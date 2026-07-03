#include <android/log.h>
#include <dlfcn.h>
#include <jni.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>
#include "nnsdk2.h"

namespace {

constexpr const char* kTag = "NnSdk2Jni";
constexpr const char* kNnSdkSoName = "libnnsdk.so";
constexpr const char* kPreloadLibs[] = {
    "libadla.so",
};

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, kTag, __VA_ARGS__)

using AmlnnInit = int (*)(void**, void*, uint32_t, amlnn_init_config*);
using AmlnnQuery = int (*)(void*, amlnn_query_cmd, void*, uint32_t);
using AmlnnInputsSet = int (*)(void*, uint32_t, amlnn_input[]);
using AmlnnRun = int (*)(void*, amlnn_run_config*);
using AmlnnOutputsGet = int (*)(void*, uint32_t, amlnn_output[]);
using AmlnnDestroy = int (*)(void*);

struct NnSdkSymbols {
    void* handle = nullptr;
    AmlnnInit init = nullptr;
    AmlnnQuery query = nullptr;
    AmlnnInputsSet inputs_set = nullptr;
    AmlnnRun run = nullptr;
    AmlnnOutputsGet outputs_get = nullptr;
    AmlnnDestroy destroy = nullptr;
    char error[256] = {};
};

NnSdkSymbols g_symbols;
std::once_flag g_load_once;

void set_last_error(const char* message) {
    std::snprintf(g_symbols.error, sizeof(g_symbols.error), "%s", message ? message : "unknown error");
}

void clear_loaded_symbols() {
    g_symbols.handle = nullptr;
    g_symbols.init = nullptr;
    g_symbols.query = nullptr;
    g_symbols.inputs_set = nullptr;
    g_symbols.run = nullptr;
    g_symbols.outputs_get = nullptr;
    g_symbols.destroy = nullptr;
}

template <typename T>
bool load_symbol(void* handle, const char* name, T* out) {
    dlerror();
    void* symbol = dlsym(handle, name);
    const char* error = dlerror();
    if (error != nullptr || symbol == nullptr) {
        char buffer[256];
        std::snprintf(buffer, sizeof(buffer), "dlsym(%s) failed: %s", name, error ? error : "null symbol");
        set_last_error(buffer);
        LOGE("%s", buffer);
        return false;
    }

    *out = reinterpret_cast<T>(symbol);
    return true;
}

void load_nnsdk_once() {
    for (const char* lib : kPreloadLibs) {
        dlerror();
        void* preload = dlopen(lib, RTLD_NOW | RTLD_GLOBAL);
        if (preload == nullptr) {
            const char* error = dlerror();
            LOGE("preload %s failed: %s (should be loaded via System.loadLibrary first)", lib, error ? error : "unknown");
        }
    }

    dlerror();
    void* handle = dlopen(kNnSdkSoName, RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        const char* error = dlerror();
        char buffer[256];
        std::snprintf(buffer, sizeof(buffer), "dlopen(%s) failed: %s", kNnSdkSoName, error ? error : "unknown");
        set_last_error(buffer);
        LOGE("%s", buffer);
        return;
    }

    g_symbols.handle = handle;

    if (!load_symbol(handle, "amlnn_init", &g_symbols.init) ||
        !load_symbol(handle, "amlnn_query", &g_symbols.query) ||
        !load_symbol(handle, "amlnn_inputs_set", &g_symbols.inputs_set) ||
        !load_symbol(handle, "amlnn_run", &g_symbols.run) ||
        !load_symbol(handle, "amlnn_outputs_get", &g_symbols.outputs_get) ||
        !load_symbol(handle, "amlnn_destroy", &g_symbols.destroy)) {
        dlclose(handle);
        clear_loaded_symbols();
        return;
    }
}

bool ensure_loaded(JNIEnv* env) {
    std::call_once(g_load_once, load_nnsdk_once);
    if (g_symbols.handle != nullptr) {
        return true;
    }

    jclass exception_class = env->FindClass("java/lang/IllegalStateException");
    if (exception_class != nullptr) {
        env->ThrowNew(exception_class, g_symbols.error[0] ? g_symbols.error : "libnnsdk.so is not loaded");
    }
    return false;
}

void throw_illegal_argument(JNIEnv* env, const char* message) {
    jclass exception_class = env->FindClass("java/lang/IllegalArgumentException");
    if (exception_class != nullptr) {
        env->ThrowNew(exception_class, message);
    }
}

void throw_illegal_state(JNIEnv* env, const char* message) {
    jclass exception_class = env->FindClass("java/lang/IllegalStateException");
    if (exception_class != nullptr) {
        env->ThrowNew(exception_class, message);
    }
}

void* require_context(JNIEnv* env, jlong context) {
    if (context == 0) {
        throw_illegal_argument(env, "context is 0");
        return nullptr;
    }
    return reinterpret_cast<void*>(context);
}

void* require_direct_buffer(JNIEnv* env, jobject buffer, const char* name, jint required_size = 0) {
    if (buffer == nullptr) {
        throw_illegal_argument(env, name);
        return nullptr;
    }

    void* address = env->GetDirectBufferAddress(buffer);
    jlong capacity = env->GetDirectBufferCapacity(buffer);
    if (address == nullptr || capacity < 0) {
        char message[128];
        std::snprintf(message, sizeof(message), "%s must be a direct ByteBuffer", name);
        throw_illegal_argument(env, message);
        return nullptr;
    }

    if (required_size > 0 && capacity < required_size) {
        char message[160];
        std::snprintf(
            message,
            sizeof(message),
            "%s capacity(%lld) is smaller than required size(%d)",
            name,
            static_cast<long long>(capacity),
            required_size);
        throw_illegal_argument(env, message);
        return nullptr;
    }

    return address;
}

uint32_t checked_size(JNIEnv* env, jlong capacity, jint requested_size, const char* name) {
    if (requested_size < 0) {
        char message[128];
        std::snprintf(message, sizeof(message), "%s size must be >= 0", name);
        throw_illegal_argument(env, message);
        return 0;
    }
    if (requested_size > capacity) {
        char message[160];
        std::snprintf(
            message,
            sizeof(message),
            "%s size(%d) is larger than buffer capacity(%lld)",
            name,
            requested_size,
            static_cast<long long>(capacity));
        throw_illegal_argument(env, message);
        return 0;
    }
    return static_cast<uint32_t>(requested_size);
}

amlnn_init_config make_init_config(
    jint backend_type,
    jint task_priority,
    jint timeout_ms,
    jboolean enable_perf,
    const char* perf_detail_path) {
    amlnn_init_config config = {};
    config.backend_type = static_cast<amlnn_backend_type>(backend_type);
    config.task_priority = static_cast<amlnn_model_task_priority>(task_priority);
    config.timeout_ms = timeout_ms < 0 ? 0 : static_cast<uint32_t>(timeout_ms);
    config.enable_perf = enable_perf ? 1U : 0U;
    config.perf_detail_path = perf_detail_path;
    return config;
}

jlong native_init_from_file(
    JNIEnv* env,
    jobject,
    jstring model_path,
    jint backend_type,
    jint task_priority,
    jint timeout_ms,
    jboolean enable_perf,
    jstring perf_detail_path) {
    if (!ensure_loaded(env)) {
        return 0;
    }
    if (model_path == nullptr) {
        throw_illegal_argument(env, "modelPath is null");
        return 0;
    }

    const char* model_path_chars = env->GetStringUTFChars(model_path, nullptr);
    const char* perf_detail_path_chars =
        perf_detail_path != nullptr ? env->GetStringUTFChars(perf_detail_path, nullptr) : nullptr;
    if (model_path_chars == nullptr || (perf_detail_path != nullptr && perf_detail_path_chars == nullptr)) {
        if (model_path_chars != nullptr) {
            env->ReleaseStringUTFChars(model_path, model_path_chars);
        }
        return 0;
    }

    amlnn_init_config config =
        make_init_config(backend_type, task_priority, timeout_ms, enable_perf, perf_detail_path_chars);
    void* context = nullptr;
    int ret = g_symbols.init(&context, const_cast<char*>(model_path_chars), 0, &config);

    env->ReleaseStringUTFChars(model_path, model_path_chars);
    if (perf_detail_path_chars != nullptr) {
        env->ReleaseStringUTFChars(perf_detail_path, perf_detail_path_chars);
    }

    if (ret != AMLNN_SUCCESS) {
        char message[96];
        std::snprintf(message, sizeof(message), "amlnn_init failed: %d", ret);
        throw_illegal_state(env, message);
        return 0;
    }
    return reinterpret_cast<jlong>(context);
}

jlong native_init_from_buffer(
    JNIEnv* env,
    jobject,
    jobject model_buffer,
    jint model_size,
    jint backend_type,
    jint task_priority,
    jint timeout_ms,
    jboolean enable_perf,
    jstring perf_detail_path) {
    if (!ensure_loaded(env)) {
        return 0;
    }
    if (model_size < 0) {
        throw_illegal_argument(env, "modelSize must be >= 0");
        return 0;
    }

    void* model = require_direct_buffer(env, model_buffer, "modelBuffer", model_size);
    if (env->ExceptionCheck()) {
        return 0;
    }

    const char* perf_detail_path_chars =
        perf_detail_path != nullptr ? env->GetStringUTFChars(perf_detail_path, nullptr) : nullptr;
    if (perf_detail_path != nullptr && perf_detail_path_chars == nullptr) {
        return 0;
    }

    amlnn_init_config config =
        make_init_config(backend_type, task_priority, timeout_ms, enable_perf, perf_detail_path_chars);
    void* context = nullptr;
    int ret = g_symbols.init(&context, model, static_cast<uint32_t>(model_size), &config);

    if (perf_detail_path_chars != nullptr) {
        env->ReleaseStringUTFChars(perf_detail_path, perf_detail_path_chars);
    }

    if (ret != AMLNN_SUCCESS) {
        char message[96];
        std::snprintf(message, sizeof(message), "amlnn_init failed: %d", ret);
        throw_illegal_state(env, message);
        return 0;
    }
    return reinterpret_cast<jlong>(context);
}

jint native_query(JNIEnv* env, jobject, jlong context, jint query_cmd, jobject info, jint size) {
    if (!ensure_loaded(env)) {
        return AMLNN_ERR_FAIL;
    }
    if (size < 0) {
        throw_illegal_argument(env, "size must be >= 0");
        return AMLNN_ERR_PARAM_INVALID;
    }

    void* native_context = require_context(env, context);
    void* info_address = require_direct_buffer(env, info, "info", size);
    if (env->ExceptionCheck()) {
        return AMLNN_ERR_PARAM_INVALID;
    }

    return g_symbols.query(
        native_context,
        static_cast<amlnn_query_cmd>(query_cmd),
        info_address,
        static_cast<uint32_t>(size));
}

jint native_inputs_set(JNIEnv* env, jobject, jlong context, jobjectArray buffers, jintArray sizes, jintArray indices) {
    if (!ensure_loaded(env)) {
        return AMLNN_ERR_FAIL;
    }

    void* native_context = require_context(env, context);
    if (native_context == nullptr) {
        return AMLNN_ERR_PARAM_INVALID;
    }
    if (buffers == nullptr) {
        throw_illegal_argument(env, "buffers is null");
        return AMLNN_ERR_PARAM_INVALID;
    }

    const jsize count = env->GetArrayLength(buffers);
    if ((sizes != nullptr && env->GetArrayLength(sizes) < count) ||
        (indices != nullptr && env->GetArrayLength(indices) < count)) {
        throw_illegal_argument(env, "sizes/indices length must be >= buffers length");
        return AMLNN_ERR_PARAM_INVALID;
    }

    jint* size_values = sizes != nullptr ? env->GetIntArrayElements(sizes, nullptr) : nullptr;
    jint* index_values = indices != nullptr ? env->GetIntArrayElements(indices, nullptr) : nullptr;
    std::vector<amlnn_input> native_inputs(static_cast<size_t>(count));

    for (jsize i = 0; i < count; ++i) {
        jobject buffer = env->GetObjectArrayElement(buffers, i);
        void* address = require_direct_buffer(env, buffer, "input buffer");
        jlong capacity = buffer != nullptr ? env->GetDirectBufferCapacity(buffer) : -1;
        if (buffer != nullptr) {
            env->DeleteLocalRef(buffer);
        }
        if (env->ExceptionCheck()) {
            if (size_values != nullptr) env->ReleaseIntArrayElements(sizes, size_values, JNI_ABORT);
            if (index_values != nullptr) env->ReleaseIntArrayElements(indices, index_values, JNI_ABORT);
            return AMLNN_ERR_PARAM_INVALID;
        }

        jint requested_size = size_values != nullptr ? size_values[i] : static_cast<jint>(capacity);
        uint32_t native_size = checked_size(env, capacity, requested_size, "input buffer");
        if (env->ExceptionCheck()) {
            if (size_values != nullptr) env->ReleaseIntArrayElements(sizes, size_values, JNI_ABORT);
            if (index_values != nullptr) env->ReleaseIntArrayElements(indices, index_values, JNI_ABORT);
            return AMLNN_ERR_PARAM_INVALID;
        }

        native_inputs[static_cast<size_t>(i)].index =
            index_values != nullptr ? static_cast<uint32_t>(index_values[i]) : static_cast<uint32_t>(i);
        native_inputs[static_cast<size_t>(i)].buf = address;
        native_inputs[static_cast<size_t>(i)].size = native_size;
    }

    if (size_values != nullptr) env->ReleaseIntArrayElements(sizes, size_values, JNI_ABORT);
    if (index_values != nullptr) env->ReleaseIntArrayElements(indices, index_values, JNI_ABORT);

    return g_symbols.inputs_set(native_context, static_cast<uint32_t>(count), native_inputs.data());
}

jint native_run(JNIEnv* env, jobject, jlong context) {
    if (!ensure_loaded(env)) {
        return AMLNN_ERR_FAIL;
    }

    void* native_context = require_context(env, context);
    if (native_context == nullptr) {
        return AMLNN_ERR_PARAM_INVALID;
    }
    return g_symbols.run(native_context, nullptr);
}

jfloatArray native_outputs_get(JNIEnv* env, jobject, jlong context, jint index) {
    if (!ensure_loaded(env)) {
        return nullptr;
    }

    void* native_context = require_context(env, context);
    if (native_context == nullptr) {
        return nullptr;
    }

    amlnn_output output = {};
    output.is_float = 1U;
    output.index = index < 0 ? 0U : static_cast<uint32_t>(index);

    int ret = g_symbols.outputs_get(native_context, 1, &output);
    if (ret != AMLNN_SUCCESS) {
        char message[96];
        std::snprintf(message, sizeof(message), "amlnn_outputs_get(float) failed: %d", ret);
        throw_illegal_state(env, message);
        return nullptr;
    }
    if (output.buf == nullptr || output.size == 0 || output.size % sizeof(float) != 0) {
        throw_illegal_state(env, "amlnn_outputs_get(float) returned invalid output");
        return nullptr;
    }

    const jsize count = static_cast<jsize>(output.size / sizeof(float));
    jfloatArray result = env->NewFloatArray(count);
    if (result == nullptr) {
        return nullptr;
    }
    env->SetFloatArrayRegion(result, 0, count, reinterpret_cast<const jfloat*>(output.buf));
    return result;
}

jint native_destroy(JNIEnv* env, jobject, jlong context) {
    if (!ensure_loaded(env)) {
        return AMLNN_ERR_FAIL;
    }

    void* native_context = require_context(env, context);
    if (native_context == nullptr) {
        return AMLNN_ERR_PARAM_INVALID;
    }
    return g_symbols.destroy(native_context);
}

jint native_size_of(JNIEnv* env, jobject, jint struct_type) {
    switch (struct_type) {
        case 0:
            return sizeof(amlnn_sdk_version);
        case 1:
            return sizeof(amlnn_input_output_num);
        case 2:
            return sizeof(amlnn_tensor_attr);
        case 3:
            return sizeof(amlnn_input_range);
        case 4:
            return sizeof(amlnn_profiling_data);
        case 5:
            return sizeof(amlnn_perf_run);
        case 6:
            return sizeof(amlnn_npu_core_status);
        default:
            throw_illegal_argument(env, "unknown struct type");
            return 0;
    }
}

JNINativeMethod g_methods[] = {
    {"nativeSizeOf", "(I)I", reinterpret_cast<void*>(native_size_of)},
    {"nativeInitFromFile", "(Ljava/lang/String;IIIZLjava/lang/String;)J", reinterpret_cast<void*>(native_init_from_file)},
    {"nativeInitFromBuffer", "(Ljava/nio/ByteBuffer;IIIIZLjava/lang/String;)J", reinterpret_cast<void*>(native_init_from_buffer)},
    {"nativeQuery", "(JILjava/nio/ByteBuffer;I)I", reinterpret_cast<void*>(native_query)},
    {"nativeInputsSet", "(J[Ljava/nio/ByteBuffer;[I[I)I", reinterpret_cast<void*>(native_inputs_set)},
    {"nativeRun", "(J)I", reinterpret_cast<void*>(native_run)},
    {"nativeOutputsGet", "(JI)[F", reinterpret_cast<void*>(native_outputs_get)},
    {"nativeDestroy", "(J)I", reinterpret_cast<void*>(native_destroy)},
};

}  // namespace

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*) {
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK || env == nullptr) {
        return JNI_ERR;
    }

    jclass clazz = env->FindClass("com/example/NN_JNI/NnSdk2");
    if (clazz == nullptr) {
        return JNI_ERR;
    }

    if (env->RegisterNatives(clazz, g_methods, sizeof(g_methods) / sizeof(g_methods[0])) != JNI_OK) {
        env->DeleteLocalRef(clazz);
        return JNI_ERR;
    }
    env->DeleteLocalRef(clazz);
    return JNI_VERSION_1_6;
}
