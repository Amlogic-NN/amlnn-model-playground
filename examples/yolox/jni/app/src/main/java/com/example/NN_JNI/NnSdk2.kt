package com.example.NN_JNI

import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder

object NnSdk2 {
    const val AMLNN_SUCCESS = 0

    const val QUERY_SDK_VERSION = 0
    const val QUERY_IN_OUT_NUM = 1
    const val QUERY_INPUT_ATTR = 2
    const val QUERY_OUTPUT_ATTR = 3
    const val QUERY_INPUT_DYNAMIC_RANGE = 4
    const val QUERY_CURRENT_INPUT_ATTR = 5
    const val QUERY_CURRENT_OUTPUT_ATTR = 6
    const val QUERY_PERF_DETAIL = 7
    const val QUERY_PERF_RUN = 8
    const val QUERY_MODEL_SOFTOP = 9
    const val QUERY_NPU_CORE_STATUS = 10

    const val BACKEND_ADLA_NPU = 0
    const val BACKEND_TF_DELEGATE_GPU = 1
    const val BACKEND_TF_DELEGATE_CPU = 2

    const val TASK_PRIORITY_MEDIUM = 0
    const val TASK_PRIORITY_LOW = 1
    const val TASK_PRIORITY_HIGH = 2

    const val STRUCT_SDK_VERSION = 0
    const val STRUCT_IN_OUT_NUM = 1
    const val STRUCT_TENSOR_ATTR = 2
    const val STRUCT_INPUT_RANGE = 3
    const val STRUCT_PROFILING_DATA = 4
    const val STRUCT_PERF_RUN = 5
    const val STRUCT_NPU_CORE_STATUS = 6

    const val TENSOR_FLOAT32 = 0
    const val TENSOR_INT8 = 2
    const val TENSOR_UINT8 = 3

    init {
        preloadNativeLibrary("adla")
        System.loadLibrary("nnsdk_jni")
    }

    private fun preloadNativeLibrary(name: String) {
        try {
            System.loadLibrary(name)
            Log.i(TAG, "preload native library: lib$name.so")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "preload lib$name.so failed", e)
            throw e
        }
    }

    private const val TAG = "NnSdk2"

    fun allocateNativeBuffer(size: Int): ByteBuffer =
        ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder())

    fun sizeOf(structType: Int): Int = nativeSizeOf(structType)

    fun initFromFile(
        modelPath: String,
        backendType: Int = BACKEND_ADLA_NPU,
        taskPriority: Int = TASK_PRIORITY_MEDIUM,
        timeoutMs: Int = 0,
        enablePerf: Boolean = false,
        perfDetailPath: String? = null,
    ): Long = nativeInitFromFile(
        modelPath,
        backendType,
        taskPriority,
        timeoutMs,
        enablePerf,
        perfDetailPath,
    )

    fun initFromBuffer(
        modelBuffer: ByteBuffer,
        modelSize: Int = modelBuffer.remaining(),
        backendType: Int = BACKEND_ADLA_NPU,
        taskPriority: Int = TASK_PRIORITY_MEDIUM,
        timeoutMs: Int = 0,
        enablePerf: Boolean = false,
        perfDetailPath: String? = null,
    ): Long = nativeInitFromBuffer(
        modelBuffer,
        modelSize,
        backendType,
        taskPriority,
        timeoutMs,
        enablePerf,
        perfDetailPath,
    )

    fun query(context: Long, queryCmd: Int, info: ByteBuffer, size: Int = info.capacity()): Int =
        nativeQuery(context, queryCmd, info, size)

    fun inputsSet(
        context: Long,
        buffers: Array<ByteBuffer>,
        sizes: IntArray? = null,
        indices: IntArray? = null,
    ): Int = nativeInputsSet(context, buffers, sizes, indices)

    fun run(context: Long): Int = nativeRun(context)

    fun outputsGet(context: Long, index: Int = 0): FloatArray =
        nativeOutputsGet(context, index)

    fun destroy(context: Long): Int = nativeDestroy(context)

    private external fun nativeSizeOf(structType: Int): Int

    private external fun nativeInitFromFile(
        modelPath: String,
        backendType: Int,
        taskPriority: Int,
        timeoutMs: Int,
        enablePerf: Boolean,
        perfDetailPath: String?,
    ): Long

    private external fun nativeInitFromBuffer(
        modelBuffer: ByteBuffer,
        modelSize: Int,
        backendType: Int,
        taskPriority: Int,
        timeoutMs: Int,
        enablePerf: Boolean,
        perfDetailPath: String?,
    ): Long

    private external fun nativeQuery(context: Long, queryCmd: Int, info: ByteBuffer, size: Int): Int

    private external fun nativeInputsSet(
        context: Long,
        buffers: Array<ByteBuffer>,
        sizes: IntArray?,
        indices: IntArray?,
    ): Int

    private external fun nativeRun(context: Long): Int

    private external fun nativeOutputsGet(context: Long, index: Int): FloatArray

    private external fun nativeDestroy(context: Long): Int
}
