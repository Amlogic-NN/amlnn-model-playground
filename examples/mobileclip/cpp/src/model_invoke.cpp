/*
 * Copyright (C) 2026 Amlogic, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "model_invoke.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

struct ModelInvokeContext {
    void* context = nullptr;
    uint32_t n_inputs = 0;
    uint32_t n_outputs = 0;
    std::vector<amlnn_tensor_attr> input_attrs;
    std::vector<amlnn_tensor_attr> output_attrs;
};

ModelInvokeContext* as_context(void* qcontext) {
    return static_cast<ModelInvokeContext*>(qcontext);
}

bool query_tensor_attrs(void* context, ModelInvokeContext* ctx) {
    if (!context || !ctx) return false;

    amlnn_input_output_num io_num;
    std::memset(&io_num, 0, sizeof(io_num));
    if (amlnn_query(context, AMLNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)) != AMLNN_SUCCESS) {
        std::printf("[Error] amlnn_query AMLNN_QUERY_IN_OUT_NUM failed.\n");
        return false;
    }

    ctx->n_inputs = io_num.n_input;
    ctx->n_outputs = io_num.n_output;
    ctx->input_attrs.assign(io_num.n_input, amlnn_tensor_attr{});
    ctx->output_attrs.assign(io_num.n_output, amlnn_tensor_attr{});

    for (uint32_t i = 0; i < io_num.n_input; ++i) {
        ctx->input_attrs[i].index = i;
        if (amlnn_query(context, AMLNN_QUERY_INPUT_ATTR, &ctx->input_attrs[i],
                        sizeof(amlnn_tensor_attr)) != AMLNN_SUCCESS) {
            std::printf("[Error] amlnn_query AMLNN_QUERY_INPUT_ATTR[%u] failed.\n", i);
            return false;
        }
    }

    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        ctx->output_attrs[i].index = i;
        if (amlnn_query(context, AMLNN_QUERY_OUTPUT_ATTR, &ctx->output_attrs[i],
                        sizeof(amlnn_tensor_attr)) != AMLNN_SUCCESS) {
            std::printf("[Error] amlnn_query AMLNN_QUERY_OUTPUT_ATTR[%u] failed.\n", i);
            return false;
        }
    }

    return true;
}

void apply_softop_options(void* context, const AdlaInitOptions& options) {
    if (!context || !options.enable_openmp) {
        return;
    }

    amlnn_softop_opt_request request;
    std::memset(&request, 0, sizeof(request));
    request.softop_type = AMLNN_Unknown;
    request.acc_type = AMLNN_SOFTOP_ACC_OPENMP;
    if (amlnn_set_softop_opt(context, &request, 1) != AMLNN_SUCCESS) {
        std::printf("[Warn] amlnn_set_softop_opt(OpenMP) failed.\n");
    }
}

std::vector<std::vector<float>> fetch_float_outputs(ModelInvokeContext* ctx) {
    if (!ctx || !ctx->context || ctx->n_outputs == 0) return {};

    std::vector<amlnn_output> outputs(ctx->n_outputs);
    for (uint32_t i = 0; i < ctx->n_outputs; ++i) {
        std::memset(&outputs[i], 0, sizeof(amlnn_output));
        outputs[i].index = i;
        outputs[i].is_float = 1;
    }

    if (amlnn_outputs_get(ctx->context, ctx->n_outputs, outputs.data()) != AMLNN_SUCCESS) {
        std::printf("[Error] amlnn_outputs_get failed.\n");
        return {};
    }

    std::vector<std::vector<float>> result;
    result.reserve(ctx->n_outputs);
    for (uint32_t i = 0; i < ctx->n_outputs; ++i) {
        if (!outputs[i].buf || outputs[i].size == 0) {
            std::printf("[Error] Output[%u] is empty.\n", i);
            return {};
        }
        if (outputs[i].size % sizeof(float) != 0) {
            std::printf("[Error] Output[%u] size %u is not aligned to float32.\n",
                        i, outputs[i].size);
            return {};
        }
        const float* output_ptr = reinterpret_cast<const float*>(outputs[i].buf);
        size_t output_elements = outputs[i].size / sizeof(float);
        result.emplace_back(output_ptr, output_ptr + output_elements);
    }
    return result;
}

}  // namespace

void* init_network_file(const char* model_path, const AdlaInitOptions& options)
{
    if (!model_path) return nullptr;

    amlnn_init_config init_config;
    std::memset(&init_config, 0, sizeof(init_config));
    init_config.backend_type = AMLNN_BACKEND_ADLA_NPU;
    init_config.task_priority = AMLNN_MODEL_TASK_PRIOR_MEDIUM;

    void* context = nullptr;
    if (amlnn_init(&context, const_cast<char*>(model_path), 0, &init_config) != AMLNN_SUCCESS) {
        std::printf("[Error] amlnn_init failed for %s\n", model_path);
        return nullptr;
    }

    apply_softop_options(context, options);

    auto* ctx = new ModelInvokeContext();
    ctx->context = context;
    if (!query_tensor_attrs(context, ctx)) {
        amlnn_destroy(context);
        delete ctx;
        return nullptr;
    }

    return ctx;
}

bool get_input_tensor_desc(void* qcontext, uint32_t input_index, ModelInputTensorDesc* desc) {
    if (!qcontext || !desc) return false;

    auto* ctx = as_context(qcontext);
    if (!ctx || input_index >= ctx->input_attrs.size()) {
        return false;
    }

    const amlnn_tensor_attr& attr = ctx->input_attrs[input_index];
    desc->dim_count = attr.n_dims;
    for (uint32_t i = 0; i < attr.n_dims && i < AMLNN_MAX_DIMS; ++i) {
        desc->dims[i] = attr.dims[i];
    }
    desc->tensor_format = attr.fmt;
    desc->tensor_type = attr.type;
    desc->scale = attr.scale;
    desc->zero_point = attr.zp;
    return true;
}

std::vector<float> run_model_with_input(ModelInvokeContext* ctx, void* input_buf, uint32_t input_size)
{
    if (!ctx || !ctx->context || !input_buf || input_size == 0) return {};

    amlnn_input input;
    std::memset(&input, 0, sizeof(input));
    input.index = 0;
    input.buf = input_buf;
    input.size = input_size;

    if (amlnn_inputs_set(ctx->context, 1, &input) != AMLNN_SUCCESS) {
        std::printf("[Error] amlnn_inputs_set failed.\n");
        return {};
    }

    if (amlnn_run(ctx->context, nullptr) != AMLNN_SUCCESS) {
        std::printf("[Error] amlnn_run failed.\n");
        return {};
    }

    auto outputs = fetch_float_outputs(ctx);
    return outputs.empty() ? std::vector<float>{} : std::move(outputs[0]);
}

std::vector<float> run_image_model(void* qcontext, const std::vector<uint8_t>& input_data)
{
    auto* ctx = as_context(qcontext);
    if (!ctx || input_data.empty()) return {};

    return run_model_with_input(
        ctx,
        const_cast<uint8_t*>(input_data.data()),
        static_cast<uint32_t>(input_data.size()));
}

std::vector<float> run_text_model(void* qcontext, const std::vector<int64_t>& input_ids)
{
    auto* ctx = as_context(qcontext);
    if (!ctx || input_ids.empty()) return {};

    return run_model_with_input(
        ctx,
        const_cast<int64_t*>(input_ids.data()),
        static_cast<uint32_t>(input_ids.size() * sizeof(int64_t)));
}

int destroy_network(void *qcontext)
{
    auto* ctx = as_context(qcontext);
    if (!ctx) return AMLNN_ERR_FAIL;

    int ret = AMLNN_SUCCESS;
    if (ctx->context) {
        ret = amlnn_destroy(ctx->context);
        if (ret != AMLNN_SUCCESS) {
            std::printf("[Error] amlnn_destroy failed. Code: %d\n", ret);
        }
    }

    delete ctx;
    return ret;
}
