/*******************************************************************************
 * Copyright (C) 2025 Amlogic, Inc. All rights reserved.
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <atomic>
#include <csignal>
#include <thread>

#include <cctype>
#include <iostream>
#include <string>
#include <vector>

#include <pthread.h>

#include "llmsdk.h"
#include <errno.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

typedef struct
{
    const char* output_path;
    int success;
} EmbeddingOutput;

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void print_usage(const char* exe)
{
    printf("Usage:\n");
    printf("  %s --model_path <embedding.adla> [--mmproj_path <mmproj.adla>]\n", exe);
    printf("\n");
    printf("Enter text directly, or use /image <path> before entering a prompt.\n");
    printf("After /image, press Enter for image-only embedding.\n");
    printf("Use /output <path> to set the next vector file.\n");
    printf("Default output: ./text.bin for text, ./image.bin for image.\n");
}

static const char* get_arg(int argc, char** argv, const char* name)
{
    for (int i = 1; i + 1 < argc; ++i)
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    return NULL;
}

static void wait_for_sigint(LLMContext context, const sigset_t* signal_set,
                            const std::atomic<bool>* stopping)
{
    while (true)
    {
        int signal_number = 0;
        if (sigwait(signal_set, &signal_number) != 0)
            continue;
        if (stopping->load())
            break;
        if (signal_number == SIGINT)
        {
            printf("\n[Signal] Call aml_llm_break\n");
            fflush(stdout);
            aml_llm_break(context);
        }
    }
}

static void callback(AML_LLMResult* result, void* userdata, AML_LLMRunStatus status)
{
    EmbeddingOutput* output = (EmbeddingOutput*)userdata;
    double squared_norm = 0.0;
    uint32_t i;
    FILE* file;
    size_t written;
    int close_result;

    if (status == AML_LLM_RUN_ERROR)
    {
        printf("run error\n");
        return;
    }
    if (status != AML_LLM_RUN_FINISH || !output || !result ||
        !result->embedding.values || result->embedding.dimension == 0)
        return;

    for (i = 0; i < result->embedding.dimension; ++i)
    {
        if (!isfinite(result->embedding.values[i])) return;
        squared_norm += (double)result->embedding.values[i] * result->embedding.values[i];
    }
    if (squared_norm <= 0.0) return;
    printf("[EMBEDDING_DEMO] dimension=%u, norm=%.9f\n",
           result->embedding.dimension, sqrt(squared_norm));
    printf("[EMBEDDING_DEMO] first_values:");
    for (i = 0; i < result->embedding.dimension && i < 8; ++i)
        printf(" %.9g", result->embedding.values[i]);
    printf("\n");

    file = fopen(output->output_path, "wb");
    if (!file)
    {
        printf("Open output failed: %s: %s\n", output->output_path, strerror(errno));
        return;
    }
    written = fwrite(result->embedding.values, sizeof(float), result->embedding.dimension, file);
    close_result = fclose(file);
    if (written != result->embedding.dimension || close_result != 0)
    {
        printf("Write output failed: %s\n", output->output_path);
        return;
    }
    printf("[EMBEDDING_DEMO] output=%s, bytes=%zu\n", output->output_path,
           written * sizeof(float));
    output->success = 1;
}

static std::string trim_copy(const std::string& text)
{
    size_t begin = 0;
    while (begin < text.size() && isspace((unsigned char)text[begin])) ++begin;
    size_t end = text.size();
    while (end > begin && isspace((unsigned char)text[end - 1])) --end;
    return text.substr(begin, end - begin);
}

static bool load_rgb_image(const char* path, std::vector<uint8_t>& rgb,
                           uint32_t* width, uint32_t* height)
{
    int w = 0, h = 0, channels = 0;
    unsigned char* data = stbi_load(path, &w, &h, &channels, 3);
    if (!data || w <= 0 || h <= 0)
    {
        if (data) stbi_image_free(data);
        printf("stbi_load failed: %s\n", path);
        return false;
    }

    rgb.assign(data, data + (size_t)w * h * 3);
    stbi_image_free(data);
    *width = (uint32_t)w;
    *height = (uint32_t)h;
    printf("image loaded: %s, original=%dx%d\n", path, w, h);
    return true;
}

int main(int argc, char** argv)
{
    if (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")))
    {
        print_usage(argv[0]);
        return 0;
    }

    const char* model_path = get_arg(argc, argv, "--model_path");
    const char* mmproj_path = get_arg(argc, argv, "--mmproj_path");
    if (!model_path)
    {
        print_usage(argv[0]);
        return -1;
    }

    sigset_t signal_set;
    sigemptyset(&signal_set);
    sigaddset(&signal_set, SIGINT);
    sigset_t old_signal_set;
    if (pthread_sigmask(SIG_BLOCK, &signal_set, &old_signal_set) != 0)
    {
        printf("pthread_sigmask failed\n");
        return -1;
    }

    AML_LLMInitConfig init_config;
    memset(&init_config, 0, sizeof(init_config));
    init_config.model_path = model_path;
    init_config.sampling_mode = AML_LLM_ARG_Max;
    init_config.top_k = 3;
    init_config.top_p = 0.9f;
    init_config.temperature = 1.0f;
    init_config.repeat_penalty = 1.1f;
    init_config.init_extend.mmproj_path = mmproj_path;

    LLMContext context = NULL;
    double start = now_ms();
    if (aml_llm_init(&context, &init_config, callback) != AML_LLM_Status_Success)
    {
        printf("aml_llm_init failed\n");
        pthread_sigmask(SIG_SETMASK, &old_signal_set, NULL);
        return -1;
    }

    printf("[EMBEDDING_DEMO] init=success, init_api_ms=%.3f\n", now_ms() - start);

    std::atomic<bool> stopping(false);
    std::thread signal_thread(wait_for_sigint, context, &signal_set, &stopping);

    AML_LLMRunConfig run_config;
    memset(&run_config, 0, sizeof(run_config));
    run_config.run_mode = AML_LLM_RUN_EMBEDDING;
    run_config.retain_history = 0;

    // Demo 显式传入图片标记、边界符，按模型选择下面的配置。
    const char* img_content = "<image>";
    const char* img_start = "<|vision_start|>"; // Qwen VLM。
    const char* img_end = "<|vision_end|>";
    // const char* img_start = "<img>"; // InternVL
    // const char* img_end = "</img>";
    // const char* img_start = ""; // FastVLM/MLP 无 Qwen 边界符。
    // const char* img_end = "";
    // const char* img_start = "<|image>"; // Gemma4 VLM
    // const char* img_end = "<image|>";


    printf("\nVLM embedding model loaded. Enter text, or use /image <path> first.\n");
    printf("After /image, press Enter for image-only embedding.\n");
    printf("Commands: exit, new_talk, break, /image <path>, /output <path>\n");
    printf("Default output: ./text.bin for text, ./image.bin for image.\n");

    std::vector<AML_LLMImageInput> images;
    std::vector<std::vector<uint8_t>> image_buffers;
    std::string next_output_path;
    unsigned int request_id = 0;

    while (true)
    {
        std::string line;
        printf("\nVLM-Embedding>>> ");
        fflush(stdout);
        if (!std::getline(std::cin, line))
        {
            break;
        }
        line = trim_copy(line);
        if (line.empty() && images.empty())
        {
            printf("Please enter your text!\n");
            continue;
        }
        if (line == "exit")
        {
            break;
        }
        if (line == "new_talk")
        {
            aml_llm_reset(context);
            images.clear();
            image_buffers.clear();
            next_output_path.clear();
            continue;
        }
        if (line == "break")
        {
            aml_llm_break(context);
            continue;
        }
        if (line == "/output" || line.rfind("/output ", 0) == 0)
        {
            std::string path = trim_copy(line.substr(7));
            if (path.empty())
            {
                printf("Usage: /output <path>\n");
                continue;
            }
            next_output_path = path;
            printf("Next output: %s\n", next_output_path.c_str());
            continue;
        }
        if (line.rfind("/image ", 0) == 0)
        {
            if (!mmproj_path)
            {
                printf("Image input requires --mmproj_path at startup.\n");
                continue;
            }
            std::string path = trim_copy(line.substr(7));
            std::vector<uint8_t> rgb;
            uint32_t width = 0;
            uint32_t height = 0;
            if (path.empty() || !load_rgb_image(path.c_str(), rgb, &width, &height))
            {
                continue;
            }
            image_buffers.push_back(std::move(rgb));
            AML_LLMImageInput image;
            memset(&image, 0, sizeof(image));
            image.type = AML_LLM_IMAGE_TYPE_BUFFER;
            image.data = image_buffers.back().data();
            image.width = width;
            image.height = height;
            images.push_back(image);
            for (size_t i = 0; i < images.size(); ++i)
            {
                images[i].data = image_buffers[i].data();
            }
            printf("image appended, total=%zu\n", images.size());
            continue;
        }

        std::string prompt = line;
        bool use_vlm = !images.empty();
        if (use_vlm && prompt.find(img_content) == std::string::npos)
        {
            std::string markers;
            for (size_t i = 0; i < images.size(); ++i)
            {
                markers += img_content;
            }
            prompt = markers + prompt;
            printf("prompt has no %s, prepend %zu marker(s)\n", img_content, images.size());
        }

        AML_LLMInput input;
        memset(&input, 0, sizeof(input));
        input.role = "user";

        if (use_vlm)
        {
            input.input_type = AML_LLM_INPUT_MULTIMODAL;
            input.multimodal_input.prompt = prompt.c_str();
            input.multimodal_input.img_inputs = images.data();
            input.multimodal_input.img_count = (uint32_t)images.size();
            input.multimodal_input.img_content = img_content;
            input.multimodal_input.img_start = img_start;
            input.multimodal_input.img_end = img_end;
        }
        else
        {
            input.input_type = AML_LLM_INPUT_PROMPT;
            input.prompt_input = prompt.c_str();
        }

        EmbeddingOutput output;
        memset(&output, 0, sizeof(output));
        output.output_path = next_output_path.empty() ? (use_vlm ? "./image.bin" : "./text.bin")
                                                     : next_output_path.c_str();
        start = now_ms();
        AML_LLMRetStatus ret = aml_llm_run(context, &input, &run_config, &output);
        printf("[EMBEDDING_DEMO] request_id=%u, run=%s, run_api_ms=%.3f\n", ++request_id,
               ret == AML_LLM_Status_Success && output.success ? "success" : "failed",
               now_ms() - start);
        next_output_path.clear();
        images.clear();
        image_buffers.clear();
    }

    stopping.store(true);
    pthread_kill(signal_thread.native_handle(), SIGINT);
    signal_thread.join();
    aml_llm_uninit(context);
    pthread_sigmask(SIG_SETMASK, &old_signal_set, NULL);
    return 0;
}
