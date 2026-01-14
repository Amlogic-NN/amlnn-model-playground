/*
 * Copyright (C) 2024–2025 Amlogic, Inc. All rights reserved.
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

#include <string.h>
#include <unistd.h>
#include <string>
#include <iostream>

#include "llmsdk.h"

typedef struct
{
    int request_id;
    bool printed;
} MyUserData;

void callback(AML_LLMResult *result, void *userdata, AML_LLMRunStatus run_status)
{
    if (!userdata) return;

    MyUserData* my_data = (MyUserData*)userdata;

    if (run_status == AML_LLM_RUN_NORMAL)
    {
        if (!my_data->printed)
        {
            printf("[Request #%d]\n", my_data->request_id);
            my_data->printed = true;
        }
        printf("%s", result->text);
        // printf("%d,", result->token_id);
        fflush(stdout);
    }
    else if (run_status == AML_LLM_RUN_FINISH)
    {
        printf("\n");
    }
    else if (run_status == AML_LLM_RUN_ERROR)
    {
        printf("run error\n");
    }
}


int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Usage: %s <model_path> <tokenizer_path>\n", argv[0]);
        return -1;
    }

    printf("\nWelcome to Amlogic LLM Demo!\n");

    LLMContext context;
    AML_LLMInitConfig init_config;
    memset(&init_config, 0, sizeof(AML_LLMInitConfig));
    init_config.model_path = (const char *)argv[1];
    init_config.tokenizer_path = (const char *)argv[2];
    init_config.sampling_mode = AML_LLM_ARG_Max;
    init_config.top_k = 3;
    init_config.top_p = 0.9f;
    init_config.temperature = 1.0f;
    init_config.repeat_penalty = 1.1f;

    aml_llm_init(&context, &init_config, callback);

    AML_LLMInput input;
    memset(&input, 0, sizeof(AML_LLMInput));
    input.input_type = AML_LLM_INPUT_PROMPT;

    AML_LLMRunConfig run_config;
    memset(&run_config, 0, sizeof(AML_LLMRunConfig));
    run_config.run_mode = AML_LLM_RUN_GENERATE;
    run_config.retain_history = 0;

    MyUserData my_data;
    memset(&my_data, 0, sizeof(MyUserData));

    printf("\nType your prompt and press Enter.\n");
    printf("Commands: [exit] to quit, [new_talk] to reset context.\n");

    while (true)
    {
        std::string input_str;
        printf("\nLLM@Amlogic>>> ");
        std::getline(std::cin, input_str);
        if (input_str == "new_talk")
        {
            aml_llm_reset(context);
            continue;
        }
        else if (input_str.empty())
        {
            printf("Please enter your question!\n");
            continue;
        }
        else if (input_str == "exit")
        {
            break;
        }

        my_data.request_id++;
        my_data.printed = false;
        input.prompt_input = (const char *)input_str.c_str();

        aml_llm_run(context, &input, &run_config, &my_data);
    }

    printf("Bye~\n");

    aml_llm_uninit(context);

    return 0;
}