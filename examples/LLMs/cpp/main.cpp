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
#include <termios.h>

extern "C" {
    #include "llmsdk.h"
}

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

std::string get_line_with_backspace() {
    std::string input;
    struct termios old_t, new_t;

    tcgetattr(STDIN_FILENO, &old_t);
    new_t = old_t;

    new_t.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &new_t);

    char c;
    while (read(STDIN_FILENO, &c, 1) == 1) {
        if (c == '\n' || c == '\r') {
            printf("\n");
            break;
        }
        else if (c == 8 || c == 127) {
            if (!input.empty()) {
                input.pop_back();
                printf("\b \b");
                fflush(stdout);
            }
        }
        else if (c >= 32 && c <= 126) {
            input.push_back(c);
            printf("%c", c);
            fflush(stdout);
        }
    }

    tcsetattr(STDIN_FILENO, TCSANOW, &old_t);
    return input;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Usage: %s <model_path>\n", argv[0]);
        return -1;
    }

    LLMContext context;
    AML_LLMInitConfig init_config;
    memset(&init_config, 0, sizeof(AML_LLMInitConfig));
    init_config.model_path = (const char *)argv[1];
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

    printf("***************************************************\n");
    printf("*                                                 *\n");
    printf("*   _____  _   _    _  _____ __     __            *\n");
    printf("*  |  ___|| \\ | |  | ||  _  |\\ \\   / /            *\n");
    printf("*  | |__  |  \\| |  | || | | | \\ \\_/ /             *\n");
    printf("*  |  __| | . ` |  | || | | |  \\   /              *\n");
    printf("*  | |___ | |\\  |__| || |_| |   | |               *\n");
    printf("*  |_____||_| \\_|\\___/ \\___/    |_|               *\n");
    printf("*                                                 *\n");
    printf("*       _    __  __ _      ____   _____ _____ ____*\n");
    printf("*      / \\  |  \\/  | |    / __ \\ / ____|_   _/ ___|*\n");
    printf("*     / _ \\ | \\  / | |   | |  | | |  __  | || |    *\n");
    printf("*    / ___ \\| |\\/| | |   | |  | | | |_ | | || |    *\n");
    printf("*   / /   \\ \\ |  | | |___| |__| | |__| |_| || |___ *\n");
    printf("*  /_/     \\_\\_|  |_|______\\____/ \\_____|_____\\____|*\n");
    printf("*                                                 *\n");
    printf("*            _      _      __  __                 *\n");
    printf("*           | |    | |    |  \\/  |                *\n");
    printf("*           | |    | |    | \\  / |                *\n");
    printf("*           | |    | |    | |\\/| |                *\n");
    printf("*           | |____| |____| |  | |                *\n");
    printf("*           |______|______|_|  |_|                *\n");
    printf("*                                                 *\n");
    printf("***************************************************\n");

    printf("\nType your prompt and press Enter.\n");
    printf("Commands: [exit] to quit, [new_talk] to reset context.\n");

    while (true)
    {
        printf("\nLLM@Amlogic>>> ");
        fflush(stdout);
        std::string input_str = get_line_with_backspace();

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