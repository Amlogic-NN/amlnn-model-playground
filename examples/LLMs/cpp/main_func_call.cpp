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
#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <regex>

#include "llmsdk.h"
#include <nlohmann/json.hpp>

// Small holder for callback behavior:
// - printed: print tokens as they come
// - buffer: collect first-pass output (for external two-run mode)
struct DemoUserData {
    bool printed = true;
    int request_id;
    std::string buffer;
};

// Stream callback: either print tokens or accumulate to buffer
static void result_callback(AML_LLMResult* result, void* userdata, AML_LLMRunStatus run_status) {

    // std::cout << result->text;
    // std::cout.flush();
    auto* mydata = reinterpret_cast<DemoUserData*>(userdata);
    if (!mydata) return;

    if (run_status == AML_LLM_RUN_NORMAL) {
        if (result && result->text) {
            if (mydata->printed) {
                std::cout << result->text;
                std::cout.flush();
            }
            mydata->buffer.append(result->text);
        }
    } else if (run_status == AML_LLM_RUN_FINISH) {
        if (mydata->printed) std::cout << "\n[FINISHED]\n";
    } else if (run_status == AML_LLM_RUN_ERROR) {
        std::cerr << "\n[ERROR]\n";
    }
}

// Internal tool execution callback (JNI-internal mode)
static int tool_callback(
    const char* tool_name,
    const char* arguments_json,
    void* /*userdata*/,
    char** out_result_buffer
) {
    std::cout << "\n[TOOL CALL - internal] name=" << tool_name
              << " args=" << (arguments_json ? arguments_json : "(null)")
              << "\n";

    if (strcmp(tool_name, "get_weather") == 0) {
        const char* resp =
            "{"
            "\"city\":\"Hangzhou\","
            "\"temperature\":\"25C\","
            "\"condition\":\"Sunny\""
            "}";
        *out_result_buffer = strdup(resp); // Freed by SDK
        return 0;
    }

    const char* fallback = "{\"error\":\"tool_not_found\"}";
    *out_result_buffer = strdup(fallback);
    return -1;
}

// External tool execution stub (for demo)
static std::string external_execute_tool(const std::string& tool_name, const std::string& /*args_json*/) {
    if (tool_name == "get_weather") {
        return std::string("{") +
               "\"city\":\"Hangzhou\","
               "\"temperature\":\"25C\","
               "\"condition\":\"Sunny\""
               "}";
    }
    return std::string("{\"error\":\"tool_not_found\"}");
}

// Parse <tool_call>{...}</tool_call> blocks from model output
struct ToolCall { std::string name; std::string args_json; };
static std::vector<ToolCall> parse_tool_calls(const std::string& text) {
    std::vector<ToolCall> calls;
    printf("model_output: %s\n", text.c_str());

    // match across newlines safely
    std::regex tool_call_pattern(
        "<tool_call>[\\n\\r\\s]*(\\{[\\n\\r\\s\\S]*?\\})[\\n\\r\\s]*</tool_call>",
        std::regex::icase | std::regex::ECMAScript
    );

    std::sregex_iterator it(text.begin(), text.end(), tool_call_pattern);
    std::sregex_iterator end;


    for (; it != end; ++it) {
        std::string json_str = (*it)[1].str();
        printf("Matched tool_call JSON: %s\n", json_str.c_str());
        nlohmann::json parsed;
        parsed = nlohmann::json::parse(json_str, nullptr, /* allow_exceptions = */ false);
        if (parsed.is_discarded()) {
            printf("Function Calling JSON parse failed: %s\n", json_str.c_str());
            continue;
        }

        std::string tool;
        if (parsed.contains("tool_name")) tool = parsed["tool_name"].get<std::string>();
        else if (parsed.contains("name")) tool = parsed["name"].get<std::string>();

        if (!tool.empty()) {
            calls.push_back(ToolCall{tool, parsed["arguments"].dump()});
        }
        printf("tool: %s, arguments: %s\n", tool.c_str(), parsed["arguments"].dump().c_str());

    }
    return calls;
}

// Build JSON array of tool results by executing each tool externally
static std::string build_tool_results_json(const std::vector<ToolCall>& calls) {
    std::string out = "[";
    for (size_t i = 0; i < calls.size(); ++i) {
        const auto& c = calls[i];
        std::string one = external_execute_tool(c.name, c.args_json);
        out += one;
        if (i + 1 < calls.size()) out += ",";
    }
    out += "]";
    return out;
}

int main(int argc, char** argv) {
    // Init LLM context
    LLMContext ctx = nullptr;

    AML_LLMInitConfig init_cfg;
    std::memset(&init_cfg, 0, sizeof(init_cfg));
    init_cfg.model_path     = argc > 1 ? argv[1] : "/path/to/your/model.bin";       // TODO: replace
    init_cfg.tokenizer_path = argc > 2 ? argv[2] : "/path/to/your/tokenizer.model"; // TODO: replace
    init_cfg.sampling_mode  = AML_LLM_TOP_P;
    init_cfg.top_k          = 0;
    init_cfg.top_p          = 0.9f;
    init_cfg.temperature    = 0.7f;
    init_cfg.repeat_penalty = 1.0f;

    if (aml_llm_init(&ctx, &init_cfg, result_callback) != AML_LLM_Status_Success) {
        std::cerr << "Init failed: aml_llm_init\n";
        return -1;
    }

    // Define tools JSON exposed to model
    const char* tools_json = R"JSON(
[
  {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "City name"
        }
      },
      "required": ["city"]
    }
  }
]
)JSON";

    const char* tool_response_tag = "tool_result";

       // Mode select: "internal" (single run) or default "external" (two runs)
    bool use_internal = (argc > 3 && std::string(argv[3]) == "internal");

    if (aml_llm_enable_function_calling(
            ctx,
            /*system_prompt=*/nullptr,
            tools_json,
            tool_response_tag) != AML_LLM_Status_Success) {
        std::cerr << "Enable function calling failed: aml_llm_enable_function_calling\n";
        aml_llm_uninit(ctx);
        return -1;
    }

    AML_LLMInput user_input;
    std::memset(&user_input, 0, sizeof(user_input));
    user_input.input_type   = AML_LLM_INPUT_PROMPT;
    user_input.prompt_input = "What's the weather in Hangzhou now?";

    AML_LLMRunConfig run_cfg;
    std::memset(&run_cfg, 0, sizeof(run_cfg));
    run_cfg.run_mode       = AML_LLM_RUN_GENERATE;
    run_cfg.retain_history = 0;

    if (use_internal) {
        std::cout << "[Mode] Internal tool callback (single run)\n";
        if (aml_llm_register_tool_callback(ctx, tool_callback, nullptr) != AML_LLM_Status_Success) {
            std::cerr << "Register tool callback failed: aml_llm_register_tool_callback\n";
            aml_llm_uninit(ctx);
            return -1;
        }

        std::cout << "[User] " << user_input.prompt_input << "\n[Assistant] ";
        DemoUserData ud; ud.printed = true;
        AML_LLMRetStatus st = aml_llm_run(ctx, &user_input, &run_cfg, &ud);
        if (st != AML_LLM_Status_Success) {
            std::cerr << "\naml_llm_run failed\n";
        }

    } else {
        std::cout << "[Mode] External tool execution (two runs)\n";
        // First pass: ask the question and collect model output containing <tool_call>{...}
        std::cout << "[User] " << user_input.prompt_input << "\n[Model] Planning tool calls...\n";
        DemoUserData ud1; ud1.printed = true;
        AML_LLMRetStatus st1 = aml_llm_run(ctx, &user_input, &run_cfg, &ud1);
        if (st1 != AML_LLM_Status_Success) {
            std::cerr << "First run aml_llm_run failed\n";
            aml_llm_uninit(ctx);
            return -1;
        }

        // Extract tool calls
        auto calls = parse_tool_calls(ud1.buffer);
        if (calls.empty()) {
            std::cerr << "No tool calls parsed. First run output:\n" << ud1.buffer << "\n";
            aml_llm_uninit(ctx);
            return -1;
        }

        // Execute tools externally and build JSON array
        std::string tool_results = build_tool_results_json(calls);
        std::cout << "[External] Tool execution complete, results: " << tool_results << "\n";

        // Second pass: feed tool results as prompt; SDK stitches full second-run prompt


        AML_LLMInput tool_input;
        std::memset(&tool_input, 0, sizeof(tool_input));
        tool_input.input_type = AML_LLM_INPUT_PROMPT;

        // ===================================================
        // 2️⃣ External Mode: Second run after tool execution
        // ===================================================
        std::string system_prompt =
            "You are a helpful assistant. "
            "Use the provided tool results to answer the user's question naturally.";

        // prefix: optional tool results block
        std::string prompt_prefix;
        prompt_prefix += "<|im_start|>user\n";
        prompt_prefix += user_input.prompt_input;  // e.g. "What's the weather in Hangzhou now?"
        prompt_prefix += "\n<|im_end|>\n";
        prompt_prefix += "<|im_start|>tool\n";
        prompt_prefix += tool_results;  // e.g. [{"city":"Hangzhou","temperature":"25C","condition":"Sunny"}]
        prompt_prefix += "\n<|im_end|>\n<|im_start|>assistant\n";

        // postfix: normally empty for direct answer
        std::string prompt_postfix = "";

        // // set a new chat template for answer mode
        aml_llm_set_chat_template(
            ctx,
            system_prompt.c_str(),
            prompt_prefix.c_str(),
            prompt_postfix.c_str()
        );

        // prepare input
        AML_LLMInput answer_input{};
        answer_input.input_type = AML_LLM_INPUT_PROMPT;
        answer_input.prompt_input = "";  // no user text, context is already in template

        // run again

        aml_llm_run(ctx, &answer_input, &run_cfg, &ud1);
        // printf("Final Result: %s \n", ud2.buffer.c_str());
    }

    aml_llm_uninit(ctx);
    return 0;
}
