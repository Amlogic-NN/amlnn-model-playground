# LLM Examples

## Resource Requirements

| Model | CPU | NPU | GPU |
| :--- | :--- | :--- | :--- |
| Qwen(0.5B) | Minimum cores: 4<br>DDR: 4G (2G reserved for NN) | At least 3.2T | NO |
| Qwen(1.8B) | Minimum cores: 4<br>DDR: 8G (6G~6.5G reserved for NN) | At least 3.2T | NO |
| Gemma(2B) | Minimum cores: 4<br>DDR: 8G (5.5G~6G reserved for NN) | At least 3.2T | NO |


 ## Performance 

ADLA2: A311D2_3.2T / S905X5_4T

| LLM Model | SOC | Dtype | Seqlen | Max_Context | New_Tokens | TTFT(ms) | Tokens/s | memory(G) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DeepSeek-R1 | A311D2 | w8a8 | 64 | 320 | 256 | 927.79 | 4.95 | 1.99 |
| DeepSeek-R1 | S905X5 | w8a8 | 64 | 320 | 256 | 514.86 | 4.47 | 1.73 |
| Gemma-2B | A311D2 | w8a8 | 64 | 320 | 256 | 846.66 | 2.64 | 3.93 |
| Gemma-2B | S905X5 | w8a8 | 64 | 320 | 256 | 482.92 | 3.08 | 2.77 |
| Gemma-3-1B | A311D2 | w8a8 | 64 | 320 | 256 | 702.88 | 5.08 | 1.9 |
| Gemma-3-1B | S905X5 | w8a8 | 64 | 320 | 256 | 468.97 | 6.44 | 1.38 |
| Llama3.2_1B | A311D2 | w8a8 | 64 | 320 | 256 | 711.64 | 5.92 | 1.69 |
| Llama3.2_1B | S905X5 | w8a8 | 64 | 320 | 256 | 695.92 | 5.42 | 1.5 |
| Qwen1.5_1.8B | A311D2 | w8a8 | 64 | 320 | 256 | 794.50 | 4.52 | 2.2 |
| Qwen1.5_1.8B | S905X5 | w8a8 | 64 | 320 | 256 | 983.93 | 4.47 | 1.9 |
| Qwen2.5_0.5B | A311D2 | w8a8 | 64 | 320 | 256 | 400.44 | 10.50 | 0.88 |
| Qwen2.5_0.5B | S905X5 | w8a8 | 64 | 320 | 256 | 400.37 | 10.97 | 0.66 |
| Qwen2.5_1.5B | A311D2 | w8a8 | 64 | 320 | 256 | 882.49 | 3.94 | 2.37 |
| Qwen2.5_1.5B | S905X5 | w8a8 | 64 | 320 | 256 | 874.06 | 4.16 | 1.76 |
| TinyLlama-1.1B-Chat-v1.0 | A311D2 | w8a8 | 64 | 320 | 256 | 763.07 | 6.51 | 1.31 |
| TinyLlama-1.1B-Chat-v1.0 | S905X5 | w8a8 | 64 | 320 | 256 | 1161.82 | 5.85 | 1.15 |
| TinyLlama-1.1B-Chat-v0.4 | A311D2 | w8a8 | 64 | 320 | 256 | 740.02 | 6.38 | 1.31 |
| TinyLlama-1.1B-Chat-v0.4 | S905X5 | w8a8 | 64 | 320 | 256 | 733.01 | 6.28 | 1.11 |


## Result
![llm-result](./model/llm_result.png)