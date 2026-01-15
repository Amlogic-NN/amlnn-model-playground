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


## Compile

### CPP
To compile the CPP project using Android NDK, follow these steps:

1. **Get the llmsdk library and header files**:
   Clone the `amlnn-toolkit` repository to get the necessary libraries for compilation.
   ```bash
   # Clone to the parent directory of amlnn-model-playground
   git clone https://github.com/Amlogic-NN/amlnn-toolkit.git
   ```

2. **Set the NDK path**:
   ```bash
   export NDK_PATH=/your/ndk/path/android-ndk-r25c
   ```

3. **Add NDK to your PATH**:
   ```bash
   export PATH=$NDK_PATH:$PATH
   ```

4. **Compile**:
   Navigate to the `cpp` directory and run `build-android.sh`:
   ```bash
   cd examples/LLMs/cpp
   ./build-android.sh
   ```

5. **Run**:
   Push the compiled executable, model, and tokenizer to your Android device.

   Optional configuration:
   - **Push `llmsdk.so`**: If not already present on the device, push it to `/data/local/tmp`.
   - **Set permissions**:
     ```bash
     chmod +x demo_llm_main
     ```
   - **Set environment variable**:
     ```bash
     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/vendor/lib64/:/data/local/tmp
     ```

   Then execute:
   ```bash
   ./demo_llm_main Qwen2.5-1.5B-Instruct-F16_quant_i8_t7c.adla tokenizer.json
   ```

## Result

| Banner | Inference Result |
| :---: | :---: |
| ![llm-result0](./model/llm-result0.png) | ![llm-result](./model/llm_result.png) |