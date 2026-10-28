[EN](README.md) | **CN**

# LLM 示例

使用 LLMSDK C++ 运行时或可选的 Python 工具包，在 Amlogic 设备上运行大语言模型推理。


## 模型与 SDK

| 资源 | 说明 |
| :--- | :--- |
| [ADLA Hub](https://huggingface.co/Amlogic-NN/amlnn-adla-models) | 预量化的 ADLA 模型 |
| [LLMSDK](https://github.com/Amlogic-NN/amlnn-toolkit/tree/main/amlnn_runtime/llm_runtime) | C++ 运行时库和头文件 |
| [amlllm_edge_toolkit_lite](https://github.com/Amlogic-NN/amlnn-toolkit/tree/main/amlnn_toolkit_lite/amlllm_edge_toolkit_lite) | 可选的 Python 推理工具包 |

运行示例前，请先下载适用于目标设备的 ADLA 模型以及与之匹配的 tokenizer 资源。

## 工作区准备

本指南假设采用以下目录结构：

```text
workspace/
├── amlnn-model-playground/
│   └── examples/LLMs/
│       ├── cpp/
│       └── py/
└── amlnn-toolkit/
```

在已有的 `amlnn-model-playground` 代码库根目录下，将 SDK 仓库克隆到其上级目录：

```bash
git clone https://github.com/Amlogic-NN/amlnn-toolkit.git ../amlnn-toolkit
```

如果 SDK 仓库已存在，请跳过此步骤。

## C++ 示例 — Android

### 1. 配置 Android NDK

将 `NDK_PATH` 设置为你的 Android NDK 安装路径。以下示例使用 NDK r25c：

```bash
export NDK_PATH=/your/ndk/path/android-ndk-r25c
export PATH="$NDK_PATH:$PATH"
```

### 2. 构建示例

在 `amlnn-model-playground` 仓库根目录下：

```bash
cd examples/LLMs/cpp
./build-android.sh
```

### 3. 部署到设备

将生成的可执行文件、ADLA 模型和 tokenizer 推送到 `/data/local/tmp`。

在主机上运行以下命令，并将每个占位符替换为实际的本地路径：

```bash
adb push <executable_path>/demo_llm_main /data/local/tmp/
adb push <model_path>/Qwen2.5-0.5B-Instruct_quant_i8_a311d2.adla /data/local/tmp/
adb push <tokenizer_path>/tokenizer.json /data/local/tmp/
```

如果设备上尚未提供 LLMSDK 动态库，也请一并部署：

```bash
adb push <llmsdk_path>/llmsdk.so /data/local/tmp/
```

请使用 SDK 软件包中提供的库文件名。

### 4. 运行推理

打开设备 shell 并运行：

```bash
adb shell
cd /data/local/tmp
chmod +x demo_llm_main
export LD_LIBRARY_PATH="/vendor/lib64:/data/local/tmp${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

./demo_llm_main Qwen2.5-0.5B-Instruct_quant_i8_a311d2.adla tokenizer.json
```

上面的模型文件名是 A311D2 的示例。请将其替换为目标设备对应的模型包。

## Python 示例 — Arm64 Ubuntu / Debian

### 前置条件

| 组件 | 要求 |
| :--- | :--- |
| 架构 | Linux aarch64 |
| 操作系统 | Ubuntu 22.04 或 Debian |
| Python | 3.10 |
| NPU 驱动/运行时 | 兼容的 ADLA 软件；LIBADLA 1.7.x 或更高版本 |

> [!CAUTION]
> 根据本指南所用的发布状态，官方系统镜像尚未发布。请使用已安装所需 NPU 驱动和运行时的板卡镜像。

### 1. 检查 ADLA 安装

通过串口控制台或终端在板卡上运行以下命令：

```bash
dmesg | grep -i adla
strings /usr/lib/libadla.so | grep LIBADLA
```

查看内核日志中的驱动信息以及动态库报告的 LIBADLA 版本。本指南要求 LIBADLA 1.7.x 或更高版本。如果库安装在其它位置，请相应调整路径。

### 2. 创建 Python 环境

如需要，请安装 Miniforge：

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh
```

按安装程序的提示操作，如有需要请重新打开 shell。然后创建并激活环境：

```bash
conda create -n amlllm_310 python=3.10 -y
conda activate amlllm_310
```

### 3. 安装 Python 包

在 `amlnn-model-playground` 仓库根目录下，从相邻的 SDK 代码库中定位适用于 Python 3.10 / aarch64 的 wheel 包：

```bash
find ../amlnn-toolkit -type f -name 'amlllm-1.0.0-cp310-cp310-linux_aarch64.whl'
```

使用该命令返回的路径进行安装：

```bash
python -m pip install <wheel_path>/amlllm-1.0.0-cp310-cp310-linux_aarch64.whl
```

如果你的 SDK 版本提供的是其它包版本，请使用与之匹配的 Python 3.10 / aarch64 wheel 包。各软件包的具体安装说明请参阅 [Python 工具包文档](https://github.com/Amlogic-NN/amlnn-toolkit/tree/main/amlnn_toolkit_lite/amlllm_edge_toolkit_lite)。

### 4. 运行对话示例

在 `amlnn-model-playground` 仓库根目录下：

```bash
cd examples/LLMs/py

python simple_chat.py \
  --model <model_path> \
  --tokenizer <tokenizer_path> \
  [options]
```

将占位符替换为实际路径。按下方所示，选择与你的模型匹配的模型模板。

### 命令行参数

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--model` | **必填。** ADLA 模型文件的路径 | — |
| `--tokenizer` | **必填。** tokenizer 资源的路径 | — |
| `--model-type` | 对话模板：`none`、`qwen`、`deepseek`、`gemma`、`gemma3`、`llama`、`tiny_llama`、`tiny_llama_v0_4`、`phi_1_5` 或 `phi_2` | `none` |
| `--sampling-mode` | 采样模式：`argmax`、`top_p` 或 `top_k` | `argmax` |
| `--top-k` | Top-K 采样参数 | `3` |
| `--top-p` | Top-P 采样参数 | `0.9` |
| `--temperature` | Softmax 温度 | `1.0` |
| `--repeat-penalty` | 重复惩罚系数 | `1.1` |
| `--loglevel` | 日志级别：`DEBUG`、`INFO`、`WARNING` 或 `ERROR` | `ERROR` |

### 用法示例

**Qwen 使用默认采样配置：**

```bash
python simple_chat.py \
  --model Qwen2.5-0.5B-Instruct_quant_i8_a311d2.adla \
  --tokenizer tokenizer.json \
  --model-type qwen
```

**Top-P 采样：**

```bash
python simple_chat.py \
  --model Qwen2.5-0.5B-Instruct_quant_i8_a311d2.adla \
  --tokenizer tokenizer.json \
  --model-type qwen \
  --sampling-mode top_p \
  --top-p 0.9 \
  --temperature 0.8
```

**Top-K 采样：**

```bash
python simple_chat.py \
  --model Qwen2.5-0.5B-Instruct_quant_i8_a311d2.adla \
  --tokenizer tokenizer.json \
  --model-type qwen \
  --sampling-mode top_k \
  --top-k 5
```

### 交互命令

对话界面启动后，可使用以下输入：

| 输入 | 操作 |
| :--- | :--- |
| 输入文本后按 Enter | 提交提示词并接收流式响应 |
| `new_talk` | 清空对话历史并开始新对话 |
| `break` | 请求中断当前响应 |
| `exit` | 退出程序 |
| `Ctrl+C` | 发送中断信号 |

## 示例输出

![LLM 推理示例输出](./model/llm-result0.png)

