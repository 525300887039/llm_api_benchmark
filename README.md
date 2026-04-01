# 大语言模型 API 基准测试工具

这是一个用于测试各种大语言模型 API 性能的 Python 工具，支持单模型测试、批量对比和 Streamlit 可视化报告。

当前可测量的指标包括：

- **首字延迟**：从发送请求到收到第一个 token 的时间
- **Token 吞吐量**：每秒生成的 token 数量
- **总响应时间**：非流式请求的完整响应耗时
- **详细统计字段**：`avg`、`min`、`max`、`median`、`p90`、`p99`、`std_dev`、`raw`

## 安装

### 使用 pip 安装

```bash
# 从 GitHub 安装
git clone https://github.com/yourusername/llm_api_benchmark.git
cd llm_api_benchmark
pip install -e .

# 或直接从 PyPI 安装（未上传）
# pip install llm-api-benchmark
```

### 使用 uv 安装（推荐）

[uv](https://github.com/astral-sh/uv) 是一个更快的 Python 包管理器和解析器。

```bash
# 首先创建并激活虚拟环境
uv venv
source .venv/bin/activate  # Linux/macOS
# 或者在 Windows 上
# .venv\Scripts\activate

# 从 GitHub 安装
git clone https://github.com/yourusername/llm_api_benchmark.git
cd llm_api_benchmark
uv pip install -e .

# 或直接从 PyPI 安装（未上传）
# uv pip install llm-api-benchmark
```

## 支持的 API 格式

本工具已原生支持四种 API 格式：

- **OpenAI 兼容格式**：包括 OpenAI，以及 SiliconFlow、DeepSeek、智谱 AI、月之暗面（Moonshot / Kimi）等提供 OpenAI 兼容接口的平台
- **Anthropic Claude 原生格式**：使用 Anthropic `messages` API
- **Azure OpenAI 格式**：使用 Azure OpenAI 的请求头和请求体格式
- **Google Gemini 原生格式**：使用 Gemini `generateContent` / `streamGenerateContent` 接口

单次测试通过 `--api_type` 选择接口格式；批量测试通过配置文件中的 `type` 字段指定。

## 使用方法

### 命令概览

```bash
llm-api-benchmark single --api_key YOUR_API_KEY
llm-api-benchmark batch --config config.toml
llm-api-benchmark report --results_dir ./results
```

### 测试单个 API

```bash
# 使用安装后的命令行工具
llm-api-benchmark single --api_key YOUR_API_KEY

# 或者直接运行模块
python -m llm_api_benchmark single --api_key YOUR_API_KEY
```

完整示例：

```bash
llm-api-benchmark single \
  --api_url "https://api.openai.com/v1/chat/completions" \
  --api_key "YOUR_API_KEY" \
  --model "gpt-4o-mini" \
  --api_type openai \
  --prompt "解释量子力学和相对论之间的关系，并给出三个实际应用的例子。" \
  --runs 3 \
  --output "results.json"
```

如果测试其他接口格式，只需要切换 `--api_type` 和对应的 `--api_url`：

```bash
# Anthropic Claude 原生格式
llm-api-benchmark single \
  --api_url "https://api.anthropic.com/v1/messages" \
  --api_key "YOUR_ANTHROPIC_API_KEY" \
  --model "claude-3-5-sonnet-20241022" \
  --api_type claude

# Azure OpenAI
llm-api-benchmark single \
  --api_url "https://your-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-01" \
  --api_key "YOUR_AZURE_API_KEY" \
  --model "gpt-4o" \
  --api_type azure

# Google Gemini
llm-api-benchmark single \
  --api_url "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent" \
  --api_key "YOUR_GEMINI_API_KEY" \
  --model "gemini-2.5-flash" \
  --api_type gemini
```

### 批量测试多个 API

批量测试允许通过 TOML 配置文件同时测试多个 LLM API，并自动生成 Markdown 对比报告。仓库内提供了示例配置文件 `config.toml.example`。

可以复制一份并按需修改：

```toml
[general]
prompt = "解释量子力学和相对论之间的关系，并给出三个实际应用的例子。"
runs = 3
# warmup_runs = 1
# max_retries = 2
# retry_delay = 1.0
# parallel = 1
output_dir = "./results"
report_file = "benchmark_report.md"
# timeout = 120

[[apis]]
name = "OpenAI-GPT4o-mini"
url = "https://api.openai.com/v1/chat/completions"
key = "sk-your-openai-api-key"
model = "gpt-4o-mini"
type = "openai"

[[apis]]
name = "Claude-3.5-Sonnet"
url = "https://api.anthropic.com/v1/messages"
key = "sk-ant-your-anthropic-api-key"
model = "claude-3-5-sonnet-20241022"
type = "claude"

[[apis]]
name = "Azure-GPT4o"
url = "https://your-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-01"
key = "your-azure-api-key"
model = "gpt-4o"
type = "azure"

[[apis]]
name = "Gemini-2.5-Flash"
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
key = "your-gemini-api-key"
model = "gemini-2.5-flash"
type = "gemini"
```

运行批量测试：

```bash
llm-api-benchmark batch --config config.toml
```

运行后会在 `results` 目录中生成每个模型的 JSON 结果文件，以及一个 Markdown 对比报告。

### 启动可视化报告

批量测试完成后，可以启动 Streamlit 可视化报告：

```bash
llm-api-benchmark report --results_dir ./results
```

如需自定义端口：

```bash
llm-api-benchmark report --results_dir ./results --port 8501
```

## 参数说明

### `single` 子命令参数

- `--api_url`：API 端点 URL，默认值为 `https://api.openai.com/v1/chat/completions`
- `--api_key`：API 密钥，必需
- `--model`：要测试的模型名称，默认值为 `gpt-3.5-turbo`
- `--prompt`：测试提示词
- `--runs`：每项测试运行次数，默认值为 `3`
- `--warmup_runs`：正式测量前的预热次数，默认值为 `0`
- `--max_retries`：单轮请求失败后的最大重试次数，默认值为 `0`
- `--retry_delay`：初始重试等待秒数，默认值为 `1.0`，后续按指数退避
- `--output`：结果输出的 JSON 文件路径，可选
- `--timeout`：请求超时秒数，默认不设置
- `--api_type`：API 类型，`choices: openai, claude, azure, gemini`，默认值为 `openai`

### `batch` 子命令参数

- `--config`：TOML 配置文件路径，必需

### `report` 子命令参数

- `--results_dir`：测试结果目录，默认值为 `./results`
- `--port`：Streamlit 服务端口，默认值为 `8501`

## 输出示例

### 单个 API 测试 JSON 输出

单次综合测试返回的 JSON 顶层会保留平均值字段，同时包含完整的 `_stats` 详细统计信息：

```json
{
  "model": "gpt-4o-mini",
  "api_url": "https://api.openai.com/v1/chat/completions",
  "api_type": "openai",
  "timestamp": "2026-04-01T12:34:56.123456",
  "prompt_length": 29,
  "runs": 3,
  "warmup_runs": 1,
  "max_retries": 2,
  "first_token_latency": 0.512,
  "token_throughput": 42.15,
  "streaming_throughput": 186.42,
  "total_time": 2.84,
  "first_token_latency_stats": {
    "avg": 0.512,
    "min": 0.481,
    "max": 0.556,
    "median": 0.499,
    "p90": 0.545,
    "p99": 0.555,
    "std_dev": 0.039,
    "raw": [0.481, 0.499, 0.556]
  },
  "token_throughput_stats": {
    "avg": 42.15,
    "min": 39.87,
    "max": 44.62,
    "median": 41.96,
    "p90": 44.09,
    "p99": 44.57,
    "std_dev": 2.38,
    "raw": [39.87, 41.96, 44.62]
  },
  "streaming_throughput_stats": {
    "avg": 186.42,
    "min": 180.12,
    "max": 193.55,
    "median": 185.59,
    "p90": 191.96,
    "p99": 193.39,
    "std_dev": 6.78,
    "raw": [180.12, 185.59, 193.55]
  },
  "total_time_stats": {
    "avg": 2.84,
    "min": 2.61,
    "max": 3.05,
    "median": 2.87,
    "p90": 3.01,
    "p99": 3.05,
    "std_dev": 0.22,
    "raw": [2.61, 2.87, 3.05]
  }
}
```

### 批量测试报告

批量测试会生成：

- 每个模型一个 JSON 结果文件
- 一个 Markdown 格式的对比报告
- 可通过 `report` 子命令打开的 Streamlit 可视化界面

## 开发

### 主要文件结构

```text
llm_api_benchmark/
├── src/
│   └── llm_api_benchmark/
│       ├── __init__.py           # 包初始化
│       ├── __main__.py           # 模块入口，兼容旧版命令行
│       ├── benchmark.py          # 核心基准测试逻辑
│       ├── batch.py              # 批量测试与 Markdown 报告生成
│       ├── providers.py          # API Provider 适配器
│       ├── visualize.py          # Streamlit 可视化报告
│       └── cli.py                # 命令行接口
├── tests/
│   ├── __init__.py
│   ├── test_benchmark.py         # Benchmark 测试
│   ├── test_cli.py               # CLI 测试
│   ├── test_providers.py         # Provider 测试
│   └── test_visualize.py         # 可视化测试
├── config.toml.example           # 批量测试配置示例
├── run_benchmark.py              # 运行辅助脚本
├── llm_api_benchmark_legacy.py   # 旧版兼容脚本
├── pyproject.toml                # 项目配置和依赖
├── README.md                     # 说明文档
├── .gitignore                    # Git 忽略文件
└── requirements.txt              # 依赖列表（兼容旧版本）
```

### 依赖管理

使用 uv 进行依赖管理：

```bash
# 创建并激活虚拟环境
uv venv
source .venv/bin/activate  # Linux/macOS
# 或者在 Windows 上
# .venv\Scripts\activate

# 安装开发依赖
uv pip install -e ".[dev]"

# 运行测试
python -m pytest

# 代码格式化
python -m black src tests
python -m isort src tests
```

## 后续开发计划

- ~~添加更多性能指标测量~~
- ~~支持更多 API 接口格式~~
- ~~添加批量测试和对比功能~~
- ~~添加可视化报告生成~~
- ~~支持 Google Gemini API~~
- ~~流式吞吐量测量~~
- ~~请求重试和超时机制~~
- ~~预热轮次~~
- ~~批量测试并行化~~
- 支持自定义请求头和请求体参数
- 支持从环境变量读取 API Key
- HTML 格式报告导出
