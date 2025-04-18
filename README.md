# 大语言模型API基准测试工具

这是一个用于测试各种大语言模型API性能的Python工具。目前支持OpenAI兼容格式的API接口，可以测量以下指标：

- **首字延迟**：从发送请求到收到第一个token的时间
- **Token吞吐量**：每秒产生的token数量

## 安装

### 使用 pip 安装

```bash
# 从GitHub安装
git clone https://github.com/yourusername/llm_api_benchmark.git
cd llm_api_benchmark
pip install -e .

# 或直接从PyPI安装（未上传）
# pip install llm-api-benchmark
```

### 使用 uv 安装 (推荐)

[uv](https://github.com/astral-sh/uv) 是一个更快的 Python 包管理器和解析器。

```bash
# 首先创建并激活虚拟环境
uv venv
source .venv/bin/activate  # Linux/macOS
# 或者在Windows上
# .venv\Scripts\activate

# 从GitHub安装
git clone https://github.com/yourusername/llm_api_benchmark.git
cd llm_api_benchmark
uv pip install -e .

# 或直接从PyPI安装（未上传）
# uv pip install llm-api-benchmark
```

## 使用方法

### 测试单个API

```bash
# 使用安装后的命令行工具
llm-api-benchmark single --api_key YOUR_API_KEY

# 或者直接运行模块
python -m llm_api_benchmark single --api_key YOUR_API_KEY
```

完整参数：

```bash
llm-api-benchmark single \
  --api_url "https://api.openai.com/v1/chat/completions" \
  --api_key "YOUR_API_KEY" \
  --model "gpt-3.5-turbo" \
  --prompt "解释量子力学和相对论之间的关系，并给出三个实际应用的例子。" \
  --runs 3 \
  --output "results.json"
```

### 批量测试多个API

批量测试允许您通过TOML配置文件同时测试多个LLM API，并自动生成对比报告。

1. 首先创建一个配置文件，例如 `config.toml`:

```toml
# LLM API 批量测试配置文件

[general]
# 通用测试参数
prompt = "解释量子力学和相对论之间的关系，并给出三个实际应用的例子。"
runs = 3  # 每个测试运行次数
output_dir = "./results"  # 结果输出目录
report_file = "benchmark_report.md"  # 最终报告文件名

# 定义多个API配置
[[apis]]
name = "OpenAI GPT-3.5"
url = "https://api.openai.com/v1/chat/completions"
key = "YOUR_OPENAI_API_KEY"  # 替换为实际的API密钥
model = "gpt-3.5-turbo"

[[apis]]
name = "OpenAI GPT-4"
url = "https://api.openai.com/v1/chat/completions"
key = "YOUR_OPENAI_API_KEY"  # 替换为实际的API密钥
model = "gpt-4"
```

2. 运行批量测试:

```bash
llm-api-benchmark batch --config config.toml
```

这将测试配置中指定的所有API，并在results目录下生成一个Markdown格式的对比报告。

### 参数说明

单个API测试参数:
- `--api_url`: API端点URL (默认为OpenAI的聊天补全API)
- `--api_key`: API密钥（必需）
- `--model`: 要测试的模型名称（默认为"gpt-3.5-turbo"）
- `--prompt`: 测试用的提示词
- `--runs`: 每项测试运行的次数（默认为3）
- `--output`: 结果输出的JSON文件路径（可选）

批量测试参数:
- `--config`: TOML配置文件路径（必需）

## 测试其他兼容OpenAI格式的API

本工具支持测试任何兼容OpenAI格式的API，包括但不限于：

- OpenAI API
- Azure OpenAI
- Claude API (使用OpenAI兼容模式)
- 本地部署的LLM (例如通过LM Studio或其他提供OpenAI兼容接口的服务)

只需要修改`--api_url`参数或在配置文件中指定相应的API端点即可。

## 输出示例

### 单个API测试JSON输出

```json
{
  "model": "gpt-3.5-turbo",
  "api_url": "https://api.openai.com/v1/chat/completions",
  "timestamp": "2023-11-15T14:32:45.123456",
  "prompt_length": 60,
  "runs": 3,
  "first_token_latency": 0.512,
  "token_throughput": 42.15
}
```

### 批量测试报告示例

批量测试会生成一个Markdown格式的对比报告，包含所有测试结果的对比表格和详细数据。

## 开发

### 项目结构

```
llm_api_benchmark/
│
├── src/                          # 源代码
│   └── llm_api_benchmark/        # 主包
│       ├── __init__.py           # 包初始化
│       ├── __main__.py           # 入口点
│       ├── benchmark.py          # 核心基准测试功能
│       ├── batch.py              # 批量测试功能
│       └── cli.py                # 命令行接口
│
├── tests/                        # 测试目录
│   ├── __init__.py
│   └── test_benchmark.py         # 单元测试
│
├── config.toml                   # 批量测试配置示例
├── pyproject.toml                # 项目配置和依赖
├── README.md                     # 说明文档
├── .gitignore                    # Git忽略文件
└── requirements.txt              # 依赖列表（兼容旧版本）
```

### 依赖管理

使用 uv 进行依赖管理：

```bash
# 创建并激活虚拟环境
uv venv
source .venv/bin/activate  # Linux/macOS
# 或者在Windows上
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

- 添加更多性能指标测量
- 支持更多API接口格式
- 添加批量测试和对比功能
- 添加可视化报告生成