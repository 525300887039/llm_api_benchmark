"""命令行界面模块，用于从命令行运行基准测试."""

import argparse
import json
import sys
from .benchmark import LLMAPIBenchmark

# 导入批量测试模块
try:
    from .batch import run_batch_benchmark
    BATCH_AVAILABLE = True
except ImportError:
    BATCH_AVAILABLE = False


def run_benchmark_cli():
    """
    运行命令行工具.

    Returns:
        dict: 测试结果
    """
    parser = argparse.ArgumentParser(description="大语言模型API基准测试工具")

    # 添加子命令解析器
    subparsers = parser.add_subparsers(dest="command", help="命令")

    # 单一API测试的子命令
    single_parser = subparsers.add_parser("single", help="测试单个API")
    single_parser.add_argument("--api_url", default="https://api.openai.com/v1/chat/completions", help="API端点URL")
    single_parser.add_argument("--api_key", required=True, help="API密钥")
    single_parser.add_argument("--model", default="gpt-3.5-turbo", help="要测试的模型名称")
    single_parser.add_argument("--prompt", default="解释量子力学和相对论之间的关系，并给出三个实际应用的例子。", help="测试提示词")
    single_parser.add_argument("--runs", type=int, default=3, help="每项测试运行次数")
    single_parser.add_argument("--output", help="结果输出的JSON文件路径")
    single_parser.add_argument("--api_type", default="openai", choices=["openai", "claude", "azure"],
                               help="API类型 (默认: openai)")

    # 批量测试的子命令
    if BATCH_AVAILABLE:
        batch_parser = subparsers.add_parser("batch", help="批量测试多个API")
        batch_parser.add_argument("--config", required=True, help="TOML配置文件路径")

    # 可视化报告的子命令
    report_parser = subparsers.add_parser("report", help="启动交互式可视化报告")
    report_parser.add_argument("--results_dir", default="./results", help="测试结果目录 (默认: ./results)")
    report_parser.add_argument("--port", type=int, default=8501, help="Streamlit 端口 (默认: 8501)")

    # 解析参数
    args = parser.parse_args()

    # 如果没有提供命令，默认为single
    if not args.command:
        if len(sys.argv) > 1:
            args.command = "single"
        else:
            parser.print_help()
            return None

    # 根据命令执行相应功能
    if args.command == "single":
        benchmark = LLMAPIBenchmark(args.api_url, args.api_key, args.model, args.api_type)
        results = benchmark.run_comprehensive_benchmark(args.prompt, args.runs)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到 {args.output}")

        return results

    elif args.command == "batch" and BATCH_AVAILABLE:
        report_path = run_batch_benchmark(args.config)
        print(f"批量测试完成，报告保存在: {report_path}")
        return {"report_path": report_path}

    elif args.command == "report":
        _launch_report(args.results_dir, args.port)
        return None

    else:
        parser.print_help()
        return None


def _launch_report(results_dir: str, port: int):
    """启动 Streamlit 可视化报告."""
    import subprocess
    import os

    visualize_path = os.path.join(os.path.dirname(__file__), "visualize.py")
    print(f"正在启动可视化报告服务 (端口: {port})...")
    print(f"数据目录: {results_dir}")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", visualize_path,
        "--server.port", str(port),
        "--", "--results_dir", results_dir,
    ])


def main():
    """入口点函数，用于setup.py配置."""
    run_benchmark_cli()


# 兼容旧版本的命令行
def legacy_cli():
    """兼容旧版本的命令行接口."""
    # 检查是否使用旧格式命令行
    if len(sys.argv) > 1 and not sys.argv[1].startswith(("single", "batch", "report")):
        # 在命令之前插入"single"
        sys.argv.insert(1, "single")
    main()


if __name__ == "__main__":
    main()