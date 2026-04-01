"""批量测试模块，用于批量测试多个LLM API并生成对比报告."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Any

import tomli

from .benchmark import LLMAPIBenchmark
from .constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPT,
    DEFAULT_REPORT_FILE,
    DEFAULT_RETRY_DELAY,
    DEFAULT_RUNS,
    DEFAULT_WARMUP_RUNS,
)


class BatchBenchmark:
    """批量测试多个LLM API并生成对比报告."""

    def __init__(self, config_file: str):
        """
        初始化批量测试对象.

        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = self._load_config()
        general_config = self.config.get("general", {})
        self.prompt = general_config.get("prompt", DEFAULT_PROMPT)
        self.runs = general_config.get("runs", DEFAULT_RUNS)
        self.output_dir = general_config.get("output_dir", DEFAULT_OUTPUT_DIR)
        self.report_file = general_config.get("report_file", DEFAULT_REPORT_FILE)
        self.timeout = general_config.get("timeout")
        self.parallel = general_config.get("parallel", 1)
        self.warmup_runs = general_config.get("warmup_runs", DEFAULT_WARMUP_RUNS)
        self.max_retries = general_config.get("max_retries", DEFAULT_MAX_RETRIES)
        self.retry_delay = general_config.get("retry_delay", DEFAULT_RETRY_DELAY)
        self.apis = self.config.get("apis", [])
        self.results = []

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_config(self) -> Dict[str, Any]:
        """
        加载TOML配置文件.

        Returns:
            Dict: 配置信息
        """
        try:
            with open(self.config_file, "rb") as f:
                return tomli.load(f)
        except Exception as e:
            raise ValueError(f"加载配置文件失败: {e}")

    def _run_single_api_test(
        self,
        api_config: Dict[str, Any],
        index: int,
    ) -> Dict[str, Any] | None:
        """执行单个 API 的基准测试，返回结果 dict 或 None（失败时）."""
        name = api_config.get("name", f"API_{index+1}")
        url = api_config.get("url")
        key = api_config.get("key")
        model = api_config.get("model")
        api_type = api_config.get("type", "openai")
        parallel_output = getattr(self, "_parallel_output", False)

        if not url or not model:
            if parallel_output:
                print(f"[{name}] 警告: API '{name}' 配置不完整，已跳过")
            else:
                print(f"警告: API '{name}' 配置不完整，已跳过")
            return None

        if parallel_output:
            print(f"\n\n[{name}] {'=' * 80}")
            print(f"[{name}] 正在测试 API: {name}")
            print(f"[{name}] {'=' * 80}\n")
        else:
            print(f"\n\n{'=' * 80}")
            print(f"正在测试 API: {name}")
            print(f"{'=' * 80}\n")

        try:
            benchmark = LLMAPIBenchmark(
                url,
                key,
                model,
                api_type,
                timeout=self.timeout,
                warmup_runs=self.warmup_runs,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
            )
            result = benchmark.run_comprehensive_benchmark(self.prompt, self.runs)

            # 添加API名称和测试时间
            result["name"] = name
            result["test_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 保存结果到文件
            result_file = os.path.join(
                self.output_dir, f"{name.replace(' ', '_').lower()}_{int(time.time())}.json"
            )
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            if parallel_output:
                print(f"\n[{name}] 结果已保存到: {result_file}")
            else:
                print(f"\n结果已保存到: {result_file}")

            return result
        except Exception as e:
            if parallel_output:
                print(f"[{name}] 测试 API '{name}' 时发生错误: {e}")
            else:
                print(f"测试 API '{name}' 时发生错误: {e}")
            return None

    def run_batch_tests(self) -> List[Dict[str, Any]]:
        """
        运行批量测试.

        Returns:
            List[Dict]: 所有API的测试结果
        """
        if not self.apis:
            raise ValueError("配置文件中未找到API配置")

        # 运行测试
        results = []
        self._parallel_output = False

        if self.parallel == 1:
            for i, api_config in enumerate(self.apis):
                result = self._run_single_api_test(api_config, i)
                if result is not None:
                    results.append(result)
        else:
            if self.parallel == 0:
                max_workers = len(self.apis)
            elif self.parallel > 1:
                max_workers = self.parallel
            else:
                raise ValueError("general.parallel 必须大于等于 0")

            self._parallel_output = True
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._run_single_api_test,
                        api_config,
                        i,
                    ): api_config.get("name", f"API_{i+1}")
                    for i, api_config in enumerate(self.apis)
                }
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        print(f"测试 API '{name}' 时发生未预期的错误: {e}")

        self.results = results
        return results

    def generate_markdown_report(self) -> str:
        """
        生成Markdown格式的对比报告.

        Returns:
            str: 报告的文件路径
        """
        if not self.results:
            raise ValueError("没有可用的测试结果")

        report_path = os.path.join(self.output_dir, self.report_file)

        # 创建报告内容
        lines = []
        lines.append("# LLM API 基准测试对比报告")
        lines.append("")
        lines.append(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **测试提示词**: {self.prompt[:100]}...")
        lines.append(f"- **每项测试运行次数**: {self.runs}")
        lines.append("")

        # 创建性能对比表格
        lines.append("## 性能对比")
        lines.append("")
        lines.append(
            "| 模型名称 | 首字延迟 (秒) | P90延迟 (秒) | 吞吐量 (tokens/秒) | 流式吞吐量 (字符/秒) | 总响应时间 (秒) | API端点 |"
        )
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | :--- |")

        # 按首字延迟排序（从快到慢）
        sorted_results = sorted(
            self.results, key=lambda x: x.get("first_token_latency", float("inf"))
        )

        for result in sorted_results:
            name = result.get("name", "未知")
            latency = result.get("first_token_latency", 0)
            latency_stats = result.get("first_token_latency_stats", {})
            p90 = latency_stats.get("p90", 0)
            throughput = result.get("token_throughput", 0)
            streaming_throughput = result.get("streaming_throughput", 0)
            total_time = result.get("total_time", 0)
            api_url = result.get("api_url", "")

            lines.append(
                f"| {name} | {latency:.3f} | {p90:.3f} | {throughput:.2f} | "
                f"{streaming_throughput:.2f} | {total_time:.2f} | {api_url} |"
            )

        lines.append("")

        # 添加详细结果部分
        lines.append("## 详细测试结果")
        lines.append("")

        for result in sorted_results:
            name = result.get("name", "未知")
            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"- **测试时间**: {result.get('test_time', '未知')}")
            lines.append(f"- **模型**: {result.get('model', '未知')}")
            lines.append(f"- **API类型**: {result.get('api_type', 'openai')}")
            lines.append(f"- **API端点**: {result.get('api_url', '未知')}")
            lines.append(f"- **提示词长度**: {result.get('prompt_length', 0)} 字符")
            lines.append("")

            # 首字延迟统计
            ls = result.get("first_token_latency_stats", {})
            if ls:
                lines.append("**首字延迟统计:**")
                lines.append("")
                lines.append(f"| 平均 | 最小 | 最大 | 中位数 | P90 | P99 | 标准差 |")
                lines.append(f"| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
                lines.append(
                    f"| {ls.get('avg',0):.3f}s | {ls.get('min',0):.3f}s | {ls.get('max',0):.3f}s "
                    f"| {ls.get('median',0):.3f}s | {ls.get('p90',0):.3f}s "
                    f"| {ls.get('p99',0):.3f}s | {ls.get('std_dev',0):.3f}s |"
                )
                lines.append("")

            # 吞吐量统计
            ts = result.get("token_throughput_stats", {})
            if ts:
                lines.append("**吞吐量统计:**")
                lines.append("")
                lines.append(f"| 平均 | 最小 | 最大 | 中位数 | P90 | P99 | 标准差 |")
                lines.append(f"| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
                lines.append(
                    f"| {ts.get('avg',0):.2f} | {ts.get('min',0):.2f} | {ts.get('max',0):.2f} "
                    f"| {ts.get('median',0):.2f} | {ts.get('p90',0):.2f} "
                    f"| {ts.get('p99',0):.2f} | {ts.get('std_dev',0):.2f} |"
                )
                lines.append("")

            sts = result.get("streaming_throughput_stats", {})
            if sts:
                lines.append("**流式吞吐量统计 (字符/秒):**")
                lines.append("")
                lines.append(f"| 平均 | 最小 | 最大 | 中位数 | P90 | P99 | 标准差 |")
                lines.append(f"| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
                lines.append(
                    f"| {sts.get('avg',0):.2f} | {sts.get('min',0):.2f} | {sts.get('max',0):.2f} "
                    f"| {sts.get('median',0):.2f} | {sts.get('p90',0):.2f} "
                    f"| {sts.get('p99',0):.2f} | {sts.get('std_dev',0):.2f} |"
                )
                lines.append("")

            # 总响应时间统计
            tt = result.get("total_time_stats", {})
            if tt:
                lines.append("**总响应时间统计:**")
                lines.append("")
                lines.append(f"| 平均 | 最小 | 最大 | 中位数 | P90 | P99 | 标准差 |")
                lines.append(f"| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
                lines.append(
                    f"| {tt.get('avg',0):.2f}s | {tt.get('min',0):.2f}s | {tt.get('max',0):.2f}s "
                    f"| {tt.get('median',0):.2f}s | {tt.get('p90',0):.2f}s "
                    f"| {tt.get('p99',0):.2f}s | {tt.get('std_dev',0):.2f}s |"
                )
                lines.append("")

        # 写入文件
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"\n对比报告已生成: {report_path}")
        return report_path


def run_batch_benchmark(config_file: str) -> str:
    """
    运行批量测试并生成报告的便捷函数.

    Args:
        config_file: 配置文件路径

    Returns:
        str: 报告文件路径
    """
    batch = BatchBenchmark(config_file)
    batch.run_batch_tests()
    report_path = batch.generate_markdown_report()

    print(
        f"\n提示: 运行 'llm-api-benchmark report --results_dir {batch.output_dir}' "
        "可启动交互式可视化报告"
    )

    return report_path
