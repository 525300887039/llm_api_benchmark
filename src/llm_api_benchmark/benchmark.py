"""LLM API基准测试的核心功能."""

import math
import time
import json
import statistics
from datetime import datetime
import requests

from .providers import create_provider


class LLMAPIBenchmark:
    """大语言模型API性能测试类."""

    def __init__(self, api_url, api_key, model, api_type="openai"):
        """
        初始化API基准测试对象.

        Args:
            api_url: API端点URL
            api_key: API密钥
            model: 要测试的模型名称
            api_type: API类型 ("openai", "claude", "azure")
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.api_type = api_type
        self.provider = create_provider(api_type, api_url, api_key, model)

    @staticmethod
    def _compute_stats(data):
        """
        计算统计指标.

        Args:
            data: 数值列表

        Returns:
            dict: 包含 avg, min, max, median, p90, p99, std_dev, raw
        """
        if not data:
            return {"avg": 0, "min": 0, "max": 0, "median": 0,
                    "p90": 0, "p99": 0, "std_dev": 0, "raw": []}

        sorted_data = sorted(data)
        n = len(sorted_data)

        def percentile(pct):
            if n == 1:
                return sorted_data[0]
            k = (n - 1) * (pct / 100.0)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_data[int(k)]
            return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)

        stats = {
            "avg": statistics.mean(data),
            "min": min(data),
            "max": max(data),
            "median": statistics.median(data),
            "p90": percentile(90),
            "p99": percentile(99),
            "std_dev": statistics.stdev(data) if n >= 2 else 0,
            "raw": list(data),
        }
        return stats

    def measure_first_token_latency(self, prompt, num_runs=3):
        """
        测量首字延迟（从发送请求到收到第一个token的时间）.

        Args:
            prompt: 测试用的提示词
            num_runs: 测试运行次数

        Returns:
            dict: 包含统计指标的字典
        """
        latencies = []

        for i in range(num_runs):
            payload = self.provider.build_chat_payload(prompt, stream=True)

            start_time = time.time()
            response = requests.post(
                self.provider.get_request_url(),
                headers=self.provider.get_headers(),
                json=payload,
                stream=True,
            )

            # 读取第一个数据包
            for line in response.iter_lines():
                if self.provider.is_first_content_event(line):
                    first_token_time = time.time()
                    latency = first_token_time - start_time
                    latencies.append(latency)
                    break

            response.close()
            print(f"运行 {i+1}/{num_runs}: 首字延迟 = {latency:.3f}秒")
            time.sleep(1)  # 避免请求过于频繁

        stats = self._compute_stats(latencies)
        print(f"\n平均首字延迟: {stats['avg']:.3f}秒")
        return stats

    def measure_token_throughput(self, prompt, runs=3):
        """
        测量模型的Token吞吐量（tokens/second）和总响应时间.

        Args:
            prompt: 测试用的提示词
            runs: 测试运行次数

        Returns:
            tuple: (throughput_stats, total_time_stats) 两个统计字典
        """
        throughputs = []
        total_times = []

        for i in range(runs):
            payload = self.provider.build_chat_payload(prompt, stream=False)

            start_time = time.time()
            response = requests.post(
                self.provider.get_request_url(),
                headers=self.provider.get_headers(),
                json=payload,
            )
            end_time = time.time()

            elapsed = end_time - start_time
            total_times.append(elapsed)
            response_json = response.json()

            output_tokens = self.provider.parse_token_count(response_json)

            if elapsed > 0 and output_tokens > 0:
                throughput = output_tokens / elapsed
                throughputs.append(throughput)
                print(f"运行 {i+1}/{runs}: 吞吐量 = {throughput:.2f} tokens/秒 "
                      f"(生成了 {output_tokens} tokens，用时 {elapsed:.2f}秒)")

            time.sleep(1)  # 避免请求过于频繁

        throughput_stats = self._compute_stats(throughputs)
        total_time_stats = self._compute_stats(total_times)

        if throughputs:
            print(f"\n平均吞吐量: {throughput_stats['avg']:.2f} tokens/秒")

        return throughput_stats, total_time_stats

    def run_comprehensive_benchmark(self, prompt, runs=3):
        """
        运行综合性能测试.

        Args:
            prompt: 测试用的提示词
            runs: 测试运行次数

        Returns:
            dict: 包含测试结果的字典
        """
        print(f"\n===== 开始对 {self.model} 进行基准测试 =====")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"API URL: {self.api_url}")
        print(f"API 类型: {self.api_type}")
        print(f"测试提示词: {prompt[:50]}... ({len(prompt)} 字符)")
        print(f"运行次数: {runs}")

        print("\n----- 测试首字延迟 -----")
        latency_stats = self.measure_first_token_latency(prompt, runs)

        print("\n----- 测试吞吐量 -----")
        throughput_stats, total_time_stats = self.measure_token_throughput(prompt, runs)

        results = {
            "model": self.model,
            "api_url": self.api_url,
            "api_type": self.api_type,
            "timestamp": datetime.now().isoformat(),
            "prompt_length": len(prompt),
            "runs": runs,
            # 向后兼容：保留顶层平均值
            "first_token_latency": latency_stats["avg"],
            "token_throughput": throughput_stats["avg"],
            "total_time": total_time_stats["avg"],
            # 详细统计数据
            "first_token_latency_stats": latency_stats,
            "token_throughput_stats": throughput_stats,
            "total_time_stats": total_time_stats,
        }

        print("\n===== 基准测试结果摘要 =====")
        print(f"模型: {self.model}")
        print(f"首字延迟: {latency_stats['avg']:.3f}秒 "
              f"(min={latency_stats['min']:.3f}, p90={latency_stats['p90']:.3f})")
        print(f"吞吐量: {throughput_stats['avg']:.2f} tokens/秒 "
              f"(min={throughput_stats['min']:.2f}, p90={throughput_stats['p90']:.2f})")
        print(f"总响应时间: {total_time_stats['avg']:.2f}秒 "
              f"(min={total_time_stats['min']:.2f}, max={total_time_stats['max']:.2f})")

        return results