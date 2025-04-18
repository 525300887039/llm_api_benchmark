"""LLM API基准测试的核心功能."""

import time
import json
import statistics
from datetime import datetime
import requests


class LLMAPIBenchmark:
    """大语言模型API性能测试类."""

    def __init__(self, api_url, api_key, model):
        """
        初始化API基准测试对象.

        Args:
            api_url: API端点URL
            api_key: API密钥
            model: 要测试的模型名称
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def measure_first_token_latency(self, prompt, num_runs=3):
        """
        测量首字延迟（从发送请求到收到第一个token的时间）.

        Args:
            prompt: 测试用的提示词
            num_runs: 测试运行次数

        Returns:
            float: 平均首字延迟（秒）
        """
        latencies = []

        for i in range(num_runs):
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True
            }

            start_time = time.time()
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                stream=True
            )

            # 读取第一个数据包
            for line in response.iter_lines():
                if line:
                    first_token_time = time.time()
                    latency = first_token_time - start_time
                    latencies.append(latency)
                    break

            response.close()
            print(f"运行 {i+1}/{num_runs}: 首字延迟 = {latency:.3f}秒")
            time.sleep(1)  # 避免请求过于频繁

        avg_latency = statistics.mean(latencies)
        print(f"\n平均首字延迟: {avg_latency:.3f}秒")
        return avg_latency

    def measure_token_throughput(self, prompt, runs=3):
        """
        测量模型的Token吞吐量（tokens/second）.

        Args:
            prompt: 测试用的提示词
            runs: 测试运行次数

        Returns:
            float: 平均token吞吐量（tokens/秒）
        """
        throughputs = []

        for i in range(runs):
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }

            start_time = time.time()
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )
            end_time = time.time()

            total_time = end_time - start_time
            response_json = response.json()

            # 获取输出的token数量，不同API可能需要调整
            try:
                # OpenAI API结构
                output_tokens = response_json.get("usage", {}).get("completion_tokens", 0)
                if output_tokens == 0:
                    # 尝试其他可能的格式
                    output_tokens = len(response_json.get("choices", [{}])[0].get("message", {}).get("content", "").split())
            except:
                # 如果无法获取确切token数，使用简单估计（按空格分割）
                output_tokens = len(response_json.get("choices", [{}])[0].get("message", {}).get("content", "").split())

            if total_time > 0 and output_tokens > 0:
                throughput = output_tokens / total_time
                throughputs.append(throughput)
                print(f"运行 {i+1}/{runs}: 吞吐量 = {throughput:.2f} tokens/秒 (生成了 {output_tokens} tokens，用时 {total_time:.2f}秒)")

            time.sleep(1)  # 避免请求过于频繁

        if throughputs:
            avg_throughput = statistics.mean(throughputs)
            print(f"\n平均吞吐量: {avg_throughput:.2f} tokens/秒")
            return avg_throughput
        return 0

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
        print(f"测试提示词: {prompt[:50]}... ({len(prompt)} 字符)")
        print(f"运行次数: {runs}")
        print("\n----- 测试首字延迟 -----")
        first_token_latency = self.measure_first_token_latency(prompt, runs)

        print("\n----- 测试吞吐量 -----")
        token_throughput = self.measure_token_throughput(prompt, runs)

        results = {
            "model": self.model,
            "api_url": self.api_url,
            "timestamp": datetime.now().isoformat(),
            "prompt_length": len(prompt),
            "runs": runs,
            "first_token_latency": first_token_latency,
            "token_throughput": token_throughput
        }

        print("\n===== 基准测试结果摘要 =====")
        print(f"模型: {self.model}")
        print(f"首字延迟: {first_token_latency:.3f}秒")
        print(f"吞吐量: {token_throughput:.2f} tokens/秒")

        return results