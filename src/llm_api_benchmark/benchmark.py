"""LLM API基准测试的核心功能."""

import math
import time
import statistics
from datetime import datetime
from typing import Any
import requests

from .providers import create_provider


class BenchmarkRunError(RuntimeError):
    """基准测试在所有运行都失败时抛出的异常."""


class LLMAPIBenchmark:
    """大语言模型API性能测试类."""

    def __init__(
        self,
        api_url,
        api_key,
        model,
        api_type="openai",
        timeout=None,
        warmup_runs=0,
        max_retries=0,
        retry_delay=1.0,
    ):
        """
        初始化API基准测试对象.

        Args:
            api_url: API端点URL
            api_key: API密钥
            model: 要测试的模型名称
            api_type: API类型 ("openai", "claude", "azure", "gemini")
            timeout: requests timeout 配置，None 表示使用 requests 默认行为
            warmup_runs: 正式测量前的预热次数，必须 >= 0
            max_retries: 单轮请求失败后的最大重试次数，必须 >= 0
            retry_delay: 初始重试等待秒数，必须 > 0
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.api_type = api_type
        self.timeout = self._normalize_timeout(timeout)
        self.warmup_runs = self._normalize_warmup_runs(warmup_runs)
        self.max_retries = self._normalize_max_retries(max_retries)
        self.retry_delay = self._normalize_retry_delay(retry_delay)
        self.provider = create_provider(api_type, api_url, api_key, model)

    @staticmethod
    def _normalize_timeout(timeout: Any):
        """标准化 timeout 配置."""
        if timeout is None:
            return None

        if isinstance(timeout, (int, float)):
            if timeout <= 0:
                raise ValueError("timeout 必须大于 0")
            return float(timeout)

        if isinstance(timeout, (list, tuple)) and len(timeout) == 2:
            connect_timeout, read_timeout = timeout
            if connect_timeout <= 0 or read_timeout <= 0:
                raise ValueError("timeout 的两个值都必须大于 0")
            return float(connect_timeout), float(read_timeout)

        raise ValueError("timeout 必须为正数，或包含两个正数的列表/元组")

    @staticmethod
    def _normalize_warmup_runs(warmup_runs: Any) -> int:
        """标准化 warmup_runs 配置."""
        if not isinstance(warmup_runs, int) or warmup_runs < 0:
            raise ValueError("warmup_runs 必须为大于等于 0 的整数")
        return warmup_runs

    @staticmethod
    def _normalize_max_retries(max_retries: Any) -> int:
        """标准化 max_retries 配置."""
        if not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries 必须为大于等于 0 的整数")
        return max_retries

    @staticmethod
    def _normalize_retry_delay(retry_delay: Any) -> float:
        """标准化 retry_delay 配置."""
        if not isinstance(retry_delay, (int, float)) or retry_delay <= 0:
            raise ValueError("retry_delay 必须大于 0")
        return float(retry_delay)

    @staticmethod
    def _is_retryable(exc: requests.RequestException) -> bool:
        """判断异常是否可重试."""
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True

        if isinstance(exc, requests.HTTPError):
            response = exc.response
            return response is not None and response.status_code in {429, 500, 502, 503, 504}

        return False

    @staticmethod
    def _format_request_error(exc: requests.RequestException) -> str:
        """返回不包含敏感 URL 的错误摘要."""
        if isinstance(exc, requests.Timeout):
            return "请求超时"

        if isinstance(exc, requests.HTTPError):
            response = exc.response
            if response is not None:
                reason = f" {response.reason}" if response.reason else ""
                return f"HTTP {response.status_code}{reason}"
            return "HTTP 错误"

        if isinstance(exc, requests.ConnectionError):
            return "连接失败"

        return exc.__class__.__name__

    @staticmethod
    def _raise_if_no_success(metric_name, values, failures, total_runs):
        """当所有运行都失败时抛出明确异常."""
        if values:
            return

        unique_failures = list(dict.fromkeys(failures)) or ["未产生可用结果"]
        reasons = "；".join(unique_failures[:3])
        raise BenchmarkRunError(
            f"{metric_name}测试失败：{total_runs} 次运行均未成功。原因：{reasons}"
        )

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
            return {
                "avg": 0,
                "min": 0,
                "max": 0,
                "median": 0,
                "p90": 0,
                "p99": 0,
                "std_dev": 0,
                "raw": [],
            }

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

    def _run_warmup_requests(self, request_factory, response_consumer):
        """按配置执行预热请求，结果不纳入统计."""
        if self.warmup_runs <= 0:
            return

        for i in range(self.warmup_runs):
            print(f"预热 {i+1}/{self.warmup_runs}...")
            response = None
            try:
                response = request_factory()
                response.raise_for_status()
                response_consumer(response)
            except (requests.RequestException, ValueError):
                pass
            finally:
                if response is not None:
                    response.close()

        print("预热完成，开始正式测量\n")

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
        failures = []

        def send_request():
            payload = self.provider.build_chat_payload(prompt, stream=True)
            return requests.post(
                self.provider.get_request_url(stream=True),
                headers=self.provider.get_headers(),
                json=payload,
                stream=True,
                timeout=self.timeout,
            )

        def consume_stream_until_first_token(response):
            for line in response.iter_lines():
                if self.provider.is_first_content_event(line):
                    break

        self._run_warmup_requests(send_request, consume_stream_until_first_token)

        for i in range(num_runs):
            latency = None
            failed = False
            attempt = 0

            while True:
                response = None
                try:
                    start_time = time.time()
                    response = send_request()
                    response.raise_for_status()

                    # 读取第一个内容数据包
                    for line in response.iter_lines():
                        if self.provider.is_first_content_event(line):
                            first_token_time = time.time()
                            latency = first_token_time - start_time
                            latencies.append(latency)
                            break
                    break
                except requests.RequestException as exc:
                    error_message = self._format_request_error(exc)
                    if self._is_retryable(exc) and attempt < self.max_retries:
                        attempt += 1
                        print(
                            f"运行 {i+1}/{num_runs}: 请求失败 ({error_message})，"
                            f"第 {attempt}/{self.max_retries} 次重试..."
                        )
                        time.sleep(self.retry_delay * (2 ** (attempt - 1)))
                        continue

                    failures.append(error_message)
                    print(f"运行 {i+1}/{num_runs}: 请求失败 ({error_message})，已跳过")
                    failed = True
                    break
                finally:
                    if response is not None:
                        response.close()

            if not failed and latency is None:
                failures.append("未检测到首个 token")
                print(f"运行 {i+1}/{num_runs}: 未检测到首个 token，跳过本轮")
            elif latency is not None:
                print(f"运行 {i+1}/{num_runs}: 首字延迟 = {latency:.3f}秒")

            if i < num_runs - 1:
                time.sleep(1)  # 避免请求过于频繁

        self._raise_if_no_success("首字延迟", latencies, failures, num_runs)
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
        failures = []

        def send_request():
            payload = self.provider.build_chat_payload(prompt, stream=False)
            return requests.post(
                self.provider.get_request_url(stream=False),
                headers=self.provider.get_headers(),
                json=payload,
                timeout=self.timeout,
            )

        self._run_warmup_requests(send_request, lambda response: response.json())

        for i in range(runs):
            attempt = 0

            while True:
                response = None
                try:
                    start_time = time.time()
                    response = send_request()
                    response.raise_for_status()
                    end_time = time.time()

                    elapsed = end_time - start_time
                    response_json = response.json()

                    output_tokens = self.provider.parse_token_count(response_json)

                    if elapsed > 0 and output_tokens > 0:
                        throughput = output_tokens / elapsed
                        throughputs.append(throughput)
                        total_times.append(elapsed)
                        print(
                            f"运行 {i+1}/{runs}: 吞吐量 = {throughput:.2f} tokens/秒 "
                            f"(生成了 {output_tokens} tokens，用时 {elapsed:.2f}秒)"
                        )
                    else:
                        failures.append("未能解析输出 token 数")
                        print(f"运行 {i+1}/{runs}: 未能解析输出 token 数，跳过本轮")
                    break
                except requests.RequestException as exc:
                    error_message = self._format_request_error(exc)
                    if self._is_retryable(exc) and attempt < self.max_retries:
                        attempt += 1
                        print(
                            f"运行 {i+1}/{runs}: 请求失败 ({error_message})，"
                            f"第 {attempt}/{self.max_retries} 次重试..."
                        )
                        time.sleep(self.retry_delay * (2 ** (attempt - 1)))
                        continue

                    failures.append(error_message)
                    print(f"运行 {i+1}/{runs}: 请求失败 ({error_message})，已跳过")
                    break
                finally:
                    if response is not None:
                        response.close()

            if i < runs - 1:
                time.sleep(1)  # 避免请求过于频繁

        self._raise_if_no_success("吞吐量", throughputs, failures, runs)
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
        print(f"预热轮次: {self.warmup_runs}")

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
            "warmup_runs": self.warmup_runs,
            "max_retries": self.max_retries,
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
        print(f"预热轮次: {self.warmup_runs}")
        print(
            f"首字延迟: {latency_stats['avg']:.3f}秒 "
            f"(min={latency_stats['min']:.3f}, p90={latency_stats['p90']:.3f})"
        )
        print(
            f"吞吐量: {throughput_stats['avg']:.2f} tokens/秒 "
            f"(min={throughput_stats['min']:.2f}, p90={throughput_stats['p90']:.2f})"
        )
        print(
            f"总响应时间: {total_time_stats['avg']:.2f}秒 "
            f"(min={total_time_stats['min']:.2f}, max={total_time_stats['max']:.2f})"
        )

        return results
