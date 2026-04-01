"""LLM API基准测试工具的单元测试."""

import unittest
from unittest.mock import patch, MagicMock, call
from datetime import datetime
import requests

from llm_api_benchmark.benchmark import BenchmarkRunError, LLMAPIBenchmark


class TestLLMAPIBenchmark(unittest.TestCase):
    """LLMAPIBenchmark类的测试."""

    def setUp(self):
        """设置测试环境."""
        self.api_url = "https://api.test.com/v1/chat/completions"
        self.api_key = "test_key"
        self.model = "test-model"
        self.benchmark = LLMAPIBenchmark(self.api_url, self.api_key, self.model)
        self.test_prompt = "这是一个测试提示词"

    @staticmethod
    def _build_stream_response(event_log=None, label=None):
        """构造带可观测消费顺序的流式响应."""
        mock_response = MagicMock()

        def iter_lines():
            if event_log is not None and label is not None:
                event_log.append(label)
            return iter([b'data: {"choices":[{"delta":{"content":"Hello"}}]}'])

        mock_response.iter_lines.side_effect = iter_lines
        return mock_response

    @staticmethod
    def _build_json_response(output_tokens):
        """构造非流式响应."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "usage": {"completion_tokens": output_tokens},
            "choices": [{"message": {"content": "测试回复" * 20}}],
        }
        return mock_response

    @staticmethod
    def _build_http_error_response(status_code, reason):
        """构造 raise_for_status 会抛出 HTTPError 的响应."""
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.reason = reason
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            f"{status_code} {reason}",
            response=mock_response,
        )
        return mock_response

    @patch("llm_api_benchmark.benchmark.requests.post")
    def test_measure_first_token_latency(self, mock_post):
        """测试首字延迟测量功能."""
        # 模拟响应
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}'
        ]
        mock_post.return_value = mock_response

        # 运行测试
        with (
            patch("llm_api_benchmark.benchmark.time.time") as mock_time,
            patch("llm_api_benchmark.benchmark.time.sleep"),
        ):
            mock_time.side_effect = [0, 0.5]  # 模拟起始时间和第一个token时间
            stats = self.benchmark.measure_first_token_latency(self.test_prompt, num_runs=1)

        # 验证结果（现在返回 stats dict）
        self.assertEqual(stats["avg"], 0.5)
        self.assertEqual(stats["raw"], [0.5])
        mock_post.assert_called_once()
        mock_response.close.assert_called_once()
        mock_response.raise_for_status.assert_called_once()
        self.assertEqual(mock_post.call_args[1]["timeout"], self.benchmark.timeout)
        self.assertEqual(mock_post.call_args[1]["json"]["model"], self.model)
        self.assertEqual(mock_post.call_args[1]["json"]["messages"][0]["content"], self.test_prompt)

    @patch("llm_api_benchmark.benchmark.requests.post")
    def test_measure_first_token_latency_without_content_event(self, mock_post):
        """测试流式响应没有内容token时会显式失败."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            b": keep-alive",
            b'data: {"choices":[{"delta":{}}]}',
            b"data: [DONE]",
        ]
        mock_post.return_value = mock_response

        with (
            patch("llm_api_benchmark.benchmark.time.time", return_value=0),
            patch("llm_api_benchmark.benchmark.time.sleep"),
        ):
            with self.assertRaises(BenchmarkRunError) as ctx:
                self.benchmark.measure_first_token_latency(self.test_prompt, num_runs=1)

        self.assertIn("未检测到首个 token", str(ctx.exception))
        mock_post.assert_called_once()
        mock_response.close.assert_called_once()
        mock_response.raise_for_status.assert_called_once()
        self.assertEqual(mock_post.call_args[1]["timeout"], self.benchmark.timeout)

    @patch("llm_api_benchmark.benchmark.requests.post")
    def test_measure_first_token_latency_with_empty_stream(self, mock_post):
        """测试空流式响应时会显式失败."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = []
        mock_post.return_value = mock_response

        with (
            patch("llm_api_benchmark.benchmark.time.time", return_value=0),
            patch("llm_api_benchmark.benchmark.time.sleep"),
            patch("builtins.print") as mock_print,
        ):
            with self.assertRaises(BenchmarkRunError):
                self.benchmark.measure_first_token_latency(self.test_prompt, num_runs=1)

        mock_post.assert_called_once()
        mock_response.close.assert_called_once()
        mock_response.raise_for_status.assert_called_once()
        self.assertEqual(mock_post.call_args[1]["timeout"], self.benchmark.timeout)
        mock_print.assert_any_call("运行 1/1: 未检测到首个 token，跳过本轮")

    @patch("llm_api_benchmark.benchmark.requests.post")
    def test_measure_first_token_latency_timeout(self, mock_post):
        """测试请求超时时若全部失败会抛出异常."""
        mock_post.side_effect = requests.Timeout("timed out")

        with patch("llm_api_benchmark.benchmark.time.sleep"):
            with self.assertRaises(BenchmarkRunError) as ctx:
                self.benchmark.measure_first_token_latency(self.test_prompt, num_runs=1)

        self.assertIn("请求超时", str(ctx.exception))
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args[1]["timeout"], self.benchmark.timeout)

    @patch("llm_api_benchmark.benchmark.requests.post")
    def test_measure_first_token_latency_http_error(self, mock_post):
        """测试HTTP错误信息会脱敏且最终抛出基准测试异常."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.reason = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "500 Server Error: https://example.com?key=secret",
            response=mock_response,
        )
        mock_post.return_value = mock_response

        with patch("llm_api_benchmark.benchmark.time.sleep"):
            with self.assertRaises(BenchmarkRunError) as ctx:
                self.benchmark.measure_first_token_latency(self.test_prompt, num_runs=1)

        self.assertIn("HTTP 500 Internal Server Error", str(ctx.exception))
        self.assertNotIn("secret", str(ctx.exception))
        self.assertNotIn("https://example.com", str(ctx.exception))
        mock_post.assert_called_once()
        mock_response.raise_for_status.assert_called_once()
        mock_response.close.assert_called_once()

    @patch("llm_api_benchmark.benchmark.requests.post")
    def test_no_retry_by_default(self, mock_post):
        """测试默认不启用重试，行为与现有逻辑一致."""
        mock_post.side_effect = requests.Timeout("timed out")

        with patch("llm_api_benchmark.benchmark.time.sleep") as mock_sleep:
            with self.assertRaises(BenchmarkRunError):
                self.benchmark.measure_first_token_latency(self.test_prompt, num_runs=1)

        self.assertEqual(self.benchmark.max_retries, 0)
        self.assertEqual(mock_post.call_count, 1)
        mock_sleep.assert_not_called()

    @patch("llm_api_benchmark.benchmark.requests.post")
    def test_retry_on_timeout(self, mock_post):
        """测试 Timeout 会触发重试并最终成功."""
        benchmark = LLMAPIBenchmark(
            self.api_url,
            self.api_key,
            self.model,
            max_retries=2,
            retry_delay=0.25,
        )
        success_response = self._build_stream_response()
        mock_post.side_effect = [requests.Timeout("timed out"), success_response]

        with (
            patch("llm_api_benchmark.benchmark.time.time") as mock_time,
            patch("llm_api_benchmark.benchmark.time.sleep") as mock_sleep,
        ):
            mock_time.side_effect = [0, 1.0, 2.0]
            stats = benchmark.measure_first_token_latency(self.test_prompt, num_runs=1)

        self.assertEqual(stats["raw"], [1.0])
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once_with(0.25)
        success_response.close.assert_called_once()

    @patch("llm_api_benchmark.benchmark.requests.post")
    def test_retry_on_429(self, mock_post):
        """测试 HTTP 429 会触发重试."""
        benchmark = LLMAPIBenchmark(
            self.api_url,
            self.api_key,
            self.model,
            max_retries=2,
            retry_delay=0.25,
        )
        rate_limit_response = self._build_http_error_response(429, "Too Many Requests")
        success_response = self._build_json_response(100)
        mock_post.side_effect = [rate_limit_response, success_response]

        with (
            patch("llm_api_benchmark.benchmark.time.time") as mock_time,
            patch("llm_api_benchmark.benchmark.time.sleep") as mock_sleep,
        ):
            mock_time.side_effect = [0, 1.0, 3.0]
            throughput_stats, total_time_stats = benchmark.measure_token_throughput(
                self.test_prompt, runs=1
            )

        self.assertEqual(throughput_stats["raw"], [50.0])
        self.assertEqual(total_time_stats["raw"], [2.0])
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once_with(0.25)
        rate_limit_response.close.assert_called_once()
        success_response.close.assert_called_once()

    @patch("llm_api_benchmark.benchmark.requests.post")
    def test_no_retry_on_400(self, mock_post):
        """测试 HTTP 400 不会触发重试."""
        benchmark = LLMAPIBenchmark(
            self.api_url,
            self.api_key,
            self.model,
            max_retries=3,
            retry_delay=0.25,
        )
        bad_request_response = self._build_http_error_response(400, "Bad Request")
        mock_post.return_value = bad_request_response

        with patch("llm_api_benchmark.benchmark.time.sleep") as mock_sleep:
            with self.assertRaises(BenchmarkRunError) as ctx:
                benchmark.measure_token_throughput(self.test_prompt, runs=1)

        self.assertIn("HTTP 400 Bad Request", str(ctx.exception))
        self.assertEqual(mock_post.call_count, 1)
        mock_sleep.assert_not_called()
        bad_request_response.close.assert_called_once()

    @patch("llm_api_benchmark.benchmark.requests.post")
    def test_retry_exponential_backoff(self, mock_post):
        """测试重试等待时间按指数退避递增."""
        benchmark = LLMAPIBenchmark(
            self.api_url,
            self.api_key,
            self.model,
            max_retries=3,
            retry_delay=0.5,
        )
        success_response = self._build_stream_response()
        mock_post.side_effect = [
            requests.Timeout("timeout-1"),
            requests.Timeout("timeout-2"),
            requests.Timeout("timeout-3"),
            success_response,
        ]

        with (
            patch("llm_api_benchmark.benchmark.time.time") as mock_time,
            patch("llm_api_benchmark.benchmark.time.sleep") as mock_sleep,
        ):
            mock_time.side_effect = [0, 1.0, 2.0, 3.0, 4.0]
            stats = benchmark.measure_first_token_latency(self.test_prompt, num_runs=1)

        self.assertEqual(stats["raw"], [1.0])
        self.assertEqual(mock_post.call_count, 4)
        self.assertEqual(mock_sleep.call_args_list, [call(0.5), call(1.0), call(2.0)])

    @patch("llm_api_benchmark.benchmark.requests.post")
    def test_retry_exhausted_raises(self, mock_post):
        """测试重试耗尽后最终抛出 BenchmarkRunError."""
        benchmark = LLMAPIBenchmark(
            self.api_url,
            self.api_key,
            self.model,
            max_retries=2,
            retry_delay=0.5,
        )
        mock_post.side_effect = requests.Timeout("timed out")

        with (
            patch("llm_api_benchmark.benchmark.time.time") as mock_time,
            patch("llm_api_benchmark.benchmark.time.sleep") as mock_sleep,
        ):
            mock_time.side_effect = [0, 1.0, 2.0]
            with self.assertRaises(BenchmarkRunError) as ctx:
                benchmark.measure_first_token_latency(self.test_prompt, num_runs=1)

        self.assertIn("请求超时", str(ctx.exception))
        self.assertEqual(mock_post.call_count, 3)
        self.assertEqual(mock_sleep.call_args_list, [call(0.5), call(1.0)])

    @patch("llm_api_benchmark.benchmark.requests.post")
    def test_measure_token_throughput(self, mock_post):
        """测试token吞吐量测量功能."""
        # 模拟响应
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "usage": {"completion_tokens": 100},
            "choices": [{"message": {"content": "测试回复" * 20}}],
        }
        mock_post.return_value = mock_response

        # 运行测试
        with (
            patch("llm_api_benchmark.benchmark.time.time") as mock_time,
            patch("llm_api_benchmark.benchmark.time.sleep"),
        ):
            mock_time.side_effect = [0, 2.0]  # 模拟起始时间和结束时间（2秒）
            throughput_stats, total_time_stats = self.benchmark.measure_token_throughput(
                self.test_prompt, runs=1
            )

        # 验证结果 (100 tokens / 2秒 = 50 tokens/秒)
        self.assertEqual(throughput_stats["avg"], 50.0)
        self.assertEqual(throughput_stats["raw"], [50.0])
        self.assertEqual(total_time_stats["avg"], 2.0)
        mock_post.assert_called_once()
        mock_response.raise_for_status.assert_called_once()
        mock_response.close.assert_called_once()
        self.assertEqual(mock_post.call_args[1]["timeout"], self.benchmark.timeout)
        self.assertEqual(mock_post.call_args[1]["json"]["messages"][0]["content"], self.test_prompt)

    def test_warmup_runs_default_zero(self):
        """测试 warmup_runs 默认值为 0."""
        self.assertEqual(self.benchmark.warmup_runs, 0)

    @patch("llm_api_benchmark.benchmark.requests.post")
    def test_warmup_runs_executes_before_measurement(self, mock_post):
        """测试预热请求会先于正式测量执行."""
        benchmark = LLMAPIBenchmark(
            self.api_url,
            self.api_key,
            self.model,
            warmup_runs=2,
        )
        event_log = []
        mock_post.side_effect = [
            self._build_stream_response(event_log, "warmup-1"),
            self._build_stream_response(event_log, "warmup-2"),
            self._build_stream_response(event_log, "measure-1"),
        ]

        with (
            patch("llm_api_benchmark.benchmark.time.time") as mock_time,
            patch("llm_api_benchmark.benchmark.time.sleep"),
        ):
            mock_time.side_effect = iter(range(20))
            stats = benchmark.measure_first_token_latency(self.test_prompt, num_runs=1)

        self.assertEqual(event_log, ["warmup-1", "warmup-2", "measure-1"])
        self.assertEqual(stats["raw"], [1])
        self.assertEqual(mock_post.call_count, 3)
        for call_args in mock_post.call_args_list:
            self.assertTrue(call_args[1]["stream"])
            self.assertEqual(call_args[1]["timeout"], benchmark.timeout)
            self.assertEqual(call_args[1]["json"]["model"], self.model)
            self.assertEqual(call_args[1]["json"]["messages"][0]["content"], self.test_prompt)

    @patch("llm_api_benchmark.benchmark.requests.post")
    def test_warmup_runs_not_included_in_stats(self, mock_post):
        """测试预热期间的数据不会进入统计结果."""
        benchmark = LLMAPIBenchmark(
            self.api_url,
            self.api_key,
            self.model,
            warmup_runs=1,
        )
        mock_post.side_effect = [
            self._build_json_response(999),
            self._build_json_response(100),
            self._build_json_response(200),
        ]

        with (
            patch("llm_api_benchmark.benchmark.time.time") as mock_time,
            patch("llm_api_benchmark.benchmark.time.sleep"),
        ):
            mock_time.side_effect = iter(range(20))
            throughput_stats, total_time_stats = benchmark.measure_token_throughput(
                self.test_prompt, runs=2
            )

        self.assertEqual(throughput_stats["raw"], [100.0, 200.0])
        self.assertEqual(total_time_stats["raw"], [1, 1])
        self.assertEqual(mock_post.call_count, 3)
        for call_args in mock_post.call_args_list:
            self.assertNotIn("stream", call_args[1])
            self.assertEqual(call_args[1]["timeout"], benchmark.timeout)
            self.assertEqual(call_args[1]["json"]["model"], self.model)
            self.assertEqual(call_args[1]["json"]["messages"][0]["content"], self.test_prompt)

    @patch("llm_api_benchmark.benchmark.requests.post")
    def test_warmup_runs_errors_ignored(self, mock_post):
        """测试预热期间的异常不会影响正式测量."""
        benchmark = LLMAPIBenchmark(
            self.api_url,
            self.api_key,
            self.model,
            warmup_runs=1,
        )
        mock_post.side_effect = [requests.Timeout("warmup timed out"), self._build_json_response(100)]

        with (
            patch("builtins.print") as mock_print,
            patch("llm_api_benchmark.benchmark.time.time") as mock_time,
            patch("llm_api_benchmark.benchmark.time.sleep"),
        ):
            mock_time.side_effect = [0, 2.0]
            throughput_stats, total_time_stats = benchmark.measure_token_throughput(
                self.test_prompt, runs=1
            )

        printed_lines = [" ".join(str(arg) for arg in call.args) for call in mock_print.call_args_list]
        self.assertEqual(throughput_stats["raw"], [50.0])
        self.assertEqual(total_time_stats["raw"], [2.0])
        self.assertEqual(mock_post.call_count, 2)
        self.assertFalse(any("请求失败" in line for line in printed_lines))

    @patch("llm_api_benchmark.benchmark.requests.post")
    def test_measure_token_throughput_timeout(self, mock_post):
        """测试吞吐量请求超时时若全部失败会抛出异常."""
        mock_post.side_effect = requests.Timeout("timed out")

        with patch("llm_api_benchmark.benchmark.time.sleep"):
            with self.assertRaises(BenchmarkRunError) as ctx:
                self.benchmark.measure_token_throughput(self.test_prompt, runs=1)

        self.assertIn("请求超时", str(ctx.exception))
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args[1]["timeout"], self.benchmark.timeout)

    @patch("llm_api_benchmark.benchmark.requests.post")
    def test_measure_token_throughput_http_error(self, mock_post):
        """测试HTTP 500时吞吐量测量会显式失败."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.reason = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "500 Server Error: https://example.com?key=secret",
            response=mock_response,
        )
        mock_post.return_value = mock_response

        with patch("llm_api_benchmark.benchmark.time.sleep"):
            with self.assertRaises(BenchmarkRunError) as ctx:
                self.benchmark.measure_token_throughput(self.test_prompt, runs=1)

        self.assertIn("HTTP 500 Internal Server Error", str(ctx.exception))
        self.assertNotIn("secret", str(ctx.exception))
        mock_post.assert_called_once()
        mock_response.raise_for_status.assert_called_once()
        mock_response.close.assert_called_once()

    @patch("llm_api_benchmark.benchmark.requests.post")
    def test_measure_token_throughput_without_output_tokens_raises(self, mock_post):
        """测试无法解析 token 数时不会伪造 0 吞吐量成功."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"usage": {}, "choices": [{"message": {"content": ""}}]}
        mock_post.return_value = mock_response

        with (
            patch("llm_api_benchmark.benchmark.time.time") as mock_time,
            patch("llm_api_benchmark.benchmark.time.sleep"),
        ):
            mock_time.side_effect = [0, 2.0]
            with self.assertRaises(BenchmarkRunError) as ctx:
                self.benchmark.measure_token_throughput(self.test_prompt, runs=1)

        self.assertIn("未能解析输出 token 数", str(ctx.exception))
        mock_response.close.assert_called_once()

    @patch("llm_api_benchmark.benchmark.requests.post")
    def test_measure_first_token_latency_allows_partial_success(self, mock_post):
        """测试存在成功运行时不会因单次失败而整体失败."""
        success_response = MagicMock()
        success_response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}'
        ]
        failure = requests.Timeout("timed out")
        mock_post.side_effect = [success_response, failure]

        with (
            patch("llm_api_benchmark.benchmark.time.time") as mock_time,
            patch("llm_api_benchmark.benchmark.time.sleep"),
        ):
            mock_time.side_effect = [0, 0.5, 1.0]
            stats = self.benchmark.measure_first_token_latency(self.test_prompt, num_runs=2)

        self.assertEqual(stats["avg"], 0.5)
        self.assertEqual(stats["raw"], [0.5])
        success_response.close.assert_called_once()

    @patch("llm_api_benchmark.benchmark.LLMAPIBenchmark.measure_first_token_latency")
    @patch("llm_api_benchmark.benchmark.LLMAPIBenchmark.measure_token_throughput")
    def test_run_comprehensive_benchmark(self, mock_throughput, mock_latency):
        """测试综合基准测试功能."""
        # 模拟方法返回值（新格式：stats dict 和 tuple）
        mock_latency.return_value = {
            "avg": 0.5,
            "min": 0.5,
            "max": 0.5,
            "median": 0.5,
            "p90": 0.5,
            "p99": 0.5,
            "std_dev": 0,
            "raw": [0.5],
        }
        throughput_stats = {
            "avg": 50.0,
            "min": 50.0,
            "max": 50.0,
            "median": 50.0,
            "p90": 50.0,
            "p99": 50.0,
            "std_dev": 0,
            "raw": [50.0],
        }
        total_time_stats = {
            "avg": 2.0,
            "min": 2.0,
            "max": 2.0,
            "median": 2.0,
            "p90": 2.0,
            "p99": 2.0,
            "std_dev": 0,
            "raw": [2.0],
        }
        mock_throughput.return_value = (throughput_stats, total_time_stats)

        # 运行测试
        with patch("llm_api_benchmark.benchmark.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 0, 0)
            results = self.benchmark.run_comprehensive_benchmark(self.test_prompt, runs=2)

        # 验证结果（向后兼容的顶层字段）
        self.assertEqual(results["model"], self.model)
        self.assertEqual(results["api_url"], self.api_url)
        self.assertEqual(results["first_token_latency"], 0.5)
        self.assertEqual(results["token_throughput"], 50.0)
        self.assertEqual(results["total_time"], 2.0)
        self.assertEqual(results["prompt_length"], len(self.test_prompt))
        self.assertEqual(results["runs"], 2)
        self.assertEqual(results["warmup_runs"], 0)
        self.assertEqual(results["max_retries"], 0)
        # 验证详细统计字段存在
        self.assertIn("first_token_latency_stats", results)
        self.assertIn("token_throughput_stats", results)
        self.assertIn("total_time_stats", results)

    def test_compute_stats(self):
        """测试统计计算功能."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = LLMAPIBenchmark._compute_stats(data)

        self.assertAlmostEqual(stats["avg"], 3.0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 5.0)
        self.assertAlmostEqual(stats["median"], 3.0)
        self.assertEqual(stats["raw"], [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertGreater(stats["p90"], stats["median"])
        self.assertGreater(stats["std_dev"], 0)

    def test_compute_stats_empty(self):
        """测试空数据的统计计算."""
        stats = LLMAPIBenchmark._compute_stats([])
        self.assertEqual(stats["avg"], 0)
        self.assertEqual(stats["raw"], [])

    def test_compute_stats_single(self):
        """测试单个数据点的统计计算."""
        stats = LLMAPIBenchmark._compute_stats([42.0])
        self.assertEqual(stats["avg"], 42.0)
        self.assertEqual(stats["min"], 42.0)
        self.assertEqual(stats["max"], 42.0)
        self.assertEqual(stats["std_dev"], 0)

    def test_init_uses_no_timeout_by_default(self):
        """测试默认不强制设置请求超时."""
        self.assertIsNone(self.benchmark.timeout)

    def test_init_accepts_scalar_timeout(self):
        """测试可以配置单值 timeout."""
        benchmark = LLMAPIBenchmark(self.api_url, self.api_key, self.model, timeout=30)
        self.assertEqual(benchmark.timeout, 30.0)

    def test_init_accepts_timeout_pair(self):
        """测试可以配置 connect/read timeout 对."""
        benchmark = LLMAPIBenchmark(self.api_url, self.api_key, self.model, timeout=[10, 120])
        self.assertEqual(benchmark.timeout, (10.0, 120.0))


if __name__ == "__main__":
    unittest.main()
