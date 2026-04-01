"""批量测试模块的单元测试."""

import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from llm_api_benchmark.batch import BatchBenchmark
from llm_api_benchmark.benchmark import BenchmarkRunError


class TestBatchBenchmark(unittest.TestCase):
    """BatchBenchmark 测试."""

    def _make_api(self, name: str) -> dict:
        model = "test-model" if name == "Test API" else f"{name.lower().replace(' ', '-')}-model"
        return {
            "name": name,
            "url": "https://api.example.com/v1/chat/completions",
            "key": "test-key",
            "model": model,
            "type": "openai",
        }

    def _write_config(self, root: str, general_lines=None, apis=None) -> str:
        general_lines = general_lines or []
        apis = apis or [self._make_api("Test API")]
        config_path = Path(root) / "config.toml"
        output_dir = (Path(root) / "results").as_posix()
        api_lines = []
        for api in apis:
            api_lines.extend(
                [
                    "[[apis]]",
                    f'name = "{api["name"]}"',
                    f'url = "{api["url"]}"',
                    f'key = "{api["key"]}"',
                    f'model = "{api["model"]}"',
                    f'type = "{api["type"]}"',
                    "",
                ]
            )

        config_path.write_text(
            "\n".join(
                [
                    "[general]",
                    'prompt = "hello"',
                    "runs = 1",
                    f'output_dir = "{output_dir}"',
                    *general_lines,
                    "",
                    *api_lines,
                ]
            ),
            encoding="utf-8",
        )
        return str(config_path)

    def test_batch_serial_by_default(self):
        """parallel 默认值应为 1，并保持串行执行."""
        apis = [self._make_api("API 1"), self._make_api("API 2")]

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir, apis=apis)
            batch = BatchBenchmark(config_path)
            call_order = []

            def fake_run(api_config, index):
                call_order.append(api_config["name"])
                return {"name": api_config["name"]}

            with patch.object(batch, "_run_single_api_test", side_effect=fake_run) as mock_run:
                with patch("llm_api_benchmark.batch.ThreadPoolExecutor") as mock_executor:
                    results = batch.run_batch_tests()

        self.assertEqual(call_order, ["API 1", "API 2"])
        self.assertEqual([result["name"] for result in results], ["API 1", "API 2"])
        self.assertEqual(mock_run.call_count, 2)
        mock_executor.assert_not_called()

    def test_batch_parallel_execution(self):
        """parallel > 1 时应通过线程池并行执行多个 API."""
        apis = [self._make_api("API 1"), self._make_api("API 2"), self._make_api("API 3")]

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir, general_lines=["parallel = 3"], apis=apis)
            batch = BatchBenchmark(config_path)
            active_calls = 0
            max_active_calls = 0
            lock = threading.Lock()

            def fake_run(api_config, index):
                nonlocal active_calls, max_active_calls
                with lock:
                    active_calls += 1
                    max_active_calls = max(max_active_calls, active_calls)
                time.sleep(0.1)
                with lock:
                    active_calls -= 1
                return {"name": api_config["name"]}

            with patch.object(batch, "_run_single_api_test", side_effect=fake_run):
                results = batch.run_batch_tests()

        self.assertGreater(max_active_calls, 1)
        self.assertEqual(sorted(result["name"] for result in results), ["API 1", "API 2", "API 3"])

    def test_batch_parallel_handles_failure(self):
        """并行模式下单个任务失败不应影响其他 API."""
        apis = [self._make_api("API 1"), self._make_api("API 2"), self._make_api("API 3")]

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir, general_lines=["parallel = 3"], apis=apis)
            batch = BatchBenchmark(config_path)

            def fake_run(api_config, index):
                if api_config["name"] == "API 2":
                    raise RuntimeError("boom")
                return {"name": api_config["name"]}

            with patch.object(batch, "_run_single_api_test", side_effect=fake_run):
                with patch("builtins.print") as mock_print:
                    results = batch.run_batch_tests()

        self.assertEqual(sorted(result["name"] for result in results), ["API 1", "API 3"])
        mock_print.assert_any_call("测试 API 'API 2' 时发生未预期的错误: boom")

    def test_batch_parallel_zero_means_auto(self):
        """parallel = 0 时线程数应自动等于 API 数量."""
        apis = [self._make_api("API 1"), self._make_api("API 2"), self._make_api("API 3")]

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir, general_lines=["parallel = 0"], apis=apis)
            batch = BatchBenchmark(config_path)
            executor_args = {}

            class RecordingExecutor(RealThreadPoolExecutor):
                def __init__(self, max_workers=None, *args, **kwargs):
                    executor_args["max_workers"] = max_workers
                    super().__init__(max_workers=max_workers, *args, **kwargs)

            def fake_run(api_config, index):
                return {"name": api_config["name"]}

            with patch.object(batch, "_run_single_api_test", side_effect=fake_run):
                with patch("llm_api_benchmark.batch.ThreadPoolExecutor", RecordingExecutor):
                    results = batch.run_batch_tests()

        self.assertEqual(executor_args["max_workers"], len(apis))
        self.assertEqual(sorted(result["name"] for result in results), ["API 1", "API 2", "API 3"])

    @patch("llm_api_benchmark.batch.LLMAPIBenchmark")
    def test_run_batch_tests_passes_timeout_from_config(self, mock_benchmark_cls):
        """general.timeout 应透传到每个 benchmark 实例."""
        mock_benchmark = mock_benchmark_cls.return_value
        mock_benchmark.run_comprehensive_benchmark.return_value = {
            "model": "test-model",
            "api_url": "https://api.example.com/v1/chat/completions",
            "api_type": "openai",
            "timestamp": "2026-04-01T00:00:00",
            "prompt_length": 5,
            "runs": 1,
            "first_token_latency": 1.0,
            "token_throughput": 2.0,
            "total_time": 3.0,
            "first_token_latency_stats": {"avg": 1.0},
            "token_throughput_stats": {"avg": 2.0},
            "total_time_stats": {"avg": 3.0},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir, ["timeout = [10, 120]"])
            batch = BatchBenchmark(config_path)
            results = batch.run_batch_tests()

        self.assertEqual(len(results), 1)
        mock_benchmark_cls.assert_called_once_with(
            "https://api.example.com/v1/chat/completions",
            "test-key",
            "test-model",
            "openai",
            timeout=[10, 120],
            warmup_runs=0,
            max_retries=0,
            retry_delay=1.0,
        )

    @patch("llm_api_benchmark.batch.LLMAPIBenchmark")
    def test_run_batch_tests_passes_retry_config(self, mock_benchmark_cls):
        """general.max_retries 和 general.retry_delay 应透传到每个 benchmark 实例."""
        mock_benchmark = mock_benchmark_cls.return_value
        mock_benchmark.run_comprehensive_benchmark.return_value = {
            "model": "test-model",
            "api_url": "https://api.example.com/v1/chat/completions",
            "api_type": "openai",
            "timestamp": "2026-04-01T00:00:00",
            "prompt_length": 5,
            "runs": 1,
            "first_token_latency": 1.0,
            "token_throughput": 2.0,
            "total_time": 3.0,
            "first_token_latency_stats": {"avg": 1.0},
            "token_throughput_stats": {"avg": 2.0},
            "total_time_stats": {"avg": 3.0},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(
                temp_dir,
                [
                    "max_retries = 2",
                    "retry_delay = 0.5",
                ],
            )
            batch = BatchBenchmark(config_path)
            results = batch.run_batch_tests()

        self.assertEqual(len(results), 1)
        mock_benchmark_cls.assert_called_once_with(
            "https://api.example.com/v1/chat/completions",
            "test-key",
            "test-model",
            "openai",
            timeout=None,
            warmup_runs=0,
            max_retries=2,
            retry_delay=0.5,
        )

    @patch("llm_api_benchmark.batch.LLMAPIBenchmark")
    def test_run_batch_tests_skips_failed_benchmarks(self, mock_benchmark_cls):
        """当 benchmark 抛错时不应生成伪成功结果."""
        mock_benchmark = mock_benchmark_cls.return_value
        mock_benchmark.run_comprehensive_benchmark.side_effect = BenchmarkRunError("failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir)
            batch = BatchBenchmark(config_path)
            results = batch.run_batch_tests()

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
