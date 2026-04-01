"""批量测试模块的单元测试."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_api_benchmark.batch import BatchBenchmark
from llm_api_benchmark.benchmark import BenchmarkRunError


class TestBatchBenchmark(unittest.TestCase):
    """BatchBenchmark 测试."""

    def _write_config(self, root: str, general_lines=None) -> str:
        general_lines = general_lines or []
        config_path = Path(root) / "config.toml"
        config_path.write_text(
            "\n".join(
                [
                    "[general]",
                    'prompt = "hello"',
                    "runs = 1",
                    'output_dir = "./results"',
                    *general_lines,
                    "",
                    "[[apis]]",
                    'name = "Test API"',
                    'url = "https://api.example.com/v1/chat/completions"',
                    'key = "test-key"',
                    'model = "test-model"',
                    'type = "openai"',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return str(config_path)

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
