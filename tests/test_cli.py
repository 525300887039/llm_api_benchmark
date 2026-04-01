"""命令行模块的单元测试."""

import unittest
from unittest.mock import patch

from llm_api_benchmark import cli


class TestLegacyCli(unittest.TestCase):
    """测试兼容旧版命令行的入口逻辑."""

    @patch("llm_api_benchmark.cli.main")
    def test_help_flag_keeps_top_level_args(self, mock_main):
        """`--help` 不应被当成旧版 single 命令."""
        argv = ["prog", "--help"]

        with patch("sys.argv", argv):
            cli.legacy_cli()

        self.assertEqual(argv, ["prog", "--help"])
        mock_main.assert_called_once_with()

    @patch("llm_api_benchmark.cli.main")
    def test_legacy_args_insert_single_command(self, mock_main):
        """旧版参数应自动补上 single 子命令."""
        argv = ["prog", "--api_key", "test-key"]

        with patch("sys.argv", argv):
            cli.legacy_cli()

        self.assertEqual(argv, ["prog", "single", "--api_key", "test-key"])
        mock_main.assert_called_once_with()


class TestRunBenchmarkCli(unittest.TestCase):
    """测试新版 CLI 参数传递."""

    @patch("llm_api_benchmark.cli.LLMAPIBenchmark")
    def test_single_command_passes_timeout(self, mock_benchmark_cls):
        """`--timeout` 应传递给基准测试对象."""
        mock_benchmark = mock_benchmark_cls.return_value
        mock_benchmark.run_comprehensive_benchmark.return_value = {"ok": True}
        argv = [
            "prog",
            "single",
            "--api_key",
            "test-key",
            "--timeout",
            "45",
        ]

        with patch("sys.argv", argv):
            result = cli.run_benchmark_cli()

        self.assertEqual(result, {"ok": True})
        mock_benchmark_cls.assert_called_once_with(
            "https://api.openai.com/v1/chat/completions",
            "test-key",
            "gpt-3.5-turbo",
            "openai",
            timeout=45.0,
            warmup_runs=0,
        )
        mock_benchmark.run_comprehensive_benchmark.assert_called_once()


if __name__ == "__main__":
    unittest.main()
