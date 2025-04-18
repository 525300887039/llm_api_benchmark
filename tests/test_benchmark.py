"""LLM API基准测试工具的单元测试."""

import unittest
from unittest.mock import patch, MagicMock
import json
from datetime import datetime

from llm_api_benchmark.benchmark import LLMAPIBenchmark


class TestLLMAPIBenchmark(unittest.TestCase):
    """LLMAPIBenchmark类的测试."""

    def setUp(self):
        """设置测试环境."""
        self.api_url = "https://api.test.com/v1/chat/completions"
        self.api_key = "test_key"
        self.model = "test-model"
        self.benchmark = LLMAPIBenchmark(self.api_url, self.api_key, self.model)
        self.test_prompt = "这是一个测试提示词"

    @patch('llm_api_benchmark.benchmark.requests.post')
    def test_measure_first_token_latency(self, mock_post):
        """测试首字延迟测量功能."""
        # 模拟响应
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [b'data: {"id":"test"}']
        mock_post.return_value = mock_response

        # 运行测试
        with patch('llm_api_benchmark.benchmark.time.time') as mock_time:
            mock_time.side_effect = [0, 0.5]  # 模拟起始时间和第一个token时间
            latency = self.benchmark.measure_first_token_latency(self.test_prompt, num_runs=1)

        # 验证结果
        self.assertEqual(latency, 0.5)
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args[1]['json']['model'], self.model)
        self.assertEqual(mock_post.call_args[1]['json']['messages'][0]['content'], self.test_prompt)

    @patch('llm_api_benchmark.benchmark.requests.post')
    def test_measure_token_throughput(self, mock_post):
        """测试token吞吐量测量功能."""
        # 模拟响应
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "usage": {"completion_tokens": 100},
            "choices": [{"message": {"content": "测试回复" * 20}}]
        }
        mock_post.return_value = mock_response

        # 运行测试
        with patch('llm_api_benchmark.benchmark.time.time') as mock_time:
            mock_time.side_effect = [0, 2.0]  # 模拟起始时间和结束时间（2秒）
            throughput = self.benchmark.measure_token_throughput(self.test_prompt, runs=1)

        # 验证结果 (100 tokens / 2秒 = 50 tokens/秒)
        self.assertEqual(throughput, 50.0)
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args[1]['json']['model'], self.model)
        self.assertEqual(mock_post.call_args[1]['json']['messages'][0]['content'], self.test_prompt)

    @patch('llm_api_benchmark.benchmark.LLMAPIBenchmark.measure_first_token_latency')
    @patch('llm_api_benchmark.benchmark.LLMAPIBenchmark.measure_token_throughput')
    def test_run_comprehensive_benchmark(self, mock_throughput, mock_latency):
        """测试综合基准测试功能."""
        # 模拟方法返回值
        mock_latency.return_value = 0.5
        mock_throughput.return_value = 50.0

        # 运行测试
        with patch('llm_api_benchmark.benchmark.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 0, 0)
            results = self.benchmark.run_comprehensive_benchmark(self.test_prompt, runs=2)

        # 验证结果
        self.assertEqual(results['model'], self.model)
        self.assertEqual(results['api_url'], self.api_url)
        self.assertEqual(results['first_token_latency'], 0.5)
        self.assertEqual(results['token_throughput'], 50.0)
        self.assertEqual(results['prompt_length'], len(self.test_prompt))
        self.assertEqual(results['runs'], 2)


if __name__ == '__main__':
    unittest.main()