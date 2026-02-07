"""API Provider 适配器的单元测试."""

import unittest
import json

from llm_api_benchmark.providers import (
    OpenAIProvider,
    ClaudeProvider,
    AzureOpenAIProvider,
    create_provider,
)


class TestOpenAIProvider(unittest.TestCase):
    """OpenAI Provider 测试."""

    def setUp(self):
        self.provider = OpenAIProvider(
            "https://api.openai.com/v1/chat/completions", "sk-test", "gpt-4"
        )

    def test_get_headers(self):
        headers = self.provider.get_headers()
        self.assertEqual(headers["Authorization"], "Bearer sk-test")
        self.assertEqual(headers["Content-Type"], "application/json")

    def test_build_chat_payload_stream(self):
        payload = self.provider.build_chat_payload("hello", stream=True)
        self.assertEqual(payload["model"], "gpt-4")
        self.assertEqual(payload["messages"][0]["content"], "hello")
        self.assertTrue(payload["stream"])

    def test_build_chat_payload_no_stream(self):
        payload = self.provider.build_chat_payload("hello", stream=False)
        self.assertFalse(payload["stream"])

    def test_parse_token_count(self):
        resp = {"usage": {"completion_tokens": 42}, "choices": []}
        self.assertEqual(self.provider.parse_token_count(resp), 42)

    def test_parse_token_count_fallback(self):
        resp = {"usage": {}, "choices": [{"message": {"content": "a b c"}}]}
        self.assertEqual(self.provider.parse_token_count(resp), 3)

    def test_parse_content(self):
        resp = {"choices": [{"message": {"content": "hello world"}}]}
        self.assertEqual(self.provider.parse_content(resp), "hello world")

    def test_is_first_content_event(self):
        self.assertTrue(self.provider.is_first_content_event(b'data: {"id":"x"}'))
        self.assertFalse(self.provider.is_first_content_event(b''))
        self.assertFalse(self.provider.is_first_content_event(b'   '))


class TestClaudeProvider(unittest.TestCase):
    """Claude Provider 测试."""

    def setUp(self):
        self.provider = ClaudeProvider(
            "https://api.anthropic.com/v1/messages", "sk-ant-test", "claude-3-sonnet"
        )

    def test_get_headers(self):
        headers = self.provider.get_headers()
        self.assertEqual(headers["x-api-key"], "sk-ant-test")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertNotIn("Authorization", headers)

    def test_build_chat_payload_stream(self):
        payload = self.provider.build_chat_payload("hello", stream=True)
        self.assertEqual(payload["model"], "claude-3-sonnet")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertTrue(payload["stream"])

    def test_build_chat_payload_no_stream(self):
        payload = self.provider.build_chat_payload("hello", stream=False)
        self.assertNotIn("stream", payload)

    def test_parse_token_count(self):
        resp = {"usage": {"output_tokens": 55}}
        self.assertEqual(self.provider.parse_token_count(resp), 55)

    def test_parse_content(self):
        resp = {"content": [{"type": "text", "text": "hello"}]}
        self.assertEqual(self.provider.parse_content(resp), "hello")

    def test_parse_content_empty(self):
        resp = {"content": []}
        self.assertEqual(self.provider.parse_content(resp), "")

    def test_is_first_content_event(self):
        delta_line = b'data: {"type":"content_block_delta","delta":{"text":"Hi"}}'
        self.assertTrue(self.provider.is_first_content_event(delta_line))
        self.assertFalse(self.provider.is_first_content_event(b''))
        self.assertFalse(self.provider.is_first_content_event(b'event: ping'))


class TestAzureOpenAIProvider(unittest.TestCase):
    """Azure OpenAI Provider 测试."""

    def setUp(self):
        self.provider = AzureOpenAIProvider(
            "https://myres.openai.azure.com/openai/deployments/gpt4/chat/completions?api-version=2024-02-01",
            "azure-key-test",
            "gpt-4",
        )

    def test_get_headers(self):
        headers = self.provider.get_headers()
        self.assertEqual(headers["api-key"], "azure-key-test")
        self.assertNotIn("Authorization", headers)

    def test_build_chat_payload(self):
        payload = self.provider.build_chat_payload("hello", stream=False)
        # Azure 不在 payload 中包含 model
        self.assertNotIn("model", payload)
        self.assertEqual(payload["messages"][0]["content"], "hello")

    def test_parse_token_count(self):
        resp = {"usage": {"completion_tokens": 30}, "choices": []}
        self.assertEqual(self.provider.parse_token_count(resp), 30)


class TestCreateProvider(unittest.TestCase):
    """工厂函数测试."""

    def test_create_openai(self):
        p = create_provider("openai", "http://test", "key", "model")
        self.assertIsInstance(p, OpenAIProvider)

    def test_create_claude(self):
        p = create_provider("claude", "http://test", "key", "model")
        self.assertIsInstance(p, ClaudeProvider)

    def test_create_azure(self):
        p = create_provider("azure", "http://test", "key", "model")
        self.assertIsInstance(p, AzureOpenAIProvider)

    def test_create_case_insensitive(self):
        p = create_provider("OpenAI", "http://test", "key", "model")
        self.assertIsInstance(p, OpenAIProvider)

    def test_create_invalid(self):
        with self.assertRaises(ValueError):
            create_provider("invalid_type", "http://test", "key", "model")


if __name__ == '__main__':
    unittest.main()
