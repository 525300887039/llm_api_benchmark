"""API Provider 适配器模块，支持不同的 LLM API 格式."""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict


class APIProvider(ABC):
    """LLM API Provider 抽象基类."""

    def __init__(self, api_url: str, api_key: str, model: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """返回 HTTP 请求头."""
        ...

    @abstractmethod
    def build_chat_payload(self, prompt: str, stream: bool) -> Dict[str, Any]:
        """构建聊天请求体."""
        ...

    @abstractmethod
    def get_request_url(self) -> str:
        """返回请求 URL."""
        ...

    @abstractmethod
    def parse_token_count(self, response_json: Dict[str, Any]) -> int:
        """从非流式响应中提取 output token 数量."""
        ...

    @abstractmethod
    def parse_content(self, response_json: Dict[str, Any]) -> str:
        """从非流式响应中提取文本内容."""
        ...

    @abstractmethod
    def is_first_content_event(self, line: bytes) -> bool:
        """判断 SSE 行是否为首个内容 token."""
        ...


class OpenAIProvider(APIProvider):
    """OpenAI 兼容 API Provider."""

    def get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def build_chat_payload(self, prompt: str, stream: bool) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }

    def get_request_url(self) -> str:
        return self.api_url

    def parse_token_count(self, response_json: Dict[str, Any]) -> int:
        count = response_json.get("usage", {}).get("completion_tokens", 0)
        if count == 0:
            content = self.parse_content(response_json)
            count = len(content.split()) if content else 0
        return count

    def parse_content(self, response_json: Dict[str, Any]) -> str:
        return (
            response_json.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

    def is_first_content_event(self, line: bytes) -> bool:
        return bool(line and line.strip())


class ClaudeProvider(APIProvider):
    """Anthropic Claude API Provider."""

    def get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

    def build_chat_payload(self, prompt: str, stream: bool) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        if stream:
            payload["stream"] = True
        return payload

    def get_request_url(self) -> str:
        return self.api_url

    def parse_token_count(self, response_json: Dict[str, Any]) -> int:
        return response_json.get("usage", {}).get("output_tokens", 0)

    def parse_content(self, response_json: Dict[str, Any]) -> str:
        content_blocks = response_json.get("content", [])
        if content_blocks:
            return content_blocks[0].get("text", "")
        return ""

    def is_first_content_event(self, line: bytes) -> bool:
        if not line:
            return False
        decoded = line.decode("utf-8", errors="ignore").strip()
        if decoded.startswith("data:"):
            try:
                data = json.loads(decoded[5:].strip())
                return data.get("type") == "content_block_delta"
            except (json.JSONDecodeError, ValueError):
                pass
        return False


class AzureOpenAIProvider(APIProvider):
    """Azure OpenAI API Provider."""

    def get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

    def build_chat_payload(self, prompt: str, stream: bool) -> Dict[str, Any]:
        return {
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }

    def get_request_url(self) -> str:
        return self.api_url

    def parse_token_count(self, response_json: Dict[str, Any]) -> int:
        count = response_json.get("usage", {}).get("completion_tokens", 0)
        if count == 0:
            content = self.parse_content(response_json)
            count = len(content.split()) if content else 0
        return count

    def parse_content(self, response_json: Dict[str, Any]) -> str:
        return (
            response_json.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

    def is_first_content_event(self, line: bytes) -> bool:
        return bool(line and line.strip())


# Provider 注册表
PROVIDERS = {
    "openai": OpenAIProvider,
    "claude": ClaudeProvider,
    "azure": AzureOpenAIProvider,
}


def create_provider(api_type: str, api_url: str, api_key: str, model: str) -> APIProvider:
    """
    工厂函数，根据 api_type 创建对应的 Provider.

    Args:
        api_type: API 类型 ("openai", "claude", "azure")
        api_url: API 端点 URL
        api_key: API 密钥
        model: 模型名称

    Returns:
        APIProvider 实例
    """
    provider_class = PROVIDERS.get(api_type.lower())
    if provider_class is None:
        supported = ", ".join(PROVIDERS.keys())
        raise ValueError(f"不支持的 API 类型: {api_type}。支持的类型: {supported}")
    return provider_class(api_url, api_key, model)
