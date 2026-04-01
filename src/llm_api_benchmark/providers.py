"""API Provider 适配器模块，支持不同的 LLM API 格式."""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def _parse_sse_json(line: bytes):
    """解析 SSE 行并返回 JSON 数据，无效行返回 None."""
    if not line:
        return None

    decoded = line.decode("utf-8", errors="ignore").strip()
    if not decoded.startswith("data:"):
        return None

    payload = decoded[5:].strip()
    if not payload or payload == "[DONE]":
        return None

    try:
        return json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return None


def _is_openai_first_content_event(line: bytes) -> bool:
    """判断 OpenAI/Azure OpenAI SSE 行是否包含首个非空内容 token."""
    data = _parse_sse_json(line)
    if data is None:
        return False

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return False

    delta = choices[0].get("delta", {})
    if not isinstance(delta, dict):
        return False

    content = delta.get("content")
    return bool(content)


def _append_query_params(url: str, params: Dict[str, str]) -> str:
    """向 URL 追加或覆盖查询参数."""
    parsed = urlsplit(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update(params)
    return urlunsplit(parsed._replace(query=urlencode(query)))


def _extract_gemini_text(response_json: Any) -> str:
    """从 Gemini 响应对象或分块数组中提取首段文本."""
    payloads = response_json if isinstance(response_json, list) else [response_json]

    for payload in payloads:
        if not isinstance(payload, dict):
            continue

        candidates = payload.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            continue

        content = candidates[0].get("content", {})
        if not isinstance(content, dict):
            continue

        parts = content.get("parts", [])
        if not isinstance(parts, list) or not parts:
            continue

        text = parts[0].get("text", "")
        if text:
            return text

    return ""


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
    def get_request_url(self, stream: bool = False) -> str:
        """返回请求 URL，必要时根据流式模式调整."""
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

    def get_request_url(self, stream: bool = False) -> str:
        return self.api_url

    def parse_token_count(self, response_json: Dict[str, Any]) -> int:
        count = response_json.get("usage", {}).get("completion_tokens", 0)
        if count == 0:
            content = self.parse_content(response_json)
            count = len(content.split()) if content else 0
        return count

    def parse_content(self, response_json: Dict[str, Any]) -> str:
        return response_json.get("choices", [{}])[0].get("message", {}).get("content", "")

    def is_first_content_event(self, line: bytes) -> bool:
        return _is_openai_first_content_event(line)


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

    def get_request_url(self, stream: bool = False) -> str:
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


class AzureOpenAIProvider(OpenAIProvider):
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


class GeminiProvider(APIProvider):
    """Google Gemini API Provider."""

    def get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

    def build_chat_payload(self, prompt: str, stream: bool) -> Dict[str, Any]:
        return {
            "contents": [{"parts": [{"text": prompt}]}],
        }

    def get_request_url(self, stream: bool = False) -> str:
        url = self.api_url
        if stream and ":streamGenerateContent" not in url:
            url = url.replace(":generateContent", ":streamGenerateContent")

        if stream:
            return _append_query_params(url, {"alt": "sse"})

        return url

    def parse_token_count(self, response_json: Dict[str, Any]) -> int:
        count = response_json.get("usageMetadata", {}).get("candidatesTokenCount", 0)
        return count if isinstance(count, int) else 0

    def parse_content(self, response_json: Dict[str, Any]) -> str:
        return _extract_gemini_text(response_json)

    def is_first_content_event(self, line: bytes) -> bool:
        data = _parse_sse_json(line)
        if data is None:
            return False
        return bool(_extract_gemini_text(data))


# Provider 注册表
PROVIDERS = {
    "openai": OpenAIProvider,
    "claude": ClaudeProvider,
    "azure": AzureOpenAIProvider,
    "gemini": GeminiProvider,
}


def create_provider(api_type: str, api_url: str, api_key: str, model: str) -> APIProvider:
    """
    工厂函数，根据 api_type 创建对应的 Provider.

    Args:
        api_type: API 类型 ("openai", "claude", "azure", "gemini")
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
