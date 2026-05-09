from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from equity_analyst.providers.anthropic_provider import AnthropicProvider
from equity_analyst.providers.openai_provider import OpenAIProvider


@dataclass
class _AnthropicUsage:
    input_tokens: int = 11
    output_tokens: int = 22


@dataclass
class _AnthropicTextBlock:
    type: str = "text"
    text: str = "anthropic-answer"


@dataclass
class _AnthropicMsg:
    content: list[Any]
    usage: Any


class _FakeAnthropicMessages:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> _AnthropicMsg:
        self.last_kwargs = kwargs
        return _AnthropicMsg(content=[_AnthropicTextBlock()], usage=_AnthropicUsage())


class _FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = _FakeAnthropicMessages()


@dataclass
class _OpenAIOutputContent:
    type: str = "output_text"
    text: str = "openai-answer"


@dataclass
class _OpenAIOutputMessage:
    content: list[Any]
    type: str = "message"


@dataclass
class _OpenAIUsage:
    input_tokens: int = 3
    output_tokens: int = 4
    total_tokens: int = 7


@dataclass
class _OpenAIResp:
    output: list[Any]
    usage: Any


class _FakeOpenAIResponses:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        return _OpenAIResp(
            output=[_OpenAIOutputMessage(content=[_OpenAIOutputContent()])],
            usage=_OpenAIUsage(),
        )


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = _FakeOpenAIResponses()


@pytest.mark.asyncio
async def test_anthropic_provider_assembles_request_and_parses_usage() -> None:
    fake = _FakeAnthropicClient()
    p = AnthropicProvider(model="claude-3-5-sonnet-latest", client=fake)  # type: ignore[arg-type]
    resp = await p.generate("hello", enable_web_search=True)

    assert resp.text == "anthropic-answer"
    assert resp.usage.input_tokens == 11
    assert resp.usage.output_tokens == 22
    assert fake.messages.last_kwargs is not None
    assert fake.messages.last_kwargs["model"] == "claude-3-5-sonnet-latest"
    assert fake.messages.last_kwargs["messages"][0]["content"] == "hello"
    assert {"type": "web_search_20260209", "name": "web_search"} in (fake.messages.last_kwargs["tools"] or [])


@pytest.mark.asyncio
async def test_openai_provider_assembles_request_and_parses_usage() -> None:
    fake = _FakeOpenAIClient()
    p = OpenAIProvider(model="gpt-5.5", client=fake)  # type: ignore[arg-type]
    resp = await p.generate("hello", enable_web_search=True)

    assert resp.text == "openai-answer"
    assert resp.usage.input_tokens == 3
    assert resp.usage.output_tokens == 4
    assert resp.usage.total_tokens == 7
    assert fake.responses.last_kwargs is not None
    assert fake.responses.last_kwargs["model"] == "gpt-5.5"
    assert fake.responses.last_kwargs["input"] == "hello"
    assert {"type": "web_search"} in (fake.responses.last_kwargs["tools"] or [])
