from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from equity_analyst.providers.anthropic_provider import AnthropicProvider
from equity_analyst.providers.gemini_provider import GeminiProvider
from equity_analyst.providers.grok_provider import XAI_BASE_URL, GrokProvider
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
    model: str = "claude-test-model"


class _FakeAnthropicStream:
    def __init__(self, msg: _AnthropicMsg) -> None:
        self._msg = msg

    async def until_done(self) -> None:
        return

    async def get_final_message(self) -> _AnthropicMsg:
        return self._msg

    async def close(self) -> None:
        return


class _FakeAnthropicStreamCM:
    def __init__(self, msg: _AnthropicMsg) -> None:
        self._stream = _FakeAnthropicStream(msg)

    async def __aenter__(self) -> _FakeAnthropicStream:
        return self._stream

    async def __aexit__(self, *args: object) -> None:
        return


class _FakeAnthropicMessages:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None
        self.stream_invoked = False

    def stream(self, **kwargs: Any) -> _FakeAnthropicStreamCM:
        self.stream_invoked = True
        self.last_kwargs = kwargs
        model_id = str(kwargs.get("model", "claude-test-model"))
        msg = _AnthropicMsg(
            content=[_AnthropicTextBlock()],
            usage=_AnthropicUsage(),
            model=model_id,
        )
        return _FakeAnthropicStreamCM(msg)


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

    assert fake.messages.stream_invoked is True
    assert resp.text == "anthropic-answer"
    assert resp.usage.input_tokens == 11
    assert resp.usage.output_tokens == 22
    assert resp.model == "claude-3-5-sonnet-latest"
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


class _FakeGeminiUsage:
    prompt_token_count = 5
    candidates_token_count = 6
    total_token_count = 11


class _FakeGeminiGenerateContentResponse:
    def __init__(self) -> None:
        self.usage_metadata = _FakeGeminiUsage()

    @property
    def text(self) -> str:
        return "gemini-answer"


class _FakeGeminiAioModels:
    def __init__(self) -> None:
        self.last_model: str | None = None
        self.last_contents: str | None = None
        self.last_config: Any = None

    async def generate_content(self, *, model: str, contents: str, config: Any = None) -> Any:
        self.last_model = model
        self.last_contents = contents
        self.last_config = config
        return _FakeGeminiGenerateContentResponse()


class _FakeGeminiAio:
    def __init__(self) -> None:
        self.models = _FakeGeminiAioModels()


class _FakeGeminiClient:
    def __init__(self) -> None:
        self.aio = _FakeGeminiAio()


@pytest.mark.asyncio
async def test_gemini_provider_assembles_request_and_parses_usage() -> None:
    fake = _FakeGeminiClient()
    p = GeminiProvider(model="gemini-2.5-flash", client=fake)  # type: ignore[arg-type]
    resp = await p.generate("hello", enable_web_search=True)

    assert resp.text == "gemini-answer"
    assert resp.usage.input_tokens == 5
    assert resp.usage.output_tokens == 6
    assert resp.usage.total_tokens == 11
    m = fake.aio.models
    assert m.last_model == "gemini-2.5-flash"
    assert m.last_contents == "hello"
    assert m.last_config is not None
    assert m.last_config.tools is not None
    assert m.last_config.tools[0].google_search is not None

    resp2 = await p.generate("hi", enable_web_search=False)
    assert resp2.text == "gemini-answer"
    assert m.last_config is None


def test_grok_provider_uses_xai_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _ctor(**kwargs: Any) -> _FakeOpenAIClient:
        captured.update(kwargs)
        return _FakeOpenAIClient()

    monkeypatch.setattr("equity_analyst.providers.grok_provider.AsyncOpenAI", _ctor)
    GrokProvider(model="grok-4.3")
    assert captured.get("base_url") == XAI_BASE_URL


@pytest.mark.asyncio
async def test_grok_provider_assembles_request_and_parses_usage() -> None:
    fake = _FakeOpenAIClient()
    p = GrokProvider(model="grok-4.3", client=fake)  # type: ignore[arg-type]
    resp = await p.generate("hello", enable_web_search=True)

    assert resp.text == "openai-answer"
    assert resp.usage.input_tokens == 3
    assert fake.responses.last_kwargs is not None
    assert fake.responses.last_kwargs["model"] == "grok-4.3"
    assert fake.responses.last_kwargs["input"] == "hello"
    tools = fake.responses.last_kwargs.get("tools") or []
    assert {"type": "web_search"} in tools

    await p.generate("hi", enable_web_search=False)
    assert fake.responses.last_kwargs is not None
    assert fake.responses.last_kwargs.get("tools") in (None, [])
