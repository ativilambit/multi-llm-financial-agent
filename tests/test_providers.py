from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from equity_analyst.gemini_cache import GeminiCacheIndex
from equity_analyst.prompt_parts import EQUITY_ANALYST_SYSTEM_PROMPT, ephemeral_cache_control
from equity_analyst.providers.anthropic_provider import AnthropicProvider
from equity_analyst.providers.gemini_provider import DEFAULT_GEMINI_MODEL, GeminiProvider
from equity_analyst.providers.grok_provider import XAI_BASE_URL, GrokProvider
from equity_analyst.providers.openai_provider import (
    EQUITY_FANOUT_PROMPT_CACHE_KEY,
    OpenAIProvider,
)


@dataclass
class _AnthropicUsage:
    input_tokens: int = 11
    output_tokens: int = 22
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


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
class _OpenAIInputTokenDetails:
    cached_tokens: int = 512


@dataclass
class _OpenAIUsage:
    input_tokens: int = 3
    output_tokens: int = 4
    total_tokens: int = 7
    input_tokens_details: Any = field(default_factory=_OpenAIInputTokenDetails)


@dataclass
class _OpenAIResp:
    output: list[Any]
    usage: Any


class _FakeOpenAIStream:
    """Minimal async iterator mimicking AsyncStream[ResponseStreamEvent]."""

    def __init__(self, *, final: _OpenAIResp) -> None:
        self._events: list[Any] = [
            SimpleNamespace(type="response.output_text.delta", delta="openai-"),
            SimpleNamespace(type="response.output_text.delta", delta="answer"),
            SimpleNamespace(type="response.completed", response=final),
        ]
        self._idx = 0

    def __aiter__(self) -> _FakeOpenAIStream:
        return self

    async def __anext__(self) -> Any:
        if self._idx >= len(self._events):
            raise StopAsyncIteration
        ev = self._events[self._idx]
        self._idx += 1
        return ev


class _FakeOpenAIResponses:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None
        self.stream_invoked = False

    async def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        if kwargs.get("stream"):
            self.stream_invoked = True
            final = _OpenAIResp(
                output=[_OpenAIOutputMessage(content=[_OpenAIOutputContent()])],
                usage=_OpenAIUsage(),
            )
            return _FakeOpenAIStream(final=final)
        return _OpenAIResp(
            output=[_OpenAIOutputMessage(content=[_OpenAIOutputContent()])],
            usage=_OpenAIUsage(),
        )


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = _FakeOpenAIResponses()


@pytest.mark.asyncio
async def test_anthropic_prompt_cache_adds_cache_control_to_system_and_tools() -> None:
    fake = _FakeAnthropicClient()
    p = AnthropicProvider(model="claude-3-5-sonnet-latest", client=fake)  # type: ignore[arg-type]
    user_body = "Today price range BODY"
    full = f"{EQUITY_ANALYST_SYSTEM_PROMPT}\n\n{user_body}"
    await p.generate(full, enable_web_search=True, prompt_cache_enabled=True)

    assert fake.messages.last_kwargs is not None
    kw = fake.messages.last_kwargs
    assert kw["messages"][0]["content"] == user_body
    sys_blocks = kw["system"]
    assert isinstance(sys_blocks, list)
    assert sys_blocks[0]["cache_control"] == ephemeral_cache_control()
    assert sys_blocks[0]["text"] == EQUITY_ANALYST_SYSTEM_PROMPT
    tools = kw.get("tools") or []
    assert tools[-1]["cache_control"] == ephemeral_cache_control()


@pytest.mark.asyncio
async def test_anthropic_prompt_cache_disabled_no_cache_markers() -> None:
    fake = _FakeAnthropicClient()
    p = AnthropicProvider(model="claude-3-5-sonnet-latest", client=fake)  # type: ignore[arg-type]
    user_body = "Today price range BODY"
    full = f"{EQUITY_ANALYST_SYSTEM_PROMPT}\n\n{user_body}"
    await p.generate(full, enable_web_search=True, prompt_cache_enabled=False)

    assert fake.messages.last_kwargs is not None
    kw = fake.messages.last_kwargs
    assert kw["messages"][0]["content"] == full
    assert "system" not in kw
    tool = (kw.get("tools") or [])[0]
    assert "cache_control" not in tool


@pytest.mark.asyncio
async def test_anthropic_cache_stats_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    @dataclass
    class _U:
        input_tokens: int = 678
        output_tokens: int = 4321
        cache_read_input_tokens: int = 12345
        cache_creation_input_tokens: int = 0

    @dataclass
    class _Msg:
        content: list[Any]
        usage: Any
        model: str = "claude-test-model"

    class _Stream2:
        def __init__(self, msg: _Msg) -> None:
            self._msg = msg

        async def until_done(self) -> None:
            return

        async def get_final_message(self) -> _Msg:
            return self._msg

    class _StreamCM2:
        def __init__(self, msg: _Msg) -> None:
            self._stream = _Stream2(msg)

        async def __aenter__(self) -> _Stream2:
            return self._stream

        async def __aexit__(self, *args: object) -> None:
            return

    class _Messages2:
        def stream(self, **kwargs: Any) -> _StreamCM2:
            msg = _Msg(content=[_AnthropicTextBlock()], usage=_U())
            return _StreamCM2(msg)

    class _Client2:
        def __init__(self) -> None:
            self.messages = _Messages2()

    fake = _Client2()
    p = AnthropicProvider(model="claude-3-5-sonnet-latest", client=fake)  # type: ignore[arg-type]
    full = f"{EQUITY_ANALYST_SYSTEM_PROMPT}\n\nBODY"
    await p.generate(full, enable_web_search=False, prompt_cache_enabled=True)

    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "Anthropic cache stats cache_read=12345 cache_creation=0 input=678 output=4321" in logged


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
    assert fake.messages.last_kwargs.get("tool_choice") == {"type": "any"}


@pytest.mark.asyncio
async def test_anthropic_tool_choice_when_force_enabled() -> None:
    fake = _FakeAnthropicClient()
    p = AnthropicProvider(model="claude-3-5-sonnet-latest", client=fake)  # type: ignore[arg-type]
    await p.generate("hello", enable_web_search=True, force_tool_use=True)

    assert fake.messages.last_kwargs is not None
    assert fake.messages.last_kwargs.get("tool_choice") == {"type": "any"}


@pytest.mark.asyncio
async def test_anthropic_no_tool_choice_when_force_disabled() -> None:
    fake = _FakeAnthropicClient()
    p = AnthropicProvider(model="claude-3-5-sonnet-latest", client=fake)  # type: ignore[arg-type]
    await p.generate("hello", enable_web_search=True, force_tool_use=False)

    assert fake.messages.last_kwargs is not None
    assert "tool_choice" not in fake.messages.last_kwargs


@pytest.mark.asyncio
async def test_openai_cache_stats_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    fake = _FakeOpenAIClient()
    p = OpenAIProvider(model="gpt-5.5", client=fake)  # type: ignore[arg-type]
    await p.generate("hello", enable_web_search=True)

    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "OpenAI cache stats cache_read=512 input=3 output=4" in logged


@pytest.mark.asyncio
async def test_openai_provider_assembles_request_and_parses_usage() -> None:
    fake = _FakeOpenAIClient()
    p = OpenAIProvider(model="gpt-5.5", client=fake)  # type: ignore[arg-type]
    resp = await p.generate("hello", enable_web_search=True)

    assert fake.responses.stream_invoked is True
    assert resp.text == "openai-answer"
    assert resp.usage.input_tokens == 3
    assert resp.usage.output_tokens == 4
    assert resp.usage.total_tokens == 7
    assert fake.responses.last_kwargs is not None
    assert fake.responses.last_kwargs["model"] == "gpt-5.5"
    assert fake.responses.last_kwargs["input"] == "hello"
    assert fake.responses.last_kwargs.get("stream") is True
    assert "prompt_cache_key" not in fake.responses.last_kwargs
    assert {"type": "web_search"} in (fake.responses.last_kwargs["tools"] or [])


@pytest.mark.asyncio
async def test_openai_fanout_structured_input_system_first_for_caching() -> None:
    fake = _FakeOpenAIClient()
    p = OpenAIProvider(model="gpt-5.5", client=fake)  # type: ignore[arg-type]
    static = "STATIC\n" * 50
    user = "USER body"
    full = f"{static}\n\n{user}"
    await p.generate(
        full,
        enable_web_search=True,
        cacheable_prefix=static,
        user_message_for_cache=user,
    )
    assert fake.responses.last_kwargs is not None
    kw = fake.responses.last_kwargs
    inp = kw["input"]
    assert isinstance(inp, list)
    assert kw.get("instructions") == static
    assert inp[0]["type"] == "message"
    assert inp[0]["role"] == "user"
    assert inp[0]["content"] == user
    assert len(inp) == 1
    assert kw.get("prompt_cache_key") == EQUITY_FANOUT_PROMPT_CACHE_KEY
    assert kw.get("prompt_cache_retention") == "24h"


@pytest.mark.asyncio
async def test_openai_prompt_cache_key_identical_for_same_prefix_different_user_bodies() -> None:
    fake = _FakeOpenAIClient()
    p = OpenAIProvider(model="gpt-5.5", client=fake)  # type: ignore[arg-type]
    static = "STATIC\n" * 50
    keys: list[Any] = []
    for user in ("first-user-body", "second-user-body" + "x" * 400):
        full = f"{static}\n\n{user}"
        await p.generate(
            full,
            enable_web_search=True,
            cacheable_prefix=static,
            user_message_for_cache=user,
        )
        assert fake.responses.last_kwargs is not None
        keys.append(fake.responses.last_kwargs.get("prompt_cache_key"))
    assert keys == [EQUITY_FANOUT_PROMPT_CACHE_KEY, EQUITY_FANOUT_PROMPT_CACHE_KEY]


@pytest.mark.asyncio
async def test_openai_debug_prefix_hash_matches_for_identical_requests(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="equity_analyst.providers.openai_provider")
    fake = _FakeOpenAIClient()
    p = OpenAIProvider(model="gpt-5.5", client=fake)  # type: ignore[arg-type]
    prompt = "identical-prefix-for-cache-check\n" + "x" * 300
    await p.generate(prompt, enable_web_search=True)
    await p.generate(prompt, enable_web_search=True)

    prefix_msgs = [
        r.getMessage()
        for r in caplog.records
        if r.levelname == "DEBUG" and "OpenAI request prefix" in r.getMessage()
    ]
    assert len(prefix_msgs) == 2
    hashes: list[str] = []
    for msg in prefix_msgs:
        assert "prefix_chars=200" in msg
        assert "input: identical-prefix-for-cache-check" in msg
        tail = msg.split("hash=", 1)[1]
        hashes.append(tail.split()[0].rstrip(","))
    assert hashes[0] == hashes[1]


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
        self.count_tokens_calls = 0
        self.count_tokens_total = 2000
        self.last_count_contents: Any = None
        self.last_count_config: Any = None

    async def count_tokens(self, *, model: str, contents: Any, config: Any = None) -> Any:
        self.count_tokens_calls += 1
        self.last_count_contents = contents
        self.last_count_config = config
        return SimpleNamespace(total_tokens=self.count_tokens_total, cached_content_token_count=0)

    async def generate_content(self, *, model: str, contents: str, config: Any = None) -> Any:
        self.last_model = model
        self.last_contents = contents
        self.last_config = config
        return _FakeGeminiGenerateContentResponse()


class _FakeGeminiAioCaches:
    def __init__(self) -> None:
        self.create_calls = 0
        self.last_create_config: Any = None

    async def create(self, *, model: str, config: Any = None) -> Any:
        self.create_calls += 1
        self.last_create_config = config
        return SimpleNamespace(name="cachedContents/fake-from-create")


class _FakeGeminiAio:
    def __init__(self) -> None:
        self.models = _FakeGeminiAioModels()
        self.caches = _FakeGeminiAioCaches()


class _FakeGeminiClient:
    def __init__(self) -> None:
        self.aio = _FakeGeminiAio()


@pytest.mark.asyncio
async def test_gemini_provider_default_model_is_latest_pro() -> None:
    fake = _FakeGeminiClient()
    p = GeminiProvider(client=fake)  # type: ignore[arg-type]
    resp = await p.generate("hello", enable_web_search=False)

    assert resp.model == DEFAULT_GEMINI_MODEL
    assert fake.aio.models.last_model == DEFAULT_GEMINI_MODEL


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


@pytest.mark.asyncio
async def test_gemini_uses_cache_on_hit(tmp_path: Path) -> None:
    fake = _FakeGeminiClient()
    idx = GeminiCacheIndex(path=tmp_path / "idx.json")
    # Rough char estimate len//4 must clear Flash min (1024) so count_tokens runs.
    static = "x" * 5000
    user = "dynamic-body"
    full = f"{static}\n\n{user}"
    p = GeminiProvider(
        model="gemini-2.5-flash",
        client=fake,  # type: ignore[arg-type]
        cache_index=idx,
        cache_ttl_s=3600,
    )
    await p.generate(
        full,
        enable_web_search=False,
        cacheable_prefix=static,
        user_message_for_cache=user,
    )
    assert fake.aio.caches.create_calls == 1
    assert fake.aio.models.last_contents == user
    assert fake.aio.models.last_config is not None
    assert fake.aio.models.last_config.cached_content == "cachedContents/fake-from-create"
    assert getattr(fake.aio.models.last_config, "tools", None) in (None, [])
    assert getattr(fake.aio.models.last_config, "system_instruction", None) is None
    assert getattr(fake.aio.models.last_config, "tool_config", None) is None
    cc = fake.aio.caches.last_create_config
    assert cc is not None
    assert cc.system_instruction == static
    assert getattr(cc, "tools", None) in (None, [])

    await p.generate(
        full,
        enable_web_search=False,
        cacheable_prefix=static,
        user_message_for_cache=user,
    )
    assert fake.aio.caches.create_calls == 1
    assert fake.aio.models.last_config is not None
    assert fake.aio.models.last_config.cached_content == "cachedContents/fake-from-create"
    assert getattr(fake.aio.models.last_config, "tools", None) in (None, [])


@pytest.mark.asyncio
async def test_gemini_cached_path_puts_tools_on_cache_create_not_on_generate(tmp_path: Path) -> None:
    fake = _FakeGeminiClient()
    idx = GeminiCacheIndex(path=tmp_path / "idx.json")
    static = "x" * 5000
    user = "dynamic-body"
    full = f"{static}\n\n{user}"
    p = GeminiProvider(
        model="gemini-2.5-flash",
        client=fake,  # type: ignore[arg-type]
        cache_index=idx,
        cache_ttl_s=3600,
    )
    await p.generate(
        full,
        enable_web_search=True,
        cacheable_prefix=static,
        user_message_for_cache=user,
    )
    assert fake.aio.caches.create_calls == 1
    assert fake.aio.models.last_config is not None
    assert fake.aio.models.last_config.cached_content == "cachedContents/fake-from-create"
    assert getattr(fake.aio.models.last_config, "tools", None) in (None, [])
    cc = fake.aio.caches.last_create_config
    assert cc is not None
    assert cc.tools is not None
    assert cc.tools[0].google_search is not None

    await p.generate(
        full,
        enable_web_search=True,
        cacheable_prefix=static,
        user_message_for_cache=user,
    )
    assert fake.aio.caches.create_calls == 1
    assert getattr(fake.aio.models.last_config, "tools", None) in (None, [])


@pytest.mark.asyncio
async def test_gemini_skips_cache_when_disabled() -> None:
    fake = _FakeGeminiClient()
    static = "x" * 4000
    user = "u"
    full = f"{static}\n\n{user}"
    p = GeminiProvider(model="gemini-2.5-flash", client=fake)  # type: ignore[arg-type]
    await p.generate(
        full,
        enable_web_search=False,
        cacheable_prefix=static,
        user_message_for_cache=user,
    )
    assert fake.aio.caches.create_calls == 0
    assert fake.aio.models.count_tokens_calls == 0
    assert fake.aio.models.last_contents == full
    assert getattr(fake.aio.models.last_config, "cached_content", None) in (None,)


@pytest.mark.asyncio
async def test_gemini_skips_cache_when_prefix_too_small(tmp_path: Path) -> None:
    fake = _FakeGeminiClient()
    fake.aio.models.count_tokens_total = 100
    idx = GeminiCacheIndex(path=tmp_path / "idx.json")
    # Long enough for rough estimate to pass; precise count still below Flash minimum.
    static = "x" * 5000
    user = "u"
    full = f"{static}\n\n{user}"
    p = GeminiProvider(
        model="gemini-2.5-flash",
        client=fake,  # type: ignore[arg-type]
        cache_index=idx,
    )
    await p.generate(
        full,
        enable_web_search=False,
        cacheable_prefix=static,
        user_message_for_cache=user,
    )
    assert fake.aio.caches.create_calls == 0
    assert fake.aio.models.count_tokens_calls == 1
    assert fake.aio.models.last_count_contents == static
    assert fake.aio.models.last_contents == full
    assert getattr(fake.aio.models.last_config, "cached_content", None) in (None,)


class _FakeGeminiAioModelsCountTokensRaise(_FakeGeminiAioModels):
    async def count_tokens(self, *, model: str, contents: Any, config: Any = None) -> Any:
        raise ValueError("contents are required.")


@pytest.mark.asyncio
async def test_gemini_count_tokens_failure_falls_back_gracefully(tmp_path: Path) -> None:
    fake = _FakeGeminiClient()
    fake.aio.models = _FakeGeminiAioModelsCountTokensRaise()
    idx = GeminiCacheIndex(path=tmp_path / "idx.json")
    static = "x" * 5000
    user = "dynamic-body"
    full = f"{static}\n\n{user}"
    p = GeminiProvider(
        model="gemini-2.5-flash",
        client=fake,  # type: ignore[arg-type]
        cache_index=idx,
    )
    resp = await p.generate(
        full,
        enable_web_search=False,
        cacheable_prefix=static,
        user_message_for_cache=user,
    )
    assert resp.text == "gemini-answer"
    assert fake.aio.caches.create_calls == 0
    assert fake.aio.models.last_contents == full
    assert getattr(fake.aio.models.last_config, "cached_content", None) in (None,)


@pytest.mark.asyncio
async def test_gemini_skips_count_tokens_when_estimate_below_minimum(tmp_path: Path) -> None:
    fake = _FakeGeminiClient()
    idx = GeminiCacheIndex(path=tmp_path / "idx.json")
    static = "a" * 200
    user = "u"
    full = f"{static}\n\n{user}"
    p = GeminiProvider(
        model="gemini-2.5-flash",
        client=fake,  # type: ignore[arg-type]
        cache_index=idx,
    )
    resp = await p.generate(
        full,
        enable_web_search=False,
        cacheable_prefix=static,
        user_message_for_cache=user,
    )
    assert resp.text == "gemini-answer"
    assert fake.aio.models.count_tokens_calls == 0
    assert fake.aio.caches.create_calls == 0
    assert fake.aio.models.last_contents == full
    assert getattr(fake.aio.models.last_config, "cached_content", None) in (None,)


def test_grok_provider_uses_xai_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _ctor(**kwargs: Any) -> _FakeOpenAIClient:
        captured.update(kwargs)
        return _FakeOpenAIClient()

    monkeypatch.setattr("equity_analyst.providers.grok_provider.AsyncOpenAI", _ctor)
    GrokProvider(model="grok-4.3")
    assert captured.get("base_url") == XAI_BASE_URL


@pytest.mark.asyncio
async def test_grok_cache_stats_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    fake = _FakeOpenAIClient()
    p = GrokProvider(model="grok-4.3", client=fake)  # type: ignore[arg-type]
    await p.generate("hello", enable_web_search=True)

    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "Grok cache stats cache_read=512 input=3 output=4" in logged


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
