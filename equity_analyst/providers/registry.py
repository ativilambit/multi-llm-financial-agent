from __future__ import annotations

from collections.abc import Callable
from typing import Any

from equity_analyst.providers.anthropic_provider import AnthropicProvider
from equity_analyst.providers.base import LLMProvider
from equity_analyst.providers.gemini_provider import GeminiProvider
from equity_analyst.providers.grok_provider import GrokProvider
from equity_analyst.providers.openai_provider import OpenAIProvider

ProviderFactory = Callable[..., LLMProvider]


class ProviderRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, name: str, factory: ProviderFactory) -> None:
        self._factories[name] = factory

    def create(self, name: str, *, model: str | None = None, client: Any | None = None) -> LLMProvider:
        try:
            factory = self._factories[name]
        except KeyError as e:
            raise KeyError(f"Unknown provider '{name}'. Registered: {sorted(self._factories)}") from e
        kwargs: dict[str, Any] = {}
        if model is not None:
            kwargs["model"] = model
        if client is not None:
            kwargs["client"] = client
        return factory(**kwargs)

    @classmethod
    def default(cls) -> ProviderRegistry:
        reg = cls()
        reg.register("anthropic", lambda **kw: AnthropicProvider(**kw))
        reg.register("openai", lambda **kw: OpenAIProvider(**kw))
        reg.register("gemini", lambda **kw: GeminiProvider(**kw))
        reg.register("grok", lambda **kw: GrokProvider(**kw))
        return reg
