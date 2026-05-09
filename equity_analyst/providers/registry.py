from __future__ import annotations

from collections.abc import Callable

from equity_analyst.providers.anthropic_provider import AnthropicProvider
from equity_analyst.providers.base import LLMProvider
from equity_analyst.providers.gemini_provider import GeminiProvider
from equity_analyst.providers.openai_provider import OpenAIProvider

ProviderFactory = Callable[[], LLMProvider]


class ProviderRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, name: str, factory: ProviderFactory) -> None:
        self._factories[name] = factory

    def create(self, name: str) -> LLMProvider:
        try:
            return self._factories[name]()
        except KeyError as e:
            raise KeyError(f"Unknown provider '{name}'. Registered: {sorted(self._factories)}") from e

    @classmethod
    def default(cls) -> ProviderRegistry:
        reg = cls()
        reg.register("anthropic", lambda: AnthropicProvider())
        reg.register("openai", lambda: OpenAIProvider())
        reg.register("gemini", lambda: GeminiProvider())
        return reg

