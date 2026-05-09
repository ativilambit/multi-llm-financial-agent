from __future__ import annotations

import abc

from equity_analyst.types import ProviderResponse


class LLMProvider(abc.ABC):
    name: str

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
    ) -> ProviderResponse:
        raise NotImplementedError
