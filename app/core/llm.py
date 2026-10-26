"""The Anthropic client, created once at startup. None when ANTHROPIC_API_KEY is unset."""

from anthropic import AsyncAnthropic

from app.core.env import Settings


def build_llm_client(settings: Settings) -> AsyncAnthropic | None:
    if settings.ANTHROPIC_API_KEY is None:
        return None
    return AsyncAnthropic(
        api_key=settings.ANTHROPIC_API_KEY.get_secret_value(),
        timeout=settings.LLM_TIMEOUT_SECONDS,
        max_retries=2,
    )
