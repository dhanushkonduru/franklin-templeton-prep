from __future__ import annotations

from groq import Groq

from sql_finance_copilot.config import AppSettings


class GroqChatClient:
    def __init__(self, settings: AppSettings):
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is required")
        self._settings = settings
        self._client = Groq(api_key=settings.groq_api_key)

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
