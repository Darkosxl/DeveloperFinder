from __future__ import annotations

import json
import re

import httpx


class LlmJsonParseError(Exception):
    """Raised when the LLM response cannot be parsed as valid JSON after retries."""


class SambaNovaClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def _strip_markdown_fences(self, text: str) -> str:
        text = text.strip()
        # Remove leading ```json ... ``` or ``` ... ``` blocks
        if text.startswith("```"):
            text = re.sub(r"^```(?:\w+)?\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return text.strip()

    def _call(self, messages: list[dict], temperature: float) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
        }
        response = self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return content

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float,
        max_retries: int = 2,
    ) -> dict:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            content = self._call(messages, temperature)
            cleaned = self._strip_markdown_fences(content)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as exc:
                last_error = exc
                # Append a reminder user message for the next attempt
                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "That was not valid JSON. Respond ONLY with a single valid JSON object, no markdown, no prose, no explanation.",
                    }
                )

        raise LlmJsonParseError(
            f"Failed to parse JSON from LLM after {max_retries + 1} attempts: {last_error}"
        )

    def close(self) -> None:
        self._client.close()
