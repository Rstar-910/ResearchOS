"""Model access through Ollama's chat API (Ollama Cloud or a local Ollama server)."""

import json

import httpx
from pydantic import BaseModel, ValidationError

from researchos.config import Settings

MAX_OUTPUT_TOKENS = 4096


class LLMFailure(Exception):
    pass


def extract_json(content: str) -> str:
    """Models sometimes wrap JSON in prose or code fences; keep the outermost object."""
    start, end = content.find("{"), content.rfind("}")
    return content[start : end + 1] if 0 <= start < end else content


class LLM:
    def __init__(
        self,
        client: httpx.Client,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120,
    ):
        self.client, self.model, self.timeout = client, model, timeout
        self.url = base_url.rstrip("/") + "/api/chat"
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def chat(self, messages: list[dict], schema: dict | None, max_tokens: int) -> tuple[str, int]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0, "num_predict": max_tokens},
        }
        if schema:
            payload["format"] = schema
        try:
            response = self.client.post(
                self.url, headers=self.headers, timeout=self.timeout, json=payload
            )
        except httpx.HTTPError:
            raise LLMFailure("model request failed: connection or timeout error") from None
        if response.status_code in {401, 403}:
            raise LLMFailure("model access rejected; check OLLAMA_API_KEY")
        if response.status_code == 404:
            raise LLMFailure(f"model {self.model} was not found; check LLM_MODEL")
        if response.status_code != 200:
            raise LLMFailure(f"model request failed: HTTP {response.status_code}")
        try:
            data = response.json()
            content = data["message"]["content"]
            used = int(data.get("prompt_eval_count") or 0) + int(data.get("eval_count") or 0)
        except (ValueError, KeyError, TypeError):
            raise LLMFailure("model returned an unreadable response") from None
        return content, used

    def structured[T: BaseModel](
        self, system: str, user: str, schema: type[T], token_limit: int
    ) -> tuple[T | None, int, str]:
        """Return (validated object or None, tokens used, error). One repair attempt.

        Tokens are reported even on failure, because failed calls still cost budget.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        json_schema = schema.model_json_schema()
        used, error = 0, ""
        for _ in range(2):
            remaining = token_limit - used
            if remaining <= 0:
                return None, used, "token budget exhausted"
            try:
                content, spent = self.chat(messages, json_schema, min(MAX_OUTPUT_TOKENS, remaining))
            except LLMFailure as failure:
                return None, used, str(failure)
            used += spent
            try:
                return schema.model_validate(json.loads(extract_json(content))), used, ""
            except (ValueError, ValidationError) as invalid:
                problems = (
                    "; ".join(
                        f"{'.'.join(map(str, item['loc']))}: {item['msg']}"
                        for item in invalid.errors()[:5]
                    )
                    if isinstance(invalid, ValidationError)
                    else "not valid JSON"
                )
                error = f"model output did not match the required format ({problems})"
                messages += [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": f"That was rejected: {problems}. "
                        "Reply again with only the corrected JSON object.",
                    },
                ]
        return None, used, error


def configured_llm(settings: Settings, client: httpx.Client) -> LLM | None:
    key = settings.ollama_api_key.get_secret_value() if settings.ollama_api_key else None
    if not key and not settings.ollama_base_url:
        return None
    base_url = str(settings.ollama_base_url or "https://ollama.com")
    return LLM(client, base_url, settings.llm_model, key, settings.llm_timeout)
