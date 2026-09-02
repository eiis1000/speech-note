"""OpenAI-compatible chat transport.

Auth, the model fallback chain, finish_reason, and the model the response *says* it
served (for llama-server the requested name is decorative). Nothing here knows what a
transcript is.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Sequence
from typing import Any, Callable

import requests

from .config import ORGANIZER_MAX_REQUEST_TIMEOUT, ORGANIZER_MIN_REQUEST_TIMEOUT
from .terminal import debug_log


def request_timeout_seconds(
    *,
    configured_timeout: float,
    estimated_prompt_tokens: int,
    requested_output_tokens: int,
) -> float:
    """Scale the request timeout with the work being asked for."""
    prompt_seconds = estimated_prompt_tokens / 250.0
    output_seconds = requested_output_tokens / 12.0
    estimated = 15.0 + prompt_seconds + output_seconds
    return min(
        ORGANIZER_MAX_REQUEST_TIMEOUT,
        max(configured_timeout, ORGANIZER_MIN_REQUEST_TIMEOUT, estimated),
    )


@dataclasses.dataclass
class ChatResponse:
    content: str
    served_model: str | None
    finish_reason: str | None


class ChatClient:
    """OpenAI-compatible chat client with model fallback."""

    # The chain exists because models differ — in availability, in capacity, and in
    # which request parameters they support. So an HTTP error advances to the next
    # model by default, and only a *credential* failure fails fast: a rejected key is
    # the one thing another model cannot fix, and retrying it six times only delays a
    # clear error. Everything else has been observed to be model-specific, including
    # the ones that look global: 400 (a provider rejecting response_format, or
    # reasoning.effort "none" — which gpt-oss rejects and the shipped free chain
    # sends), 402 (out of credit, which is exactly when the free fallbacks matter),
    # 404 (the model left the catalog since our check), and 413 (context limits are
    # per model). Every failure is accumulated, so the raised error still names them
    # all rather than hiding the first one behind the last.
    FATAL_STATUS = {401, 403}

    def __init__(
        self,
        *,
        api_base: str,
        models: Sequence[str],
        timeout: float,
        auth_env: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.api_base = api_base
        self.configured_models = [m for m in models if m]
        self.timeout = timeout
        self.auth_env = auth_env
        self.reasoning_effort = reasoning_effort
        self.last_served_model: str | None = None
        self._catalog_checked = False
        self._available_models: list[str] | None = None

    def models_url(self) -> str:
        if self.api_base.endswith("/chat/completions"):
            return self.api_base[: -len("/chat/completions")] + "/models"
        return self.api_base.rstrip("/") + "/models"

    def request_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth_env is not None:
            api_key = os.environ.get(self.auth_env, "").strip()
            if not api_key:
                raise RuntimeError(f"{self.auth_env} is required for this organizer provider")
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def candidate_models(self) -> list[str]:
        """Configured models filtered against the live catalog when possible.

        The configured list is a preference order, not a claim of existence:
        entries missing from the provider's /models catalog are dropped (and
        the drop is logged). If the catalog cannot be fetched, fall back to the
        configured list unfiltered.
        """
        if self._catalog_checked:
            return self._available_models or self.configured_models
        self._catalog_checked = True
        try:
            response = requests.get(self.models_url(), timeout=10.0, headers=self.request_headers())
            response.raise_for_status()
            catalog = {entry.get("id") for entry in response.json().get("data", [])}
        except Exception as exc:
            debug_log(f"model catalog check failed; using configured list: {exc}")
            return self.configured_models
        available = [model for model in self.configured_models if model in catalog]
        dropped = [model for model in self.configured_models if model not in catalog]
        if dropped:
            debug_log(f"models missing from provider catalog, skipped: {dropped}")
        # An empty intersection means our list is fully stale; trying the
        # configured names anyway gives clearer errors than failing silently.
        self._available_models = available or self.configured_models
        return self._available_models

    def _worth_another_model(self, response: requests.Response) -> bool:
        """Could a different model in the chain succeed after this failure?"""
        return response.status_code not in self.FATAL_STATUS

    def _advance_or_raise(
        self,
        failures: list[str],
        *,
        index: int,
        models: list[str],
        on_model_failure: Callable[[str, str], None] | None,
        from_exc: Exception | None = None,
    ) -> None:
        """Either advance to the next model (notifying on_model_failure) or, if this
        was the last model, raise the accumulated failures. Returns normally only when
        the caller should `continue` to the next model."""
        if index + 1 >= len(models):
            if from_exc is not None:
                raise RuntimeError("; ".join(failures)) from from_exc
            raise RuntimeError("; ".join(failures))
        if on_model_failure is not None:
            on_model_failure(models[index], models[index + 1])

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        timeout: float,
        on_model_attempt: Callable[[str], None] | None = None,
        on_model_failure: Callable[[str, str], None] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        failures: list[str] = []
        models = self.candidate_models()
        for index, model in enumerate(models):
            if on_model_attempt is not None:
                on_model_attempt(model)
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": 0.0,
                # Some providers reject temperature 0 unless top_p is pinned to 1
                # (mistral: "top_p must be 1 when using greedy sampling"), and it is a
                # no-op everywhere else.
                "top_p": 1,
                "max_tokens": max_tokens,
            }
            if response_format is not None:
                # A schema here is *enforced*: llama.cpp compiles it to a GBNF grammar and
                # constrains sampling, so a reply cannot come back malformed or truncated
                # mid-string. See annotate.RESPONSE_SCHEMA.
                payload["response_format"] = response_format
                if "openrouter" in self.api_base:
                    # OpenRouter routes to providers that silently IGNORE parameters
                    # they don't support — an unenforced schema is how malformed
                    # replies happened. Only route where the schema is honored; a
                    # model with no such provider fails over to the next in the chain.
                    payload["provider"] = {"require_parameters": True}
            if self.reasoning_effort is not None:
                payload["reasoning"] = {"effort": self.reasoning_effort}
            try:
                response = requests.post(
                    self.api_base,
                    timeout=timeout,
                    headers=self.request_headers(),
                    data=json.dumps(payload),
                )
                if not response.ok:
                    body = " ".join(response.text.split())[:500]
                    failures.append(f"{model}: HTTP {response.status_code} {body}")
                    # A bad credential is the only failure another model cannot fix
                    # (see FATAL_STATUS); everything else falls through the chain.
                    if not self._worth_another_model(response):
                        raise RuntimeError("; ".join(failures))
                    self._advance_or_raise(
                        failures, index=index, models=models, on_model_failure=on_model_failure
                    )
                    continue
                # A 200 OK can still carry a gateway/provider error and no choices:
                # OpenRouter surfaces upstream rate limits and outages this way. Treat
                # an unusable body as a failed attempt and fall through instead of
                # crashing on data["choices"][0].
                try:
                    data = response.json()
                except ValueError:
                    snippet = " ".join(response.text.split())[:300]
                    failures.append(f"{model}: HTTP 200 with non-JSON body: {snippet}")
                    self._advance_or_raise(
                        failures, index=index, models=models, on_model_failure=on_model_failure
                    )
                    continue
                choices = data.get("choices")
                if not choices:
                    detail = data.get("error", data)
                    failures.append(f"{model}: HTTP 200 but no choices ({json.dumps(detail)[:300]})")
                    self._advance_or_raise(
                        failures, index=index, models=models, on_model_failure=on_model_failure
                    )
                    continue
                choice = choices[0]
                # Record what the server says it served; for llama-server the
                # requested name is decorative, the response is authoritative.
                served = data.get("model") or model
                self.last_served_model = served
                return ChatResponse(
                    content=_content_to_text(choice.get("message", {}).get("content")),
                    served_model=served,
                    finish_reason=choice.get("finish_reason"),
                )
            except requests.RequestException as exc:
                failures.append(f"{model}: {exc}")
                self._advance_or_raise(
                    failures, index=index, models=models,
                    on_model_failure=on_model_failure, from_exc=exc,
                )
                continue
        raise RuntimeError("; ".join(failures) or "no organizer models configured")


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part).strip()
    return ""
