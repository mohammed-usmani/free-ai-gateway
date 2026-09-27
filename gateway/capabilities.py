"""What a request needs (vision, tools, JSON schema, prompt size) and whether a step can do it.

A step is skipped only when we *know* it can't: a `supports:` override in the yaml, or
LiteLLM's model info. Unknown → the step is tried.
"""

import json
from dataclasses import dataclass

import litellm

from gateway.config import Step

GATEWAY_STEP_OPTIONS = {"supports", "first_byte_timeout"}  # ours, never sent to a provider


@dataclass(frozen=True)
class Needs:
    vision: bool = False
    tools: bool = False
    json_schema: bool = False
    prompt_tokens: int = 0

    def describe(self) -> str:
        parts = [n for n in ("vision", "tools", "json_schema") if getattr(self, n)]
        return ", ".join(parts + ([f"{self.prompt_tokens} prompt tokens"] if self.prompt_tokens else []))


_IMAGE_TYPES = {"image_url", "input_image", "image"}


def _has_image(items) -> bool:
    for item in items or []:
        content = item.get("content") if isinstance(item, dict) else None
        if isinstance(item, dict) and item.get("type") in _IMAGE_TYPES:
            return True
        if isinstance(content, list) and any(isinstance(p, dict) and p.get("type") in _IMAGE_TYPES for p in content):
            return True
    return False


def _tokens(messages) -> int:
    try:
        return litellm.token_counter(model="gpt-4o", messages=messages)
    except Exception:  # noqa: BLE001 — rough size is enough
        return len(json.dumps(messages, default=str)) // 4


def needs_for(method: str, kwargs: dict) -> Needs:
    if method == "acompletion":
        messages = kwargs.get("messages") or []
        fmt = kwargs.get("response_format") or {}
        return Needs(vision=_has_image(messages), tools=bool(kwargs.get("tools") or kwargs.get("functions")),
                     json_schema=isinstance(fmt, dict) and fmt.get("type") == "json_schema",
                     prompt_tokens=_tokens(messages))
    if method == "aresponses":
        items = kwargs.get("input")
        items = items if isinstance(items, list) else [{"role": "user", "content": str(items or "")}]
        fmt = ((kwargs.get("text") or {}).get("format") or {})
        return Needs(vision=_has_image(items), tools=bool(kwargs.get("tools")),
                     json_schema=fmt.get("type") == "json_schema",
                     prompt_tokens=len(json.dumps(items, default=str)) // 4)
    if method == "aanthropic_messages":
        messages = kwargs.get("messages") or []
        return Needs(vision=_has_image(messages), tools=bool(kwargs.get("tools")),
                     prompt_tokens=len(json.dumps(messages, default=str)) // 4)
    return Needs()


def _info(step: Step) -> dict:
    try:
        return litellm.get_model_info(f"{step.provider}/{step.model}")
    except Exception:  # noqa: BLE001 — unknown model: no info
        return {}


def step_can(step: Step, needs: Needs) -> tuple[bool, str]:
    if needs == Needs():
        return True, ""
    override = step.options.get("supports") or {}
    info = None

    def known(flag: str, info_key: str):
        nonlocal info
        if flag in override:
            return override[flag]
        info = _info(step) if info is None else info
        return info.get(info_key)

    if needs.vision and known("vision", "supports_vision") is False:
        return False, "no vision"
    if needs.tools and known("tools", "supports_function_calling") is False:
        return False, "no tools"
    if needs.json_schema and known("json_schema", "supports_response_schema") is False:
        return False, "no json_schema"
    limit = known("max_input_tokens", "max_input_tokens")
    if needs.prompt_tokens and isinstance(limit, int) and limit < needs.prompt_tokens:
        return False, f"context {limit} < {needs.prompt_tokens}"
    return True, ""
