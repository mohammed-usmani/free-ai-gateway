"""Provider keys from the environment and fallback chains from providers.yaml."""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

_KEY_VAR = re.compile(r"^([A-Z0-9_]+)_API_KEYS?$")
_NOT_PROVIDERS = {"GATEWAY"}


def load_keys(env: Mapping[str, str], prefix: str = "") -> dict[str, list[str]]:
    """`GROQ_API_KEYS=a,b` (or `GROQ_API_KEY=a`) → {"groq": ["a", "b"]}.

    The CSV form wins when both are set; blanks and repeats are dropped. With
    prefix="PAID_" only `PAID_*` variables are read; without, they are ignored.
    """
    raw: dict[str, str] = {}
    for var, value in env.items():
        if prefix:
            if not var.startswith(prefix):
                continue
            var = var[len(prefix):]
        elif var.startswith("PAID_"):
            continue
        m = _KEY_VAR.match(var)
        if not m or m.group(1) in _NOT_PROVIDERS:
            continue
        provider = m.group(1).lower()
        if var.endswith("_KEYS") or provider not in raw:
            raw[provider] = value
    keys = {p: list(dict.fromkeys(k.strip() for k in v.split(",") if k.strip())) for p, v in raw.items()}
    return {p: k for p, k in keys.items() if k}


@dataclass(frozen=True)
class Step:
    provider: str
    model: str
    options: dict = field(default_factory=dict, hash=False, compare=False)


@dataclass
class RealtimeChain:
    s2s: list[Step]
    stt: list[Step]
    llm_alias: str
    tts: list[Step]
    fallback_to: str | None = None


@dataclass
class ModeSettings:
    cooldown_seconds: float = 60
    num_retries: int | None = None  # None: one retry per extra key
    first_byte_timeout: float = 20


@dataclass
class Catalog:
    aliases: dict[str, list[Step]]
    realtime: RealtimeChain
    settings: ModeSettings = field(default_factory=ModeSettings)
    fallback_to: dict[str, str] = field(default_factory=dict)
    prices: dict[str, list[dict]] = field(default_factory=dict)


def _steps(items: list[dict]) -> list[Step]:
    return [Step(i["provider"], i["model"], {k: v for k, v in i.items() if k not in ("provider", "model")}) for i in items]


def _check_prices(prices: dict) -> dict:
    for key, entries in prices.items():
        for entry in entries:
            if "effective_from" not in entry:
                raise ValueError(f"prices.{key}: every entry needs effective_from")
            date.fromisoformat(str(entry["effective_from"]))
            for field_name, value in entry.items():
                if field_name != "effective_from" and not isinstance(value, (int, float)):
                    raise ValueError(f"prices.{key}.{field_name} must be a number, got {value!r}")
    return prices


def load_catalog(path: str | Path) -> Catalog:
    data = yaml.safe_load(Path(path).read_text())
    rt = data["realtime"]
    aliases, fallback_to = {}, {}
    for name, entry in data["aliases"].items():
        if isinstance(entry, dict):
            aliases[name] = _steps(entry["steps"])
            if entry.get("fallback_to"):
                fallback_to[name] = entry["fallback_to"]
        else:
            aliases[name] = _steps(entry)
    return Catalog(
        aliases=aliases,
        realtime=RealtimeChain(_steps(rt["s2s"]), _steps(rt["stt"]), rt["llm"], _steps(rt["tts"]),
                               rt.get("fallback_to")),
        settings=ModeSettings(**(data.get("settings") or {})),
        fallback_to=fallback_to,
        prices=_check_prices(data.get("prices") or {}),
    )


def check_fallbacks(catalogs: Mapping[str, Catalog]) -> None:
    """Every `fallback_to` must name an existing `<mode>/<alias>`, and they must not loop."""
    edges: dict[str, str] = {}
    for mode, catalog in catalogs.items():
        for alias, target in catalog.fallback_to.items():
            edges[f"{mode}/{alias}"] = target
        if catalog.realtime.fallback_to:
            edges[f"{mode}/realtime"] = catalog.realtime.fallback_to
    for source, target in edges.items():
        t_mode, _, t_alias = target.partition("/")
        known = catalogs.get(t_mode)
        if not known or (t_alias not in known.aliases and t_alias != "realtime"):
            raise ValueError(f"{source} falls back to unknown {target}")
        seen, node = {source}, target
        while node in edges:
            if node in seen:
                raise ValueError(f"fallback cycle through {node}")
            seen.add(node)
            node = edges[node]


def known_providers(keys: Mapping[str, list[str]], catalogs) -> dict[str, list[str]]:
    """Drop env vars that merely look like provider keys (LANGSMITH_API_KEY…): keep providers a
    catalog uses or LiteLLM can call (for `provider/model` passthrough)."""
    import litellm

    used = set()
    for c in catalogs:
        used |= {s.provider for steps in c.aliases.values() for s in steps}
        used |= {s.provider for s in c.realtime.s2s + c.realtime.stt + c.realtime.tts}
    callable_ = {str(getattr(p, "value", p)) for p in litellm.provider_list}
    return {p: k for p, k in keys.items() if p in used or p in callable_}


def available(steps: list[Step], keys: Mapping[str, list[str]]) -> list[Step]:
    return [s for s in steps if keys.get(s.provider)]
