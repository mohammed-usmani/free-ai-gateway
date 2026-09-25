"""Provider keys from the environment and fallback chains from providers.yaml."""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_KEY_VAR = re.compile(r"^([A-Z0-9_]+)_API_KEYS?$")
_NOT_PROVIDERS = {"GATEWAY"}


def load_keys(env: Mapping[str, str]) -> dict[str, list[str]]:
    """`GROQ_API_KEYS=a,b` (or `GROQ_API_KEY=a`) → {"groq": ["a", "b"]}.

    The CSV form wins when both are set; blanks and repeats are dropped.
    """
    raw: dict[str, str] = {}
    for var, value in env.items():
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


@dataclass
class Catalog:
    aliases: dict[str, list[Step]]
    realtime: RealtimeChain


def _steps(items: list[dict]) -> list[Step]:
    return [Step(i["provider"], i["model"], {k: v for k, v in i.items() if k not in ("provider", "model")}) for i in items]


def load_catalog(path: str | Path) -> Catalog:
    data = yaml.safe_load(Path(path).read_text())
    rt = data["realtime"]
    return Catalog(
        aliases={name: _steps(steps) for name, steps in data["aliases"].items()},
        realtime=RealtimeChain(_steps(rt["s2s"]), _steps(rt["stt"]), rt["llm"], _steps(rt["tts"])),
    )


def available(steps: list[Step], keys: Mapping[str, list[str]]) -> list[Step]:
    return [s for s in steps if keys.get(s.provider)]
