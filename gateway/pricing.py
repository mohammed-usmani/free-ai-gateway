"""Prices, from the most to the least trustworthy source:

1. cost the provider reported on the response (handled where usage is recorded),
2. owner overrides in the mode's yaml (`prices:`), with effective dates,
3. LiteLLM's maintained price map, re-fetched every few hours (last good copy kept),
4. the map bundled with the pinned LiteLLM.

No price → cost is None, never a guess. Counts are kept as disjoint buckets so cost is
a plain sum and can be recomputed later.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, fields
from datetime import date, datetime, timezone
from importlib import resources

import httpx

log = logging.getLogger(__name__)

LITELLM_PRICES_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

# our unit → LiteLLM map keys (first present wins)
_LITELLM_KEYS = {
    "input": ["input_cost_per_token"],
    "output": ["output_cost_per_token"],
    "cache_read": ["cache_read_input_token_cost"],
    "cache_write": ["cache_creation_input_token_cost"],
    "cache_write_1h": ["cache_creation_input_token_cost_above_1hr"],
    "audio_input": ["input_cost_per_audio_token"],
    "audio_output": ["output_cost_per_audio_token"],
    "audio_cache_read": ["cache_read_input_audio_token_cost"],
    "per_second": ["input_cost_per_second"],
    "per_character": ["input_cost_per_character", "output_cost_per_character"],
}


@dataclass
class Counts:
    input: int = 0  # uncached text input
    cache_read: int = 0
    cache_write: int = 0  # 5-minute (default) cache writes
    cache_write_1h: int = 0  # 1-hour cache writes (Anthropic), priced higher
    output: int = 0  # text output, reasoning included
    reasoning: int = 0  # informational: already inside output
    audio_input: int = 0  # uncached audio input
    audio_output: int = 0
    audio_cache_read: int = 0
    audio_seconds: float = 0.0
    characters: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(**{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)})


# bucket → unit that prices it (reasoning is inside output, so it has none)
_BUCKET_UNIT = {"input": "input", "cache_read": "cache_read", "cache_write": "cache_write",
                "cache_write_1h": "cache_write_1h", "output": "output",
                "audio_input": "audio_input", "audio_output": "audio_output",
                "audio_cache_read": "audio_cache_read", "audio_seconds": "per_second", "characters": "per_character"}


@dataclass
class Price:
    units: dict[str, float]
    source: str


def _prompt_size(c: Counts) -> int:
    return c.input + c.cache_read + c.cache_write + c.cache_write_1h + c.audio_input + c.audio_cache_read


def _rate(units: dict[str, float], unit: str, prompt: int) -> float | None:
    """Base rate, or the highest long-context tier (`unit@<threshold>`) the prompt exceeds."""
    tiers = [(int(k.split("@")[1]), v) for k, v in units.items() if k.startswith(unit + "@")]
    above = [t for t in tiers if prompt > t[0]]
    return max(above)[1] if above else units.get(unit)


def cost(counts: Counts, price: Price | None) -> float | None:
    if price is None:
        return None
    total, prompt = 0.0, _prompt_size(counts)
    for bucket, unit in _BUCKET_UNIT.items():
        n = getattr(counts, bucket)
        if not n:
            continue
        rate = _rate(price.units, unit, prompt)
        if rate is None:
            return None  # a billed bucket we can't price: unknown, not zero
        total += n * rate
    return total


_TIER = re.compile(r"^(input_cost_per_token|output_cost_per_token|cache_read_input_token_cost|"
                   r"cache_creation_input_token_cost)_above_(\d+)k_tokens$")
_TIER_UNIT = {"input_cost_per_token": "input", "output_cost_per_token": "output",
              "cache_read_input_token_cost": "cache_read", "cache_creation_input_token_cost": "cache_write"}


def _units_from_litellm(entry: dict) -> dict[str, float]:
    units = {}
    for unit, keys in _LITELLM_KEYS.items():
        for k in keys:
            if entry.get(k) is not None:
                units[unit] = float(entry[k])
                break
    for k, v in entry.items():  # long-context tiers; _priority/_flex service tiers don't match
        if v is not None and (m := _TIER.match(k)):
            units[f"{_TIER_UNIT[m.group(1)]}@{int(m.group(2)) * 1000}"] = float(v)
    return units


def _load_bundled() -> dict:
    path = resources.files("litellm") / "model_prices_and_context_window_backup.json"
    return json.loads(path.read_text())


def fetch_litellm_prices(url: str = LITELLM_PRICES_URL) -> dict:
    resp = httpx.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict) or len(data) < 100:
        raise ValueError("price map looks wrong; keeping the previous one")
    return data


class Pricer:
    def __init__(self, overrides: dict[str, dict[str, list[dict]]], bundled: dict | None = None,
                 fetch=fetch_litellm_prices):
        self._overrides = overrides
        self._bundled = _load_bundled() if bundled is None else bundled
        self._fetch = fetch
        self._live: dict | None = None
        self._live_at: datetime | None = None
        self._last_error: str | None = None
        self._unpriced: set[str] = set()

    def price(self, provider: str, model: str, mode: str, at: datetime) -> Price | None:
        key = f"{provider}/{model}"
        entries = [e for e in self._overrides.get(mode, {}).get(key, [])
                   if _as_date(e["effective_from"]) <= at.date()]
        if entries:
            entry = max(entries, key=lambda e: _as_date(e["effective_from"]))
            return Price({k: float(v) for k, v in entry.items() if k != "effective_from"}, "override")
        for table, source in ((self._live, f"litellm-live@{self._live_at and self._live_at.isoformat(timespec='minutes')}"),
                              (self._bundled, "litellm-bundled")):
            if not table:
                continue
            entry = table.get(key)
            bare = table.get(model)
            if entry is None and bare and bare.get("litellm_provider") in (None, provider):
                entry = bare  # a bare name only counts for the provider it belongs to
            if entry and (units := _units_from_litellm(entry)):
                return Price(units, source)
        self._unpriced.add(key)
        return None

    def health(self, models, at: datetime) -> list[dict]:
        """For each (mode, provider, model): where its price comes from and what looks off —
        no price, a price identical to a different model's (a likely copied row), a live price
        that differs from the bundled one, or an override older than 90 days."""
        report = []
        for mode, provider, model in models:
            price = self.price(provider, model, mode, at)
            flags = [] if price else ["unpriced"]
            if price and price.source.startswith("litellm"):
                table = self._live if price.source.startswith("litellm-live") and self._live else self._bundled
                twin = _twin(table, provider, model, price.units)
                if twin:
                    flags.append(f"same_price_as:{twin}")
                if self._live and price.source.startswith("litellm-live"):
                    bundled = self._bundled.get(f"{provider}/{model}") or self._bundled.get(model)
                    if bundled and _units_from_litellm(bundled) != price.units:
                        flags.append("price_changed_since_bundled")
            if price and price.source == "override":
                entries = self._overrides.get(mode, {}).get(f"{provider}/{model}", [])
                newest = max(_as_date(e["effective_from"]) for e in entries)
                if (at.date() - newest).days > 90:
                    flags.append("override_older_than_90_days")
            report.append({"mode": mode, "provider": provider, "model": model, "priced": price is not None,
                           "source": price.source if price else None, "flags": flags})
        return report

    async def refresh(self):
        if not self._fetch:
            return
        try:
            self._live = await asyncio.to_thread(self._fetch)
            self._live_at = datetime.now(timezone.utc)
            self._last_error = None
        except Exception as exc:  # noqa: BLE001 — keep serving the last good map
            self._last_error = f"{type(exc).__name__}: {exc}"
            log.warning("price refresh failed, keeping the previous map: %s", self._last_error)

    async def refresh_forever(self, hours: float):
        while True:
            await self.refresh()
            await asyncio.sleep(hours * 3600)

    def status(self) -> dict:
        age = (datetime.now(timezone.utc) - self._live_at).total_seconds() / 3600 if self._live_at else None
        return {"source": "litellm-live" if self._live else "litellm-bundled",
                "fetched_at": self._live_at and self._live_at.isoformat(timespec="minutes"),
                "age_hours": age and round(age, 1), "last_error": self._last_error,
                "unpriced": sorted(self._unpriced)}


_BASIC = ("input", "output", "cache_read")


def _family(name: str) -> str:
    return re.sub(r"(-\d{4}-\d{2}-\d{2}|-\d{8}|-latest|-preview)$", "", name.split("/")[-1])


def _twin(table: dict, provider: str, model: str, units: dict) -> str | None:
    """Another model of the same provider with identical base prices — often a row copied
    from an older sibling when a new model was added before its real price was known."""
    probe = {k: units.get(k) for k in _BASIC}
    if not probe["input"]:
        return None
    for name, entry in table.items():
        if not isinstance(entry, dict) or entry.get("litellm_provider") not in (provider, None):
            continue
        if _family(name) == _family(model) or name.split("/")[-1].startswith(model) or model.startswith(name.split("/")[-1]):
            continue  # a dated snapshot or alias of the same model
        other = _units_from_litellm(entry)
        if {k: other.get(k) for k in _BASIC} == probe:
            return name.split("/")[-1] if "/" in name else name
    return None


def _as_date(value) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


# --- usage → buckets --------------------------------------------------------------------


def _int(v) -> int:
    return int(v or 0)


def counts_from_litellm(usage) -> Counts:
    """LiteLLM usage is gross: prompt_tokens includes cache reads/writes (verified for
    Anthropic in 1.102.1) and audio."""
    pd = getattr(usage, "prompt_tokens_details", None)
    cd = getattr(usage, "completion_tokens_details", None)
    cache_read = _int(getattr(pd, "cached_tokens", 0))
    cache_write = _int(getattr(usage, "cache_creation_input_tokens", None) or getattr(usage, "_cache_creation_input_tokens", 0))
    ccd = getattr(usage, "cache_creation_token_details", None) or {}
    write_1h = _int(ccd.get("ephemeral_1h_input_tokens") if isinstance(ccd, dict) else getattr(ccd, "ephemeral_1h_input_tokens", 0))
    audio_in = _int(getattr(pd, "audio_tokens", 0))
    audio_out = _int(getattr(cd, "audio_tokens", 0))
    prompt, completion = _int(usage.prompt_tokens), _int(usage.completion_tokens)
    return Counts(input=max(0, prompt - cache_read - cache_write - audio_in), cache_read=cache_read,
                  cache_write=cache_write - write_1h, cache_write_1h=write_1h, output=max(0, completion - audio_out),
                  reasoning=_int(getattr(cd, "reasoning_tokens", 0)), audio_input=audio_in, audio_output=audio_out)


def counts_from_pipecat(u, net: bool, reasoning_in_output: bool = True) -> Counts:
    """Pipecat LLMTokenUsage: `prompt_tokens` is net of cache for Anthropic/Bedrock, gross
    otherwise. Audio counts include their cached part. Gemini Live reports thinking tokens
    outside `completion_tokens` (reasoning_in_output=False); they're billed as output."""
    cr, cw = _int(u.cache_read_input_tokens), _int(u.cache_creation_input_tokens)
    gross = _int(u.prompt_tokens) + (cr + cw if net else 0)
    audio_cached = _int(u.cache_read_input_audio_tokens)
    audio_in = _int(u.input_audio_tokens) - audio_cached
    audio_out = _int(u.output_audio_tokens)
    return Counts(input=max(0, gross - cr - cw - audio_in), cache_read=cr - audio_cached, cache_write=cw,
                  output=max(0, _int(u.completion_tokens) - audio_out) + (0 if reasoning_in_output else _int(u.reasoning_tokens)),
                  reasoning=_int(u.reasoning_tokens), audio_input=audio_in, audio_output=audio_out,
                  audio_cache_read=audio_cached)
