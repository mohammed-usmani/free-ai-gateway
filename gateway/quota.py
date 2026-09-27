"""Tell a daily quota exhaustion apart from a per-minute rate limit, and say when it resets.

Daily limits cool a key down until the reset instead of retrying it every minute all day.
Signals (researched 2026-09-27): Gemini quotaId "...PerDay..." (resets midnight Pacific);
Groq "per day (RPD|TPD)" + x-ratelimit-reset-requests; OpenRouter "free-models-per-day" +
X-RateLimit-Reset (ms); Cloudflare error 4006 (00:00 UTC); SambaNova *-requests-day headers.
"""

import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")
_DURATION = re.compile(r"(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?$")


def _headers(exc) -> dict:
    raw = getattr(exc, "litellm_response_headers", None) or getattr(getattr(exc, "response", None), "headers", None) or {}
    return {k.lower(): v for k, v in dict(raw).items()}


def _duration(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    m = _DURATION.match(str(value).strip())
    if not m or not any(m.groups()):
        return None
    h, mi, s = m.groups()
    return int(h or 0) * 3600 + int(mi or 0) * 60 + float(s or 0)


def _until_midnight(now: datetime, tz) -> float:
    local = now.astimezone(tz)
    nxt = (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (nxt - local).total_seconds()


def daily_reset_seconds(exc, provider: str, now: datetime | None = None) -> float | None:
    """Seconds until a daily quota resets, or None when this isn't a daily limit."""
    if getattr(exc, "status_code", None) != 429:
        return None
    now = now or datetime.now(timezone.utc)
    text, h = str(exc), _headers(exc)
    if re.search(r"PerDay", text):
        return _until_midnight(now, PACIFIC if provider == "gemini" else timezone.utc)
    if re.search(r"per day \((RPD|TPD)\)", text, re.I):
        return _duration(h.get("x-ratelimit-reset-requests") or h.get("x-ratelimit-reset-tokens")) or 3600
    if "free-models-per-day" in text:
        reset = h.get("x-ratelimit-reset")
        if reset and reset.isdigit():
            return max(0.0, int(reset) / 1000 - now.timestamp())
        return _until_midnight(now, timezone.utc)
    if re.search(r'"code"\s*:\s*4006|daily free allocation', text):
        return _until_midnight(now, timezone.utc)
    if h.get("x-ratelimit-remaining-requests-day") == "0":
        return _duration(h.get("x-ratelimit-reset-requests-day")) or _until_midnight(now, timezone.utc)
    return None
