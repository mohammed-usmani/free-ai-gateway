"""Free and paid gateways behind one model namespace: `paid/<alias>`, `free/<alias>`, bare = free."""

from dataclasses import dataclass
from typing import Any

from gateway.router import Exhausted, Gateway

DEFAULT_MODE = "free"


@dataclass
class Answer:
    resp: Any
    mode: str
    group: str
    skipped: int


class Modes:
    def __init__(self, gateways: dict[str, Gateway]):
        self.gateways = gateways

    def split(self, model: str) -> tuple[str, str]:
        head, _, rest = model.partition("/")
        if head in self.gateways and rest:
            return head, rest
        return DEFAULT_MODE, model

    def is_alias(self, model: str) -> bool:
        mode, name = self.split(model)
        return name in self.gateways[mode].catalog.aliases

    async def call(self, method: str, model: str, **kwargs) -> Answer:
        """Run in the model's mode; when its chain is exhausted (or has no keys at all),
        continue at the alias's `fallback_to`. Fallbacks are checked acyclic at load."""
        mode, name = self.split(model)
        gateway = self.gateways[mode]
        try:
            resp, group, skipped = await gateway.call(method, name, **kwargs)
            return Answer(resp, mode, group, skipped)
        except (Exhausted, KeyError) as exc:
            target = gateway.catalog.fallback_to.get(name)
            if not target or name not in gateway.catalog.aliases:
                raise
            try:
                return await self.call(method, target, **kwargs)
            except KeyError:
                # The fallback has no keys either: report why this chain failed.
                if isinstance(exc, Exhausted):
                    raise exc from None
                raise

    def aliases(self) -> list[str]:
        out = []
        for mode, gateway in self.gateways.items():
            usable = set(gateway.aliases()) | set(gateway.catalog.fallback_to)
            names = [a for a in gateway.catalog.aliases if a in usable]
            out += names if mode == DEFAULT_MODE else [f"{mode}/{a}" for a in names]
        return out

    def keys(self) -> list[str]:
        return [k for g in self.gateways.values() for keys in g.keys.values() for k in keys]
