"""Key rotation and cross-provider fallback on top of LiteLLM.

Each chain step is a LiteLLM model group with one deployment per key, so LiteLLM rotates
keys within a step (a failing key is cooled down and the retry lands on another). Walking
from step to step is ours: LiteLLM's own fallbacks also fire on 400s, which we must return.
"""

import litellm
from litellm import Router

from gateway.config import Catalog, Step, available

COOLDOWN_SECONDS = 60


def status_of(exc: Exception) -> int | None:
    return getattr(exc, "status_code", None)


def should_fall_back(exc: Exception) -> bool:
    """Provider trouble moves on; a request the provider rejects as malformed does not."""
    status = status_of(exc)
    if status is None or not isinstance(status, int):
        return True  # connection errors, timeouts
    return status in (401, 403, 404, 408, 429) or status >= 500


class Exhausted(Exception):
    """Every step failed with a fall-back-able error."""

    def __init__(self, last: Exception | None, retry_after: float | None):
        super().__init__(str(last) if last else "no provider configured")
        self.last = last
        self.retry_after = retry_after


class Gateway:
    def __init__(self, catalog: Catalog, keys: dict[str, list[str]]):
        self.catalog = catalog
        self.keys = keys
        self._chains: dict[str, list[str]] = {}
        model_list: list[dict] = []
        registered: set[str] = set()

        for alias, steps in catalog.aliases.items():
            groups = []
            for step in available(steps, keys):
                group = f"{step.provider}/{step.model}"
                groups.append(group)
                if group not in registered:
                    registered.add(group)
                    model_list += self._deployments(group, step)
            if groups:
                self._chains[alias] = groups

        self._registered = registered
        self.router = Router(
            model_list=model_list,
            cooldown_time=COOLDOWN_SECONDS,
            allowed_fails=0,
            # Retries land on the step's other keys, since the failed key is cooling down.
            num_retries=max((len(k) for k in keys.values()), default=1) - 1,
            retry_after=0,
        )

    def _deployments(self, group: str, step: Step) -> list[dict]:
        return [
            {
                "model_name": group,
                "litellm_params": {"model": f"{step.provider}/{step.model}", "api_key": key, **step.options},
            }
            for key in self.keys[step.provider]
        ]

    def chain(self, model: str) -> list[str]:
        """Alias → its usable steps; `provider/model` → just that model on the provider's keys."""
        if model in self._chains:
            return self._chains[model]
        provider, _, name = model.partition("/")
        if not name or not self.keys.get(provider):
            raise KeyError(model)
        if model not in self._registered:
            self._registered.add(model)
            for d in self._deployments(model, Step(provider, name)):
                self.router.add_deployment(litellm.types.router.Deployment(**d))
        return [model]

    async def call(self, method: str, model: str, **kwargs):
        """Run `router.<method>` down the chain. Returns (response, group used, steps skipped).

        For streams, the first chunk is fetched here, so a provider that fails before
        sending anything is still fallen back from; after that the stream is committed.
        """
        last: Exception | None = None
        for skipped, group in enumerate(self.chain(model)):
            try:
                resp = await getattr(self.router, method)(model=group, **kwargs)
                if kwargs.get("stream"):
                    resp = await _prefetch(resp)
                return resp, group, skipped
            except Exception as exc:  # noqa: BLE001 — classified below
                if not should_fall_back(exc):
                    raise
                last = exc
        raise Exhausted(last, self._retry_after(model))

    def _retry_after(self, model: str) -> float | None:
        return COOLDOWN_SECONDS if self._chains.get(model) or "/" in model else None

    def aliases(self) -> list[str]:
        return list(self._chains)


async def _prefetch(stream):
    iterator = stream.__aiter__()
    first = await iterator.__anext__()

    async def replay():
        yield first
        async for chunk in iterator:
            yield chunk

    return replay()
