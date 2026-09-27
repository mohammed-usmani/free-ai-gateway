"""Cloudflare Workers AI text-to-image as a LiteLLM custom provider (`cloudflare_images`).

LiteLLM has no Cloudflare image handler; Workers AI's `/ai/run/<model>` returns the image as
base64 JSON. The account id comes from the step's `account_id` or CLOUDFLARE_ACCOUNT_ID."""

import os
import time

import httpx
import litellm
from litellm import CustomLLM
from litellm.types.utils import ImageObject, ImageResponse

DEFAULT_BASE = "https://api.cloudflare.com"


class CloudflareImages(CustomLLM):
    async def aimage_generation(self, model, prompt, model_response, api_key, api_base, optional_params,
                                logging_obj, timeout=None, client=None) -> ImageResponse:
        params = dict(optional_params or {})
        account = params.pop("account_id", None) or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        if not account:
            raise litellm.BadRequestError("cloudflare_images needs account_id or CLOUDFLARE_ACCOUNT_ID",
                                          model=model, llm_provider="cloudflare_images")
        model = model.split("/", 1)[1] if model.startswith("cloudflare/") else model
        url = f"{(api_base or DEFAULT_BASE).rstrip('/')}/client/v4/accounts/{account}/ai/run/{model}"
        body = {"prompt": prompt, **{k: v for k, v in params.items() if k in ("steps", "seed", "width", "height")}}
        async with httpx.AsyncClient(timeout=timeout or 120) as http:
            resp = await http.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=body)
        if resp.status_code >= 400:
            error = {429: litellm.RateLimitError, 401: litellm.AuthenticationError}.get(resp.status_code)
            if error:
                raise error(resp.text[:500], llm_provider="cloudflare_images", model=model)
            raise litellm.APIError(resp.status_code, resp.text[:500], llm_provider="cloudflare_images", model=model)
        image = resp.json()["result"]["image"]
        return ImageResponse(created=int(time.time()), data=[ImageObject(b64_json=image)])


def register():
    entry = {"provider": "cloudflare_images", "custom_handler": CloudflareImages()}
    if not any(p.get("provider") == "cloudflare_images" for p in litellm.custom_provider_map):
        litellm.custom_provider_map.append(entry)
        litellm._custom_providers.append("cloudflare_images")
        from litellm.utils import custom_llm_setup

        custom_llm_setup()
