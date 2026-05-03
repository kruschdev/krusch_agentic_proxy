"""
Krusch Agentic Proxy — Unified Waterfall Router.

Consolidates the three duplicated waterfall routing implementations into a single,
well-tested class with consistent error handling, timeouts, and key resolution.
"""

import os
import logging
from typing import Dict, Any, List, Optional

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)


def resolve_api_key(raw_key: str) -> str:
    """Resolve an API key, supporting ENV: prefix for environment variable lookup."""
    if raw_key.startswith("ENV:"):
        return os.environ.get(raw_key[4:], "")
    return raw_key


class WaterfallRouter:
    """
    Iterates through configured routes until one succeeds.
    Supports both streaming and non-streaming responses.
    """

    def __init__(self, routes: List[Dict[str, Any]], timeout: float = 120.0):
        self.routes = routes
        self.timeout = timeout

    async def route_chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """
        Route a simple chat completion through the waterfall and return the content string.
        Used by the internal KruschEngine client for LLM calls.
        Raises RuntimeError if all routes fail.
        """
        if not self.routes:
            raise RuntimeError("No waterfall_routes configured for auto-cloud routing.")

        last_error = ""

        for route in self.routes:
            route_name = route.get("name", "Unknown")
            api_url = route.get("api_url")
            logger.info(f"[Waterfall] Trying route: {route_name}")

            payload = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            }

            # Set model(s) from route config
            if "models" in route:
                payload["models"] = route["models"]
            elif "model" in route:
                payload["model"] = route["model"]

            api_key = resolve_api_key(route.get("api_key", ""))
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "http://localhost:5440",
                "X-Title": "Krusch Agentic Waterfall Router",
            }

            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(api_url, json=payload, headers=headers)
                    if response.status_code == 200:
                        return response.json()["choices"][0]["message"]["content"]
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    logger.warning(f"[Waterfall] Route {route_name} failed: {last_error}")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[Waterfall] Route {route_name} exception: {last_error}")

        raise RuntimeError(f"All waterfall routes failed. Last error: {last_error}")

    async def route_proxy(
        self,
        data: dict,
        is_stream: bool = False,
    ) -> JSONResponse | StreamingResponse:
        """
        Route a full OpenAI-compatible request through the waterfall.
        Returns a FastAPI response (JSON or Streaming).
        Used by the API gateway for external HTTP requests.
        """
        if not self.routes:
            return JSONResponse(
                {"error": "No waterfall_routes configured for auto-cloud routing."},
                status_code=500,
            )

        last_error = "No routes attempted."

        for route in self.routes:
            api_url = route.get("api_url")
            route_name = route.get("name", "Unknown Route")
            logger.info(f"[Waterfall] Attempting route: {route_name} ({api_url})")

            # Build payload — replace model with route's model(s)
            payload = {k: v for k, v in data.items() if k != "model"}
            if "models" in route:
                payload["models"] = route["models"]
            elif "model" in route:
                payload["model"] = route["model"]

            api_key = resolve_api_key(route.get("api_key", ""))
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5440",
                "X-Title": "Krusch Agentic Waterfall Router",
            }

            try:
                if is_stream:
                    response = await self._try_stream(api_url, payload, headers)
                    if response is not None:
                        return response
                    last_error = f"Stream connection failed for route {route_name}"
                else:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        resp = await client.post(api_url, json=payload, headers=headers)
                        if resp.status_code == 200:
                            return JSONResponse(resp.json())
                        last_error = f"HTTP {resp.status_code}: {resp.text}"
                        logger.warning(f"[Waterfall] Route {route_name} failed: {last_error}")
            except Exception as e:
                last_error = f"Connection error: {str(e)}"
                logger.warning(f"[Waterfall] Route {route_name} exception: {last_error}")

        return JSONResponse(
            {"error": f"All waterfall routes failed. Last error: {last_error}"},
            status_code=502,
        )

    async def _try_stream(
        self,
        api_url: str,
        payload: dict,
        headers: dict,
    ) -> Optional[StreamingResponse]:
        """
        Attempt a streaming connection. Returns a StreamingResponse on success,
        or None on failure. Ensures httpx client is properly cleaned up.
        """
        client = httpx.AsyncClient(timeout=self.timeout)
        try:
            req = client.build_request("POST", api_url, json=payload, headers=headers)
            resp = await client.send(req, stream=True)

            if resp.status_code != 200:
                error_body = await resp.aread()
                logger.warning(
                    f"[Waterfall] Stream route failed: HTTP {resp.status_code}: "
                    f"{error_body.decode('utf-8', errors='ignore')}"
                )
                await resp.aclose()
                await client.aclose()
                return None

            async def stream_generator():
                try:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                finally:
                    await resp.aclose()
                    await client.aclose()

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        except Exception as e:
            logger.warning(f"[Waterfall] Stream connection exception: {e}")
            await client.aclose()
            return None
