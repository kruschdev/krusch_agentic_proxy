"""
Krusch Agentic Proxy — LLM Chat Client.

Provides the low-level async chat function used by KruschEngine.
Supports both single-endpoint and waterfall multi-provider routing.
"""

import httpx
import os
import logging
from typing import Dict, Any

from src.router import WaterfallRouter, resolve_api_key

logger = logging.getLogger(__name__)


async def chat(system_prompt: str, user_message: str, config: Dict[str, Any]) -> str:
    """
    Standard OpenAI-compatible async chat client using httpx.
    Routes through the WaterfallRouter when model is 'auto-cloud' or USE_AUTO_CLOUD=1.
    """
    api_url = config.get('api_url', 'http://localhost:11434/v1/chat/completions')
    model = config.get('model', 'qwen2.5-coder:7b')
    temperature = config.get('temperature', 0.1)
    max_tokens = config.get('max_tokens', 2048)

    # Waterfall routing for cloud fallback
    if model == "auto-cloud" or os.environ.get("USE_AUTO_CLOUD") == "1":
        routes = config.get("waterfall_routes", [])
        router = WaterfallRouter(routes, timeout=120.0)
        return await router.route_chat(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # Standard single-endpoint execution
    api_key = config.get('api_key', 'sk-none')
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {resolve_api_key(api_key)}",
    }

    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
