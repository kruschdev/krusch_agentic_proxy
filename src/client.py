import httpx
import os
from typing import Dict, Any

async def chat(system_prompt: str, user_message: str, config: Dict[str, Any]) -> str:
    """
    Standard OpenAI-compatible async chat client using httpx.
    """
    api_url = config.get('api_url', 'http://localhost:11434/v1/chat/completions')
    model = config.get('model', 'qwen2.5-coder:7b')
    temperature = config.get('temperature', 0.1)
    max_tokens = config.get('max_tokens', 2048)
    
    # Internal Waterfall Support
    if model == "auto-cloud" or os.environ.get("USE_AUTO_CLOUD") == "1":
        routes = config.get("waterfall_routes", [])
        if not routes:
            raise RuntimeError("No waterfall_routes configured for auto-cloud routing.")
            
        last_error = ""
        for route in routes:
            route_api_url = route.get("api_url")
            route_name = route.get("name", "Unknown")
            print(f"[Internal Routing] Trying route: {route_name}")
            
            payload = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            }
            if "models" in route:
                payload["models"] = route["models"]
            elif "model" in route:
                payload["model"] = route["model"]
                
            raw_key = route.get("api_key", "")
            api_key = os.environ.get(raw_key[4:], "") if raw_key.startswith("ENV:") else raw_key
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "http://localhost:5440"
            }
            
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(route_api_url, json=payload, headers=headers)
                    if response.status_code == 200:
                        return response.json()['choices'][0]['message']['content']
                    last_error = f"HTTP {response.status_code}: {response.text}"
            except Exception as e:
                last_error = str(e)
                
        raise RuntimeError(f"All waterfall routes failed. Last error: {last_error}")

    # Standard Local/Single execution
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.get('api_key', 'sk-none')}"
    }
    
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    }
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
