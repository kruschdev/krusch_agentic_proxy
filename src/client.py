import httpx
from typing import Dict, Any

async def chat(system_prompt: str, user_message: str, config: Dict[str, Any]) -> str:
    """
    Standard OpenAI-compatible async chat client using httpx.
    """
    api_url = config.get('api_url', 'http://localhost:11434/v1/chat/completions')
    model = config.get('model', 'qwen2.5-coder:7b')
    temperature = config.get('temperature', 0.1)
    max_tokens = config.get('max_tokens', 2048)
    
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
