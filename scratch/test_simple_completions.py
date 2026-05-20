import requests
import json
import sys

def test_simple_query():
    url = "http://127.0.0.1:5443/v1/chat/completions"
    payload = {
        "model": "qwen2.5-coder:7b",
        "messages": [
            {
                "role": "user",
                "content": "Explain what GPU we are currently using for our homelab fleet."
            }
        ]
    }
    
    print(f"Sending request to agentic proxy gateway at: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        print(f"\nResponse Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n--- RESPONSE FROM PROXY ---")
            print(json.dumps(result, indent=2))
            
            # Extract content
            choices = result.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                print("\n--- ASSISTANT CONTENT ---")
                print(content)
        else:
            print(f"Error: {response.text}", file=sys.stderr)
            
    except Exception as e:
        print(f"Connection failed: {e}", file=sys.stderr)

if __name__ == "__main__":
    test_simple_query()
