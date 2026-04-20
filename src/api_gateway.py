import os
import sys
import json
import uuid
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

# Ensure src is in the python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core import KruschEngine

app = FastAPI(title="Krusch Agentic Proxy (OpenAI API)")

def load_config():
    try:
        config_path = os.environ.get("KRUSCH_PROXY_CONFIG", os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json'))
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {"llm": {"model": "qwen2.5-coder:7b", "api_url": "http://127.0.0.1:11434/v1/chat/completions"}}

config = load_config()
krusch_engine = KruschEngine(config)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    model = data.get("model", config.get("llm", {}).get("model", "qwen2.5-coder:7b"))
    
    # Extract the last user message
    user_prompt = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_prompt = msg.get("content", "")
            break
            
    if not user_prompt:
        user_prompt = "No user prompt provided."

    print(f"[API Gateway] Received request for model {model}. Prompt length: {len(user_prompt)}")

    provided_tools = data.get("tools", [])
    is_passive_mode = os.environ.get("PASSIVE_MODE") == "1"
    force_autonomous = data.get("force_autonomous", False)
    force_dual_engine = os.environ.get("FORCE_DUAL_ENGINE") == "1"
    
    # --- SMART ROUTING GATE ---
    # The Dual-Engine pipeline acts as a structural multiplier for capable models (>=7B).
    # However, it imposes a severe "Cognitive Penalty" on small models (<=3B), degrading 
    # their baseline logic as they hyper-focus on escaping JSON constraints.
    model_lower = model.lower()
    is_small_model = any(tag in model_lower for tag in ["0.5b", "1.5b", "2b", "3b"])
    
    if is_small_model and (provided_tools or force_dual_engine) and not is_passive_mode:
        print(f"⚠️ [ROUTING GATE WARNING]: {model} detected as a Small Model (<= 3B parameters).")
        print(f"⚠️ The Dual-Engine JSON constraint will likely degrade its performance due to the Cognitive Penalty.")
        print(f"⚠️ It is highly recommended to route this model to a standard NLP pipeline instead.")
    
    # Router: Fast-Path for Standard NLP
    # If no tools are provided and we aren't forcing the autonomous loop or dual engine, just answer normally.
    if not provided_tools and not is_passive_mode and not force_autonomous and not force_dual_engine:
        print(f"[API Gateway] Fast-path routing: Standard NLP request detected.")
        
        # Build full conversation context for normal chat
        chat_context = ""
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            chat_context += f"{role}: {content}\n"
            
        _, response_text = await krusch_engine.generate(
            prompt=chat_context,
            is_code_exec=False,
            is_tool_call=False,
            target_model=model,
            max_tokens=2048,
            temperature=0.6
        )
        
        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": len(chat_context)//4, "completion_tokens": len(response_text)//4, "total_tokens": (len(chat_context)+len(response_text))//4}
        })

    # Router: Standard NLP via Dual-Engine (For Benchmarking)
    if force_dual_engine and not provided_tools:
        print(f"[API Gateway] Forced Dual-Engine NLP routing active.")
        chat_context = ""
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            chat_context += f"{role}: {content}\n"
            
        blueprint, response_text = await krusch_engine.generate(
            prompt=chat_context,
            is_code_exec=False,
            is_tool_call=False,
            target_model=model,
            max_tokens=2048,
            temperature=0.1
        )
        
        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": len(chat_context)//4, "completion_tokens": len(response_text)//4, "total_tokens": (len(chat_context)+len(response_text))//4}
        })
    
    if provided_tools or is_passive_mode:
        print(f"[API Gateway] Tool-calling Mode Active. Using Dual-Engine pipeline.")
        chat_history = f"OBJECTIVE: {user_prompt}\nAVAILABLE TOOLS:\n{json.dumps(provided_tools, indent=2)}\n" if provided_tools else f"OBJECTIVE: {user_prompt}\n"
        blueprint, response_text = await krusch_engine.generate(
            prompt=chat_history,
            is_code_exec=False,
            is_tool_call=True,
            exact_signature="",
            target_model=model,
            max_tokens=2048,
            temperature=0.1
        )
        
        tool_calls = []
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, list):
                tool_calls = parsed
            elif isinstance(parsed, dict) and "name" in parsed:
                tool_calls = [parsed]
        except Exception:
            pass
            
        openai_tool_calls = []
        for tc in tool_calls:
            if isinstance(tc, dict) and "name" in tc:
                openai_tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc.get("arguments", tc.get("parameters", {})))
                    }
                })
        
        # If the benchmark expects a raw JSON string in the text response (Weklund style)
        # We should just return the raw response_text if openai_tool_calls is empty or not requested via 'tools'
        if not provided_tools:
            msg_obj = {
                "role": "assistant",
                "content": response_text
            }
        else:
            msg_obj = {
                "role": "assistant",
                "content": None,
                "tool_calls": openai_tool_calls
            } if openai_tool_calls else {
                "role": "assistant",
                "content": response_text
            }
        
        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": msg_obj,
                "finish_reason": "tool_calls" if openai_tool_calls else "stop"
            }],
            "usage": {"prompt_tokens": len(user_prompt)//4, "completion_tokens": len(response_text)//4, "total_tokens": (len(user_prompt)+len(response_text))//4}
        })

    # Execute through Dual-Engine Thinker (Autonomous Loop)
    # We treat standard API calls as tool_call=False for the initial pass, 
    # but the engine handles the autonomous loop internally if we wrap it properly,
    # or we can manually run the autonomous loop here like we did in mcp_server.py
    
    max_loops = 5
    chat_history = f"You are the Krusch Autonomous Proxy. Achieve the objective.\\nOBJECTIVE: {user_prompt}\\n"
    
    from src.mcp_server import execute_internal_tool
    
    final_answer = "Error: Agent exceeded max iterations."
    
    for iteration in range(max_loops):
        print(f"  -> Iteration {iteration+1}...")
        blueprint, response_text = await krusch_engine.generate(
            prompt=chat_history,
            is_code_exec=False,
            is_tool_call=True,
            exact_signature="",
            target_model=model,
            max_tokens=2048,
            temperature=0.1
        )
        
        # The tools are returned in response_text as a JSON array when is_tool_call=True
        tool_calls = []
        try:
            # Attempt to parse response_text as JSON
            parsed = json.loads(response_text)
            if isinstance(parsed, list):
                tool_calls = parsed
            elif isinstance(parsed, dict) and "name" in parsed:
                tool_calls = [parsed]
        except json.JSONDecodeError:
            pass

        if tool_calls:
            for tool in tool_calls:
                if not isinstance(tool, dict):
                    chat_history += "\\nSYSTEM: Invalid tool call format. Expected a JSON object, got: " + str(type(tool)) + ". Please output a list of valid tool call objects."
                    break
                    
                tool_name = tool.get("name")
                tool_args = tool.get("arguments", tool.get("parameters", {}))
                
                if tool_name == "final_answer":
                    final_answer = tool_args.get("answer", "Objective complete.")
                    break
                    
                tool_result = execute_internal_tool(tool_name, tool_args)
                chat_history += f"\\nASSISTANT CALLED TOOL: {tool_name}\\nWITH ARGS: {json.dumps(tool_args)}\\nRESULT:\\n{tool_result}\\n"
                
            if final_answer != "Error: Agent exceeded max iterations.":
                break
        else:
            if response_text and "final_answer" not in response_text:
                final_answer = response_text
                break
            else:
                chat_history += "\\nSYSTEM: You must output a valid JSON array of tool calls."

    # Package as OpenAI JSON
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    
    return JSONResponse({
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": final_answer
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(user_prompt) // 4,
            "completion_tokens": len(final_answer) // 4,
            "total_tokens": (len(user_prompt) + len(final_answer)) // 4
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5440))
    print(f"[*] Starting Krusch Agentic Universal API Gateway on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
