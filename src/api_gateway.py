"""
Krusch Agentic Proxy — OpenAI-Compatible API Gateway.

Serves as a FastAPI REST endpoint on port 5440, routing requests through
the Dual-Engine pipeline, waterfall cloud routing, or fast-path NLP.
"""

import os
import sys
import json
import uuid
import time
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

# Ensure src is in the python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core import KruschEngine
from src.router import WaterfallRouter
from src.tools import INTERNAL_TOOLS, execute_internal_tool
from src.models import ChatCompletionRequest

logger = logging.getLogger(__name__)

# Max wall-clock time for the autonomous agent loop (seconds)
AGENT_TIMEOUT = float(os.environ.get("KRUSCH_AGENT_TIMEOUT", "120"))


# --- Configuration ---
def load_config():
    try:
        config_path = os.environ.get(
            "KRUSCH_PROXY_CONFIG",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
        )
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {"llm": {"model": "qwen2.5-coder:7b", "api_url": "http://127.0.0.1:11434/v1/chat/completions"}}


# --- Application Lifecycle ---
# Engine and router are initialized at startup, not import time (H2)
_engine: KruschEngine = None
_router: WaterfallRouter = None
_config: dict = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize engine and router on startup, clean up on shutdown."""
    global _engine, _router, _config
    _config = load_config()
    _engine = KruschEngine(_config)
    _router = WaterfallRouter(_config.get("waterfall_routes", []))
    logger.info("[Gateway] Engine and WaterfallRouter initialized.")
    yield
    logger.info("[Gateway] Shutting down.")


app = FastAPI(title="Krusch Agentic Proxy (OpenAI API)", lifespan=lifespan)


# --- Helper ---
def _build_response(model: str, content: str, finish_reason: str = "stop",
                    tool_calls=None) -> dict:
    """Build a standard OpenAI-compatible response dict."""
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["content"] = None
        msg["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
        # Token counts omitted — accurate counting requires a tokenizer (M1)
    }


# --- Endpoints ---

@app.get("/health")
async def health():
    """Health check endpoint for monitoring and benchmark scripts."""
    return {"status": "ok", "engine": "krusch-agentic-proxy"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # Validate request body via Pydantic (C4)
    try:
        raw_data = await request.json()
        validated = ChatCompletionRequest(**raw_data)
    except Exception as e:
        return JSONResponse({"error": f"Invalid request: {str(e)}"}, status_code=400)

    data = raw_data  # Keep raw dict for waterfall passthrough
    messages = [msg.model_dump(exclude_none=True) for msg in validated.messages]
    model = validated.model or _config.get("llm", {}).get("model", "qwen2.5-coder:7b")

    # --- WATERFALL AUTO-CLOUD ROUTING ---
    if model == "auto-cloud" or os.environ.get("USE_AUTO_CLOUD") == "1":
        logger.info("[Gateway] 'auto-cloud' detected. Intercepting for Waterfall Auto-Routing.")
        is_stream = validated.stream or False
        return await _router.route_proxy(data, is_stream=is_stream)

    # Extract the last user message
    user_prompt = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_prompt = msg.get("content", "")
            break

    if not user_prompt:
        user_prompt = "No user prompt provided."

    logger.info(f"[Gateway] Request for model {model}. Prompt length: {len(user_prompt)}")

    provided_tools = validated.tools or []
    is_passive_mode = os.environ.get("PASSIVE_MODE") == "1"
    force_autonomous = validated.force_autonomous or False
    force_dual_engine = os.environ.get("FORCE_DUAL_ENGINE") == "1"

    # --- SMART ROUTING GATE ---
    model_lower = model.lower()
    is_small_model = any(tag in model_lower for tag in ["0.5b", "1.5b", "2b", "3b"])

    if is_small_model and (provided_tools or force_dual_engine) and not is_passive_mode:
        logger.warning(
            f"⚠️ [ROUTING GATE]: {model} detected as Small Model (≤ 3B). "
            f"Dual-Engine will likely degrade performance due to Cognitive Penalty."
        )

    # --- FAST-PATH: Standard NLP ---
    if not provided_tools and not is_passive_mode and not force_autonomous and not force_dual_engine:
        logger.info("[Gateway] Fast-path routing: Standard NLP request.")

        chat_context = ""
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            chat_context += f"{role}: {content}\n"

        _, response_text = await _engine.generate(
            prompt=chat_context,
            is_code_exec=False,
            is_tool_call=False,
            target_model=model,
            max_tokens=validated.max_tokens or 2048,
            temperature=0.6
        )

        return JSONResponse(_build_response(model, response_text))

    # --- FORCED DUAL-ENGINE NLP (Benchmarking) ---
    if force_dual_engine and not provided_tools:
        logger.info("[Gateway] Forced Dual-Engine NLP routing active.")
        chat_context = ""
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            chat_context += f"{role}: {content}\n"

        blueprint, response_text = await _engine.generate(
            prompt=chat_context,
            is_code_exec=False,
            is_tool_call=False,
            target_model=model,
            max_tokens=validated.max_tokens or 2048,
            temperature=0.1
        )

        combined = response_text
        if blueprint:
            combined = f"<holodata>\n{blueprint}\n</holodata>\n\n{response_text}"

        return JSONResponse(_build_response(model, combined))

    # --- TOOL-CALLING MODE ---
    if provided_tools or is_passive_mode:
        logger.info("[Gateway] Tool-calling Mode Active. Using Dual-Engine pipeline.")
        chat_history = (
            f"OBJECTIVE: {user_prompt}\nAVAILABLE TOOLS:\n{json.dumps(provided_tools, indent=2)}\n"
            if provided_tools
            else f"OBJECTIVE: {user_prompt}\n"
        )

        blueprint, response_text = await _engine.generate(
            prompt=chat_history,
            is_code_exec=False,
            is_tool_call=True,
            exact_signature="",
            target_model=model,
            max_tokens=validated.max_tokens or 2048,
            temperature=0.1
        )

        # Parse tool calls from response
        tool_calls = []
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, list):
                tool_calls = parsed
            elif isinstance(parsed, dict) and "name" in parsed:
                tool_calls = [parsed]
        except json.JSONDecodeError:
            logger.warning(f"[Gateway] Failed to parse tool-call JSON: {response_text[:200]}")

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

        if not provided_tools:
            return JSONResponse(_build_response(model, response_text))

        if openai_tool_calls:
            return JSONResponse(_build_response(model, None, finish_reason="tool_calls",
                                                tool_calls=openai_tool_calls))
        else:
            return JSONResponse(_build_response(model, response_text))

    # --- AUTONOMOUS AGENT LOOP ---
    async def _autonomous_loop():
        chat_history = f"You are the Krusch Autonomous Proxy. Achieve the objective.\nOBJECTIVE: {user_prompt}\n"
        max_loops = 5
        final_answer = "Error: Agent exceeded max iterations."

        for iteration in range(max_loops):
            logger.info(f"  -> Iteration {iteration+1}...")
            blueprint, response_text = await _engine.generate(
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
            except json.JSONDecodeError:
                logger.warning(f"[Gateway] Autonomous loop parse failure: {response_text[:200]}")

            if tool_calls:
                for tool in tool_calls:
                    if not isinstance(tool, dict):
                        chat_history += f"\nSYSTEM: Invalid tool call format. Expected JSON object, got: {type(tool)}\n"
                        break

                    tool_name = tool.get("name")
                    tool_args = tool.get("arguments", tool.get("parameters", {}))

                    if tool_name == "final_answer":
                        return tool_args.get("answer", "Objective complete.")

                    tool_result = execute_internal_tool(tool_name, tool_args)
                    chat_history += f"\nASSISTANT CALLED TOOL: {tool_name}\nWITH ARGS: {json.dumps(tool_args)}\nRESULT:\n{tool_result}\n"

                if final_answer != "Error: Agent exceeded max iterations.":
                    break
            else:
                if response_text and "final_answer" not in response_text:
                    return response_text
                else:
                    chat_history += "\nSYSTEM: You must output a valid JSON array of tool calls.\n"

        return final_answer

    try:
        final_answer = await asyncio.wait_for(_autonomous_loop(), timeout=AGENT_TIMEOUT)
    except asyncio.TimeoutError:
        final_answer = f"Error: Autonomous agent timed out after {AGENT_TIMEOUT}s."

    return JSONResponse(_build_response(model, final_answer))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5440))
    print(f"[*] Starting Krusch Agentic Universal API Gateway on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
