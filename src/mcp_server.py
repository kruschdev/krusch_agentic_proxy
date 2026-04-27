import os
import sys
import json
import subprocess
import asyncio
import httpx
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core import KruschEngine

# Initialize the MCP Server
mcp = FastMCP("KruschAgenticMCP", dependencies=["httpx", "pydantic"])

# Load Krusch Configuration
def load_config():
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json'), 'r') as f:
            return json.load(f)
    except Exception:
        return {"llm": {"model": "qwen2.5-coder:7b", "api_url": "http://127.0.0.1:11434/v1/chat/completions"}}

config = load_config()
krusch_engine = KruschEngine(config)

# Internal tools for the Krusch Agent to use during its autonomous loop
INTERNAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_bash_command",
            "description": "Executes a shell command on the host homelab and returns the stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to run"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads the contents of a file on the local filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Call this tool to return the final answer to the user once the objective is complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "The final detailed answer to the user's objective"}
                },
                "required": ["answer"]
            }
        }
    }
]

def execute_internal_tool(tool_name, parameters):
    if tool_name == "run_bash_command":
        cmd = parameters.get("command", "")
        if os.getenv("TAC_BRIDGE_MODE") == "1":
            try:
                response = httpx.post("http://127.0.0.1:11441/tac/run", json={"command": cmd}, timeout=120)
                data = response.json()
                if "error" in data and not data.get("output"):
                    return f"Execution Failed:\\n{data['error']}"
                return f"STDOUT:\\n{data.get('output', '')}\\nEXIT_CODE: {data.get('exit_code', '')}"
            except Exception as e:
                return f"TAC Bridge Execution Failed: {str(e)}"
                
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            return f"STDOUT:\\n{result.stdout}\\nSTDERR:\\n{result.stderr}"
        except Exception as e:
            return f"Execution Failed: {str(e)}"
    elif tool_name == "read_file":
        path = parameters.get("path", "")
        try:
            with open(path, "r") as f:
                return f.read()
        except Exception as e:
            return f"Failed to read file: {str(e)}"
    return f"Unknown tool: {tool_name}"

@mcp.tool()
async def spawn_autonomous_clone(objective: str) -> str:
    """
    Spawns an autonomous sub-agent powered by the Krusch Dual-Engine.
    Use this to delegate complex, multi-step coding or bash tasks that require deep reasoning.
    The clone will work independently and return the final summary when complete.
    """
    system_prompt = f"""You are the Krusch Autonomous Sub-Agent. 
You have been delegated a complex objective by a frontend agent (like OpenClaw or Hermes).
Your goal is to use your available tools to achieve the objective, and then call 'final_answer' with the result.

OBJECTIVE: {objective}

AVAILABLE TOOLS:
{json.dumps(INTERNAL_TOOLS, indent=2)}
"""
    
    chat_history = system_prompt
    max_loops = 10
    
    for iteration in range(max_loops):
        print(f"\\n--- Krusch Iteration {iteration+1} ---")
        
        target_model = config.get("llm", {}).get("model", "qwen2.5-coder:7b")
        
        # We pass the prompt to the dual-engine Thinker
        blueprint, response_text = await krusch_engine.generate(
            prompt=chat_history,
            is_code_exec=False,
            is_tool_call=True,
            exact_signature="",
            target_model=target_model,
            max_tokens=2048,
            temperature=0.1
        )
        
        # Parse the JSON tool call from the text response
        tool_call = None
        try:
            tools = json.loads(response_text)
            if isinstance(tools, list) and len(tools) > 0:
                tool_call = tools[0]
            elif isinstance(tools, dict):
                tool_call = tools
        except Exception:
            pass

        # Check if the engine called a tool
        if tool_call and "name" in tool_call:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("arguments", {})
            
            print(f"[*] Calling Tool: {tool_name}")
            
            if tool_name == "final_answer":
                return tool_args.get("answer", "Objective complete.")
                
            tool_result = execute_internal_tool(tool_name, tool_args)
            print(f"[*] Tool Result: {tool_result[:100]}...")
            
            # Append the result to the chat history for the next iteration
            chat_history += f"\\n\\nASSISTANT CALLED TOOL: {tool_name}\\nWITH ARGS: {json.dumps(tool_args)}\\nTOOL RESULT:\\n{tool_result}\\n"
        else:
            # If no tool was called but we have a text response, maybe the agent is confused.
            chat_history += f"\\n\\nASSISTANT RESPONSE: {response_text}\\nSYSTEM: You must call a tool. Use 'final_answer' to finish."
            
    return "Error: Krusch Agent exceeded maximum iterations without calling final_answer."

def main():
    mcp.run()

if __name__ == "__main__":
    main()
