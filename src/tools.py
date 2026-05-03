"""
Krusch Agentic Proxy — Internal Tool Definitions & Sandboxed Executor.

This module centralizes all internal tools available to the autonomous agent loop.
It is imported by both mcp_server.py and api_gateway.py to avoid circular imports
and ensure consistent tool behavior.

Security: All shell/file operations are sandboxed behind allowlists and path restrictions.
"""

import os
import re
import shlex
import subprocess
import logging
from typing import Dict, Any, List, Optional

import httpx

logger = logging.getLogger(__name__)

# --- SECURITY: Command Allowlist ---
# Only these command prefixes are permitted for shell execution.
# Extend this list carefully — every addition is a new attack surface.
ALLOWED_COMMAND_PREFIXES = [
    "ls", "cat", "head", "tail", "wc", "grep", "find", "echo",
    "pwd", "whoami", "date", "uname", "env",
    "python", "python3", "pip", "node",
    "git status", "git log", "git diff", "git show", "git branch",
    "docker ps", "docker logs", "docker inspect",
    "curl",  # Permit outbound HTTP for API testing
]

# --- SECURITY: Filesystem Read Restrictions ---
# Only paths under these roots can be read by the read_file tool.
DEFAULT_ALLOWED_ROOTS = [
    os.path.expanduser("~"),
]


def _get_allowed_roots() -> List[str]:
    """Return the list of allowed filesystem roots, configurable via env var."""
    env_roots = os.environ.get("KRUSCH_ALLOWED_ROOTS")
    if env_roots:
        return [os.path.expanduser(r.strip()) for r in env_roots.split(":")]
    return DEFAULT_ALLOWED_ROOTS


def _is_command_allowed(command: str) -> bool:
    """Check if a command matches the allowlist prefixes."""
    cmd_stripped = command.strip()
    for prefix in ALLOWED_COMMAND_PREFIXES:
        if cmd_stripped.startswith(prefix):
            return True
    return False


def _is_path_allowed(path: str) -> bool:
    """Check if a file path is within the allowed filesystem roots."""
    resolved = os.path.realpath(os.path.expanduser(path))
    for root in _get_allowed_roots():
        resolved_root = os.path.realpath(root)
        if resolved.startswith(resolved_root + os.sep) or resolved == resolved_root:
            return True
    return False


# --- Tool Schema Definitions (OpenAI function calling format) ---
INTERNAL_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_bash_command",
            "description": "Executes an allowlisted shell command on the host and returns stdout/stderr. Commands are restricted to a safety allowlist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to run (must match the allowlist)"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads the contents of a file on the local filesystem. Paths are restricted to allowed roots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file (must be within allowed roots)"}
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


def execute_internal_tool(tool_name: str, parameters: Dict[str, Any]) -> str:
    """
    Execute an internal tool by name with the given parameters.
    All operations are sandboxed behind security checks.
    """
    if tool_name == "run_bash_command":
        cmd = parameters.get("command", "")

        if not _is_command_allowed(cmd):
            logger.warning(f"[Security] Blocked disallowed command: {cmd}")
            return f"BLOCKED: Command not in allowlist. Allowed prefixes: {', '.join(ALLOWED_COMMAND_PREFIXES)}"

        # TAC Bridge mode (remote execution via HTTP)
        if os.getenv("TAC_BRIDGE_MODE") == "1":
            try:
                response = httpx.post(
                    "http://127.0.0.1:11441/tac/run",
                    json={"command": cmd},
                    timeout=120
                )
                data = response.json()
                if "error" in data and not data.get("output"):
                    return f"Execution Failed:\n{data['error']}"
                return f"STDOUT:\n{data.get('output', '')}\nEXIT_CODE: {data.get('exit_code', '')}"
            except Exception as e:
                return f"TAC Bridge Execution Failed: {str(e)}"

        # Local sandboxed execution
        try:
            result = subprocess.run(
                shlex.split(cmd),
                shell=False,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.path.expanduser("~")
            )
            return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        except Exception as e:
            return f"Execution Failed: {str(e)}"

    elif tool_name == "read_file":
        path = parameters.get("path", "")

        if not _is_path_allowed(path):
            logger.warning(f"[Security] Blocked read of restricted path: {path}")
            return f"BLOCKED: Path is outside allowed roots. Allowed roots: {', '.join(_get_allowed_roots())}"

        try:
            resolved = os.path.realpath(os.path.expanduser(path))
            with open(resolved, "r") as f:
                content = f.read()
            # Cap output size to prevent context window flooding
            if len(content) > 50_000:
                return content[:50_000] + f"\n\n[TRUNCATED — file is {len(content)} bytes, showing first 50,000]"
            return content
        except Exception as e:
            return f"Failed to read file: {str(e)}"

    elif tool_name == "final_answer":
        # This is handled by the caller — just return the answer string
        return parameters.get("answer", "Objective complete.")

    logger.warning(f"[Tools] Unknown tool called: {tool_name}")
    return f"Unknown tool: {tool_name}"
