#!/bin/bash
# start_proxies.sh
# Boots the NVIDIA instance of the Krusch Agentic Proxy.
# PID files are written for clean shutdown via stop_proxies.sh.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_DIR="$SCRIPT_DIR/.pids"
mkdir -p "$PID_DIR"

echo "[*] Starting NVIDIA Agentic Proxy (Port 5443) -> TabbyAPI"
PORT=5443 nohup uv run python src/api_gateway.py > nvidia_proxy.log 2>&1 &
echo $! > "$PID_DIR/nvidia.pid"
echo "    PID: $(cat "$PID_DIR/nvidia.pid")"

echo "[*] NVIDIA proxy launched in the background!"
echo "    - NVIDIA Log: nvidia_proxy.log"
echo "[*] Use './stop_proxies.sh' to stop."

