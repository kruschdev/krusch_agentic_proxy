#!/bin/bash
# start_proxies.sh
# Boots both the NVIDIA and AMD instances of the Krusch Agentic Proxy from the unified codebase.
# PID files are written for clean shutdown via stop_proxies.sh.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_DIR="$SCRIPT_DIR/.pids"
mkdir -p "$PID_DIR"

echo "[*] Starting NVIDIA Agentic Proxy (Port 5440) -> TabbyAPI"
PORT=5440 nohup uv run python src/api_gateway.py > nvidia_proxy.log 2>&1 &
echo $! > "$PID_DIR/nvidia.pid"
echo "    PID: $(cat "$PID_DIR/nvidia.pid")"

echo "[*] Starting AMD Agentic Proxy (Port 5442) -> llama.cpp/Vulkan"
PORT=5442 KRUSCH_PROXY_CONFIG=config.amd.json nohup uv run python src/api_gateway.py > amd_proxy.log 2>&1 &
echo $! > "$PID_DIR/amd.pid"
echo "    PID: $(cat "$PID_DIR/amd.pid")"

echo "[*] Both proxies launched in the background!"
echo "    - NVIDIA Log: nvidia_proxy.log"
echo "    - AMD Log:    amd_proxy.log"
echo "[*] Use './stop_proxies.sh' to stop both."
