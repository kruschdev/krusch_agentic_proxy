#!/bin/bash
# stop_proxies.sh
# Gracefully stops both NVIDIA and AMD proxy instances using saved PID files.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_DIR="$SCRIPT_DIR/.pids"

stop_proxy() {
    local name="$1"
    local pidfile="$PID_DIR/$name.pid"

    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "[*] Stopping $name proxy (PID: $pid)..."
            kill -TERM "$pid"
            # Wait up to 5 seconds for graceful shutdown
            for i in $(seq 1 10); do
                if ! kill -0 "$pid" 2>/dev/null; then
                    echo "    $name stopped."
                    rm -f "$pidfile"
                    return 0
                fi
                sleep 0.5
            done
            echo "    $name didn't stop gracefully, sending SIGKILL..."
            kill -9 "$pid" 2>/dev/null
            rm -f "$pidfile"
        else
            echo "[*] $name proxy (PID: $pid) is not running."
            rm -f "$pidfile"
        fi
    else
        echo "[*] No PID file for $name proxy."
    fi
}

stop_proxy "nvidia"
stop_proxy "amd"

echo "[*] All proxies stopped."
