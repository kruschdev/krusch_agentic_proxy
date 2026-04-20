import os
import subprocess
import json
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROXY_DIR = os.path.dirname(SCRIPT_DIR)
LOG_FILE = os.path.join(PROXY_DIR, "ifeval_distributed.log")

NODE_CONFIGS = {
    "RTX_3060": {
        "url": "http://127.0.0.1:11434/v1/chat/completions",
        "models": ["llama3.1:8b", "qwen2.5-coder:7b"]
    },
    "RTX_3050": {
        "url": "http://127.0.0.1:11434/v1/chat/completions",
        "models": ["qwen2.5-coder:0.5b", "llama3.2:1b"]
    },
    "AMD_VULKAN": {
        "url": "http://127.0.0.1:11435/v1/chat/completions",
        "models": ["qwen2.5-coder:3b", "llama3.2:3b"]
    }
}

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def run_lm_eval(model_name, api_url, output_dir):
    cmd = [
        "uv", "run", "lm_eval",
        "--model", "local-chat-completions",
        "--tasks", "ifeval",
        "--model_args", f"base_url={api_url},model={model_name}",
        "--apply_chat_template", "true",
        "--output_path", output_dir
    ]
    
    log(f"Running: {' '.join(cmd)}")
    start = time.time()
    
    # We pipe the output to the log file
    with open(LOG_FILE, "a") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=os.path.dirname(PROXY_DIR) + "/mlx_transformers_benchmark")
        process.wait()
        
    log(f"Finished in {time.time() - start:.1f}s")
    
def start_proxy(backend_url, port=5440):
    # Temporarily write config to point to the correct backend node
    config_path = os.path.join(PROXY_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump({"llm": {"api_url": backend_url, "temperature": 0.1}}, f)
        
    env = os.environ.copy()
    env["FORCE_DUAL_ENGINE"] = "1"
    env["PORT"] = str(port)
    
    cmd = ["python3", "src/api_gateway.py"]
    log(f"Starting Proxy on port {port} pointing to {backend_url}...")
    proxy_proc = subprocess.Popen(cmd, env=env, cwd=PROXY_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3) # Wait for proxy to start
    return proxy_proc

def main():
    log("=== Starting Distributed IFEval ===")
    
    for node, config in NODE_CONFIGS.items():
        backend_url = config["url"]
        models = config["models"]
        
        for model in models:
            log(f"\n==============================================")
            log(f"🏃 [{node}] IFEval for: {model}")
            log(f"==============================================")
            
            # 1. Unassisted
            log(f"\n--- 1. NAKED / UNASSISTED ({model}) ---")
            output_dir = os.path.join(PROXY_DIR, "benchmarks", "results", f"ifeval_naked_{model.replace(':', '_')}")
            run_lm_eval(model, backend_url, output_dir)
            
            # 2. Dual Engine
            log(f"\n--- 2. DUAL ENGINE ({model}) ---")
            # Start proxy pointing to this node
            proxy_proc = start_proxy(backend_url, port=5440)
            try:
                output_dir = os.path.join(PROXY_DIR, "benchmarks", "results", f"ifeval_dual_{model.replace(':', '_')}")
                run_lm_eval(model, "http://127.0.0.1:5440/v1/chat/completions", output_dir)
            finally:
                proxy_proc.terminate()
                proxy_proc.wait()
                
    log("=== All IFEval Benchmarks Complete ===")

if __name__ == "__main__":
    main()
