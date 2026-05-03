#!/bin/bash
set -e

echo "============================================================"
echo " Krusch Agentic Proxy - Official LM Evaluation Harness Benchmark Tool"
echo "============================================================"
echo ""
echo "This script will install the EleutherAI LM Evaluation Harness"
echo "and run the core structural benchmarks (IFEval, HumanEval)"
echo "against your local Krusch Agentic Proxy."
echo ""
echo "WARNING: This suite explicitly EXCLUDES pure math and reasoning"
echo "benchmarks (like GSM8K or BBH) because the Agentic Proxy is"
echo "designed strictly for tool-calling and code execution, not"
echo "conversational abstract logic."
echo ""

# Ensure API is running
if ! curl -s http://localhost:5440/health > /dev/null; then
    echo "[!] Error: The Krusch Agentic Proxy API is not running on localhost:5440."
    echo "    Please start the API first: python src/api_gateway.py"
    exit 1
fi

echo "[*] Krusch Agentic Proxy API detected on port 5440."

# Create isolated venv for benchmarking
if [ ! -d "bench_venv" ]; then
    echo "[*] Creating virtual environment for benchmarks..."
    python3 -m venv bench_venv
fi

source bench_venv/bin/activate

echo "[*] Installing lm-eval and dependencies (this may take a moment)..."
# We install lm-eval with the api extra
pip install --quiet "lm-eval[api]"

echo ""
echo "[*] Running Structural Benchmarks (Limit: 5 questions per task)..."
echo "    (Tasks: ifeval, humaneval)"
echo "------------------------------------------------------------"

# Run the evaluation harness targeting the local OpenAI-compatible endpoint
# We use local-chat-completions since Krusch Agentic Proxy serves /v1/chat/completions
lm_eval \
    --model local-chat-completions \
    --tasks ifeval,humaneval \
    --model_args model=krusch-brain,base_url=http://localhost:5440/v1/chat/completions \
    --apply_chat_template \
    --num_fewshot 5 \
    --limit 5 \
    --batch_size 1

echo "------------------------------------------------------------"
echo "[*] Running Weklund Tool-Calling Benchmark..."
echo "------------------------------------------------------------"
export PASSIVE_MODE=1
python benchmarks/weklund_tools.py

echo "------------------------------------------------------------"
echo "[*] Benchmark complete!"
echo "To run a full, unabridged benchmark (WARNING: Can take days on a single GPU), run:"
echo "source bench_venv/bin/activate && lm_eval --model local-chat-completions --tasks ifeval,humaneval --model_args model=krusch-brain,base_url=http://localhost:5440/v1/chat/completions"
