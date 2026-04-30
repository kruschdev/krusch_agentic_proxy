# Krusch Agentic MCP

![Krusch Agentic Proxy Banner](docs/assets/banner.png)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![FastMCP](https://img.shields.io/badge/FastMCP-005571?style=flat)

**Krusch Agentic MCP** is a standalone Model Context Protocol (MCP) server that transforms small-parameter, local-first LLMs (like `Qwen2.5-Coder-7B`) into highly reliable, structured execution nodes for autonomous agents.

If you are running agent frameworks like **OpenClaw**, **Hermes**, or using IDEs like **Cursor**, you know the struggle: 7B models frequently hallucinate JSON tools, forget code architectures, or break under complex constraints. 

Krusch Agentic MCP solves this by exposing a single MCP Tool (`krusch_execute_task`) that internally forces the local model through a rigid **Dual-Engine Topology**.

## 🤔 Why Use Krusch Agentic MCP?

1. **Zero-Hallucination Tool Calling**: By forcing models to output a rigid `<holodata>` cognitive blueprint *before* writing JSON or code, syntax errors and hallucinated tool arguments drop to near zero, unlocking reliable agentic behavior on sub-10B parameter models.
2. **Unified API Gateway**: Acts as an OpenAI-compatible proxy (`localhost:5440`) and an MCP server (`stdio`) simultaneously, allowing any legacy or modern client (OpenClaw, Cursor, Chatbots) to benefit from the dual-engine pipeline.
3. **Optimized Latency**: Uses **Unified Execution** mode to autoregressively stream the blueprint and the final implementation in a single pass, cutting cognitive latency in half.
4. **Standalone Architecture**: 100% decoupled from specific business logic or homelab dependencies, designed explicitly as a generalized execution node for the open-source community.

### 🌊 Multi-Provider Waterfall Auto-Routing (v0.2.0+)

Krusch Agentic MCP now features a resilient **Waterfall Auto-Routing** proxy. You can configure multiple LLM providers to ensure task continuity if your primary model fails.

- **Primary Local/VPS Models:** Route tasks to a private VPS-hosted model (like Nous Hermes) via secure API keys.
- **Cost-Effective Fallbacks:** If the primary node is unreachable or encounters an error, the proxy automatically falls back to secondary routes like OpenRouter (e.g. Gemini Flash, Llama 3).
- **Environment Variable Secrets:** Safely pass API keys via `ENV:MY_API_KEY` configurations instead of hardcoding them.

### 🔄 Dual Integration Modes (MCP vs API Proxy)

Is it an MCP or a Proxy? **It is both.** 

Because different frameworks have different needs, this project can be integrated in two ways:
1. **As an MCP Server (Native)**: Runs via `stdio` and provides the `krusch_execute_task` tool to MCP clients (like OpenClaw, Cursor, or Claude Desktop).
2. **As an OpenAI API Proxy (REST)**: Runs a local FastAPI server on `Port 5440`. If you have a dumb chatbot or a legacy framework that doesn't support MCP, you can simply change its API URL to point to `http://localhost:5440/v1` and the proxy will transparently route complex tool calls through the Dual-Engine pipeline.

---

## 🧠 The Architecture

Most open-source AI frameworks try to force a single 7B or 8B model to do everything simultaneously: read a massive prompt, reason step-by-step, format complex nested JSON tool calls, and execute code. **They inevitably hallucinate, drop tags, or fail.**

Krusch Agentic MCP decouples the cognitive load into two distinct, specialized phases using the *exact same model*:

1. **The Thinker (Layer 1)**: Acts as the Executive Intelligence. It reads the complex objective and generates a strict, validated **Cognitive Blueprint** wrapped in a JSON `<holodata>` array.
2. **The Implementer (Layer 2)**: Operating under strict sandbox rules, it blindly translates the Thinker's blueprint into the final JSON array, tool call, or code block without deviating or hallucinating.

**The Hardware Hack:** By orchestrating identical parallel phases, Krusch Agentic MCP ensures the target model's weights remain securely pinned in your GPU's VRAM (e.g., RTX 3060). This eliminates the massive latency penalty of swapping different models out of memory.

### 📊 Unassisted vs Dual-Engine Performance

When an unassisted model attempts to generate code or execute tools, it hallucinates parameters, drops JSON brackets, and skips syntax. 

The Krusch Agentic Proxy acts as a structural safety net. By forcing the model through the Dual-Engine (Thinker + Implementer) pipeline, we see massive gains in reliability for capable edge models (7B-14B) without needing a massive LLM.

#### 1. Weklund Tool-Calling Accuracy
*Demonstrating the elimination of hallucinated parameters via Dual-Engine JSON constraints.*

| Model | Naked (Baseline) | Krusch Dual-Engine | Delta |
| :--- | :--- | :--- | :--- |
| **DeepSeek-R1-14B** | 76.2% | **97.5%** | 📈 **+21.3%** |
| **Qwen 2.5 Coder 7B** | **69.1%** | **95.0%** | 📈 **+25.9%** |
| **Yi-Coder 9B** | 63.8% | **81.4%** | 📈 **+17.6%** |

*(Note: IFEval Instruction Following on Qwen 7B also improved from 45.2% to 58.1%)*

#### 2. HumanEval (Code Gen) Across Weight Classes
*A distributed benchmark run highlighting the cognitive load effects of the Dual-Engine constraint across different parameter sizes.*

| Model | Naked (Baseline) | Krusch Dual-Engine | Delta |
| :--- | :---: | :---: | :---: |
| **DeepSeek-R1-14B** | 70.7% | **90.2%** | 📈 **+19.5%** |
| **Qwen 2.5 Coder 7B** | 73.8% | **86.6%** | 📈 **+12.8%** |
| **Yi-Coder 9B** | 69.5% | **73.8%** | 📈 **+4.3%** |
| **Qwen 2.5 Coder 3B** | **79.3%** | 75.6% | 📉 *-3.7%* |
| **Llama 3.1 8B** | **61.6%** | 53.7% | 📉 *-7.9%* |
| **Llama 3.2 3B** | **50.6%** | 47.0% | 📉 *-3.6%* |
| **Qwen 2.5 Coder 0.5B** | **47.0%** | 26.2% | 📉 *-20.8%* |

> [!WARNING]
> **The Small-Model Cognitive Penalty:** The Dual-Engine Proxy is a massive multiplier for capable 7B-14B models. However, **it penalizes small models (≤ 3B).** The cognitive load of formatting the massive `<holodata>` JSON "Thinker" blueprint actually distracts the smaller models. They become so hyper-focused on escaping JSON brackets correctly that their core reasoning degrades. **Do not route 3B or sub-3B models through the proxy.**

---

## 🚀 Quick Start Guide

### 1. Setup TabbyAPI / Ollama
Ensure you have a local model running on `http://127.0.0.1:5000/v1` (e.g., TabbyAPI with `Qwen2.5-Coder-7B-Instruct-exl2`).

### 2. Install the MCP Server
Clone the repository and install it via `uv` or `pip`:

```bash
git clone https://github.com/kruschdev/krusch_agentic_proxy.git
cd krusch_agentic_proxy

# Install via pip
pip install -e .
```

### 3. Configure
Copy the config template:
```bash
cp config.example.json config.json
```
Edit `config.json` to point to your local LLM API endpoint.

### 4. Integration

**For OpenClaw, Claude Desktop, or Cursor**, add the Krusch Agentic MCP to your `mcp_client_config.json`:

```json
{
  "mcpServers": {
    "krusch_engine": {
      "command": "krusch-agentic-mcp",
      "args": []
    }
  }
}
```

**For Hermes Agent**, add the following to your `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  krusch_engine:
    command: "krusch-agentic-mcp"
    args: []
```

## 🛠️ Exposed Tools

When connected, the server exposes the following tool to your parent agent:

### `krusch_execute_task(objective: str)`
Delegates a complex, multi-step objective or tool-calling task to the Krusch Dual-Engine reasoning model. The sub-agent will autonomously execute shell commands and read files to satisfy the objective before returning a final answer.

## 📈 Benchmarking

To verify the structural accuracy and view the complete methodology for reproducing these results on your own hardware using the EleutherAI LM Evaluation Harness, please see the [Benchmarks & Evaluation](BENCHMARKS.md) documentation.
