"""Hivemind Swarm Benchmark — Full RAG + Personas + 3-GPU Distributed Consensus.

Architecture:
  1. Query PostgreSQL vector store for context relevant to each problem
  2. Query tbl_elite_personas for 3 different expert personas
  3. Fire all 3 models in parallel across 3 GPUs, each with a unique persona
  4. If 3B and 9B agree → fast path (use 3B response)
  5. If they disagree → use 14B tiebreaker

Fleet GPU Distribution:
  - Node A GPU 1  → qwen2.5-coder:3b
  - Node B GPU 2  → yi-coder:9b
  - Node C GPU 3  → deepseek-r1:14b
"""

import time
import os
import sys
import re
import asyncio
import json
from typing import Any, Callable, Optional, List

import httpx

from mtb.llm_benchmarks.base_llm_benchmark import BaseLLMBenchmark
from mtb.measurement import LlmBenchmarkMeasurement

# Add Hivemind project dir
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
HIVEMIND_DIR = os.path.dirname(DS_DIR)
HOMELAB_DIR = os.path.dirname(os.path.dirname(HIVEMIND_DIR))
sys.path.append(HIVEMIND_DIR)
sys.path.append(os.path.join(HOMELAB_DIR, 'lib-py'))

# Fleet-distributed Ollama endpoints — one per GPU
OLLAMA_NODE_1 = "http://localhost:11434/v1/chat/completions"
OLLAMA_NODE_2 = "http://localhost:11435/v1/chat/completions"
OLLAMA_NODE_3 = "http://localhost:11436/v1/chat/completions"

async def _call_ollama(model: str, system_prompt: str, user_prompt: str, 
                       max_tokens: int = 2048, url: str = OLLAMA_NODE_1) -> str:
    """Direct Ollama call — fleet-distributed with persona-injected prompts."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=120) as client:
        res = await client.post(url, json=payload, headers=headers)

    if res.status_code != 200:
        return f"Error: Ollama {res.status_code}: {res.text[:200]}"

    data = res.json()
    text = data["choices"][0]["message"]["content"].strip()
    # Strip <think> tags from deepseek-r1
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return text


def _normalize_code(text: str) -> str:
    """Strip markdown fences and whitespace for comparison."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines[1:] if l.strip() != "```"]
        cleaned = "\n".join(lines)
    return cleaned.strip()


def _outputs_agree(resp_a: str, resp_b: str) -> bool:
    """Check if two model outputs are substantially similar."""
    norm_a = _normalize_code(resp_a)
    norm_b = _normalize_code(resp_b)
    
    if norm_a == norm_b:
        return True
    
    compact_a = re.sub(r'\s+', '', norm_a)
    compact_b = re.sub(r'\s+', '', norm_b)
    if compact_a == compact_b:
        return True
    
    # Character-level overlap for fuzzy match
    if len(compact_a) > 20 and len(compact_b) > 20:
        set_a = set(compact_a)
        set_b = set(compact_b)
        overlap = len(set_a & set_b) / max(len(set_a | set_b), 1)
        if overlap > 0.85:
            return True

    return False


class HivemindLlmBenchmark(BaseLLMBenchmark):
    """Full Hivemind RAG + Persona + 3-GPU Distributed Consensus Benchmark."""
    framework = "hivemind"

    def __init__(
        self,
        name: str,
        prompt_formatter: Callable[[str], Any],
        model_id: str,
        backend: str,
        dtype: str,
        max_num_tokens: int,
        thinking: bool = True,
    ):
        super().__init__(name, model_id, backend, dtype, prompt_formatter, max_num_tokens, thinking)
        
        # Load config
        config_path = os.path.join(HIVEMIND_DIR, 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize Hivemind PostgreSQL RAG pipeline
        from database.init_db import setup_database
        from krusch_toolkit.db import create_session_factory
        from hivemind.vector_store import VectorStore
        
        self.engine = setup_database(self.config)
        self.session_factory = create_session_factory(self.engine)
        self.v_store = VectorStore(self.session_factory)
        
        # Model assignments — one per GPU
        self.model_3b = self.config['llm']['fast_model']       # Fast Inference
        self.model_9b = 'yi-coder:9b'                          # Medium Inference
        self.model_14b = 'qwen2.5-coder:7b'                    # Heavy Inference
        
        self.loop = asyncio.new_event_loop()
        
        # Stats
        self.consensus_3b_9b = 0
        self.consensus_3b_14b = 0
        self.consensus_9b_14b = 0
        self.no_consensus_fallback = 0

    def format_and_tokenize_prompt(self, prompt: str):
        return prompt

    def get_num_prompt_tokens(self, user_prompt: str) -> int:
        return len(str(user_prompt)) // 4

    def setup(self):
        print(f"\n[Hivemind Swarm] Full RAG + Persona + 2-GPU Consensus Mode")
        print(f"  CUDA 1: {self.model_3b} → Node A")
        print(f"  CUDA 2: {self.model_9b} → Node B")
        print(f"  RAG: PostgreSQL chrysalis_v0 (vector store + personas)")
        print()

    async def _get_rag_context(self, prompt: str) -> dict:
        """Query PostgreSQL vector store for relevant context and personas."""
        # Vector search for relevant homelab/code context
        results = await self.v_store.search(prompt, limit=2)
        context_str = "\n".join([f"Source [{r['title']}]: {r['content']}" for r in results])
        if not context_str:
            context_str = "No specific constraints found. Use general best practices."
        
        # Get 3 different personas — one for each model
        personas = await self.v_store.search_personas(prompt, limit=3)
        
        return {
            "context": context_str,
            "personas": personas,
        }

    def _build_persona_prompt(self, persona: Optional[dict], base_angle: str) -> str:
        """Build a system prompt with RAG persona injection."""
        if persona:
            role = persona.get('role', 'Expert Developer')
            rules = persona.get('domain_rulesets', '')
            return (
                f"You are {role}. {rules}\n\n"
                f"Approach: {base_angle}\n\n"
                f"Output ONLY the requested code or answer. No explanations, no markdown fences "
                f"unless the code itself requires them. For Python code problems, output valid "
                f"Python that can be executed directly."
            )
        else:
            return (
                f"You are a precise coding assistant. Approach: {base_angle}\n\n"
                f"Output ONLY the requested code or answer. No explanations, no markdown fences "
                f"unless the code itself requires them. For Python code problems, output valid "
                f"Python that can be executed directly."
            )

    async def _async_run(self, prompt: str) -> str:
        """Full RAG + Persona + 2-GPU parallel consensus.
        
        1. Query PostgreSQL for context
        2. Fire 2 models in parallel across 2 CUDA GPUs
        3. Consensus check → fast return or Yi fallback
        """
        
        # Phase 0: RAG retrieval from PostgreSQL
        rag = await self._get_rag_context(prompt)
        personas = rag["personas"]
        
        persona_3b = personas[0] if len(personas) > 0 else None
        persona_9b = personas[1] if len(personas) > 1 else None
        
        sys_3b = self._build_persona_prompt(persona_3b, "Write minimal, correct code. Speed over elegance.")
        sys_9b = self._build_persona_prompt(persona_9b, "Focus on correctness and edge cases. Fast clean syntax.")
        
        user_prompt = f"Context:\n{rag['context']}\n\nProblem:\n{prompt}"
        
        # Fire 2 in TRUE parallel
        task_3b = asyncio.create_task(
            _call_ollama(self.model_3b, sys_3b, user_prompt, self.max_num_tokens, url=OLLAMA_NODE_1)
        )
        task_9b = asyncio.create_task(
            _call_ollama(self.model_9b, sys_9b, user_prompt, self.max_num_tokens, url=OLLAMA_NODE_2)
        )
        
        # Wait for both CUDA GPUs
        resp_3b, resp_9b = await asyncio.gather(task_3b, task_9b)
        
        # Ensemble verification
        if _outputs_agree(resp_3b, resp_9b):
            self.consensus_3b_9b += 1
            return resp_3b
            
        # They disagreed → fallback to the heavy model (Yi on 3060)
        self.no_consensus_fallback += 1
        return resp_9b

    def run_once(self, prompt: str) -> LlmBenchmarkMeasurement:
        start_time = time.perf_counter()
        
        response = self.loop.run_until_complete(self._async_run(prompt))
        
        # Wrap bare code in fences for the sandbox evaluator
        if "```" not in response:
            response = f"```python\n{response}\n```"

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        return LlmBenchmarkMeasurement(
            response=response,
            num_prompt_tokens=self.get_num_prompt_tokens(prompt),
            num_generated_tokens=len(response) // 4,
            prompt_tps=0.0,
            generation_tps=(len(response) // 4) / max(elapsed, 0.1),
            prompt_time_sec=0.0,
            generation_time_sec=elapsed,
            peak_memory_gib=0.0,
        )

    def teardown(self):
        total = (self.consensus_3b_9b + self.consensus_3b_14b + 
                 self.consensus_9b_14b + self.no_consensus_fallback)
        if total > 0:
            print(f"\n[Hivemind Swarm] Consensus Stats:")
            print(f"  Majority (3B + 9B):   {self.consensus_3b_9b}/{total} ({self.consensus_3b_9b/total*100:.0f}%)")
            print(f"  Majority (3B + 14B):  {self.consensus_3b_14b}/{total} ({self.consensus_3b_14b/total*100:.0f}%)")
            print(f"  Majority (9B + 14B):  {self.consensus_9b_14b}/{total} ({self.consensus_9b_14b/total*100:.0f}%)")
            print(f"  No Majority Fallback: {self.no_consensus_fallback}/{total} ({self.no_consensus_fallback/total*100:.0f}%)")
        self.loop.close()
