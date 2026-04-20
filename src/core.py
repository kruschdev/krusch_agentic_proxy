import re
import os
import sys
from typing import Tuple, Dict, Any, Optional

from src.client import chat

class KruschEngine:
    """
    Standalone implementation of the Dual-Engine Krusch Cognitive Architecture.
    Exposes the synchronous standard generation pipeline without internal RAG logic.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        llm_conf = self.config.get('llm', {})
        self.default_model = llm_conf.get('model', 'qwen2.5-coder:7b')
        self.base_url = llm_conf.get('api_url', 'http://127.0.0.1:11434/v1/chat/completions')
        self.unified_execution = True # Optimized: Merges Thinker/Implementer into a single autoregressive pass
        
    async def generate(self, 
                       prompt: str, 
                       is_code_exec: bool = False, 
                       is_tool_call: bool = False, 
                       exact_signature: str = "", 
                       target_model: Optional[str] = None,
                       max_tokens: int = 2048,
                       temperature: float = 0.1) -> Tuple[str, str]:
        """
        Executes the Dual-Engine pipeline.
        Returns a tuple: (Cognitive Blueprint, Final Response)
        """
        model = target_model if target_model else self.default_model
        
        # --- FAST-PATH NLP ROUTING ---
        if not is_code_exec and not is_tool_call:
            ai_config_standard = {
                 'provider': 'ollama',
                 'api_url': self.base_url,
                 'model': model,
                 'temperature': temperature,
                 'max_tokens': max_tokens
            }
            system_prompt = "You are a highly capable AI assistant. Answer the user's question clearly and accurately. If doing math, show your work step-by-step."
            
            try:
                # Fast-Path Direct Pass-through
                # Removed the slow Auditor loop based on 2026-04 benchmark findings.
                # The Krusch Gateway now acts as a lightning-fast pass-through for standard NLP/Math.
                final_response = await chat(system_prompt, prompt, ai_config_standard)
                return "FAST_PATH_DIRECT", final_response
            except Exception as e:
                raise RuntimeError(f"Standard Generation failed: {e}")
                
        if is_code_exec:
            components_schema = """  "components": [
    {
      "file_path": "string (The target filename for this component, e.g., index.js)",
      "description_constraint": "string (Meticulously inject any hard constraints, exact API prototypes, or architectural design from the Retrieved Context here so the generation factory doesn't guess)"
    }
  ]"""
        elif is_tool_call:
            components_schema = """  "tool_plan": "string (A step-by-step plan of which tools to call and what arguments to pass)" """
        else:
            components_schema = """  "solution_format": "string (Specify the structural rules for the final response. If the original problem requires a strict ending format like '#### <number>', enforce it here. Otherwise, instruct the Implementer to answer naturally.)" """



        target_schema = f"""{{
  "cognitive_scratchpad": "string (MANDATORY: Step-by-step logic, math analysis, and problem constraint breakdown. Think out loud here before writing any constraints.)",
  "task_name": "string (Short title of the task)",
  "strict_problem_constraints": "string (MANDATORY: Extract and list any explicit 'Do Not' or 'Must' rules mentioned in the ORIGINAL PROBLEM here so they are preserved)",
{components_schema}
}}"""

        holodata_enforcement = f"""You MUST wrap your JSON blueprint strictly inside <holodata> xml tags exactly like this:
<holodata>
{target_schema}
</holodata>
"""

        # --- LAYER 1: The Thinker ---
        if is_code_exec and exact_signature:
            thinker_enforcement = f"MANDATORY SIGNATURE: You must instruct the Implementer to use the exact function signature `{exact_signature}` in the description_constraint. If returning a JSON schema, it MUST be wrapped inside this Python function returning a dictionary."
        else:
            thinker_enforcement = ""

        thinker_system_prompt = f"""You are the Krusch Executive Intelligence (Layer 1: The Thinker).
Your job is NOT to write final code. 
Your job is to read the user's problem, and write a strict "Cognitive Blueprint" telling the implementation layer exactly how to solve it, what bugs to fix, and what constraints to follow.
DO NOT write the final implementation logic yourself. You are strictly responsible for architecting the Blueprint boundaries."""
        
        # Heuristic: Prevent few-shot examples from hijacking the Thinker's JSON format
        thinker_prompt = prompt
        if "Question:" in prompt and prompt.count("Question:") > 1:
            parts = prompt.rsplit("Question:", 1)
            final_question_part = "Question:" + parts[-1]
            thinker_prompt = f"[Context: The user provided formatting examples earlier, but they have been omitted here to prevent you from mimicking their raw text syntax. Your ONLY task is to solve the final question below and output a <holodata> JSON blueprint.]\n\n{final_question_part}"

        thinker_msg = f"ORIGINAL PROBLEM:\n{thinker_prompt}\n\n{thinker_enforcement}\n\n{holodata_enforcement}\n\nCRITICAL: You MUST output ONLY a valid JSON object wrapped in <holodata> matching the requested schema. Do not output markdown explanations or math formulas outside of the JSON. NOTE: Do not leak your own <holodata> formatting instructions into the 'strict_problem_constraints'. The Implementer uses a different output format."
        
        ai_config_thinker = {
             'provider': 'ollama',
             'api_url': self.base_url,
             'model': model,
             'temperature': 0.1,
             'max_tokens': 1500
        }

        # --- DYNAMIC RULES FOR IMPLEMENTER ---
        if is_code_exec:
            sig_text = f" You MUST define your solution using the exact Python function signature: `{exact_signature}`. If the problem asks for a JSON schema, you must return it as a Python dictionary from this exact function. Return the function definition so it can be evaluated by test cases." if exact_signature else ""
            coding_rule = f"CRITICAL RULE 1: Respond ONLY with exactly ONE markdown code block containing your ENTIRE solution.{sig_text} DO NOT output any conversational text, 'Step-by-Step' explanations, or markdown text outside of the code block. DO NOT include example execution code, if __name__ == '__main__' blocks, or print statements. Your output will be piped directly into a strict Python sandbox."
        elif is_tool_call:
            coding_rule = """CRITICAL RULE 1: If valid tools match the request, respond ONLY with a raw JSON array representing the Tool Calls. DO NOT wrap the output in markdown code blocks. DO NOT output conversational text. 
CRITICAL RULE 2: The JSON array MUST conform exactly to this flat schema: [{"name": "<function_name>", "arguments": {"<param1>": "<val1>"}}]
DO NOT use the OpenAI "type" or "function" wrappers. ALWAYS use the key "arguments", never "parameters".
CRITICAL RULE 3: If the prompt asks you to plan a sequence of tool calls, you MUST output ALL the tool calls in the sequence as a single JSON array, even if later tools depend on the results of earlier tools (use placeholder values for unknown arguments).
CRITICAL RULE 4: IF NO TOOLS MATCH the request, OR if you must refuse a request (e.g. it is unsafe), OR if you must ask the user a question for missing required parameters: IGNORE RULE 1. DO NOT output a JSON array of tool calls. Instead, output ONLY a normal natural language message to the user explaining the situation. NEVER output a tool call with empty placeholder values if a required parameter is missing. NEVER bypass this by smuggling conversational messages into the arguments of an unrelated tool (e.g., using a translation or echo tool to 'talk' to the user).
CRITICAL RULE 5: YOU MUST ONLY USE TOOLS EXPLICITLY LISTED IN THE PROMPT. DO NOT HALLUCINATE TOOLS."""
        else:
            coding_rule = "CRITICAL RULE 1: Do NOT write any Python code or functions. Explain your reasoning clearly in pure text. You MUST use the step-by-step logic provided in the Cognitive Blueprint's 'cognitive_scratchpad' to arrive at your answer so you don't make math mistakes. Follow the 'solution_format' provided in the Cognitive Blueprint for how to format your final answer."

        if self.unified_execution:
            unified_system_prompt = f"""You are the Krusch Agentic Proxy. You operate in a strict autonomous environment.
Your task is to solve the user's problem by FIRST outputting a strict "Cognitive Blueprint" detailing your logic, and THEN immediately outputting your final answer.
{coding_rule}

{holodata_enforcement}

CRITICAL: You MUST FIRST output the <holodata> block, and THEN output your final answer directly after the </holodata> closing tag. DO NOT wait for the user to reply."""
            
            unified_msg = f"ORIGINAL PROBLEM:\n{thinker_prompt}\n\n{thinker_enforcement}"
            ai_config_unified = {
                 'provider': 'ollama',
                 'api_url': self.base_url,
                 'model': model,
                 'temperature': temperature,
                 'max_tokens': max_tokens + 1500 # space for both blueprint and answer
            }
            
            try:
                raw_response = await chat(unified_system_prompt, unified_msg, ai_config_unified)
                blueprint = "No Blueprint Found"
                response_text = raw_response
                match = re.search(r"<holodata>(.*?)</holodata>", raw_response, re.DOTALL)
                if match:
                    blueprint = match.group(1).strip()
                    response_text = raw_response[match.end():].strip()
            except Exception as e:
                raise RuntimeError(f"Unified Intelligence Generation failed: {e}")
        else:
            # --- SEQUENTIAL FALLBACK ---
            try:
                blueprint = await chat(thinker_system_prompt, thinker_msg, ai_config_thinker)
                print(f"\n--- BLUEPRINT ---\n{blueprint}\n-----------------\n")
                # Add absolute-bottom schema extraction to maximize precision
                match = re.search(r"<holodata>(.*?)</holodata>", blueprint, re.DOTALL)
                if match:
                    blueprint = match.group(1).strip()
            except Exception as e:
                raise RuntimeError(f"Intelligence Generation failed on Layer 1: {e}")
            
            # --- LAYER 2: The Implementer ---
            implementer_system_prompt = f"""You are the Krusch Implementation Node (Layer 2: The Builder).
You are operating in a strict autonomous sandbox environment.
{coding_rule}
If the Cognitive Blueprint defines strict problem constraints or exact domain roles, you MUST adhere to them. Let the constraints dictate your final output structure.
You will be provided an original problem, and a "Cognitive Blueprint" from the Executive Intelligence tier.
Your job is to blindly follow the Cognitive Blueprint to write the final output or code. 
DO NOT regurgitate the blueprint back to the user. Produce exactly what the original problem requested, guided by the blueprint."""
            implementer_prompt = prompt
            if "Question:" in prompt and prompt.count("Question:") > 1:
                dummy_holodata = """<holodata>
{
  "cognitive_scratchpad": "Analyzing the numerical relationships...",
  "task_name": "Math Calculation",
  "strict_problem_constraints": "None",
  "solution_format": "#### <number>"
}
</holodata>
"""
                implementer_prompt = prompt.replace("Answer:", f"Answer:\n{dummy_holodata}")

            implementer_msg = f"ORIGINAL PROBLEM (Context):\n{implementer_prompt}\n\nCOGNITIVE BLUEPRINT (Strict Instructions):\n{blueprint}"
            
            ai_config_implementer = {
                 'provider': 'ollama',
                 'api_url': self.base_url,
                 'model': model,
                 'temperature': temperature,
                 'max_tokens': max_tokens
            }
            
            try:
                response_text = await chat(implementer_system_prompt, implementer_msg, ai_config_implementer)
            except Exception as e:
                raise RuntimeError(f"Intelligence Generation failed on Layer 2: {e}")

        # Post-Processing: Strip conversational markdown fences if present for code/tools
        cleaned = response_text.strip()
        if is_code_exec or is_tool_call:
            match = re.search(r"```[a-zA-Z]*\n(.*?)```", cleaned, re.DOTALL)
            if match:
                response_text = match.group(1).strip()
            elif cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [l for l in lines[1:] if not l.strip().startswith("```")]
                response_text = "\n".join(lines).strip()
            elif is_tool_call and cleaned.startswith("[") and "]" in cleaned:
                pass

                
        return blueprint, response_text
