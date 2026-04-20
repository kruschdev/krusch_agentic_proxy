from mtb.llm_benchmarks.models.base import ModelSpec

def format_hivemind_prompt(prompt: str) -> str:
    from mtb.quality_benchmarks.eval_problems import EvalProblem
    return prompt

MODEL_SPEC = ModelSpec(
    name="hivemind-swarm",
    num_params=0.0,
    prompt_formatter=format_hivemind_prompt,
    model_ids={
        "hivemind": {
            "int4": "hivemind-swarm",
            "int8": "hivemind-swarm",
            "fp16": "hivemind-swarm",
        }
    },
    thinking=True,
)
