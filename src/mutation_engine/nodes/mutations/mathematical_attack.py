from langchain.tools import tool
import random
from src.mutation_engine.mutation_workflow_state import BasePrompt


def math_wrap(prompt: str) -> str:
    """Substitutes the provided prompt with a mathematical transformation."""
    equations = [
            "{prompt} + 1",
            "{prompt} - 1",
            "{prompt} * 2",
            "{prompt} / 2",
            "{prompt} ** 2",
            "sqrt({prompt})",
            "log({prompt})",
            "exp({prompt})",
            "abs({prompt})",
            "round({prompt})",
            "floor({prompt})",
            "ceil({prompt})",
            "max({prompt}, 100)",
            "min({prompt}, 0)",
            "sum([{prompt}, 42, 27])",
            "prod([{prompt}, 3, 7])",
            "diff([{prompt}, 5, 2])",
            "quot([{prompt}, 8, 4])",
            "remainder({prompt}, 3)",
            "power({prompt}, 3)"
    ]
    equation = random.choice(equations)
    return equation.format(prompt=prompt)

@tool
def mathematical_attack(prompt: BasePrompt) -> BasePrompt:
    """Applies a mathematical transformation to the input prompt."""
    final_prompts = []
    for p in prompt["prompt"]:
        mutated_prompt = math_wrap(p)
        final_prompts.append(mutated_prompt)
    return BasePrompt(prompt=final_prompts)
