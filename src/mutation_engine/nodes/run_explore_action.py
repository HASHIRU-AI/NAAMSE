import random
from typing_extensions import List

from src.mutation_engine.mutation_workflow_state import BasePrompt, MutationEngineState, ScoredPrompt


def run_explore_action(state: MutationEngineState) -> MutationEngineState:
    """3a. Fetches a random prompt."""
    print("--- [Mutation Engine] Running Action: EXPLORE ---")
    all_prompts: List[ScoredPrompt] = state["input_prompts"]
    newly_generated_prompt = f"explored_({random.choice(all_prompts)['prompt'][0]})"
    return {"newly_generated_prompt": BasePrompt(prompt=[newly_generated_prompt])}
