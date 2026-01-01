import random
from typing_extensions import List

from src.mutation_engine.mutation_workflow_state import BasePrompt, ClusterInfo, MutationEngineState, ScoredPrompt
from src.cluster_engine.utilities import get_random_prompt

def run_explore_action(state: MutationEngineState) -> MutationEngineState:
    """3a. Fetches a random prompt."""
    print("--- [Mutation Engine] Running Action: EXPLORE ---")

    random_prompt_info = get_random_prompt()
    prompt_text = random_prompt_info["prompt"]

    cluster_info: ClusterInfo = {}
    if "cluster_label" in random_prompt_info:
        cluster_info["cluster_label"] = random_prompt_info["cluster_label"]

    result: BasePrompt = {"prompt": [prompt_text]}
    if cluster_info:
        result["cluster_info"] = cluster_info

    return {
        "newly_generated_prompt": result
    }

    # all_prompts: List[ScoredPrompt] = state["input_prompts"]
    # newly_generated_prompt = f"explored_({random.choice(all_prompts)['prompt'][0]})"
    # return {"newly_generated_prompt": BasePrompt(prompt=[newly_generated_prompt])}
