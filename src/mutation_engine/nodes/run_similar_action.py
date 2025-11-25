from src.mutation_engine.mutation_workflow_state import BasePrompt, MutationEngineState, ScoredPrompt

from src.cluster_engine.utilities import find_nearest_prompts


def run_similar_action(state: MutationEngineState) -> MutationEngineState:
    """3b. Fetches a similar prompt."""
    print("--- [Mutation Engine] Running Action: SIMILAR ---")
    selected_prompt: ScoredPrompt = state['selected_prompt']
    query_prompt = selected_prompt["prompt"][0]

    # Use embedding-based nearest neighbor search
    # (uses default JSONL data source + auto device selection)
    nearest_list = find_nearest_prompts(query_prompt, n=1)

    if not nearest_list:
        raise ValueError("No similar prompts found in the corpus")

    nearest = nearest_list[0]
    similar_text = nearest["prompt"]

    return {
        "newly_generated_prompt": BasePrompt(prompt=[similar_text]),
    }
    # similar_prompt = BasePrompt(prompt=[f"similar_to_({selected_prompt['prompt'][0]})"])
    # return {"newly_generated_prompt": similar_prompt}
