from src.mutation_engine.mutation_workflow_state import BasePrompt, MutationEngineState, ScoredPrompt


def run_similar_action(state: MutationEngineState) -> MutationEngineState:
    """3b. Fetches a similar prompt."""
    print("--- [Mutation Engine] Running Action: SIMILAR ---")
    selected_prompt: ScoredPrompt = state['selected_prompt']
    similar_prompt = BasePrompt(prompt=[f"similar_to_({selected_prompt['prompt'][0]})"])
    return {"newly_generated_prompt": similar_prompt}
