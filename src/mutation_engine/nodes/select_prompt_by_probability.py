import random

from src.mutation_engine.mutation_workflow_state import MutationEngineState, ScoredPrompt


def select_prompt_by_probability(state: MutationEngineState) -> MutationEngineState:
    """1. Selects a parent prompt."""
    # Use task-specific seed if provided (for deterministic parallel execution)
    task_seed = state.get("task_seed")
    rng = random.Random(task_seed) if task_seed is not None else random
    selected: ScoredPrompt = rng.choices(
        state["input_prompts"],
        weights=state["prompt_probabilities"],
        k=1
    )[0]

    print(
        f"\n--- [Mutation Engine] Selected prompt (Score: {selected['score']}) ---")
    return {"selected_prompt": selected}
