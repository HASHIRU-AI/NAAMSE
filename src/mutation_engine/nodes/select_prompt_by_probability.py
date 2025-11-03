import random


def select_prompt_by_probability(state):
    """1. Selects a parent prompt."""
    selected = random.choices(
        state["input_prompts"],
        weights=state["prompt_probabilities"],
        k=1
    )[0]
    print(
        f"\n--- [Mutation Engine] Selected prompt (Score: {selected[1]}) ---")
    return {"selected_prompt": selected}