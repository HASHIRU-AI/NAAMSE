import random


def run_explore_action(state):
    """3a. Fetches a random prompt."""
    print("--- [Mutation Engine] Running Action: EXPLORE ---")
    all_prompts = [p[0] for p in state["input_prompts"]]
    return {"newly_generated_prompt": f"explored_({random.choice(all_prompts)})"}