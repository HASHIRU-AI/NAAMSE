from typing_extensions import List, Tuple
from src.mutation_engine.mutation_workflow_state import MutationEngineState, ScoredPrompt


def calculate_probabilities(state: MutationEngineState) -> MutationEngineState:
    """Entrypoint: Calculates selection probability and initializes output list."""
    print("--- [Mutation Engine] Calculating prompt probabilities... ---")

    input_prompts: List[ScoredPrompt] = state["input_prompts"]
    scores = [prompt["score"] for prompt in input_prompts]

    if not scores or all(s == 0 for s in scores):
        # Fallback to uniform distribution if no scores available
        count = len(input_prompts)
        probabilities = [1.0 / count] * count
    else:
        # Normalize scores to create probabilities
        total_score = sum(scores)
        probabilities = [score / total_score for score in scores]

    return {
        "prompt_probabilities": probabilities,
        "final_generated_prompts": []
    }
