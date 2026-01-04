import random

from src.mutation_engine.mutation_workflow_state import MutationEngineState, ScoredPrompt

def decide_action_by_score(state: MutationEngineState) -> MutationEngineState:
    """2. Decides which action to take based on score."""
    prompt: ScoredPrompt = state["selected_prompt"]
    
    # Use task-specific seed if provided (for deterministic parallel execution)
    task_seed = state.get("task_seed")
    if task_seed is not None:
        print(f"[Seeding] decide_action using task_seed={task_seed}")
    rng = random.Random(task_seed) if task_seed is not None else random

    if prompt['score'] < 0.5:  # <50%
        action = rng.choices(["explore", "similar", "mutate"], weights=[
                                0.7, 0.2, 0.1], k=1)[0]
    elif prompt['score'] < 0.8:  # 50-80%
        action = rng.choices(["explore", "similar", "mutate"], weights=[
                                0.1, 0.7, 0.2], k=1)[0]
    else:  # >80%
        action = rng.choices(["explore", "similar", "mutate"], weights=[
                                0.1, 0.2, 0.7], k=1)[0]

    print(f"--- [Mutation Engine] Decided action: {action} ---")
    return {"action_to_take": action}