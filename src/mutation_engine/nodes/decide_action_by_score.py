import random

from langchain_core.runnables import RunnableConfig

from src.experiments.ablation_config import get_ablation
from src.mutation_engine.mutation_workflow_state import MutationEngineState, ScoredPrompt
from src.mutation_engine.selectors import select_action

def decide_action_by_score(state: MutationEngineState, config: RunnableConfig) -> MutationEngineState:
    """2. Decides which action to take, using the ablation's action selector (default: score thresholds)."""
    prompt: ScoredPrompt = state["selected_prompt"]
    ablation = get_ablation(config)

    # Use task-specific seed if provided (for deterministic parallel execution)
    task_seed = state.get("task_seed")
    if task_seed is not None:
        print(f"[Seeding] decide_action using task_seed={task_seed}")
    rng = random.Random(task_seed) if task_seed is not None else random
    print(f"--- [Mutation Engine] Deciding action for prompt with score {prompt['score']:.4f} "
          f"(selector={ablation.action_selector}) ---")
    action, probabilities = select_action(prompt, rng, ablation, state.get("uncovered_categories"))
    print(f"--- [Mutation Engine] Decided action: {action} ---")
    return {"action_to_take": action, "action_probabilities": probabilities}
