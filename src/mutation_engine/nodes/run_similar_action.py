from typing import Any, Dict
from src.mutation_engine.mutation_workflow_state import BasePrompt, ClusterInfo, Metadata, Mutation, MutationEngineState, ScoredPrompt
from src.cluster_engine.utilities import find_nearest_prompts
import random
from langchain_core.runnables import RunnableConfig


def run_similar_action(state: MutationEngineState, config: RunnableConfig) -> MutationEngineState:
    """3b. Fetches a similar prompt."""
    print("--- [Mutation Engine] Running Action: SIMILAR ---")

    # Use task-specific seed if provided (for deterministic parallel execution)
    task_seed = state.get("task_seed")
    if task_seed is not None:
        print(f"[Seeding] run_similar_action using task_seed={task_seed}")
    rng = random.Random(task_seed) if task_seed is not None else random

    selected_prompt: ScoredPrompt = state['selected_prompt']
    query_prompt = selected_prompt["prompt"][0]

    # Use embedding-based nearest neighbor search
    # (uses default JSONL data source + auto device selection)
    # database is in metadata.database
    database = config.get("configurable", {}).get("database", None)

    nearest_list = find_nearest_prompts(
        query_prompt, n=10, seed=task_seed, data_source=database)

    if not nearest_list:
        raise ValueError("No similar prompts found in the corpus")

    # Pick a random one from the top 10
    nearest = rng.choice(nearest_list)
    similar_text = nearest["prompt"]

    cluster_info: ClusterInfo = {}
    if "cluster_id" in nearest:
        cluster_info["cluster_id"] = nearest["cluster_id"]
    if "cluster_label" in nearest:
        cluster_info["cluster_label"] = nearest["cluster_label"]

    metadata: Metadata = {"mutation_type": Mutation.SIMILAR}
    if cluster_info:
        metadata["cluster_info"] = cluster_info

    result: BasePrompt = {"prompt": [similar_text], "metadata": metadata}

    return {
        "newly_generated_prompt": result,
    }
