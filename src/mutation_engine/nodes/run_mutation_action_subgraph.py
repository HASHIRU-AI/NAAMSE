import random
from src.mutation_engine.mutation_workflow_state import BasePrompt, MutationEngineState, MutationWorkflowState, Mutation

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from src.experiments.ablation_config import get_ablation
from src.mutation_engine.nodes.invoke_mutation_llm import invoke_llm_with_tools
from src.mutation_engine.mutation_workflow_state import MutationWorkflowState
from src.mutation_engine.selectors import select_mutation

def select_mutation_type(state: MutationWorkflowState, config: RunnableConfig) -> MutationWorkflowState:
    """1. Selects a mutation type using the ablation's mutation selector (default: uniform random)."""
    # Use task-specific seed if provided (for deterministic parallel execution)
    task_seed = state.get("task_seed")
    rng = random.Random(task_seed) if task_seed is not None else random
    ablation = get_ablation(config)
    selected_type, probabilities = select_mutation(state["prompt_to_mutate"], rng, ablation)
    print(f"  [Mutation Subgraph] Selected mutation: {selected_type} "
          f"(selector={ablation.mutation_selector}, task_seed={task_seed})")

    return {"mutation_type": selected_type, "mutation_probabilities": probabilities}

# Build and compile the subgraph
mutation_workflow_builder = StateGraph(MutationWorkflowState)
mutation_workflow_builder.add_node(
    "select_mutation_type", select_mutation_type)
mutation_workflow_builder.add_node(
    "invoke_llm_with_tools", invoke_llm_with_tools)
mutation_workflow_builder.add_edge(START, "select_mutation_type")
mutation_workflow_builder.add_edge(
    "select_mutation_type", "invoke_llm_with_tools")
mutation_workflow_builder.add_edge("invoke_llm_with_tools", END)

mutation_action_subgraph = mutation_workflow_builder.compile()


def run_mutation_action_subgraph(state: MutationEngineState, config: RunnableConfig) -> MutationEngineState:
    """3c. Invokes the mutation subgraph."""
    print("--- [Mutation Engine] Running Action: MUTATE ---")
    prompt_to_mutate: BasePrompt = state['selected_prompt']

    subgraph_output: MutationWorkflowState = mutation_action_subgraph.invoke({
        "prompt_to_mutate": prompt_to_mutate,
        "task_seed": state.get("task_seed")  # Pass seed for deterministic selection
    }, config=config)  # Pass config so the ablation reaches select_mutation_type
    mutated_prompt = dict(subgraph_output["mutated_prompt"])
    metadata = dict(mutated_prompt.get("metadata") or {})
    metadata["selector"] = {
        "mutation_type": subgraph_output.get("mutation_type"),
        "mutation_probabilities": subgraph_output.get("mutation_probabilities"),
    }
    mutated_prompt["metadata"] = metadata
    return {"newly_generated_prompt": mutated_prompt}
