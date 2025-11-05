import random
from src.mutation_engine.mutation_workflow_state import BasePrompt, MutationEngineState, MutationWorkflowState


from langgraph.graph import StateGraph, START, END

from src.mutation_engine.nodes.invoke_mutation_llm import invoke_llm_with_tools
from src.mutation_engine.mutation_workflow_state import MutationWorkflowState

def select_mutation_type(state: MutationWorkflowState) -> MutationWorkflowState:
    """1. Selects a random mutation type."""
    mutation_types = ["paraphrase", "add_negation",
                      "synonym_swap", "split_prompt"]
    selected_type = random.choice(mutation_types)
    print(f"  [Mutation Subgraph] Selected mutation: {selected_type}")
    return {"mutation_type": selected_type}

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


def run_mutation_action_subgraph(state: MutationEngineState) -> MutationEngineState:
    """3c. Invokes the mutation subgraph."""
    print("--- [Mutation Engine] Running Action: MUTATE ---")
    prompt_to_mutate: BasePrompt = state['selected_prompt']

    subgraph_output: MutationWorkflowState = mutation_action_subgraph.invoke(
        {"prompt_to_mutate": prompt_to_mutate}
    )
    return {"newly_generated_prompt": subgraph_output["mutated_prompt"]}
