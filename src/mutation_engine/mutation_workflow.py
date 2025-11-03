# mutation_engine_graph.py

from typing_extensions import List, Tuple, Literal, TypedDict
from langgraph.graph import StateGraph, START, END

from src.mutation_engine.nodes.calculate_probabilities import calculate_probabilities
from src.mutation_engine.nodes.select_prompt_by_probability import select_prompt_by_probability
from src.mutation_engine.nodes.decide_action_by_score import decide_action_by_score
from src.mutation_engine.nodes.run_explore_action import run_explore_action
from src.mutation_engine.nodes.run_similar_action import run_similar_action
from src.mutation_engine.nodes.run_mutation_action_subgraph import run_mutation_action_subgraph
from src.mutation_engine.nodes.add_prompt_to_output_list import add_prompt_to_output_list
from src.mutation_engine.nodes.should_continue_loop import should_continue_loop


# --- 2. Define Main Mutation Engine Graph ---

class MutationEngineState(TypedDict):
    """
    This is the "interface" for the main mutation engine.
    It defines the inputs and outputs the parent graph will interact with.
    """
    # Inputs
    input_prompts: List[Tuple[str, float]]
    n_to_generate: int

    # Internal state
    prompt_probabilities: List[float]
    selected_prompt: Tuple[str, float]
    action_to_take: Literal["mutate", "similar", "explore"]
    newly_generated_prompt: str

    # Output
    final_generated_prompts: List[str]


# --- Build the Main Mutation Engine Graph ---
main_graph_builder = StateGraph(MutationEngineState)

# Add nodes
main_graph_builder.add_node("calculate_probabilities", calculate_probabilities)
main_graph_builder.add_node("select_prompt", select_prompt_by_probability)
main_graph_builder.add_node("decide_action", decide_action_by_score)
main_graph_builder.add_node("action_explore", run_explore_action)
main_graph_builder.add_node("action_similar", run_similar_action)
main_graph_builder.add_node(
    "action_mutate", run_mutation_action_subgraph)  # Subgraph as node
main_graph_builder.add_node("add_to_output", add_prompt_to_output_list)

# Define Edges
main_graph_builder.add_edge(START, "calculate_probabilities")
main_graph_builder.add_edge("calculate_probabilities", "select_prompt")
main_graph_builder.add_edge("select_prompt", "decide_action")

main_graph_builder.add_conditional_edges(
    "decide_action",
    lambda state: state["action_to_take"],
    {"explore": "action_explore", "similar": "action_similar", "mutate": "action_mutate"}
)

main_graph_builder.add_edge("action_explore", "add_to_output")
main_graph_builder.add_edge("action_similar", "add_to_output")
main_graph_builder.add_edge("action_mutate", "add_to_output")

main_graph_builder.add_conditional_edges(
    "add_to_output",
    should_continue_loop,
    {"select_prompt": "select_prompt", END: END}
)

# --- Compile the graph for export ---
mutation_engine_graph = main_graph_builder.compile()


# --- Test block (to run this file directly) ---
if __name__ == "__main__":
    print(
        "--- [TEST RUN] Compiling and running mutation_engine_graph.py independently ---")

    initial_prompts = [
        ("low_score_prompt", 0.3),
        ("mid_score_prompt", 0.7),
        ("high_score_prompt", 0.9)
    ]
    num_to_gen = 3

    final_state = mutation_engine_graph.invoke({
        "input_prompts": initial_prompts,
        "n_to_generate": num_to_gen
    })

    print("\n--- [TEST RUN] Graph execution complete ---")
    print(f"Generated {len(final_state['final_generated_prompts'])} prompts:")
    for i, prompt in enumerate(final_state['final_generated_prompts']):
        print(f"  {i+1}: {prompt}")
