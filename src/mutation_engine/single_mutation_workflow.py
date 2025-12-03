from langgraph.graph import StateGraph, START, END

from src.mutation_engine.mutation_workflow_state import MutationEngineState
from src.mutation_engine.nodes.select_prompt_by_probability import select_prompt_by_probability
from src.mutation_engine.nodes.decide_action_by_score import decide_action_by_score
from src.mutation_engine.nodes.run_explore_action import run_explore_action
from src.mutation_engine.nodes.run_similar_action import run_similar_action
from src.mutation_engine.nodes.run_mutation_action_subgraph import run_mutation_action_subgraph
from src.mutation_engine.nodes.add_prompt_to_output_list import add_prompt_to_output_list


def initialize_single_mutation(state: MutationEngineState):
    scores = [p["score"] for p in state["input_prompts"]]
    if not scores or all(s == 0 for s in scores):
        probabilities = [1.0 / len(state["input_prompts"])] * len(state["input_prompts"])
    else:
        total = sum(scores)
        probabilities = [s / total for s in scores]
    return {"prompt_probabilities": probabilities, "final_generated_prompts": []}


builder = StateGraph(MutationEngineState)

builder.add_node("initialize", initialize_single_mutation)
builder.add_node("select_prompt", select_prompt_by_probability)
builder.add_node("decide_action", decide_action_by_score)
builder.add_node("action_explore", run_explore_action)
builder.add_node("action_similar", run_similar_action)
builder.add_node("action_mutate", run_mutation_action_subgraph)
builder.add_node("add_to_output", add_prompt_to_output_list)

builder.add_edge(START, "initialize")
builder.add_edge("initialize", "select_prompt")
builder.add_edge("select_prompt", "decide_action")
builder.add_conditional_edges(
    "decide_action",
    lambda state: state["action_to_take"],
    {"explore": "action_explore", "similar": "action_similar", "mutate": "action_mutate"}
)
builder.add_edge("action_explore", "add_to_output")
builder.add_edge("action_similar", "add_to_output")
builder.add_edge("action_mutate", "add_to_output")
builder.add_edge("add_to_output", END)

single_mutation_graph = builder.compile()
