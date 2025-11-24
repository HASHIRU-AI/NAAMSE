from typing_extensions import List, TypedDict, Optional
from langgraph.graph import StateGraph, START, END

# --- Import subgraphs and their states ---
from src.behavioral_engine.behavior_engine_workflow_state import ConversationHistory
from src.mutation_engine.mutation_workflow_state import BasePrompt, ScoredPrompt
from src.mutation_engine.mutation_workflow import mutation_engine_graph
from src.agent.invoke_and_score_subgraph import invoke_and_score_subgraph, InvokeAndScoreState
import asyncio

# --- 1. Define Parent Graph State (now a subgraph state) ---

class FuzzerGraphState(TypedDict):
    """
    This is the state for the fuzzer iteration subgraph.

    It must contain all keys needed by the mutation and invoke_and_score subgraphs.
    It will also receive the output keys for generated and scored prompts.
    """
    # Inputs for the mutation engine
    input_prompts: List[ScoredPrompt]
    n_to_generate: int
    a2a_agent_url: str

    # Output from the mutation engine
    final_generated_prompts: List[BasePrompt]

    # Output from the invoke and score subgraph
    conversation_histories: List[ConversationHistory]
    agent_test_scores: Optional[List[float]]
    scored_mutations: List[ScoredPrompt] # New: The results of scoring the generated_mutations


# --- 2. Define Parent Graph Nodes ---

def setup_run(state: FuzzerGraphState):
    """
    This node sets up the state for a single fuzzer iteration.
    It initializes lists that will be populated during the iteration.
    """
    print(f"--- FUZZER ITERATION SUBGRAPH: Setting up for {state['n_to_generate']} mutations ---")
    return {
        # Ensure outputs are initialized
        "final_generated_prompts": [],
        "agent_test_scores": [],
        "conversation_histories": [],
        "scored_mutations": []
    }

async def invoke_and_score_node(state: FuzzerGraphState):
    """
    This node acts as a proxy to the invoke_and_score_subgraph.
    It passes the necessary state to the subgraph and collects its outputs.
    """
    print("--- FUZZER ITERATION SUBGRAPH: Invoking and Scoring Mutated Prompts ---")

    subgraph_input: InvokeAndScoreState = {
        "final_generated_prompts": state["final_generated_prompts"],
        "a2a_agent_url": state["a2a_agent_url"],
        "current_prompt_index": 0,
        "current_prompt": None,
        "conversation_histories": [],
        "agent_test_scores": []
    }
    
    subgraph_output = await invoke_and_score_subgraph.ainvoke(subgraph_input)
    
    # Combine generated prompts with their scores
    scored_mutations = []
    if state["final_generated_prompts"] and subgraph_output.get("agent_test_scores"):
        for i, prompt_dict in enumerate(state["final_generated_prompts"]):
            if i < len(subgraph_output["agent_test_scores"]):
                scored_mutations.append(ScoredPrompt(prompt=prompt_dict["prompt"], score=subgraph_output["agent_test_scores"][i]))

    return {
        "agent_test_scores": subgraph_output.get("agent_test_scores", []),
        "conversation_histories": subgraph_output.get("conversation_histories", []),
        "scored_mutations": scored_mutations
    }


# --- 3. Build the Fuzzer Iteration Subgraph ---

print("--- Building Fuzzer Iteration Subgraph ---")
parent_builder = StateGraph(FuzzerGraphState) # Keep name as parent_builder for consistency

# Add the nodes
parent_builder.add_node("setup_run", setup_run)
parent_builder.add_node("mutation_engine", mutation_engine_graph)
parent_builder.add_node("invoke_and_score_prompts", invoke_and_score_node)


# --- 4. Wire the Fuzzer Iteration Subgraph Edges ---

parent_builder.add_edge(START, "setup_run")
parent_builder.add_edge("setup_run", "mutation_engine")
parent_builder.add_edge("mutation_engine", "invoke_and_score_prompts")
parent_builder.add_edge("invoke_and_score_prompts", END)


# --- 5. Compile the Fuzzer Iteration Subgraph ---
graph = parent_builder.compile() # Keep graph as the compiled object name

print("--- Fuzzer Iteration Subgraph Compiled Successfully ---")


# --- Test block (to run this file) ---
if __name__ == "__main__":
    async def main_test():
        print("\n--- [TEST RUN] Invoking fuzzer_iteration_subgraph.py ---")

        initial_input = {
            "input_prompts": [
                {"prompt": ["low_score_prompt"], "score":0.3},
                {"prompt": ["medium_score_prompt"], "score":0.5},
                {"prompt": ["high_score_prompt"], "score":0.9}
            ],
            "n_to_generate": 3,
            "a2a_agent_url": "http://localhost:5000"
        }

        final_state = await graph.ainvoke(initial_input)

        print("\n--- [TEST RUN] Fuzzer Iteration Subgraph Run Complete ---")
        print("\nFinal State:")
        print(final_state)
        print("\nScored Mutations:")
        print(final_state.get("scored_mutations"))

    asyncio.run(main_test())
