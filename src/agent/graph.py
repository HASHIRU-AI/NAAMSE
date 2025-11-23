# parent_graph.py

import random
from typing_extensions import List, Tuple, TypedDict, Optional
from langgraph.graph import StateGraph, START, END

# --- Import the compiled subgraph and its state ---
# This assumes 'mutation_engine_graph.py' is in the same directory.
from src.behavioral_engine.behavior_engine_workflow_state import ConversationHistory
from src.mutation_engine.mutation_workflow_state import BasePrompt, ScoredPrompt
from src.mutation_engine.mutation_workflow import mutation_engine_graph
from src.behavioral_engine.behavior_engine_workflow import behavior_engine_graph
from src.invoke_agent.invoke_agent_workflow import invoke_agent_graph
from src.invoke_agent.invoke_agent_state import InvokeAgentWorkflowState
from src.agent.invoke_and_score_subgraph import invoke_and_score_subgraph, InvokeAndScoreState
import asyncio # Add this import
# --- 1. Define Parent Graph State ---

class FuzzerGraphState(TypedDict):
    """
    This is the state for the main parent graph.

    It must contain all keys needed by the subgraph
    (input_prompts, n_to_generate)

    It will also receive the output keys
    (final_generated_prompts)
    """
    # Inputs for the mutation engine
    input_prompts: List[ScoredPrompt]
    n_to_generate: int
    a2a_agent_url: str

    # Output from the mutation engine
    final_generated_prompts: List[BasePrompt]
    conversation_histories: List[ConversationHistory]

    # Other data for the parent graph
    agent_test_scores: Optional[List[float]]


# --- 2. Define Parent Graph Nodes ---

def setup_run(state: FuzzerGraphState):
    """
    This node loads the initial seed prompts.
    For this example, we just print and assume inputs are passed at the start.
    """
    print("--- PARENT: (1) Setting up run ---")
    # In a real app, this might load data from a DB
    # and populate 'input_prompts' and 'n_to_generate'
    return {
        # Ensure outputs are initialized
        "final_generated_prompts": [],
        "agent_test_scores": [],
        "conversation_histories": []
    }

async def invoke_and_score_node(state: FuzzerGraphState): # Made async
    """
    This node acts as a proxy to the invoke_and_score_subgraph.
    It passes the necessary state to the subgraph and collects its outputs.
    """
    print("--- PARENT: (3) Invoking and Scoring Mutated Prompts ---")

    subgraph_input: InvokeAndScoreState = {
        "final_generated_prompts": state["final_generated_prompts"],
        "a2a_agent_url": state["a2a_agent_url"],
        "current_prompt_index": 0, # Subgraph initializes this
        "current_prompt": None,    # Subgraph initializes this
        "conversation_histories": [], # Subgraph initializes this
        "agent_test_scores": []       # Subgraph initializes this
    }
    
    subgraph_output = await invoke_and_score_subgraph.ainvoke(subgraph_input) # Changed to ainvoke
    
    return {
        "agent_test_scores": subgraph_output.get("agent_test_scores", []),
        "conversation_histories": subgraph_output.get("conversation_histories", [])
    }


# --- 3. Build the Parent Graph ---

print("--- Building Parent Fuzzer Graph ---")
parent_builder = StateGraph(FuzzerGraphState)

# Add the nodes
parent_builder.add_node("setup_run", setup_run)

parent_builder.add_node("mutation_engine", mutation_engine_graph)

# Add the new node for the subgraph
parent_builder.add_node("invoke_and_score_prompts", invoke_and_score_node)


# --- 4. Wire the Parent Graph Edges ---

parent_builder.add_edge(START, "setup_run")
parent_builder.add_edge("setup_run", "mutation_engine")
parent_builder.add_edge("mutation_engine", "invoke_and_score_prompts")
parent_builder.add_edge("invoke_and_score_prompts", END)


# --- 5. Compile the Parent Graph ---
graph = parent_builder.compile()

print("--- Parent Graph Compiled Successfully ---")


# --- Test block (to run this file) ---
if __name__ == "__main__":
    async def main_test(): # Define an async test function
        print("\n--- [TEST RUN] Invoking graph.py ---")

        # This is the input to the parent graph
        initial_input = {
            "input_prompts": [
                {"prompt": ["low_score_prompt"], "score":0.3},
                {"prompt": ["medium_score_prompt"], "score":0.5},
                {"prompt": ["high_score_prompt"], "score":0.9}
            ],
            "n_to_generate": 3,
            "a2a_agent_url": "http://localhost:5000"
        }

        final_state = await graph.ainvoke(initial_input) # Changed to ainvoke

        print("\n--- [TEST RUN] Parent Graph Run Complete ---")
        print("\nFinal State:")
        print(final_state)
        print("\nConversation Histories:")
        print(final_state.get("conversation_histories"))

    asyncio.run(main_test()) # Run the async test function
