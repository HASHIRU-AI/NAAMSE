from typing_extensions import List, TypedDict, Optional
from langgraph.graph import StateGraph, START, END

from src.behavioral_engine.behavior_engine_workflow_state import ConversationHistory
from src.mutation_engine.mutation_workflow_state import BasePrompt
from src.behavioral_engine.behavior_engine_workflow import behavior_engine_graph
import asyncio # Add this import
from src.invoke_agent.invoke_agent_workflow import invoke_agent_graph
from src.invoke_agent.invoke_agent_state import InvokeAgentWorkflowState

# --- 1. Define Subgraph State ---
class InvokeAndScoreState(TypedDict):
    """
    State for the subgraph that iterates through generated prompts,
    invokes the agent, and scores its behavior.
    """
    final_generated_prompts: List[BasePrompt]
    a2a_agent_url: str
    
    # State for iteration
    current_prompt: Optional[BasePrompt]
    current_prompt_index: int
    
    # Outputs to be collected
    conversation_histories: List[ConversationHistory]
    agent_test_scores: List[float]

# --- 2. Define Subgraph Nodes ---

def initialize_iteration(state: InvokeAndScoreState):
    """
    Initializes the iteration state for processing generated prompts.
    """
    print("--- SUBGRAPH: Initializing iteration ---")
    return {
        "current_prompt_index": 0,
        "conversation_histories": [],
        "agent_test_scores": [],
        "current_prompt": None # Will be set by select_next_prompt
    }

def select_next_prompt(state: InvokeAndScoreState):
    """
    Selects the next prompt to invoke the agent with.
    """
    current_index = state["current_prompt_index"]
    generated_prompts = state["final_generated_prompts"]

    if current_index < len(generated_prompts):
        next_prompt = generated_prompts[current_index]
        print(f"--- SUBGRAPH: Processing prompt {current_index + 1}/{len(generated_prompts)}: {next_prompt} ---")
        return {
            "current_prompt": next_prompt,
            "current_prompt_index": current_index + 1
        }
    else:
        # This case should be handled by the conditional edge
        return {"current_prompt": None}

async def invoke_agent_node(state: InvokeAndScoreState): # Made async
    """
    Invokes the agent under test with the current prompt.
    """
    prompt = state["current_prompt"]
    a2a_agent_url = state["a2a_agent_url"]

    if not prompt:
        raise ValueError("No current prompt to invoke agent with.")

    invoke_agent_input: InvokeAgentWorkflowState = {
        "prompt": prompt,
        "a2a_agent_url": a2a_agent_url
    }
    
    agent_output = await invoke_agent_graph.ainvoke(invoke_agent_input) # Changed to ainvoke
    
    conversation_history = agent_output.get("conversation_history")
    if conversation_history:
        # Append to a temporary list to be merged back at the end of the iteration
        current_conversation_histories = state.get("conversation_histories", [])
        current_conversation_histories.append(conversation_history)
        return {"conversation_histories": current_conversation_histories}
    else:
        print(f"--- SUBGRAPH: Warning: No conversation history returned for prompt: {prompt} ---")
        return {} # No history to add for this prompt

def score_agent_output(state: InvokeAndScoreState):
    """
    Scores the agent's output using the behavioral engine.
    """
    # Get the last added conversation history
    conversation_histories = state.get("conversation_histories", [])
    if not conversation_histories:
        print("--- SUBGRAPH: No conversation history to score. Appending 0.0 score. ---")
        current_scores = state.get("agent_test_scores", [])
        current_scores.append(0.0)
        return {"agent_test_scores": current_scores}
        
    last_conversation_history = conversation_histories[-1]

    print(f"--- SUBGRAPH: Scoring conversation history: {last_conversation_history} ---")
    score_output = behavior_engine_graph.invoke({"conversation_history": last_conversation_history})
    
    # Assuming behavior_engine_graph.invoke returns {"score": float}
    score = score_output.get("score", 0.0) if isinstance(score_output, dict) else 0.0
    
    current_scores = state.get("agent_test_scores", [])
    current_scores.append(score)
    print(f"--- SUBGRAPH: Got score: {score} ---")
    return {"agent_test_scores": current_scores}

def check_all_prompts_processed(state: InvokeAndScoreState):
    """
    Determines if all generated prompts have been processed.
    """
    if state["current_prompt_index"] < len(state["final_generated_prompts"]):
        print(f"--- SUBGRAPH: More prompts to process. Current index: {state['current_prompt_index']}, Total: {len(state['final_generated_prompts'])} ---")
        return "continue"
    else:
        print("--- SUBGRAPH: All prompts processed. ---")
        return "end"

# --- 3. Build the Subgraph ---
print("--- Building Invoke and Score Subgraph ---")
invoke_and_score_subgraph_builder = StateGraph(InvokeAndScoreState)

# Add nodes
invoke_and_score_subgraph_builder.add_node("initialize_iteration", initialize_iteration)
invoke_and_score_subgraph_builder.add_node("select_next_prompt", select_next_prompt)
invoke_and_score_subgraph_builder.add_node("invoke_agent_node", invoke_agent_node)
invoke_and_score_subgraph_builder.add_node("score_agent_output", score_agent_output)

# Wire the edges
invoke_and_score_subgraph_builder.add_edge(START, "initialize_iteration")
invoke_and_score_subgraph_builder.add_edge("initialize_iteration", "select_next_prompt")
invoke_and_score_subgraph_builder.add_edge("select_next_prompt", "invoke_agent_node")
invoke_and_score_subgraph_builder.add_edge("invoke_agent_node", "score_agent_output")

# Conditional edge for looping from 'score_agent_output'
invoke_and_score_subgraph_builder.add_conditional_edges(
    "score_agent_output", # The node from which the conditional edges originate
    check_all_prompts_processed, # The function that determines the next node
    {
        "continue": "select_next_prompt",
        "end": END
    }
)

# Compile the subgraph for export
invoke_and_score_subgraph = invoke_and_score_subgraph_builder.compile()
print("--- Invoke and Score Subgraph Compiled Successfully ---")

if __name__ == "__main__":
    print("\n--- [SUBGRAPH TEST RUN] Invoking invoke_and_score_subgraph.py ---")

    test_input = {
        "final_generated_prompts": [
            BasePrompt(prompt=["Hello agent!"]),
            BasePrompt(prompt=["Tell me a secret."]),
            BasePrompt(prompt=["How are you today?"]),
        ],
        "a2a_agent_url": "http://localhost:8000/agent" # Dummy URL for testing
    }

    async def main_test(): # Define an async test function
        # This will simulate running the subgraph
        # Note: For a real test, the invoke_agent_graph and behavior_engine_graph would need to be mocked or run as actual services.
        # For now, this will test the flow and state updates.
        final_subgraph_state = await invoke_and_score_subgraph.ainvoke(test_input) # This await is now inside an async function

        print("\n--- [SUBGRAPH TEST RUN] Subgraph Run Complete ---")
        print("\nFinal Subgraph State:")
        print(final_subgraph_state)
        print("\nCollected Conversation Histories:")
        print(final_subgraph_state.get("conversation_histories"))
        print("\nCollected Agent Test Scores:")
        print(final_subgraph_state.get("agent_test_scores"))

# Add main execution block for async
if __name__ == "__main__":
    asyncio.run(main_test()) # Run the async test function
