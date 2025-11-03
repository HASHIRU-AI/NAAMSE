from typing_extensions import List, Tuple, Literal, TypedDict
from langgraph.graph import StateGraph, START, END

from src.behavioral_engine.nodes.sanitize_inputs import sanitize_inputs
from src.behavioral_engine.nodes.calculate_pii_score import calculate_pii_score
from src.behavioral_engine.nodes.calculate_moe_score import calculate_moe_score
from src.behavioral_engine.nodes.calculate_final_score import calculate_final_score
class BehaviorEngineWorkflowState(TypedDict):
    """
    This is the "interface" for the main mutation engine.
    It defines the inputs and outputs the parent graph will interact with.
    """
    # Inputs
    conversation_history: List[str]
    input_prompt: str

    # Internal state
    sanitized_conversation_history: List[str]
    sanitized_input_prompt: str
    pii_score: int
    moe_score: int

    #outputs
    final_score: int
    

# --- Build the Main Mutation Engine Graph ---
main_graph_builder = StateGraph(BehaviorEngineWorkflowState)

# Add nodes
main_graph_builder.add_node("SanitizationLayer", sanitize_inputs)
main_graph_builder.add_node("PIIScoreLayer", calculate_pii_score)
main_graph_builder.add_node("MOEScoreLayer", calculate_moe_score)
main_graph_builder.add_node("FinalScoreLayer", calculate_final_score)

# Define Edges
main_graph_builder.add_edge(START, "SanitizationLayer")
main_graph_builder.add_edge("SanitizationLayer", "PIIScoreLayer")
main_graph_builder.add_edge("SanitizationLayer", "MOEScoreLayer")
main_graph_builder.add_edge("PIIScoreLayer", "FinalScoreLayer")
main_graph_builder.add_edge("MOEScoreLayer", "FinalScoreLayer")
main_graph_builder.add_edge("FinalScoreLayer", END)

# --- Compile the graph for export ---
behavior_engine_graph = main_graph_builder.compile()


# --- Test block (to run this file directly) ---
if __name__ == "__main__":
    print(
        "--- [TEST RUN] Compiling and running behavior_engine_graph.py independently ---")

    conversation_history = [
        "Hello, how are you today?",
    ]
    input_prompt = "What is your name?"

    final_state = behavior_engine_graph.invoke({
        "ConversationHistory": conversation_history,
        "input_prompt": input_prompt
    })

    print("\n--- [TEST RUN] Graph execution complete ---")
    print(f"Final score: {final_state['final_score']}")