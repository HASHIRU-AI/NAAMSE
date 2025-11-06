from typing_extensions import List, Tuple, Literal, TypedDict, Dict
from langgraph.graph import StateGraph, START, END

from src.behavioral_engine.nodes.sanitize_inputs import sanitize_inputs
from src.behavioral_engine.nodes.calculate_pii_score import calculate_pii_score
from src.behavioral_engine.nodes.calculate_moe_score import calculate_moe_score
from src.behavioral_engine.nodes.calculate_final_score import calculate_final_score
from src.behavioral_engine.nodes.translate_inputs import translate_inputs
from src.behavioral_engine.nodes.decode_inputs import decode_inputs
from src.behavioral_engine.nodes.convert_to_ascii import convert_to_ascii
class BehaviorEngineWorkflowState(TypedDict):
    """
    This is the "interface" for the main mutation engine.
    It defines the inputs and outputs the parent graph will interact with.
    """
    # Inputs
    conversation_history: List[Dict[str, str]]

    # Internal state
    decoded_conversation_history: List[Dict[str, str]]
    translated_conversation_history: List[Dict[str, str]]
    sanitized_conversation_history: List[Dict[str, str]]
    pii_score: int
    moe_score: int

    #outputs
    final_score: int
    

# --- Build the Main Mutation Engine Graph ---
main_graph_builder = StateGraph(BehaviorEngineWorkflowState)

# Add nodes
main_graph_builder.add_node("FixEncodingAndDecodeLayer", decode_inputs)
main_graph_builder.add_node("LanguageTranslationLayer", translate_inputs)
main_graph_builder.add_node("ASCIIConversionLayer", convert_to_ascii)
main_graph_builder.add_node("PIIScoreLayer", calculate_pii_score)
main_graph_builder.add_node("MOEScoreLayer", calculate_moe_score)
main_graph_builder.add_node("FinalScoreLayer", calculate_final_score)

# Define Edges
main_graph_builder.add_edge(START, "FixEncodingAndDecodeLayer")
main_graph_builder.add_edge("FixEncodingAndDecodeLayer", "LanguageTranslationLayer")
main_graph_builder.add_edge("LanguageTranslationLayer", "ASCIIConversionLayer")  
main_graph_builder.add_edge("ASCIIConversionLayer", "PIIScoreLayer")
main_graph_builder.add_edge("ASCIIConversionLayer", "MOEScoreLayer")
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