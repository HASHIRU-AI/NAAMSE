from langgraph.graph import StateGraph, START, END

from src.behavioral_engine.nodes.sanitize_inputs import sanitize_inputs
from src.behavioral_engine.nodes.calculate_pii_score import calculate_pii_score
from src.behavioral_engine.nodes.calculate_moe_score import calculate_moe_score
from src.behavioral_engine.nodes.calculate_final_score import calculate_final_score
from src.behavioral_engine.nodes.translate_inputs import translate_inputs
from src.behavioral_engine.nodes.decode_inputs import decode_inputs
from src.behavioral_engine.nodes.convert_to_ascii import convert_to_ascii
from src.behavioral_engine.moe_score_subgraph.moe_score_workflow import moe_score_graph
from src.behavioral_engine.behavior_engine_workflow_state import BehaviorEngineWorkflowState


# --- Build the Main Mutation Engine Graph ---
main_graph_builder = StateGraph(BehaviorEngineWorkflowState)

# Add nodes
main_graph_builder.add_node("FixEncodingAndDecodeLayer", decode_inputs)
main_graph_builder.add_node("LanguageTranslationLayer", translate_inputs)
main_graph_builder.add_node("ASCIIConversionLayer", convert_to_ascii)
main_graph_builder.add_node("PIIScoreLayer", calculate_pii_score)
main_graph_builder.add_node("MOEScoreLayer", moe_score_graph)
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
