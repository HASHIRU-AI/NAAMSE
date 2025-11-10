from langgraph.graph import StateGraph, START, END
from src.behavioral_engine.moe_score_subgraph.more_score_state import MOESubgraphState
from src.behavioral_engine.moe_score_subgraph.nodes.create_llm_judges import create_judge_node
from src.behavioral_engine.moe_score_subgraph.nodes.aggregate_score import aggregate_scores
from src.behavioral_engine.moe_score_subgraph.llm_judges.gemini_judge import GeminiJudge
main_graph_builder = StateGraph(MOESubgraphState)

judges = [GeminiJudge(), GeminiJudge(model_name="gemini-2.5-flash")] 
judge_node_names = []
for judge in judges:
    node_name = f"MOEJudge_{judge.get_judge_id()}"
    judge_node_names.append(node_name)
    main_graph_builder.add_node(node_name, create_judge_node(judge))
    
main_graph_builder.add_node("AggregateScores", aggregate_scores)

# Define Edges
for node_name in judge_node_names:
    main_graph_builder.add_edge(START, node_name)
    main_graph_builder.add_edge(node_name, "AggregateScores")
main_graph_builder.add_edge("AggregateScores", END)

moe_score_graph = main_graph_builder.compile()