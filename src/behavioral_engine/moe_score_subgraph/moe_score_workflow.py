from langgraph.graph import StateGraph, START, END
from src.behavioral_engine.moe_score_subgraph.more_score_state import MOESubgraphState
from src.behavioral_engine.moe_score_subgraph.nodes.create_llm_judges import create_judge_node
from src.behavioral_engine.moe_score_subgraph.nodes.aggregate_score import aggregate_scores
from src.behavioral_engine.moe_score_subgraph.llm_judges.gemini_judge import GeminiJudge
from src.behavioral_engine.moe_score_subgraph.llm_judges.ollama_judge import OllamaJudge
from src.behavioral_engine.moe_score_subgraph.nodes.response_alignment_judge_node import create_response_alignment_judge_node
from src.behavioral_engine.moe_score_subgraph.moe_score_judge_prompts import eval_type_to_prompt, EvalType
main_graph_builder = StateGraph(MOESubgraphState)

judges = []
for eval_type, prompt in eval_type_to_prompt.items():
    judge = GeminiJudge(judge_id=eval_type.value, eval_type=eval_type)
    judge.set_system_prompt(prompt)
    judges.append(judge)

judge_node_names = []
for judge in judges:
    node_name = f"MOEJudge_{judge.get_judge_id()}"
    judge_node_names.append(node_name)
    if judge.get_eval_type() == EvalType.RESPONSE_ALIGNMENT:
        main_graph_builder.add_node(node_name, create_response_alignment_judge_node(judge))
    else:
        main_graph_builder.add_node(node_name, create_judge_node(judge))
    
main_graph_builder.add_node("AggregateScores", aggregate_scores)

# Define Edges
for node_name in judge_node_names:
    main_graph_builder.add_edge(START, node_name)
    main_graph_builder.add_edge(node_name, "AggregateScores")
main_graph_builder.add_edge("AggregateScores", END)

moe_score_graph = main_graph_builder.compile()