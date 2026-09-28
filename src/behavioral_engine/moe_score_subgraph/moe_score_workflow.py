import threading

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from src.behavioral_engine.moe_score_subgraph.more_score_state import MOESubgraphState
from src.behavioral_engine.moe_score_subgraph.nodes.create_llm_judges import create_judge_node
from src.behavioral_engine.moe_score_subgraph.nodes.aggregate_score import aggregate_scores
from src.behavioral_engine.moe_score_subgraph.nodes.jev_judge_node import create_jev_judge_node
from src.behavioral_engine.moe_score_subgraph.llm_judges.gemini_judge import GeminiJudge
from src.behavioral_engine.moe_score_subgraph.llm_judges.meta_judge import MetaJudge
from src.behavioral_engine.moe_score_subgraph.llm_judges.jev_judge import JevJudge
from src.behavioral_engine.moe_score_subgraph.nodes.response_alignment_judge_node import create_response_alignment_judge_node
from src.behavioral_engine.moe_score_subgraph.moe_score_judge_prompts import eval_type_to_prompt, EvalType
from src.experiments.ablation_config import get_ablation

# Judge backends selectable per run (ablation --fitness-judge / referee)
JUDGE_CLASSES = {
    "gemini": GeminiJudge,
    "meta": MetaJudge,
    "jev": JevJudge,
}

_judge_node_cache = {}
_judge_node_cache_lock = threading.Lock()


def resolve_judge_backend(config: RunnableConfig) -> str:
    """Backend for this invocation: an explicit override (used by the referee), else the ablation's fitness judge."""
    configurable = (config or {}).get("configurable", {})
    return configurable.get("judge_backend") or get_ablation(config).fitness_judge


def get_judge_node(backend: str, eval_type: EvalType):
    """Build (once) and return the node function for a backend and eval type"""
    key = (backend, eval_type)
    with _judge_node_cache_lock:
        if key not in _judge_node_cache:
            judge = JUDGE_CLASSES[backend](judge_id=eval_type.value, eval_type=eval_type)
            judge.set_system_prompt(eval_type_to_prompt[eval_type])
            if backend == "jev":
                node = create_jev_judge_node(judge)
            elif eval_type == EvalType.RESPONSE_ALIGNMENT:
                node = create_response_alignment_judge_node(judge)
            else:
                node = create_judge_node(judge)
            _judge_node_cache[key] = node
        return _judge_node_cache[key]


def create_dispatch_node(eval_type: EvalType):
    """Node that routes one eval type to the judge backend chosen for this run"""
    def dispatch_node(state: MOESubgraphState, config: RunnableConfig):
        return get_judge_node(resolve_judge_backend(config), eval_type)(state)
    return dispatch_node


main_graph_builder = StateGraph(MOESubgraphState)

judge_node_names = []
for eval_type in eval_type_to_prompt:
    node_name = f"MOEJudge_{eval_type.value}"
    judge_node_names.append(node_name)
    main_graph_builder.add_node(node_name, create_dispatch_node(eval_type))

main_graph_builder.add_node("AggregateScores", aggregate_scores)

# Define Edges
for node_name in judge_node_names:
    main_graph_builder.add_edge(START, node_name)
    main_graph_builder.add_edge(node_name, "AggregateScores")
main_graph_builder.add_edge("AggregateScores", END)

moe_score_graph = main_graph_builder.compile()
