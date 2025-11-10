from typing import Dict, Any
from src.behavioral_engine.moe_score_subgraph.more_score_state import MOESubgraphState


def aggregate_scores(state: MOESubgraphState) -> Dict[str, Any]:
    """
    Aggregates scores from all judges
    """
    judge_results = state["judge_results"]
    aggregated_moe_score = sum(result.score for result in judge_results) / len(judge_results) if judge_results else 0
    
    # Print individual results
    for result in judge_results:
        print(f"  {result.judge_name}: {result.score}")
    
    print(f"\nAggregated MOE Score: {aggregated_moe_score}")
    print(f"{'='*60}\n")
    
    return {
        "moe_score": aggregated_moe_score
    }