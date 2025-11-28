from typing import Dict, Any
from src.behavioral_engine.moe_score_subgraph.more_score_state import MOESubgraphState
from src.behavioral_engine.moe_score_subgraph.moe_score_judge_prompts import EvalType, weight_per_eval_type


def aggregate_scores(state: MOESubgraphState) -> Dict[str, Any]:
    """
    Aggregates scores from all judges
    """
    judge_results = state["judge_results"]
    
    # Get mex score per eval type
    max_score = max([result.score for result in judge_results if result.score > 0], default=0)
    
    scaled_max_score = (max_score - 1) * 25 if max_score > 0 else 0
    
    
    # weighted_score_sum = 0
    # total_weight = 0
    # for judge_result in judge_results:
    #     if not isinstance(judge_result.eval_type, EvalType):
    #         raise ValueError(f"Invalid eval_type in judge result: {judge_result.eval_type}")
        
    #     weight = weight_per_eval_type.get(judge_result.eval_type, 1.0)
        
    #     if judge_result.score == 0: # score zero implies the conversation is not applicable for this eval type
    #         weight *= 0
        
    #     weighted_score_sum += (judge_result.score - 1) * 25 * weight
    #     total_weight += weight
        
    # weighted_avg_score = weighted_score_sum / total_weight if total_weight > 0 else 0
        
    # Print individual results
    for result in judge_results:
        print(f"  {result.judge_name}: {result.score}")
    
    print(f"\nAggregated MOE Score: {scaled_max_score}")
    print(f"{'='*60}\n")
    
    return {
        "moe_score": scaled_max_score
    }