"""Fixed referee: re-score every arm's prompts with one judge so arms are comparable.

Keeps a Jev-driven run from being graded by Jev. When the referee is the same
backend as the fitness judge, the fitness judgment is reused instead of paying
for a second identical call. With gated=True, harm judges are skipped on refusals
(see gated_judge.py).
"""
import asyncio
import logging
from typing import Any, Dict, List

from src.behavioral_engine.behavior_engine_workflow import behavior_engine_graph
from src.experiments.ablation_config import AblationConfig
from src.experiments.coverage import summarize_judge_results
from src.experiments.gated_judge import gated_behavior_engine_graph, judge_calls

logger = logging.getLogger(__name__)


async def _referee_one(prompt: Dict[str, Any], referee: str, semaphore: asyncio.Semaphore,
                       gated: bool = False) -> Dict[str, Any]:
    history = prompt.get("conversation_history") or prompt.get("conversation")
    if not history:
        return {"score": None, "judge_results": None, "reused_fitness_judgment": False, "error": "no conversation"}
    async with semaphore:
        try:
            graph = gated_behavior_engine_graph if gated else behavior_engine_graph
            result = await graph.ainvoke(
                {"conversation_history": history, "is_score_flipped": False},
                config={"configurable": {"judge_backend": referee}},
            )
        except Exception as e:  # keep the run's other results if one referee call fails
            logger.error("Referee call failed: %s", e)
            return {"score": None, "judge_results": None, "reused_fitness_judgment": False, "error": str(e)}
    return {
        "score": result.get("final_score"),
        "judge_results": summarize_judge_results(result.get("judge_results"), result.get("ra_judge_results")),
        "reused_fitness_judgment": False,
        "judge_calls": judge_calls(result.get("judge_results")) if gated else 7,
    }


async def referee_prompts(prompts: List[Dict[str, Any]], ablation: AblationConfig,
                          concurrency: int = 8, gated: bool = False) -> List[Dict[str, Any]]:
    """Return one referee record per prompt, in order."""
    referee = ablation.referee_judge
    if referee == "none":
        return [{"score": None, "judge_results": None, "reused_fitness_judgment": False} for _ in prompts]
    if referee == ablation.fitness_judge:
        return [{
            "score": p.get("metadata", {}).get("judge_score"),
            "judge_results": p.get("metadata", {}).get("judge_results"),
            "reused_fitness_judgment": True,
        } for p in prompts]
    semaphore = asyncio.Semaphore(concurrency)
    return await asyncio.gather(*(_referee_one(p, referee, semaphore, gated) for p in prompts))
