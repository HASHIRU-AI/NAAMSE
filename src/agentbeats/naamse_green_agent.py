"""
NAAMSE Green Agent - Evaluator agent that calls the NAAMSE LangGraph fuzzer.
"""
import json

from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.utils import new_agent_text_message

from src.agentbeats.models import NAAMSERequest
from src.agentbeats.green_executor import GreenAgent
from src.agent.graph import graph


class NAAMSEGreenAgent(GreenAgent):
    """
    NAAMSE Green Agent implementation.
    
    Receives a NAAMSERequest, runs the fuzzer, returns result.
    """

    def validate_request(self, request: NAAMSERequest) -> tuple[bool, str]:
        """Validate the incoming request."""
        if not str(request.target_url).startswith(("http://", "https://")):
            return False, f"Invalid target URL: {request.target_url}"
        
        if request.iterations_limit < 1:
            return False, "iterations_limit must be at least 1"
        
        return True, "Request is valid"

    async def run_eval(self, request: NAAMSERequest, updater: TaskUpdater) -> None:
        """Run the NAAMSE fuzzer evaluation."""
        target_url = str(request.target_url)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Running NAAMSE fuzzer against {target_url} with {request.iterations_limit} iterations...",
                context_id=updater.context_id
            )
        )

        # Build the fuzzer input state
        initial_prompts = [p.model_dump() for p in request.initial_prompts] if request.initial_prompts else [
            {"prompt": ["seed prompt"], "score": 0.0}
        ]

        fuzzer_input = {
            "iterations_limit": request.iterations_limit,
            "mutations_per_iteration": request.mutations_per_iteration,
            "score_threshold": request.score_threshold,
            "a2a_agent_url": target_url,
            "input_prompts_for_iteration": initial_prompts
        }

        try:
            final_state = await graph.ainvoke(fuzzer_input)
        except Exception as e:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Fuzzer error: {e}", context_id=updater.context_id)
            )
            raise

        # Extract results
        all_prompts = final_state.get("all_fuzzer_prompts_with_scores", [])
        
        # Calculate summary statistics and cluster averages in one pass
        total_score = 0
        max_score = 0
        high_score_count = 0
        
        # Track per-cluster: {cluster_label: {"sum": x, "count": y, "max": z}}
        cluster_stats: dict[str, dict[str, float]] = {}
        
        for prompt in all_prompts:
            score = prompt.get("score", 0)
            total_score += score
            max_score = max(max_score, score)
            if score >= request.score_threshold:
                high_score_count += 1
            
            # Get cluster label
            cluster_info = prompt.get("cluster_info", {})
            cluster_label = cluster_info.get("cluster_label")
            
            if cluster_label not in cluster_stats:
                cluster_stats[cluster_label] = {"sum": 0, "count": 0, "max": 0}
            
            cluster_stats[cluster_label]["sum"] += score
            cluster_stats[cluster_label]["count"] += 1
            cluster_stats[cluster_label]["max"] = max(cluster_stats[cluster_label]["max"], score)
        
        avg_score = total_score / len(all_prompts) if all_prompts else 0
        
        # Build cluster report sorted by avg score descending
        cluster_report = [
            {
                "cluster": label,
                "count": int(stats["count"]),
                "avg_score": round(stats["sum"] / stats["count"], 2),
                "max_score": round(stats["max"], 2),
            }
            for label, stats in cluster_stats.items()
        ]
        cluster_report.sort(key=lambda x: -x["avg_score"])

        print(f"Fuzzer completed. Tested {len(all_prompts)} prompts.")
        print(f"Max score: {max_score}, Avg score: {avg_score}, High scoring prompts: {high_score_count}")
        print(f"Cluster breakdown: {len(cluster_report)} clusters")
        
        result = {
            "total_prompts_tested": len(all_prompts),
            "max_score": max_score,
            "avg_score": round(avg_score, 2),
            "high_score_count": high_score_count,
            "cluster_report": cluster_report,
            "all_prompts": all_prompts
        }
        
        # Pretty print for terminal readability
        print(json.dumps(result, indent=2))
        
        # Send final result
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(json.dumps(result, indent=2), context_id=updater.context_id)
        )

