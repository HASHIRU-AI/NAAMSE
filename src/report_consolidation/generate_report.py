import json
from typing import Any, Dict, List

from ..cluster_engine.utilities import get_cluster_info_for_prompt


def generate_report_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a consolidated report from the fuzzer's final state.
    This function is designed to be a node in a LangGraph.
    """
    all_prompts: List[Dict[str, Any]] = state.get("all_fuzzer_prompts_with_scores", [])
    request_score_threshold = state.get(
        "score_threshold", 0.0)  # Extract from state

    # Calculate summary statistics and cluster averages in one pass
    total_score = 0
    max_score = 0
    high_score_count = 0

    # Track per-cluster: {cluster_label: {"sum": x, "count": y, "max": z, "description": ""}}
    cluster_stats: dict[str, dict[str, Any]] = {}

    for prompt in all_prompts:
        score = prompt.get("score", 0)
        total_score += score
        max_score = max(max_score, score)
        if score >= request_score_threshold:
            high_score_count += 1

        # Get cluster label from metadata
        metadata = prompt.get("metadata", {})
        cluster_info = metadata.get("cluster_info", {})
        cluster_label = cluster_info.get("cluster_label")
        description = cluster_info.get("description", "")

        if not cluster_label:
            # Try to get cluster info using the utility function
            prompt_text = prompt.get("prompt", "")
            if prompt_text:
                fetched_info = get_cluster_info_for_prompt(prompt_text)
                if fetched_info:
                    cluster_label = fetched_info.get("label")
                    description = fetched_info.get("description", "")

        if cluster_label:
            if cluster_label not in cluster_stats:
                cluster_stats[cluster_label] = {"sum": 0, "count": 0, "max": 0, "description": description}
            elif not cluster_stats[cluster_label]["description"]:
                cluster_stats[cluster_label]["description"] = description

            cluster_stats[cluster_label]["sum"] += score
            cluster_stats[cluster_label]["count"] += 1
            cluster_stats[cluster_label]["max"] = max(
                cluster_stats[cluster_label]["max"], score)

    avg_score = total_score / len(all_prompts) if all_prompts else 0

    # Build cluster report sorted by avg score descending
    cluster_report = [
        {
            "cluster": label,
            "count": int(stats["count"]),
            "avg_score": round(stats["sum"] / stats["count"], 2),
            "max_score": round(stats["max"], 2),
            "description": stats["description"]
        }
        for label, stats in cluster_stats.items()
    ]
    cluster_report.sort(key=lambda x: -x["avg_score"])

    # print statements for terminal readability can be kept here or moved to the agent
    print(f"Fuzzer completed. Tested {len(all_prompts)} prompts.")
    print(
        f"Max score: {max_score}, Avg score: {avg_score}, High scoring prompts: {high_score_count}")
    print(f"Cluster breakdown: {len(cluster_report)} clusters")

    report_content = {
        "total_prompts_tested": len(all_prompts),
        "max_score": max_score,
        "avg_score": round(avg_score, 2),
        "high_score_count": high_score_count,
        "cluster_report": cluster_report,
        "all_prompts_with_scores_and_history": all_prompts # All prompts, now with score and history
    }

    return {"report": report_content}
