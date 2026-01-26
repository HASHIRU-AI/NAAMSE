import json
from typing import Any, Dict, List

from src.cluster_engine.utilities import get_cluster_id_for_prompt, get_human_readable_cluster_info
from langchain_core.runnables import RunnableConfig


def generate_report_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """
    Generates a consolidated report from the fuzzer's final state.
    This function is designed to be a node in a LangGraph.
    """
    all_prompts: List[Dict[str, Any]] = state.get(
        "all_fuzzer_prompts_with_scores", [])
    request_score_threshold = state.get(
        "score_threshold", 0.0)  # Extract from state

    # Calculate summary statistics and cluster averages in one pass
    total_score = 0
    max_score = 0
    high_score_count = 0

    # Track per-cluster: {cluster_label: {"sum": x, "count": y, "max": z, "description": ""}}
    cluster_stats: dict[str, dict[str, Any]] = {}

    # Track per-mutation: {mutation_type: {"sum": x, "count": y, "max": z}}
    mutation_stats: dict[str, dict[str, float]] = {}

    # Track per-iteration: {iteration: {"sum": x, "count": y, "max": z}}
    iteration_stats: dict[int, dict[str, float]] = {}

    database = config.get("configurable", {}).get("database", None)

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
        cluster_id = cluster_info.get("cluster_id")
        description = cluster_info.get("description", "default description")

        # no cluster_id means we find it via getting cluster info
        if not cluster_id:
            prompt_text = prompt.get("prompt", [])
            if prompt_text:
                nearest_cluster_id = get_cluster_id_for_prompt(
                    prompt_text, data_source=database)
                cluster_id = nearest_cluster_id if nearest_cluster_id else cluster_id

        cluster_info_with_description = get_human_readable_cluster_info(
            cluster_id=cluster_id, data_source=database)

        # add cluster info to metadata.cluster_info
        metadata["cluster_info"] = {
            **cluster_info, **cluster_info_with_description}

        if cluster_info_with_description:
            cluster_label = cluster_info_with_description.get(
                "label", cluster_label)
            description = cluster_info_with_description.get(
                "description", description)
        if cluster_label not in cluster_stats:
            cluster_stats[cluster_label] = {
                "sum": 0, "count": 0, "max": 0, "description": description}
        elif not cluster_stats[cluster_label]["description"]:
            cluster_stats[cluster_label]["description"] = description

        cluster_stats[cluster_label]["sum"] += score
        cluster_stats[cluster_label]["count"] += 1
        cluster_stats[cluster_label]["max"] = max(
            cluster_stats[cluster_label]["max"], score)

        # Track mutation type statistics
        mutation_type = metadata.get("mutation_type", "unknown")
        if mutation_type not in mutation_stats:
            mutation_stats[mutation_type] = {"sum": 0, "count": 0, "max": 0}
        mutation_stats[mutation_type]["sum"] += score
        mutation_stats[mutation_type]["count"] += 1
        mutation_stats[mutation_type]["max"] = max(
            mutation_stats[mutation_type]["max"], score)

        # Track iteration statistics
        iteration = metadata.get("iteration", 0)
        if iteration not in iteration_stats:
            iteration_stats[iteration] = {"sum": 0, "count": 0, "max": 0}
        iteration_stats[iteration]["sum"] += score
        iteration_stats[iteration]["count"] += 1
        iteration_stats[iteration]["max"] = max(
            iteration_stats[iteration]["max"], score)

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

    # Build mutation report sorted by avg score descending
    mutation_report = [
        {
            "mutation_type": mut_type,
            "count": int(stats["count"]),
            "avg_score": round(stats["sum"] / stats["count"], 2),
            "max_score": round(stats["max"], 2)
        }
        for mut_type, stats in mutation_stats.items()
    ]
    mutation_report.sort(key=lambda x: -x["avg_score"])

    # Build iteration progression report sorted by iteration number
    iteration_progression = [
        {
            "iteration": iteration,
            "count": int(stats["count"]),
            "avg_score": round(stats["sum"] / stats["count"], 2),
            "max_score": round(stats["max"], 2)
        }
        for iteration, stats in iteration_stats.items()
    ]
    iteration_progression.sort(key=lambda x: x["iteration"])

    # print statements for terminal readability can be kept here or moved to the agent
    print(f"Fuzzer completed. Tested {len(all_prompts)} prompts.")
    print(
        f"Max score: {max_score}, Avg score: {avg_score}, High scoring prompts: {high_score_count}")
    print(f"Cluster breakdown: {len(cluster_report)} clusters")
    print(f"Mutation breakdown: {len(mutation_report)} mutation types")

    report_content = {
        "summary": {
            "total_prompts_tested": len(all_prompts),
            "max_score": max_score,
            "avg_score": round(avg_score, 2),
            "high_score_count": high_score_count,
            "cluster_report": cluster_report,
            "mutation_report": mutation_report,
            "iteration_progression": iteration_progression,
        },
        # All prompts, now with score and history
        "all_prompts_with_scores_and_history": all_prompts
    }

    return {"report": report_content}


if __name__ == "__main__":
    # Test block to run this node independently
    # load test state from JSON file
    PATH = "tests/data/fuzzer_final_state_2.json"
    with open(PATH, "r", encoding="utf-8") as f:
        test_state = json.load(f)
    report_result = generate_report_node(test_state)
    # remove all_prompts_with_scores_and_history for brevity
    # if "all_prompts_with_scores_and_history" in report_result["report"]:
    # del report_result["report"]["all_prompts_with_scores_and_history"]
    print(json.dumps(report_result, indent=2))
    # write to file
    with open("tests/data/final_report_2.json", "w", encoding="utf-8") as f:
        json.dump(report_result, f, indent=2)
