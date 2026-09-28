"""Run one arm of the Jev ablation from the command line.

Example:
    uv run python -m src.experiments.run_ablation \\
        --target http://localhost:5000 --iterations 7 --mutations 4 --seed 1 \\
        --action-selector jev --mutation-selector uniform --fitness-judge meta

With no ablation flags this is the unmodified NAAMSE baseline. Each run writes
outputs/<arm>_seed<seed>_<timestamp>/{config.json, prompts.jsonl, final_state.json}.
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.experiments.ablation_config import AblationConfig, add_ablation_args, ablation_from_args

logger = logging.getLogger(__name__)

DATABASES = {
    "adversarial": {
        "db_file": "src/cluster_engine/data_access/adversarial/naamse.db",
        "centroids_file": "src/cluster_engine/data_access/adversarial/centroids.pkl",
        "lookup_file": "src/cluster_engine/data_access/adversarial/cluster_lookup_table.json",
    },
    "benign": {
        "db_file": "src/cluster_engine/data_access/benign/naamse_benign.db",
        "centroids_file": "src/cluster_engine/data_access/adversarial/centroids.pkl",
        "lookup_file": "src/cluster_engine/data_access/benign/cluster_lookup_table.json",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one arm of the NAAMSE x Jev ablation.")
    run = parser.add_argument_group("run")
    run.add_argument("--target", required=True, help="A2A URL of the agent under test")
    run.add_argument("--iterations", type=int, default=7, help="Fuzzer iterations")
    run.add_argument("--mutations", type=int, default=4, help="Mutations per iteration")
    run.add_argument("--score-threshold", type=float, default=80,
                     help="Prompts scoring at or above this seed the next iteration")
    run.add_argument("--seed", type=int, default=1234, help="Random seed for the run")
    run.add_argument("--corpus", choices=sorted(DATABASES), default="adversarial",
                     help="Prompt corpus to draw EXPLORE/SIMILAR prompts from")
    run.add_argument("--mutation-llm", choices=["gemini", "meta"], default="gemini",
                     help="Provider for the mutation engine's LLM (meta = Muse Spark)")
    run.add_argument("--max-concurrency", type=int, default=4, help="Parallel workers per iteration")
    run.add_argument("--output-dir", default="outputs", help="Parent directory for run output")
    run.add_argument("--dry-run", action="store_true", help="Print the resolved config and exit")
    add_ablation_args(parser)
    return parser


def make_run_dir(parent: str, ablation: AblationConfig, seed: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(parent) / f"{ablation.arm_name}_seed{seed}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def resolved_config(args: argparse.Namespace, ablation: AblationConfig) -> Dict[str, Any]:
    return {
        "arm": ablation.arm_name,
        "ablation": ablation.to_dict(),
        "run": {
            "target": args.target, "iterations": args.iterations, "mutations": args.mutations,
            "score_threshold": args.score_threshold, "seed": args.seed, "corpus": args.corpus,
            "mutation_llm": args.mutation_llm, "max_concurrency": args.max_concurrency,
        },
        "argv": sys.argv[1:],
    }


def prompt_record(prompt: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a scored prompt into one JSON-serializable record."""
    metadata = prompt.get("metadata") or {}
    return {
        "iteration": metadata.get("iteration"),
        "prompt": prompt.get("prompt"),
        "fitness": prompt.get("score"),
        "judge_score": metadata.get("judge_score"),
        "judge_results": metadata.get("judge_results"),
        "mutation_type": metadata.get("mutation_type"),
        "selector": metadata.get("selector"),
        "history": metadata.get("history"),
        "conversation": prompt.get("conversation_history"),
    }


def write_outputs(run_dir: Path, final_state: Dict[str, Any]) -> None:
    prompts: List[Dict[str, Any]] = final_state.get("all_fuzzer_prompts_with_scores") or []
    with open(run_dir / "prompts.jsonl", "w") as f:
        for p in prompts:
            f.write(json.dumps(prompt_record(p), default=str) + "\n")
    with open(run_dir / "final_state.json", "w") as f:
        json.dump({
            "iterations_completed": final_state.get("current_iteration"),
            "n_prompts": len(prompts),
            "covered_categories": final_state.get("covered_categories"),
        }, f, indent=2, default=str)


async def run(args: argparse.Namespace, ablation: AblationConfig, run_dir: Path) -> Dict[str, Any]:
    # Read by the mutation engine when it builds its (cached) agent
    os.environ["MUTATION_ENGINE_PROVIDER"] = args.mutation_llm
    # Imported here so --dry-run and --help stay fast and need no credentials
    from src.agent.graph import graph
    from src.cluster_engine.data_access.sqlite_source import SQLiteDataSource
    from src.config import Config

    os.environ["NAAMSE_RANDOM_SEED"] = str(args.seed)  # initialize_fuzzer re-reads this
    Config.set_seed(args.seed)

    corpus = DATABASES[args.corpus]
    missing = [p for p in (corpus["db_file"], corpus["lookup_file"]) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Corpus files missing: {missing}. See the README for the database download.")

    config = {
        "max_concurrency": args.max_concurrency,
        "recursion_limit": 100,
        "configurable": {
            "database": SQLiteDataSource(**corpus),
            "output_path": str(run_dir / "report.pdf"),
            "is_score_flipped": args.corpus == "benign",
            "ablation": ablation,
        },
    }
    fuzzer_input = {
        "iterations_limit": args.iterations,
        "mutations_per_iteration": args.mutations,
        "score_threshold": args.score_threshold,
        "a2a_agent_url": args.target,
        "is_score_flipped": args.corpus == "benign",
    }
    return await graph.ainvoke(fuzzer_input, config=config)


def main(argv: List[str] = None) -> int:
    args = build_parser().parse_args(argv)
    ablation = ablation_from_args(args)
    config = resolved_config(args, ablation)

    if args.dry_run:
        print(json.dumps(config, indent=2))
        return 0

    run_dir = make_run_dir(args.output_dir, ablation, args.seed)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Run directory: {run_dir}")

    final_state = asyncio.run(run(args, ablation, run_dir))
    write_outputs(run_dir, final_state)
    print(f"Done: {len(final_state.get('all_fuzzer_prompts_with_scores') or [])} prompts, "
          f"covered {len(final_state.get('covered_categories') or [])}/6 categories -> {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
