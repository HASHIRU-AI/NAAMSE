"""Controlled operator experiment: apply a fixed set of operators to a fixed set of parents.

The sweep's operator evidence is observational (who got mutated, with what, depended on the
selector and the seed). Here every operator is applied once to every parent, with a fixed
per-(parent, operator) seed, sent to the target, and scored by the refusal-gated Meta judges.
This isolates the operator's effect from parent quality and selection.

Parents: stratified by stored judge score (<30, 30-50, 50-80) from the score-objective,
Meta-judged runs, excluding texts that are themselves mutator refusals.
Operators: a no-op control (echo), Jev's favorites, the observed top operators, a non-LLM
operator (synonym), and a weak one.

Full records (prompt and response text) go to outputs/operator_experiment/ (gitignored);
a text-free summary goes to analysis-output/jev-ablation/operator_experiment.csv. Resumable.

Usage: uv run python scripts/operator_experiment.py --target http://localhost:5050 [--limit N]
"""
import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()
os.environ["MUTATION_ENGINE_PROVIDER"] = "meta"  # same mutation LLM as the sweep

from analyze_offpolicy import is_mutator_refusal  # noqa: E402

logger = logging.getLogger(__name__)

OPERATORS = [
    "echo",                              # no-op control
    "synonym_mutation",                  # no LLM call
    "semantic_steganography_mutation",   # Jev favorite; often refused by the mutator
    "adversarial_prefix_mutation",       # Jev favorite
    "persona_roleplay_mutation",         # Jev favorite
    "artprompt",                         # observed top
    "many_shot_jailbreaking",            # observed top
    "dual_response_divider_mutation",    # observed upper
    "code_exec",                         # observed upper
    "language_translation_mutation",     # observed weak
]
STRATA = {"<30": (0, 30, 8), "30-50": (30, 50, 4), "50-80": (50, 80, 8)}
SCORE_ARMS = ("act-static_thresholds_mut-uniform_fit-meta_obj-score", "act-uniform_mut-uniform_fit-meta_obj-score",
              "act-jev_mut-uniform_fit-meta_obj-score", "act-static_thresholds_mut-jev_fit-meta_obj-score")
MAX_CHARS = 4000
SEED = 0
RECORDS = Path("outputs/operator_experiment/records.jsonl")
SUMMARY = Path("analysis-output/jev-ablation/operator_experiment.csv")


def pick_parents(runs: Path) -> List[Dict[str, Any]]:
    pool, seen = [], set()
    for run_dir in sorted(runs.glob("act-*_seed*")):
        if run_dir.name.split("_seed")[0] not in SCORE_ARMS:
            continue
        for idx, line in enumerate((run_dir / "prompts.jsonl").read_text().splitlines()):
            r = json.loads(line)
            text = " ".join(map(str, r["prompt"]))
            if text in seen or len(text) > MAX_CHARS or is_mutator_refusal(text):
                continue
            seen.add(text)
            pool.append({"parent_id": f"{run_dir.name}:{idx}", "text": text, "score": r["judge_score"]})
    rng = random.Random(SEED)
    parents = []
    for name, (lo, hi, k) in STRATA.items():
        cands = [p for p in pool if lo <= p["score"] < hi]
        parents += [{**p, "bucket": name} for p in rng.sample(cands, k)]
    return parents


def task_seed(parent_id: str, op: str) -> int:
    return int(hashlib.sha256(f"{parent_id}|{op}".encode()).hexdigest()[:12], 16)


def mutate(parent: Dict[str, Any], op: str, database) -> Dict[str, Any]:
    from src.mutation_engine.nodes.invoke_mutation_llm import invoke_llm_with_tools
    state = {"prompt_to_mutate": {"prompt": [parent["text"]], "score": parent["score"], "metadata": {}},
             "mutation_type": op, "task_seed": task_seed(parent["parent_id"], op)}
    out = invoke_llm_with_tools(state, {"configurable": {"database": database}})
    text = " ".join(map(str, out["mutated_prompt"]["prompt"]))
    return {"text": text,
            "mutation_failed": op != "echo" and text.strip() == parent["text"].strip(),
            "mutator_refusal": is_mutator_refusal(text)}


async def run_one(parent, op, target, database, sem) -> Dict[str, Any]:
    from src.experiments.gated_judge import gated_behavior_engine_graph, judge_calls
    from src.invoke_agent.nodes.invoke_agent import _sync_a2a_interaction
    from src.experiments.coverage import summarize_judge_results
    async with sem:
        rec = {"parent_id": parent["parent_id"], "bucket": parent["bucket"], "parent_score": parent["score"],
               "operator": op}
        try:
            m = await asyncio.to_thread(mutate, parent, op, database)
            conv = await asyncio.to_thread(_sync_a2a_interaction, [m["text"][:20000]], target)
            res = await gated_behavior_engine_graph.ainvoke(
                {"conversation_history": conv["conversation_history"], "is_score_flipped": False},
                config={"configurable": {"judge_backend": "meta"}})
            summ = summarize_judge_results(res.get("judge_results"), res.get("ra_judge_results"))
            rec.update({**{k: m[k] for k in ("mutation_failed", "mutator_refusal")},
                        "score": res.get("final_score"), "alignment": summ.get("alignment"),
                        "categories": summ.get("categories"), "judge_calls": judge_calls(res.get("judge_results")),
                        "prompt": m["text"], "conversation": conv["conversation_history"], "error": None})
        except Exception as e:  # keep going; failed cells are retried on rerun
            logger.error("cell failed (%s, %s): %s", parent["parent_id"], op, e)
            rec["error"] = str(e)
        return rec


def write_summary() -> None:
    import pandas as pd
    recs = [json.loads(l) for l in RECORDS.read_text().splitlines()]
    recs = [r for r in recs if r.get("error") is None]
    rows = [{k: r.get(k) for k in ("parent_id", "bucket", "parent_score", "operator", "score", "alignment",
                                   "mutation_failed", "mutator_refusal", "judge_calls")} |
            {"max_harm": max((v["harm"] for v in (r.get("categories") or {}).values() if v.get("is_related")),
                             key=["not_harmful", "low_risk", "high_risk", "harmful"].index, default="not_harmful")}
            for r in recs]
    pd.DataFrame(rows).to_csv(SUMMARY, index=False)


async def main_async(args) -> None:
    from src.cluster_engine.data_access.sqlite_source import SQLiteDataSource
    from src.experiments.run_ablation import DATABASES
    database = SQLiteDataSource(**DATABASES["adversarial"])
    RECORDS.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if RECORDS.exists():
        for l in RECORDS.read_text().splitlines():
            r = json.loads(l)
            if r.get("error") is None:
                done.add((r["parent_id"], r["operator"]))
    parents = pick_parents(Path(args.runs))
    cells = [(p, op) for p in parents for op in OPERATORS if (p["parent_id"], op) not in done]
    cells = cells[: args.limit or None]
    logger.info("%d parents x %d operators; %d cells to run", len(parents), len(OPERATORS), len(cells))
    sem = asyncio.Semaphore(args.concurrency)
    n = calls = 0
    with open(RECORDS, "a") as f:
        for coro in asyncio.as_completed([run_one(p, op, args.target, database, sem) for p, op in cells]):
            rec = await coro
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            n += 1
            calls += rec.get("judge_calls") or 0
            if n % 10 == 0 or n == len(cells):
                logger.info("%d/%d cells, %d judge calls", n, len(cells), calls)
    write_summary()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--runs", default="outputs")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=3)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
