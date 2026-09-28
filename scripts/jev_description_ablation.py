"""Offline test of what drives Jev's zero-shot operator choice (RQ3). No target or judge calls.

Replays the parents that were actually mutated in the score-objective, Meta-judged arms to Jev's
operator question under five framings of the 26 options:

  original    operator name -> its curated description (as deployed)
  names_only  operator name -> the name in words (no description)
  anonymized  "option_NN" (shuffled) -> the original description (no name)
  swapped     operator name -> another operator's description (fixed derangement)
  evidence    original + the operator's observed outcomes in the sweep (n, mean score, refusal rate)

If picks follow descriptions, `swapped` moves the mass to whichever key carries the favored
description and `anonymized` preserves it. Whether Jev's choice varies with the parent is measured
by the spread of per-parent distributions around the condition mean (Jensen-Shannon).

Usage: uv run python scripts/jev_description_ablation.py [--out analysis-output/jev-ablation]
"""
import argparse
import json
import logging
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()
from src.mutation_engine.selectors.jev_decisions import ask_jev_choice, describe_parent  # noqa: E402
from src.mutation_engine.selectors.mutation_selectors import (  # noqa: E402
    JEV_MUTATION_CRITERIA, JEV_MUTATION_INSTRUCTIONS,
)

logger = logging.getLogger(__name__)

OPS = sorted(JEV_MUTATION_CRITERIA)
CONDITIONS = ("original", "names_only", "anonymized", "swapped", "evidence")
SEED = 0
REPEAT_N = 20  # parents asked twice under `original` to check Jev's determinism


def load_parents(runs_dir: Path) -> List[Dict[str, Any]]:
    """Unique parents of MUTATE children, matched by lineage score within the run."""
    parents, seen = [], set()
    for run_dir in sorted(runs_dir.glob("act-*obj-score_seed*")):
        if "fit-jev" in run_dir.name or not (run_dir / "final_state.json").exists():
            continue
        recs = [json.loads(line) for line in (run_dir / "prompts.jsonl").read_text().splitlines()]
        for r in recs:
            if r["mutation_type"] in ("explore", "similar"):
                continue
            ps = r["history"][-1]["score"]
            cands = [i for i, q in enumerate(recs)
                     if q["iteration"] < r["iteration"] and abs(q["judge_score"] - ps) < 1e-9]
            texts = {" ".join(map(str, recs[i]["prompt"])) for i in cands}
            if len(texts) != 1 or (run_dir.name, cands[0]) in seen:
                continue  # unmatched or ambiguous parent
            seen.add((run_dir.name, cands[0]))
            q = recs[cands[0]]
            parents.append({"run": run_dir.name, "idx": cands[0], "prompt": q["prompt"],
                            "score": q["judge_score"], "metadata": {"judge_results": q["judge_results"]}})
    return parents


def framings(ops_stats: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """criteria dict and key -> operator map for each condition."""
    rng = random.Random(SEED)
    anon_order = OPS[:]
    rng.shuffle(anon_order)
    derange = OPS[:]
    while any(a == b for a, b in zip(OPS, derange)):
        rng.shuffle(derange)
    stats = ops_stats.set_index("operator")

    def evidence(op: str) -> str:
        if op not in stats.index:
            return f"{JEV_MUTATION_CRITERIA[op]} Observed so far in this study: never tried."
        s = stats.loc[op]
        return (f"{JEV_MUTATION_CRITERIA[op]} Observed so far in this study: {int(s.n)} uses, mean judge "
                f"score {s.mean_all:.1f}/100, target refused {100 * s.refusal_all:.0f}% of them.")

    return {
        "original": {"criteria": dict(JEV_MUTATION_CRITERIA), "key_to_op": {o: o for o in OPS}},
        "names_only": {"criteria": {o: o.replace("_", " ") for o in OPS}, "key_to_op": {o: o for o in OPS}},
        "anonymized": {"criteria": {f"option_{i:02d}": JEV_MUTATION_CRITERIA[o] for i, o in enumerate(anon_order)},
                       "key_to_op": {f"option_{i:02d}": o for i, o in enumerate(anon_order)}},
        # key o carries the description of desc_of[o]
        "swapped": {"criteria": {o: JEV_MUTATION_CRITERIA[d] for o, d in zip(OPS, derange)},
                    "key_to_op": {o: o for o in OPS}, "desc_of": dict(zip(OPS, derange))},
        "evidence": {"criteria": {o: evidence(o) for o in OPS}, "key_to_op": {o: o for o in OPS}},
    }


def ask(parent: Dict[str, Any], cond: str, frame: Dict[str, Any]) -> Dict[str, Any]:
    try:
        _, probs = ask_jev_choice("mutation", JEV_MUTATION_INSTRUCTIONS, frame["criteria"],
                                  describe_parent(parent), 0.0, random.Random(0))
    except Exception as e:  # keep the other calls if one fails; failed rows are retried on rerun
        logger.error("Jev call failed (%s, %s/%s): %s", cond, parent["run"], parent["idx"], e)
        return {"error": str(e)}
    return {"probs": {frame["key_to_op"][k]: v for k, v in probs.items()}}


def run_calls(parents: List[Dict[str, Any]], frames: Dict[str, Dict[str, Any]], raw: Path, workers: int) -> None:
    done = set()
    if raw.exists():
        for line in raw.read_text().splitlines():
            r = json.loads(line)
            if "probs" in r:
                done.add((r["condition"], r["run"], r["idx"], r["rep"]))
    jobs = [(c, p, 0) for c in CONDITIONS for p in parents]
    jobs += [("original", p, 1) for p in parents[:REPEAT_N]]
    jobs = [j for j in jobs if (j[0], j[1]["run"], j[1]["idx"], j[2]) not in done]
    logger.info("%d Jev calls to make", len(jobs))
    with ThreadPoolExecutor(workers) as pool, open(raw, "a") as f:
        futures = [(c, p, rep, pool.submit(ask, p, c, frames[c])) for c, p, rep in jobs]
        for i, (c, p, rep, fut) in enumerate(futures, 1):
            f.write(json.dumps({"condition": c, "run": p["run"], "idx": p["idx"], "rep": rep,
                                "parent_score": p["score"], **fut.result()}) + "\n")
            f.flush()
            if i % 50 == 0:
                logger.info("%d/%d", i, len(jobs))


def js(p: np.ndarray, q: np.ndarray) -> float:
    m = (p + q) / 2
    kl = lambda a, b: float(np.sum(np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / np.where(b > 0, b, 1e-12)), 0)))
    return (kl(p, m) + kl(q, m)) / 2


def summarize(raw: Path, frames: Dict[str, Dict[str, Any]], ops_stats: pd.DataFrame) -> Dict[str, Any]:
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    rows = [r for r in rows if "probs" in r]
    mat = {c: np.array([[r["probs"].get(o, 0.0) for o in OPS] for r in rows if r["condition"] == c and r["rep"] == 0])
           for c in CONDITIONS}
    mat = {c: m / m.sum(axis=1, keepdims=True) for c, m in mat.items()}
    mean = {c: m.mean(axis=0) for c, m in mat.items()}
    perf = ops_stats.set_index("operator").mean_all.reindex(OPS)
    out: Dict[str, Any] = {"n_parents": {c: int(len(m)) for c, m in mat.items()}, "conditions": {}}
    for c in CONDITIONS:
        mu, m = mean[c], mat[c]
        top = np.argsort(-mu)[:3]
        argmaxes = pd.Series([OPS[i] for i in m.argmax(axis=1)]).value_counts()
        out["conditions"][c] = {
            "top3": {OPS[i]: round(float(mu[i]), 3) for i in top},
            "p_steg_plus_synonym": round(float(mu[OPS.index("semantic_steganography_mutation")]
                                               + mu[OPS.index("synonym_mutation")]), 3),
            "effective_n_operators": round(float(2 ** -np.sum(mu[mu > 0] * np.log2(mu[mu > 0]))), 2),
            "distinct_argmax": int(len(argmaxes)), "argmax_counts": argmaxes.head(5).to_dict(),
            "mean_js_to_condition_mean": round(float(np.mean([js(p, mu) for p in m])), 3),
            "spearman_prob_vs_observed_mean_score": round(float(pd.Series(mu, index=OPS).corr(perf, method="spearman")), 3),
        }
    # Name- vs description-following in the swapped condition
    desc_of = frames["swapped"]["desc_of"]
    by_name = pd.Series(mean["original"], index=OPS)
    swapped = pd.Series(mean["swapped"], index=OPS)
    by_desc = pd.Series({desc_of[k]: v for k, v in swapped.items()})  # mass re-indexed by description owner
    out["swapped_follows"] = {
        "corr_with_original_by_name": round(float(swapped.corr(by_name, method="spearman")), 3),
        "corr_with_original_by_description": round(float(by_desc.reindex(OPS).corr(by_name, method="spearman")), 3),
        "mass_on_key_semantic_steganography": round(float(swapped["semantic_steganography_mutation"]), 3),
        "mass_on_key_carrying_steganography_description": round(float(by_desc["semantic_steganography_mutation"]), 3),
    }
    anon = pd.Series(mean["anonymized"], index=OPS)
    out["anonymized_corr_with_original"] = round(float(anon.corr(by_name, method="spearman")), 3)
    reps = [r for r in rows if r["condition"] == "original"]
    pairs = {}
    for r in reps:
        pairs.setdefault((r["run"], r["idx"]), {})[r["rep"]] = np.array([r["probs"].get(o, 0.0) for o in OPS])
    diffs = [np.abs(v[0] - v[1]).max() for v in pairs.values() if len(v) == 2]
    out["repeat_max_abs_prob_diff"] = {"n": len(diffs), "median": float(np.median(diffs)) if diffs else None,
                                       "max": float(np.max(diffs)) if diffs else None}
    out["mean_probs"] = {c: {o: round(float(v), 4) for o, v in zip(OPS, mean[c])} for c in CONDITIONS}
    return out


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="outputs")
    parser.add_argument("--out", default="analysis-output/jev-ablation")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-parents", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    ops_stats = pd.read_csv(out / "operators_mutator_refusals.csv")  # from analyze_offpolicy.py
    parents = load_parents(Path(args.runs))
    if args.limit_parents:
        parents = parents[: args.limit_parents]
    logger.info("%d parents", len(parents))
    frames = framings(ops_stats)
    raw = out / "jev_description_ablation_raw.jsonl"
    run_calls(parents, frames, raw, args.workers)
    summary = summarize(raw, frames, ops_stats)
    (out / "jev_description_ablation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "mean_probs"}, indent=2))


if __name__ == "__main__":
    main()
