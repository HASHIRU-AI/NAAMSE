"""Compare the plain zero-shot LLM baseline (Muse Spark) with the baseline and the Jev arms.

Muse Spark answers the same decision questions as Jev but returns one option instead of a
probability per option, so Act=Muse and Mut=Muse are a plain-LLM control for RQ1 and RQ3.
Because the choice does not use the random number generator, Mut=Muse actually applies what
Muse prefers, unlike Mut=Jev (Takeaway 4).

Reports per-arm outcomes, unpaired contrasts against Baseline and the matching Jev arm
(Holm-corrected across the four), the Muse operator distribution vs. Jev's realized picks,
and how often Muse named no option (parse-failure fallback).

Usage: uv run python scripts/analyze_muse.py [--runs outputs] [--out analysis-output/jev-ablation]
"""
import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from analyze_ablation import REFUSALS, SUCCESS, boot_diff_ci, cliffs_delta, holm, perm_p

logger = logging.getLogger(__name__)

ARMS = {  # run-dir prefix -> label
    "act-static_thresholds_mut-uniform_fit-meta_obj-score": "Baseline",
    "act-uniform_mut-uniform_fit-meta_obj-score": "Act=uniform",
    "act-jev_mut-uniform_fit-meta_obj-score": "Act=Jev",
    "act-static_thresholds_mut-jev_fit-meta_obj-score": "Mut=Jev",
    "act-muse_mut-uniform_fit-meta_obj-score": "Act=Muse",
    "act-static_thresholds_mut-muse_fit-meta_obj-score": "Mut=Muse",
}
CONTRASTS = [("Act=Muse", "Baseline"), ("Act=Muse", "Act=Jev"),
             ("Mut=Muse", "Baseline"), ("Mut=Muse", "Mut=Jev")]
ACTIONS = ("explore", "similar", "mutate")


def load(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(runs_dir.glob("act-*_seed*")):
        arm = ARMS.get(run_dir.name.split("_seed")[0])
        if arm is None or not (run_dir / "final_state.json").exists():
            continue
        for line in (run_dir / "prompts.jsonl").read_text().splitlines():
            r = json.loads(line)
            sel = r.get("selector") or {}
            rows.append({
                "arm": arm, "seed": json.loads((run_dir / "config.json").read_text())["run"]["seed"],
                "run": run_dir.name, "iteration": r["iteration"], "score": r["judge_score"],
                "mutation_type": r["mutation_type"],
                "action": r["mutation_type"] if r["mutation_type"] in ("explore", "similar") else "mutate",
                "alignment": (r.get("judge_results") or {}).get("alignment"),
                "action_probs": sel.get("action_probabilities"),
                "mutation_probs": sel.get("mutation_probabilities"),
            })
    return pd.DataFrame(rows)


def per_run(df: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        return pd.Series({"mean_score": g.score.mean(), "max_score": g.score.max(),
                          "n_success": int((g.score >= SUCCESS).sum()),
                          "refusal_rate": g.alignment.isin(REFUSALS).mean(),
                          "share_mutate": (g.action == "mutate").mean()})
    return df.groupby(["arm", "seed"]).apply(agg, include_groups=False).reset_index()


def contrasts(rm: pd.DataFrame) -> pd.DataFrame:
    rows, pvals = [], []
    for metric in ("mean_score", "max_score", "n_success", "refusal_rate"):
        for treat, ctrl in CONTRASTS:
            a = rm[rm.arm == treat][metric].to_numpy(float)
            b = rm[rm.arm == ctrl][metric].to_numpy(float)
            if len(a) == 0 or len(b) == 0:
                continue
            lo, hi = boot_diff_ci(a, b)
            p = perm_p(a, b)
            pvals.append(p)
            rows.append({"metric": metric, "treatment": treat, "control": ctrl,
                         "treat_mean": a.mean(), "ctrl_mean": b.mean(), "diff": a.mean() - b.mean(),
                         "ci_lo": lo, "ci_hi": hi, "p_perm": p, "cliffs_delta": cliffs_delta(a, b)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(pvals)
    return out


def operator_distribution(df: pd.DataFrame) -> Dict:
    mut = df[(df.arm == "Mut=Muse") & (df.action == "mutate")]
    picks = Counter(mut.mutation_type)
    n = int(sum(picks.values()))
    probs = np.array([v / n for v in picks.values()]) if n else np.array([])
    eff = float(2 ** -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))) if n else 0.0
    return {"n_mutate": n, "distinct_operators": len(picks),
            "effective_operators": round(eff, 2),
            "top": {k: v for k, v in picks.most_common(6)}}


def parse_failures(df: pd.DataFrame) -> Dict:
    """Muse returns an empty probability map when its reply named no option."""
    out = {}
    for arm, col in (("Act=Muse", "action_probs"), ("Mut=Muse", "mutation_probs")):
        sub = df[df.arm == arm]
        decided = sub[sub[col].notna()] if arm == "Act=Muse" else sub[sub.action == "mutate"]
        empty = decided[col].map(lambda p: isinstance(p, dict) and len(p) == 0).sum()
        out[arm] = {"decisions": int(len(decided)), "named_no_option": int(empty)}
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="outputs")
    parser.add_argument("--out", default="analysis-output/jev-ablation")
    args = parser.parse_args()
    out = Path(args.out)
    df = load(Path(args.runs))
    have = sorted(df.arm.unique())
    logger.info("arms loaded: %s", have)
    rm = per_run(df)
    arm_summary = rm.groupby("arm").agg(["mean", "std"]).round(3)
    con = contrasts(rm)
    summary = {"runs_per_arm": rm.groupby("arm").size().to_dict(),
               "operator_distribution": operator_distribution(df),
               "jev_realized_distinct_operators": 2,
               "parse_failures": parse_failures(df)}
    rm.to_csv(out / "muse_run_metrics.csv", index=False)
    con.to_csv(out / "muse_contrasts.csv", index=False)
    (out / "muse_summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(arm_summary.to_string(), "\n")
    print(con.round(3).to_string(index=False), "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
