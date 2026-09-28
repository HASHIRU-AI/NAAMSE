"""RQ2 with a common referee: re-scored Fit=Jev prompts vs. the Meta-judged arms.

The Fit=Jev arm was scored by Jev, all other arms by Meta. run_referee.py re-scores the Fit=Jev
prompts with the (gated) Meta judge, which puts that arm on the same scale as the others and
pairs a Jev and a Meta verdict on each of its 140 prompts. This script reports:

1. Arm comparison on the Meta scale: Fit=Jev (refereed) vs. Baseline, per run.
2. Jev-Meta agreement: alignment confusion matrix, Cohen's kappa (unweighted, quadratic),
   refusal and success agreement, score rank correlation.
3. Agreement-calibration of Jev's probabilities, using Meta's verdicts as the reference:
   P(refusal) from alignment probabilities, and per-category relatedness (non-gated rows only,
   since gated rows carry no Meta relatedness verdict). Brier score and 10-bin ECE.

Usage: uv run python scripts/analyze_referee.py [--referee analysis-output/jev-ablation/referee_fitjev_meta.jsonl]
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from analyze_ablation import REFUSALS, SUCCESS, boot_diff_ci, cliffs_delta, perm_p
from analyze_pooled import load_prompts

logger = logging.getLogger(__name__)

LEVELS = ["full_refusal", "weak_refusal", "weak_acceptance", "full_acceptance"]


def kappa(a: pd.Series, b: pd.Series, weights: str = "none") -> float:
    idx = {k: i for i, k in enumerate(LEVELS)}
    n = len(LEVELS)
    obs = np.zeros((n, n))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    i, j = np.indices((n, n))
    w = (i != j).astype(float) if weights == "none" else ((i - j) / (n - 1)) ** 2
    return float(1 - (w * obs).sum() / (w * exp).sum())


def brier_ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> Tuple[float, float]:
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = sum(abs(p[which == k].mean() - y[which == k].mean()) * (which == k).mean()
              for k in range(bins) if (which == k).any())
    return brier, float(ece)


def load_pairs(referee_path: Path, runs_dir: Path) -> pd.DataFrame:
    ref = pd.DataFrame([json.loads(line) for line in referee_path.read_text().splitlines()])
    ref = ref[ref.error.isna()].drop_duplicates(["run", "idx"], keep="last")
    jev = {}
    for run in ref.run.unique():
        for idx, line in enumerate((runs_dir / run / "prompts.jsonl").read_text().splitlines()):
            jev[(run, idx)] = json.loads(line).get("judge_results") or {}
    ref["jev_results"] = [jev[(r, i)] for r, i in zip(ref.run, ref.idx)]
    return ref


def arm_comparison(pairs: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    fit = pairs.assign(refused=pairs.referee_alignment.isin(REFUSALS),
                       success=pairs.referee_score >= SUCCESS)
    per_run = {"Fit=Jev (Meta referee)": fit.groupby("run").agg(
                   mean_score=("referee_score", "mean"), refusal=("refused", "mean"), successes=("success", "sum")),
               "Fit=Jev (own Jev judge)": fit.assign(refused_j=fit.fitness_alignment.isin(REFUSALS),
                                                     success_j=fit.fitness_score >= SUCCESS).groupby("run").agg(
                   mean_score=("fitness_score", "mean"), refusal=("refused_j", "mean"), successes=("success_j", "sum"))}
    for arm in ("Baseline", "Act=uniform", "Act=Jev", "Mut=Jev"):
        per_run[arm] = df[df.arm == arm].groupby("run").agg(
            mean_score=("score", "mean"), refusal=("refused", "mean"), successes=("success", "sum"))
    rows = []
    base = per_run["Baseline"]
    for arm, g in per_run.items():
        row = {"arm": arm, "n_runs": len(g)}
        for m in ("mean_score", "refusal", "successes"):
            row[m] = g[m].mean()
            if arm != "Baseline" and len(g) == len(base):
                a, b = g[m].to_numpy(float), base[m].to_numpy(float)
                lo, hi = boot_diff_ci(a, b)
                row.update({f"{m}_diff": a.mean() - b.mean(), f"{m}_ci": f"[{lo:+.2f}, {hi:+.2f}]",
                            f"{m}_p": perm_p(a, b), f"{m}_delta": cliffs_delta(a, b)})
        rows.append(row)
    return pd.DataFrame(rows)


def agreement(pairs: pd.DataFrame) -> Dict:
    j, m = pairs.fitness_alignment, pairs.referee_alignment
    conf = pd.crosstab(j.rename("Jev"), m.rename("Meta")).reindex(index=LEVELS, columns=LEVELS, fill_value=0)
    js, ms = pairs.fitness_score >= SUCCESS, pairs.referee_score >= SUCCESS
    return {
        "n": int(len(pairs)),
        "alignment_exact_agreement": float((j == m).mean()),
        "kappa_unweighted": kappa(j, m), "kappa_quadratic": kappa(j, m, "quadratic"),
        "refusal_agreement": float((j.isin(REFUSALS) == m.isin(REFUSALS)).mean()),
        "jev_refusal_rate": float(j.isin(REFUSALS).mean()), "meta_refusal_rate": float(m.isin(REFUSALS).mean()),
        "score_spearman": float(pairs.fitness_score.corr(pairs.referee_score, method="spearman")),
        "successes_jev": int(js.sum()), "successes_meta": int(ms.sum()), "successes_both": int((js & ms).sum()),
        "confusion_jev_rows_meta_cols": conf.to_dict(orient="index"),
    }


def calibration(pairs: pd.DataFrame) -> Dict:
    out = {}
    p_ref = pairs.jev_results.map(lambda r: sum((r.get("alignment_probabilities") or {}).get(k, 0.0) for k in REFUSALS))
    y_ref = pairs.referee_alignment.isin(REFUSALS).astype(float)
    b, e = brier_ece(p_ref.to_numpy(float), y_ref.to_numpy())
    out["refusal"] = {"n": int(len(pairs)), "brier": b, "ece": e,
                      "mean_p": float(p_ref.mean()), "meta_rate": float(y_ref.mean())}
    rows = []
    for jr, cats in zip(pairs.jev_results, pairs.referee_categories):
        for c, v in (cats or {}).items():
            jp = (jr.get("categories") or {}).get(c, {}).get("related_probability")
            if jp is not None:
                rows.append({"category": c, "p": jp, "y": float(bool(v.get("is_related")))})
    rel = pd.DataFrame(rows)
    if len(rel):
        b, e = brier_ece(rel.p.to_numpy(float), rel.y.to_numpy())
        out["relatedness"] = {"n": int(len(rel)), "brier": b, "ece": e,
                              "mean_p": float(rel.p.mean()), "meta_rate": float(rel.y.mean())}
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="outputs")
    parser.add_argument("--referee", default="analysis-output/jev-ablation/referee_fitjev_meta.jsonl")
    parser.add_argument("--out", default="analysis-output/jev-ablation")
    args = parser.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    pairs = load_pairs(Path(args.referee), runs)
    # Relatedness calibration needs real Meta relatedness verdicts, which gated rows lack
    ungated = pairs[pairs.judge_calls > 1]
    comp = arm_comparison(pairs, load_prompts(runs))
    summary = {"judge_calls": int(pairs.judge_calls.sum()), "ungated_equivalent": 7 * len(pairs),
               "agreement": agreement(pairs), "calibration_all": calibration(pairs.assign(referee_categories=None)),
               "calibration_ungated": calibration(ungated)}
    comp.to_csv(out / "referee_arm_comparison.csv", index=False)
    (out / "referee_summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(comp.round(3).to_string(), "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
