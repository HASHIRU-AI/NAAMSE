"""Analyze the controlled operator experiment (scripts/operator_experiment.py).

Every operator is applied to the same 20 parents, so operators are compared *within parent*:
each operator's judge score minus the echo (no-op) score on the same parent, with a bootstrap
over parents, a Wilcoxon signed-rank test, and Holm correction across operators. Echo re-sends
the parent unchanged, so echo vs. the parent's stored score also measures test-retest noise.

Usage: uv run python scripts/analyze_operator_experiment.py [--out analysis-output/jev-ablation]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from analyze_ablation import SUCCESS, _save, holm

REFUSALS = ("full_refusal", "weak_refusal")
N_BOOT = 5000
BLUE, ORANGE, GREY = "#1f77b4", "#ff7f0e", "#9e9e9e"
SHORT = lambda o: o.replace("_mutation", "").replace("_attack", "")


def boot_mean(x: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def contrasts(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    wide = df.pivot(index="parent_id", columns="operator", values="score")
    rows = []
    for op in [c for c in wide.columns if c != "echo"]:
        d = (wide[op] - wide["echo"]).dropna().to_numpy()
        if len(d) < 3:
            continue
        lo, hi = boot_mean(d)
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        rows.append({"subset": subset, "operator": op, "n_parents": len(d), "mean_diff_vs_echo": d.mean(),
                     "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": p})
    out = pd.DataFrame(rows)
    out["holm_p"] = holm(out.wilcoxon_p.tolist())
    return out.sort_values("mean_diff_vs_echo", ascending=False)


def per_operator(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("operator")
    t = pd.DataFrame({
        "n": g.size(), "mean_score": g.score.mean(), "refusal": g.refused.mean(),
        "successes": g.success.sum(), "high_risk_or_harmful": g.high.sum(),
        "mutator_refusal": g.mutator_refusal.sum(), "mutation_failed": g.mutation_failed.sum(),
    })
    real = df[df.real].groupby("operator")
    t["n_real"] = real.size()
    t["mean_score_real"] = real.score.mean()
    return t.reset_index().sort_values("mean_score", ascending=False)


def fig(ops: pd.DataFrame, con: pd.DataFrame, out: Path) -> None:
    con = con[con.subset == "all"].set_index("operator")
    order = ops.sort_values("mean_score").operator.tolist()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    y = np.arange(len(order))
    t = ops.set_index("operator").loc[order]
    a1.barh(y, t.mean_score, color=[GREY if o == "echo" else BLUE for o in order])
    for yi, v, r in zip(y, t.mean_score, t.refusal):
        a1.text(v + 0.5, yi, f"{v:.1f}  (refused {r:.0%})", va="center", fontsize=7)
    a1.set_yticks(y, [SHORT(o) for o in order], fontsize=8)
    a1.set_xlabel("Mean judge score over 20 parents", fontsize=8)
    a1.set_xlim(0, max(t.mean_score) * 1.45)
    a1.spines[["top", "right"]].set_visible(False)
    a1.set_title("Per-operator outcome (grey = echo control)", fontsize=9)
    c = con.reindex([o for o in order if o != "echo"])
    yc = np.arange(len(c))
    a2.errorbar(c.mean_diff_vs_echo, yc, xerr=[c.mean_diff_vs_echo - c.ci_lo, c.ci_hi - c.mean_diff_vs_echo],
                fmt="o", color=ORANGE, ecolor="0.6", capsize=2, ms=5)
    a2.axvline(0, color="0.5", ls="--", lw=1)
    a2.set_yticks(yc, [SHORT(o) for o in c.index], fontsize=8)
    a2.set_xlabel("Score minus echo on the same parent\n(mean, 95% parent-bootstrap CI)", fontsize=8)
    a2.spines[["top", "right"]].set_visible(False)
    a2.set_title("Within-parent effect vs. no-op", fontsize=9)
    _save(fig, out, "figure-17-operator-experiment")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="outputs/operator_experiment/records.jsonl")
    parser.add_argument("--out", default="analysis-output/jev-ablation")
    args = parser.parse_args()
    out = Path(args.out)
    recs = [json.loads(l) for l in Path(args.records).read_text().splitlines()]
    df = pd.DataFrame([r for r in recs if r.get("error") is None])
    df["refused"] = df.alignment.isin(REFUSALS)
    df["success"] = df.score >= SUCCESS
    df["high"] = df.categories.map(lambda c: any(v.get("is_related") and v.get("harm") in ("high_risk", "harmful")
                                                 for v in (c or {}).values()))
    df["real"] = ~(df.mutator_refusal | df.mutation_failed) | (df.operator == "echo")
    ops = per_operator(df)
    con = pd.concat([contrasts(df, "all"), contrasts(df[df.real], "real_attacks_only")])
    echo = df[df.operator == "echo"]
    retest = {"n": int(len(echo)), "mean_abs_diff": float((echo.score - echo.parent_score).abs().mean()),
              "same_alignment_band": float(((echo.score >= 50) == (echo.parent_score >= 50)).mean()),
              "spearman": float(echo.score.corr(echo.parent_score, method="spearman"))}
    summary = {"cells": int(len(df)), "parents": int(df.parent_id.nunique()), "operators": int(df.operator.nunique()),
               "judge_calls": int(df.judge_calls.sum()), "successes": int(df.success.sum()),
               "high_risk_or_harmful": int(df.high.sum()), "mutator_refusals": int(df.mutator_refusal.sum()),
               "mutation_failed": int(df.mutation_failed.sum()), "echo_retest": retest}
    ops.to_csv(out / "operator_experiment_per_operator.csv", index=False)
    con.to_csv(out / "operator_experiment_contrasts.csv", index=False)
    (out / "operator_experiment_summary.json").write_text(json.dumps(summary, indent=2))
    fig(ops, con, out / "figures")
    pd.set_option("display.width", 200)
    print(json.dumps(summary, indent=2))
    print(ops.round(2).to_string(index=False), "\n")
    print(con.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
