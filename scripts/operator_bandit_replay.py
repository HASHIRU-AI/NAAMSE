"""Offline replay of operator selectors on the controlled operator grid (no API calls).

The controlled experiment observed every (parent, operator) cell once, so any operator policy
can be replayed against it without new queries: each simulated decision draws a parent at
random, the policy picks an operator, and the reward is that cell's recorded judge score.

Policies: uniform, Jev's intended preferences (renormalized over the tested operators),
UCB1, Gaussian Thompson sampling, and the best fixed operator (in-sample ceiling).
Horizons span a single run's MUTATE budget (~10 decisions) up to a long campaign.

Usage: uv run python scripts/operator_bandit_replay.py [--out analysis-output/jev-ablation]
"""
import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_ablation import _save

HORIZONS = [5, 10, 20, 50, 100, 200]
N_CAMPAIGNS = 4000
SEED = 0
SHORT_OP = lambda o: o.replace("_mutation", "")
COLORS = {"uniform": "#9e9e9e", "Jev intended": "#ff7f0e", "UCB1": "#1f77b4",
          "Thompson": "#2ca02c", "best fixed (oracle)": "#000000"}


def load_grid(records: Path) -> pd.DataFrame:
    recs = [json.loads(l) for l in records.read_text().splitlines()]
    df = pd.DataFrame([r for r in recs if r.get("error") is None])
    grid = df.pivot_table(index="parent_id", columns="operator", values="score", aggfunc="first")
    grid = grid.dropna()  # keep parents observed under every operator
    if grid.empty:
        raise SystemExit("No parent has been observed under every operator yet; finish the experiment first.")
    return grid


def jev_weights(ops: List[str], out: Path) -> np.ndarray:
    intended = pd.read_csv(out / "seed_coupling_operators.csv").set_index("operator").intended_share
    w = np.array([intended.get(o, 0.0) for o in ops], float)
    return w / w.sum()


def run_policy(policy: str, grid: np.ndarray, horizon: int, rng: np.random.Generator,
               jev_w: np.ndarray, best: int) -> float:
    n_par, n_ops = grid.shape
    counts, sums, sq = np.zeros(n_ops), np.zeros(n_ops), np.zeros(n_ops)
    total = 0.0
    for t in range(1, horizon + 1):
        p = rng.integers(n_par)
        if policy == "uniform":
            a = rng.integers(n_ops)
        elif policy == "Jev intended":
            a = rng.choice(n_ops, p=jev_w)
        elif policy == "best fixed (oracle)":
            a = best
        elif policy == "UCB1":
            if (counts == 0).any():
                a = rng.choice(np.flatnonzero(counts == 0))
            else:
                a = int(np.argmax(sums / counts + 100 * np.sqrt(2 * np.log(t) / counts)))
        elif policy == "Thompson":
            mean = np.where(counts > 0, sums / np.maximum(counts, 1), 50.0)
            var = np.where(counts > 1, (sq / np.maximum(counts, 1) - mean ** 2), 20.0 ** 2)
            a = int(np.argmax(rng.normal(mean, np.sqrt(np.maximum(var, 1.0) / np.maximum(counts, 1)))))
        r = grid[p, a]
        counts[a] += 1
        sums[a] += r
        sq[a] += r * r
        total += r
    return total / horizon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="outputs/operator_experiment/records.jsonl")
    parser.add_argument("--out", default="analysis-output/jev-ablation")
    args = parser.parse_args()
    out = Path(args.out)
    grid_df = load_grid(Path(args.records))
    ops = list(grid_df.columns)
    grid = grid_df.to_numpy(float)
    best = int(np.argmax(grid.mean(axis=0)))
    jev_w = jev_weights(ops, out)
    rng = np.random.default_rng(SEED)
    rows = []
    for policy in COLORS:
        for h in HORIZONS:
            vals = np.array([run_policy(policy, grid, h, rng, jev_w, best) for _ in range(N_CAMPAIGNS)])
            rows.append({"policy": policy, "horizon": h, "mean_score": vals.mean(),
                         "ci_lo": np.percentile(vals, 2.5), "ci_hi": np.percentile(vals, 97.5)})
    res = pd.DataFrame(rows)
    meta = {"parents": int(grid.shape[0]), "operators": ops, "best_fixed": ops[best],
            "operator_means": dict(zip(ops, np.round(grid.mean(axis=0), 2))),
            "jev_intended_weights": dict(zip(ops, np.round(jev_w, 3)))}
    res.to_csv(out / "operator_bandit_replay.csv", index=False)
    (out / "operator_bandit_replay.json").write_text(json.dumps(meta, indent=2))

    best_name = ops[best]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for policy, color in COLORS.items():
        r = res[res.policy == policy]
        ls = "--" if "oracle" in policy else "-"
        label = f"best fixed: {SHORT_OP(best_name)} (in-sample)" if "oracle" in policy else policy
        ax.plot(r.horizon, r.mean_score, ls, color=color, marker="o", ms=4, lw=2, label=label)
    ax.axvspan(4, 11, color="0.92", zorder=0)
    ax.text(6.6, res[res.policy == "uniform"].mean_score.min() - 0.45, "one run's\nMUTATE budget", fontsize=7,
            color="0.35", ha="center", va="top")
    ax.set_xscale("log")
    ax.set_xticks(HORIZONS, [str(h) for h in HORIZONS])
    ax.set_xlabel("Operator decisions in the campaign", fontsize=8)
    ax.set_ylabel("Mean judge score per decision", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Operator selectors replayed on the controlled grid", fontsize=9)
    _save(fig, out / "figures", "figure-18-operator-bandit-replay")
    pd.set_option("display.width", 200)
    print(json.dumps(meta, indent=2))
    print(res.pivot(index="policy", columns="horizon", values="mean_score").round(2).to_string())


if __name__ == "__main__":
    main()
