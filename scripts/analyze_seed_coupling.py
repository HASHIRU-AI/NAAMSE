"""How NAAMSE's shared per-task seed shapes the Jev ablation results (offline, no API calls).

Each parallel task re-seeds a fresh random.Random(task_seed) for the action choice, the mutation
operator choice, and the EXPLORE corpus draw, so all three start from the same first draw u:

- MUTATE (last action) happens only for high u, and the operator is then drawn with the same u
  over an alphabetically sorted CDF, so picks land on late-alphabet operators.
- EXPLORE (first action) happens only for u < P(explore), and the corpus offset is ~u * 2^17,
  so EXPLORE samples only the first P(explore) fraction of the corpus (ordered by source).

This is the configured behavior of the system under test. The script measures its effect:
1. Operator selection: realized picks vs. what Jev's logged probabilities intended.
2. Cluster breadth: re-draws each post-initial EXPLORE prompt's cluster from the coupled window
   and from the full corpus (other prompts keep their observed clusters), and reports how much of
   the Act=Jev vs. Baseline breadth gap the window explains. First-order only: downstream
   children of a different EXPLORE draw are not simulated.

Usage: uv run python scripts/analyze_seed_coupling.py [--runs outputs] [--out analysis-output/jev-ablation]
"""
import argparse
import collections
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DB = "src/cluster_engine/data_access/adversarial/naamse.db"
BITS17 = 131072  # randint(0, count-1) draws getrandbits(17) for a 129k-row corpus
N_SIM = 500
ARMS = {"act-static_thresholds_mut-uniform_fit-meta_obj-score": "Baseline",
        "act-jev_mut-uniform_fit-meta_obj-score": "Act=Jev",
        "act-uniform_mut-uniform_fit-meta_obj-score": "Act=uniform"}


def static_p_explore(score: float) -> float:
    return 0.7 if score < 50 else 0.1 if score < 100 else 0.4


def operator_tables(runs: Path) -> pd.DataFrame:
    """Per operator: realized Jev picks vs. expected picks under Jev's own probabilities."""
    realized, intended, n = collections.Counter(), collections.Counter(), 0
    for run_dir in runs.glob("act-*_mut-jev_*_seed*"):
        for line in (run_dir / "prompts.jsonl").read_text().splitlines():
            sel = json.loads(line).get("selector") or {}
            p = sel.get("mutation_probabilities")
            if not p:
                continue
            n += 1
            realized[sel["mutation_type"]] += 1
            total = sum(max(v, 0.0) for v in p.values())
            for k, v in p.items():
                intended[k] += max(v, 0.0) / total
    ops = sorted(set(realized) | set(intended))
    df = pd.DataFrame({"operator": ops, "realized": [realized[o] for o in ops],
                       "intended": [intended[o] for o in ops]})
    df["realized_share"] = df.realized / n
    df["intended_share"] = df.intended / n
    return df.sort_values("intended", ascending=False)


def corpus_clusters() -> np.ndarray:
    con = sqlite3.connect(DB)
    rows = [(r[0] or "").split("/")[0] for r in con.execute("SELECT cluster_id FROM prompts ORDER BY id")]
    con.close()
    return np.array(rows)


def explore_windows(runs: Path, clusters_csv: Path) -> Dict[str, List[dict]]:
    """Per run: list of prompts with observed top-level cluster and, for post-initial EXPLORE, P(explore)."""
    pc = pd.read_csv(clusters_csv)
    pc["pos"] = pc.groupby("run").cumcount()
    by_run = {}
    for run_dir in sorted(runs.glob("act-*_seed*")):
        arm = ARMS.get(run_dir.name.split("_seed")[0])
        if arm is None:
            continue
        obs = pc[pc.run == run_dir.name].set_index("pos").cluster
        items = []
        for i, line in enumerate((run_dir / "prompts.jsonl").read_text().splitlines()):
            r = json.loads(line)
            window = None
            if r["mutation_type"] == "explore" and r["iteration"] >= 1:
                ap = (r.get("selector") or {}).get("action_probabilities")
                window = (ap["explore"] / sum(ap.values())) if ap else \
                    (1 / 3 if arm == "Act=uniform" else static_p_explore(r["history"][-1]["score"]))
            items.append({"cluster": obs.get(i), "window": window})
        by_run[run_dir.name] = {"arm": arm, "items": items}
    return by_run


def simulate(by_run: Dict, corpus: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    n = len(corpus)
    rows = []
    for run, d in by_run.items():
        for mode in ("coupled", "decoupled"):
            vals = []
            for _ in range(N_SIM):
                cl = []
                for it in d["items"]:
                    if it["window"] is None:
                        cl.append(it["cluster"])
                        continue
                    hi = n if mode == "decoupled" else min(n, max(1, int(it["window"] * BITS17)))
                    cl.append(corpus[rng.integers(0, hi)])
                vals.append(len({c for c in cl if isinstance(c, str)}))
            rows.append({"run": run, "arm": d["arm"], "mode": mode, "n_clusters": float(np.mean(vals))})
        rows.append({"run": run, "arm": d["arm"], "mode": "observed",
                     "n_clusters": float(len({it["cluster"] for it in d["items"] if isinstance(it["cluster"], str)}))})
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="outputs")
    parser.add_argument("--out", default="analysis-output/jev-ablation")
    args = parser.parse_args()
    runs, out = Path(args.runs), Path(args.out)

    ops = operator_tables(runs)
    by_run = explore_windows(runs, out / "prompt_clusters.csv")
    sim = simulate(by_run, corpus_clusters(), np.random.default_rng(0))
    arm_means = sim.groupby(["mode", "arm"]).n_clusters.mean().unstack()
    gaps = (arm_means["Act=Jev"] - arm_means["Baseline"]).rename("Act=Jev − Baseline")
    windows = pd.DataFrame([{"arm": d["arm"], "window": it["window"]} for d in by_run.values()
                            for it in d["items"] if it["window"] is not None])

    ops.to_csv(out / "seed_coupling_operators.csv", index=False)
    sim.to_csv(out / "seed_coupling_clusters.csv", index=False)
    pd.set_option("display.width", 200)
    print(ops.round(3).to_string(index=False), "\n")
    print("Mean EXPLORE window (fraction of corpus reachable), post-initial:")
    print(windows.groupby("arm").window.agg(["count", "mean"]).round(3).to_string(), "\n")
    print(arm_means.round(2).to_string(), "\n")
    print(gaps.round(2).to_string())
    share = 1 - gaps["decoupled"] / gaps["coupled"] if gaps["coupled"] else float("nan")
    print(f"\nShare of the coupled gap explained by the EXPLORE window: {share:.0%}")


if __name__ == "__main__":
    main()
