"""Corpus-cluster coverage of the Jev ablation sweep (no API calls).

The runner does not log cluster_info, so clusters are recovered offline:
  * prompts found verbatim in the corpus DB get their exact cluster;
  * other prompts (rewritten by a mutation) get the majority top-level cluster of their
    k nearest corpus prompts, embedded with the engine's all-MiniLM-L6-v2 model.
Assignment accuracy is estimated by leave-one-out kNN on the exactly matched prompts.

Usage: uv run python scripts/analyze_clusters.py [--runs outputs] [--out analysis-output/jev-ablation]
"""
import argparse
import itertools
import json
import logging
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_ablation import ARMS, CONTRASTS, REFUSALS, SUCCESS, _save, boot_ci, boot_diff_ci, cliffs_delta, holm, perm_p

logger = logging.getLogger(__name__)

DB = "src/cluster_engine/data_access/adversarial/naamse.db"
LOOKUP = "src/cluster_engine/data_access/adversarial/cluster_lookup_table.json"
K = 10
METRICS = ["n_clusters", "cluster_entropy", "n_clusters_exact"]


def top(cluster_id: Optional[str]) -> Optional[str]:
    return cluster_id.split("/")[0] if isinstance(cluster_id, str) and cluster_id else None


def load_corpus(db: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    """Corpus embeddings (L2-normalized), their top-level clusters, and text -> cluster_id."""
    con = sqlite3.connect(db)
    rows = con.execute("""SELECT p.user_content, p.cluster_id, c.embedding_vector
                          FROM prompts p JOIN centroids c ON c.prompt_id = p.id
                          WHERE p.source != 'NAAMSE_mutation' ORDER BY p.id""").fetchall()
    con.close()
    emb = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    tops = np.array([top(r[1]) for r in rows])
    text_to_cluster = {r[0]: r[1] for r in rows}
    return emb, tops, text_to_cluster


def knn_top(queries: np.ndarray, emb: np.ndarray, tops: np.ndarray, k: int = K,
            exclude_self: bool = False) -> List[str]:
    """Majority top-level cluster of the k nearest corpus prompts (cosine)."""
    out = []
    for q in queries:
        sims = emb @ q
        idx = np.argpartition(-sims, k + 1)[:k + 1]
        idx = idx[np.argsort(-sims[idx])]
        if exclude_self:
            idx = idx[1:]  # drop the prompt itself (similarity ~1)
        out.append(Counter(tops[idx[:k]]).most_common(1)[0][0])
    return out


def assign_clusters(runs_dir: Path, emb: np.ndarray, tops: np.ndarray,
                    text_to_cluster: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    from sentence_transformers import SentenceTransformer

    rows = []
    for run_dir in sorted(runs_dir.glob("act-*")):
        if not (run_dir / "final_state.json").exists():
            continue
        cfg = json.loads((run_dir / "config.json").read_text())
        if cfg["arm"] not in ARMS:
            continue
        for line in (run_dir / "prompts.jsonl").read_text().splitlines():
            r = json.loads(line)
            mt = r["mutation_type"]
            rows.append({"arm": ARMS[cfg["arm"]], "seed": cfg["run"]["seed"], "run": run_dir.name,
                         "iteration": r["iteration"], "text": r["prompt"][0], "score": r["judge_score"],
                         "action": mt if mt in ("explore", "similar") else "mutate", "mutation_type": mt,
                         "refused": (r.get("judge_results") or {}).get("alignment") in REFUSALS})
    df = pd.DataFrame(rows)

    df["cluster_id"] = df.text.map(text_to_cluster)
    df["exact"] = df.cluster_id.notna()
    df["cluster"] = df.cluster_id.map(top)

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    novel = df[~df.exact]
    q = model.encode(novel.text.tolist(), normalize_embeddings=True, batch_size=32)
    df.loc[~df.exact, "cluster"] = knn_top(q, emb, tops)

    # Leave-one-out accuracy of kNN assignment on prompts whose true cluster is known
    known = df[df.exact].drop_duplicates("text")
    qk = model.encode(known.text.tolist(), normalize_embeddings=True, batch_size=32)
    pred = knn_top(qk, emb, tops, exclude_self=True)
    acc = {"n_exact": int(df.exact.sum()), "n_knn": int((~df.exact).sum()),
           "knn_loo_accuracy": float(np.mean(np.array(pred) == known.cluster.to_numpy())),
           "knn_loo_n": len(known), "n_top_level_clusters": int(len(set(tops)))}
    return df, acc


def run_metrics(df: pd.DataFrame) -> pd.DataFrame:
    def per_run(g: pd.DataFrame) -> pd.Series:
        p = g.cluster.value_counts(normalize=True).to_numpy()
        return pd.Series({"n_clusters": g.cluster.nunique(),
                          "cluster_entropy": float(-(p * np.log2(p)).sum()),
                          "n_clusters_exact": g[g.exact].cluster.nunique()})
    return df.groupby(["arm", "seed"]).apply(per_run, include_groups=False).reset_index()


def random_reference(tops: np.ndarray, n_draws: int = 28, sims: int = 5000, seed: int = 0) -> Tuple[float, float]:
    """Distinct top-level clusters from n_draws uniform corpus samples (EXPLORE-only search)."""
    rng = np.random.default_rng(seed)
    d = [len(set(tops[rng.integers(0, len(tops), n_draws)])) for _ in range(sims)]
    return float(np.mean(d)), float(np.std(d))


def contrasts(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        block = []
        for treat, ctrl, rq in CONTRASTS:
            a = metrics[metrics.arm == treat][metric].to_numpy(float)
            b = metrics[metrics.arm == ctrl][metric].to_numpy(float)
            lo, hi = boot_diff_ci(a, b)
            block.append({"metric": metric, "rq": rq, "treatment": treat, "control": ctrl,
                          "treat_mean": a.mean(), "ctrl_mean": b.mean(), "diff": a.mean() - b.mean(),
                          "diff_ci_lo": lo, "diff_ci_hi": hi, "p_perm": perm_p(a, b),
                          "cliffs_delta": cliffs_delta(a, b)})
        for row, p in zip(block, holm([r["p_perm"] for r in block])):
            row["p_holm"] = p
        rows += block
    return pd.DataFrame(rows)


def cluster_outcomes(df: pd.DataFrame, labels: Dict[str, str]) -> pd.DataFrame:
    """Per top-level cluster outcomes, pooled over Meta-judged arms."""
    m = df[df.arm != "Fit=Jev*"]
    out = m.groupby("cluster").agg(n=("score", "size"), n_runs=("run", "nunique"), mean_score=("score", "mean"),
                                   refusal=("refused", "mean"), n_success=("score", lambda s: int((s >= SUCCESS).sum())))
    out["label"] = [labels.get(c, "(unlabeled)") for c in out.index]
    return out.sort_values("mean_score", ascending=False).reset_index()


# ---------------------------------------------------------------- figures
def fig_breadth(metrics: pd.DataFrame, ref: Tuple[float, float], out: Path) -> None:
    arms = list(ARMS.values())
    fig, ax = plt.subplots(figsize=(8, 4))
    rng = np.random.default_rng(0)
    for i, arm in enumerate(arms):
        v = metrics[metrics.arm == arm].n_clusters.to_numpy(float)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=18, alpha=0.6,
                   color="0.55" if arm == "Fit=Jev*" else ("C1" if arm.startswith("Cov") else "C0"))
        lo, hi = boot_ci(v) if v.std() > 0 else (v.mean(), v.mean())
        ax.errorbar(i, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt="D", color="k", ms=5, capsize=3)
    ax.axhspan(ref[0] - ref[1], ref[0] + ref[1], color="C2", alpha=0.15)
    ax.axhline(ref[0], color="C2", ls="--", lw=1, label=f"28 uniform corpus draws ({ref[0]:.1f} ± {ref[1]:.1f})")
    ax.set_xticks(range(len(arms)), arms, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Distinct top-level clusters / run")
    ax.set_ylim(5.5, 19)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Corpus-cluster breadth per run (dots = seeds, ◆ = mean, 95% seed-bootstrap CI)", fontsize=9)
    _save(fig, out, "figure-11-cluster-breadth")


def fig_cluster_outcomes(co: pd.DataFrame, out: Path) -> None:
    co = co[co.n >= 5].sort_values("refusal")
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.3 * len(co) + 1.4), sharey=True)
    y = np.arange(len(co))
    names = [f"{c}: {l[:38]} (n={n})" for c, l, n in zip(co.cluster, co.label, co.n)]
    axes[0].barh(y, co.refusal, color="C0")
    axes[1].barh(y, co.mean_score, color="C3")
    axes[1].axvline(SUCCESS, color="0.7", ls=":", lw=0.8)
    axes[0].set_yticks(y, names, fontsize=7)
    axes[0].set_xlabel("Refusal rate")
    axes[1].set_xlabel("Mean judge score")
    for yi, s in zip(y, co.n_success):
        if s:
            axes[1].annotate(f"{s} ≥80", (co.mean_score.iloc[yi] + 1, yi), va="center", fontsize=7)
    fig.suptitle("Outcomes by corpus cluster (top-level, n ≥ 5), pooled over Meta-judged arms", fontsize=9)
    _save(fig, out, "figure-12-cluster-outcomes")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="outputs")
    parser.add_argument("--out", default="analysis-output/jev-ablation")
    args = parser.parse_args()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    emb, tops, text_to_cluster = load_corpus(DB)
    df, acc = assign_clusters(Path(args.runs), emb, tops, text_to_cluster)
    labels = {k: (v["label"] if isinstance(v, dict) else v) for k, v in json.loads(Path(LOOKUP).read_text()).items()}
    metrics = run_metrics(df)
    ref = random_reference(tops)
    acc["random_28_draws_mean"], acc["random_28_draws_std"] = ref
    con = contrasts(metrics)
    co = cluster_outcomes(df, labels)
    by_action = df[df.arm != "Fit=Jev*"].groupby("action").apply(
        lambda g: pd.Series({"n": len(g), "distinct_clusters": g.cluster.nunique()}), include_groups=False)

    df.drop(columns=["text"]).to_csv(out / "prompt_clusters.csv", index=False)
    metrics.to_csv(out / "cluster_run_metrics.csv", index=False)
    con.to_csv(out / "cluster_contrasts.csv", index=False)
    co.to_csv(out / "cluster_outcomes.csv", index=False)
    (out / "cluster_assignment.json").write_text(json.dumps(acc, indent=2))
    fig_breadth(metrics, ref, out / "figures")
    fig_cluster_outcomes(co, out / "figures")

    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(json.dumps(acc, indent=2))
    print(metrics.groupby("arm")[METRICS].agg(["mean", "std"]).loc[list(ARMS.values())].round(2).to_string(), "\n")
    print(con.round(3).to_string(), "\n")
    print(co.round(2).to_string(), "\n")
    print(by_action.to_string())


if __name__ == "__main__":
    main()
