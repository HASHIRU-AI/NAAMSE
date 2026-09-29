"""Budget-free robustness checks for the pooled action and operator analyses.

1. Off-policy action values. The direct-method estimate in analyze_pooled.py pools decisions
   made by different selectors, so action choice can be confounded with parent quality inside
   a bucket. Every decision's behavior propensity is known (static weights, 1/3 for uniform,
   the logged Jev probabilities), so we re-estimate each policy's value with self-normalized
   importance sampling (SNIPS) and doubly robust (DR) estimators, and separately from the
   Act=uniform arm alone, where actions are randomized independently of the parent.
2. Mutator refusals. The mutation LLM sometimes refuses to transform the parent and its
   refusal text becomes the "attack". We flag these children and recompute per-operator
   outcomes on the children that are actual attacks.

Usage: uv run python scripts/analyze_offpolicy.py [--runs outputs] [--out analysis-output/jev-ablation]
"""
import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from analyze_ablation import ACTIONS
from analyze_pooled import BUCKET_LABELS, SCORE_ARMS, cluster_boot, jev_policy, load_prompts, transitions

logger = logging.getLogger(__name__)

# Mutator refusals: STRICT is the opener-only pattern; BROAD adds refusal phrases anywhere
# in the first 400 characters (catches "I understand you are asking ... I can't help").
STRICT = re.compile(r"^\s*(I understand you're|I can(no|')t|I cannot|I'm sorry|I am sorry|Sorry|"
                    r"I won't|I will not|I'm not able|I am not able|I'm unable|As an AI)", re.I)
BROAD = re.compile(r"(I can(no|')t|I cannot|I won't|I will not|I'm unable|I am unable|I'm not able)"
                   r"( help| assist| provide| create| comply| fulfill| support| do that)"
                   r"|I understand you(’|')?(re| are) asking", re.I)
# Jailbreak templates that forbid refusals ("NEVER say 'Sorry', 'I can't assist'") are attacks
TEMPLATE = re.compile(r"never say", re.I)


def is_mutator_refusal(text: str) -> bool:
    """Refusal opener, or a refusal phrase early in the text that is not a quoted template rule.

    Also catches refusals written earlier in the lineage and wrapped by a later operator.
    """
    head = text[:400]
    return bool(STRICT.search(text) or (BROAD.search(head) and not TEMPLATE.search(head)))


def static_probs(score: float) -> np.ndarray:
    """The deployed static policy, exactly as in action_selectors.static_thresholds."""
    if score < 50:
        return np.array([0.7, 0.2, 0.1])
    if score < 80:
        return np.array([0.1, 0.7, 0.2])
    if score < 100:
        return np.array([0.1, 0.2, 0.7])
    return np.array([0.4, 0.4, 0.2])


def logged_jev_probs(runs_dir: Path) -> Dict[tuple, np.ndarray]:
    """(run, prompt index) -> normalized Jev action probabilities, for Act=Jev decisions."""
    out = {}
    for run_dir in runs_dir.glob("act-jev_mut-uniform_fit-meta_obj-score_seed*"):
        for i, line in enumerate((run_dir / "prompts.jsonl").read_text().splitlines()):
            p = (json.loads(line).get("selector") or {}).get("action_probabilities")
            if p:
                v = np.array([max(p.get(a, 0.0), 0.0) for a in ACTIONS])
                out[(run_dir.name, i)] = v / v.sum()
    return out


def with_propensities(df: pd.DataFrame, runs_dir: Path) -> pd.DataFrame:
    """Score-arm transitions with behavior propensity mu(a|x) and each target policy's pi(a|x)."""
    df = df.copy()
    df["idx"] = df.groupby("run").cumcount()
    t = transitions(df)
    jev_logged = logged_jev_probs(runs_dir)
    jev_bucket = jev_policy(df, runs_dir)
    a_idx = t.action.map({a: i for i, a in enumerate(ACTIONS)}).to_numpy()

    def behavior(row) -> np.ndarray:
        if row.arm == "Act=uniform":
            return np.full(3, 1 / 3)
        if row.arm == "Act=Jev":
            return jev_logged[(row.run, row.idx)]
        return static_probs(row.parent_score)

    mu = np.stack([behavior(r) for r in t.itertuples()])
    t["mu"] = mu[np.arange(len(t)), a_idx]
    t["pi_static"] = [static_probs(s)[a] for s, a in zip(t.parent_score, a_idx)]
    t["pi_uniform"] = 1 / 3
    # Jev is only observed on Act=Jev parents; elsewhere use its bucket-mean probabilities
    t["pi_Jev"] = [jev_logged[(r.run, r.idx)][a] if r.arm == "Act=Jev" else jev_bucket[r.bucket][a]
                   for r, a in zip(t.itertuples(), a_idx)]
    t["pi_Jev_vec"] = [jev_logged[(r.run, r.idx)] if r.arm == "Act=Jev" else jev_bucket[r.bucket]
                       for r in t.itertuples()]
    t["pi_static_vec"] = [static_probs(s) for s in t.parent_score]
    t["pi_uniform_vec"] = [np.full(3, 1 / 3)] * len(t)
    return t


def snips(t: pd.DataFrame, policy: str, outcome: str) -> float:
    w = t[f"pi_{policy}"] / t.mu
    return float((w * t[outcome]).sum() / w.sum())


def doubly_robust(t: pd.DataFrame, policy: str, outcome: str) -> float:
    """DR with a bucket x action mean as the outcome model (skips cells with no data)."""
    q = t.groupby(["bucket", "action"], observed=True)[outcome].mean()
    vals = []
    for r in t.itertuples():
        qs = np.array([q.get((r.bucket, a), np.nan) for a in ACTIONS])
        if np.isnan(qs).any():
            continue
        dm = float(np.dot(getattr(r, f"pi_{policy}_vec"), qs))
        w = getattr(r, f"pi_{policy}") / r.mu
        vals.append(dm + w * (getattr(r, outcome) - q[(r.bucket, r.action)]))
    return float(np.mean(vals)) if vals else np.nan


def offpolicy_table(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in ("score", "refused"):
        for est_name, est in (("SNIPS", snips), ("DR", doubly_robust)):
            for pol in ("static", "uniform", "Jev"):
                lo, hi = cluster_boot(t, lambda s: est(s, pol, outcome))
                rows.append({"outcome": outcome, "estimator": est_name, "policy": pol,
                             "value": est(t, pol, outcome), "ci_lo": lo, "ci_hi": hi})
            diff = lambda s: est(s, "Jev", outcome) - est(s, "static", outcome)
            lo, hi = cluster_boot(t, diff)
            rows.append({"outcome": outcome, "estimator": est_name, "policy": "Jev − static",
                         "value": diff(t), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def uniform_arm_table(t: pd.DataFrame) -> pd.DataFrame:
    """Per-bucket action outcomes from Act=uniform only (randomized, so unconfounded)."""
    u = t[t.arm == "Act=uniform"]
    rows = []
    for (b, a), g in u.groupby(["bucket", "action"], observed=True):
        lo, hi = cluster_boot(g, lambda s: s.score.mean()) if len(g) > 1 else (np.nan, np.nan)
        rows.append({"bucket": b, "action": a, "n": len(g), "mean_child": g.score.mean(),
                     "ci_lo": lo, "ci_hi": hi, "refusal": g.refused.mean()})
    return pd.DataFrame(rows)


def mutator_refusals(runs_dir: Path) -> pd.DataFrame:
    """One row per MUTATE child in the Meta-judged arms, with mutator-refusal flags."""
    rows = []
    for run_dir in sorted(runs_dir.glob("act-*_seed*")):
        if "fit-jev" in run_dir.name or not (run_dir / "final_state.json").exists():
            continue
        recs = [json.loads(line) for line in (run_dir / "prompts.jsonl").read_text().splitlines()]
        texts = [" ".join(str(p) for p in q["prompt"]).strip() for q in recs]
        for r, text in zip(recs, texts):
            if r["mutation_type"] in ("explore", "similar"):
                continue
            # Parent = earlier prompt in the run whose score matches the lineage's last score
            parent_score = r["history"][-1]["score"]
            parents = {t for q, t in zip(recs, texts)
                       if q["iteration"] < r["iteration"] and abs(q["judge_score"] - parent_score) < 1e-9}
            rows.append({"run": run_dir.name, "operator": r["mutation_type"],
                         "identical_to_parent": text in parents,
                         "selector": "Jev" if "_mut-jev_" in run_dir.name else "uniform",
                         "score": r["judge_score"],
                         "refused": (r.get("judge_results") or {}).get("alignment") in
                         ("full_refusal", "weak_refusal"),
                         "strict": bool(STRICT.search(text)),
                         "broad": is_mutator_refusal(text)})
    return pd.DataFrame(rows)


def operator_table_clean(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for op, g in m.groupby("operator"):
        real = g[~g.broad]
        rows.append({"operator": op, "n": len(g), "mutator_refused": int(g.broad.sum()),
                     "mean_all": g.score.mean(), "refusal_all": g.refused.mean(),
                     "n_real": len(real), "mean_real": real.score.mean() if len(real) else np.nan,
                     "refusal_real": real.refused.mean() if len(real) else np.nan})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="outputs")
    parser.add_argument("--out", default="analysis-output/jev-ablation")
    args = parser.parse_args()
    runs, out = Path(args.runs), Path(args.out)

    df = load_prompts(runs)
    t = with_propensities(df[df.arm.isin(SCORE_ARMS)], runs)
    op = offpolicy_table(t)
    ua = uniform_arm_table(t)
    m = mutator_refusals(runs)
    ops = operator_table_clean(m)

    op.to_csv(out / "offpolicy_values.csv", index=False)
    ua.to_csv(out / "uniform_arm_transitions.csv", index=False)
    ops.to_csv(out / "operators_mutator_refusals.csv", index=False)
    m.to_csv(out / "mutate_children.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(f"transitions: {len(t)}; max importance weight {(t[[f'pi_{p}' for p in ('static', 'Jev')]].max(axis=1) / t.mu).max():.1f}\n")
    print(op.round(3).to_string(), "\n")
    print(ua.round(2).to_string(), "\n")
    print(f"MUTATE children: {len(m)}; mutator refusals strict={m.strict.sum()} broad={m.broad.sum()}")
    print(m.groupby("selector")[["strict", "broad"]].mean().round(3).to_string(), "\n")
    print(m.groupby("broad").agg(n=("score", "size"), mean=("score", "mean"), max=("score", "max"),
                                 target_refusal=("refused", "mean")).round(2).to_string(), "\n")
    print(ops.round(2).to_string())


if __name__ == "__main__":
    main()
