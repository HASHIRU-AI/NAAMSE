"""Figures 13 to 16 for the follow-up analyses (no API calls; reads the follow-up outputs).

13  Jev operator selection: intended (Jev probabilities) vs. realized picks
14  Description ablation: Jev's mean operator probabilities under five framings
15  Cross-model annotation: alignment agreement with Meta, and severity of the 13 successes
16  Seed coupling: EXPLORE corpus offset vs. the policy's P(explore)

Usage: uv run python scripts/make_followup_figures.py [--out analysis-output/jev-ablation]
"""
import argparse
import bisect
import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_ablation import _save
from analyze_referee import LEVELS
from analyze_seed_coupling import BITS17, DB, static_p_explore

BLUE, ORANGE = "#1f77b4", "#ff7f0e"
HARM = ["not_harmful", "low_risk", "high_risk", "harmful"]
SHORT = lambda o: o.replace("_mutation", "").replace("_attack", "")


def fig_operator_intended(out: Path, figs: Path) -> None:
    ops = pd.read_csv(out / "seed_coupling_operators.csv")
    ops = ops[(ops.intended_share >= 0.01) | (ops.realized > 0)].sort_values("intended_share")
    y = np.arange(len(ops))
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(ops) + 1.3))
    ax.barh(y + 0.2, ops.intended_share, 0.38, color=BLUE, label="Intended (mean Jev probability)")
    ax.barh(y - 0.2, ops.realized_share, 0.38, color=ORANGE, label="Realized share of 35 picks")
    for yi, a, b in zip(y, ops.intended_share, ops.realized_share):
        ax.text(a + 0.005, yi + 0.2, f"{a:.2f}", va="center", fontsize=6.5)
        if b > 0:
            ax.text(b + 0.005, yi - 0.2, f"{b:.2f}", va="center", fontsize=6.5)
    ax.set_yticks(y, [SHORT(o) for o in ops.operator], fontsize=7.5)
    ax.set_xlabel("Share of MUTATE decisions (Mut=Jev arms)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title("Jev operator selection: intended vs. realized under the shared per-task seed", fontsize=9)
    _save(fig, figs, "figure-13-operator-intended-vs-realized")


def fig_description_ablation(out: Path, figs: Path) -> None:
    s = json.loads((out / "jev_description_ablation.json").read_text())
    conds = list(s["mean_probs"])
    mp = pd.DataFrame(s["mean_probs"])  # operators x conditions
    top = mp.max(axis=1).sort_values(ascending=False).index[:12]
    mat = mp.loc[top, conds].to_numpy()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max(), aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}" if v >= 0.005 else "·", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 0.6 * mat.max() else "0.15")
    ax.set_xticks(range(len(conds)), [c.replace("_", " ") for c in conds], fontsize=7.5)
    ax.set_yticks(range(len(top)), [SHORT(o) for o in top], fontsize=7.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Mean probability over 66 parents", fontsize=7.5)
    sw = s["swapped_follows"]
    ax.set_title(f"Jev operator preference by framing (swapped: follows description ρ={sw['corr_with_original_by_description']:.2f},\n"
                 f"name ρ={sw['corr_with_original_by_name']:.2f}; swapped column indexed by key)", fontsize=8.5)
    _save(fig, figs, "figure-14-description-ablation")


def fig_annotation(out: Path, figs: Path) -> None:
    j = pd.read_csv(out / "labeling" / "labels_claude_joined.csv")
    key = pd.read_csv(out / "labeling" / "label_key.csv")
    lab = {json.loads(l)["item_id"]: json.loads(l) for l in (out / "labeling" / "labels_claude.jsonl").read_text().splitlines()}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.8), gridspec_kw={"width_ratios": [1, 1.1]})
    conf = pd.crosstab(j.meta_alignment, j.label_alignment).reindex(index=LEVELS, columns=LEVELS, fill_value=0)
    a1.imshow(conf.to_numpy(), cmap="Blues", aspect="auto")
    for (r, c), v in np.ndenumerate(conf.to_numpy()):
        a1.text(c, r, str(v), ha="center", va="center", fontsize=8, color="white" if v > conf.to_numpy().max() * 0.6 else "0.15")
    ticks = [l.replace("_", "\n") for l in LEVELS]
    a1.set_xticks(range(4), ticks, fontsize=7)
    a1.set_yticks(range(4), ticks, fontsize=7)
    a1.set_xlabel("Independent annotator", fontsize=8)
    a1.set_ylabel("Meta (fitness judge or referee)", fontsize=8)
    a1.set_title("Alignment, 103 prompts (κ = 0.74)", fontsize=9)

    succ = key[key.stratum == "success"]
    order = lambda d: max(d, key=HARM.index) if d else "not_harmful"
    judge = [order(list(json.loads(v).values())) for v in succ.harm_verdicts]
    annot = [order([c["harm"] for c in lab[i]["categories"].values() if c["related"]]) for i in succ.item_id]
    counts = pd.DataFrame({"Judge (own)": pd.Series(judge).value_counts(),
                           "Annotator": pd.Series(annot).value_counts()}).reindex(HARM).fillna(0)
    shades = [plt.cm.Reds(x) for x in (0.2, 0.4, 0.65, 0.9)]
    x = np.arange(2)
    bottom = np.zeros(2)
    for lvl, col in zip(HARM, shades):
        vals = counts.loc[lvl].to_numpy()
        a2.bar(x, vals, 0.55, bottom=bottom, color=col, edgecolor="white", linewidth=2, label=lvl.replace("_", " "))
        for xi, v, b in zip(x, vals, bottom):
            if v:
                a2.text(xi, b + v / 2, f"{int(v)}", ha="center", va="center", fontsize=8,
                        color="white" if lvl in ("high_risk", "harmful") else "0.15")
        bottom += vals
    a2.set_xticks(x, counts.columns, fontsize=8)
    a2.set_ylabel("Successes (score ≥ 80)", fontsize=8)
    a2.spines[["top", "right"]].set_visible(False)
    a2.legend(title="Strongest harm verdict", fontsize=7, title_fontsize=7, frameon=False, bbox_to_anchor=(1.0, 1.0))
    a2.set_title("Severity of the 13 successes", fontsize=9)
    _save(fig, figs, "figure-15-cross-model-annotation")


def fig_explore_window(runs: Path, figs: Path) -> None:
    con = sqlite3.connect(DB)
    ids = [r[0] for r in con.execute("SELECT id FROM prompts ORDER BY id")]
    pts = {"static": [], "Jev": []}
    for run_dir in runs.glob("act-*obj-score_seed*"):
        if "_fit-jev_" in run_dir.name or "act-uniform" in run_dir.name:
            continue
        for line in (run_dir / "prompts.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["mutation_type"] != "explore" or r["iteration"] < 1:
                continue
            q = con.execute("SELECT id FROM prompts WHERE user_content = ? LIMIT 1", (r["prompt"][0],)).fetchone()
            if not q:
                continue
            u = bisect.bisect_left(ids, q[0]) / BITS17
            ap = (r.get("selector") or {}).get("action_probabilities")
            if ap:
                pts["Jev"].append((ap["explore"] / sum(ap.values()), u))
            else:
                pts["static"].append((static_p_explore(r["history"][-1]["score"]), u))
    con.close()
    fig, ax = plt.subplots(figsize=(5, 4))
    rng = np.random.default_rng(0)
    for (name, v), col, mk in zip(pts.items(), (BLUE, ORANGE), ("o", "s")):
        p, u = np.array(v).T
        jitter = rng.uniform(-0.012, 0.012, len(p)) if name == "static" else 0
        ax.scatter(p + jitter, u, s=16, color=col, marker=mk, alpha=0.75, edgecolor="white", linewidth=0.5,
                   label=f"{name} policy (n={len(p)})")
    ax.plot([0, 1], [0, 1], color="0.5", ls="--", lw=1)
    ax.text(0.62, 0.7, "offset = P(explore)", fontsize=7, color="0.35", rotation=38)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("P(explore) of the decision", fontsize=8)
    ax.set_ylabel("Corpus offset of the EXPLORE draw (fraction of 2^17)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("Every post-initial EXPLORE draw lies below its P(explore)", fontsize=9)
    _save(fig, figs, "figure-16-explore-window")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="outputs")
    parser.add_argument("--out", default="analysis-output/jev-ablation")
    args = parser.parse_args()
    out, runs = Path(args.out), Path(args.runs)
    figs = out / "figures"
    fig_operator_intended(out, figs)
    fig_description_ablation(out, figs)
    fig_annotation(out, figs)
    fig_explore_window(runs, figs)
    print("wrote figures 13-16 to", figs)


if __name__ == "__main__":
    main()
