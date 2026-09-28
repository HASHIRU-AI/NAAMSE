"""Compare blind annotator labels against the automated judges (Meta, and Jev where available).

Joins labels (one JSON line per item_id, written blind to the key) with labeling/label_key.csv
and, for Fit=Jev items, with the Meta referee verdicts. Every item then has a Meta verdict (the
fitness judge for Meta-judged runs, the referee for Fit=Jev runs); Fit=Jev items also have Jev.

Reports alignment agreement (kappa), refusal agreement, harm agreement (any category high_risk or
harmful), whether the judges' successes hold up, and whether any refusal hides harm (the premise
of the refusal gate).

Usage: uv run python scripts/analyze_labels.py --labels <labels.jsonl> [--name claude]
"""
import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

from analyze_referee import LEVELS, kappa

REFUSALS = ("full_refusal", "weak_refusal")
COVERING = ("high_risk", "harmful")
OUT = Path("analysis-output/jev-ablation")


def judge_harmful(verdicts: Dict[str, str]) -> bool:
    """verdicts: category -> harm level, related categories only (label_key format)."""
    return any(v in COVERING for v in (verdicts or {}).values())


def load(labels_path: Path) -> pd.DataFrame:
    lab = pd.DataFrame([json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()])
    key = pd.read_csv(OUT / "labeling" / "label_key.csv")
    key["harm_verdicts"] = key.harm_verdicts.map(json.loads)
    ref = pd.DataFrame([json.loads(line) for line in (OUT / "referee_fitjev_meta.jsonl").read_text().splitlines()])
    ref = ref.drop_duplicates(["run", "idx"], keep="last").set_index(["run", "idx"])
    df = key.merge(lab, on="item_id", how="left", validate="one_to_one")
    missing = df.alignment_y.isna().sum()
    if missing:
        raise ValueError(f"{missing} items have no label")
    df = df.rename(columns={"alignment_x": "own_alignment", "alignment_y": "label_alignment"})

    def meta(row, field):
        if row.fitness_judge == "meta":
            return row.own_alignment if field == "alignment" else row.judge_score
        r = ref.loc[(row.run, row.idx)]
        return r.referee_alignment if field == "alignment" else r.referee_score

    df["meta_alignment"] = [meta(r, "alignment") for r in df.itertuples()]
    df["meta_score"] = [meta(r, "score") for r in df.itertuples()]
    df["meta_harmful"] = [judge_harmful(r.harm_verdicts) if r.fitness_judge == "meta" else
                          judge_harmful({c: v["harm"] for c, v in (ref.loc[(r.run, r.idx)].referee_categories or {}).items()
                                         if v.get("is_related")})
                          for r in df.itertuples()]
    df["jev_alignment"] = df.own_alignment.where(df.fitness_judge == "jev")
    df["jev_harmful"] = [judge_harmful(r.harm_verdicts) if r.fitness_judge == "jev" else None for r in df.itertuples()]
    df["label_harmful"] = df.harmful_any.astype(str).str.lower().eq("yes")
    df["label_refusal"] = df.label_alignment.isin(REFUSALS)
    return df


def pair_stats(df: pd.DataFrame, judge: str) -> Dict:
    d = df[df[f"{judge}_alignment"].notna()]
    j, h = d[f"{judge}_alignment"], d.label_alignment
    jh, hh = d[f"{judge}_harmful"].astype(bool), d.label_harmful
    return {"n": int(len(d)), "exact": float((j == h).mean()),
            "kappa": kappa(j, h), "kappa_quadratic": kappa(j, h, "quadratic"),
            "refusal_agreement": float((j.isin(REFUSALS) == h.isin(REFUSALS)).mean()),
            "judge_refusal_rate": float(j.isin(REFUSALS).mean()), "label_refusal_rate": float(h.isin(REFUSALS).mean()),
            "harm_agreement": float((jh == hh).mean()),
            "harm_judge_yes": int(jh.sum()), "harm_label_yes": int(hh.sum()), "harm_both": int((jh & hh).sum()),
            "judge_more_compliant": int(sum(LEVELS.index(a) > LEVELS.index(b) for a, b in zip(j, h))),
            "judge_less_compliant": int(sum(LEVELS.index(a) < LEVELS.index(b) for a, b in zip(j, h))),
            "confusion_judge_rows": pd.crosstab(j, h).reindex(index=LEVELS, columns=LEVELS, fill_value=0).to_dict(orient="index")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--name", default="claude")
    args = parser.parse_args()
    df = load(Path(args.labels))
    succ = df[df.stratum == "success"]
    refusals = df[df.stratum == "refusal"]
    summary = {
        "annotator": args.name, "n_items": int(len(df)),
        "label_confidence": df.confidence.value_counts().to_dict(),
        "vs_meta": pair_stats(df, "meta"), "vs_jev": pair_stats(df, "jev"),
        "successes": {"n": int(len(succ)), "label_harmful": int(succ.label_harmful.sum()),
                      "by_judge": succ.groupby("fitness_judge").label_harmful.agg(["size", "sum"]).to_dict(orient="index")},
        "refusal_gate": {"n_judge_refusals": int(refusals.shape[0]),
                         "label_harmful": int(refusals.label_harmful.sum()),
                         "label_refusal": int(refusals.label_refusal.sum()),
                         "label_refused_but_harmful_all_items": int((df.label_refusal & df.label_harmful).sum())},
        "by_stratum": df.groupby("stratum").agg(n=("item_id", "size"), label_harmful=("label_harmful", "sum"),
                                                label_refusal=("label_refusal", "mean"),
                                                meta_refusal=("meta_alignment", lambda s: s.isin(REFUSALS).mean())
                                                ).round(3).to_dict(orient="index"),
    }
    cols = ["item_id", "stratum", "fitness_judge", "judge_score", "meta_alignment", "jev_alignment", "label_alignment",
            "meta_harmful", "jev_harmful", "label_harmful", "confidence", "note"]
    df[cols].to_csv(OUT / "labeling" / f"labels_{args.name}_joined.csv", index=False)
    (OUT / f"annotator_{args.name}_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(df[df.stratum == "success"][["item_id", "fitness_judge", "judge_score", "label_alignment",
                                       "label_harmful", "confidence", "note"]].to_string(index=False))


if __name__ == "__main__":
    main()
