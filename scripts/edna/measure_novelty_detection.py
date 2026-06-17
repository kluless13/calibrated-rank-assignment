#!/usr/bin/env python3
"""First-pass novelty-detection measurement for the 'reading the unknown' story.

Question: can the encoder's geometry tell when a query's species is NOT in the
reference database? We use the held-out splits as a labelled novelty benchmark:

  seen_test     -> species IS in reference        -> KNOWN  (label 0)
  eval_c        -> species held out, genus present -> NOVEL  (species-novel)
  unseen_genera -> genus held out                  -> NOVEL  (genus-novel, "more novel")

Novelty score = how poorly the query matches its nearest candidate. We read the
top-1 similarity (pred_score) and the top1-top2 margin straight from the saved
zero_shot_candidate_predictions.csv files. No GPU, no retraining.

If KNOWN vs NOVEL separates (AUROC well above 0.5), the discovery framing is
real and Layer B graduates from 'proposed' to 'measured'. If not, we learn that
now for free.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RUN = Path("results/remote_runs/2026-05-30/rtx_pro_6000")
SPLITS = {
    "seen_test": RUN / "coi_fish_tree_clean_phylo_mamba_cosine512_seqval_seen_test/zero_shot_candidate_predictions.csv",
    "eval_c": RUN / "coi_fish_tree_clean_phylo_mamba_cosine512_seqval/zero_shot_candidate_predictions.csv",
    "unseen_genera": RUN / "coi_fish_tree_clean_phylo_mamba_cosine512_seqval_unseen_genera/zero_shot_candidate_predictions.csv",
}
OUT_DIR = Path("results/paper1_phylo_calibrated_assignment/source_tables")


def top_two(scores_json: str) -> tuple[float, float]:
    try:
        s = json.loads(scores_json)
        if isinstance(s, list) and s:
            top1 = float(s[0])
            top2 = float(s[1]) if len(s) > 1 else float(s[0])
            return top1, top2
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return np.nan, np.nan


def load(name: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    tops = df["top_scores"].map(top_two)
    df["top1"] = [t[0] for t in tops]
    df["top2"] = [t[1] for t in tops]
    # fall back to pred_score if top_scores was unparseable
    df["top1"] = df["top1"].fillna(pd.to_numeric(df.get("pred_score"), errors="coerce"))
    df["margin"] = df["top1"] - df["top2"]
    df["split"] = name
    return df[["processid", "split", "top1", "margin"]].dropna(subset=["top1"])


def dist(label: str, x: np.ndarray) -> dict:
    return {
        "split": label,
        "n": int(len(x)),
        "mean": round(float(np.mean(x)), 4),
        "median": round(float(np.median(x)), 4),
        "p25": round(float(np.percentile(x, 25)), 4),
        "p75": round(float(np.percentile(x, 75)), 4),
    }


def auroc_novel(known: np.ndarray, novel: np.ndarray) -> float:
    """AUROC for detecting NOVEL as positive. Novelty score = -similarity
    (a less-similar query is more novel). Returns AUROC in [0,1]."""
    y = np.concatenate([np.zeros(len(known)), np.ones(len(novel))])
    score = np.concatenate([-known, -novel])  # higher = more novel
    return float(roc_auc_score(y, score))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {name: load(name, path) for name, path in SPLITS.items()}
    for name, df in frames.items():
        print(f"loaded {name}: {len(df):,} queries")

    seen = frames["seen_test"]
    evalc = frames["eval_c"]
    unseen = frames["unseen_genera"]

    print("\n=== top-1 similarity distribution per split (sanity: known should be HIGHEST) ===")
    rows = []
    for name, df in [("seen_test (KNOWN)", seen), ("eval_c (species-novel)", evalc), ("unseen_genera (genus-novel)", unseen)]:
        d = dist(name, df["top1"].to_numpy())
        rows.append(d)
        print(f"  {name:<30} n={d['n']:>6}  mean={d['mean']:.4f}  median={d['median']:.4f}  [p25={d['p25']:.4f}, p75={d['p75']:.4f}]")

    print("\n=== top1-top2 MARGIN distribution per split ===")
    margin_rows = []
    for name, df in [("seen_test (KNOWN)", seen), ("eval_c (species-novel)", evalc), ("unseen_genera (genus-novel)", unseen)]:
        d = dist(name, df["margin"].to_numpy())
        d["feature"] = "margin"
        margin_rows.append(d)
        print(f"  {name:<30} mean={d['mean']:.4f}  median={d['median']:.4f}")

    novel_all = np.concatenate([evalc["top1"].to_numpy(), unseen["top1"].to_numpy()])
    novel_all_margin = np.concatenate([evalc["margin"].to_numpy(), unseen["margin"].to_numpy()])

    print("\n=== NOVELTY DETECTION (the decisive numbers) ===")
    results = {}
    results["auroc_known_vs_novel__top1"] = auroc_novel(seen["top1"].to_numpy(), novel_all)
    results["auroc_known_vs_novel__margin"] = auroc_novel(seen["margin"].to_numpy(), novel_all_margin)
    results["auroc_known_vs_evalc__top1"] = auroc_novel(seen["top1"].to_numpy(), evalc["top1"].to_numpy())
    results["auroc_known_vs_unseengenera__top1"] = auroc_novel(seen["top1"].to_numpy(), unseen["top1"].to_numpy())
    # graded: among novels, can we tell species-novel from the MORE-novel genus-novel?
    results["auroc_graded_evalc_vs_unseengenera__top1"] = auroc_novel(evalc["top1"].to_numpy(), unseen["top1"].to_numpy())

    for k, v in results.items():
        print(f"  {k:<42} AUROC = {v:.4f}")

    # operating point: flag the most-novel X% by lowest top1; what fraction are truly novel?
    print("\n=== operating point: threshold on top-1 similarity ===")
    combined = pd.concat([
        seen.assign(novel=0),
        evalc.assign(novel=1),
        unseen.assign(novel=1),
    ], ignore_index=True)
    op_rows = []
    for q in [0.10, 0.25, 0.50]:
        thr = combined["top1"].quantile(q)  # flag queries BELOW this similarity as novel
        flagged = combined[combined["top1"] <= thr]
        prec = float(flagged["novel"].mean()) if len(flagged) else float("nan")
        recall = float(flagged["novel"].sum() / combined["novel"].sum())
        op_rows.append({"flag_below_quantile": q, "threshold": round(float(thr), 4),
                        "n_flagged": int(len(flagged)), "precision_novel": round(prec, 4),
                        "recall_novel": round(recall, 4)})
        print(f"  flag bottom {int(q*100):>2}% by similarity -> precision(novel)={prec:.3f}, recall(novel)={recall:.3f}")

    summary = {
        "experiment": "novelty_detection_first_pass",
        "source": "cosine512 seed1206 zero_shot_candidate_predictions (top-1 similarity)",
        "note": "First-pass proxy from full-tree candidate scores. Rigorous version uses reference-only distance + EVT/OpenMax.",
        "distributions_top1": rows,
        "distributions_margin": margin_rows,
        "auroc": {k: round(v, 4) for k, v in results.items()},
        "operating_points": op_rows,
    }
    out = OUT_DIR / "novelty_detection_first_pass.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(rows + margin_rows).to_csv(OUT_DIR / "novelty_detection_distributions.csv", index=False)
    print(f"\nsaved: {out}")

    # verdict
    main_auroc = results["auroc_known_vs_novel__top1"]
    print("\n" + "=" * 60)
    if main_auroc >= 0.80:
        print(f"VERDICT: STRONG separation (AUROC {main_auroc:.3f}). Discovery framing is real.")
    elif main_auroc >= 0.65:
        print(f"VERDICT: MODERATE separation (AUROC {main_auroc:.3f}). Promising; needs the rigorous EVT version.")
    else:
        print(f"VERDICT: WEAK separation (AUROC {main_auroc:.3f}). Top-1 proxy insufficient; needs reference-only distance / embeddings.")
    print("=" * 60)


if __name__ == "__main__":
    main()
