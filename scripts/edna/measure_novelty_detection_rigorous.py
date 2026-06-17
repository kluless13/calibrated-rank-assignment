#!/usr/bin/env python3
"""Rigorous novelty detection: reference-only distance + multi-feature detector.

Fixes the two confounds of the first pass:
  1. Full-tree candidates -> we now compare each query ONLY against the 3,839
     reference species (the species we actually have sequences for). A novel
     query can then only match a RELATIVE, not its own held-out position.
  2. Single feature -> we add margin, genus-consensus, and top-k mean, then
     train a small detector and evaluate it on SPECIES IT NEVER CALIBRATED ON
     (species-disjoint split), so there is no leakage.

Labels:
  seen_test     = KNOWN  (species in reference)        -> 0
  eval_c        = NOVEL  (species-novel, genus present) -> 1
  unseen_genera = NOVEL  (genus-novel)                  -> 1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

RUN = Path("results/remote_runs/2026-05-30/rtx_pro_6000")
REF_CSV = Path("data/phylo/fish_tree_clean_splits/reference_train.csv")
SPLITS = {
    "seen_test": RUN / "coi_fish_tree_clean_phylo_mamba_cosine512_seqval_seen_test/zero_shot_candidate_predictions.csv",
    "eval_c": RUN / "coi_fish_tree_clean_phylo_mamba_cosine512_seqval/zero_shot_candidate_predictions.csv",
    "unseen_genera": RUN / "coi_fish_tree_clean_phylo_mamba_cosine512_seqval_unseen_genera/zero_shot_candidate_predictions.csv",
}
OUT_DIR = Path("results/paper1_phylo_calibrated_assignment/source_tables")
FEATURES = ["ref_top1", "ref_margin", "ref_genus_consensus", "ref_topk_mean", "ref_n_ref_in_topk"]


def reference_only_features(labels_json: str, scores_json: str, ref_species: set[str], topk: int = 10) -> dict:
    """Filter candidates to reference species, then compute novelty features."""
    try:
        labels = json.loads(labels_json)
        scores = [float(s) for s in json.loads(scores_json)]
    except (json.JSONDecodeError, TypeError, ValueError):
        return {f: np.nan for f in FEATURES}
    # keep only candidates that are reference species, preserve descending order
    ref_pairs = [(lab, sc) for lab, sc in zip(labels, scores) if lab in ref_species]
    if not ref_pairs:
        return {f: np.nan for f in FEATURES}
    ref_labels = [p[0] for p in ref_pairs]
    ref_scores = [p[1] for p in ref_pairs]
    top1 = ref_scores[0]
    top2 = ref_scores[1] if len(ref_scores) > 1 else ref_scores[0]
    top_labels = ref_labels[:topk]
    genera = [lab.split("_", 1)[0] for lab in top_labels]
    modal = max(set(genera), key=genera.count)
    consensus = genera.count(modal) / len(genera)
    return {
        "ref_top1": top1,
        "ref_margin": top1 - top2,
        "ref_genus_consensus": consensus,
        "ref_topk_mean": float(np.mean(ref_scores[:topk])),
        "ref_n_ref_in_topk": float(min(len(ref_pairs), topk)),
    }


def build_table(name: str, path: Path, ref_species: set[str], label: int, depth: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    feats = df.apply(lambda r: reference_only_features(r["top_tree_labels"], r["top_scores"], ref_species), axis=1)
    feat_df = pd.DataFrame(list(feats))
    out = pd.concat([df[["species_name"]].reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    out["split"] = name
    out["novel"] = label
    out["novelty_depth"] = depth  # 0 known, 1 species-novel, 2 genus-novel
    return out.dropna(subset=FEATURES)


def single_feature_auroc(known: np.ndarray, novel: np.ndarray) -> float:
    y = np.concatenate([np.zeros(len(known)), np.ones(len(novel))])
    score = np.concatenate([-known, -novel])  # lower similarity = more novel
    return float(roc_auc_score(y, score))


def species_split_auroc(table: pd.DataFrame, n_repeats: int = 5) -> dict:
    """Train a detector on calibration species, test on disjoint eval species."""
    rng_seeds = [11, 22, 33, 44, 55]
    results = {"logistic": [], "hgb": []}
    species = table["species_name"].dropna().unique()
    for seed in rng_seeds[:n_repeats]:
        rng = np.random.RandomState(seed)
        # split species (not rows) into calibration / evaluation, disjoint
        perm = rng.permutation(species)
        cut = int(len(perm) * 0.7)
        calib_sp, eval_sp = set(perm[:cut]), set(perm[cut:])
        tr = table[table["species_name"].isin(calib_sp)]
        te = table[table["species_name"].isin(eval_sp)]
        if te["novel"].nunique() < 2 or tr["novel"].nunique() < 2:
            continue
        Xtr, ytr = tr[FEATURES].to_numpy(), tr["novel"].to_numpy()
        Xte, yte = te[FEATURES].to_numpy(), te["novel"].to_numpy()
        scaler = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=1000).fit(scaler.transform(Xtr), ytr)
        results["logistic"].append(roc_auc_score(yte, lr.predict_proba(scaler.transform(Xte))[:, 1]))
        hgb = HistGradientBoostingClassifier(random_state=seed).fit(Xtr, ytr)
        results["hgb"].append(roc_auc_score(yte, hgb.predict_proba(Xte)[:, 1]))
    return {
        m: {"mean": round(float(np.mean(v)), 4), "min": round(float(np.min(v)), 4), "max": round(float(np.max(v)), 4)}
        for m, v in results.items() if v
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref_df = pd.read_csv(REF_CSV)
    ref_species = {s.replace(" ", "_") for s in ref_df["species_name"].dropna().unique()}
    print(f"reference species: {len(ref_species):,}")

    seen = build_table("seen_test", SPLITS["seen_test"], ref_species, 0, 0)
    evalc = build_table("eval_c", SPLITS["eval_c"], ref_species, 1, 1)
    unseen = build_table("unseen_genera", SPLITS["unseen_genera"], ref_species, 1, 2)
    print(f"loaded (reference-only features): seen={len(seen):,} eval_c={len(evalc):,} unseen_genera={len(unseen):,}")

    print("\n=== reference-only top-1 similarity per split (should be cleaner than full-tree) ===")
    for nm, df in [("seen_test (KNOWN)", seen), ("eval_c (species-novel)", evalc), ("unseen_genera (genus-novel)", unseen)]:
        x = df["ref_top1"].to_numpy()
        print(f"  {nm:<30} mean={np.mean(x):.4f}  median={np.median(x):.4f}")

    novel_all = np.concatenate([evalc["ref_top1"].to_numpy(), unseen["ref_top1"].to_numpy()])
    print("\n=== SINGLE-FEATURE AUROC (reference-only top-1; no training, no leakage) ===")
    a_all = single_feature_auroc(seen["ref_top1"].to_numpy(), novel_all)
    a_evalc = single_feature_auroc(seen["ref_top1"].to_numpy(), evalc["ref_top1"].to_numpy())
    a_unseen = single_feature_auroc(seen["ref_top1"].to_numpy(), unseen["ref_top1"].to_numpy())
    print(f"  known vs novel (all):        {a_all:.4f}   (first-pass full-tree was 0.750)")
    print(f"  known vs eval_c (species):   {a_evalc:.4f}   (first-pass 0.679)")
    print(f"  known vs unseen (genus):     {a_unseen:.4f}   (first-pass 0.840)")

    table = pd.concat([seen, evalc, unseen], ignore_index=True)
    print("\n=== MULTI-FEATURE DETECTOR (species-disjoint split, leakage-safe, 5 repeats) ===")
    sup = species_split_auroc(table)
    for model, stats in sup.items():
        print(f"  {model:<10} AUROC mean={stats['mean']:.4f}  range [{stats['min']:.4f}, {stats['max']:.4f}]")

    summary = {
        "experiment": "novelty_detection_rigorous",
        "reference_species": len(ref_species),
        "n_queries": {"seen_test": len(seen), "eval_c": len(evalc), "unseen_genera": len(unseen)},
        "single_feature_auroc_reference_only": {
            "known_vs_novel_all": round(a_all, 4),
            "known_vs_eval_c_species_novel": round(a_evalc, 4),
            "known_vs_unseen_genera_genus_novel": round(a_unseen, 4),
        },
        "multi_feature_detector_species_split": sup,
        "features": FEATURES,
        "notes": [
            "Reference-only = candidates filtered to the 3,839 species with sequences.",
            "Multi-feature detector trained on calibration species, tested on disjoint eval species.",
            "Known=trained=in-reference; this is the operational open-set definition.",
        ],
    }
    out = OUT_DIR / "novelty_detection_rigorous.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nsaved: {out}")

    best = max([a_all] + [s["mean"] for s in sup.values()]) if sup else a_all
    print("\n" + "=" * 60)
    if best >= 0.85:
        print(f"VERDICT: STRONG (best AUROC {best:.3f}). Discovery framing is real and compelling.")
    elif best >= 0.75:
        print(f"VERDICT: SOLID (best AUROC {best:.3f}). Defensible discovery result; genus-level especially.")
    else:
        print(f"VERDICT: MODERATE (best AUROC {best:.3f}). Honest graded-novelty story, marker-limited at species.")
    print("=" * 60)


if __name__ == "__main__":
    main()
