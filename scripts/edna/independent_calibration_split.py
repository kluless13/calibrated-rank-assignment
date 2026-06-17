#!/usr/bin/env python3
"""Prospective rank/no-call validation via species-disjoint calibration.

The headline rank/no-call numbers used seen-test-derived thresholds. This is the
stricter, prospective test the manuscript needs:

  1. Pool the held-out queries (Eval C + unseen-genera) from the CNN seed1206
     p-distance-reranked pipeline.
  2. Repeatedly split the SPECIES into calibration vs evaluation (disjoint).
  3. Fit per-rank confidence thresholds on calibration species only (target
     precision 0.99, feature = per-rank top-10 consensus).
  4. Apply the locked thresholds to evaluation species NEVER used to set them.
  5. Report coverage, assigned precision, false-species-call rate, rank counts,
     and whether the 0% false-species result survives.

No GPU; uses existing pipeline_rank_assignments.csv.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

RUNS = Path("results/paper1_phylo_calibrated_assignment/pipeline_runs")
FILES = {
    "eval_c": RUNS / "coi_cnn_seed1206_eval_c_target099_pdistance_experimental/pipeline_rank_assignments.csv",
    "unseen_genera": RUNS / "coi_cnn_seed1206_unseen_genera_target099_pdistance_experimental/pipeline_rank_assignments.csv",
}
RANKS = ["species", "genus", "family", "order"]
TARGET = 0.99
N_REPEATS = 30
OUT = Path("results/paper1_phylo_calibrated_assignment/source_tables/independent_calibration_split.json")


def load():
    frames = []
    for split, f in FILES.items():
        d = pd.read_csv(f)
        d["split"] = split
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["true_species"] = df["true_tree_label"].astype(str)
    for r in RANKS:
        true_col = "true_species" if r == "species" else f"true_{r}"
        df[f"correct_{r}"] = (df[f"pred_{r}"].astype(str) == df[true_col].astype(str)).astype(float)
        df[f"conf_{r}"] = pd.to_numeric(df[f"{r}_top10_consensus"], errors="coerce").fillna(0.0)
    return df


def fit_thresholds(calib: pd.DataFrame, target: float) -> dict:
    """Per rank: smallest confidence threshold whose accepted set hits target precision."""
    thr = {}
    for r in RANKS:
        conf = calib[f"conf_{r}"].to_numpy()
        corr = calib[f"correct_{r}"].to_numpy()
        order = np.argsort(-conf, kind="stable")
        corr_s = corr[order]
        conf_s = conf[order]
        cum_prec = np.cumsum(corr_s) / np.arange(1, len(corr_s) + 1)
        ok = np.where(cum_prec >= target)[0]
        thr[r] = float(conf_s[ok[-1]]) if len(ok) else np.inf
    return thr


def apply_policy(ev: pd.DataFrame, thr: dict) -> dict:
    conf = ev[[f"conf_{r}" for r in RANKS]].to_numpy()
    corr = ev[[f"correct_{r}" for r in RANKS]].to_numpy()
    thr_arr = np.array([thr[r] for r in RANKS])
    meets = conf >= thr_arr
    any_meet = meets.any(axis=1)
    first = np.argmax(meets, axis=1)  # deepest = species(0) first
    n = len(ev)
    na = int(any_meet.sum())
    assigned_correct = corr[np.arange(n), first][any_meet]
    return {
        "coverage": na / n,
        "assigned_precision": float(assigned_correct.mean()) if na else float("nan"),
        "false_species_rate": float(((first == 0) & any_meet).mean()),
        "n_species": int(((first == 0) & any_meet).sum()),
        "n_genus": int(((first == 1) & any_meet).sum()),
        "n_family": int(((first == 2) & any_meet).sum()),
        "n_order": int(((first == 3) & any_meet).sum()),
        "n_no_call": int((~any_meet).sum()),
    }


def main():
    df = load()
    species = df["true_species"].unique()
    print("held-out queries=%d  species=%d  (eval_c+unseen_genera, CNN seed1206 p-distance)" % (len(df), len(species)), flush=True)
    reps = []
    for seed in range(N_REPEATS):
        rng = np.random.RandomState(seed)
        perm = rng.permutation(species)
        cut = len(perm) // 2
        calib_sp, eval_sp = set(perm[:cut]), set(perm[cut:])
        thr = fit_thresholds(df[df["true_species"].isin(calib_sp)], TARGET)
        reps.append(apply_policy(df[df["true_species"].isin(eval_sp)], thr))

    def agg(key):
        v = np.array([r[key] for r in reps], dtype=float)
        v = v[~np.isnan(v)]
        return {"mean": round(float(v.mean()), 4), "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}

    summary = {
        "experiment": "independent_species_disjoint_calibration",
        "model": "CNN seed1206 p-distance reranked",
        "splits_pooled": list(FILES.keys()),
        "target_precision": TARGET,
        "feature": "per-rank top-10 consensus",
        "n_repeats": N_REPEATS,
        "coverage": agg("coverage"),
        "assigned_precision": agg("assigned_precision"),
        "false_species_rate": agg("false_species_rate"),
        "mean_rank_counts": {k: int(np.mean([r[k] for r in reps])) for k in ["n_species", "n_genus", "n_family", "n_order", "n_no_call"]},
        "zero_false_species_survives": bool(max(r["false_species_rate"] for r in reps) == 0.0),
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== PROSPECTIVE (species-disjoint) rank/no-call, target 0.99, %d repeats ===" % N_REPEATS)
    print("  coverage:           mean %.3f  range [%.3f, %.3f]" % (summary["coverage"]["mean"], summary["coverage"]["min"], summary["coverage"]["max"]))
    print("  assigned precision: mean %.3f  range [%.3f, %.3f]" % (summary["assigned_precision"]["mean"], summary["assigned_precision"]["min"], summary["assigned_precision"]["max"]))
    print("  false species-call: mean %.4f  range [%.4f, %.4f]" % (summary["false_species_rate"]["mean"], summary["false_species_rate"]["min"], summary["false_species_rate"]["max"]))
    print("  mean rank counts:", summary["mean_rank_counts"])
    print("  ZERO false-species survives across all repeats:", summary["zero_false_species_survives"])
    print("saved:", OUT)


if __name__ == "__main__":
    main()
