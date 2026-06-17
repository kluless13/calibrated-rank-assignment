#!/usr/bin/env python3
"""Layer C: rediscovery. Cluster held-out (Eval C) barcode embeddings with NO
labels and ask whether the clusters recover the latent species/genus/family.

These 11,594 reads come from 531 species the model never trained on. If reads
from the same unknown species land in the same cluster, the geometry is
'rediscovering' species without ever being told they exist.

Metrics (all label-free clustering scored against held-out truth):
  AMI  (Adjusted Mutual Information): 0 = random, 1 = perfect agreement.
  ARI  (Adjusted Rand Index): same idea, pair-counting based.
  homogeneity: are clusters pure (one species each)?
  completeness: is each species kept in one cluster (not split)?
We score against species, genus, and family truth to see the graded picture.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
)
from sklearn.preprocessing import normalize

EMB = Path("results/paper1_phylo_calibrated_assignment/raw_sequence_production_v1/eval_c/embedding_export/query_embeddings.npz")
PRED = Path("results/remote_runs/2026-05-30/rtx_pro_6000/coi_fish_tree_clean_phylo_mamba_cosine512_seqval/zero_shot_candidate_predictions.csv")
OUT_DIR = Path("results/paper1_phylo_calibrated_assignment/source_tables")


def score(labels_true: np.ndarray, labels_pred: np.ndarray) -> dict:
    h, c, v = homogeneity_completeness_v_measure(labels_true, labels_pred)
    return {
        "AMI": round(float(adjusted_mutual_info_score(labels_true, labels_pred)), 4),
        "ARI": round(float(adjusted_rand_score(labels_true, labels_pred)), 4),
        "homogeneity": round(float(h), 4),
        "completeness": round(float(c), 4),
        "n_clusters": int(len(set(labels_pred))),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(EMB, allow_pickle=True)
    pids = data["processids"]
    X = normalize(data["embeddings"].astype(np.float32))  # L2-normalize -> cosine geometry
    print(f"embeddings: {X.shape}")

    # map processid -> true taxonomy
    pred = pd.read_csv(PRED, usecols=["processid", "species_name", "genus_name", "family_name"])
    tax = pred.set_index("processid")
    df = pd.DataFrame({"processid": pids}).set_index("processid").join(tax).reset_index().dropna(subset=["species_name"])
    keep = df.index.to_numpy()
    X = X[keep]
    sp = df["species_name"].to_numpy()
    ge = df["genus_name"].to_numpy()
    fa = df["family_name"].to_numpy()
    n_sp, n_ge, n_fa = len(set(sp)), len(set(ge)), len(set(fa))
    print(f"truth: {len(sp):,} reads, {n_sp} species, {n_ge} genera, {n_fa} families")

    results = {}

    # --- 1) KMeans with the TRUE species count (upper-bound: how separable given k) ---
    print("\n=== (1) KMeans, k = true #species (tests separability given the count) ===")
    km = MiniBatchKMeans(n_clusters=n_sp, random_state=0, n_init=3, batch_size=2048).fit(X)
    results["kmeans_k_species"] = {
        "species": score(sp, km.labels_), "genus": score(ge, km.labels_), "family": score(fa, km.labels_),
    }
    for rank in ["species", "genus", "family"]:
        s = results["kmeans_k_species"][rank]
        print(f"  vs {rank:<8} AMI={s['AMI']:.3f} ARI={s['ARI']:.3f} homog={s['homogeneity']:.3f} complete={s['completeness']:.3f}")

    # --- 2) Agglomerative WITHOUT the count (real discovery: threshold on cosine distance) ---
    print("\n=== (2) Agglomerative, NO known count (real discovery setting) ===")
    for thr in [0.10, 0.15, 0.20]:
        ac = AgglomerativeClustering(n_clusters=None, distance_threshold=thr, metric="cosine", linkage="average").fit(X)
        s_sp = score(sp, ac.labels_)
        s_ge = score(ge, ac.labels_)
        results[f"agglomerative_thr{thr}"] = {"species": s_sp, "genus": s_ge}
        print(f"  thr={thr}: found {s_sp['n_clusters']} clusters (true species={n_sp}) | "
              f"species AMI={s_sp['AMI']:.3f} | genus AMI={s_ge['AMI']:.3f}")

    summary = {
        "experiment": "rediscovery_clustering_eval_c",
        "n_reads": int(len(sp)), "n_true_species": n_sp, "n_true_genera": n_ge, "n_true_families": n_fa,
        "results": results,
        "notes": [
            "Eval C reads are from species the encoder never trained on.",
            "Clustering uses ONLY the embeddings (no labels); truth used only to score.",
            "AMI/ARI: 0=random, 1=perfect. homogeneity=cluster purity, completeness=species kept whole.",
        ],
    }
    out = OUT_DIR / "rediscovery_clustering.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nsaved: {out}")

    sp_ami = results["kmeans_k_species"]["species"]["AMI"]
    ge_ami = results["kmeans_k_species"]["genus"]["AMI"]
    print("\n" + "=" * 60)
    print(f"Rediscovery (KMeans k=species): species AMI={sp_ami:.3f}, genus AMI={ge_ami:.3f}")
    if sp_ami >= 0.85:
        print("VERDICT: STRONG species rediscovery. The geometry recovers unknown species.")
    elif ge_ami >= 0.80:
        print("VERDICT: Strong GENUS rediscovery; species marker-limited (consistent w/ detection).")
    else:
        print("VERDICT: Moderate; graded recovery, strongest at higher ranks.")
    print("=" * 60)


if __name__ == "__main__":
    main()
