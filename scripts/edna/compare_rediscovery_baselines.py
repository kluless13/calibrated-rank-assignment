#!/usr/bin/env python3
"""Head-to-head: does our learned tree-geometry embedding recover species better
than the species-delimitation prior art?

BIN (Ratnasingham & Hebert 2013) and ABGD/ASAP delimit species by clustering raw
sequences at a genetic-distance threshold. We reproduce that *mechanism* with
k-mer distance on the raw Eval C sequences (VSEARCH/cd-hit are not installed
locally; k-mer distance is a faithful raw-sequence-distance basis). We then
cluster our learned 512-D tree-geometry embedding the same ways and compare.

The honest open question: our embedding was trained for COARSE tree structure,
which may SMOOTH OVER the FINE differences species delimitation needs. So raw
sequence distance might win at species and lose at genus. We let the data decide.

Scored against held-out truth (Eval C = 531 species the model never trained on):
  AMI 0=random 1=perfect; homogeneity=cluster purity; completeness=species whole.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
)
from sklearn.preprocessing import normalize

SEQS = Path("data/phylo/fish_tree_clean_splits/eval_c_query.csv")
EMB = Path("results/paper1_phylo_calibrated_assignment/raw_sequence_production_v1/eval_c/embedding_export/query_embeddings.npz")
OUT_DIR = Path("results/paper1_phylo_calibrated_assignment/source_tables")


def score(true, pred) -> dict:
    h, c, _ = homogeneity_completeness_v_measure(true, pred)
    return {"AMI": round(float(adjusted_mutual_info_score(true, pred)), 4),
            "ARI": round(float(adjusted_rand_score(true, pred)), 4),
            "homogeneity": round(float(h), 4), "completeness": round(float(c), 4),
            "n_clusters": int(len(set(pred)))}


def cluster_and_score(name: str, X: np.ndarray, sp, ge, fa, n_sp: int) -> dict:
    print(f"\n--- {name} (dim={X.shape[1]}) ---")
    out = {"representation": name, "dim": int(X.shape[1])}
    # KMeans with true count (separability given k)
    km = MiniBatchKMeans(n_clusters=n_sp, random_state=0, n_init=3, batch_size=2048).fit(X)
    out["kmeans_k_species"] = {"species": score(sp, km.labels_), "genus": score(ge, km.labels_), "family": score(fa, km.labels_)}
    s = out["kmeans_k_species"]
    print(f"  KMeans k={n_sp}:  species AMI={s['species']['AMI']:.3f}  genus AMI={s['genus']['AMI']:.3f}  family AMI={s['family']['AMI']:.3f}")
    # Blind agglomerative (BIN/ABGD-style threshold, no known count)
    Xd = X.toarray() if hasattr(X, "toarray") else X  # agglomerative needs dense
    out["blind"] = {}
    for thr in [0.05, 0.10, 0.15]:
        ac = AgglomerativeClustering(n_clusters=None, distance_threshold=thr, metric="cosine", linkage="average").fit(Xd)
        ssp, sge = score(sp, ac.labels_), score(ge, ac.labels_)
        out["blind"][f"thr{thr}"] = {"species": ssp, "genus": sge}
        print(f"  blind thr={thr}: {ssp['n_clusters']:>4} clusters (true={n_sp}) | species AMI={ssp['AMI']:.3f} | genus AMI={sge['AMI']:.3f}")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seqs = pd.read_csv(SEQS)
    data = np.load(EMB, allow_pickle=True)
    emb_pids = list(data["processids"])
    emb = data["embeddings"].astype(np.float32)
    emb_idx = {p: i for i, p in enumerate(emb_pids)}

    # align: keep reads that have both a sequence and an embedding
    seqs = seqs[seqs["processid"].isin(emb_idx)].dropna(subset=["nucleotides", "species_name"]).reset_index(drop=True)
    order = [emb_idx[p] for p in seqs["processid"]]
    X_learned = normalize(emb[order])
    sp = seqs["species_name"].to_numpy()
    ge = seqs["genus_name"].to_numpy()
    fa = seqs["family_name"].to_numpy()
    n_sp = len(set(sp))
    print(f"aligned reads: {len(seqs):,} | species={n_sp} genera={len(set(ge))} families={len(set(fa))}")

    # raw-sequence k-mer representation (the BIN/ABGD-style basis)
    print("building 6-mer vectors for raw sequences (BIN/ABGD-style basis)...")
    vec = CountVectorizer(analyzer="char", ngram_range=(6, 6), lowercase=False)
    X_kmer = normalize(vec.fit_transform(seqs["nucleotides"].str.upper()))
    print(f"  k-mer vocab: {len(vec.vocabulary_)}")

    results = {
        "experiment": "rediscovery_head_to_head",
        "n_reads": int(len(seqs)), "n_species": n_sp,
        "note": "k-mer distance is a faithful BIN/ABGD-style raw-sequence-distance basis; VSEARCH/cd-hit not installed locally.",
        "learned_tree_geometry": cluster_and_score("LEARNED tree-geometry embedding", X_learned, sp, ge, fa, n_sp),
        "raw_kmer_BIN_ABGD_style": cluster_and_score("RAW k-mer (BIN/ABGD-style)", X_kmer, sp, ge, fa, n_sp),
    }
    out = OUT_DIR / "rediscovery_head_to_head.json"
    out.write_text(json.dumps(results, indent=2) + "\n")

    # verdict table
    L = results["learned_tree_geometry"]["kmeans_k_species"]
    R = results["raw_kmer_BIN_ABGD_style"]["kmeans_k_species"]
    print("\n" + "=" * 64)
    print("HEAD-TO-HEAD (KMeans k=species, AMI):")
    print(f"  {'rank':<10}{'LEARNED':>10}{'RAW k-mer':>12}{'winner':>12}")
    for rank in ["species", "genus", "family"]:
        l, r = L[rank]["AMI"], R[rank]["AMI"]
        w = "LEARNED" if l > r + 0.01 else ("RAW" if r > l + 0.01 else "tie")
        print(f"  {rank:<10}{l:>10.3f}{r:>12.3f}{w:>12}")
    print("=" * 64)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
