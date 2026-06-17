#!/usr/bin/env python3
"""Plot the species-clustering vs tree-recovery frontier across loss weightings.

The open question: is there a tree-vs-species weighting that keeps BOTH high
species-clustering AMI (~0.9, matching VSEARCH) AND high tree-recovery Pearson
(~0.9, our DETECT/novelty value)? Or is it a strict tradeoff?

For each model embedding we compute:
  - species-clustering AMI (blind agglomerative best + KMeans k=species)
  - tree-recovery Pearson (embedding distance vs species-tree distance, unseen-unseen)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.preprocessing import normalize
from scipy.stats import pearsonr

BASE = Path("/Users/kluless/marinemamba/results/paper1_phylo_calibrated_assignment/source_tables")
FRONTIER_DIR = BASE / "frontier_embeddings"
RUNROOT = Path("/Users/kluless/marinemamba/results/remote_runs")
TREE_NPZ = Path("/Users/kluless/marinemamba/results/remote_runs/2026-05-30/rtx_pro_6000/coi_fish_tree_clean_phylo_mamba_hier512_seqval/tree_embeddings.npz")
TAX = pd.read_csv("/Users/kluless/marinemamba/data/phylo/fish_tree_clean_splits/eval_c_query.csv").set_index("processid")

# model name -> embedding npz
MODELS = {
    "ws0.0 (pure tree, wc1.0)": FRONTIER_DIR / "ws0.0_query_embeddings.npz",
    "ws0.1 (tree-heavy)": FRONTIER_DIR / "ws0.1_query_embeddings.npz",
    "ws0.3 (tree-leaning)": FRONTIER_DIR / "ws0.3_query_embeddings.npz",
    "ws1.0 (balanced)": FRONTIER_DIR / "ws1.0_query_embeddings.npz",
    "ANCHOR tree-only cosine512": RUNROOT / "../paper1_phylo_calibrated_assignment/raw_sequence_production_v1/eval_c/embedding_export/query_embeddings.npz",
    "ANCHOR hybrid (species-heavy)": RUNROOT / "2026-06-02/rtx_pro_6000/paper1_retrieval_dl_sweep/coi_cnn_retrieval_hybrid_seed1301/query_embeddings.npz",
    "ANCHOR contrastive (species-only)": RUNROOT / "2026-06-02/rtx_pro_6000/paper1_retrieval_dl_sweep/coi_cnn_retrieval_contrastive_seed1301/query_embeddings.npz",
}


def load_tree_species_vecs():
    d = np.load(TREE_NPZ, allow_pickle=True)
    labels = [str(x) for x in d["labels"]] if "labels" in d else [str(x) for x in d[d.files[0]]]
    # find the embedding array
    emb = None
    for k in d.files:
        a = d[k]
        if getattr(a, "ndim", 0) == 2:
            emb = a
    vecs = normalize(emb.astype(np.float32))
    return {lab: vecs[i] for i, lab in enumerate(labels)}


def tree_recovery_pearson(emb, sp_labels, tree_vecs, n_pairs=30000, seed=0):
    rng = np.random.RandomState(seed)
    # keep reads whose species has a tree vector
    keep = [i for i, s in enumerate(sp_labels) if s in tree_vecs]
    keep = np.array(keep)
    if len(keep) < 100:
        return float("nan")
    i = rng.choice(keep, n_pairs)
    j = rng.choice(keep, n_pairs)
    ok = i != j
    i, j = i[ok], j[ok]
    emb_d = 1 - (emb[i] * emb[j]).sum(1)  # cosine distance between reads
    tv_i = np.stack([tree_vecs[sp_labels[k]] for k in i])
    tv_j = np.stack([tree_vecs[sp_labels[k]] for k in j])
    tree_d = 1 - (tv_i * tv_j).sum(1)
    return float(pearsonr(emb_d, tree_d)[0])


def species_clustering_ami(emb, sp):
    n_sp = len(set(sp))
    km = MiniBatchKMeans(n_clusters=n_sp, random_state=0, n_init=3, batch_size=2048).fit(emb)
    kmeans_ami = float(AMI(sp, km.labels_))
    blind_best = 0.0
    for thr in [0.05, 0.10, 0.15]:
        ac = AgglomerativeClustering(n_clusters=None, distance_threshold=thr, metric="cosine", linkage="average").fit(emb)
        blind_best = max(blind_best, float(AMI(sp, ac.labels_)))
    return kmeans_ami, blind_best


def main():
    tree_vecs = load_tree_species_vecs()
    print("tree species vecs:", len(tree_vecs))
    rows = []
    for name, npz in MODELS.items():
        if not npz.exists():
            print("MISSING:", name, npz)
            continue
        d = np.load(npz, allow_pickle=True)
        pids = [str(p) for p in d["processids"]]
        df = pd.DataFrame({"processid": pids, "row": range(len(pids))}).set_index("processid")
        df = df.join(TAX[["species_name"]], how="inner").dropna(subset=["species_name"])
        emb = normalize(d["embeddings"].astype(np.float32)[df["row"].to_numpy()])
        sp = df["species_name"].str.replace(" ", "_").to_numpy()
        kmeans_ami, blind_ami = species_clustering_ami(emb, sp)
        tr = tree_recovery_pearson(emb, sp, tree_vecs)
        rows.append({"model": name, "species_AMI_blind": round(blind_ami, 4),
                     "species_AMI_kmeans": round(kmeans_ami, 4), "tree_recovery_pearson": round(tr, 4)})
        print("%-38s species_AMI(blind)=%.3f  (kmeans)=%.3f  tree_recovery=%.3f" % (
            name, blind_ami, kmeans_ami, tr), flush=True)
    out = BASE / "weight_frontier.json"
    json.dump({"frontier": rows, "vsearch_species_ami": 0.9147, "vsearch_tree_recovery": None}, open(out, "w"), indent=2)
    print("\n=== FRONTIER (species clustering vs tree recovery) ===")
    print("  best-of-both target: species_AMI>=0.89 AND tree_recovery>=0.85")
    for r in sorted(rows, key=lambda x: -x["tree_recovery_pearson"]):
        flag = "  <-- BEST OF BOTH" if r["species_AMI_blind"] >= 0.89 and r["tree_recovery_pearson"] >= 0.85 else ""
        print("  %-38s species=%.3f tree=%.3f%s" % (r["model"], r["species_AMI_blind"], r["tree_recovery_pearson"], flag))
    print("saved:", out)


if __name__ == "__main__":
    main()
