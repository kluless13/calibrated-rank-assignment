#!/usr/bin/env python3
"""Does adding a species objective to the tree-distance objective improve
rediscovery clustering? The retrieval-DL sweep already TRAINED these multi-task
CNNs and exported their Eval C embeddings; they were never clustered. We cluster
them now (free, local) and compare to tree-only, VSEARCH, cd-hit, BarcodeBERT.

  cnn_hybrid          = tree-distance + species (the 'mix' / Direction A)
  cnn_contrastive     = species only (~ BarcodeBERT-style)
  cnn_hier_contrastive= tree + species + genus/family/order soft targets
  (baseline tree-only cosine512: species AMI 0.874, already measured)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI, homogeneity_completeness_v_measure as HCV
from sklearn.preprocessing import normalize

SWEEP = Path("results/remote_runs/2026-06-02/rtx_pro_6000/paper1_retrieval_dl_sweep")
MODELS = {
    "cnn_hybrid (tree+species)": SWEEP / "coi_cnn_retrieval_hybrid_seed1301/query_embeddings.npz",
    "cnn_contrastive (species only)": SWEEP / "coi_cnn_retrieval_contrastive_seed1301/query_embeddings.npz",
    "cnn_hier_contrastive (tree+species+ranks)": SWEEP / "coi_cnn_retrieval_hier_contrastive_seed1301/query_embeddings.npz",
}
TAX = pd.read_csv("data/phylo/fish_tree_clean_splits/eval_c_query.csv").set_index("processid")
OUT = Path("results/paper1_phylo_calibrated_assignment/source_tables/multitask_rediscovery.json")


def sc(t, p):
    h, c, _ = HCV(t, p)
    return {"AMI": round(float(AMI(t, p)), 4), "ARI": round(float(ARI(t, p)), 4),
            "homogeneity": round(float(h), 4), "completeness": round(float(c), 4), "n_clusters": int(len(set(p)))}


def evaluate(name: str, npz: Path) -> dict:
    d = np.load(npz, allow_pickle=True)
    pids = [str(p) for p in d["processids"]]
    df = pd.DataFrame({"processid": pids, "row": range(len(pids))}).set_index("processid")
    df = df.join(TAX[["species_name", "genus_name", "family_name"]], how="inner").dropna(subset=["species_name"])
    X = normalize(d["embeddings"].astype(np.float32)[df["row"].to_numpy()])
    sp, ge, fa = df["species_name"].to_numpy(), df["genus_name"].to_numpy(), df["family_name"].to_numpy()
    n_sp = len(set(sp))
    print("%-42s matched %d reads, %d species, dim=%d" % (name, len(df), n_sp, X.shape[1]), flush=True)
    km = MiniBatchKMeans(n_clusters=n_sp, random_state=0, n_init=3, batch_size=2048).fit(X)
    out = {"matched_reads": int(len(df)), "n_species": n_sp, "dim": int(X.shape[1]),
           "kmeans_k_species": {"species": sc(sp, km.labels_), "genus": sc(ge, km.labels_), "family": sc(fa, km.labels_)},
           "blind": {}}
    for thr in [0.05, 0.10]:
        ac = AgglomerativeClustering(n_clusters=None, distance_threshold=thr, metric="cosine", linkage="average").fit(X)
        out["blind"]["thr%s" % thr] = {"species": sc(sp, ac.labels_)}
    s = out["kmeans_k_species"]
    print("    KMeans k=species: species AMI=%.3f  genus AMI=%.3f  family AMI=%.3f" % (
        s["species"]["AMI"], s["genus"]["AMI"], s["family"]["AMI"]), flush=True)
    return out


def main():
    results = {"experiment": "multitask_rediscovery", "models": {}}
    for name, npz in MODELS.items():
        if npz.exists():
            results["models"][name] = evaluate(name, npz)
        else:
            print("MISSING:", npz, flush=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print("\n" + "=" * 70)
    print("REDISCOVERY HEAD-TO-HEAD (species recovery AMI, KMeans k=species):")
    print("  %-44s %s" % ("method", "species / genus / family"))
    print("  %-44s %s" % ("tree-only cosine512 (current model)", "0.874 / 0.859 / 0.756"))
    for name, r in results["models"].items():
        s = r["kmeans_k_species"]
        print("  %-44s %.3f / %.3f / %.3f" % (name, s["species"]["AMI"], s["genus"]["AMI"], s["family"]["AMI"]))
    print("  %-44s %s" % ("VSEARCH (real BIN/ABGD)", "0.915 / 0.876 / 0.720"))
    print("  %-44s %s" % ("cd-hit (real BIN-style)", "0.886 / 0.847 / 0.692"))
    print("  %-44s %s" % ("BarcodeBERT (frozen)", "0.492 / -- / --"))
    print("=" * 70)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
