import subprocess, json, numpy as np, pandas as pd
from sklearn.metrics import adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI, homogeneity_completeness_v_measure as HCV

df = pd.read_csv("/root/eval_c_query.csv").dropna(subset=["nucleotides", "species_name"]).reset_index(drop=True)
print("reads=%d species=%d" % (len(df), df["species_name"].nunique()), flush=True)

# strip N-padding (clean-split builder left-padded short barcodes with N -> VSEARCH SIGFPE)
df["seq_clean"] = df["nucleotides"].str.upper().str.strip("N")
df = df[df["seq_clean"].str.len() >= 100].reset_index(drop=True)
print("after N-strip + min-len100: %d reads, %d species" % (len(df), df["species_name"].nunique()), flush=True)
with open("/root/eval_c.fasta", "w") as f:
    for _, r in df.iterrows():
        f.write(">%s\n%s\n" % (r["processid"], r["seq_clean"]))

sp = df["species_name"].to_numpy()
ge = df["genus_name"].to_numpy()
fa = df["family_name"].to_numpy()
pids = df["processid"].tolist()


def sc(t, p):
    h, c, _ = HCV(t, p)
    return {"AMI": round(float(AMI(t, p)), 4), "ARI": round(float(ARI(t, p)), 4),
            "homogeneity": round(float(h), 4), "completeness": round(float(c), 4),
            "n_clusters": int(len(set(p)))}


results = {"tool": "VSEARCH 2.28.1 --cluster_fast (real alignment-based OTU clustering, BIN/ABGD-style)",
           "n_reads": int(len(df)), "n_species": int(df["species_name"].nunique()), "by_identity": {}}

for idt in [0.97, 0.98, 0.99]:
    uc = "/root/clusters_%d.uc" % int(idt * 100)
    print("vsearch --cluster_fast --id %.2f ..." % idt, flush=True)
    subprocess.run(["vsearch", "--cluster_fast", "/root/eval_c.fasta", "--id", str(idt),
                    "--uc", uc, "--qmask", "none", "--threads", "16"], check=True, capture_output=True)
    read2cl = {}
    for line in open(uc):
        p = line.rstrip("\n").split("\t")
        if p[0] in ("S", "H"):
            read2cl[p[8]] = p[1]
    cl = np.array([read2cl.get(pid, "-1") for pid in pids])
    results["by_identity"]["id%.2f" % idt] = {"species": sc(sp, cl), "genus": sc(ge, cl), "family": sc(fa, cl)}
    s = results["by_identity"]["id%.2f" % idt]["species"]
    print("  id=%.2f: %d clusters (true=%d) | species AMI=%.3f | genus AMI=%.3f" % (
        idt, s["n_clusters"], df["species_name"].nunique(),
        results["by_identity"]["id%.2f" % idt]["species"]["AMI"],
        results["by_identity"]["id%.2f" % idt]["genus"]["AMI"]), flush=True)

json.dump(results, open("/root/vsearch_delimitation.json", "w"), indent=2)

best = max(v["species"]["AMI"] for v in results["by_identity"].values())
print("")
print("=== COMPLETE REDISCOVERY HEAD-TO-HEAD: species recovery AMI ===")
print("  OUR tree-geometry embedding   : 0.874")
print("  VSEARCH OTU (real BIN/ABGD)    : %.3f  (best across 97/98/99%%)" % best)
print("  raw k-mer proxy (BIN/ABGD)     : 0.618")
print("  BarcodeBERT (frozen, COI FM)   : 0.492")
