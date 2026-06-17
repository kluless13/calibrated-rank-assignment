import subprocess, json, numpy as np, pandas as pd
from sklearn.metrics import adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI, homogeneity_completeness_v_measure as HCV

df = pd.read_csv("/root/eval_c_query.csv").dropna(subset=["nucleotides", "species_name"]).reset_index(drop=True)
df["seq_clean"] = df["nucleotides"].str.upper().str.strip("N")
df = df[df["seq_clean"].str.len() >= 100].reset_index(drop=True)
with open("/root/eval_c.fasta", "w") as f:
    for _, r in df.iterrows():
        f.write(">%s\n%s\n" % (r["processid"], r["seq_clean"]))
sp = df["species_name"].to_numpy(); ge = df["genus_name"].to_numpy(); fa = df["family_name"].to_numpy()
pids = df["processid"].tolist()


def sc(t, p):
    h, c, _ = HCV(t, p)
    return {"AMI": round(float(AMI(t, p)), 4), "ARI": round(float(ARI(t, p)), 4),
            "homogeneity": round(float(h), 4), "completeness": round(float(c), 4), "n_clusters": int(len(set(p)))}


results = {"tool": "cd-hit-est 4.8.1 (greedy alignment clustering, BIN-style)", "by_identity": {}}
for idt, n in [(0.97, 8), (0.98, 8), (0.99, 10)]:
    out = "/root/cdhit_%d" % int(idt * 100)
    subprocess.run(["cd-hit-est", "-i", "/root/eval_c.fasta", "-o", out, "-c", str(idt),
                    "-n", str(n), "-d", "0", "-T", "16", "-M", "0"], check=True, capture_output=True)
    read2cl = {}; cid = -1
    for line in open(out + ".clstr"):
        if line.startswith(">Cluster"):
            cid += 1
        else:
            rid = line.split(">", 1)[1].split("...")[0].strip()
            read2cl[rid] = cid
    cl = np.array([read2cl.get(pid, -1) for pid in pids])
    results["by_identity"]["id%.2f" % idt] = {"species": sc(sp, cl), "genus": sc(ge, cl), "family": sc(fa, cl)}
    s = results["by_identity"]["id%.2f" % idt]["species"]
    print("  cd-hit id=%.2f: %d clusters (true=%d) | species AMI=%.3f | genus AMI=%.3f" % (
        idt, s["n_clusters"], df["species_name"].nunique(), s["AMI"],
        results["by_identity"]["id%.2f" % idt]["genus"]["AMI"]), flush=True)
json.dump(results, open("/root/cdhit_delimitation.json", "w"), indent=2)
best = max(v["species"]["AMI"] for v in results["by_identity"].values())
print("cd-hit best species AMI: %.3f" % best, flush=True)
