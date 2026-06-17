import torch, json, time, numpy as np, pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI, homogeneity_completeness_v_measure as HCV
from sklearn.preprocessing import normalize

t0 = time.time()
df = pd.read_csv("/root/eval_c_query.csv").dropna(subset=["nucleotides", "species_name"]).reset_index(drop=True)
seqs = df["nucleotides"].str.upper().tolist()
sp = df["species_name"].to_numpy()
ge = df["genus_name"].to_numpy()
fa = df["family_name"].to_numpy()
n_sp = len(set(sp))
print("reads=%d species=%d genera=%d families=%d" % (len(seqs), n_sp, len(set(ge)), len(set(fa))), flush=True)

tok = AutoTokenizer.from_pretrained("bioscan-ml/BarcodeBERT", trust_remote_code=True)
model = AutoModel.from_pretrained("bioscan-ml/BarcodeBERT", trust_remote_code=True).to("cuda").eval()
pad = tok.pad_token_id or 0

print("tokenizing...", flush=True)
ids_list = [tok(s, return_tensors="pt")["input_ids"].squeeze(0)[:512] for s in seqs]

print("embedding %d reads..." % len(ids_list), flush=True)
embs = []
B = 256
for i in range(0, len(ids_list), B):
    b = ids_list[i:i + B]
    L = max(len(x) for x in b)
    ii = torch.full((len(b), L), pad, dtype=torch.long)
    mm = torch.zeros((len(b), L), dtype=torch.long)
    for j, x in enumerate(b):
        ii[j, :len(x)] = x
        mm[j, :len(x)] = 1
    with torch.no_grad():
        out = model(ii.to("cuda"), attention_mask=mm.to("cuda"))
        hs = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
        m = mm.unsqueeze(-1).float().cuda()
        pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1)
    embs.append(pooled.float().cpu().numpy())
    if i % (B * 10) == 0:
        print("  %d/%d" % (i, len(ids_list)), flush=True)
X = normalize(np.vstack(embs))
print("embeddings:", X.shape, "(%.0fs)" % (time.time() - t0), flush=True)


def sc(t, p):
    h, c, _ = HCV(t, p)
    return {"AMI": round(float(AMI(t, p)), 4), "ARI": round(float(ARI(t, p)), 4),
            "homogeneity": round(float(h), 4), "completeness": round(float(c), 4),
            "n_clusters": int(len(set(p)))}


print("clustering (KMeans k=species)...", flush=True)
km = MiniBatchKMeans(n_clusters=n_sp, random_state=0, n_init=3, batch_size=2048).fit(X)
res = {"model": "bioscan-ml/BarcodeBERT", "embed_dim": int(X.shape[1]), "n_reads": int(len(seqs)), "n_species": n_sp,
       "kmeans_k_species": {"species": sc(sp, km.labels_), "genus": sc(ge, km.labels_), "family": sc(fa, km.labels_)},
       "blind": {}}
for thr in [0.05, 0.10, 0.15]:
    ac = AgglomerativeClustering(n_clusters=None, distance_threshold=thr, metric="cosine", linkage="average").fit(X)
    res["blind"]["thr%s" % thr] = {"species": sc(sp, ac.labels_), "genus": sc(ge, ac.labels_)}
json.dump(res, open("/root/barcodebert_rediscovery.json", "w"), indent=2)

s = res["kmeans_k_species"]
sp_ami = s["species"]["AMI"]
print("")
print("=== BarcodeBERT rediscovery (KMeans k=species) ===")
print("  species AMI=%.3f  genus AMI=%.3f  family AMI=%.3f" % (sp_ami, s["genus"]["AMI"], s["family"]["AMI"]))
for thr, d in res["blind"].items():
    print("  blind %s: %d clusters | species AMI=%.3f | genus AMI=%.3f" % (
        thr, d["species"]["n_clusters"], d["species"]["AMI"], d["genus"]["AMI"]))
print("")
print("=== HEAD-TO-HEAD: species recovery AMI (KMeans k=species) ===")
print("  OUR tree-geometry : 0.874")
print("  BarcodeBERT       : %.3f" % sp_ami)
print("  raw k-mer (BIN)   : 0.618")
