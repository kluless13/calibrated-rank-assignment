import os, torch
from transformers import AutoConfig, AutoTokenizer, AutoModel

NAME = "zhihan1996/DNABERT-S"
cfg = AutoConfig.from_pretrained(NAME, trust_remote_code=True)
attn = {k: v for k, v in cfg.__dict__.items() if any(s in k.lower() for s in ["flash", "attn", "attention", "triton", "unpad"])}
print("attn-related config:", attn, flush=True)

# disable any boolean flash/triton flags
for k in list(cfg.__dict__):
    if any(s in k.lower() for s in ["flash", "triton"]) and isinstance(getattr(cfg, k), bool):
        setattr(cfg, k, False)
        print("  set", k, "-> False", flush=True)
for k in ["use_flash_attn", "flash_attn", "use_triton"]:
    if hasattr(cfg, k):
        setattr(cfg, k, False)

seq = "ACGTACGTAC" * 6


def smoke(device, use_cfg):
    tok = AutoTokenizer.from_pretrained(NAME, trust_remote_code=True)
    kw = {"trust_remote_code": True}
    if use_cfg:
        kw["config"] = cfg
    model = AutoModel.from_pretrained(NAME, **kw).to(device).eval()
    ids = tok(seq, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        hs = model(ids)[0]
    return hs.shape[-1]


for label, device, use_cfg in [("GPU+disabled-flash", "cuda", True), ("CPU", "cpu", False)]:
    try:
        dim = smoke(device, use_cfg)
        print("DNABERT-S OK [%s] | dim=%d" % (label, dim), flush=True)
        print("WORKING_DEVICE=%s USE_CFG=%d" % (device, int(use_cfg)), flush=True)
        break
    except Exception as e:
        print("DNABERT-S FAILED [%s]: %s" % (label, repr(e)[:200]), flush=True)
