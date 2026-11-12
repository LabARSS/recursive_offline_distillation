import glob, json, os, torch
from safetensors import safe_open
from transformers import AutoTokenizer

ckpts = sorted(glob.glob("checkpoint-*"), key=lambda p: int(p.split("-")[1]))
base = json.load(open(f"{ckpts[0]}/adapter_config.json"))["base_model_name_or_path"]
print("base:", base)

def base_tensor(name):
    for f in sorted(glob.glob(f"{base}/*.safetensors")):
        with safe_open(f, "pt") as s:
            if name in s.keys():
                return s.get_tensor(name).float()
    return None

emb = base_tensor("model.embed_tokens.weight")
lm = base_tensor("lm_head.weight")
tok = AutoTokenizer.from_pretrained(base)
print("vocab rows:", emb.shape[0], "| lm_head in base:", lm is not None)
mean_norm = emb.norm(dim=1).mean().item()
print(f"mean row norm (all base rows): {mean_norm:.4f}")

prev = None
for c in ckpts:
    p = f"{c}/thinking_token_rows.pt"
    if not os.path.exists(p):
        print(f"{c}: MISSING thinking_token_rows.pt"); continue
    d = torch.load(p, map_location="cpu", weights_only=True)
    ids, rows = d["new_ids"], d["input_rows"].float()
    b = emb[ids]
    print(f"\n{c}: ids={ids} -> {tok.convert_ids_to_tokens(ids)} | keys={list(d)} | nan={rows.isnan().any().item()}")
    for i, t in enumerate(ids):
        delta = (rows[i] - b[i]).norm().item()
        cos = torch.nn.functional.cosine_similarity(rows[i], b[i], dim=0).item()
        step = (rows[i] - prev[i]).norm().item() if prev is not None else float("nan")
        print(f"  id {t}: |base|={b[i].norm():.4f} |trained|={rows[i].norm():.4f} "
              f"|Δ vs base|={delta:.4f} cos(base)={cos:.4f} |Δ vs prev ckpt|={step:.4f}")
    prev = rows
