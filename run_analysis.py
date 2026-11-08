"""Computes every number in the README and writes results/analysis.json."""
import json, pathlib
from dataclasses import replace
from ctxmath.models import MODELS
from ctxmath.formulas import GB, params, nonembed, kv_bytes, kv_per_token, attention_share, crossover, max_context

CTX = [1024, 4096, 8192, 32768, 131072, 524288, 1048576]
out = dict(models=[], contexts=CTX)
for a in MODELS:
    out["models"].append(dict(name=a.name, params_b=params(a) / 1e9, kv_per_token_kib=kv_per_token(a) / 1024,
                              kv_gib={str(n): kv_bytes(a, n) / GB for n in CTX}, crossover=crossover(a),
                              max_ctx_80gb_bf16=max_context(a, 80, 2, 2), max_ctx_80gb_fp8kv=max_context(a, 80, 2, 1),
                              attention_share={str(n): attention_share(a, n) for n in CTX}))
m = {a.name: a for a in MODELS}
l70, g3, l8 = m["Llama 3.1 70B"], m["Gemma 3 27B"], m["Llama 3.1 8B"]
n = 131072
out["gqa_70b"] = {f"{h} KV heads": kv_bytes(l70, n, kv_heads=h) / GB for h in (64, 8, 1)}
mha = kv_bytes(g3, n, kv_heads=g3.heads, window=False) / GB
gqa = kv_bytes(g3, n, window=False) / GB
win = kv_bytes(g3, n) / GB
fp8 = kv_bytes(g3, n, bytes_per=1) / GB
out["gemma3_steps"] = {"All global, 32 KV heads, bf16": mha, "Grouped-query attention, 16 KV heads": gqa,
                       "Sliding window on 52 of 62 layers": win, "FP8 KV cache": fp8}
glob = 2 * g3.kv_heads * g3.head_dim * 2 * (g3.layers - g3.local_layers) * n / GB
out["gemma3_split"] = {"Global layers (10)": glob, "Local layers (52)": win - glob}
out["budget_8b"] = {str(c): dict(weights=params(l8) * 2 / GB, kv=kv_bytes(l8, c) / GB) for c in (4096, 32768, 131072, 262144, 524288)}
out["users_8b_32k"] = int((80 * GB - params(l8) * 2) // kv_bytes(l8, 32768))
pathlib.Path("results").mkdir(exist_ok=True)
json.dump(out, open("results/analysis.json", "w"), indent=1)
for x in out["models"]:
    print(f"{x['name']:16s} {x['params_b']:7.2f}B kv/tok {x['kv_per_token_kib']:5.0f}KiB 128k {x['kv_gib']['131072']:6.1f}GiB 1M {x['kv_gib']['1048576']:7.1f}GiB cross {x['crossover']} max80 {x['max_ctx_80gb_bf16']:,} fp8 {x['max_ctx_80gb_fp8kv']:,} att@128k {100*x['attention_share']['131072']:.0f}%")
print(out["gqa_70b"]); print(out["gemma3_steps"]); print(out["gemma3_split"]); print(out["budget_8b"]); print("users", out["users_8b_32k"])
