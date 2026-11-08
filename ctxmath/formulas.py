"""All equations used in this project.

KV cache, bytes:       M_kv(n) = 2 * h_kv * d_h * b * ( L_global * n + L_local * min(n, W) )
Weights, bytes:        M_w = N * b_w
Linear FLOPs / token:  F_lin ~ 2 * N_nonembed
Attention FLOPs at position t: F_att(t) ~ 4 * L * t * h * d_h
Crossover (F_att = F_lin): t* = N_nonembed / (2 * L * h * d_h)
"""
from .models import Arch

GB = 1024 ** 3


def params(a: Arch):
    """Parameter count from the architecture (embeddings, attention, gated MLP, norms)."""
    attn = a.dim * a.heads * a.head_dim + 2 * a.dim * a.kv_heads * a.head_dim + a.heads * a.head_dim * a.dim
    mlp = 3 * a.dim * a.ffn
    norms = (4 if a.family == "Gemma" and a.name != "Gemma 7B" else 2) * a.dim
    emb = a.vocab * a.dim * (1 if a.tied else 2)
    return emb + a.layers * (attn + mlp + norms) + a.dim


def nonembed(a: Arch):
    return params(a) - a.vocab * a.dim * (1 if a.tied else 2)


def kv_bytes(a: Arch, n, bytes_per=2, batch=1, kv_heads=None, window=True):
    h = a.kv_heads if kv_heads is None else kv_heads
    local = a.local_layers if window else 0
    g = a.layers - local
    return 2 * h * a.head_dim * bytes_per * batch * (g * n + local * min(n, a.window if local else n))


def kv_per_token(a: Arch, bytes_per=2):
    """KV bytes per token when every layer keeps the full context."""
    return 2 * a.kv_heads * a.head_dim * bytes_per * a.layers


def attention_share(a: Arch, t):
    att = 4 * a.layers * t * a.heads * a.head_dim
    if a.local_layers:
        att = 4 * a.heads * a.head_dim * ((a.layers - a.local_layers) * t + a.local_layers * min(t, a.window))
    return att / (att + 2 * nonembed(a))


def crossover(a: Arch):
    """Context position where attention FLOPs equal the linear-layer FLOPs."""
    lo, hi = 1, 1 << 26
    while lo < hi:
        mid = (lo + hi) // 2
        if attention_share(a, mid) < 0.5:
            lo = mid + 1
        else:
            hi = mid
    return lo if attention_share(a, lo) >= 0.5 else None


def max_context(a: Arch, gpu_gb=80, weight_bytes=2, kv_bytes_per=2, reserve_gb=0):
    free = gpu_gb * GB - params(a) * weight_bytes - reserve_gb * GB
    if free <= 0:
        return 0
    lo, hi = 0, 1 << 30
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if kv_bytes(a, mid, kv_bytes_per) <= free:
            lo = mid
        else:
            hi = mid - 1
    return lo
