"""Architectures from official configuration files.

Llama: https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py
Gemma: https://github.com/google/gemma_pytorch/blob/main/gemma/config.py
"""
from dataclasses import dataclass


@dataclass
class Arch:
    name: str
    family: str
    layers: int
    dim: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    vocab: int
    tied: bool
    local_layers: int = 0     # layers using sliding-window attention
    window: int = 0
    max_context: int = 0


def llama_ffn(dim, mult, multiple_of):
    """FFN width exactly as computed in the Llama reference code."""
    h = int(2 * (4 * dim) / 3)
    h = int(mult * h)
    return multiple_of * ((h + multiple_of - 1) // multiple_of)


V3 = 128256
MODELS = [
    Arch("Llama 3.2 1B", "Llama", 16, 2048, 32, 8, 64, llama_ffn(2048, 1.5, 256), V3, True, max_context=131072),
    Arch("Llama 3.2 3B", "Llama", 28, 3072, 24, 8, 128, llama_ffn(3072, 1.0, 256), V3, True, max_context=131072),
    Arch("Llama 3.1 8B", "Llama", 32, 4096, 32, 8, 128, llama_ffn(4096, 1.3, 1024), V3, False, max_context=131072),
    Arch("Llama 3.1 70B", "Llama", 80, 8192, 64, 8, 128, llama_ffn(8192, 1.3, 4096), V3, False, max_context=131072),
    Arch("Llama 3.1 405B", "Llama", 126, 16384, 128, 8, 128, llama_ffn(16384, 1.2, 4096), V3, False, max_context=131072),
    Arch("Gemma 7B", "Gemma", 28, 3072, 16, 16, 256, 24576, 256000, True, max_context=8192),
    Arch("Gemma 2 27B", "Gemma", 46, 4608, 32, 16, 128, 36864, 256000, True, local_layers=23, window=4096, max_context=8192),
    Arch("Gemma 3 27B", "Gemma", 62, 5376, 32, 16, 128, 5376 * 4, 262144, True, local_layers=52, window=1024, max_context=131072),
]
