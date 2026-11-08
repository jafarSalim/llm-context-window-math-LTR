import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ctxmath.models import MODELS
from ctxmath.formulas import params, kv_bytes, kv_per_token, attention_share, crossover, GB

M = {a.name: a for a in MODELS}
NAMED = {"Llama 3.2 1B": 1.0, "Llama 3.2 3B": 3.0, "Llama 3.1 8B": 8.0, "Llama 3.1 70B": 70.0, "Llama 3.1 405B": 405.0, "Gemma 2 27B": 27.0, "Gemma 3 27B": 27.0}


def test_parameter_counts_match_model_names():
    for name, size in NAMED.items():
        assert abs(params(M[name]) / 1e9 / size - 1) < 0.25, name


def test_llama_8b_kv_cache():
    a = M["Llama 3.1 8B"]
    assert kv_per_token(a) == 131072            # 128 KiB per token in bf16
    assert kv_bytes(a, 131072) / GB == 16.0     # 16 GiB at 128K tokens


def test_sliding_window_caps_local_layers():
    a = M["Gemma 3 27B"]
    full = kv_bytes(a, 131072, window=False)
    win = kv_bytes(a, 131072)
    assert win < full / 5


def test_attention_share_grows_and_crossover_is_half():
    a = M["Llama 3.1 8B"]
    assert attention_share(a, 1024) < attention_share(a, 131072)
    t = crossover(a)
    assert abs(attention_share(a, t) - 0.5) < 0.001
