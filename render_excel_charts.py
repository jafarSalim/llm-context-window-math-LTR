"""Exports the workbook's Excel charts as PNG images for the README (needs LibreOffice)."""
from xlkit.charts import render

PAGES = {"Overview": "overview.png", "Chart KV Growth": "kv_growth.png", "KV Cache": "kv_heatmap.png", "Chart Memory Budget": "memory_budget.png",
         "Chart GQA": "gqa.png", "Chart Attention Share": "attention_share.png", "Chart Compute Split": "compute_split.png",
         "Chart Gemma Waterfall": "gemma_waterfall.png", "Chart Gemma Split": "gemma_split.png", "Chart GPU Limit": "gpu_limit.png", "Chart Size vs KV": "size_vs_kv.png"}
for n in render("excel/llm_context_window_math.xlsx", PAGES, "results"):
    print(n, "->", PAGES[n])
