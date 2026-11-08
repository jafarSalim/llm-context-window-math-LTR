"""Builds excel/llm_context_window_math.xlsx. Every number is a live formula on the architecture table."""
import json, pathlib
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.chart import BarChart, LineChart, BubbleChart, DoughnutChart, Reference, Series
from openpyxl.chart.series import DataPoint
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.chart.label import DataLabelList, DataLabel
from openpyxl.utils import get_column_letter as L
from ctxmath.models import MODELS
from xlkit.charts import BLUE_IN, NOTE, OFFICE, header, style, fill, labels, chart_sheet, tiles

wb = Workbook(); wb.remove(wb.active)
thin = Side(style="thin", color="BFBFBF")

# ---- Inputs ----
inp = wb.create_sheet("Inputs")
inp.column_dimensions["A"].width, inp.column_dimensions["B"].width = 52, 12
inp["A1"] = "Inputs (blue cells can be changed)"; inp["A1"].font = Font(name="Calibri", bold=True, size=15, color="1F4E79")
rows = [("Bytes per KV value (2 = bf16, 1 = fp8)", 2), ("Bytes per weight (2 = bf16)", 2), ("GPU memory, GiB (an 80 GB card reports about 80 GiB)", 80), ("Context for single-number tiles", 131072)]
for i, (a, b) in enumerate(rows, 3):
    inp.cell(i, 1, a); c = inp.cell(i, 2, b); c.font = BLUE_IN; c.number_format = "#,##0"
KVB, WB, GPU, NREF = "Inputs!$B$3", "Inputs!$B$4", "Inputs!$B$5", "Inputs!$B$6"

# ---- Architecture table ----
md = wb.create_sheet("Models")
cols = ["Model", "Family", "Layers", "Dim", "Heads", "KV heads", "Head dim", "FFN", "Vocab", "Tied emb.", "Local layers", "Window",
        "Norms/layer", "Trained context", "Params (B)", "Non-emb. params", "KV per token (KiB)", "Attention = MLP at token"]
header(md, 1, cols, [16, 8, 8, 8, 8, 9, 9, 9, 9, 8, 9, 9, 9, 11, 10, 14, 12, 14])
md.row_dimensions[1].height = 45
for i, a in enumerate(MODELS, 2):
    vals = [a.name, a.family, a.layers, a.dim, a.heads, a.kv_heads, a.head_dim, a.ffn, a.vocab, int(a.tied), a.local_layers, a.window,
            4 if a.family == "Gemma" and a.name != "Gemma 7B" else 2, a.max_context]
    for j, v in enumerate(vals, 1):
        c = md.cell(i, j, v)
        if j >= 3:
            c.font = BLUE_IN; c.number_format = "#,##0"
    md.cell(i, 15, f"=(I{i}*D{i}*(2-J{i})+C{i}*(D{i}*E{i}*G{i}+2*D{i}*F{i}*G{i}+E{i}*G{i}*D{i}+3*D{i}*H{i}+M{i}*D{i})+D{i})/1E9").number_format = "0.00"
    md.cell(i, 16, f"=O{i}*1E9-I{i}*D{i}*(2-J{i})").number_format = "#,##0"
    md.cell(i, 17, f"=2*F{i}*G{i}*{KVB}*C{i}/1024").number_format = "#,##0"
    md.cell(i, 18, f"=IF(K{i}=0,P{i}/(2*C{i}*E{i}*G{i}),\"see note\")").number_format = "#,##0"
NM = len(MODELS) + 1
md.cell(NM + 2, 1, "Sources: Llama configs from github.com/meta-llama/llama-models (models/sku_list.py); Gemma configs from github.com/google/gemma_pytorch (gemma/config.py).").font = NOTE
md.cell(NM + 3, 1, "Params = embeddings + layers × (attention + gated MLP + norms). Matches the size in each model's name; Gemma 7B counts its 256k-token embedding.").font = NOTE
md.cell(NM + 4, 1, "Attention = MLP point for sliding-window models is computed in Python (results/analysis.json), because the window changes the formula.").font = NOTE
md.freeze_panes = "B2"
ROW = {a.name: i for i, a in enumerate(MODELS, 2)}


def kv_formula(r, n_ref, heads=None, window=True, bytes_ref=KVB):
    h = heads if heads is not None else f"Models!$F${r}"
    if window:
        return f"2*{h}*Models!$G${r}*{bytes_ref}*((Models!$C${r}-Models!$K${r})*{n_ref}+Models!$K${r}*MIN({n_ref},IF(Models!$K${r}>0,Models!$L${r},{n_ref})))/1024^3"
    return f"2*{h}*Models!$G${r}*{bytes_ref}*Models!$C${r}*{n_ref}/1024^3"


# ---- KV grid (heatmap + line chart) ----
CTX = [1024, 4096, 8192, 32768, 131072, 524288, 1048576]
CL = ["1K", "4K", "8K", "32K", "128K", "512K", "1M"]
kv = wb.create_sheet("KV Cache")
kv["A1"] = "KV cache for one sequence, GiB (bf16)"; kv["A1"].font = Font(name="Calibri", bold=True, size=15, color="1F4E79")
kv["A2"] = "Grey italic = beyond the context the model was trained for: memory it would need, not something it supports."; kv["A2"].font = NOTE
header(kv, 3, ["Model"] + CL, [16] + [10] * len(CTX))
for j, n in enumerate(CTX, 2):
    kv.cell(4 + len(MODELS) + 1, j, n).font = Font(name="Calibri", color="FFFFFF", size=6)
CR = 4 + len(MODELS) + 1  # hidden row with numeric contexts
for i, a in enumerate(MODELS, 4):
    kv.cell(i, 1, a.name)
    for j, n in enumerate(CTX, 2):
        c = kv.cell(i, j, "=" + kv_formula(ROW[a.name], f"{L(j)}${CR}"))
        c.number_format, c.alignment, c.border = "#,##0.0", Alignment(horizontal="center"), Border(thin, thin, thin, thin)
        if n > a.max_context:
            c.font = Font(name="Calibri", italic=True, color="7F7F7F")
last = 3 + len(MODELS)
kv.conditional_formatting.add(f"B4:{L(1 + len(CTX))}{last}", ColorScaleRule(start_type="num", start_value=0, start_color="63BE7B", mid_type="num", mid_value=40, mid_color="FFEB84", end_type="num", end_value=300, end_color="F8696B"))
kv.page_setup.orientation = "landscape"; kv.sheet_properties.pageSetUpPr.fitToPage = True; kv.print_area = f"A1:{L(1 + len(CTX))}{last}"

# ---- Memory budget of Llama 3.1 8B ----
bd = wb.create_sheet("Budget Data")
header(bd, 1, ["Context", "Weights (GiB)", "KV cache (GiB)", "GPU memory"], [12, 14, 15, 12])
r8 = ROW["Llama 3.1 8B"]
for i, n in enumerate([4096, 32768, 131072, 262144, 524288], 2):
    bd.cell(i, 1, f"{n // 1024}K"); bd.cell(i, 6, n)
    bd.cell(i, 2, f"=Models!$O${r8}*1E9*{WB}/1024^3").number_format = "0.0"
    bd.cell(i, 3, "=" + kv_formula(r8, f"F{i}")).number_format = "0.0"
    bd.cell(i, 4, f"={GPU}").number_format = "0.0"

# ---- GQA ----
gq = wb.create_sheet("GQA Data")
r70 = ROW["Llama 3.1 70B"]
header(gq, 1, ["Attention type", "KV heads", "KV at 128K (GiB)"], [30, 10, 16])
for i, (nm, h) in enumerate([("Multi-head (MHA)", 64), ("Grouped-query (GQA), actual", 8), ("Multi-query (MQA)", 1)], 2):
    gq.cell(i, 1, nm); gq.cell(i, 2, h).font = BLUE_IN
    gq.cell(i, 3, "=" + kv_formula(r70, NREF, heads=f"B{i}")).number_format = "0.0"

# ---- Attention share ----
at = wb.create_sheet("Attention Data")
TS = [1024, 4096, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
sel = ["Llama 3.1 8B", "Llama 3.1 70B", "Gemma 3 27B"]
header(at, 1, ["Context position"] + sel + ["8B: attention", "8B: MLP and projections"], [16, 13, 13, 13, 13, 18])
for i, t in enumerate(TS, 2):
    at.cell(i, 1, f"{t // 1024}K" if t < 1048576 else "1M"); at.cell(i, 8, t)
    for j, nm in enumerate(sel, 2):
        r = ROW[nm]
        att = f"4*Models!$E${r}*Models!$G${r}*((Models!$C${r}-Models!$K${r})*$H{i}+Models!$K${r}*MIN($H{i},IF(Models!$K${r}>0,Models!$L${r},$H{i})))"
        at.cell(i, j, f"={att}/({att}+2*Models!$P${r})").number_format = "0%"
    at.cell(i, 5, f"=B{i}").number_format = "0%"; at.cell(i, 6, f"=1-B{i}").number_format = "0%"

# ---- Gemma 3 27B reductions (waterfall) ----
gw = wb.create_sheet("Gemma3 Data")
r3 = ROW["Gemma 3 27B"]
header(gw, 1, ["Step", "KV at 128K (GiB)", "Base", "Saved", "Total"], [36, 16, 10, 10, 10])
steps = [("All global, 32 KV heads, bf16", "=" + kv_formula(r3, NREF, heads=f"Models!$E${r3}", window=False, bytes_ref="2")),
         ("Grouped-query: 16 KV heads", "=" + kv_formula(r3, NREF, window=False, bytes_ref="2")),
         ("Sliding window: 52 of 62 layers", "=" + kv_formula(r3, NREF, bytes_ref="2")),
         ("FP8 KV cache", "=" + kv_formula(r3, NREF, bytes_ref="1"))]
for i, (nm, f) in enumerate(steps, 2):
    gw.cell(i, 1, nm); gw.cell(i, 2, f).number_format = "0.0"
gw.cell(6, 1, "Result"); gw.cell(6, 2, "=B5").number_format = "0.0"
for i in range(2, 7):
    if i in (2, 6):
        gw.cell(i, 3, 0); gw.cell(i, 4, 0); gw.cell(i, 5, f"=B{i}")
    else:
        gw.cell(i, 3, f"=B{i}"); gw.cell(i, 4, f"=B{i - 1}-B{i}"); gw.cell(i, 5, 0)
    for c in (3, 4, 5):
        gw.cell(i, c).number_format = "0.0"
gw["A8"], gw["B8"] = "Global layers (10)", f"=2*Models!$F${r3}*Models!$G${r3}*2*(Models!$C${r3}-Models!$K${r3})*{NREF}/1024^3"
gw["A9"], gw["B9"] = "Local layers (52)", "=B4-B8"
for c in ("B8", "B9"):
    gw[c].number_format = "0.0"

# ---- Max context on one GPU ----
mx = wb.create_sheet("GPU Data")
fit = ["Llama 3.2 1B", "Llama 3.2 3B", "Llama 3.1 8B", "Gemma 7B", "Gemma 2 27B", "Gemma 3 27B"]
header(mx, 1, ["Model", "bf16 KV", "fp8 KV", "Trained context"], [16, 14, 14, 14])
for i, nm in enumerate(fit, 2):
    r = ROW[nm]
    mx.cell(i, 1, nm)
    for col, b in ((2, 2), (3, 1)):
        free = f"({GPU}*1024^3-Models!$O${r}*1E9*{WB})"
        unit = f"(2*Models!$F${r}*Models!$G${r}*{b})"
        f = (f"=IF(Models!$K${r}=0,{free}/({unit}*Models!$C${r}),"
             f"IF({free}/({unit}*Models!$C${r})<=Models!$L${r},{free}/({unit}*Models!$C${r}),({free}/{unit}-Models!$K${r}*Models!$L${r})/(Models!$C${r}-Models!$K${r})))")
        mx.cell(i, col, f).number_format = "#,##0"
    mx.cell(i, 4, f"=Models!$N${r}").number_format = "#,##0"
mx.cell(len(fit) + 3, 1, "One 80 GB GPU, bf16 weights, batch of 1, no other memory reserved. Llama 3.1 70B and 405B do not fit in 80 GB at bf16.").font = NOTE

# ---- Calculator ----
cc = wb.create_sheet("KV Calculator", 0)
cc.column_dimensions["A"].width, cc.column_dimensions["B"].width = 36, 14
cc["A1"] = "KV cache calculator"; cc["A1"].font = Font(name="Calibri", bold=True, size=16, color="1F4E79")
items = [("Layers", 32), ("KV heads", 8), ("Head dimension", 128), ("Bytes per value", 2), ("Context length (tokens)", 131072), ("Sequences in the batch", 1),
         ("Layers with sliding window", 0), ("Window size", 4096)]
for i, (a, b) in enumerate(items, 3):
    cc.cell(i, 1, a); c = cc.cell(i, 2, b); c.font = BLUE_IN; c.number_format = "#,##0"
cc["A12"], cc["B12"] = "KV cache (GiB)", "=2*B4*B5*B6*B8*((B3-B9)*B7+B9*MIN(B7,IF(B9>0,B10,B7)))/1024^3"
cc["A13"], cc["B13"] = "KV per token, all layers (KiB)", "=2*B4*B5*B6*B3/1024"
cc["B12"].number_format, cc["B13"].number_format = "#,##0.00", "#,##0"
cc["A12"].font = cc["B12"].font = Font(name="Calibri", bold=True, size=13)
cc["A15"] = "Defaults are Llama 3.1 8B at 128K tokens."; cc["A15"].font = NOTE

# ================= Charts =================
lc = LineChart()
for k, a in enumerate(MODELS):
    s = Series(Reference(kv, min_col=2, min_row=4 + k, max_col=1 + len(CTX), max_row=4 + k), title=a.name)
    fill(s, OFFICE[k], line=True); s.marker.symbol = "circle"; s.marker.size = 6; s.marker.graphicalProperties = GraphicalProperties(solidFill=OFFICE[k])
    lc.series.append(s)
lc.set_categories(Reference(kv, min_col=2, min_row=3, max_col=1 + len(CTX), max_row=3))
lc.y_axis.scaling.logBase = 10; lc.y_axis.title = "KV cache, GiB (log)"; lc.x_axis.title = "Context length"; lc.legend.position = "r"
style(lc, "KV cache grows linearly with context, for one sequence (bf16)", 26, 14)
chart_sheet(wb, "Chart KV Growth", lc)

bc = BarChart(); bc.type = "col"; bc.grouping = "stacked"; bc.overlap = 100; bc.gapWidth = 60
for col, colr in ((2, "4472C4"), (3, "ED7D31")):
    s = Series(Reference(bd, min_col=col, min_row=1, max_row=6), title_from_data=True); fill(s, colr); labels(s, "0.0"); bc.series.append(s)
bc.set_categories(Reference(bd, min_col=1, min_row=2, max_row=6))
lg = LineChart(); s = Series(Reference(bd, min_col=4, min_row=1, max_row=6), title="GPU memory (80 GiB)"); fill(s, "C00000", line=True); s.marker.symbol = "none"; lg.series.append(s)
bc += lg
bc.y_axis.title = "GiB"; bc.x_axis.title = "Context length"; bc.legend.position = "b"
style(bc, "Llama 3.1 8B on one 80 GB GPU: weights + KV cache (the trained limit is 128K)")
chart_sheet(wb, "Chart Memory Budget", bc)

gc = BarChart(); gc.type = "col"; gc.gapWidth = 60
s = Series(Reference(gq, min_col=3, min_row=1, max_row=4), title_from_data=True); fill(s, "4472C4"); labels(s, "0.0", "outEnd")
for k, colr in enumerate(["A5A5A5", "70AD47", "FFC000"]):
    p = DataPoint(idx=k); p.graphicalProperties = GraphicalProperties(solidFill=colr); s.dPt.append(p)
gc.series.append(s); gc.set_categories(Reference(gq, min_col=1, min_row=2, max_row=4))
gc.legend = None; gc.y_axis.title = "GiB at 128K tokens"
style(gc, "Llama 3.1 70B: why grouped-query attention matters (128K context)")
chart_sheet(wb, "Chart GQA", gc)

ac = LineChart()
for j, colr in zip(range(2, 5), ["4472C4", "ED7D31", "70AD47"]):
    s = Series(Reference(at, min_col=j, min_row=1, max_row=len(TS) + 1), title_from_data=True)
    fill(s, colr, line=True); s.marker.symbol = "circle"; s.marker.size = 7; s.marker.graphicalProperties = GraphicalProperties(solidFill=colr); ac.series.append(s)
ac.set_categories(Reference(at, min_col=1, min_row=2, max_row=len(TS) + 1))
ac.y_axis.numFmt = "0%"; ac.y_axis.scaling.max = 1; ac.y_axis.title = "Share of compute for the next token"; ac.x_axis.title = "Position in the context"
ac.legend.position = "b"
style(ac, "How much of the work is attention? It grows with the context")
chart_sheet(wb, "Chart Attention Share", ac)

pc = BarChart(); pc.type = "col"; pc.grouping = "percentStacked"; pc.overlap = 100; pc.gapWidth = 40
for j, colr in ((5, "ED7D31"), (6, "4472C4")):
    s = Series(Reference(at, min_col=j, min_row=1, max_row=len(TS) + 1), title_from_data=True); fill(s, colr); labels(s, "0%"); pc.series.append(s)
pc.set_categories(Reference(at, min_col=1, min_row=2, max_row=len(TS) + 1))
pc.y_axis.numFmt = "0%"; pc.legend.position = "b"; pc.x_axis.title = "Position in the context"
style(pc, "Llama 3.1 8B: compute for one new token, attention vs everything else")
chart_sheet(wb, "Chart Compute Split", pc)

wc = BarChart(); wc.type = "col"; wc.grouping = "stacked"; wc.overlap = 100; wc.gapWidth = 50
for col, colr in ((3, None), (4, "70AD47"), (5, "4472C4")):
    s = Series(Reference(gw, min_col=col, min_row=2, max_row=6), title=gw.cell(1, col).value)
    if colr is None:
        s.graphicalProperties = GraphicalProperties(noFill=True); s.graphicalProperties.line.noFill = True
    else:
        fill(s, colr); labels(s, "0.0")
        for k in range(5):
            show = (col == 4 and 0 < k < 4) or (col == 5 and k in (0, 4))
            if not show:
                s.dLbls.dLbl.append(DataLabel(idx=k, showVal=False, showSerName=False, showCatName=False, showLegendKey=False, showPercent=False))
    wc.series.append(s)
wc.set_categories(Reference(gw, min_col=1, min_row=2, max_row=6))
wc.legend = None; wc.y_axis.title = "GiB at 128K tokens"
style(wc, "Gemma 3 27B: three design choices shrink the KV cache 24×")
chart_sheet(wb, "Chart Gemma Waterfall", wc, "Green = memory saved by each step. Steps 1 and 2 are what the model would need without its actual design.")

dc = DoughnutChart(holeSize=55)
dc.add_data(Reference(gw, min_col=2, min_row=8, max_row=9)); dc.set_categories(Reference(gw, min_col=1, min_row=8, max_row=9))
for k, colr in enumerate(["4472C4", "70AD47"]):
    p = DataPoint(idx=k); p.graphicalProperties = GraphicalProperties(solidFill=colr); dc.series[0].dPt.append(p)
dc.dataLabels = DataLabelList(); dc.dataLabels.showPercent = True
for k in ("showVal", "showCatName", "showSerName", "showLegendKey"):
    setattr(dc.dataLabels, k, False)
dc.legend.position = "r"
style(dc, "Gemma 3 27B at 128K: 10 global layers hold almost all the KV cache", 18, 12)
chart_sheet(wb, "Chart Gemma Split", dc)

mc = BarChart(); mc.type = "bar"; mc.gapWidth = 40
for col, colr in ((2, "4472C4"), (3, "70AD47"), (4, "A5A5A5")):
    s = Series(Reference(mx, min_col=col, min_row=1, max_row=len(fit) + 1), title_from_data=True); fill(s, colr); labels(s, "#,##0", "outEnd"); mc.series.append(s)
mc.set_categories(Reference(mx, min_col=1, min_row=2, max_row=len(fit) + 1))
mc.x_axis.scaling.orientation = "maxMin"; mc.y_axis.scaling.logBase = 10; mc.y_axis.title = "Tokens (log)"; mc.legend.position = "b"
style(mc, "Longest context that fits in one 80 GB GPU (memory only), vs the trained context", 26, 15)
chart_sheet(wb, "Chart GPU Limit", mc)

bb = BubbleChart(); bb.bubbleScale = 40
for k, a in enumerate(MODELS):
    r = ROW[a.name]
    s = Series(values=Reference(md, min_col=17, min_row=r), xvalues=Reference(md, min_col=15, min_row=r), zvalues=Reference(md, min_col=3, min_row=r), title=a.name)
    s.graphicalProperties = GraphicalProperties(solidFill=OFFICE[k]); bb.series.append(s)
bb.x_axis.scaling.logBase = 10; bb.x_axis.scaling.min = 0.5; bb.x_axis.scaling.max = 1000
bb.x_axis.title = "Parameters, billions (log)"; bb.y_axis.title = "KV per token, KiB (bf16)"; bb.legend.position = "r"
style(bb, "Bigger model, bigger KV per token? Not always (bubble = layers)", 26, 14)
chart_sheet(wb, "Chart Size vs KV", bb)

tiles(wb, [(f"=Models!Q{r8}*{NREF}/1024^2", '0" GiB"', "One 128K chat, Llama 3.1 8B", "F4B183"),
           ("='GQA Data'!C2/'GQA Data'!C3", '0"×"', "Less KV with GQA, 70B", "A9D18E"),
           ("='Gemma3 Data'!B3/'Gemma3 Data'!B4", '0.0"×"', "Less KV, sliding window", "9DC3E6"),
           (f"=Models!R{r8}", '#,##0', "Attention passes MLP, 8B", "FFD966")])
order = ["Overview", "KV Calculator", "Chart KV Growth", "KV Cache", "Chart Memory Budget", "Chart GQA", "Chart Attention Share", "Chart Compute Split",
         "Chart Gemma Waterfall", "Chart Gemma Split", "Chart GPU Limit", "Chart Size vs KV", "Inputs", "Models", "Budget Data", "GQA Data",
         "Attention Data", "Gemma3 Data", "GPU Data"]
wb._sheets = [wb[n] for n in order]
pathlib.Path("excel").mkdir(exist_ok=True)
wb.save("excel/llm_context_window_math.xlsx")
print("saved")
