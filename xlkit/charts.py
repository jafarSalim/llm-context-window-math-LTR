"""Small helpers for native Excel charts with openpyxl, and a LibreOffice renderer."""
import pathlib, subprocess, tempfile
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.chart.label import DataLabelList
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.drawing.text import ParagraphProperties, CharacterProperties, Font as DFont

BLUE_IN = Font(name="Calibri", color="0000FF")
HEAD = PatternFill("solid", fgColor="1F4E79")
HEADF = Font(name="Calibri", bold=True, color="FFFFFF")
NOTE = Font(name="Calibri", italic=True, color="595959", size=9)
OFFICE = ["4472C4", "ED7D31", "A5A5A5", "FFC000", "5B9BD5", "70AD47", "264478", "9E480E", "636363", "997300"]


def header(ws, row, names, widths=None, start_col=1):
    for i, n in enumerate(names):
        c = ws.cell(row, start_col + i, n)
        c.fill, c.font, c.alignment = HEAD, HEADF, Alignment(horizontal="center", vertical="center", wrap_text=True)
    if widths:
        from openpyxl.utils import get_column_letter
        for i, w in enumerate(widths):
            ws.column_dimensions[get_column_letter(start_col + i)].width = w


def style(ch, title, w=24, h=13):
    ch.title = title
    ch.width, ch.height = w, h
    cp = CharacterProperties(latin=DFont(typeface="Calibri"), sz=1400, b=True, solidFill="404040")
    ch.title.tx.rich.p[0].pPr = ParagraphProperties(defRPr=cp)
    return ch


def fill(series, color, line=False, width=28575):
    series.graphicalProperties = GraphicalProperties()
    if line:
        series.graphicalProperties.line.solidFill = color
        series.graphicalProperties.line.width = width
        series.smooth = False
    else:
        series.graphicalProperties.solidFill = color
        series.graphicalProperties.line.solidFill = color


def labels(series, fmt=None, pos=None):
    series.dLbls = DataLabelList()
    series.dLbls.showVal = True
    for k in ("showSerName", "showCatName", "showLegendKey", "showPercent"):
        setattr(series.dLbls, k, False)
    if fmt:
        series.dLbls.numFmt = fmt
    if pos:
        series.dLbls.position = pos


def chart_sheet(wb, name, ch, note=None):
    ws = wb.create_sheet(name)
    ws.add_chart(ch, "B2")
    if note:
        ws["B29"] = note
        ws["B29"].font = NOTE
    ws.page_setup.orientation = "landscape"
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth = ws.page_setup.fitToHeight = 1
    ws.print_area = "A1:Q30"
    return ws


def tiles(wb, items, name="Overview"):
    """items: list of (formula, number format, label, value color)."""
    ws = wb.create_sheet(name, 0)
    ws.sheet_view.showGridLines = False
    ws.row_dimensions[3].height, ws.row_dimensions[4].height = 62, 30
    ws.column_dimensions["A"].width = 2
    from openpyxl.utils import get_column_letter
    for k, (f, fmt, lab, fg) in enumerate(items):
        col = get_column_letter(2 + 2 * k)
        ws.column_dimensions[col].width = 26
        ws.column_dimensions[get_column_letter(3 + 2 * k)].width = 2
        a, b = ws[f"{col}3"], ws[f"{col}4"]
        a.value, a.number_format = f, fmt
        a.font, a.fill, a.alignment = Font(name="Calibri", bold=True, size=34, color=fg), PatternFill("solid", fgColor="1F4E79"), Alignment(horizontal="center", vertical="center")
        b.value, b.font, b.fill, b.alignment = lab, Font(name="Calibri", size=12, color="FFFFFF"), PatternFill("solid", fgColor="1F4E79"), Alignment(horizontal="center", vertical="center")
    ws.page_setup.orientation = "landscape"
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.print_area = f"A1:{get_column_letter(1 + 2 * len(items))}5"
    return ws


def render(xlsx, pages, out_dir):
    """pages: {sheet name: png file}. Exports those sheets to cropped PNGs."""
    from openpyxl import load_workbook
    from PIL import Image, ImageChops
    tmp = pathlib.Path(tempfile.mkdtemp())
    wb = load_workbook(xlsx)
    for ws in wb.worksheets:
        ws.sheet_state = "visible" if ws.title in pages else "hidden"
    wb.active = wb.sheetnames.index(next(n for n in wb.sheetnames if n in pages))
    wb.save(tmp / "r.xlsx")
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(tmp), str(tmp / "r.xlsx")], check=True, capture_output=True, timeout=240)
    subprocess.run(["pdftoppm", "-r", "200", "-png", str(tmp / "r.pdf"), str(tmp / "p")], check=True)
    imgs = sorted(tmp.glob("p-*.png"), key=lambda p: int(p.stem.split("-")[1]))
    names = [n for n in wb.sheetnames if n in pages]
    if len(imgs) != len(names):
        raise SystemExit(f"{len(imgs)} pages for {len(names)} sheets")
    out_dir = pathlib.Path(out_dir)
    for n, p in zip(names, imgs):
        im = Image.open(p).convert("RGB")
        box = ImageChops.difference(im, Image.new("RGB", im.size, "white")).point(lambda v: 255 if v > 12 else 0).getbbox()
        im.crop((max(box[0] - 24, 0), max(box[1] - 24, 0), min(box[2] + 24, im.width), min(box[3] + 24, im.height))).save(out_dir / pages[n])
    return names
