from __future__ import annotations
import io
import textwrap
from fpdf import FPDF
from app.reports.composer import ReportData


def _safe(text: str) -> str:
    """Strip characters outside latin-1 range (built-in PDF fonts only support latin-1)."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


class _PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Survey Analytics Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title: str) -> None:
        self.ln(4)
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, _safe(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 0, 0)
        self.line(self.get_x(), self.get_y(), self.get_x() + self.epw, self.get_y())
        self.ln(2)

    def body_text(self, text: str) -> None:
        self.set_font("Helvetica", "", 10)
        wrapped = textwrap.fill(_safe(text), width=90)
        self.multi_cell(0, 6, wrapped)
        self.ln(2)


def render_pdf(report: ReportData, narrative: str) -> bytes:
    pdf = _PDF()
    pdf.add_page()

    # Dataset summary
    pdf.section_title("Dataset Overview")
    pdf.body_text(
        f"File: {report.filename}  |  Rows: {report.row_count}  |  Columns: {report.col_count}"
    )
    pdf.body_text(narrative)

    # Column table
    pdf.section_title("Column Summary")
    pdf.set_font("Helvetica", "B", 9)
    col_w = [50, 25, 25, 25, 65]
    headers = ["Column", "Type", "Missing %", "Unique", "Top Values / Mean"]
    for w, h in zip(col_w, headers):
        pdf.cell(w, 7, _safe(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for col in report.column_summary:
        top = str(col.get("mean") or col.get("top_values") or "")[:30]
        row_vals = [col["name"][:20], col["dtype"], str(col["missing_pct"]), str(col["n_unique"]), top]
        for w, v in zip(col_w, row_vals):
            pdf.cell(w, 6, _safe(v), border=1)
        pdf.ln()

    # Key insights
    if report.insights:
        pdf.section_title("Key Findings")
        for ins in report.insights:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, _safe(f"{ins['rank']}. {ins['title']}"), new_x="LMARGIN", new_y="NEXT")
            pdf.body_text(ins["summary"])

    # Pinned charts
    if report.pinned_charts:
        pdf.section_title("Charts")
        for chart in report.pinned_charts:
            png_bytes = chart.get("png_bytes")
            if not png_bytes:
                continue
            title = chart.get("title", "Chart")
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 6, _safe(title), new_x="LMARGIN", new_y="NEXT")
            img_buf = io.BytesIO(png_bytes)
            # Place image; max width = page width - margins
            try:
                pdf.image(img_buf, w=min(170, pdf.epw))
            except Exception:
                pdf.body_text("[Chart could not be rendered]")
            pdf.ln(4)

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
