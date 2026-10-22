"""HTML and PDF report generation.

Reports are built fully in memory — no temporary files — and returned as
bytes for the API to stream to the browser. Charts come from the same curated
specs shown in the UI, rendered with Matplotlib at export time (at most
``settings.report_chart_limit`` images in the PDF).

The old module was dead code (imported nowhere) and duplicated its imports;
this one is wired into the Insights page and covered by tests.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from html import escape
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.formatting import format_metric, format_number, format_pct

INK = "#0f172a"
MUTED = "#5b6b82"
BRAND = "#4f46e5"
BORDER = "#e2e8f0"
WARN = "#b45309"

PDF_SAFE = {
    "\u2194": "-",
    "\u2264": "<=",
    "\u2265": ">=",
    "\u00d7": "x",
    "\u2013": "-",
    "\u2014": "-",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00b7": "-",
    "\u00b2": "2",
}


def _pdf_text(value) -> str:
    text = str(value)
    for source, replacement in PDF_SAFE.items():
        text = text.replace(source, replacement)
    return text


def _slug(name: str) -> str:
    safe = "".join(character if character.isalnum() else "-" for character in str(name).lower())
    return "-".join(part for part in safe.split("-") if part)[:48] or "dataset"


# ---------------------------------------------------------------------------
# Shared preparation
# ---------------------------------------------------------------------------


def _dataset_meta(context: dict) -> dict:
    dataset = context.get("dataset") or {}
    profile = context.get("profile") or {}
    overview = profile.get("overview") or {}
    return {
        "name": dataset.get("name") or "Dataset",
        "source": dataset.get("source") or "uploaded file",
        "rows": overview.get("rows", dataset.get("rows", 0)),
        "columns": overview.get("columns", dataset.get("columns", 0)),
        "missing_pct": overview.get("missing_pct", 0),
        "missing_cells": overview.get("missing_cells", 0),
        "duplicate_rows": overview.get("duplicate_rows", 0),
        "revision": dataset.get("revision", 1),
        "memory_mb": overview.get("memory_mb"),
        "generated_at": context.get("generated_at")
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "truncated": overview.get("truncated") or dataset.get("truncated"),
    }


def _metrics_table_rows(ml: dict) -> list[tuple[str, str]]:
    metrics = ml.get("metrics") or {}
    baseline = ml.get("baseline") or {}
    rows = []
    if ml.get("task") == "classification":
        rows.append(("Accuracy", format_pct((metrics.get("accuracy") or 0) * 100)))
        rows.append(("Balanced accuracy", format_pct((metrics.get("balanced_accuracy") or 0) * 100)))
        rows.append(("Macro F1", format_pct((metrics.get("f1_macro") or 0) * 100)))
        if baseline.get("accuracy") is not None:
            rows.append(
                (
                    "Majority-class baseline",
                    format_pct(baseline["accuracy"] * 100),
                )
            )
        if metrics.get("roc_auc") is not None:
            rows.append(("ROC AUC", format_metric(metrics["roc_auc"], 3)))
    else:
        rows.append(("R2", format_metric(metrics.get("r2"), 3)))
        rows.append(("MAE", format_number(metrics.get("mae"), 2)))
        rows.append(("RMSE", format_number(metrics.get("rmse"), 2)))
        if metrics.get("mape") is not None:
            rows.append(("MAPE", format_pct(metrics["mape"])))
        if baseline.get("mae") is not None:
            rows.append(("Baseline MAE (mean)", format_number(baseline["mae"], 2)))
    return rows


def _chart_images(context: dict, limit: int) -> list[tuple[dict, bytes]]:
    specs = context.get("charts") or []
    if not specs:
        return []

    from src.visualizer import render_spec

    images = []
    for spec in specs[:limit]:
        figure = render_spec(spec)
        try:
            buffer = BytesIO()
            figure.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
            images.append((spec, buffer.getvalue()))
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)
    return images


def _quality_rows(profile: dict) -> list[dict]:
    """Quality issues computed by the profiler, ready to render."""
    return list(profile.get("quality") or [])


def _column_summary(column: dict) -> str:
    kind = column.get("kind")
    stats = column.get("stats") or {}
    if kind == "numeric" and stats:
        return (
            f"mean {format_number(stats.get('mean'))} - "
            f"range {format_number(stats.get('min'))} to {format_number(stats.get('max'))}"
        )
    if kind == "datetime" and stats:
        return f"{stats.get('min', '')} to {stats.get('max', '')}"
    top = column.get("top_values") or []
    if top:
        return f"most common: {top[0]['value']} ({top[0]['pct']}%)"
    return "-"


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


HTML_STYLE = f"""
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  padding: 32px 20px 56px;
  background: #f6f7fb;
  color: {INK};
  font: 15px/1.6 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}}
.page {{ max-width: 940px; margin: 0 auto; }}
header {{
  background: linear-gradient(135deg, #111a2e 0%, #1e2a4a 55%, #312e81 100%);
  color: #fff;
  border-radius: 18px;
  padding: 34px 36px;
}}
header .eyebrow {{ text-transform: uppercase; letter-spacing: .14em; font-size: 11px; opacity: .7; margin: 0 0 10px; }}
header h1 {{ margin: 0 0 6px; font-size: 30px; line-height: 1.2; }}
header p {{ margin: 4px 0 0; color: #c7d2fe; font-size: 13px; }}
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 18px 0 26px; }}
.kpi {{ background: #fff; border: 1px solid {BORDER}; border-radius: 14px; padding: 14px 16px; }}
.kpi span {{ display: block; font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: {MUTED}; margin-bottom: 6px; }}
.kpi strong {{ font-size: 22px; font-variant-numeric: tabular-nums; }}
section {{ background: #fff; border: 1px solid {BORDER}; border-radius: 16px; padding: 24px 26px; margin-bottom: 18px; }}
h2 {{ font-size: 17px; margin: 0 0 14px; }}
h3 {{ font-size: 14px; margin: 18px 0 8px; }}
p {{ margin: 0 0 10px; }}
ul {{ margin: 0 0 10px; padding-left: 20px; }}
li {{ margin-bottom: 6px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ text-align: left; font-size: 11px; letter-spacing: .05em; text-transform: uppercase; color: {MUTED}; border-bottom: 1px solid {BORDER}; padding: 8px 10px; }}
td {{ border-bottom: 1px solid #eef2f7; padding: 8px 10px; vertical-align: top; }}
tr:last-child td {{ border-bottom: none; }}
.finding {{ border-left: 3px solid {BRAND}; padding: 2px 0 2px 14px; margin: 0 0 14px; }}
.finding.warning {{ border-color: {WARN}; }}
.finding strong {{ display: block; }}
.finding span {{ color: {MUTED}; font-size: 13px; }}
.chart {{ margin: 0 0 24px; }}
.chart img {{ width: 100%; height: auto; border: 1px solid {BORDER}; border-radius: 12px; }}
.chart figcaption {{ color: {MUTED}; font-size: 13px; margin-top: 8px; }}
.muted {{ color: {MUTED}; font-size: 13px; }}
.pill {{ display: inline-block; padding: 2px 9px; border-radius: 999px; background: #eef2ff; color: {BRAND}; font-size: 11px; font-weight: 600; }}
.warn {{ color: {WARN}; }}
footer {{ color: {MUTED}; font-size: 12px; text-align: center; margin-top: 26px; }}
@media print {{
  body {{ background: #fff; padding: 0; }}
  header {{ border-radius: 0; }}
  section, .chart img {{ break-inside: avoid; }}
  .kpi {{ break-inside: avoid; }}
}}
"""


def build_html(context: dict, *, include_charts: bool = True, chart_limit: int = 4) -> str:
    meta = _dataset_meta(context)
    profile = context.get("profile") or {}
    insights = context.get("insights") or {}
    ml = context.get("ml")
    cleaning = context.get("cleaning")

    parts: list[str] = []
    parts.append(
        f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(meta['name'])} — Data report</title>
<style>{HTML_STYLE}</style></head>
<body><div class="page">
<header>
  <p class="eyebrow">AI Data Visualizer · Report</p>
  <h1>{escape(meta['name'])}</h1>
  <p>{escape(str(meta['source']))} · revision {meta['revision']} ·
     generated {escape(str(meta['generated_at']))}</p>
</header>"""
    )

    kpis = [
        ("Rows", f"{meta['rows']:,}"),
        ("Columns", f"{meta['columns']:,}"),
        ("Missing", format_pct(meta["missing_pct"])),
        ("Duplicate rows", f"{meta['duplicate_rows']:,}"),
        ("Memory", f"{meta['memory_mb']} MB" if meta["memory_mb"] else "-"),
    ]
    parts.append('<div class="kpis">')
    for label, value in kpis:
        parts.append(f'<div class="kpi"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>')
    parts.append("</div>")

    summary = insights.get("summary")
    findings = insights.get("findings") or []
    recommendations = insights.get("recommendations") or []

    parts.append("<section><h2>Executive summary</h2>")
    if summary:
        parts.append(f"<p>{escape(summary)}</p>")
    if findings:
        for finding in findings:
            severity = " warning" if finding.get("severity") == "warning" else ""
            parts.append(
                f'<div class="finding{severity}"><strong>{escape(str(finding.get("title", "")))}</strong>'
                f'<span>{escape(str(finding.get("detail", "")))}</span></div>'
            )
    else:
        parts.append('<p class="muted">No notable findings were recorded.</p>')
    parts.append("</section>")

    overview = profile.get("overview") or {}
    quality = _quality_rows(profile)
    parts.append("<section><h2>Data quality</h2>")
    parts.append(
        '<p class="muted">'
        f"{int(overview.get('missing_cells', 0)):,} missing cells "
        f"({format_pct(overview.get('missing_pct', 0))}) across "
        f"{int(overview.get('cells', 0)):,} cells · "
        f"{int(overview.get('duplicate_rows', 0)):,} duplicate rows · "
        f"{int(overview.get('identifier_columns', 0)):,} identifier column(s) · "
        f"{int(overview.get('constant_columns', 0)):,} constant column(s)"
        "</p>"
    )
    if quality:
        parts.append("<ul>")
        for issue in quality:
            parts.append(
                f"<li><strong>{escape(str(issue.get('title', '')))}</strong> — "
                f"{escape(str(issue.get('detail', '')))}</li>"
            )
        parts.append("</ul>")
    else:
        parts.append('<p class="muted">No quality issues were detected.</p>')
    parts.append("</section>")

    if include_charts:
        images = _chart_images(context, chart_limit)
        if images:
            parts.append("<section><h2>Key charts</h2>")
            for spec, png in images:
                encoded = base64.b64encode(png).decode("ascii")
                parts.append(
                    f'<figure class="chart"><img alt="{escape(spec.get("title", ""))}" '
                    f'src="data:image/png;base64,{encoded}">'
                    f'<figcaption><strong>{escape(spec.get("title", ""))}.</strong> '
                    f'{escape(spec.get("insight", ""))}</figcaption></figure>'
                )
            parts.append("</section>")

    parts.append("<section><h2>Columns</h2><table><thead><tr>"
                 "<th>Column</th><th>Type</th><th>Missing</th><th>Distinct</th><th>Summary</th>"
                 "</tr></thead><tbody>")
    for column in profile.get("columns") or []:
        parts.append(
            "<tr>"
            f"<td>{escape(str(column.get('name')))}</td>"
            f"<td>{escape(str(column.get('kind')))}</td>"
            f"<td>{escape(format_pct(column.get('missing_pct', 0)))}</td>"
            f"<td>{int(column.get('unique', 0)):,}</td>"
            f"<td>{escape(_column_summary(column))}</td>"
            "</tr>"
        )
    parts.append("</tbody></table></section>")

    if cleaning:
        parts.append("<section><h2>Cleaning applied</h2>")
        actions = cleaning.get("actions") or []
        if actions:
            parts.append("<ul>")
            for action in actions:
                parts.append(f"<li>{escape(str(action))}</li>")
            parts.append("</ul>")
        parts.append(
            '<p class="muted">'
            f"Rows {cleaning.get('rows_before', 0):,} → {cleaning.get('rows_after', 0):,} · "
            f"missing cells {cleaning.get('missing_before', 0):,} → {cleaning.get('missing_after', 0):,}"
            "</p></section>"
        )

    if ml:
        parts.append("<section><h2>Machine learning</h2>")
        parts.append(
            f"<p><span class='pill'>{escape(str(ml.get('task')))}</span> "
            f"Predicting <strong>{escape(str(ml.get('target')))}</strong> with "
            f"{escape(str(ml.get('model_label') or ml.get('model_id')))}.</p>"
        )
        review = (insights.get("model") or {}).get("headline")
        if review:
            parts.append(f"<p>{escape(str(review))}</p>")

        parts.append("<h3>Metrics</h3><table><tbody>")
        for label, value in _metrics_table_rows(ml):
            parts.append(f"<tr><td>{escape(label)}</td><td>{escape(value)}</td></tr>")
        parts.append("</tbody></table>")

        per_class = ml.get("per_class") or []
        if per_class:
            parts.append("<h3>Per class</h3><table><thead><tr><th>Class</th>"
                         "<th>Precision</th><th>Recall</th><th>F1</th><th>Support</th>"
                         "</tr></thead><tbody>")
            for row in per_class:
                parts.append(
                    "<tr>"
                    f"<td>{escape(str(row.get('label')))}</td>"
                    f"<td>{format_metric(row.get('precision'), 3)}</td>"
                    f"<td>{format_metric(row.get('recall'), 3)}</td>"
                    f"<td>{format_metric(row.get('f1'), 3)}</td>"
                    f"<td>{int(row.get('support', 0)):,}</td>"
                    "</tr>"
                )
            parts.append("</tbody></table>")

        importance = ml.get("importance") or []
        if importance:
            parts.append("<h3>Feature importance</h3><table><thead><tr><th>Feature</th>"
                         "<th>Importance</th></tr></thead><tbody>")
            for row in importance[:10]:
                parts.append(
                    f"<tr><td>{escape(str(row.get('feature')))}</td>"
                    f"<td>{format_pct((row.get('importance') or 0) * 100)}</td></tr>"
                )
            parts.append("</tbody></table>")

        warnings = ml.get("warnings") or []
        if warnings:
            parts.append('<h3>Notes</h3><ul class="warn">')
            for warning in warnings:
                parts.append(f"<li>{escape(str(warning))}</li>")
            parts.append("</ul>")
        parts.append("</section>")

    if recommendations:
        parts.append("<section><h2>Recommended next steps</h2><ul>")
        for item in recommendations:
            detail = item.get("detail") if isinstance(item, dict) else str(item)
            parts.append(f"<li>{escape(str(detail))}</li>")
        parts.append("</ul></section>")

    parts.append(
        "<footer>Generated by AI Data Visualizer from computed statistics. "
        f"No values are estimated or fabricated. · {escape(str(meta['generated_at']))}</footer>"
        "</div></body></html>"
    )
    return "".join(parts)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def _pdf_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor(INK),
            spaceAfter=2,
            alignment=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor(INK),
            spaceBefore=14,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["Normal"],
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#1f2937"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor(MUTED),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Cell",
            parent=styles["Normal"],
            fontSize=8,
            leading=10.5,
        )
    )
    return styles


def _pdf_table(rows: list[list], styles, header: bool = True, widths=None) -> Table:
    table = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")))
    table.setStyle(TableStyle(style))
    return table


def build_pdf(context: dict, *, include_charts: bool = True, chart_limit: int = 4) -> bytes:
    meta = _dataset_meta(context)
    profile = context.get("profile") or {}
    insights = context.get("insights") or {}
    ml = context.get("ml")
    cleaning = context.get("cleaning")
    styles = _pdf_styles()

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"{meta['name']} report",
        author="AI Data Visualizer",
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(18 * mm, 9 * mm, "AI Data Visualizer · generated from computed statistics")
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()

    story = []
    story.append(Paragraph(_pdf_text(meta["name"]), styles["ReportTitle"]))
    story.append(
        Paragraph(
            _pdf_text(
                f"Data report · revision {meta['revision']} · "
                f"{meta['rows']:,} rows × {meta['columns']:,} columns · "
                f"generated {meta['generated_at']}"
            ),
            styles["Small"],
        )
    )
    story.append(Spacer(1, 10))

    kpi_row = [
        ["Rows", "Columns", "Missing", "Duplicate rows"],
        [
            f"{meta['rows']:,}",
            f"{meta['columns']:,}",
            format_pct(meta["missing_pct"]),
            f"{meta['duplicate_rows']:,}",
        ],
    ]
    story.append(_pdf_table(kpi_row, styles))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Executive summary", styles["SectionHeading"]))
    if insights.get("summary"):
        story.append(Paragraph(_pdf_text(insights["summary"]), styles["Body"]))
        story.append(Spacer(1, 4))

    for finding in (insights.get("findings") or [])[:6]:
        story.append(
            Paragraph(
                f"<b>{_pdf_text(finding.get('title', ''))}</b> — "
                f"{_pdf_text(finding.get('detail', ''))}",
                styles["Body"],
            )
        )
        story.append(Spacer(1, 3))

    overview = profile.get("overview") or {}
    quality = _quality_rows(profile)
    story.append(Paragraph("Data quality", styles["SectionHeading"]))
    story.append(
        Paragraph(
            _pdf_text(
                f"{int(overview.get('missing_cells', 0)):,} missing cells "
                f"({format_pct(overview.get('missing_pct', 0))}) across "
                f"{int(overview.get('cells', 0)):,} cells; "
                f"{int(overview.get('duplicate_rows', 0)):,} duplicate rows; "
                f"{int(overview.get('constant_columns', 0)):,} constant column(s)."
            ),
            styles["Body"],
        )
    )
    for issue in quality:
        story.append(
            Paragraph(
                f"<b>{_pdf_text(issue.get('title', ''))}</b> — "
                f"{_pdf_text(issue.get('detail', ''))}",
                styles["Small"],
            )
        )
    if not quality:
        story.append(Paragraph("No quality issues were detected.", styles["Small"]))
    story.append(Spacer(1, 6))

    columns = profile.get("columns") or []
    if columns:
        story.append(Paragraph("Columns", styles["SectionHeading"]))
        rows = [["Column", "Type", "Missing", "Distinct", "Summary"]]
        for column in columns[:60]:
            rows.append(
                [
                    Paragraph(_pdf_text(column.get("name")), styles["Cell"]),
                    Paragraph(_pdf_text(column.get("kind")), styles["Cell"]),
                    Paragraph(format_pct(column.get("missing_pct", 0)), styles["Cell"]),
                    Paragraph(f"{int(column.get('unique', 0)):,}", styles["Cell"]),
                    Paragraph(_pdf_text(_column_summary(column)), styles["Cell"]),
                ]
            )
        story.append(
            _pdf_table(rows, styles, widths=[80, 50, 42, 40, 200])
        )

    if include_charts:
        images = _chart_images(context, chart_limit)
        if images:
            story.append(PageBreak())
            story.append(Paragraph("Key charts", styles["SectionHeading"]))
            for spec, png in images:
                story.append(
                    Paragraph(_pdf_text(spec.get("title", "")), styles["Body"])
                )
                story.append(
                    Image(BytesIO(png), width=165 * mm, height=98 * mm)
                )
                story.append(
                    Paragraph(_pdf_text(spec.get("insight", "")), styles["Small"])
                )
                story.append(Spacer(1, 8))

    if cleaning:
        story.append(Paragraph("Cleaning applied", styles["SectionHeading"]))
        for action in cleaning.get("actions") or []:
            story.append(Paragraph(f"• {_pdf_text(action)}", styles["Body"]))
        story.append(
            Paragraph(
                _pdf_text(
                    f"Rows {cleaning.get('rows_before', 0):,} to "
                    f"{cleaning.get('rows_after', 0):,}; missing cells "
                    f"{cleaning.get('missing_before', 0):,} to "
                    f"{cleaning.get('missing_after', 0):,}."
                ),
                styles["Small"],
            )
        )

    if ml:
        story.append(Paragraph("Machine learning", styles["SectionHeading"]))
        story.append(
            Paragraph(
                _pdf_text(
                    f"Task: {ml.get('task')} · target: {ml.get('target')} · "
                    f"model: {ml.get('model_label') or ml.get('model_id')}"
                ),
                styles["Body"],
            )
        )
        review = (insights.get("model") or {}).get("headline")
        if review:
            story.append(Paragraph(_pdf_text(review), styles["Small"]))
            story.append(Spacer(1, 4))

        rows = [["Metric", "Value"]]
        for label, value in _metrics_table_rows(ml):
            rows.append([Paragraph(_pdf_text(label), styles["Cell"]), Paragraph(_pdf_text(value), styles["Cell"])])
        story.append(_pdf_table(rows, styles, widths=[120, 120]))

        per_class = ml.get("per_class") or []
        if per_class:
            rows = [["Class", "Precision", "Recall", "F1", "Support"]]
            for row in per_class:
                rows.append(
                    [
                        Paragraph(_pdf_text(row.get("label")), styles["Cell"]),
                        format_metric(row.get("precision"), 3),
                        format_metric(row.get("recall"), 3),
                        format_metric(row.get("f1"), 3),
                        f"{int(row.get('support', 0)):,}",
                    ]
                )
            story.append(Spacer(1, 6))
            story.append(_pdf_table(rows, styles, widths=[80, 70, 70, 70, 60]))

        importance = ml.get("importance") or []
        if importance:
            story.append(Spacer(1, 6))
            rows = [["Feature", "Importance"]]
            for row in importance[:10]:
                rows.append(
                    [
                        Paragraph(_pdf_text(row.get("feature")), styles["Cell"]),
                        Paragraph(format_pct((row.get("importance") or 0) * 100), styles["Cell"]),
                    ]
                )
            story.append(_pdf_table(rows, styles, widths=[140, 100]))

        warnings = ml.get("warnings") or []
        if warnings:
            story.append(Spacer(1, 6))
            for warning in warnings:
                story.append(Paragraph(f"• {_pdf_text(warning)}", styles["Small"]))

    recommendations = insights.get("recommendations") or []
    if recommendations:
        story.append(Paragraph("Recommended next steps", styles["SectionHeading"]))
        for item in recommendations:
            detail = item.get("detail") if isinstance(item, dict) else str(item)
            story.append(Paragraph(f"• {_pdf_text(detail)}", styles["Body"]))

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Entry point used by the API
# ---------------------------------------------------------------------------


def render_report(
    context: dict,
    fmt: str = "html",
    *,
    include_charts: bool = True,
    chart_limit: int = 4,
) -> tuple[bytes, str, str]:
    """Return ``(payload, mimetype, filename)`` for the requested format."""
    meta = _dataset_meta(context)
    slug = _slug(meta["name"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")

    if fmt == "pdf":
        payload = build_pdf(context, include_charts=include_charts, chart_limit=chart_limit)
        return payload, "application/pdf", f"{slug}-report-{stamp}.pdf"

    if fmt == "html":
        payload = build_html(context, include_charts=include_charts, chart_limit=chart_limit)
        return payload.encode("utf-8"), "text/html; charset=utf-8", f"{slug}-report-{stamp}.html"

    raise ValueError("Report format must be 'html' or 'pdf'.")


def write_html(context: dict, output_path: str | Path, **kwargs) -> str:
    """Write an HTML report to disk (CLI/tests)."""
    payload, _mimetype, _name = render_report(context, "html", **kwargs)
    path = Path(output_path).with_suffix(".html")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return str(path)


def write_pdf(context: dict, output_path: str | Path, **kwargs) -> str:
    """Write a PDF report to disk (CLI/tests)."""
    payload, _mimetype, _name = render_report(context, "pdf", **kwargs)
    path = Path(output_path).with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return str(path)
