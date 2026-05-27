# =============================================================================
# PDF EXPORT — generates professional investment report PDF
# =============================================================================

import io
import re
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Brand colors
NAVY   = colors.HexColor("#0F172A")
BLUE   = colors.HexColor("#1D4ED8")
GREEN  = colors.HexColor("#16A34A")
RED    = colors.HexColor("#DC2626")
GOLD   = colors.HexColor("#D97706")
LIGHT  = colors.HexColor("#F8F9FC")
BORDER = colors.HexColor("#DDE3EE")
WHITE  = colors.white
GRAY   = colors.HexColor("#475569")

VERDICT_COLORS = {
    "DEEP VALUE": GREEN,
    "ATTRACTIVE": BLUE,
    "FAIR":       GOLD,
    "RICH":       colors.HexColor("#EA580C"),
    "EXPENSIVE":  RED,
    "ETF":        GRAY,
}


def _fig_to_image(fig, width_cm=17, dpi=150) -> Image:
    """Convert matplotlib figure to ReportLab Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    w = width_cm * cm
    # Maintain aspect ratio
    fig_w, fig_h = fig.get_size_inches()
    aspect = fig_h / fig_w
    return Image(buf, width=w, height=w * aspect)


def _markdown_to_para(text: str, style) -> list:
    """
    Convert simple markdown (bold, bullets) to ReportLab Paragraphs.
    Handles **bold**, bullet lines starting with -, *, •
    """
    elements = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            elements.append(Spacer(1, 4))
            continue

        # Convert **text** to <b>text</b>
        formatted = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", stripped)
        # Convert *text* to <i>text</i> (only if not **bold**)
        formatted = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", formatted)

        if stripped.startswith(("- ", "* ", "• ")):
            # Bullet
            bullet_text = "• " + formatted[2:].strip()
            elements.append(Paragraph(bullet_text, style))
        elif re.match(r"^\*\*[A-Z ]+\*\*$", stripped) or re.match(r"^##", stripped):
            # Section header
            header_text = formatted.replace("##", "").strip()
            elements.append(Spacer(1, 4))
            elements.append(Paragraph(header_text, style))
        else:
            elements.append(Paragraph(formatted, style))

    return elements


def build_report_pdf(
    report_date: str,
    portfolio_stats: dict,
    risk_metrics: dict,
    risk_flags: list,
    attribution_summary: dict,
    holdings_df,
    clean_data: dict,
    verdicts: dict,
    ai_report: dict,
    figures: dict,           # {"overview": fig, "sector": fig, "attribution": fig, "risk": fig, "valuation": fig}
) -> bytes:
    """
    Assemble the full portfolio intelligence report as PDF bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Portfolio Intelligence Report",
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        fontName="Helvetica-Bold", fontSize=22, textColor=NAVY,
        spaceAfter=4, alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "SubTitle", parent=styles["Normal"],
        fontName="Helvetica", fontSize=11, textColor=GRAY,
        spaceAfter=6, alignment=TA_CENTER,
    )
    h1_style = ParagraphStyle(
        "H1", parent=styles["Heading1"],
        fontName="Helvetica-Bold", fontSize=14, textColor=NAVY,
        spaceBefore=16, spaceAfter=6, borderPad=4,
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=11, textColor=BLUE,
        spaceBefore=10, spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontName="Helvetica", fontSize=9, textColor=NAVY,
        spaceAfter=4, leading=14,
    )
    bold_style = ParagraphStyle(
        "Bold", parent=body_style,
        fontName="Helvetica-Bold", fontSize=9,
    )
    small_style = ParagraphStyle(
        "Small", parent=body_style, fontSize=8, textColor=GRAY,
    )
    verdict_note_style = ParagraphStyle(
        "VerdictNote", parent=body_style, fontSize=8.5, leading=13,
    )

    elements = []

    # =========================================================================
    # COVER PAGE
    # =========================================================================
    elements.append(Spacer(1, 3 * cm))
    elements.append(Paragraph("PORTFOLIO INTELLIGENCE REPORT", title_style))
    elements.append(Paragraph(f"Analysis Date: {report_date}", subtitle_style))
    elements.append(Spacer(1, 0.5 * cm))
    elements.append(HRFlowable(width="100%", thickness=2, color=BLUE))
    elements.append(Spacer(1, 1 * cm))

    # Key stats box
    ps = portfolio_stats
    stats_data = [
        ["Total Invested", f"₹{ps.get('total_invested', 0):,.0f}",
         "Total Value", f"₹{ps.get('total_current', 0):,.0f}"],
        ["Unrealised P&L", f"₹{ps.get('total_pnl', 0):+,.0f}",
         "Total Return", f"{ps.get('total_return', 0):+.2f}%"],
        ["Holdings", f"{ps.get('num_holdings', 0)}",
         "Winners / Losers", f"{ps.get('num_winners', 0)} / {ps.get('num_losers', 0)}"],
        ["Best Performer", f"{ps.get('best_performer', '—')[:20]}  {ps.get('best_return', 0):+.1f}%",
         "Active Share", f"{risk_metrics.get('active_share', 0):.1f}%"],
    ]
    stats_table = Table(stats_data, colWidths=[4.5 * cm, 5 * cm, 4.5 * cm, 4 * cm])
    stats_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), NAVY),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, LIGHT]),
    ]))
    elements.append(stats_table)
    elements.append(Spacer(1, 0.5 * cm))
    elements.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    elements.append(Spacer(1, 1 * cm))

    # Table of contents
    toc_items = [
        "1.  Executive Summary",
        "2.  Portfolio Holdings & Performance",
        "3.  Sector Allocation & Attribution",
        "4.  Risk Analytics",
        "5.  Valuation & Peer Analysis",
        "6.  Individual Investment Theses",
    ]
    elements.append(Paragraph("Contents", h2_style))
    for item in toc_items:
        elements.append(Paragraph(item, body_style))
    elements.append(PageBreak())

    # =========================================================================
    # SECTION 1: EXECUTIVE SUMMARY
    # =========================================================================
    elements.append(Paragraph("1. Executive Summary", h1_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=8))

    if ai_report.get("summary"):
        for para in _markdown_to_para(ai_report["summary"], body_style):
            elements.append(para)
    else:
        elements.append(Paragraph("Executive summary not generated.", small_style))

    elements.append(PageBreak())

    # =========================================================================
    # SECTION 2: HOLDINGS TABLE
    # =========================================================================
    elements.append(Paragraph("2. Portfolio Holdings & Performance", h1_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=8))

    h_data = [["#", "Company", "Sector", "Buy ₹", "CMP ₹", "Invested", "Value", "P&L", "Ret%", "Wt%"]]
    for i, row in holdings_df.iterrows():
        h_data.append([
            str(i),
            row["Company"][:22],
            row["Sector"][:16],
            f"{row['Avg Buy Price']:.2f}",
            f"{row['Current Price']:.2f}" if row.get("Current Price") else "—",
            f"₹{row['Invested Value']:,.0f}",
            f"₹{row['Current Value']:,.0f}",
            f"₹{row['P&L']:+,.0f}",
            f"{row['Return %']:+.1f}%",
            f"{row['Portfolio Wt%']:.1f}%",
        ])

    h_table = Table(h_data, colWidths=[0.7*cm, 4.2*cm, 2.8*cm, 1.5*cm, 1.5*cm, 2*cm, 2*cm, 2*cm, 1.5*cm, 1.3*cm])
    ts = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
        ("ALIGN", (3, 0), (-1, -1), "RIGHT"),
    ])
    # Color P&L column
    for i, row in holdings_df.iterrows():
        color = GREEN if row["P&L"] >= 0 else RED
        ts.add("TEXTCOLOR", (7, i), (8, i), color)
    h_table.setStyle(ts)
    elements.append(h_table)

    # Sector overview figure
    if "overview" in figures:
        elements.append(Spacer(1, 0.5 * cm))
        elements.append(_fig_to_image(figures["overview"], width_cm=17))

    elements.append(PageBreak())

    # =========================================================================
    # SECTION 3: SECTOR ALLOCATION & ATTRIBUTION
    # =========================================================================
    elements.append(Paragraph("3. Sector Allocation & Attribution", h1_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=8))

    if "sector" in figures:
        elements.append(_fig_to_image(figures["sector"], width_cm=17))
        elements.append(Spacer(1, 0.5 * cm))

    # Attribution summary
    if attribution_summary:
        elements.append(Paragraph("Return Attribution (BHB Model)", h2_style))
        attr_data = [
            ["Portfolio Return", f"{attribution_summary.get('total_port_return', 0):+.2f}%"],
            ["Benchmark Return", f"{attribution_summary.get('total_bench_return', 0):+.2f}%"],
            ["Active Alpha", f"{attribution_summary.get('total_alpha', 0):+.2f}%"],
            ["  Allocation Effect", f"{attribution_summary.get('allocation_effect', 0):+.4f}%"],
            ["  Selection Effect", f"{attribution_summary.get('selection_effect', 0):+.4f}%"],
            ["  Interaction Effect", f"{attribution_summary.get('interaction_effect', 0):+.4f}%"],
        ]
        attr_table = Table(attr_data, colWidths=[8 * cm, 4 * cm])
        attr_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 2), (0, 2), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, LIGHT]),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(attr_table)
        elements.append(Spacer(1, 0.3 * cm))

    if "attribution" in figures:
        elements.append(_fig_to_image(figures["attribution"], width_cm=17))

    elements.append(PageBreak())

    # =========================================================================
    # SECTION 4: RISK ANALYTICS
    # =========================================================================
    elements.append(Paragraph("4. Risk Analytics", h1_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=8))

    # Risk metrics table
    rm = risk_metrics
    risk_data = [
        ["Metric", "Value", "Interpretation"],
        ["Active Share", f"{rm.get('active_share', 0):.1f}%", "> 60% = genuinely active"],
        ["HHI (Concentration)", f"{rm.get('hhi', 0):.0f}", "< 1000 = diversified"],
        ["Effective Holdings", f"{rm.get('effective_n', 0):.1f}", "Equivalent equal-weight N"],
        ["Effective Sectors", f"{rm.get('effective_sectors', 0):.1f}", "Sector diversification"],
        ["Top 5 Weight", f"{rm.get('top5_wt', 0):.1f}%", "< 60% = healthy"],
        ["Hit Rate", f"{rm.get('hit_rate', 0):.1f}%", "% profitable positions"],
        ["Win/Loss Ratio", f"{rm.get('win_loss_ratio', 0):.2f}x", "> 1.5x = good"],
    ]
    risk_table = Table(risk_data, colWidths=[5 * cm, 3 * cm, 9 * cm])
    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(risk_table)
    elements.append(Spacer(1, 0.5 * cm))

    # Risk flags
    elements.append(Paragraph("Risk Flags", h2_style))
    for flag in risk_flags:
        elements.append(Paragraph(f"  {flag['level']}", body_style))

    if "risk" in figures:
        elements.append(Spacer(1, 0.3 * cm))
        elements.append(_fig_to_image(figures["risk"], width_cm=17))

    elements.append(PageBreak())

    # =========================================================================
    # SECTION 5: VALUATION & PEER ANALYSIS
    # =========================================================================
    elements.append(Paragraph("5. Valuation & Peer Analysis", h1_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=8))

    if "valuation" in figures:
        elements.append(_fig_to_image(figures["valuation"], width_cm=17))
        elements.append(Spacer(1, 0.5 * cm))

    # Valuation summary table
    val_data = [["Company", "Sector", "PE", "PB", "EV/EBITDA", "ROCE%", "Score", "Verdict"]]
    for _, row in holdings_df.iterrows():
        ticker = row["Ticker"]
        d = clean_data.get(ticker) or {}
        v = verdicts.get(ticker) or {}
        val_data.append([
            row["Company"][:18],
            row["Sector"][:14],
            f"{d.get('pe'):.1f}" if d.get("pe") else "—",
            f"{d.get('pb'):.2f}" if d.get("pb") else "—",
            f"{d.get('ev_ebitda'):.1f}" if d.get("ev_ebitda") else "—",
            f"{d.get('roce'):.1f}" if d.get("roce") else "—",
            f"{v.get('score'):.0f}" if v.get("score") else "—",
            v.get("label", "—"),
        ])

    val_table = Table(val_data, colWidths=[3.5*cm, 2.8*cm, 1.5*cm, 1.5*cm, 2.2*cm, 1.7*cm, 1.5*cm, 2.8*cm])
    val_ts = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (2, 0), (-1, -1), "CENTER"),
    ])
    # Color verdict column
    for i, row in enumerate(holdings_df.itertuples(), start=1):
        v = verdicts.get(row.Ticker) or {}
        lbl = v.get("label", "")
        color = VERDICT_COLORS.get(lbl, NAVY)
        val_ts.add("TEXTCOLOR", (7, i), (7, i), color)
        val_ts.add("FONTNAME", (7, i), (7, i), "Helvetica-Bold")
    val_table.setStyle(val_ts)
    elements.append(val_table)

    elements.append(PageBreak())

    # =========================================================================
    # SECTION 6: INDIVIDUAL INVESTMENT THESES
    # =========================================================================
    elements.append(Paragraph("6. Individual Investment Theses", h1_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=8))

    for _, row in holdings_df.iterrows():
        ticker = row["Ticker"]
        company = row["Company"]
        d = clean_data.get(ticker) or {}
        v = verdicts.get(ticker) or {}
        thesis = ai_report.get("theses", {}).get(ticker, "")

        # Holding header card
        verdict_label = v.get("label", "—")
        verdict_color = VERDICT_COLORS.get(verdict_label, GRAY)
        score_str = f"{v.get('score'):.0f}/100" if v.get("score") else "N/A"

        card_data = [
            [
                Paragraph(f"<b>{company}</b>", ParagraphStyle("ch", fontName="Helvetica-Bold", fontSize=11, textColor=WHITE)),
                Paragraph(f"{ticker}  |  {row['Sector']}", ParagraphStyle("cs", fontName="Helvetica", fontSize=8, textColor=colors.HexColor("#BFDBFE"))),
                Paragraph(f"{verdict_label}  {score_str}", ParagraphStyle("cv", fontName="Helvetica-Bold", fontSize=10, textColor=WHITE, alignment=TA_RIGHT)),
            ]
        ]
        card_table = Table(card_data, colWidths=[6.5 * cm, 6.5 * cm, 5 * cm])
        card_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), NAVY),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elements.append(KeepTogether([card_table]))

        # Key metrics row under the header
        metrics_row = []
        for label, val in [
            ("Weight", f"{row['Portfolio Wt%']:.1f}%"),
            ("Return", f"{row['Return %']:+.1f}%"),
            ("PE", f"{d.get('pe'):.1f}x" if d.get('pe') else "—"),
            ("PB", f"{d.get('pb'):.2f}x" if d.get('pb') else "—"),
            ("ROCE", f"{d.get('roce'):.1f}%" if d.get('roce') else "—"),
            ("ROE", f"{d.get('roe'):.1f}%" if d.get('roe') else "—"),
        ]:
            metrics_row.append([
                Paragraph(label, ParagraphStyle("ml", fontName="Helvetica", fontSize=7.5, textColor=GRAY, alignment=TA_CENTER)),
                Paragraph(val, ParagraphStyle("mv", fontName="Helvetica-Bold", fontSize=9, textColor=NAVY, alignment=TA_CENTER)),
            ])

        m_table = Table(metrics_row, colWidths=[3 * cm] * 6)
        m_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(m_table)

        # Investment flags
        flags = v.get("flags", [])
        if flags:
            elements.append(Spacer(1, 4))
            flags_text = "  |  ".join(flags[:4])
            elements.append(Paragraph(
                f"<i><font color='#475569' size='8'>Signals: {flags_text}</font></i>",
                ParagraphStyle("flags", fontSize=8, leading=11)
            ))

        # Thesis text
        elements.append(Spacer(1, 6))
        if thesis:
            for para in _markdown_to_para(thesis, verdict_note_style):
                elements.append(para)
        else:
            elements.append(Paragraph(
                "<i>Investment thesis not generated (upload Screener file or provide API key)</i>",
                small_style,
            ))

        elements.append(Spacer(1, 0.5 * cm))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
        elements.append(Spacer(1, 0.4 * cm))

    # Footer
    elements.append(Spacer(1, 1 * cm))
    elements.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(
        f"Generated by Portfolio Intelligence Platform  |  {datetime.now().strftime('%d %b %Y %H:%M')}  |  For internal use only",
        ParagraphStyle("footer", fontName="Helvetica", fontSize=7.5, textColor=GRAY, alignment=TA_CENTER),
    ))

    doc.build(elements)
    return buf.getvalue()
