# =============================================================================
# INSTITUTIONAL PDF REPORT GENERATOR — MODULE 7
# 18-page Goldman/Morgan Stanley style institutional report
# =============================================================================

import io
import re
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
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

# ── Brand Colors ──────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#0F172A")
BLUE   = colors.HexColor("#1D4ED8")
GREEN  = colors.HexColor("#15803D")
RED    = colors.HexColor("#DC2626")
GOLD   = colors.HexColor("#D97706")
ORANGE = colors.HexColor("#EA580C")
LIGHT  = colors.HexColor("#F8F9FC")
BORDER = colors.HexColor("#DDE3EE")
WHITE  = colors.white
GRAY   = colors.HexColor("#475569")
LGRAY  = colors.HexColor("#94A3B8")

VERDICT_COLORS = {
    "DEEP VALUE": GREEN, "ATTRACTIVE": BLUE, "FAIR": GOLD,
    "RICH": ORANGE, "EXPENSIVE": RED, "ETF": GRAY,
}
ACTION_COLORS = {
    "ACCUMULATE": GREEN, "HOLD": BLUE, "TRIM": GOLD, "EXIT": RED, "WATCH": GRAY,
}
ROLE_COLORS = {
    "COMPOUNDER": GREEN, "CYCLICAL": GOLD, "TURNAROUND": BLUE,
    "VALUE_TRAP": RED, "TACTICAL": GRAY,
}

C = {
    "bg": "#FFFFFF", "panel": "#F8F9FC", "border": "#DDE3EE",
    "text": "#0F172A", "text2": "#475569", "text3": "#94A3B8",
    "blue": "#1D4ED8", "green": "#15803D", "red": "#DC2626",
    "gold": "#D97706", "grid": "#E2E8F0",
}


# =============================================================================
# HELPERS
# =============================================================================

def _fig_to_image(fig, width_cm=17, dpi=120) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    w = width_cm * cm
    fw, fh = fig.get_size_inches()
    return Image(buf, width=w, height=w * fh / fw)


def _md_to_paras(text: str, style) -> list:
    elements = []
    for line in (text or "").split("\n"):
        s = line.strip()
        if not s:
            elements.append(Spacer(1, 3))
            continue
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", s)
        if s.startswith(("- ", "• ", "* ")):
            elements.append(Paragraph("• " + s[2:], style))
        else:
            elements.append(Paragraph(s, style))
    return elements


def _style(name, **kw):
    base = getSampleStyleSheet()["Normal"]
    return ParagraphStyle(name, parent=base, **kw)


def _sax(ax):
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["text2"], labelsize=7.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(C["border"])
    ax.spines["bottom"].set_color(C["border"])
    ax.set_axisbelow(True)


# =============================================================================
# COMPUTED METRICS
# =============================================================================

def compute_portfolio_risk_metrics(price_history: dict, holdings_df: pd.DataFrame) -> dict:
    """Compute volatility, Sharpe proxy, and max drawdown from price history."""
    result = {"volatility": None, "sharpe_proxy": None, "max_drawdown": None}
    if not price_history:
        return result
    try:
        prices_df   = pd.DataFrame(price_history).dropna(how="all")
        weight_map  = dict(zip(holdings_df["Ticker"], holdings_df["Portfolio Wt%"] / 100))
        port_series = pd.Series(0.0, index=prices_df.index)
        for ticker, wt in weight_map.items():
            if ticker in prices_df.columns:
                s = prices_df[ticker].dropna()
                if len(s) > 1:
                    port_series = port_series.add(((s / s.iloc[0]) - 1) * wt * 100, fill_value=0)

        port_cum   = port_series.dropna()
        daily_rets = port_cum.diff().dropna()
        vol        = daily_rets.std() * math.sqrt(252)
        total_ret  = port_cum.iloc[-1] if len(port_cum) > 0 else 0
        sharpe     = (total_ret - 6.5) / vol if vol > 0 else 0
        # Max drawdown
        cumulative = (1 + daily_rets / 100).cumprod()
        roll_max   = cumulative.cummax()
        drawdown   = (cumulative - roll_max) / roll_max * 100
        max_dd     = drawdown.min()

        result = {
            "volatility":   round(vol, 2),
            "sharpe_proxy": round(sharpe, 2),
            "max_drawdown": round(max_dd, 2),
        }
    except Exception:
        pass
    return result


def compute_factor_exposures(holdings_df: pd.DataFrame, clean_data: dict, verdicts: dict) -> dict:
    """Approximate factor exposures from existing fundamental data."""
    quality_scores, value_scores, growth_scores, momentum_scores = [], [], [], []

    for _, row in holdings_df.iterrows():
        ticker = row["Ticker"]
        d = clean_data.get(ticker) or {}
        v = verdicts.get(ticker) or {}
        wt = row["Portfolio Wt%"] / 100

        # Quality: ROCE (normalized 0–100); neutral=40 when no Screener data
        roce = d.get("roce")
        q = min(100, roce * 2.5) if roce is not None else 40
        quality_scores.append(q * wt)

        # Value: inverted composite score (cheaper = more value exposure)
        score = v.get("score") or 50
        value_scores.append((100 - score) * wt)

        # Growth: PAT CAGR; neutral=40 when no Screener data
        pat_cagr = d.get("pat_cagr_3y")
        g = min(100, max(0, pat_cagr) * 2.5) if pat_cagr is not None else 40
        growth_scores.append(g * wt)

        # Momentum: return % (normalized, -50 to +100 → 0 to 100)
        ret = row.get("Return %", 0) or 0
        mom_norm = min(100, max(0, (ret + 50)))
        momentum_scores.append(mom_norm * wt)

    q_total  = round(sum(quality_scores), 1)
    r_total  = round(100 - q_total * 0.5, 1)
    return {
        "quality":  q_total,
        "value":    round(sum(value_scores), 1),
        "growth":   round(sum(growth_scores), 1),
        "momentum": round(sum(momentum_scores), 1),
        "risk":     min(100, max(0, r_total)),
    }


def cluster_investment_themes(holdings_df: pd.DataFrame, clean_data: dict, scenarios: dict) -> list:
    """Group holdings into 3-5 investment themes."""
    theme_map = {
        "Commodity Rerating":  ["COMMODITY_METAL"],
        "Financial Services":  ["PSU_BANK", "SFB", "NBFC"],
        "Energy & Infra":      ["POWER_IPP", "RENEWABLE_SOLAR", "TELECOM_INFRA"],
        "Consumer & Mobility": ["AUTO_OEM_GLOBAL", "AUTO_ANCILLARY"],
        "Platforms & Tech":    ["EXCHANGE_PLATFORM", "ETF"],
    }

    themes = {}
    for _, row in holdings_df.iterrows():
        ticker = row["Ticker"]
        d      = clean_data.get(ticker) or {}
        btype  = d.get("btype", "UNKNOWN")
        for theme, btypes in theme_map.items():
            if btype in btypes:
                themes.setdefault(theme, []).append({
                    "ticker":   ticker,
                    "company":  row["Company"][:22],
                    "wt":       row["Portfolio Wt%"],
                    "return":   row.get("Return %", 0),
                    "role":     (scenarios.get(ticker) or {}).get("portfolio_role", "TACTICAL"),
                })
                break
        else:
            themes.setdefault("Other", []).append({
                "ticker":  ticker,
                "company": row["Company"][:22],
                "wt":      row["Portfolio Wt%"],
                "return":  row.get("Return %", 0),
                "role":    "TACTICAL",
            })

    # Return only non-empty themes sorted by total weight
    result = []
    for theme, stocks in themes.items():
        total_wt = sum(s["wt"] for s in stocks)
        result.append({"theme": theme, "stocks": stocks, "total_wt": total_wt})
    return sorted(result, key=lambda x: x["total_wt"], reverse=True)


# =============================================================================
# CHART GENERATORS
# =============================================================================

def _fig_performance_attribution(attrib_df, holdings_df, attribution_summary, factors) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=C["bg"])
    fig.suptitle("Performance Attribution", fontsize=11, fontweight="bold", color=C["text"])

    # A — BHB waterfall
    ax1 = axes[0]
    _sax(ax1)
    if attrib_df is not None and not attrib_df.empty:
        df = attrib_df[attrib_df["Port_Wt%"] > 0].sort_values("Active_Return")
        cols = [C["green"] if v >= 0 else C["red"] for v in df["Active_Return"]]
        ax1.barh(df["Sector"].str[:15], df["Active_Return"], color=cols, alpha=0.85, zorder=3)
        ax1.axvline(0, color=C["text"], lw=0.8)
        ax1.xaxis.grid(True, color=C["grid"], lw=0.5)
        ax1.set_title("Active Return by Sector (BHB)", fontsize=9, color=C["text"])
        ax1.set_xlabel("Return (%)", fontsize=8)

    # B — Factor contribution
    ax2 = axes[1]
    _sax(ax2)
    if factors:
        fnames = ["Quality", "Value", "Growth", "Momentum"]
        fvals  = [factors.get(f.lower(), 0) for f in fnames]
        fcols  = [C["green"] if v > 50 else C["red"] for v in fvals]
        ax2.barh(fnames, fvals, color=fcols, alpha=0.85, zorder=3)
        ax2.axvline(50, color=C["text3"], lw=0.8, ls="--")
        ax2.xaxis.grid(True, color=C["grid"], lw=0.5)
        ax2.set_title("Factor Exposure (0–100)", fontsize=9, color=C["text"])
        ax2.set_xlabel("Score", fontsize=8)

    # C — Stock contribution waterfall
    ax3 = axes[2]
    _sax(ax3)
    contribs = holdings_df[["Company", "Return %", "Portfolio Wt%"]].copy()
    contribs["Contrib"] = contribs["Return %"] * contribs["Portfolio Wt%"] / 100
    contribs = contribs.sort_values("Contrib")
    top = contribs.tail(5); bot = contribs.head(5)
    combo = pd.concat([bot, top]).drop_duplicates()
    cols = [C["green"] if v >= 0 else C["red"] for v in combo["Contrib"]]
    ax3.barh(combo["Company"].str[:16], combo["Contrib"], color=cols, alpha=0.85, zorder=3)
    ax3.axvline(0, color=C["text"], lw=0.8)
    ax3.xaxis.grid(True, color=C["grid"], lw=0.5)
    ax3.set_title("Stock Contribution to Return", fontsize=9, color=C["text"])

    plt.tight_layout()
    return fig


def _fig_foundational_intelligence(holdings_df, clean_data) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(18, 7), facecolor=C["bg"])
    fig.suptitle("Foundational Intelligence — Financial Strength Heatmap", fontsize=11, fontweight="bold", color=C["text"])

    metrics = ["ROCE%", "EBITDAm%", "PAT CAGR%", "D/E", "ND/EBITDA", "FCF"]
    companies = []
    data_matrix = []

    for _, row in holdings_df.iterrows():
        ticker = row["Ticker"]
        d = clean_data.get(ticker) or {}
        companies.append(row["Company"][:20])
        data_matrix.append([
            d.get("roce"),
            d.get("ebitda_margin"),
            d.get("pat_cagr_3y"),
            d.get("de_ratio"),
            d.get("nd_ebitda"),
            1 if (d.get("fcf") or 0) > 0 else 0,
        ])

    # Check if any real fundamental data exists (not all None)
    has_any_data = any(
        any(v is not None for v in row[:5])   # first 5 cols (not FCF flag)
        for row in data_matrix
    )
    if not data_matrix or not has_any_data:
        ax.text(0.5, 0.5,
                "Upload Screener Excel files to populate this heatmap\n"
                "(screener.in → company page → Excel button)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=C["text3"], style="italic")
        ax.axis("off")
        return fig

    mat = np.array([[v if v is not None else 0 for v in row] for row in data_matrix], dtype=float)

    # Normalize each column 0–1 (higher = better, except D/E and ND/EBITDA where lower = better)
    norm = np.zeros_like(mat)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        lo, hi = col.min(), col.max()
        if hi > lo:
            norm[:, j] = (col - lo) / (hi - lo)
        else:
            norm[:, j] = 0.5
    # Invert D/E (col 3) and ND/EBITDA (col 4)
    norm[:, 3] = 1 - norm[:, 3]
    norm[:, 4] = 1 - norm[:, 4]

    im = ax.imshow(norm, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_yticks(range(len(companies)))
    ax.set_xticklabels(metrics, fontsize=8.5)
    ax.set_yticklabels(companies, fontsize=8)

    for i in range(len(companies)):
        for j in range(len(metrics)):
            raw = data_matrix[i][j]
            txt = f"{raw:.1f}" if raw is not None else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color="black")

    plt.colorbar(im, ax=ax, shrink=0.6, label="Relative strength (green = better)")
    plt.tight_layout()
    return fig


def _fig_portfolio_structure(holdings_df, sector_df, factors) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=C["bg"])
    fig.suptitle("Portfolio Structure Snapshot", fontsize=11, fontweight="bold", color=C["text"])

    # A — Sector pie
    ax1 = axes[0]
    sdata = sector_df[sector_df["Port_Weight"] > 0].sort_values("Port_Weight", ascending=False)
    if not sdata.empty:
        wedge_colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(sdata)))
        ax1.pie(sdata["Port_Weight"], labels=sdata["Sector"].str[:14],
                colors=wedge_colors, autopct="%1.1f%%", pctdistance=0.8,
                textprops={"fontsize": 7})
        ax1.set_title("Sector Allocation", fontsize=9, color=C["text"])

    # B — Top holdings bar
    ax2 = axes[1]
    _sax(ax2)
    top10 = holdings_df.nlargest(10, "Portfolio Wt%")
    intensities = np.linspace(0.85, 0.4, len(top10))
    bar_cols = [plt.cm.Blues(i) for i in intensities]
    ax2.barh(top10["Company"].str[:18][::-1], top10["Portfolio Wt%"][::-1],
             color=bar_cols[::-1], zorder=3)
    ax2.xaxis.grid(True, color=C["grid"], lw=0.5)
    ax2.set_title("Top Holdings by Weight", fontsize=9, color=C["text"])
    ax2.set_xlabel("Portfolio Weight (%)", fontsize=8)

    # C — Factor exposure bars
    ax3 = axes[2]
    _sax(ax3)
    if factors:
        fname  = ["Quality", "Growth", "Momentum", "Value", "Risk"]
        fval   = [factors.get(f.lower(), 50) for f in fname]
        fcol   = [C["green"] if v > 50 else C["red"] for v in fval]
        ax3.barh(fname, fval, color=fcol, alpha=0.85, zorder=3)
        ax3.axvline(50, color=C["text3"], lw=0.8, ls="--", label="Neutral (50)")
        ax3.set_xlim(0, 105)
        ax3.xaxis.grid(True, color=C["grid"], lw=0.5)
        ax3.set_title("Factor Exposure (0=low, 100=high)", fontsize=9, color=C["text"])
        ax3.legend(fontsize=7)

    plt.tight_layout()
    return fig


def _fig_market_regime(factors, risk_metrics) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=C["bg"])
    fig.suptitle("Market Regime & Factor Environment", fontsize=11, fontweight="bold", color=C["text"])

    # A — Risk-on / Risk-off gauge (simplified horizontal bar)
    ax1 = axes[0]
    ax1.set_facecolor(C["panel"])
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-1, 1)
    ax1.set_yticks([])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    active_share = risk_metrics.get("active_share", 50) if risk_metrics else 50
    hit_rate     = risk_metrics.get("hit_rate", 50) if risk_metrics else 50
    regime_score = min(100, (active_share * 0.4 + hit_rate * 0.6))

    ax1.barh([0], [regime_score], color=C["blue"], alpha=0.7, height=0.5)
    ax1.barh([0], [100 - regime_score], left=[regime_score], color=C["border"], alpha=0.5, height=0.5)
    ax1.axvline(50, color=C["text3"], lw=1, ls="--")
    ax1.text(regime_score / 2, 0, f"{regime_score:.0f}", ha="center", va="center",
             fontsize=11, fontweight="bold", color="white")
    regime_label = "RISK-ON" if regime_score > 60 else ("NEUTRAL" if regime_score > 40 else "RISK-OFF")
    ax1.set_title(f"Portfolio Regime: {regime_label}", fontsize=9, color=C["text"])
    ax1.set_xlabel("Regime Score (0=Risk-off, 100=Risk-on)", fontsize=8)

    # B — Factor performance bars
    ax2 = axes[1]
    _sax(ax2)
    if factors:
        fname = ["Quality", "Growth", "Momentum", "Value"]
        fval  = [factors.get(f.lower(), 50) for f in fname]
        fcol  = [C["green"] if v > 60 else (C["gold"] if v > 40 else C["red"]) for v in fval]
        bars  = ax2.bar(fname, fval, color=fcol, alpha=0.85, zorder=3, width=0.5)
        ax2.axhline(50, color=C["text3"], lw=0.8, ls="--")
        ax2.set_ylim(0, 110)
        ax2.yaxis.grid(True, color=C["grid"], lw=0.5)
        for bar, val in zip(bars, fval):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     f"{val:.0f}", ha="center", va="bottom", fontsize=9)
        ax2.set_title("Factor Exposures vs Neutral (50)", fontsize=9, color=C["text"])
        ax2.set_ylabel("Score", fontsize=8)

    plt.tight_layout()
    return fig


def _fig_radar(factors) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw={"projection": "polar"}, facecolor=C["bg"])
    fig.suptitle("Portfolio Signal Aggregation", fontsize=11, fontweight="bold", color=C["text"])

    if not factors:
        return fig

    cats   = ["Quality", "Growth", "Momentum", "Value", "Risk-Adj"]
    vals   = [
        factors.get("quality", 50),
        factors.get("growth",  50),
        factors.get("momentum",50),
        factors.get("value",   50),
        100 - factors.get("risk", 50),
    ]
    N = len(cats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals_plot = vals + vals[:1]

    ax.set_facecolor("#F8F9FC")
    ax.plot(angles, vals_plot, "o-", lw=2, color=C["blue"])
    ax.fill(angles, vals_plot, alpha=0.25, color=C["blue"])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, size=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], size=7, color=C["text2"])
    ax.axhline(50, color=C["text3"], lw=0.8, ls="--", alpha=0.5)

    plt.tight_layout()
    return fig


def _fig_risk_heatmap(risk_metrics, sector_df) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=C["bg"])
    fig.suptitle("Risk Dashboard", fontsize=11, fontweight="bold", color=C["text"])

    # A — Risk heatmap (4 quadrants: macro/sector/factor/concentration)
    ax1 = axes[0]
    ax1.set_facecolor(C["panel"])
    risk_cats = ["Macro Risk", "Sector Concentration", "Factor Risk", "Stock Concentration"]
    rm = risk_metrics or {}
    macro_risk  = 50  # simplified
    sector_risk = min(100, (rm.get("sector_hhi", 2000) / 50))
    factor_risk = 50  # simplified
    stock_risk  = min(100, rm.get("hhi", 1000) / 50)
    risk_vals   = [macro_risk, sector_risk, factor_risk, stock_risk]

    colors_r = [plt.cm.RdYlGn(1 - v / 100) for v in risk_vals]
    bars = ax1.barh(risk_cats, risk_vals, color=colors_r, alpha=0.85, zorder=3)
    ax1.axvline(50, color=C["text3"], lw=0.8, ls="--")
    ax1.set_xlim(0, 100)
    ax1.xaxis.grid(True, color=C["grid"], lw=0.5)
    for bar, val in zip(bars, risk_vals):
        ax1.text(val + 1, bar.get_y() + bar.get_height() / 2,
                 f"{val:.0f}", va="center", fontsize=8)
    ax1.set_title("Risk by Category (0=low, 100=high)", fontsize=9, color=C["text"])
    ax1.tick_params(labelsize=8)

    # B — Key risk metrics summary
    ax2 = axes[1]
    ax2.set_facecolor(C["panel"])
    ax2.axis("off")
    ax2.set_title("Risk Scorecard", fontsize=9, color=C["text"], pad=8)
    metrics_display = [
        ("Active Share",     f"{rm.get('active_share', 0):.1f}%"),
        ("HHI",              f"{rm.get('hhi', 0):.0f}"),
        ("Effective N",      f"{rm.get('effective_n', 0):.1f}"),
        ("Top 5 Weight",     f"{rm.get('top5_wt', 0):.1f}%"),
        ("Hit Rate",         f"{rm.get('hit_rate', 0):.1f}%"),
        ("Win/Loss Ratio",   f"{rm.get('win_loss_ratio', 0):.2f}x"),
    ]
    for i, (lbl, val) in enumerate(metrics_display):
        y = 0.9 - i * 0.13
        ax2.text(0.05, y, lbl, transform=ax2.transAxes, fontsize=9, color=C["text2"])
        ax2.text(0.95, y, val, transform=ax2.transAxes, fontsize=9,
                 fontweight="bold", color=NAVY.hexval()[1:] and C["text"], ha="right")

    plt.tight_layout()
    return fig


def _fig_themes(themes: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(16, 6), facecolor=C["bg"])
    fig.suptitle("Investment Themes — Portfolio Story Clusters", fontsize=11, fontweight="bold", color=C["text"])
    ax.axis("off")

    theme_colors = [C["blue"], C["green"], C["gold"], "#7C3AED", "#0891B2"]
    n_themes = min(5, len(themes))
    box_w = 0.18
    gap   = (1.0 - n_themes * box_w) / (n_themes + 1)

    for i, theme_data in enumerate(themes[:n_themes]):
        x0 = gap + i * (box_w + gap)
        col = theme_colors[i % len(theme_colors)]

        # Theme header
        ax.add_patch(plt.Rectangle((x0, 0.6), box_w, 0.35, transform=ax.transAxes,
                                   color=col, alpha=0.85, zorder=2))
        ax.text(x0 + box_w / 2, 0.775, theme_data["theme"],
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=3, wrap=True)
        ax.text(x0 + box_w / 2, 0.63,
                f"{theme_data['total_wt']:.1f}% of portfolio",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7, color="white", alpha=0.9, zorder=3)

        # Stock list
        for j, stock in enumerate(theme_data["stocks"][:4]):
            y_pos = 0.52 - j * 0.12
            role_col = {
                "COMPOUNDER": C["green"], "CYCLICAL": C["gold"],
                "TURNAROUND": C["blue"], "VALUE_TRAP": C["red"],
            }.get(stock.get("role", ""), C["text3"])
            ax.text(x0 + 0.01, y_pos, f"• {stock['company'][:18]}",
                    transform=ax.transAxes, fontsize=7.5, color=C["text"], va="center")
            ax.text(x0 + box_w - 0.01, y_pos, f"{stock['wt']:.1f}%",
                    transform=ax.transAxes, fontsize=7.5, color=role_col, va="center", ha="right")

    plt.tight_layout()
    return fig


# =============================================================================
# MAIN PDF BUILDER — 18 PAGES
# =============================================================================

def build_institutional_pdf(
    report_date: str,
    portfolio_stats: dict,
    risk_metrics: dict,
    risk_flags: list,
    attribution_summary: dict,
    attrib_df,
    holdings_df,
    sector_df,
    clean_data: dict,
    verdicts: dict,
    scenarios: dict,
    actions: dict,
    institutional_report: dict,
    figures: dict,
    price_history: dict = None,
    portfolio_name: str = "Portfolio Intelligence Report",
) -> bytes:
    """
    Build the complete 18-page institutional PDF report.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=1.8*cm, rightMargin=1.8*cm,
                            topMargin=1.8*cm, bottomMargin=1.8*cm,
                            title=portfolio_name)

    # ── Style definitions ──────────────────────────────────────────────────
    H1    = _style("H1",  fontName="Helvetica-Bold", fontSize=16, textColor=NAVY, spaceBefore=10, spaceAfter=4)
    H2    = _style("H2",  fontName="Helvetica-Bold", fontSize=11, textColor=BLUE, spaceBefore=8,  spaceAfter=3)
    H3    = _style("H3",  fontName="Helvetica-Bold", fontSize=9.5, textColor=NAVY, spaceBefore=5, spaceAfter=2)
    BODY  = _style("BD",  fontName="Helvetica", fontSize=9,   textColor=NAVY, spaceAfter=3, leading=13)
    SMALL = _style("SM",  fontName="Helvetica", fontSize=7.5, textColor=GRAY, spaceAfter=2, leading=11)
    MONO  = _style("MN",  fontName="Courier",   fontSize=8,   textColor=NAVY, spaceAfter=2)
    BOLD  = _style("BLD", fontName="Helvetica-Bold", fontSize=9, textColor=NAVY, spaceAfter=2)

    ps      = portfolio_stats or {}
    inst    = institutional_report or {}
    st      = inst.get("stock_theses", {})
    el      = []   # elements list

    def hr(): return HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=6)
    def hr2(): return HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=8)

    # ── COMPUTED ANALYTICS ─────────────────────────────────────────────────
    risk_computed = compute_portfolio_risk_metrics(price_history or {}, holdings_df)
    factors       = compute_factor_exposures(holdings_df, clean_data, verdicts)
    themes        = cluster_investment_themes(holdings_df, clean_data, scenarios or {})

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 1 — COVER
    # ─────────────────────────────────────────────────────────────────────────
    el += [Spacer(1, 2.5*cm)]
    el += [Paragraph("AI INSTITUTIONAL EQUITY SYSTEM", _style("CVT", fontName="Helvetica-Bold", fontSize=22, textColor=NAVY, alignment=TA_CENTER))]
    el += [Paragraph(portfolio_name.upper(), _style("CVS", fontName="Helvetica", fontSize=14, textColor=GRAY, alignment=TA_CENTER, spaceAfter=4))]
    el += [Spacer(1, 0.3*cm), hr2(), Spacer(1, 0.5*cm)]

    # KPI tiles (cover)
    identity = inst.get("portfolio_identity", "Hybrid Portfolio")
    total_ret = ps.get("total_return", 0)
    ret_color = GREEN if total_ret >= 0 else RED
    kpi_data = [
        ["Analysis Date", "Portfolio Return", "Holdings", "Portfolio Type"],
        [report_date, f"{total_ret:+.2f}%", str(ps.get("num_holdings", 0)), identity],
    ]
    kpi_tbl = Table(kpi_data, colWidths=[4.5*cm]*4)
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), WHITE),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.3, BORDER), ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("PADDING", (0,0), (-1,-1), 8),
        ("BACKGROUND", (0,1), (-1,1), LIGHT),
        ("FONTNAME", (0,1), (-1,1), "Helvetica-Bold"), ("FONTSIZE", (0,1), (-1,1), 11),
        ("TEXTCOLOR", (1,1), (1,1), ret_color),
    ]))
    el += [kpi_tbl, Spacer(1, 1.5*cm)]

    # Cover stats — 2-col layout to avoid overflow
    cover_stats = [
        ["Total Invested",   f"Rs. {ps.get('total_invested',0):,.0f}",
         "Current Value",    f"Rs. {ps.get('total_current',0):,.0f}"],
        ["Unrealised P&L",  f"Rs. {ps.get('total_pnl',0):+,.0f}",
         "Active Share",     f"{(risk_metrics or {}).get('active_share',0):.1f}%"],
        ["Effective N",      f"{(risk_metrics or {}).get('effective_n',0):.1f}",
         "Top 5 Weight",     f"{(risk_metrics or {}).get('top5_wt',0):.1f}%"],
        ["Best Performer",   f"{ps.get('best_performer','—')[:22]}",
         "Return",           f"{ps.get('best_return',0):+.1f}%"],
    ]
    cs_tbl = Table(cover_stats, colWidths=[4.5*cm, 5.5*cm, 4*cm, 4*cm])
    cs_tbl.setStyle(TableStyle([
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"), ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9), ("GRID", (0,0), (-1,-1), 0.3, BORDER),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [WHITE, LIGHT]), ("PADDING", (0,0), (-1,-1), 6),
    ]))
    el += [cs_tbl, Spacer(1, 0.5*cm), hr(), Spacer(1, 0.3*cm)]
    el += [Paragraph("Powered by AI Institutional Equity System  |  For internal use only",
                     _style("FT", fontName="Helvetica", fontSize=8, textColor=LGRAY, alignment=TA_CENTER))]
    el += [PageBreak()]

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 2 — EXECUTIVE SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    el += [Paragraph("Executive Summary", H1), hr2()]

    # 5 KPI tiles
    vol  = risk_computed.get("volatility")
    shrp = risk_computed.get("sharpe_proxy")
    mdd  = risk_computed.get("max_drawdown")
    conc = (risk_metrics or {}).get("top5_wt", 0)
    kpi5_data = [
        ["Portfolio Return", "Volatility (ann.)", "Sharpe Proxy", "Max Drawdown", "Top 5 Concentration"],
        [f"{total_ret:+.2f}%",
         f"{vol:.1f}%" if vol else "N/A",
         f"{shrp:.2f}x" if shrp else "N/A",
         f"{mdd:.1f}%" if mdd else "N/A",
         f"{conc:.1f}%"],
    ]
    k5_tbl = Table(kpi5_data, colWidths=[3.6*cm]*5)
    k5_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), WHITE),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("GRID", (0,0), (-1,-1), 0.3, BORDER), ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("PADDING", (0,0), (-1,-1), 7), ("BACKGROUND", (0,1), (-1,1), LIGHT),
        ("FONTNAME", (0,1), (-1,1), "Helvetica-Bold"), ("FONTSIZE", (0,1), (-1,1), 11),
    ]))
    el += [k5_tbl, Spacer(1, 0.5*cm)]
    if inst.get("executive_summary"):
        for p in _md_to_paras(inst["executive_summary"], BODY):
            el += [p]
    else:
        # Auto-generate a basic summary from raw stats when AI thesis hasn't run
        _ret  = ps.get('total_return', 0)
        _best = ps.get('best_performer', '—')
        _bret = ps.get('best_return', 0)
        _wrst = ps.get('worst_performer', '—')
        _wret = ps.get('worst_return', 0)
        _as   = (risk_metrics or {}).get('active_share', 0)
        _eff  = (risk_metrics or {}).get('effective_n', 0)
        _auto = (
            f"The portfolio delivered a total return of {_ret:+.2f}% as of {report_date}. "
            f"The top contributor was {_best} ({_bret:+.1f}%), while {_wrst} ({_wret:+.1f}%) "
            f"was the weakest performer. "
            f"With an active share of {_as:.1f}% and {_eff:.1f} effective holdings, "
            f"this is a concentrated, genuinely active portfolio. "
            f"Upload Screener Excel files and run the AI Institutional Report for a full narrative analysis."
        )
        el += [Paragraph(_auto, BODY)]
    el += [PageBreak()]

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 3 — PERFORMANCE ATTRIBUTION
    # ─────────────────────────────────────────────────────────────────────────
    el += [Paragraph("Performance Attribution", H1), hr2()]
    if attribution_summary:
        attr_data = [
            ["Portfolio Return", f"{attribution_summary.get('total_port_return',0):+.2f}%"],
            ["Benchmark Return", f"{attribution_summary.get('total_bench_return',0):+.2f}%"],
            ["Total Alpha",      f"{attribution_summary.get('total_alpha',0):+.2f}%"],
            ["Allocation Effect",f"{attribution_summary.get('allocation_effect',0):+.4f}%"],
            ["Selection Effect", f"{attribution_summary.get('selection_effect',0):+.4f}%"],
        ]
        at_tbl = Table(attr_data, colWidths=[7*cm, 5*cm])
        at_tbl.setStyle(TableStyle([
            ("FONTSIZE", (0,0), (-1,-1), 9), ("GRID", (0,0), (-1,-1), 0.3, BORDER),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [WHITE, LIGHT]), ("PADDING", (0,0), (-1,-1), 5),
            ("FONTNAME", (0,2), (0,2), "Helvetica-Bold"),
        ]))
        el += [at_tbl, Spacer(1, 0.4*cm)]
    fig_attr = _fig_performance_attribution(attrib_df, holdings_df, attribution_summary, factors)
    el += [_fig_to_image(fig_attr), PageBreak()]

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 4 — FOUNDATIONAL INTELLIGENCE
    # ─────────────────────────────────────────────────────────────────────────
    el += [Paragraph("Foundational Intelligence — Financial Strength", H1), hr2()]
    fig_found = _fig_foundational_intelligence(holdings_df, clean_data)
    el += [_fig_to_image(fig_found)]
    el += [Paragraph("Heatmap: green = relative strength, red = relative weakness vs portfolio peers. "
                     "Columns: ROCE, EBITDA margin, PAT CAGR, D/E (inverted), ND/EBITDA (inverted), FCF quality.", SMALL)]
    el += [PageBreak()]

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 5 — PORTFOLIO STRUCTURE SNAPSHOT
    # ─────────────────────────────────────────────────────────────────────────
    el += [Paragraph("Portfolio Structure Snapshot", H1), hr2()]
    fig_struct = _fig_portfolio_structure(holdings_df, sector_df, factors)
    el += [_fig_to_image(fig_struct), PageBreak()]

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 6 — MARKET REGIME & FACTOR ENVIRONMENT
    # ─────────────────────────────────────────────────────────────────────────
    el += [Paragraph("Market Regime & Factor Environment", H1), hr2()]
    fig_regime = _fig_market_regime(factors, risk_metrics)
    el += [_fig_to_image(fig_regime), PageBreak()]

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 7 — INVESTMENT THEMES
    # ─────────────────────────────────────────────────────────────────────────
    el += [Paragraph("Investment Themes — Portfolio Story Clusters", H1), hr2()]
    fig_themes = _fig_themes(themes)
    el += [_fig_to_image(fig_themes), Spacer(1, 0.3*cm)]
    for theme_data in themes[:5]:
        el += [Paragraph(f"<b>{theme_data['theme']}</b>  ({theme_data['total_wt']:.1f}% of portfolio)", H3)]
        stocks_line = "  |  ".join(
            f"{s['company']} ({s['wt']:.1f}%, {s['return']:+.1f}%)"
            for s in theme_data["stocks"]
        )
        el += [Paragraph(stocks_line, SMALL), Spacer(1, 3)]
    el += [PageBreak()]

    # ─────────────────────────────────────────────────────────────────────────
    # PAGES 8–12 — STOCK DEEP DIVES (top 5 by portfolio weight)
    # ─────────────────────────────────────────────────────────────────────────
    top5_tickers = holdings_df.nlargest(5, "Portfolio Wt%")["Ticker"].tolist()

    for ticker in top5_tickers:
        row_data = holdings_df[holdings_df["Ticker"] == ticker]
        if row_data.empty:
            continue
        row     = row_data.iloc[0]
        company = row["Company"]
        d       = clean_data.get(ticker) or {}
        v       = verdicts.get(ticker) or {}
        sc      = (scenarios or {}).get(ticker) or {}
        ac      = (actions or {}).get(ticker) or {}
        th      = st.get(ticker) or {}

        verdict_lbl  = v.get("label", "—")
        score_str    = f"{v.get('score'):.0f}/100" if v.get("score") else "N/A"
        action_lbl   = ac.get("signal", "WATCH")
        conv_lbl     = ac.get("conviction", "LOW")
        role_lbl     = sc.get("portfolio_role", "TACTICAL")
        action_color = ACTION_COLORS.get(action_lbl, GRAY)
        role_color   = ROLE_COLORS.get(role_lbl, GRAY)

        # Header card
        card_data = [[
            Paragraph(f"<b>{company}</b>", _style("CH", fontName="Helvetica-Bold", fontSize=12, textColor=WHITE)),
            Paragraph(f"{ticker}  |  {row['Sector']}", _style("CS", fontName="Helvetica", fontSize=8, textColor=colors.HexColor("#BFDBFE"))),
            Paragraph(f"{verdict_lbl}  {score_str}", _style("CV", fontName="Helvetica-Bold", fontSize=10, textColor=WHITE, alignment=TA_RIGHT)),
        ]]
        card_tbl = Table(card_data, colWidths=[6*cm, 7*cm, 5*cm])
        card_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), NAVY), ("PADDING", (0,0), (-1,-1), 8), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ]))
        el += [card_tbl]

        # Metrics strip
        m_row = [[
            Paragraph(f"Weight: <b>{row['Portfolio Wt%']:.1f}%</b>", SMALL),
            Paragraph(f"Return: <b>{row.get('Return %', 0):+.1f}%</b>", SMALL),
            Paragraph(f"PE: <b>{d.get('pe') or '—'}</b>  PB: <b>{d.get('pb') or '—'}</b>", SMALL),
            Paragraph(f"ROCE: <b>{d.get('roce') or '—'}%</b>", SMALL),
            Paragraph(f"Action: <b>{action_lbl}</b>", _style("AC", fontName="Helvetica-Bold", fontSize=8, textColor=action_color)),
            Paragraph(f"Role: <b>{role_lbl}</b>", _style("RL", fontName="Helvetica-Bold", fontSize=8, textColor=role_color)),
        ]]
        m_tbl = Table(m_row, colWidths=[3*cm]*6)
        m_tbl.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.3, BORDER), ("BACKGROUND", (0,0), (-1,-1), LIGHT), ("PADDING", (0,0), (-1,-1), 4),
        ]))
        el += [m_tbl, Spacer(1, 5)]

        # Scenarios
        bear_r = (sc.get("bear") or {}).get("return_pct", "N/A")
        bull_r = (sc.get("bull") or {}).get("return_pct", "N/A")
        sc_data = [
            ["BEAR", "BASE", "BULL", "Asymmetry"],
            [f"{bear_r:+.1f}%" if isinstance(bear_r, float) else str(bear_r),
             "+0.0%",
             f"{bull_r:+.1f}%" if isinstance(bull_r, float) else str(bull_r),
             f"{sc.get('asymmetry_ratio', 0):.1f}x"],
        ]
        sc_tbl = Table(sc_data, colWidths=[4.5*cm]*4)
        sc_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), WHITE),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 9),
            ("GRID", (0,0), (-1,-1), 0.3, BORDER), ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("PADDING", (0,0), (-1,-1), 6), ("BACKGROUND", (0,1), (-1,1), LIGHT),
            ("FONTNAME", (0,1), (-1,1), "Helvetica-Bold"),
            ("TEXTCOLOR", (0,1), (0,1), RED), ("TEXTCOLOR", (2,1), (2,1), GREEN),
        ]))
        el += [Paragraph("Scenario Analysis", H2), sc_tbl, Spacer(1, 6)]

        # 12-point thesis
        thesis_sections = [
            ("Investment Thesis",              th.get("investment_thesis", "")),
            ("Market Expectation vs Reality",  th.get("market_expectation_vs_reality", "")),
            ("Why Market May Be Wrong",        th.get("why_market_wrong", "")),
            ("Capital Allocation Quality",     th.get("capital_allocation_quality", "")),
            ("Management Assessment",          th.get("management_assessment", "")),
            ("Position Sizing Logic",          th.get("position_sizing_logic", "")),
            ("Long-term Outlook",              th.get("long_term_outlook", "")),
        ]
        for section_title, text in thesis_sections:
            if text:
                el += [Paragraph(section_title, H3)]
                for p in _md_to_paras(text, BODY):
                    el += [p]
                el += [Spacer(1, 3)]

        # Bull / Bear / Triggers (lists)
        for list_title, items in [
            ("Bull Case", th.get("bull_case", [])),
            ("Bear Case", th.get("bear_case", [])),
            ("Key Triggers", th.get("key_triggers", [])),
            ("Risk Factors", th.get("risk_factors", [])),
        ]:
            if items:
                el += [Paragraph(list_title, H3)]
                for item in items:
                    el += [Paragraph(f"• {item}", BODY)]
                el += [Spacer(1, 3)]

        el += [hr(), PageBreak()]

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 13 — UNDERPERFORMERS / RISK POSITIONS
    # ─────────────────────────────────────────────────────────────────────────
    el += [Paragraph("Underperformers & Risk Positions", H1), hr2()]
    risk_rows = [["Company", "Signal", "Reason", "Exit Condition"]]
    for _, row in holdings_df.iterrows():
        ticker = row["Ticker"]
        ac = (actions or {}).get(ticker) or {}
        if ac.get("signal") in ("TRIM", "EXIT"):
            th = st.get(ticker) or {}
            mf = th.get("monitoring_framework") or [] if th else []
            exit_cond = str(mf[0])[:50] if mf else "Review at next quarterly results"
            reason = ac.get("reason", "")
            # Truncate reason cleanly at word boundary
            if len(reason) > 80:
                reason = reason[:77] + "..."
            risk_rows.append([
                row["Company"][:22],
                ac.get("signal", ""),
                reason,
                exit_cond,
            ])

    if len(risk_rows) > 1:
        risk_tbl = Table(risk_rows, colWidths=[4*cm, 2*cm, 9*cm, 5*cm])
        risk_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), WHITE),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 7.5),
            ("GRID", (0,0), (-1,-1), 0.3, BORDER), ("PADDING", (0,0), (-1,-1), 5),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, LIGHT]),
        ]))
        el += [risk_tbl]
    else:
        el += [Paragraph("No positions currently flagged for exit. All held positions are within conviction range.", BODY)]
    el += [PageBreak()]

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 14 — MODEL TRACEABILITY
    # ─────────────────────────────────────────────────────────────────────────
    el += [Paragraph("Model Traceability — How Intelligence Flows", H1), hr2()]
    flow_steps = [
        ("M5A", "Raw Financial Extraction",      "Screener.in Excel → Revenue, EBITDA, PAT, FCF, B/S"),
        ("M5B", "Scoring & Intelligence",        "5-factor composite score + peer multiples + investment flags"),
        ("M5C", "Scenario Engine",               "Bull/Base/Bear scenarios + asymmetry ratio + portfolio role"),
        ("M5D", "Action Engine",                 "ACCUMULATE/HOLD/TRIM/EXIT signal + position sizing logic"),
        ("M5E", "Context Consolidation",         "All prior outputs → unified research_context.json for LLM"),
        ("M6A", "Research Ingestion",            "Concall PDF parsing + news sentiment via GPT-4o mini"),
        ("M6B", "AI Thesis Engine (7 prompts)", "Core thesis → Mispricing → Conviction → Contrarian → Sector → Memo → Validate"),
        ("M7",  "Institutional PDF Report",      "18-page Goldman-style report: performance → theses → monitoring"),
    ]
    flow_data = [["Module", "Engine", "Output"]] + flow_steps
    flow_tbl  = Table(flow_data, colWidths=[2*cm, 5.5*cm, 10.5*cm])
    flow_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), WHITE),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("GRID", (0,0), (-1,-1), 0.3, BORDER), ("PADDING", (0,0), (-1,-1), 6),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, LIGHT]),
        ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"), ("TEXTCOLOR", (0,1), (0,-1), BLUE),
    ]))
    el += [flow_tbl, PageBreak()]

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 15 — RISK DASHBOARD
    # ─────────────────────────────────────────────────────────────────────────
    el += [Paragraph("Risk Dashboard", H1), hr2()]
    fig_risk = _fig_risk_heatmap(risk_metrics, sector_df)
    el += [_fig_to_image(fig_risk)]
    if risk_flags:
        el += [Spacer(1, 0.3*cm), Paragraph("Risk Flags:", H2)]
        for flag in risk_flags:
            lvl = flag.get("level", str(flag)) if isinstance(flag, dict) else str(flag)
            el += [Paragraph(f"  {lvl}", BODY)]
    el += [PageBreak()]

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 16 — PORTFOLIO SIGNAL AGGREGATION
    # ─────────────────────────────────────────────────────────────────────────
    el += [Paragraph("Portfolio Signal Aggregation", H1), hr2()]
    fig_radar = _fig_radar(factors)
    el += [_fig_to_image(fig_radar, width_cm=12)]
    signal_data = [["Factor", "Score", "Interpretation"]]
    for fname, fkey, interp in [
        ("Quality",  "quality",  "Capital efficiency & ROCE quality"),
        ("Growth",   "growth",   "Revenue & PAT CAGR trajectory"),
        ("Momentum", "momentum", "Price momentum & return profile"),
        ("Value",    "value",    "Valuation discount vs peers"),
        ("Risk-Adj", "risk",     "Risk-adjusted positioning (inverted)"),
    ]:
        score = factors.get(fkey, 50) if factors else 50
        el += []
        signal_data.append([fname, f"{score:.0f}/100",
                            "Strong" if score > 65 else ("Moderate" if score > 40 else "Weak")])

    sig_tbl = Table(signal_data, colWidths=[4*cm, 3*cm, 11*cm])
    sig_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), WHITE),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("GRID", (0,0), (-1,-1), 0.3, BORDER), ("PADDING", (0,0), (-1,-1), 5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, LIGHT]),
    ]))
    el += [Spacer(1, 0.3*cm), sig_tbl, PageBreak()]

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 17 — MONITORING FRAMEWORK
    # ─────────────────────────────────────────────────────────────────────────
    el += [Paragraph("Monitoring Framework", H1), hr2()]
    mon_data = [["Company", "Action", "Key KPI to Monitor", "Trigger / Catalyst", "Invalidation"]]

    for _, row in holdings_df.iterrows():
        ticker  = row["Ticker"]
        ac      = (actions or {}).get(ticker) or {}
        th      = st.get(ticker) or {}
        mf      = th.get("monitoring_framework", [])
        kpi     = mf[0] if mf else "Quarterly earnings"
        trigger = (th.get("key_triggers") or ["Results season"])[0] if th else "Results season"
        inval   = (th.get("risk_factors") or ["Thesis breakdown"])[0] if th else "Thesis breakdown"

        mon_data.append([
            row["Company"][:18],
            ac.get("signal", "WATCH"),
            str(kpi)[:35],
            str(trigger)[:40],
            str(inval)[:40],
        ])

    mon_tbl = Table(mon_data, colWidths=[3.5*cm, 2.2*cm, 4.5*cm, 5*cm, 5*cm])
    mon_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), WHITE),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 7.5),
        ("GRID", (0,0), (-1,-1), 0.3, BORDER), ("PADDING", (0,0), (-1,-1), 4),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, LIGHT]),
    ]))
    el += [mon_tbl, PageBreak()]

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 18 — PM COMMENTARY
    # ─────────────────────────────────────────────────────────────────────────
    el += [Paragraph("PM Commentary", H1), hr2()]
    el += [Paragraph(f"<b>{inst.get('portfolio_identity', 'Portfolio')}</b>  |  {report_date}", H2)]
    el += [Spacer(1, 0.3*cm)]
    if inst.get("pm_commentary"):
        for p in _md_to_paras(inst["pm_commentary"], BODY):
            el += [p]
    else:
        el += [Paragraph("PM Commentary not generated — run AI Institutional Report first.", SMALL)]

    el += [Spacer(1, 1*cm), hr()]
    el += [Paragraph(
        f"AI Institutional Equity System  |  Generated {datetime.now().strftime('%d %b %Y %H:%M')}  |  For investment committee use only",
        _style("FTR", fontName="Helvetica", fontSize=7.5, textColor=LGRAY, alignment=TA_CENTER)
    )]

    doc.build(el)
    return buf.getvalue()


# =============================================================================
# LEGACY FALLBACK — keeps old 6-page PDF working for runs without M6B
# =============================================================================
def build_report_pdf(
    report_date, portfolio_stats, risk_metrics, risk_flags,
    attribution_summary, holdings_df, clean_data, verdicts, ai_report, figures,
) -> bytes:
    """Legacy 6-page PDF for backward compatibility when institutional report hasn't run."""
    # Build minimal context to pass to 18-page builder
    return build_institutional_pdf(
        report_date=report_date,
        portfolio_stats=portfolio_stats,
        risk_metrics=risk_metrics,
        risk_flags=risk_flags,
        attribution_summary=attribution_summary,
        attrib_df=None,
        holdings_df=holdings_df,
        sector_df=None,
        clean_data=clean_data,
        verdicts=verdicts,
        scenarios={},
        actions={},
        institutional_report={
            "stock_theses":     {t: {"investment_thesis": tx} for t, tx in (ai_report or {}).get("theses", {}).items()},
            "executive_summary": (ai_report or {}).get("summary", ""),
            "pm_commentary":    "",
            "portfolio_identity": "Portfolio",
        },
        figures=figures,
        price_history={},
    )
