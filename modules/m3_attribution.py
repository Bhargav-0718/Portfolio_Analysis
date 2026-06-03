# =============================================================================
# MODULE 3 — BRINSON-HOOD-BEEBOWER ATTRIBUTION ANALYSIS
# Exact replication of Final_Project_Code.ipynb Module 3
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

C = {
    "bg":     "#FFFFFF", "panel":  "#F8F9FC", "border": "#DDE3EE",
    "text":   "#0F172A", "text2":  "#475569", "text3":  "#94A3B8",
    "blue":   "#1D4ED8", "green":  "#16A34A", "red":    "#DC2626",
    "gold":   "#D97706", "purple": "#7C3AED", "grid":   "#E2E8F0",
}

# Sector → NSE index proxy (same as notebook SECTOR_PROXIES)
SECTOR_PROXIES = {
    "Financials":             "^NSEBANK",
    "Materials":              "^CNXMETAL",
    "Consumer Discretionary": "^CNXAUTO",
    "Industrials":            "^CNXINFRA",
    "Information Technology": "^CNXIT",
    "Consumer Staples":       "^CNXFMCG",
    "Energy":                 "^CNXENERGY",
    "Healthcare":             "^CNXPHARMA",
    "Communication Services": "^CNXMEDIA",
    "Real Estate":            "^CNXREALTY",
    "Utilities":              "^CNXENERGY",
    "ETF":                    "^NSEI",
}

ATTR_LOOKBACK_YEARS = 2


def fetch_sector_benchmark_returns(
    sectors: list,
    analysis_date: datetime,
    lookback_years: int = ATTR_LOOKBACK_YEARS,
) -> tuple:
    """
    Fetch 2-year sector index returns + total benchmark return.
    Returns (sector_bench_returns dict, total_bench_return float).

    Fallback: sector uses total_bench_return if proxy unavailable (matches notebook).
    """
    attr_start = (analysis_date - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")
    attr_end   = analysis_date.strftime("%Y-%m-%d")

    # Total benchmark first (Nifty 500, fallback Nifty 50)
    total_bench_return = 18.0   # last-resort default
    for bench in ["^CRSLDX", "^NSEI"]:
        try:
            bd = yf.download(bench, start=attr_start, end=attr_end,
                             progress=False, auto_adjust=True)
            if isinstance(bd.columns, pd.MultiIndex):
                bd.columns = bd.columns.get_level_values(0)
            if not bd.empty and "Close" in bd.columns:
                bc = bd["Close"].dropna()
                total_bench_return = round(float((bc.iloc[-1] / bc.iloc[0] - 1) * 100), 2)
                break
        except Exception:
            pass

    # Sector proxies — fall back to total_bench_return (matches notebook behaviour)
    sector_bench_returns = {}
    for sector in sectors:
        proxy = SECTOR_PROXIES.get(sector)
        if not proxy:
            sector_bench_returns[sector] = total_bench_return
            continue
        try:
            data = yf.download(proxy, start=attr_start, end=attr_end,
                               progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if not data.empty and "Close" in data.columns:
                closes = data["Close"].dropna()
                ret    = (closes.iloc[-1] / closes.iloc[0] - 1) * 100
                sector_bench_returns[sector] = round(float(ret), 2)
            else:
                sector_bench_returns[sector] = total_bench_return   # ← notebook fallback
        except Exception:
            sector_bench_returns[sector] = total_bench_return       # ← notebook fallback

    return sector_bench_returns, total_bench_return


def bhb_attribution(
    holdings: pd.DataFrame,
    sector_df: pd.DataFrame,
    sector_bench_returns: dict,
    total_bench_return: float,
) -> tuple:
    """
    Brinson-Fachler Attribution — exact notebook logic.

    Key notebook fixes applied:
    1. Sector portfolio return = weight-avg of stock returns in that sector
    2. Interaction effect = 0.0 (Brinson-Fachler simplification)
    3. Reconciliation scaling to force BHB totals to match actual active return
    4. Bench return fallback = total_bench_return (not 0)

    Returns (attrib_df, summary_dict, holdings_with_contribution)
    """
    # ── Total portfolio return (notebook formula) ─────────────────────────
    total_port_return = (
        holdings["P&L"].sum() / holdings["Invested Value"].sum()
    ) * 100
    active_return = total_port_return - total_bench_return

    # ── Sector portfolio return: weight-avg of stock returns (notebook fix) ─
    sector_port_returns = (
        holdings.groupby("Sector")
        .apply(lambda x: np.average(x["Return %"], weights=x["Portfolio Wt%"]))
        .to_dict()
    )

    # ── BHB loop (exact notebook) ─────────────────────────────────────────
    attribution_rows = []

    for _, row in sector_df.iterrows():
        sector   = row["Sector"]
        port_wt  = row["Port_Weight"] / 100
        bench_wt = row["Bench_Weight"] / 100

        port_ret  = sector_port_returns.get(sector, 0)
        bench_ret = sector_bench_returns.get(sector, total_bench_return)

        # Safety fallback (notebook: if port_ret is NaN use bench_ret)
        if pd.isna(port_ret):
            port_ret = bench_ret

        # Brinson-Fachler Attribution (interaction = 0, same as notebook)
        allocation_effect  = (port_wt - bench_wt) * (bench_ret - total_bench_return)
        selection_effect   = port_wt * (port_ret - bench_ret)
        interaction_effect = 0.0
        total_alpha        = allocation_effect + selection_effect

        attribution_rows.append({
            "Sector":       sector,
            "Port_Wt%":     round(port_wt * 100, 2),
            "Bench_Wt%":    round(bench_wt * 100, 2),
            "Active_Bet%":  round((port_wt - bench_wt) * 100, 2),
            "Port_Return%": round(port_ret, 2),
            "Bench_Return%":round(bench_ret, 2),
            "Return_Diff%": round(port_ret - bench_ret, 2),
            "Allocation":   round(allocation_effect,  6),
            "Selection":    round(selection_effect,   6),
            "Interaction":  round(interaction_effect, 6),
            "Total_Alpha":  round(total_alpha, 6),
        })

    attrib_df = pd.DataFrame(attribution_rows)
    attrib_df = attrib_df.sort_values("Total_Alpha", ascending=False).reset_index(drop=True)

    # ── Reconciliation scaling (notebook: force totals to match active return) ─
    explained_total = attrib_df["Total_Alpha"].sum()
    if abs(explained_total) > 1e-6:
        scaling_factor = active_return / explained_total
        attrib_df["Allocation"]  *= scaling_factor
        attrib_df["Selection"]   *= scaling_factor
        attrib_df["Total_Alpha"] *= scaling_factor

    explained_total      = attrib_df["Total_Alpha"].sum()
    reconciliation_error = active_return - explained_total
    total_allocation     = attrib_df["Allocation"].sum()
    total_selection      = attrib_df["Selection"].sum()
    total_interaction    = attrib_df["Interaction"].sum()

    # ── Stock-level contribution (notebook formula) ────────────────────────
    holdings = holdings.copy()
    holdings["Contribution%"] = (
        (holdings["Portfolio Wt%"] / 100) * holdings["Return %"]
    ).round(4)

    summary = {
        "total_port_return":   round(total_port_return, 2),
        "total_bench_return":  round(total_bench_return, 2),
        "total_alpha":         round(total_port_return - total_bench_return, 2),
        "active_return":       round(active_return, 4),
        "allocation_effect":   round(total_allocation, 4),
        "selection_effect":    round(total_selection, 4),
        "interaction_effect":  round(total_interaction, 4),
        "total_explained":     round(explained_total, 4),
        "reconciliation_error":round(reconciliation_error, 6),
    }

    return attrib_df, summary, holdings


def plot_attribution_charts(
    attrib_df: pd.DataFrame,
    holdings: pd.DataFrame,
    summary: dict,
    report_date: str,
) -> plt.Figure:
    """
    3-panel combined figure (platform version of notebook's fig4 + fig5 + fig6).
    Exactly matches notebook styling and data selection.
    """

    def style_ax(ax):
        ax.set_facecolor(C["panel"])
        ax.tick_params(labelsize=8.5, colors=C["text2"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(C["border"])
        ax.spines["bottom"].set_color(C["border"])
        ax.grid(axis="x", color=C["grid"], linewidth=0.6, linestyle="--", zorder=0)
        ax.set_axisbelow(True)

    # Notebook filter: show sectors with portfolio weight OR benchmark weight > 0.5%
    plot_df = attrib_df[
        (attrib_df["Port_Wt%"] > 0) | (attrib_df["Bench_Wt%"] > 0.5)
    ].copy().sort_values("Total_Alpha", ascending=True)

    fig, axes = plt.subplots(3, 1, figsize=(16, 22), facecolor=C["bg"])
    fig.suptitle(f"Attribution Analysis  |  {report_date}",
                 fontsize=13, fontweight="bold", color=C["text"], y=0.995)

    # ── FIGURE 4 — Total Attribution by Sector ────────────────────────────
    ax1 = axes[0]
    style_ax(ax1)
    bar_colors = [C["green"] if v >= 0 else C["red"] for v in plot_df["Total_Alpha"]]
    bars = ax1.barh(plot_df["Sector"], plot_df["Total_Alpha"],
                    color=bar_colors, alpha=0.85, height=0.6, zorder=3)

    # Data labels (notebook style: +0.4272%)
    for bar, val in zip(bars, plot_df["Total_Alpha"]):
        offset = 0.003 if val >= 0 else -0.003
        ax1.text(bar.get_width() + offset,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:+.4f}%", va="center",
                 ha="left" if val >= 0 else "right",
                 fontsize=8.5, color=C["text"], fontweight="bold")

    ax1.axvline(x=0, color=C["text2"], linewidth=1.2, zorder=4)
    ax1.set_title(f"Total Attribution by Sector  |  {report_date}",
                  fontsize=11, fontweight="bold", color=C["text"], pad=14)
    ax1.set_xlabel("Total Alpha (%)", fontsize=9, color=C["text2"])

    # Summary box (notebook-exact text)
    summary_text = (
        f"Portfolio Return:  {summary['total_port_return']:+.2f}%\n"
        f"Benchmark Return:  {summary['total_bench_return']:+.2f}%\n"
        f"Active Return:     {summary['total_alpha']:+.2f}%\n"
        f"Allocation Effect: {summary['allocation_effect']:+.4f}%\n"
        f"Selection Effect:  {summary['selection_effect']:+.4f}%"
    )
    ax1.text(0.98, 0.05, summary_text, transform=ax1.transAxes, fontsize=8.5,
             va="bottom", ha="right", color=C["text"],
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor=C["border"], alpha=0.9))
    ax1.legend(handles=[
        mpatches.Patch(color=C["green"], alpha=0.85, label="Positive Alpha"),
        mpatches.Patch(color=C["red"],   alpha=0.85, label="Negative Alpha"),
    ], fontsize=8, facecolor=C["bg"], edgecolor=C["border"], labelcolor=C["text"])

    # ── FIGURE 5 — Allocation vs Selection vs Interaction ─────────────────
    ax2 = axes[1]
    style_ax(ax2)
    plot_df2 = attrib_df[
        (attrib_df["Port_Wt%"] > 0) | (attrib_df["Bench_Wt%"] > 0.5)
    ].copy().sort_values("Total_Alpha", ascending=True)

    y = np.arange(len(plot_df2))
    h = 0.28

    b1 = ax2.barh(y + h, plot_df2["Allocation"],  h,
                  color=C["blue"],   alpha=0.85, label="Allocation Effect",  zorder=3)
    b2 = ax2.barh(y,     plot_df2["Selection"],   h,
                  color=C["gold"],   alpha=0.85, label="Selection Effect",   zorder=3)
    b3 = ax2.barh(y - h, plot_df2["Interaction"], h,
                  color=C["purple"], alpha=0.70, label="Interaction Effect", zorder=3)

    # Data labels on allocation bars
    for bar in b1:
        val = bar.get_width()
        if abs(val) > 0.0005:
            ax2.text(val + (0.002 if val >= 0 else -0.002),
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:+.3f}", va="center",
                     ha="left" if val >= 0 else "right",
                     fontsize=7, color=C["blue"])

    # Data labels on selection bars
    for bar in b2:
        val = bar.get_width()
        if abs(val) > 0.0005:
            ax2.text(val + (0.002 if val >= 0 else -0.002),
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:+.3f}", va="center",
                     ha="left" if val >= 0 else "right",
                     fontsize=7, color=C["gold"])

    ax2.set_yticks(y)
    ax2.set_yticklabels(plot_df2["Sector"], fontsize=9)
    ax2.axvline(x=0, color=C["text2"], linewidth=1.2, zorder=4)
    ax2.set_title(f"Attribution Decomposition: Allocation vs Selection  |  {report_date}",
                  fontsize=11, fontweight="bold", color=C["text"], pad=14)
    ax2.set_xlabel("Attribution Effect (%)", fontsize=9, color=C["text2"])
    ax2.legend(fontsize=9, facecolor=C["bg"], edgecolor=C["border"], labelcolor=C["text"])

    # ── FIGURE 6 — Stock-Level Contributors / Detractors ──────────────────
    ax3 = axes[2]
    style_ax(ax3)

    if "Contribution%" in holdings.columns:
        all_stocks = holdings.copy()
        all_stocks["Short"] = all_stocks["Company"].str[:20]

        # Top 8 contributors + bottom 8 detractors (notebook: top/bottom 8)
        top8 = all_stocks.nlargest(8, "Contribution%").sort_values("Contribution%", ascending=True)
        bot8 = all_stocks.nsmallest(8, "Contribution%").sort_values("Contribution%", ascending=False)

        # Combined in one plot (top green on right, bottom red on left)
        combined = pd.concat([bot8, top8]).drop_duplicates()
        combined = combined.sort_values("Contribution%", ascending=True)
        bar_colors_c = [C["green"] if v >= 0 else C["red"]
                        for v in combined["Contribution%"]]

        bars3 = ax3.barh(combined["Short"], combined["Contribution%"],
                         color=bar_colors_c, alpha=0.85, height=0.6, zorder=3)

        # Data labels (notebook style: +61.60%  (ret: +178.3%))
        for bar, (_, row) in zip(bars3, combined.iterrows()):
            val = row["Contribution%"]
            offset = 0.01 if val >= 0 else -0.01
            ax3.text(val + offset,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:+.2f}%  (ret: {row['Return %']:+.1f}%)",
                     va="center", ha="left" if val >= 0 else "right",
                     fontsize=8, color=C["text"])

        ax3.axvline(x=0, color=C["text2"], linewidth=1.0, zorder=4)
        ax3.set_title(f"Stock-Level Return Attribution  |  {report_date}",
                      fontsize=11, fontweight="bold", color=C["text"], pad=14)
        ax3.set_xlabel("Contribution to Portfolio Return (%)", fontsize=9, color=C["text2"])
        ax3.tick_params(axis="y", labelsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.995], pad=2.0)
    return fig
