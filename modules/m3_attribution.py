# =============================================================================
# MODULE 3 — BRINSON-HOOD-BEEBOWER ATTRIBUTION ANALYSIS
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

C = {
    "bg": "#FFFFFF", "panel": "#F8F9FC", "border": "#DDE3EE",
    "text": "#0F172A", "text2": "#475569", "text3": "#94A3B8",
    "blue": "#1D4ED8", "green": "#16A34A", "red": "#DC2626",
    "gold": "#D97706", "grid": "#E2E8F0",
}

# Best sector proxies available on NSE/yfinance
SECTOR_PROXIES = {
    "Financials":             "^CNXFIN",
    "Materials":              "^CNXMETAL",
    "Consumer Discretionary": "^CNXAUTO",
    "Industrials":            "^CNXINFRA",
    "Information Technology": "^CNXIT",
    "Consumer Staples":       "^CNXFMCG",
    "Energy":                 "^CNXENERGY",
    "Healthcare":             "^CNXPHARMA",
    "Communication Services": "^CNXMEDIA",
    "Utilities":              "^CNXENERGY",
    "ETF":                    None,
}


def fetch_sector_benchmark_returns(
    sectors: list,
    analysis_date: datetime,
    lookback_years: int = 3,
) -> tuple:
    """
    Fetch returns for each sector proxy and total benchmark over lookback period.
    Returns (sector_returns dict, total_bench_return float).
    """
    start = (analysis_date - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")
    end = analysis_date.strftime("%Y-%m-%d")

    sector_returns = {}
    for sector in sectors:
        proxy = SECTOR_PROXIES.get(sector)
        if not proxy:
            sector_returns[sector] = 0.0
            continue
        try:
            data = yf.download(proxy, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if not data.empty and "Close" in data.columns:
                closes = data["Close"].dropna()
                if len(closes) > 1:
                    ret = (closes.iloc[-1] / closes.iloc[0] - 1) * 100
                    sector_returns[sector] = round(float(ret), 2)
                    continue
        except Exception:
            pass
        sector_returns[sector] = 0.0

    # Total benchmark
    total_bench = 0.0
    for bench in ["^CRSLDX", "^NSEI"]:
        try:
            bd = yf.download(bench, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(bd.columns, pd.MultiIndex):
                bd.columns = bd.columns.get_level_values(0)
            if not bd.empty:
                closes = bd["Close"].dropna()
                if len(closes) > 1:
                    total_bench = round(float((closes.iloc[-1] / closes.iloc[0] - 1) * 100), 2)
                    break
        except Exception:
            pass

    return sector_returns, total_bench


def bhb_attribution(
    holdings: pd.DataFrame,
    sector_df: pd.DataFrame,
    sector_bench_returns: dict,
    total_bench_return: float,
) -> tuple:
    """
    Compute Brinson-Hood-Beebower attribution by sector.
    Returns (attrib_df, summary dict).
    """
    # Total portfolio return (weighted)
    total_port_return = (
        holdings["Return %"] * holdings["Portfolio Wt%"] / 100
    ).sum()

    rows = []
    for _, row in sector_df.iterrows():
        sector = row["Sector"]
        if row["Port_Weight"] == 0 and row["Bench_Weight"] == 0:
            continue

        pw = row["Port_Weight"] / 100
        bw = row["Bench_Weight"] / 100
        pr = row.get("Sector_Return%", 0) or 0
        br = sector_bench_returns.get(sector, 0) or 0
        tb = total_bench_return or 0

        allocation = (pw - bw) * (br - tb)
        selection = pw * (pr - br)
        interaction = (pw - bw) * (pr - br)
        active = allocation + selection + interaction

        rows.append({
            "Sector": sector,
            "Port_Wt%": row["Port_Weight"],
            "Bench_Wt%": row["Bench_Weight"],
            "Active_Bet%": row["Active_Bet"],
            "Port_Return%": pr,
            "Bench_Return%": br,
            "Allocation": round(allocation, 4),
            "Selection": round(selection, 4),
            "Interaction": round(interaction, 4),
            "Active_Return": round(active, 4),
        })

    attrib_df = pd.DataFrame(rows)

    # Stock-level contribution
    holdings = holdings.copy()
    holdings["Contribution%"] = (
        holdings["Return %"] * holdings["Portfolio Wt%"] / 100
    ).round(4)

    summary = {
        "total_port_return": round(total_port_return, 2),
        "total_bench_return": round(total_bench_return, 2),
        "total_alpha": round(total_port_return - total_bench_return, 2),
        "allocation_effect": round(attrib_df["Allocation"].sum(), 4),
        "selection_effect": round(attrib_df["Selection"].sum(), 4),
        "interaction_effect": round(attrib_df["Interaction"].sum(), 4),
        "total_explained": round(attrib_df["Active_Return"].sum(), 4),
    }

    return attrib_df, summary, holdings


def plot_attribution_charts(
    attrib_df: pd.DataFrame,
    holdings: pd.DataFrame,
    summary: dict,
    report_date: str,
) -> plt.Figure:
    """
    Figure 4+5+6: Attribution waterfall, decomposition, stock contributors.
    """
    fig = plt.figure(figsize=(18, 14), facecolor=C["bg"])
    fig.suptitle(
        f"Brinson-Hood-Beebower Attribution  |  {report_date}",
        fontsize=13, fontweight="bold", color=C["text"], y=0.99,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])   # Waterfall
    ax2 = fig.add_subplot(gs[0, 1])   # Decomposition
    ax3 = fig.add_subplot(gs[1, :])   # Stock contributors

    def style_ax(ax):
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["text2"], labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(C["border"])
        ax.spines["bottom"].set_color(C["border"])
        ax.set_axisbelow(True)

    # --- Panel 1: Total active return by sector ---
    style_ax(ax1)
    if not attrib_df.empty:
        df1 = attrib_df[attrib_df["Port_Wt%"] > 0].sort_values("Active_Return")
        colors = [C["green"] if v >= 0 else C["red"] for v in df1["Active_Return"]]
        ax1.barh(df1["Sector"].str[:18], df1["Active_Return"], color=colors, alpha=0.85, zorder=3)
        ax1.axvline(0, color=C["text"], lw=0.8)
        ax1.xaxis.grid(True, color=C["grid"], lw=0.5, zorder=0)
        ax1.yaxis.grid(False)
        ax1.tick_params(axis="y", labelsize=7)
        ax1.set_title("Total Active Return by Sector", fontsize=9, color=C["text"])
        ax1.set_xlabel("Active Return (%)", fontsize=8)
        # Summary box
        tb = f"Allocation: {summary['allocation_effect']:+.2f}%\nSelection:  {summary['selection_effect']:+.2f}%\nTotal Alpha:{summary['total_alpha']:+.2f}%"
        ax1.text(0.97, 0.05, tb, transform=ax1.transAxes, fontsize=7,
                 ha="right", va="bottom", color=C["text"],
                 bbox=dict(boxstyle="round,pad=0.4", facecolor=C["panel"], edgecolor=C["border"]))

    # --- Panel 2: Allocation vs Selection per sector ---
    style_ax(ax2)
    if not attrib_df.empty:
        df2 = attrib_df[attrib_df["Port_Wt%"] > 0]
        x = np.arange(len(df2))
        w = 0.35
        ax2.bar(x - w / 2, df2["Allocation"], w, label="Allocation", color=C["blue"], alpha=0.8, zorder=3)
        ax2.bar(x + w / 2, df2["Selection"], w, label="Selection", color=C["gold"], alpha=0.8, zorder=3)
        ax2.set_xticks(x)
        ax2.set_xticklabels(df2["Sector"].str[:12], rotation=40, ha="right", fontsize=7)
        ax2.axhline(0, color=C["text"], lw=0.8)
        ax2.yaxis.grid(True, color=C["grid"], lw=0.5, zorder=0)
        ax2.legend(fontsize=8)
        ax2.set_title("Allocation vs Selection Effect", fontsize=9, color=C["text"])
        ax2.set_ylabel("Return Attribution (%)", fontsize=8)

    # --- Panel 3: Stock contributors ---
    style_ax(ax3)
    contribs = holdings[["Company", "Sector", "Portfolio Wt%", "Return %", "Contribution%"]].copy()
    contribs = contribs.sort_values("Contribution%")
    top = contribs.tail(5)
    bot = contribs.head(5)
    combined = pd.concat([bot, top]).drop_duplicates()
    bar_colors = [C["green"] if v >= 0 else C["red"] for v in combined["Contribution%"]]
    ax3.barh(combined["Company"].str[:28], combined["Contribution%"], color=bar_colors, alpha=0.85, zorder=3)
    ax3.axvline(0, color=C["text"], lw=0.8)
    ax3.xaxis.grid(True, color=C["grid"], lw=0.5, zorder=0)
    ax3.yaxis.grid(False)
    ax3.tick_params(axis="y", labelsize=7.5)
    ax3.set_title("Top 5 Contributors / Detractors (by return contribution)", fontsize=9, color=C["text"])
    ax3.set_xlabel("Contribution to Portfolio Return (%)", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig
