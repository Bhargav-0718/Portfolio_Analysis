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
    3-panel attribution figure matching notebook output:
    Fig 4: Total alpha by sector (waterfall horizontal)
    Fig 5: Attribution Decomposition — Allocation vs Selection vs Interaction (horizontal grouped)
    Fig 6: Stock-level contributors / detractors (horizontal)
    """
    fig = plt.figure(figsize=(18, 20), facecolor=C["bg"])
    fig.suptitle(
        f"Brinson-Hood-Beebower Attribution  |  {report_date}",
        fontsize=13, fontweight="bold", color=C["text"], y=0.99,
    )
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55)
    ax1 = fig.add_subplot(gs[0])   # Fig 4 — Total alpha waterfall
    ax2 = fig.add_subplot(gs[1])   # Fig 5 — Decomposition (Allocation + Selection + Interaction)
    ax3 = fig.add_subplot(gs[2])   # Fig 6 — Stock contributors / detractors

    def style_ax(ax):
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["text2"], labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(C["border"])
        ax.spines["bottom"].set_color(C["border"])
        ax.set_axisbelow(True)

    # ── Figure 4 — Total Attribution by Sector (waterfall) ────────────────────
    style_ax(ax1)
    if not attrib_df.empty:
        # All sectors sorted by Active_Return
        df1 = attrib_df.sort_values("Active_Return")
        colors = [C["green"] if v >= 0 else C["red"] for v in df1["Active_Return"]]
        bars = ax1.barh(df1["Sector"].str[:20], df1["Active_Return"], color=colors, alpha=0.85, zorder=3)
        # Data labels
        for bar, val in zip(bars, df1["Active_Return"]):
            ax1.text(
                val + (0.05 if val >= 0 else -0.05), bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}%", va="center", ha="left" if val >= 0 else "right",
                fontsize=7, color=C["text"],
            )
        ax1.axvline(0, color=C["text"], lw=0.8)
        ax1.xaxis.grid(True, color=C["grid"], lw=0.5, zorder=0)
        ax1.yaxis.grid(False)
        ax1.tick_params(axis="y", labelsize=8)
        ax1.set_title(f"Total Attribution by Sector  |  {report_date}", fontsize=10, color=C["text"])
        ax1.set_xlabel("Total Alpha (%)", fontsize=9)
        # Summary inset
        tb = (f"Portfolio Return: {summary['total_port_return']:+.2f}%\n"
              f"Benchmark Return: {summary['total_bench_return']:+.2f}%\n"
              f"Active Return:    {summary['total_alpha']:+.2f}%\n"
              f"Allocation Effect: {summary['allocation_effect']:+.4f}%\n"
              f"Selection Effect:  {summary['selection_effect']:+.4f}%")
        ax1.text(0.98, 0.05, tb, transform=ax1.transAxes, fontsize=7.5,
                 ha="right", va="bottom", color=C["text"],
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=C["border"], alpha=0.9))
        # Legend patches
        from matplotlib.patches import Patch
        ax1.legend(handles=[Patch(color=C["green"], label="Positive Alpha"),
                             Patch(color=C["red"],   label="Negative Alpha")],
                   fontsize=8, loc="lower right")

    # ── Figure 5 — Attribution Decomposition: Allocation vs Selection vs Interaction ──
    style_ax(ax2)
    if not attrib_df.empty:
        # All sectors — sorted by allocation effect ascending (matches notebook)
        df2 = attrib_df.sort_values("Allocation")
        n   = len(df2)
        y   = np.arange(n)
        h   = 0.25  # bar height for each component

        PURPLE = "#7C3AED"

        ax2.barh(y + h,     df2["Allocation"],  h, label="Allocation Effect",  color=C["blue"],  alpha=0.85, zorder=3)
        ax2.barh(y,         df2["Selection"],   h, label="Selection Effect",   color=C["gold"],  alpha=0.85, zorder=3)
        ax2.barh(y - h,     df2["Interaction"], h, label="Interaction Effect", color=PURPLE,     alpha=0.85, zorder=3)

        ax2.set_yticks(y)
        ax2.set_yticklabels(df2["Sector"].str[:20], fontsize=8)
        ax2.axvline(0, color=C["text"], lw=0.8)
        ax2.xaxis.grid(True, color=C["grid"], lw=0.5, zorder=0)
        ax2.yaxis.grid(False)
        ax2.legend(fontsize=8, loc="lower right")
        ax2.set_title(f"Attribution Decomposition: Allocation vs Selection  |  {report_date}", fontsize=10, color=C["text"])
        ax2.set_xlabel("Attribution Effect (%)", fontsize=9)

    # ── Figure 6 — Stock-Level Return Attribution ─────────────────────────────
    style_ax(ax3)
    if "Contribution%" in holdings.columns:
        contribs = holdings[["Company", "Portfolio Wt%", "Return %", "Contribution%"]].copy().dropna(subset=["Contribution%"])
        contribs = contribs.sort_values("Contribution%")

        # Top 8 contributors + bottom 8 detractors (matching notebook)
        top8 = contribs.tail(8)
        bot8 = contribs.head(8)
        combined = pd.concat([bot8, top8]).drop_duplicates()

        bar_colors = [C["green"] if v >= 0 else C["red"] for v in combined["Contribution%"]]
        bars = ax3.barh(
            combined["Company"].str[:28] + "  (wt:" + combined["Portfolio Wt%"].map("{:.1f}%".format) +
            "  ret:" + combined["Return %"].map("{:+.1f}%".format) + ")",
            combined["Contribution%"], color=bar_colors, alpha=0.85, zorder=3,
        )
        ax3.axvline(0, color=C["text"], lw=0.8)
        ax3.xaxis.grid(True, color=C["grid"], lw=0.5, zorder=0)
        ax3.yaxis.grid(False)
        ax3.tick_params(axis="y", labelsize=7.5)
        ax3.set_title(f"Stock-Level Return Attribution  |  {report_date}", fontsize=10, color=C["text"])
        ax3.set_xlabel("Contribution to Portfolio Return (%)", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig
