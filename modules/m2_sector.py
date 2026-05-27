# =============================================================================
# MODULE 2 — SECTOR ALLOCATION + PERFORMANCE CHARTS
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import yfinance as yf
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# Nifty 500 benchmark sector weights (normalized to 100%)
NIFTY500_RAW = {
    "Financials": 33.10,
    "Materials": 8.20,
    "Consumer Discretionary": 7.80,
    "Industrials": 7.50,
    "Information Technology": 12.40,
    "Consumer Staples": 7.20,
    "Energy": 9.80,
    "Healthcare": 5.90,
    "Communication Services": 2.30,
    "Real Estate": 1.80,
    "Utilities": 2.50,
    "ETF": 0.00,
}
_total = sum(NIFTY500_RAW.values())
NIFTY500 = {k: round(v / _total * 100, 2) for k, v in NIFTY500_RAW.items()}

# Chart color palette
C = {
    "bg": "#FFFFFF", "panel": "#F8F9FC", "border": "#DDE3EE",
    "text": "#0F172A", "text2": "#475569", "text3": "#94A3B8",
    "blue": "#1D4ED8", "blue_light": "#93C5FD", "green": "#16A34A",
    "red": "#DC2626", "gold": "#D97706", "grid": "#E2E8F0",
    "port_line": "#1D4ED8", "bench_line": "#6B7280",
    "alpha_pos": "#86EFAC", "alpha_neg": "#FCA5A5",
}


def aggregate_sectors(holdings: pd.DataFrame) -> pd.DataFrame:
    """Aggregate holdings by sector and merge with Nifty 500 benchmark weights."""
    sector_grouped = holdings.groupby("Sector").agg(
        Port_Weight=("Portfolio Wt%", "sum"),
        Total_Invested=("Invested Value", "sum"),
        Total_Current=("Current Value", "sum"),
        Total_PnL=("P&L", "sum"),
        Num_Stocks=("Ticker", "count"),
    ).reset_index()

    sector_grouped["Sector_Return%"] = (
        (sector_grouped["Total_Current"] - sector_grouped["Total_Invested"])
        / sector_grouped["Total_Invested"]
        * 100
    ).round(2)

    benchmark_df = pd.DataFrame(list(NIFTY500.items()), columns=["Sector", "Bench_Weight"])
    sector_df = sector_grouped.merge(benchmark_df, on="Sector", how="outer")
    sector_df["Port_Weight"] = sector_df["Port_Weight"].fillna(0)
    sector_df["Bench_Weight"] = sector_df["Bench_Weight"].fillna(0)
    sector_df["Num_Stocks"] = sector_df["Num_Stocks"].fillna(0).astype(int)
    sector_df["Active_Bet"] = (sector_df["Port_Weight"] - sector_df["Bench_Weight"]).round(2)
    sector_df = sector_df.sort_values("Active_Bet", ascending=False).reset_index(drop=True)
    return sector_df


def _style_ax(ax, grid_axis="y"):
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["text2"], labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C["border"])
    ax.spines["bottom"].set_color(C["border"])
    if grid_axis:
        ax.yaxis.grid(True, color=C["grid"], linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def plot_sector_overview(
    sector_df: pd.DataFrame,
    holdings: pd.DataFrame,
    report_date: str,
) -> plt.Figure:
    """
    Figure 1: 3-panel sector overview
      A — Portfolio weight by sector (bar)
      B — Active bets vs benchmark (horizontal bar)
      C — Top 5 / Bottom 5 holdings by return
    """
    fig = plt.figure(figsize=(18, 10), facecolor=C["bg"])
    fig.suptitle(
        f"Sector Allocation & Holdings Overview  |  {report_date}",
        fontsize=13, fontweight="bold", color=C["text"], y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # --- Panel A: Sector weights ---
    active = sector_df[(sector_df["Port_Weight"] > 0) | (sector_df["Bench_Weight"] > 0)].copy()
    labels = active["Sector"].str[:18].tolist()
    x = np.arange(len(labels))
    w = 0.35
    _style_ax(ax1)
    bars1 = ax1.bar(x - w / 2, active["Port_Weight"], w, label="Portfolio", color=C["blue"], alpha=0.85, zorder=3)
    bars2 = ax1.bar(x + w / 2, active["Bench_Weight"], w, label="Nifty 500", color=C["text3"], alpha=0.7, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax1.set_ylabel("Weight (%)", fontsize=8, color=C["text2"])
    ax1.set_title("Portfolio vs Benchmark Weights", fontsize=9, color=C["text"], pad=8)
    ax1.legend(fontsize=7, framealpha=0.5)

    # --- Panel B: Active bets ---
    _style_ax(ax2, grid_axis="x")
    ab_sorted = active.sort_values("Active_Bet")
    colors = [C["green"] if v >= 0 else C["red"] for v in ab_sorted["Active_Bet"]]
    ax2.barh(ab_sorted["Sector"].str[:18], ab_sorted["Active_Bet"], color=colors, alpha=0.85, zorder=3)
    ax2.axvline(0, color=C["text"], linewidth=0.8)
    ax2.set_xlabel("Active Bet (%)", fontsize=8, color=C["text2"])
    ax2.set_title("Active Bets vs Nifty 500", fontsize=9, color=C["text"], pad=8)
    ax2.xaxis.grid(True, color=C["grid"], linewidth=0.5, zorder=0)
    ax2.yaxis.grid(False)
    ax2.tick_params(axis="y", labelsize=7)

    # --- Panel C: Top 5 + Bottom 5 by return ---
    _style_ax(ax3)
    valid = holdings.dropna(subset=["Return %"])
    top5 = valid.nlargest(5, "Return %")
    bot5 = valid.nsmallest(5, "Return %")
    combined = pd.concat([top5, bot5]).drop_duplicates()
    combined = combined.sort_values("Return %")
    bar_colors = [C["green"] if v >= 0 else C["red"] for v in combined["Return %"]]
    ax3.barh(
        combined["Company"].str[:28],
        combined["Return %"],
        color=bar_colors, alpha=0.85, zorder=3,
    )
    ax3.axvline(0, color=C["text"], linewidth=0.8)
    for i, (_, row) in enumerate(combined.iterrows()):
        ax3.text(
            row["Return %"] + (0.5 if row["Return %"] >= 0 else -0.5),
            i, f"{row['Return %']:+.1f}%",
            va="center", ha="left" if row["Return %"] >= 0 else "right",
            fontsize=7, color=C["text"],
        )
    ax3.set_xlabel("Return (%)", fontsize=8, color=C["text2"])
    ax3.set_title("Top 5 / Bottom 5 Holdings by Return", fontsize=9, color=C["text"], pad=8)
    ax3.xaxis.grid(True, color=C["grid"], linewidth=0.5, zorder=0)
    ax3.yaxis.grid(False)
    ax3.tick_params(axis="y", labelsize=7.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_cumulative_performance(
    holdings: pd.DataFrame,
    analysis_date: datetime,
    period_years: int = 5,
) -> plt.Figure:
    """
    Figure 2: Portfolio vs Nifty 500 cumulative return over period_years.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor=C["bg"])
    fig.suptitle(
        f"{period_years}-Year Cumulative Return vs Benchmark",
        fontsize=12, fontweight="bold", color=C["text"],
    )

    start_str = (analysis_date - timedelta(days=365 * period_years)).strftime("%Y-%m-%d")
    end_str = analysis_date.strftime("%Y-%m-%d")

    # Fetch portfolio prices
    price_data = {}
    for _, row in holdings.iterrows():
        ticker = row["Ticker"]
        try:
            data = yf.download(ticker, start=start_str, end=end_str, progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if not data.empty and "Close" in data.columns:
                price_data[ticker] = data["Close"].dropna()
        except Exception:
            pass

    # Benchmark
    bench_prices = pd.Series(dtype=float)
    for bench in ["^CRSLDX", "^NSEI"]:
        try:
            bd = yf.download(bench, start=start_str, end=end_str, progress=False, auto_adjust=True)
            if isinstance(bd.columns, pd.MultiIndex):
                bd.columns = bd.columns.get_level_values(0)
            if not bd.empty:
                bench_prices = bd["Close"].dropna()
                break
        except Exception:
            pass

    ax1, ax2 = axes
    for ax in axes:
        _style_ax(ax)

    if price_data:
        prices_df = pd.DataFrame(price_data).dropna(how="all")
        weight_map = dict(zip(holdings["Ticker"], holdings["Portfolio Wt%"] / 100))
        port_series = pd.Series(0.0, index=prices_df.index)
        for ticker, wt in weight_map.items():
            if ticker in prices_df.columns:
                s = prices_df[ticker].dropna()
                if len(s) > 0:
                    port_series = port_series.add(((s / s.iloc[0]) - 1) * wt * 100, fill_value=0)
        port_cum = port_series.dropna()

        if len(bench_prices) > 1:
            bench_cum = ((bench_prices / bench_prices.iloc[0]) - 1) * 100
            common_idx = port_cum.index.intersection(bench_cum.index)
            if len(common_idx) > 1:
                port_plot = port_cum.loc[common_idx]
                bench_plot = bench_cum.loc[common_idx]
                ax1.plot(port_plot.index, port_plot, color=C["port_line"], lw=2, label="Portfolio")
                ax1.plot(bench_plot.index, bench_plot, color=C["bench_line"], lw=1.5, ls="--", label="Nifty 500/50")
                alpha = port_plot - bench_plot
                ax1.fill_between(alpha.index, 0, alpha, where=alpha >= 0, alpha=0.25, color=C["green"], label="Alpha (+)")
                ax1.fill_between(alpha.index, 0, alpha, where=alpha < 0, alpha=0.25, color=C["red"])
                ax1.axhline(0, color=C["border"], lw=0.8)
                ax1.set_ylabel("Cumulative Return (%)", fontsize=8)
                ax1.legend(fontsize=8)
                ax1.set_title("Cumulative Return", fontsize=9, color=C["text"])

                ax2.fill_between(alpha.index, 0, alpha, where=alpha >= 0, color=C["alpha_pos"], alpha=0.8, label="Positive alpha")
                ax2.fill_between(alpha.index, 0, alpha, where=alpha < 0, color=C["alpha_neg"], alpha=0.8, label="Negative alpha")
                ax2.axhline(0, color=C["text"], lw=1)
                ax2.set_ylabel("Alpha vs Benchmark (%)", fontsize=8)
                ax2.legend(fontsize=8)
                ax2.set_title("Rolling Alpha (Portfolio − Benchmark)", fontsize=9, color=C["text"])
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "Price data unavailable", transform=ax.transAxes,
                    ha="center", va="center", color=C["text3"])

    plt.tight_layout()
    return fig
