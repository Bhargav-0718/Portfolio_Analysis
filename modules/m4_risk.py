# =============================================================================
# MODULE 4 — PORTFOLIO RISK ANALYTICS
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
    "blue": "#1D4ED8", "blue_light": "#BFDBFE", "green": "#16A34A",
    "red": "#DC2626", "gold": "#D97706", "grid": "#E2E8F0",
}

CORR_YEARS = 2


def calculate_risk_metrics(
    holdings: pd.DataFrame,
    sector_df: pd.DataFrame,
) -> dict:
    """Compute active share, HHI, effective N, hit rate, and other risk metrics."""
    weights = holdings["Portfolio Wt%"].values / 100
    bench_weights = holdings["Benchmark Wt%"].values / 100

    # Active Share
    active_share = 0.5 * np.abs(weights - bench_weights).sum() * 100

    # HHI
    hhi = np.sum(weights ** 2) * 10000

    # Effective N
    effective_n = 1 / np.sum(weights ** 2) if np.sum(weights ** 2) > 0 else len(holdings)

    # Sector HHI
    sector_weights = sector_df[sector_df["Port_Weight"] > 0]["Port_Weight"].values / 100
    sector_hhi = np.sum(sector_weights ** 2) * 10000
    eff_sectors = 1 / np.sum(sector_weights ** 2) if np.sum(sector_weights ** 2) > 0 else 1

    # Concentration
    sorted_w = np.sort(weights)[::-1]
    top1_wt = sorted_w[0] * 100 if len(sorted_w) >= 1 else 0
    top3_wt = sorted_w[:3].sum() * 100 if len(sorted_w) >= 3 else sorted_w.sum() * 100
    top5_wt = sorted_w[:5].sum() * 100 if len(sorted_w) >= 5 else sorted_w.sum() * 100
    top10_wt = sorted_w[:10].sum() * 100 if len(sorted_w) >= 10 else sorted_w.sum() * 100

    # Hit rate + win/loss
    valid = holdings.dropna(subset=["Return %"])
    winners = valid[valid["P&L"] > 0]
    losers = valid[valid["P&L"] < 0]
    hit_rate = len(winners) / len(valid) * 100 if len(valid) > 0 else 0
    avg_win = winners["Return %"].mean() if len(winners) > 0 else 0
    avg_loss = losers["Return %"].mean() if len(losers) > 0 else 0
    win_loss = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # Largest position
    largest = holdings.loc[holdings["Portfolio Wt%"].idxmax()]
    largest_pnl_impact = (
        largest["P&L"] / holdings["Invested Value"].sum() * 100
    )

    # Top sector
    top_sector = sector_df[sector_df["Port_Weight"] > 0].sort_values("Port_Weight", ascending=False).iloc[0]

    return {
        "active_share": round(active_share, 1),
        "hhi": round(hhi, 0),
        "effective_n": round(effective_n, 1),
        "effective_sectors": round(eff_sectors, 1),
        "sector_hhi": round(sector_hhi, 0),
        "top1_wt": round(top1_wt, 1),
        "top3_wt": round(top3_wt, 1),
        "top5_wt": round(top5_wt, 1),
        "top10_wt": round(top10_wt, 1),
        "hit_rate": round(hit_rate, 1),
        "avg_win": round(avg_win, 1),
        "avg_loss": round(avg_loss, 1),
        "win_loss_ratio": round(win_loss, 2) if win_loss != float("inf") else 99.0,
        "largest_company": largest["Company"],
        "largest_wt": round(largest["Portfolio Wt%"], 1),
        "largest_pnl_impact": round(largest_pnl_impact, 2),
        "top_sector": top_sector["Sector"],
        "top_sector_wt": round(top_sector["Port_Weight"], 1),
    }


def compute_return_based_metrics(
    holdings: pd.DataFrame,
    corr_returns: pd.DataFrame,
    analysis_date: datetime,
    risk_free_rate: float = 6.0,
) -> dict:
    """
    Compute Beta, Sharpe Ratio, Sortino Ratio, Jensen Alpha, Max Drawdown.
    Uses 2-year daily returns. risk_free_rate in % annually.
    """
    result = {
        "beta": None, "sharpe": None, "sortino": None,
        "jensen_alpha": None, "max_drawdown": None,
        "annualised_vol": None, "portfolio_ann_return": None,
    }

    if corr_returns.empty or len(corr_returns.columns) < 2:
        return result

    try:
        # Build weighted portfolio daily returns
        weight_map = {
            row["Company"][:15]: row["Portfolio Wt%"] / 100
            for _, row in holdings.iterrows()
        }
        port_daily = pd.Series(0.0, index=corr_returns.index)
        for col in corr_returns.columns:
            wt = weight_map.get(col, 0)
            port_daily = port_daily.add(corr_returns[col] * wt, fill_value=0)
        port_daily = port_daily.dropna()

        if len(port_daily) < 30:
            return result

        # Fetch benchmark returns for same period
        start = (analysis_date - timedelta(days=365 * CORR_YEARS)).strftime("%Y-%m-%d")
        end   = analysis_date.strftime("%Y-%m-%d")
        bench_daily = pd.Series(dtype=float)
        for bench in ["^CRSLDX", "^NSEI"]:
            try:
                bd = yf.download(bench, start=start, end=end, progress=False, auto_adjust=True)
                if isinstance(bd.columns, pd.MultiIndex):
                    bd.columns = bd.columns.get_level_values(0)
                if not bd.empty and "Close" in bd.columns:
                    bench_daily = bd["Close"].pct_change().dropna()
                    break
            except Exception:
                pass

        # Annualised stats
        ann_factor   = 252
        rf_daily     = risk_free_rate / 100 / ann_factor
        port_ann_ret = float(port_daily.mean() * ann_factor * 100)
        port_vol     = float(port_daily.std() * np.sqrt(ann_factor) * 100)

        # Sharpe
        excess_daily = port_daily - rf_daily
        sharpe = float(excess_daily.mean() / port_daily.std() * np.sqrt(ann_factor)) if port_daily.std() > 0 else None

        # Sortino (downside deviation only)
        downside = port_daily[port_daily < rf_daily]
        sortino = None
        if len(downside) > 5 and downside.std() > 0:
            sortino = float(excess_daily.mean() / downside.std() * np.sqrt(ann_factor))

        # Beta vs benchmark
        beta = None
        jensen_alpha = None
        bench_ann_ret = None
        if not bench_daily.empty:
            common_idx = port_daily.index.intersection(bench_daily.index)
            if len(common_idx) > 30:
                p = port_daily.loc[common_idx]
                b = bench_daily.loc[common_idx]
                cov = np.cov(p.values, b.values)
                bench_var = cov[1, 1]
                beta = round(cov[0, 1] / bench_var, 2) if bench_var > 0 else None
                bench_ann_ret = float(b.mean() * ann_factor * 100)
                if beta is not None:
                    # Jensen Alpha = portfolio return - [Rf + Beta × (Bench - Rf)]
                    capm_ret = risk_free_rate + beta * (bench_ann_ret - risk_free_rate)
                    jensen_alpha = round(port_ann_ret - capm_ret, 2)

        # Max Drawdown (1-year lookback)
        one_year_ago = analysis_date - timedelta(days=365)
        port_1y = port_daily[port_daily.index >= one_year_ago.strftime("%Y-%m-%d")]
        max_dd = None
        if len(port_1y) > 10:
            cum = (1 + port_1y).cumprod()
            roll_max = cum.cummax()
            drawdown = (cum - roll_max) / roll_max * 100
            max_dd = round(float(drawdown.min()), 2)

        result = {
            "beta":               round(beta, 2) if beta else None,
            "sharpe":             round(sharpe, 2) if sharpe else None,
            "sortino":            round(sortino, 2) if sortino else None,
            "jensen_alpha":       jensen_alpha,
            "max_drawdown":       max_dd,
            "annualised_vol":     round(port_vol, 2) if port_vol else None,
            "portfolio_ann_return": round(port_ann_ret, 2) if port_ann_ret else None,
        }
    except Exception:
        pass

    return result


def generate_risk_flags(metrics: dict, sector_df: pd.DataFrame) -> list:
    """Generate human-readable risk flag list."""
    flags = []

    if metrics["top1_wt"] > 25:
        flags.append({"level": f"⚠️  HIGH: Top position {metrics['largest_company'][:20]} = {metrics['top1_wt']:.1f}% of portfolio", "color": "red"})
    elif metrics["top1_wt"] > 15:
        flags.append({"level": f"🟡 MODERATE: Top position {metrics['largest_company'][:20]} = {metrics['top1_wt']:.1f}%", "color": "orange"})

    if metrics["top5_wt"] > 75:
        flags.append({"level": f"⚠️  HIGH: Top 5 positions = {metrics['top5_wt']:.1f}% — concentrated portfolio", "color": "red"})

    if metrics["top_sector_wt"] > 40:
        flags.append({"level": f"⚠️  HIGH: {metrics['top_sector']} = {metrics['top_sector_wt']:.1f}% — sector concentration risk", "color": "red"})

    if metrics["effective_n"] < 5:
        flags.append({"level": f"🟡 MODERATE: Effective holdings = {metrics['effective_n']:.1f} (behaves like {metrics['effective_n']:.0f} stocks)", "color": "orange"})

    zero_sectors = sector_df[
        (sector_df["Bench_Weight"] > 5) & (sector_df["Port_Weight"] == 0)
    ]["Sector"].tolist()
    if zero_sectors:
        flags.append({"level": f"ℹ️  INFO: Zero exposure — {', '.join(zero_sectors[:3])}", "color": "blue"})

    if metrics["win_loss_ratio"] < 1.0 and metrics["hit_rate"] < 50:
        flags.append({"level": f"🟡 MODERATE: Hit rate {metrics['hit_rate']:.0f}% with avg loss > avg win", "color": "orange"})

    if metrics["active_share"] < 40:
        flags.append({"level": f"ℹ️  INFO: Low active share {metrics['active_share']:.0f}% — portfolio tracks benchmark closely", "color": "blue"})

    if metrics["active_share"] > 90:
        flags.append({"level": f"⚠️  HIGH: Very high active share {metrics['active_share']:.0f}% — extreme divergence from benchmark", "color": "red"})

    if not flags:
        flags.append({"level": "✅  No major risk flags detected", "color": "green"})

    return flags


def get_risk_bucket(max_drawdown: float) -> str:
    """Map 1-year max drawdown to risk bucket."""
    if max_drawdown is None:
        return "Unknown"
    if max_drawdown > -5:
        return "Low"
    elif max_drawdown > -12:
        return "Moderate"
    elif max_drawdown > -20:
        return "High"
    else:
        return "Very High"


def fetch_correlation_data(
    holdings: pd.DataFrame,
    analysis_date: datetime,
    years: int = CORR_YEARS,
) -> pd.DataFrame:
    """Fetch historical returns for correlation matrix."""
    start = (analysis_date - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    end = analysis_date.strftime("%Y-%m-%d")
    price_data = {}

    for _, row in holdings.iterrows():
        ticker = row["Ticker"]
        company = row["Company"][:15]
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if not data.empty and "Close" in data.columns:
                price_data[company] = data["Close"].dropna()
        except Exception:
            pass

    if len(price_data) < 2:
        return pd.DataFrame()

    prices_df = pd.DataFrame(price_data).dropna(how="all")
    return prices_df.pct_change().dropna()


def plot_risk_dashboard(
    holdings: pd.DataFrame,
    sector_df: pd.DataFrame,
    metrics: dict,
    corr_returns: pd.DataFrame,
    report_date: str,
) -> plt.Figure:
    """
    Figure 7-9: Concentration chart, correlation heatmap, risk dashboard.
    """
    has_corr = not corr_returns.empty and len(corr_returns.columns) > 2

    fig = plt.figure(figsize=(18, 14), facecolor=C["bg"])
    fig.suptitle(
        f"Portfolio Risk Analytics  |  {report_date}",
        fontsize=13, fontweight="bold", color=C["text"], y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :2])
    ax5 = fig.add_subplot(gs[1, 2])

    def style_ax(ax):
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["text2"], labelsize=7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(C["border"])
        ax.spines["bottom"].set_color(C["border"])
        ax.set_axisbelow(True)

    # --- A: Position weights bar ---
    style_ax(ax1)
    sorted_h = holdings.sort_values("Portfolio Wt%", ascending=False).head(12)
    intensities = np.linspace(0.9, 0.4, len(sorted_h))
    bar_colors = [plt.cm.Blues(i) for i in intensities]
    bars = ax1.barh(sorted_h["Company"].str[:20][::-1], sorted_h["Portfolio Wt%"][::-1],
                    color=bar_colors[::-1], zorder=3)
    ax1.xaxis.grid(True, color=C["grid"], lw=0.5, zorder=0)
    ax1.yaxis.grid(False)
    ax1.set_title("Position Weights", fontsize=9, color=C["text"])
    ax1.set_xlabel("Weight (%)", fontsize=8)

    # --- B: Cumulative weight curve ---
    style_ax(ax2)
    cum_wt = holdings.sort_values("Portfolio Wt%", ascending=False)["Portfolio Wt%"].cumsum().values
    x_pos = np.arange(1, len(cum_wt) + 1)
    ax2.plot(x_pos, cum_wt, color=C["blue"], lw=2, zorder=3)
    ax2.fill_between(x_pos, 0, cum_wt, alpha=0.15, color=C["blue"])
    for ref, label in [(50, "50%"), (75, "75%"), (90, "90%")]:
        ax2.axhline(ref, color=C["text3"], lw=0.8, ls="--")
        ax2.text(len(cum_wt) * 0.98, ref + 1, label, fontsize=7, color=C["text3"], ha="right")
    for n, _ in [(1, ""), (3, ""), (5, "")]:
        if n <= len(cum_wt):
            ax2.axvline(n, color=C["gold"], lw=0.8, ls=":")
    ax2.set_xlabel("# Holdings (sorted by weight)", fontsize=8)
    ax2.set_ylabel("Cumulative Weight (%)", fontsize=8)
    ax2.set_title("Concentration Curve", fontsize=9, color=C["text"])
    ax2.yaxis.grid(True, color=C["grid"], lw=0.5, zorder=0)

    # --- C: Return distribution ---
    style_ax(ax3)
    valid = holdings.dropna(subset=["Return %"])
    ret_colors = [C["green"] if v >= 0 else C["red"] for v in valid["Return %"]]
    ax3.barh(valid.sort_values("Return %")["Company"].str[:16],
             valid.sort_values("Return %")["Return %"],
             color=sorted(ret_colors, key=lambda x: (x == C["green"])), alpha=0.85, zorder=3)
    ax3.axvline(0, color=C["text"], lw=0.8)
    ax3.xaxis.grid(True, color=C["grid"], lw=0.5, zorder=0)
    ax3.yaxis.grid(False)
    ax3.tick_params(axis="y", labelsize=6.5)
    ax3.set_title("Return Distribution", fontsize=9, color=C["text"])
    ax3.set_xlabel("Return (%)", fontsize=8)

    # --- D: Correlation heatmap ---
    if has_corr:
        corr_mat = corr_returns.corr()
        n = len(corr_mat)
        import matplotlib.colors as mcolors
        cmap = plt.cm.RdYlGn
        im = ax4.imshow(corr_mat.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        ax4.set_xticks(range(n))
        ax4.set_yticks(range(n))
        ax4.set_xticklabels(corr_mat.columns, rotation=45, ha="right", fontsize=7)
        ax4.set_yticklabels(corr_mat.columns, fontsize=7)
        for i in range(n):
            for j in range(n):
                ax4.text(j, i, f"{corr_mat.values[i, j]:.2f}",
                         ha="center", va="center", fontsize=6, color="black")
        plt.colorbar(im, ax=ax4, shrink=0.8)
        avg_corr = (corr_mat.values.sum() - n) / (n * (n - 1)) if n > 1 else 0
        ax4.set_title(f"Correlation Heatmap (avg: {avg_corr:.2f})", fontsize=9, color=C["text"])
    else:
        ax4.set_facecolor(C["panel"])
        ax4.text(0.5, 0.5, "Correlation data unavailable", transform=ax4.transAxes,
                 ha="center", va="center", color=C["text3"])
        ax4.set_title("Correlation Heatmap", fontsize=9, color=C["text"])

    # --- E: Key metrics summary ---
    ax5.set_facecolor(C["panel"])
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    ax5.spines["left"].set_visible(False)
    ax5.spines["bottom"].set_visible(False)

    def risk_color(val, low, high):
        return C["green"] if val < low else (C["red"] if val > high else C["gold"])

    metrics_display = [
        ("Active Share", f"{metrics['active_share']:.1f}%",
         risk_color(metrics["active_share"], 60, 90)),
        ("HHI", f"{metrics['hhi']:.0f}",
         risk_color(metrics["hhi"], 0, 1500)),
        ("Effective N", f"{metrics['effective_n']:.1f}",
         risk_color(-metrics["effective_n"], -10, -5)),
        ("Hit Rate", f"{metrics['hit_rate']:.1f}%",
         risk_color(-metrics["hit_rate"], -70, -50)),
        ("Win/Loss", f"{metrics['win_loss_ratio']:.2f}x",
         risk_color(-metrics["win_loss_ratio"], -2.0, -1.0)),
        ("Top 5 Wt", f"{metrics['top5_wt']:.1f}%",
         risk_color(metrics["top5_wt"], 0, 65)),
    ]
    ax5.set_title("Risk Scorecard", fontsize=9, color=C["text"], pad=8)
    for i, (label, value, color) in enumerate(metrics_display):
        y_pos = 0.9 - i * 0.14
        ax5.text(0.05, y_pos, label, transform=ax5.transAxes,
                 fontsize=8.5, color=C["text2"], va="center")
        ax5.text(0.95, y_pos, value, transform=ax5.transAxes,
                 fontsize=9, fontweight="bold", color=color, va="center", ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig
