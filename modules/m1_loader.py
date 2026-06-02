# =============================================================================
# MODULE 1 — PORTFOLIO LOADER WITH LIVE PRICES
# =============================================================================

import pandas as pd
import numpy as np
import yfinance as yf
import logging
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


# Tickers known to be broken on yfinance — require manual price input
DEFAULT_MANUAL_PRICES = {
    "TATAMOTORS.NS": None,
    "BBG.NS": None,
}


def load_portfolio(csv_file) -> pd.DataFrame:
    """
    Parse portfolio CSV (same format as portfolio.csv).
    Returns clean holdings DataFrame.
    """
    raw = pd.read_csv(csv_file, header=0)
    raw.columns = raw.columns.str.strip()

    # Auto-detect MarketCap column
    mktcap_col = next(
        (c for c in raw.columns if "market" in c.lower() or "cap" in c.lower()), None
    )

    # Keep only valid stock rows (No. between 1-100)
    holdings = raw[
        raw["No."].notna()
        & (pd.to_numeric(raw["No."], errors="coerce").between(1, 100))
    ].copy().reset_index(drop=True)

    keep_cols = ["Ticker", "Company", "Sector", "Quantity", "Avg Buy Price", "Benchmark Wt%"]
    if mktcap_col:
        keep_cols.append(mktcap_col)
    holdings = holdings[keep_cols].copy()

    if mktcap_col:
        holdings = holdings.rename(columns={mktcap_col: "MarketCap"})
    else:
        holdings["MarketCap"] = 0

    for col in ["Quantity", "Avg Buy Price", "Benchmark Wt%", "MarketCap"]:
        holdings[col] = pd.to_numeric(holdings[col], errors="coerce").fillna(0)

    holdings["Invested Value"] = (holdings["Quantity"] * holdings["Avg Buy Price"]).round(2)
    return holdings


def fetch_prices(
    holdings: pd.DataFrame,
    analysis_date: datetime,
    manual_prices: dict = None,
    progress_callback=None,
) -> dict:
    """
    Fetch closing prices for all holdings as of analysis_date.
    manual_prices: dict of {ticker: price} for tickers that fail yfinance.
    Returns dict of {ticker: price}.
    """
    manual_prices = manual_prices or {}
    fetch_start = (analysis_date - timedelta(days=10)).strftime("%Y-%m-%d")
    fetch_end = (analysis_date + timedelta(days=2)).strftime("%Y-%m-%d")

    prices = {}
    results = []  # list of (ticker, company, price, status)

    for _, row in holdings.iterrows():
        ticker = row["Ticker"]
        company = row["Company"]

        # Manual override
        if ticker in manual_prices and manual_prices[ticker]:
            prices[ticker] = float(manual_prices[ticker])
            results.append((ticker, company, prices[ticker], "manual"))
            if progress_callback:
                progress_callback(ticker, company, prices[ticker], "manual")
            continue

        # yfinance fetch
        price = None
        try:
            data = yf.download(
                ticker,
                start=fetch_start,
                end=fetch_end,
                progress=False,
                auto_adjust=True,
            )
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if not data.empty and "Close" in data.columns:
                closes = data["Close"].dropna()
                if len(closes) > 0:
                    price = round(float(closes.iloc[-1]), 2)
        except Exception:
            pass

        if price and price > 0:
            prices[ticker] = price
            results.append((ticker, company, price, "live"))
        else:
            prices[ticker] = row["Avg Buy Price"]
            results.append((ticker, company, row["Avg Buy Price"], "fallback"))

        if progress_callback:
            status = "live" if (price and price > 0) else "fallback"
            progress_callback(ticker, company, prices[ticker], status)

    return prices, results


def calculate_metrics(holdings: pd.DataFrame, prices: dict) -> pd.DataFrame:
    """
    Add current price, current value, P&L, return%, weights to holdings.
    Returns enriched DataFrame sorted by portfolio weight.
    """
    df = holdings.copy()
    df["Current Price"] = df["Ticker"].map(prices)
    df["Current Value"] = (df["Quantity"] * df["Current Price"]).round(2)
    df["P&L"] = (df["Current Value"] - df["Invested Value"]).round(2)
    df["Return %"] = (df["P&L"] / df["Invested Value"] * 100).round(2)

    total_current = df["Current Value"].sum()
    df["Portfolio Wt%"] = (df["Current Value"] / total_current * 100).round(2)
    df["Active Bet%"] = (df["Portfolio Wt%"] - df["Benchmark Wt%"]).round(2)

    df = df.sort_values("Portfolio Wt%", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    return df


def compute_max_drawdown(
    holdings: pd.DataFrame,
    analysis_date: datetime,
    lookback_years: int = 1,
) -> tuple:
    """
    Compute portfolio-weighted 1-year rolling max drawdown.
    Returns (max_drawdown_pct, risk_bucket).
    """
    import warnings
    warnings.filterwarnings("ignore")

    start = (analysis_date - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")
    end   = analysis_date.strftime("%Y-%m-%d")

    price_data = {}
    for _, row in holdings.iterrows():
        ticker = row["Ticker"]
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if not data.empty and "Close" in data.columns:
                price_data[ticker] = data["Close"].dropna()
        except Exception:
            pass

    if not price_data:
        return None, "Unknown"

    prices_df  = pd.DataFrame(price_data).dropna(how="all")
    weight_map = dict(zip(holdings["Ticker"], holdings["Portfolio Wt%"] / 100))
    port_idx   = pd.Series(0.0, index=prices_df.index)

    for ticker, wt in weight_map.items():
        if ticker in prices_df.columns:
            s = prices_df[ticker].dropna()
            if len(s) > 1:
                port_idx = port_idx.add(((s / s.iloc[0]) * wt), fill_value=0)

    port_cum  = port_idx.dropna()
    if len(port_cum) < 10:
        return None, "Unknown"

    roll_max  = port_cum.cummax()
    drawdown  = (port_cum - roll_max) / roll_max * 100
    max_dd    = round(float(drawdown.min()), 2)

    # Risk bucket per spec
    if max_dd > -5:      bucket = "Low"
    elif max_dd > -12:   bucket = "Moderate"
    elif max_dd > -20:   bucket = "High"
    else:                bucket = "Very High"

    return max_dd, bucket


def portfolio_summary(holdings: pd.DataFrame) -> dict:
    """Compute top-level portfolio statistics."""
    total_invested = holdings["Invested Value"].sum()
    total_current = holdings["Current Value"].sum()
    total_pnl = holdings["P&L"].sum()
    total_return = (total_pnl / total_invested) * 100

    valid = holdings.dropna(subset=["Return %"])
    best = valid.loc[valid["Return %"].idxmax()]
    worst = valid.loc[valid["Return %"].idxmin()]
    largest = holdings.loc[holdings["Portfolio Wt%"].idxmax()]

    return {
        "total_invested": total_invested,
        "total_current": total_current,
        "total_pnl": total_pnl,
        "total_return": total_return,
        "num_holdings": len(holdings),
        "num_winners": int((holdings["P&L"] > 0).sum()),
        "num_losers": int((holdings["P&L"] < 0).sum()),
        "best_performer": best["Company"],
        "best_return": best["Return %"],
        "worst_performer": worst["Company"],
        "worst_return": worst["Return %"],
        "largest_position": largest["Company"],
        "largest_wt": largest["Portfolio Wt%"],
        "top5_wt": holdings.nlargest(5, "Portfolio Wt%")["Portfolio Wt%"].sum(),
        # Populated later after max drawdown is computed
        "max_drawdown": None,
        "risk_bucket": "Unknown",
    }
