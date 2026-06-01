# =============================================================================
# MODULE 5D — ACTION ENGINE
# ACCUMULATE / HOLD / TRIM / EXIT / WATCH signals + position sizing logic
# =============================================================================

import pandas as pd

# Signal labels in priority order
SIGNALS = ["ACCUMULATE", "HOLD", "TRIM", "EXIT", "WATCH"]

# Conviction → position sizing targets (% of portfolio)
SIZING_TARGETS = {
    "HIGH":   (8.0, 12.0),   # target 8–12%
    "MEDIUM": (4.0,  7.0),   # target 4–7%
    "LOW":    (1.0,  3.0),   # target 1–3%
}

ACTION_COLORS = {
    "ACCUMULATE": "#15803D",
    "HOLD":       "#1D4ED8",
    "TRIM":       "#D97706",
    "EXIT":       "#DC2626",
    "WATCH":      "#6B7280",
}


def _conviction_level(score: float) -> str:
    if score is None:
        return "LOW"
    if score >= 70:
        return "HIGH"
    if score >= 50:
        return "MEDIUM"
    return "LOW"


def _sizing_rationale(conviction: str, current_wt: float, target_lo: float, target_hi: float, signal: str) -> str:
    target_mid = (target_lo + target_hi) / 2
    if signal == "ACCUMULATE":
        if current_wt < target_lo:
            return f"Underweight at {current_wt:.1f}% vs {target_lo:.0f}–{target_hi:.0f}% target — add to build position"
        return f"At {current_wt:.1f}%; high conviction warrants scaling toward {target_hi:.0f}%"
    if signal == "HOLD":
        return f"Current {current_wt:.1f}% within {target_lo:.0f}–{target_hi:.0f}% conviction range — maintain"
    if signal == "TRIM":
        return f"Reduce toward {target_mid:.0f}%; current {current_wt:.1f}% exceeds risk-adjusted conviction"
    if signal == "EXIT":
        return f"Exit position; thesis broken. Current {current_wt:.1f}% — full divestment recommended"
    return f"Monitor at {current_wt:.1f}%; no near-term action warranted"


def run_action_engine(
    verdicts: dict,
    scenarios: dict,
    holdings_df: pd.DataFrame,
) -> dict:
    """
    Generate action signal + position sizing logic for every holding.

    Returns: {ticker: action_dict}
    """
    actions = {}

    for _, row in holdings_df.iterrows():
        ticker     = row["Ticker"]
        v          = verdicts.get(ticker) or {}
        sc         = scenarios.get(ticker) or {}
        score      = v.get("score")
        role       = sc.get("portfolio_role", "TACTICAL")
        asymmetry  = sc.get("asymmetry_ratio", 1.0) or 1.0
        bear_ret   = sc.get("bear", {}).get("return_pct", -20)
        current_wt = row.get("Portfolio Wt%", 0)
        d_flags    = v.get("flags", [])
        btype      = sc.get("btype", "UNKNOWN")

        # ETF — always HOLD
        if btype == "ETF":
            actions[ticker] = {
                "signal":           "HOLD",
                "conviction":       "MEDIUM",
                "target_wt_lo":     4.0,
                "target_wt_hi":     8.0,
                "sizing_rationale": "ETF — hold for index exposure. Adjust based on NASDAQ view.",
                "action_color":     ACTION_COLORS["HOLD"],
                "reason":           "ETF position — no fundamental action signal applicable",
            }
            continue

        # ── Determine signal ───────────────────────────────────────────────
        signal = "WATCH"

        if score is not None:
            # EXIT: thesis broken
            if (score < 25 or
                    (role == "VALUE_TRAP") or
                    (bear_ret is not None and bear_ret < -50 and score < 35)):
                signal = "EXIT"

            # TRIM: deteriorating or overvalued
            elif (score < 40 or
                  role == "VALUE_TRAP" or
                  (bear_ret is not None and bear_ret < -35)):
                signal = "TRIM"

            # ACCUMULATE: high conviction + good asymmetry
            elif (score >= 65 and
                  asymmetry >= 1.5 and
                  role != "VALUE_TRAP"):
                signal = "ACCUMULATE"

            # HOLD: solid but not high conviction
            elif score >= 50 and asymmetry >= 1.0:
                signal = "HOLD"

            # TRIM if score OK but asymmetry poor
            elif score >= 50 and asymmetry < 0.8:
                signal = "TRIM"

        # ── Conviction level ──────────────────────────────────────────────
        conviction = _conviction_level(score)
        if signal == "EXIT":
            conviction = "LOW"
        elif signal == "ACCUMULATE":
            conviction = "HIGH"

        # ── Position sizing ───────────────────────────────────────────────
        target_lo, target_hi = SIZING_TARGETS[conviction]
        rationale = _sizing_rationale(conviction, current_wt, target_lo, target_hi, signal)

        # ── Human-readable reason ─────────────────────────────────────────
        if signal == "ACCUMULATE":
            reason = (f"Score {score:.0f}/100, asymmetry {asymmetry:.1f}x, "
                      f"role: {role}. Upside significantly exceeds downside risk.")
        elif signal == "HOLD":
            reason = (f"Score {score:.0f}/100, asymmetry {asymmetry:.1f}x. "
                      f"Conviction supports current weight.")
        elif signal == "TRIM":
            reason = (f"Score {score:.0f}/100. "
                      f"{'Weak asymmetry.' if asymmetry < 1.0 else ''} "
                      f"Reduce to free capital for higher-conviction names.")
        elif signal == "EXIT":
            reason = (f"Score {score:.0f}/100, role: {role}. "
                      f"Thesis integrity compromised — exit before further deterioration.")
        else:
            reason = f"Score {f'{score:.0f}' if score else 'N/A'}/100. Monitor for entry/exit trigger."

        actions[ticker] = {
            "signal":           signal,
            "conviction":       conviction,
            "target_wt_lo":     target_lo,
            "target_wt_hi":     target_hi,
            "current_wt":       round(current_wt, 2),
            "sizing_rationale": rationale,
            "action_color":     ACTION_COLORS[signal],
            "reason":           reason.strip(),
        }

    return actions


def portfolio_action_summary(actions: dict, holdings_df: pd.DataFrame) -> dict:
    """Roll up action signals into a portfolio-level summary."""
    counts = {s: 0 for s in SIGNALS}
    for a in actions.values():
        sig = a.get("signal", "WATCH")
        counts[sig] = counts.get(sig, 0) + 1

    # Stocks to add / trim
    accumulate_tickers = [t for t, a in actions.items() if a["signal"] == "ACCUMULATE"]
    exit_tickers       = [t for t, a in actions.items() if a["signal"] == "EXIT"]
    trim_tickers       = [t for t, a in actions.items() if a["signal"] == "TRIM"]

    # Map ticker → company name
    ticker_name = dict(zip(holdings_df["Ticker"], holdings_df["Company"]))
    def names(tickers): return [ticker_name.get(t, t)[:20] for t in tickers]

    return {
        "signal_counts":       counts,
        "accumulate_names":    names(accumulate_tickers),
        "trim_names":          names(trim_tickers),
        "exit_names":          names(exit_tickers),
        "net_bias":            (
            "BULLISH" if counts["ACCUMULATE"] > counts["TRIM"] + counts["EXIT"]
            else "BEARISH" if counts["TRIM"] + counts["EXIT"] > counts["ACCUMULATE"]
            else "NEUTRAL"
        ),
    }
