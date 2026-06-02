# =============================================================================
# MODULE 5D — ACTION ENGINE
# STRONG_INCREASE / INCREASE / HOLD / REDUCE / STRONG_REDUCE signals
# Conviction score + Decision score per spec
# =============================================================================

import pandas as pd
import numpy as np

SIGNALS = ["STRONG_INCREASE", "INCREASE", "HOLD", "REDUCE", "STRONG_REDUCE"]

# Legacy aliases for backward compat with PDF / app
SIGNAL_DISPLAY = {
    "STRONG_INCREASE": "ACCUMULATE ↑↑",
    "INCREASE":        "ACCUMULATE ↑",
    "HOLD":            "HOLD",
    "REDUCE":          "TRIM ↓",
    "STRONG_REDUCE":   "EXIT ↓↓",
}

ACTION_COLORS = {
    "STRONG_INCREASE": "#15803D",
    "INCREASE":        "#1D4ED8",
    "HOLD":            "#D97706",
    "REDUCE":          "#EA580C",
    "STRONG_REDUCE":   "#DC2626",
}

SIZING_TARGETS = {
    "STRONG_INCREASE": (10.0, 14.0),
    "INCREASE":        (6.0,  10.0),
    "HOLD":            (3.0,   7.0),
    "REDUCE":          (1.0,   3.0),
    "STRONG_REDUCE":   (0.0,   1.0),
}

# Template-specific adjustments per Req.md spec
TEMPLATE_ADJUSTMENTS = {
    "PSU_BANK":          +5,
    "SFB":               +5,
    "NBFC":              +5,
    "POWER_IPP":         -5,
    "RENEWABLE_SOLAR":   -5,
}


def _build_conviction_score(score: float, d: dict, btype: str) -> float:
    """
    Build conviction score 0–100 from multiple factors.
    Per Req.md Module 5.6 spec.
    """
    conviction = 50.0

    # Valuation score contribution
    if score is not None:
        s_norm = score / 10.0          # convert 0-100 scale to 0-10
        if s_norm >= 8:     conviction += 15
        elif s_norm >= 6:   conviction += 10
        elif s_norm >= 4:   conviction += 5
        else:               conviction -= 10

    # ROE quality
    roe = d.get("roe") or 0
    if roe >= 20:    conviction += 15
    elif roe >= 15:  conviction += 10
    elif roe < 8:    conviction -= 10

    # Data completeness (confidence)
    completeness = d.get("confidence", 100)
    if completeness < 70:   conviction -= 10
    elif completeness >= 95: conviction += 5

    # Count valid metrics
    metric_keys = ["pe", "pb", "ev_ebitda", "roce", "roe", "ebitda_margin", "rev_cagr_3y"]
    valid_metrics = sum(1 for k in metric_keys if d.get(k) is not None)
    if valid_metrics >= 4:   conviction += 10
    elif valid_metrics <= 1: conviction -= 10

    # Template-specific adjustments
    conviction += TEMPLATE_ADJUSTMENTS.get(btype, 0)

    # PE-level adjustment
    pe = d.get("pe")
    if pe:
        if pe <= 15:   conviction += 5
        elif pe >= 40: conviction -= 5

    return round(min(100, max(0, conviction)), 1)


def _build_decision_score(composite_score: float, conviction: float) -> float:
    """
    Decision Score = (Composite Score × 0.65) + (Conviction × 0.35)
    Per Req.md spec: (Composite Score × 10 × 0.65) + (Conviction × 0.35)
    We use 0-100 scale throughout.
    """
    if composite_score is None:
        composite_score = 44.0   # neutral
    return round((composite_score * 0.65) + (conviction * 0.35), 1)


def _signal_from_decision(decision_score: float) -> str:
    """Map decision score to action signal per Req.md thresholds."""
    if decision_score >= 85:   return "STRONG_INCREASE"
    elif decision_score >= 72: return "INCREASE"
    elif decision_score >= 55: return "HOLD"
    elif decision_score >= 40: return "REDUCE"
    else:                      return "STRONG_REDUCE"


def _sizing_rationale(signal: str, current_wt: float, lo: float, hi: float) -> str:
    mid = (lo + hi) / 2
    if signal == "STRONG_INCREASE":
        return f"High conviction — scale to {lo:.0f}–{hi:.0f}% target; current {current_wt:.1f}%"
    if signal == "INCREASE":
        return f"Add to position toward {lo:.0f}–{hi:.0f}%; current {current_wt:.1f}%"
    if signal == "HOLD":
        return f"Maintain at {current_wt:.1f}%; within {lo:.0f}–{hi:.0f}% conviction range"
    if signal == "REDUCE":
        return f"Trim toward {mid:.0f}%; current {current_wt:.1f}% exceeds conviction"
    return f"Exit; target ≤{hi:.0f}%. Current {current_wt:.1f}% — divest progressively"


def run_action_engine(
    verdicts: dict,
    scenarios: dict,
    holdings_df: pd.DataFrame,
    clean_data: dict = None,
) -> dict:
    """
    Generate action signal for every holding with full conviction model.
    Returns: {ticker: action_dict}
    """
    clean_data = clean_data or {}
    actions = {}

    for _, row in holdings_df.iterrows():
        ticker      = row["Ticker"]
        v           = verdicts.get(ticker) or {}
        sc          = scenarios.get(ticker) or {}
        d           = clean_data.get(ticker) or {}
        score       = v.get("score")
        btype       = sc.get("btype", d.get("btype", "UNKNOWN"))
        current_wt  = row.get("Portfolio Wt%", 0)

        # ETF — always HOLD
        if btype == "ETF":
            actions[ticker] = {
                "signal": "HOLD", "signal_display": "HOLD",
                "conviction_score": 50.0, "decision_score": 50.0,
                "conviction": "MEDIUM",
                "target_wt_lo": 4.0, "target_wt_hi": 8.0,
                "current_wt": round(current_wt, 2),
                "sizing_rationale": "ETF — hold for index exposure",
                "action_color": ACTION_COLORS["HOLD"],
                "reason": "ETF — no fundamental action signal",
            }
            continue

        # Build conviction and decision scores
        conviction_score = _build_conviction_score(score, d, btype)
        decision_score   = _build_decision_score(score, conviction_score)
        signal           = _signal_from_decision(decision_score)

        # Override: VALUE_TRAP → force REDUCE/STRONG_REDUCE
        role = sc.get("portfolio_role", "TACTICAL")
        if role == "VALUE_TRAP" and signal in ("STRONG_INCREASE", "INCREASE"):
            signal = "REDUCE"
            decision_score = min(decision_score, 45.0)

        # Conviction label (HIGH/MEDIUM/LOW for display)
        if decision_score >= 72:    conv_label = "HIGH"
        elif decision_score >= 55:  conv_label = "MEDIUM"
        else:                       conv_label = "LOW"

        lo, hi = SIZING_TARGETS[signal]
        rationale = _sizing_rationale(signal, current_wt, lo, hi)

        # Human-readable reason
        score_str = f"{score:.0f}/100" if score is not None else "N/A"
        reason = (
            f"Decision score {decision_score:.0f} "
            f"[Valuation {score_str} × 0.65 + Conviction {conviction_score:.0f} × 0.35]. "
            f"ROE {d.get('roe') or '—'}%."
        )

        actions[ticker] = {
            "signal":           signal,
            "signal_display":   SIGNAL_DISPLAY[signal],
            "conviction_score": conviction_score,
            "decision_score":   decision_score,
            "conviction":       conv_label,
            "target_wt_lo":     lo,
            "target_wt_hi":     hi,
            "current_wt":       round(current_wt, 2),
            "sizing_rationale": rationale,
            "action_color":     ACTION_COLORS[signal],
            "reason":           reason,
        }

    return actions


def portfolio_action_summary(actions: dict, holdings_df: pd.DataFrame) -> dict:
    """Roll up action signals into portfolio-level summary + regime detection."""
    counts = {s: 0 for s in SIGNALS}
    for a in actions.values():
        sig = a.get("signal", "HOLD")
        counts[sig] = counts.get(sig, 0) + 1

    total = len(actions)
    increase = counts["STRONG_INCREASE"] + counts["INCREASE"]
    reduce   = counts["REDUCE"] + counts["STRONG_REDUCE"]

    # Regime detection per Req.md Module 5.7
    if total > 0 and increase / total > 0.75:
        regime = "Risk-On / Aggressive Alpha Tilt"
    elif total > 0 and reduce / total > 0.35:
        regime = "Risk-Off / Defensive Tilt"
    else:
        regime = "Balanced / Rotational Market"

    ticker_name = dict(zip(holdings_df["Ticker"], holdings_df["Company"]))
    def names(sig_list):
        return [ticker_name.get(t, t)[:20] for t in sig_list]

    strong_increase = [t for t, a in actions.items() if a["signal"] == "STRONG_INCREASE"]
    increase_tickers = [t for t, a in actions.items() if a["signal"] == "INCREASE"]
    reduce_tickers   = [t for t, a in actions.items() if a["signal"] == "REDUCE"]
    exit_tickers     = [t for t, a in actions.items() if a["signal"] == "STRONG_REDUCE"]

    # Top opportunities by decision_score
    sorted_by_decision = sorted(
        [(t, a.get("decision_score", 0)) for t, a in actions.items()],
        key=lambda x: x[1], reverse=True,
    )
    top_opportunities = [
        ticker_name.get(t, t)[:20] for t, ds in sorted_by_decision if ds >= 70
    ][:5]
    reduce_candidates = [
        ticker_name.get(t, t)[:20] for t, a in actions.items()
        if a["signal"] in ("REDUCE", "STRONG_REDUCE")
    ][:5]

    return {
        "signal_counts":      counts,
        "increase_count":     increase,
        "reduce_count":       reduce,
        "hold_count":         counts["HOLD"],
        "regime":             regime,
        "accumulate_names":   names(strong_increase + increase_tickers)[:5],
        "trim_names":         names(reduce_tickers),
        "exit_names":         names(exit_tickers),
        "top_opportunities":  top_opportunities,
        "reduce_candidates":  reduce_candidates,
        "net_bias": (
            "BULLISH"  if increase > reduce
            else "BEARISH" if reduce > increase
            else "NEUTRAL"
        ),
    }
