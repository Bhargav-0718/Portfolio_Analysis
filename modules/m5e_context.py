# =============================================================================
# MODULE 5E — CONTEXT CONSOLIDATION ENGINE (M6.1)
# Merges ALL prior module outputs into a single research_context dict / JSON
# This becomes the "memory + knowledge base" for the LLM in M6B
# =============================================================================

import json
from datetime import datetime
import pandas as pd


def _safe(val):
    """Make a value JSON-serializable."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if pd.isna(val):
            return None
        return round(float(val), 4) if isinstance(val, float) else int(val)
    if isinstance(val, (bool, str)):
        return val
    if isinstance(val, (list, tuple)):
        return [_safe(v) for v in val]
    if isinstance(val, dict):
        return {k: _safe(v) for k, v in val.items()}
    return str(val)


def build_context_json(
    holdings_df: pd.DataFrame,
    portfolio_stats: dict,
    sector_df: pd.DataFrame,
    attribution_summary: dict,
    risk_metrics: dict,
    risk_flags: list,
    clean_data: dict,
    verdicts: dict,
    scenarios: dict,
    actions: dict,
    research_data: dict = None,
    analysis_date: str = None,
) -> dict:
    """
    Merge all module outputs into a single structured dict.
    This dict is passed as context to the LLM (M6B thesis engine).

    Returns: context dict (JSON-serializable)
    """
    research_data = research_data or {}

    # ── PORTFOLIO SUMMARY ─────────────────────────────────────────────────
    ps = {k: _safe(v) for k, v in portfolio_stats.items()}

    # ── SECTOR ALLOCATION ─────────────────────────────────────────────────
    sector_list = []
    if sector_df is not None and not sector_df.empty:
        for _, row in sector_df.iterrows():
            if row.get("Port_Weight", 0) > 0 or row.get("Bench_Weight", 0) > 0:
                sector_list.append({
                    "sector":       row["Sector"],
                    "port_wt":      _safe(row.get("Port_Weight")),
                    "bench_wt":     _safe(row.get("Bench_Weight")),
                    "active_bet":   _safe(row.get("Active_Bet")),
                    "sector_ret":   _safe(row.get("Sector_Return%")),
                    "num_stocks":   _safe(row.get("Num_Stocks")),
                })

    # ── ATTRIBUTION ───────────────────────────────────────────────────────
    attr = {k: _safe(v) for k, v in (attribution_summary or {}).items()}

    # ── RISK ──────────────────────────────────────────────────────────────
    rm = {k: _safe(v) for k, v in (risk_metrics or {}).items()}
    flag_texts = [f.get("level", str(f)) for f in (risk_flags or [])]

    # ── ACTION SUMMARY ────────────────────────────────────────────────────
    from modules.m5d_action import portfolio_action_summary
    action_summary = portfolio_action_summary(actions or {}, holdings_df)

    # ── PER-HOLDING ───────────────────────────────────────────────────────
    holdings_context = {}

    for _, row in holdings_df.iterrows():
        ticker  = row["Ticker"]
        d       = clean_data.get(ticker) or {}
        v       = verdicts.get(ticker) or {}
        sc      = scenarios.get(ticker) or {}
        ac      = (actions or {}).get(ticker) or {}
        res     = research_data.get(ticker) or {}

        # Fundamentals — only include non-None values
        fundamentals = {
            "revenue":        _safe(d.get("revenue")),
            "ebitda":         _safe(d.get("ebitda")),
            "ebitda_margin":  _safe(d.get("ebitda_margin")),
            "pat":            _safe(d.get("pat")),
            "pat_margin":     _safe(d.get("pat_margin")),
            "roce":           _safe(d.get("roce")),
            "roe":            _safe(d.get("roe")),
            "de_ratio":       _safe(d.get("de_ratio")),
            "nd_ebitda":      _safe(d.get("nd_ebitda")),
            "fcf":            _safe(d.get("fcf")),
            "cfo":            _safe(d.get("cfo")),
            "rev_cagr_3y":    _safe(d.get("rev_cagr_3y")),
            "pat_cagr_3y":    _safe(d.get("pat_cagr_3y")),
            "bvps":           _safe(d.get("bvps")),
            "bname":          d.get("bname", ""),
            "btype":          d.get("btype", ""),
            "confidence":     _safe(d.get("confidence")),
            "accounting_flags": d.get("accounting_flags", []),
        }

        # Valuation
        valuation = {
            "pe":          _safe(d.get("pe")),
            "pb":          _safe(d.get("pb")),
            "ev_ebitda":   _safe(d.get("ev_ebitda")),
            "live_price":  _safe(d.get("live_price")),
            "live_mcap":   _safe(d.get("live_mcap")),
            "score":       _safe(v.get("score")),
            "label":       v.get("label", ""),
            "flags":       v.get("flags", []),
            "breakdown":   [
                {"factor": name, "score": s, "max": mx}
                for name, s, mx in v.get("breakdown", [])
            ],
            "peer_med":    _safe(v.get("peer_med", {})),
        }

        # Portfolio position
        position = {
            "current_price":  _safe(row.get("Current Price")),
            "avg_buy_price":  _safe(row.get("Avg Buy Price")),
            "portfolio_wt":   _safe(row.get("Portfolio Wt%")),
            "benchmark_wt":   _safe(row.get("Benchmark Wt%")),
            "active_bet":     _safe(row.get("Active Bet%")),
            "return_pct":     _safe(row.get("Return %")),
            "pnl":            _safe(row.get("P&L")),
            "current_value":  _safe(row.get("Current Value")),
            "invested_value": _safe(row.get("Invested Value")),
        }

        holdings_context[ticker] = {
            "company":      row["Company"],
            "sector":       row["Sector"],
            "position":     position,
            "fundamentals": fundamentals,
            "valuation":    valuation,
            "scenarios":    _safe(sc),
            "action":       _safe(ac),
            "research":     _safe(res),
        }

    context = {
        "generated_at":    analysis_date or datetime.now().strftime("%Y-%m-%d"),
        "portfolio_summary": ps,
        "sector_allocation": sector_list,
        "attribution":      attr,
        "risk_metrics":     rm,
        "risk_flags":       flag_texts,
        "action_summary":   _safe(action_summary),
        "holdings":         holdings_context,
    }

    return context


def context_to_json_string(context: dict) -> str:
    """Serialize context dict to compact JSON string for LLM consumption."""
    return json.dumps(context, ensure_ascii=False, indent=2)


def context_portfolio_brief(context: dict) -> str:
    """
    Produce a concise text brief of the portfolio for LLM system context.
    Keeps token count low while preserving essential intelligence.
    """
    ps   = context.get("portfolio_summary", {})
    rm   = context.get("risk_metrics", {})
    attr = context.get("attribution", {})
    asig = context.get("action_summary", {})

    lines = [
        f"Portfolio: {ps.get('num_holdings', '?')} holdings | "
        f"Invested ₹{ps.get('total_invested', 0):,.0f} | "
        f"Return {ps.get('total_return', 0):+.1f}%",

        f"Risk: Active share {rm.get('active_share', '?')}% | "
        f"HHI {rm.get('hhi', '?')} | "
        f"Effective N {rm.get('effective_n', '?')}",

        f"Attribution: Total alpha {attr.get('total_alpha', 0):+.2f}% | "
        f"Allocation {attr.get('allocation_effect', 0):+.4f}% | "
        f"Selection {attr.get('selection_effect', 0):+.4f}%",

        f"Action signals: {asig.get('signal_counts', {})} | "
        f"Net bias: {asig.get('net_bias', '?')}",
    ]

    # Sector overweights
    sectors = context.get("sector_allocation", [])
    overweights = [
        f"{s['sector']} ({s['active_bet']:+.1f}%)"
        for s in sectors
        if s.get("active_bet", 0) > 2 and s.get("port_wt", 0) > 0
    ]
    if overweights:
        lines.append(f"Key overweights: {', '.join(overweights)}")

    return "\n".join(lines)
