# =============================================================================
# MODULE 5B VALUATION — NOTEBOOK-EXACT SCORING ENGINE
# Implements Final_Project_Code.ipynb Modules 5.1 + 5.2 + 5.3 exactly:
#   - Peer-comparison scoring (premium/discount vs peer median)
#   - Template-weighted composite (0–10 scale)
#   - SCORING_BANDS_MULTIPLE / SCORING_BANDS_QUALITY
#   - Manual bank metrics (GNPA, NIM, CASA)
#   - 95th-percentile winsorization
# =============================================================================

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import logging
from typing import Optional

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# =============================================================================
# BLOCK 1 — HOLDING → TEMPLATE MAPPING  (Module 5.1 HOLDINGS)
# Keys use .NS suffix (yfinance format). BBNL and MON100 skipped.
# =============================================================================

HOLDING_TEMPLATES = {
    "HINDCOPPER.NS": ("METALS_MINING",    "Hindustan Copper"),
    "HINDZINC.NS":   ("METALS_MINING",    "Hindustan Zinc"),
    "TATASTEEL.NS":  ("STEEL",            "Tata Steel"),
    "TATAMOTORS.NS": ("AUTO_OEM",         "Tata Motors"),
    "GABRIEL.NS":    ("AUTO_ANCILLARY",   "Gabriel India"),
    "PNB.NS":        ("PSU_BANK",         "Punjab National Bank"),
    "BANKINDIA.NS":  ("PSU_BANK",         "Bank of India"),
    "IDFCFIRSTB.NS": ("PRIVATE_BANK",     "IDFC First Bank"),
    "EQUITASBNK.NS": ("SMALL_FINANCE_BANK","Equitas SFB"),
    "JIOFIN.NS":     ("NBFC_FINSERV",     "Jio Financial"),
    "INDUSTOWER.NS": ("TOWER_INFRA",      "Indus Towers"),
    "ADANIPOWER.NS": ("POWER_UTILITY",    "Adani Power"),
    "BORORENEW.NS":  ("SOLAR_GLASS",      "Borosil Renewables"),
    "IEX.NS":        ("ENERGY_EXCHANGE",  "Indian Energy Exchange"),
    # BBG.NS → BBNL → SKIP (unlisted)
    # MON100.NS → ETF → SKIP
}

SKIP_VALUATION = {"BBG.NS", "MON100.NS"}

FLAG_NOTES = {
    "TATAMOTORS.NS": "Transient yfinance no_data issue possible. Retry engine enabled.",
    "JIOFIN.NS":     "Early-stage NBFC. P/B more reliable than earnings metrics.",
    "ADANIPOWER.NS": "EV multiples preferred due to leverage profile.",
    "BORORENEW.NS":  "Proxy peers used due to niche solar glass exposure.",
}

# =============================================================================
# BLOCK 2 — PEER GROUPS  (Module 5.1 PEER_GROUPS)
# =============================================================================

PEER_GROUPS = {
    "HINDCOPPER.NS": ["VEDL.NS",      "HINDALCO.NS",  "NATIONALUM.NS", "NMDC.NS"],
    "HINDZINC.NS":   ["VEDL.NS",      "HINDALCO.NS",  "NATIONALUM.NS", "HINDCOPPER.NS"],
    "TATASTEEL.NS":  ["JSWSTEEL.NS",  "SAIL.NS",      "JINDALSTEL.NS", "RATNAMANI.NS"],
    "TATAMOTORS.NS": ["M&M.NS",       "MARUTI.NS",    "ASHOKLEY.NS",   "EICHERMOT.NS"],
    "GABRIEL.NS":    ["UNOMINDA.NS",  "SUPRAJIT.NS",  "MMFL.NS",       "ENDURANCE.NS"],
    "PNB.NS":        ["CANBK.NS",     "UNIONBANK.NS", "INDIANB.NS",    "BANKINDIA.NS"],
    "BANKINDIA.NS":  ["CANBK.NS",     "UNIONBANK.NS", "INDIANB.NS",    "PNB.NS"],
    "IDFCFIRSTB.NS": ["YESBANK.NS",   "RBLBANK.NS",   "BANDHANBNK.NS", "FEDERALBNK.NS"],
    "EQUITASBNK.NS": ["AUBANK.NS",    "UJJIVANSFB.NS","JANA.NS",       "SURYODAY.NS"],
    "JIOFIN.NS":     ["BAJFINANCE.NS","CHOLAFIN.NS",  "MUTHOOTFIN.NS", "BAJAJFINSV.NS"],
    "INDUSTOWER.NS": ["BHARTIARTL.NS","TATACOMM.NS",  "GTLINFRA.NS",   "STLTECH.NS"],
    "ADANIPOWER.NS": ["NTPC.NS",      "TATAPOWER.NS", "TORNTPOWER.NS", "CESC.NS"],
    "BORORENEW.NS":  ["ASAHIINDIA.NS","WAAREEENER.NS","ADANIGREEN.NS", "TATAPOWER.NS"],
    "IEX.NS":        ["MCX.NS",       "BSE.NS",       "CDSL.NS",       "CAMS.NS"],
}

# =============================================================================
# BLOCK 3 — VALUATION TEMPLATES  (Module 5.1 VALUATION_TEMPLATES)
# =============================================================================

VALUATION_TEMPLATES = {
    "PSU_BANK":          ["pb", "pe", "roe", "gnpa", "nim", "casa"],
    "PRIVATE_BANK":      ["pb", "pe", "roe", "gnpa", "nim", "casa"],
    "SMALL_FINANCE_BANK":["pb", "pe", "roe", "gnpa", "nim", "casa"],
    "METALS_MINING":     ["ev_ebitda", "pb", "ev_sales", "net_debt_ebitda", "pe"],
    "STEEL":             ["ev_ebitda", "pb", "ev_sales", "net_debt_ebitda", "pe"],
    "AUTO_OEM":          ["ev_ebitda", "pe", "ev_sales", "ps"],
    "AUTO_ANCILLARY":    ["pe", "ev_ebitda", "pb", "ev_sales"],
    "POWER_UTILITY":     ["ev_ebitda", "pb", "pe", "div_yield"],
    "TOWER_INFRA":       ["ev_ebitda", "pe", "ev_sales"],
    "NBFC_FINSERV":      ["pb", "pe", "roe"],
    "SOLAR_GLASS":       ["ev_ebitda", "pb", "pe", "ev_sales"],
    "ENERGY_EXCHANGE":   ["pe", "ev_ebitda", "ps", "roe"],
}

# =============================================================================
# BLOCK 4 — TEMPLATE WEIGHTS  (Module 5.1 TEMPLATE_WEIGHTS)
# =============================================================================

TEMPLATE_WEIGHTS = {
    "PSU_BANK":           {"pb":0.30,"pe":0.20,"roe":0.15,"gnpa":0.20,"nim":0.10,"casa":0.05},
    "PRIVATE_BANK":       {"pb":0.25,"pe":0.20,"roe":0.20,"gnpa":0.15,"nim":0.15,"casa":0.05},
    "SMALL_FINANCE_BANK": {"pb":0.25,"pe":0.20,"roe":0.15,"gnpa":0.25,"nim":0.10,"casa":0.05},
    "METALS_MINING":      {"ev_ebitda":0.40,"pb":0.20,"ev_sales":0.20,"net_debt_ebitda":0.10,"pe":0.10},
    "STEEL":              {"ev_ebitda":0.40,"pb":0.20,"ev_sales":0.20,"net_debt_ebitda":0.10,"pe":0.10},
    "AUTO_OEM":           {"ev_ebitda":0.40,"pe":0.30,"ev_sales":0.20,"ps":0.10},
    "AUTO_ANCILLARY":     {"pe":0.40,"ev_ebitda":0.30,"pb":0.20,"ev_sales":0.10},
    "POWER_UTILITY":      {"ev_ebitda":0.40,"pb":0.25,"pe":0.25,"div_yield":0.10},
    "TOWER_INFRA":        {"ev_ebitda":0.50,"pe":0.30,"ev_sales":0.20},
    "NBFC_FINSERV":       {"pb":0.50,"pe":0.30,"roe":0.20},
    "SOLAR_GLASS":        {"ev_ebitda":0.35,"pb":0.25,"pe":0.25,"ev_sales":0.15},
    "ENERGY_EXCHANGE":    {"pe":0.35,"ev_ebitda":0.30,"ps":0.20,"roe":0.15},
}

# =============================================================================
# BLOCK 5 — METRIC METADATA  (lower_is_cheaper flag)
# =============================================================================

METRIC_META = {
    "pe":              {"label": "P/E",             "lower_is_cheaper": True},
    "pb":              {"label": "P/B",             "lower_is_cheaper": True},
    "ev_ebitda":       {"label": "EV/EBITDA",       "lower_is_cheaper": True},
    "ev_sales":        {"label": "EV/Sales",        "lower_is_cheaper": True},
    "ps":              {"label": "P/Sales",         "lower_is_cheaper": True},
    "net_debt_ebitda": {"label": "Net Debt/EBITDA", "lower_is_cheaper": True},
    "div_yield":       {"label": "Dividend Yield",  "lower_is_cheaper": False},
    "roe":             {"label": "ROE",             "lower_is_cheaper": False},
    "gnpa":            {"label": "GNPA",            "lower_is_cheaper": True},
    "nim":             {"label": "NIM",             "lower_is_cheaper": False},
    "casa":            {"label": "CASA",            "lower_is_cheaper": False},
}

# =============================================================================
# BLOCK 6 — SCORING BANDS  (Module 5.1 SCORING_BANDS)
# =============================================================================

SCORING_BANDS_MULTIPLE = [  # lower prem/disc = cheaper = higher score
    (-0.40, 10), (-0.25, 9), (-0.15, 8), (-0.05, 7), (0.05, 6),
    (0.15, 5), (0.25, 4), (0.40, 3), (0.60, 2), (float("inf"), 1),
]

SCORING_BANDS_QUALITY = [   # higher = better = higher score
    (-0.40, 1), (-0.25, 2), (-0.15, 3), (-0.05, 4), (0.05, 5),
    (0.15, 6), (0.25, 7), (0.40, 8), (0.60, 9), (float("inf"), 10),
]

# =============================================================================
# BLOCK 7 — SCORE LABELS  (Module 5.1 SCORE_LABELS, composite on 0-10)
# =============================================================================

SCORE_LABELS = [
    (3.0,  "🔴 Expensive",           "Significantly above peers"),
    (4.5,  "🟠 Moderately Expensive","Above peer average"),
    (5.5,  "🟡 Fair Value",          "Near peer median"),
    (7.0,  "🟢 Moderately Cheap",   "Below peer average"),
    (10.1, "💎 Cheap",              "Deep valuation discount"),
]

# =============================================================================
# BLOCK 8 — MANUAL BANK METRICS  (Module 5.1 MANUAL_BANK_METRICS)
# All bank holding + peer tickers included so peer scoring works
# =============================================================================

MANUAL_BANK_METRICS = {
    "PNB.NS":        {"gnpa": 4.98, "nim": 3.25, "casa": 42.3},
    "BANKINDIA.NS":  {"gnpa": 4.98, "nim": 3.02, "casa": 44.4},
    "IDFCFIRSTB.NS": {"gnpa": 1.97, "nim": 6.34, "casa": 46.9},
    "EQUITASBNK.NS": {"gnpa": 2.95, "nim": 8.73, "casa": 33.8},
    # Peers
    "CANBK.NS":      {"gnpa": 4.23, "nim": 2.92, "casa": 32.7},
    "UNIONBANK.NS":  {"gnpa": 4.76, "nim": 3.05, "casa": 35.4},
    "INDIANB.NS":    {"gnpa": 3.95, "nim": 3.26, "casa": 41.2},
    "YESBANK.NS":    {"gnpa": 1.74, "nim": 2.40, "casa": 30.9},
    "RBLBANK.NS":    {"gnpa": 2.65, "nim": 5.47, "casa": 33.4},
    "BANDHANBNK.NS": {"gnpa": 4.19, "nim": 7.93, "casa": 35.8},
    "FEDERALBNK.NS": {"gnpa": 2.11, "nim": 3.34, "casa": 30.1},
    "AUBANK.NS":     {"gnpa": 1.67, "nim": 6.27, "casa": 33.3},
    "UJJIVANSFB.NS": {"gnpa": 2.73, "nim": 9.30, "casa": 25.7},
    "JANA.NS":       {"gnpa": 7.40, "nim": 9.80, "casa": 19.2},
    "SURYODAY.NS":   {"gnpa": 3.20, "nim": 10.1, "casa": 21.5},
}

# =============================================================================
# BLOCK 9 — DATA QUALITY RULES
# =============================================================================

DQ = {
    "max_pe":              150.0,
    "max_pb":               30.0,
    "max_ev_ebitda":        60.0,
    "max_net_debt_ebitda":  15.0,
    "winsorise_pct":        0.95,
    "min_peers":             1,
}

# =============================================================================
# BLOCK 10 — YFINANCE DATA FETCH  (Module 5.2 equivalent)
# =============================================================================

def _safe(val) -> float:
    try:
        if val is None: return np.nan
        f = float(val)
        return np.nan if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return np.nan


def fetch_one_ticker(ticker: str) -> dict:
    """
    Fetch fundamental metrics for one ticker via yfinance.info.
    Returns dict with keys: pe, pb, ev_ebitda, ev_sales, ps,
    net_debt_ebitda, div_yield, roe + bank metrics from MANUAL_BANK_METRICS.
    """
    rec = {}
    try:
        info = yf.Ticker(ticker).info

        pe  = _safe(info.get("trailingPE"))
        pb  = _safe(info.get("priceToBook"))
        ev  = _safe(info.get("enterpriseValue"))
        rev = _safe(info.get("totalRevenue"))
        ebitda = _safe(info.get("ebitda"))
        debt   = _safe(info.get("totalDebt", 0)) or 0
        cash   = _safe(info.get("totalCash", 0)) or 0
        roe    = _safe(info.get("returnOnEquity"))
        ps_val = _safe(info.get("priceToSalesTrailing12Months"))
        div    = _safe(info.get("dividendYield"))

        # Sanitize extremes
        if not np.isnan(pe)  and pe  > DQ["max_pe"]:          pe  = np.nan
        if not np.isnan(pb)  and pb  > DQ["max_pb"]:          pb  = np.nan
        if not np.isnan(ebitda) and ebitda <= 0:               ebitda = np.nan

        ev_ebitda = _safe(info.get("enterpriseToEbitda"))
        if np.isnan(ev_ebitda) and not np.isnan(ev) and not np.isnan(ebitda):
            ev_ebitda = ev / ebitda
        if not np.isnan(ev_ebitda) and ev_ebitda > DQ["max_ev_ebitda"]:
            ev_ebitda = np.nan

        ev_sales = np.nan
        if not np.isnan(ev) and not np.isnan(rev) and rev > 0:
            ev_sales = ev / rev

        net_debt = debt - cash
        net_debt_ebitda = np.nan
        if not np.isnan(ebitda) and ebitda > 0:
            nd_e = net_debt / ebitda
            if abs(nd_e) <= DQ["max_net_debt_ebitda"]:
                net_debt_ebitda = nd_e

        rec = {
            "pe":              pe,
            "pb":              pb,
            "ev_ebitda":       ev_ebitda,
            "ev_sales":        ev_sales,
            "ps":              ps_val,
            "net_debt_ebitda": net_debt_ebitda,
            "div_yield":       (_safe(div) * 100) if not np.isnan(_safe(div)) else np.nan,
            "roe":             (_safe(roe) * 100) if not np.isnan(_safe(roe)) else np.nan,
        }
    except Exception:
        pass

    # Inject manual bank metrics
    bank = MANUAL_BANK_METRICS.get(ticker, {})
    for k, v in bank.items():
        rec[k] = float(v)

    return rec


def build_ticker_lookup(
    holding_tickers: list,
    progress_callback=None,
) -> dict:
    """
    Fetch data for all holdings + their peers.
    Returns TICKER_LOOKUP: {ticker: {metric: value}}
    """
    # Collect all unique tickers
    all_tickers = set(holding_tickers)
    for t in holding_tickers:
        ns_key = t.replace(".NS", "")
        peers = PEER_GROUPS.get(t, [])
        all_tickers.update(peers)

    lookup = {}
    for ticker in sorted(all_tickers):
        if ticker in SKIP_VALUATION:
            continue
        try:
            lookup[ticker] = fetch_one_ticker(ticker)
        except Exception:
            lookup[ticker] = {}
        if progress_callback:
            progress_callback(ticker)

    return lookup


# =============================================================================
# BLOCK 11 — SCORING FUNCTIONS  (Module 5.3 exact logic)
# =============================================================================

def get_peer_values(holding_ticker: str, metric: str, lookup: dict) -> pd.Series:
    """
    Returns Series of valid peer values for a metric, with 95th-pct winsorization.
    """
    peers = PEER_GROUPS.get(holding_ticker, [])
    values = []
    for p in peers:
        v = _safe(lookup.get(p, {}).get(metric))
        if not np.isnan(v):
            values.append(v)
    if not values:
        return pd.Series(dtype=float)
    s   = pd.Series(values, dtype=float)
    cap = s.quantile(DQ["winsorise_pct"])
    return s.clip(upper=cap)


def score_one_metric(h_val: float, peer_med: float, lower_is_cheaper: bool) -> float:
    """
    premium/discount vs peer median → score 1–10.
    Exact same logic as notebook's score_one_metric().
    """
    if np.isnan(h_val) or np.isnan(peer_med) or peer_med == 0:
        return np.nan
    prem_disc = (h_val / peer_med) - 1.0
    bands     = SCORING_BANDS_MULTIPLE if lower_is_cheaper else SCORING_BANDS_QUALITY
    for upper, score in bands:
        if prem_disc < upper:
            return float(score)
    return 1.0


def compute_composite(metric_scores: dict, template: str) -> float:
    """
    Weighted composite from metric scores. Renormalises weights if some NaN.
    Returns 0–10 scale (same as notebook).
    """
    weights   = TEMPLATE_WEIGHTS.get(template, {})
    available = {m: s for m, s in metric_scores.items()
                 if m in weights and not np.isnan(s)}
    if not available:
        return np.nan
    total_w = sum(weights[m] for m in available)
    if total_w == 0:
        return np.nan
    return sum(weights[m] * available[m] for m in available) / total_w


def get_label(score: float) -> str:
    """Map composite score (0–10) to label string."""
    if np.isnan(score):
        return "⚪ No Data"
    for threshold, label, _ in SCORE_LABELS:
        if score <= threshold:
            return label
    return "⚪ No Data"


# =============================================================================
# BLOCK 12 — MAIN VALUATION ENGINE  (Module 5.3 main loop)
# =============================================================================

def run_notebook_valuation(
    holdings_df,
    progress_callback=None,
) -> dict:
    """
    Full notebook M5.1 + M5.2 + M5.3 scoring pipeline.

    Returns verdicts dict compatible with downstream modules (M5C, M5D, M5E, PDF):
    {
        ticker: {
            "score":            composite * 10,      # 0-100 for compatibility
            "score_10":         composite,            # 0-10 notebook scale
            "label":            "💎 Cheap",
            "template":         "ENERGY_EXCHANGE",
            "flags":            [...],
            "breakdown":        [(name, score, max_pts), ...],
            "peer_med":         {"pe": ..., "pb": ..., ...},
            "metric_details":   [...],               # full per-metric table
            "data_completeness": 100.0,
            "valid_metrics":    4,
            "fw":               {},
        }
    }
    """
    if progress_callback:
        progress_callback("Fetching peer data from yfinance for all holdings + peers...")

    # Step 1: Collect holding tickers from holdings_df
    holding_tickers = [
        t for t in holdings_df["Ticker"].tolist()
        if t not in SKIP_VALUATION and t in HOLDING_TEMPLATES
    ]

    # Step 2: Build TICKER_LOOKUP (M5.2 equivalent)
    if progress_callback:
        progress_callback("Fetching fundamentals from yfinance (holdings + all peers)...")

    def _fetch_cb(ticker):
        if progress_callback:
            progress_callback(f"  fetched {ticker}")

    lookup = build_ticker_lookup(holding_tickers, _fetch_cb)

    verdicts = {}

    # Step 3: Score each holding (M5.3 equivalent)
    for ticker in holdings_df["Ticker"].tolist():

        if ticker in SKIP_VALUATION:
            verdicts[ticker] = {
                "score": None, "score_10": None, "label": "ETF",
                "template": "ETF", "flags": [], "breakdown": [],
                "peer_med": {}, "metric_details": [], "data_completeness": 0,
                "valid_metrics": 0, "fw": {},
            }
            continue

        if ticker not in HOLDING_TEMPLATES:
            verdicts[ticker] = {
                "score": None, "score_10": None, "label": "⚪ No Data",
                "template": "UNKNOWN", "flags": ["Ticker not in scoring universe"],
                "breakdown": [], "peer_med": {}, "metric_details": [],
                "data_completeness": 0, "valid_metrics": 0, "fw": {},
            }
            continue

        template, company = HOLDING_TEMPLATES[ticker]
        metrics   = VALUATION_TEMPLATES.get(template, [])
        h_data    = lookup.get(ticker, {})
        flag_note = FLAG_NOTES.get(ticker, "")

        metric_scores   = {}
        metric_details  = []
        peer_med_dict   = {}
        valid_count     = 0

        for metric in metrics:
            meta      = METRIC_META.get(metric, {})
            lower_ok  = meta.get("lower_is_cheaper", True)
            label_str = meta.get("label", metric)
            weight    = TEMPLATE_WEIGHTS.get(template, {}).get(metric, 0)

            h_val     = _safe(h_data.get(metric))
            peer_vals = get_peer_values(ticker, metric, lookup)
            peer_n    = len(peer_vals)
            peer_med  = float(peer_vals.median()) if peer_n >= DQ["min_peers"] else np.nan

            score = score_one_metric(h_val, peer_med, lower_ok)

            prem_disc = np.nan
            if not np.isnan(h_val) and not np.isnan(peer_med) and peer_med != 0:
                prem_disc = ((h_val / peer_med) - 1.0) * 100.0

            metric_scores[metric] = score
            peer_med_dict[metric] = peer_med

            if peer_n >= DQ["min_peers"] and not np.isnan(h_val):
                valid_count += 1

            metric_details.append({
                "metric":     metric,
                "label":      label_str,
                "weight":     weight,
                "h_val":      round(h_val, 2) if not np.isnan(h_val) else None,
                "peer_med":   round(peer_med, 2) if not np.isnan(peer_med) else None,
                "prem_disc":  round(prem_disc, 1) if not np.isnan(prem_disc) else None,
                "score":      round(score, 0) if not np.isnan(score) else None,
                "peer_n":     peer_n,
                "lower_is_cheaper": lower_ok,
            })

        # Composite score (0-10 scale, same as notebook)
        comp_10 = compute_composite(metric_scores, template)
        label   = get_label(comp_10)

        # Convert to 0-100 for compatibility with downstream M5C/M5D/M5E/PDF
        comp_100 = round(comp_10 * 10, 1) if not np.isnan(comp_10) else None

        data_completeness = (valid_count / len(metrics) * 100) if metrics else 0

        # Build breakdown list for radar chart (normalize each metric score to 0-100)
        breakdown = []
        for metric in metrics:
            s   = metric_scores.get(metric, np.nan)
            wt  = TEMPLATE_WEIGHTS.get(template, {}).get(metric, 0)
            lbl = METRIC_META.get(metric, {}).get("label", metric)
            if not np.isnan(s):
                breakdown.append((lbl, round(s, 1), 10.0))

        # Flags
        flags = []
        if flag_note:
            flags.append(flag_note)
        if data_completeness < 70:
            flags.append(f"Low data coverage ({data_completeness:.0f}%) — scores may be less reliable")
        if comp_10 is not None and not np.isnan(comp_10):
            if comp_10 >= 7:
                flags.append(f"Score {comp_10:.1f}/10 — trading at meaningful discount to peers")
            elif comp_10 <= 3:
                flags.append(f"Score {comp_10:.1f}/10 — trading at significant premium to peers")

        # roe and pe for conviction engine (M5D)
        roe_val = _safe(h_data.get("roe"))
        pe_val  = _safe(h_data.get("pe"))

        verdicts[ticker] = {
            "score":            comp_100,
            "score_10":         round(comp_10, 2) if not np.isnan(comp_10) else None,
            "label":            label,
            "template":         template,
            "flags":            flags,
            "breakdown":        breakdown,
            "peer_med":         {k: (round(v, 2) if not np.isnan(v) else None)
                                 for k, v in peer_med_dict.items()},
            "metric_details":   metric_details,
            "data_completeness": round(data_completeness, 1),
            "valid_metrics":    valid_count,
            "roe":              roe_val,
            "pe":               pe_val,
            "fw":               {},
        }

        if progress_callback:
            sc_str = f"{comp_10:.2f}/10" if (comp_10 is not None and not np.isnan(comp_10)) else "N/A"
            progress_callback(f"  {company}: Score {sc_str} → {label}")

    return verdicts
