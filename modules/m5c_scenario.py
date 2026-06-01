# =============================================================================
# MODULE 5C — SCENARIO ENGINE
# Bull / Base / Bear scenarios per holding + portfolio role classification
# =============================================================================

import pandas as pd
import numpy as np

IS_FINANCIAL = {"PSU_BANK", "SFB", "NBFC"}

# Direct 1–2 year return estimates per scenario by business type
# (bear, bull) — realistic ranges for Indian listed equities
SCENARIO_RETURNS = {
    "COMMODITY_METAL":   {"bear": -40, "bull": +65},
    "AUTO_OEM_GLOBAL":   {"bear": -35, "bull": +55},
    "AUTO_ANCILLARY":    {"bear": -25, "bull": +45},
    "PSU_BANK":          {"bear": -30, "bull": +50},
    "SFB":               {"bear": -35, "bull": +60},
    "NBFC":              {"bear": -30, "bull": +55},
    "POWER_IPP":         {"bear": -25, "bull": +45},
    "RENEWABLE_SOLAR":   {"bear": -35, "bull": +70},
    "EXCHANGE_PLATFORM": {"bear": -20, "bull": +40},
    "TELECOM_INFRA":     {"bear": -20, "bull": +35},
    "ETF":               {"bear": -25, "bull": +40},
    "UNKNOWN":           {"bear": -25, "bull": +40},
}

CATALYSTS = {
    "COMMODITY_METAL": {
        "bear": "Global demand slowdown + commodity price collapse; China construction weakness",
        "bull": "Commodity supercycle; infrastructure spending surge; supply disruption",
    },
    "AUTO_OEM_GLOBAL": {
        "bear": "JLR volume decline; EV transition cost; India CV cycle downturn",
        "bull": "JLR order book execution; India PV/EV ramp; margin recovery",
    },
    "AUTO_ANCILLARY": {
        "bear": "OEM production cuts; raw material cost spike; customer concentration risk",
        "bull": "EV component content gain; OEM premiumization; export order wins",
    },
    "PSU_BANK": {
        "bear": "NPA spike from MSME/agriculture stress; NIM compression; credit cost surge",
        "bull": "Asset quality normalization; CASA improvement; credit growth acceleration",
    },
    "SFB": {
        "bear": "MFI stress; collection efficiency drop; deposit franchise vulnerability",
        "bull": "Graduation to full bank; CASA ramp; liability franchise build-out",
    },
    "NBFC": {
        "bear": "Funding cost spike; credit quality deterioration; regulatory tightening",
        "bull": "AUM growth acceleration; ROE trajectory improvement; franchise optionality",
    },
    "POWER_IPP": {
        "bear": "PPA renegotiation; fuel cost spike; regulatory tariff caps",
        "bull": "Merchant power price surge; capacity addition completion; RE transition",
    },
    "RENEWABLE_SOLAR": {
        "bear": "Module price spike; project execution delays; policy reversal",
        "bull": "Solar glass ASP expansion; capacity utilization ramp; export opportunity",
    },
    "EXCHANGE_PLATFORM": {
        "bear": "Volume market share loss; regulatory fee cap; new competition",
        "bull": "Volume growth acceleration; new product launches; market deepening",
    },
    "TELECOM_INFRA": {
        "bear": "Tenancy ratio decline; customer concentration (Jio/Airtel); debt burden",
        "bull": "5G tower densification; tenancy additions; lease rate escalations",
    },
    "ETF": {
        "bear": "NASDAQ correction; USD/INR depreciation; tech valuation compression",
        "bull": "AI/tech earnings acceleration; USD strength; global risk-on rally",
    },
    "UNKNOWN": {
        "bear": "Sector headwinds; execution risk; leverage concerns",
        "bull": "Sector tailwinds; re-rating potential; earnings beat",
    },
}


def _adjust_scenarios(base_bear: float, base_bull: float, d: dict, score: float) -> tuple:
    """
    Apply modifiers to base scenario returns based on company fundamentals.
    Returns (adjusted_bear, adjusted_bull).
    """
    bear, bull = base_bear, base_bull

    # Valuation adjustment
    if score is not None:
        if score >= 70:   bull = min(bull + 10, 120)   # cheap → more upside
        elif score <= 35: bear = max(bear - 10, -60)   # expensive → worse downside

    # Leverage adjustment
    nd_e = d.get("nd_ebitda") or 0
    de   = d.get("de_ratio") or 0
    if nd_e > 4 or de > 2.0:
        bear = max(bear - 8, -60)   # leverage amplifies downside

    # Growth momentum
    pat_cagr = d.get("pat_cagr_3y") or 0
    if pat_cagr > 20:  bull = min(bull + 8, 120)   # strong earnings momentum
    if pat_cagr < -5:  bear = max(bear - 8, -60)   # declining earnings

    # Quality buffer (high ROCE = less downside)
    roce = d.get("roce") or 0
    if roce > 25 and (d.get("de_ratio") or 0) < 0.5:
        bear = min(bear + 8, -5)   # quality buffer reduces downside

    return round(bear, 1), round(bull, 1)


def _assign_portfolio_role(d: dict, score: float, btype: str) -> str:
    """
    Deterministic portfolio role assignment per roadmap:
    COMPOUNDER / CYCLICAL / TURNAROUND / VALUE_TRAP / TACTICAL
    """
    roce     = d.get("roce") or 0
    de       = d.get("de_ratio") or 0
    pat_cagr = d.get("pat_cagr_3y") or 0
    nd_e     = d.get("nd_ebitda") or 0

    if score is not None and score < 35 and nd_e > 4 and pat_cagr < 0:
        return "VALUE_TRAP"

    if (btype not in IS_FINANCIAL and
            roce > 18 and de < 0.5 and pat_cagr > 12 and
            score is not None and score >= 55):
        return "COMPOUNDER"

    if btype in ("COMMODITY_METAL", "POWER_IPP", "RENEWABLE_SOLAR"):
        return "CYCLICAL"

    if score is not None and 30 <= score <= 55 and pat_cagr > 0:
        return "TURNAROUND"

    return "TACTICAL"


def run_scenario_engine(
    clean_data: dict,
    verdicts: dict,
    holdings_df: pd.DataFrame,
) -> dict:
    """
    Generate bull/base/bear scenarios for every holding.
    Returns: {ticker: scenario_dict}
    """
    scenarios = {}

    for _, row in holdings_df.iterrows():
        ticker = row["Ticker"]
        d      = clean_data.get(ticker) or {}
        v      = verdicts.get(ticker) or {}
        btype  = d.get("btype", "UNKNOWN")
        score  = v.get("score")
        cats   = CATALYSTS.get(btype, CATALYSTS["UNKNOWN"])

        live_price = d.get("live_price") or row.get("Current Price") or 0

        # Base scenario returns
        base_returns = SCENARIO_RETURNS.get(btype, SCENARIO_RETURNS["UNKNOWN"])
        base_bear    = float(base_returns["bear"])
        base_bull    = float(base_returns["bull"])

        # Apply fundamental modifiers
        bear_ret, bull_ret = _adjust_scenarios(base_bear, base_bull, d, score)

        # Implied prices
        bear_price = round(live_price * (1 + bear_ret / 100), 2) if live_price > 0 else None
        bull_price = round(live_price * (1 + bull_ret / 100), 2) if live_price > 0 else None

        # Asymmetry ratio
        asymmetry = round(abs(bull_ret) / abs(bear_ret), 2) if bear_ret != 0 else 1.0

        # Portfolio role
        role = _assign_portfolio_role(d, score, btype)

        scenarios[ticker] = {
            "bear": {
                "return_pct":    bear_ret,
                "implied_price": bear_price,
                "catalyst":      cats["bear"],
            },
            "base": {
                "return_pct":    0.0,
                "implied_price": round(live_price, 2) if live_price else None,
                "catalyst":      f"Continuation; PAT CAGR {d.get('pat_cagr_3y') or 'N/A'}%",
            },
            "bull": {
                "return_pct":    bull_ret,
                "implied_price": bull_price,
                "catalyst":      cats["bull"],
            },
            "asymmetry_ratio": asymmetry,
            "portfolio_role":  role,
            "btype":           btype,
        }

    return scenarios
