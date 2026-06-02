# =============================================================================
# MODULE 5A — SCREENER DATA LAYER
# Parses Screener.in Excel exports for fundamental data
# =============================================================================

import pandas as pd
import numpy as np
import openpyxl
import yfinance as yf
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

IS_FINANCIAL = {"PSU_BANK", "SFB", "NBFC"}
IS_ASSET_LIGHT = {"EXCHANGE_PLATFORM"}

# =============================================================================
# MANUAL BANK METRICS — yfinance doesn't provide NIM, GNPA, CASA
# Source: latest annual reports / RBI data
# =============================================================================
MANUAL_BANK_METRICS = {
    "PNB.NS":        {"gnpa_pct": 4.5,  "nim_pct": 3.0,  "casa_pct": 43.0},
    "BANKINDIA.NS":  {"gnpa_pct": 5.4,  "nim_pct": 2.9,  "casa_pct": 40.0},
    "IDFCFIRSTB.NS": {"gnpa_pct": 2.0,  "nim_pct": 6.0,  "casa_pct": 47.0},
    "EQUITASBNK.NS": {"gnpa_pct": 2.8,  "nim_pct": 8.5,  "casa_pct": 32.0},
    "JIOFIN.NS":     {"gnpa_pct": None, "nim_pct": None,  "casa_pct": None},
    "BBG.NS":        {"gnpa_pct": None, "nim_pct": None,  "casa_pct": None},
}

# Business model configuration
BUSINESS_MODEL = {
    "HINDCOPPER.NS": ("COMMODITY_METAL",    "Commodity Metal — Copper"),
    "HINDZINC.NS":   ("COMMODITY_METAL",    "Commodity Metal — Zinc/Silver"),
    "TATASTEEL.NS":  ("COMMODITY_METAL",    "Commodity Metal — Steel"),
    "TATAMOTORS.NS": ("AUTO_OEM_GLOBAL",    "Auto OEM — Global (JLR + India)"),
    "GABRIEL.NS":    ("AUTO_ANCILLARY",     "Auto Ancillary — Ride Control"),
    "PNB.NS":        ("PSU_BANK",           "PSU Bank — Mid-size"),
    "BANKINDIA.NS":  ("PSU_BANK",           "PSU Bank — Mid-size"),
    "IDFCFIRSTB.NS": ("SFB",               "Small Finance Bank"),
    "EQUITASBNK.NS": ("SFB",               "Small Finance Bank"),
    "JIOFIN.NS":     ("NBFC",              "NBFC — New Age Financial Services"),
    "BBG.NS":        ("NBFC",              "NBFC — MF Distribution"),
    "ADANIPOWER.NS": ("POWER_IPP",         "Power — Independent Power Producer"),
    "BORORENEW.NS":  ("RENEWABLE_SOLAR",   "Renewable Energy — Solar Glass"),
    "IEX.NS":        ("EXCHANGE_PLATFORM", "Exchange Platform — Power Trading"),
    "INDUSTOWER.NS": ("TELECOM_INFRA",     "Telecom Infrastructure — Tower Co"),
    "MON100.NS":     ("ETF",              "ETF — NASDAQ 100"),
}


def parse_screener_annual(filepath: str) -> dict:
    """
    Parse a Screener.in Excel export.
    Uses fixed row boundaries:
      META     : rows 5-14
      P&L      : rows 16-39  (Annual only)
      B/S      : rows 55-79
      Cash Flow: rows 80-89
      Derived  : rows 92-93
    """
    wb = openpyxl.load_workbook(filepath, data_only=True)
    ws = wb["Data Sheet"]
    rows = list(ws.iter_rows(min_row=1, max_row=93, values_only=True))

    def parse_float(v):
        try:
            if v is not None and str(v).strip() not in ("", "—"):
                return float(str(v).replace(",", ""))
        except Exception:
            pass
        return None

    def read_section(row_start, row_end):
        result = {}
        for row in rows[row_start - 1: row_end]:
            label = str(row[0]).strip() if row[0] else ""
            if not label or label == "—":
                continue
            if "Report Date" in label:
                continue
            if any(s in label.upper() for s in
                   ["PROFIT & LOSS", "BALANCE SHEET", "CASH FLOW",
                    "QUARTERS", "META", "PRICE:", "DERIVED:"]):
                continue
            vals = [parse_float(v) for v in row[1:]]
            clean = [v for v in vals if v is not None]
            if clean:
                result[label] = clean
        return result

    def read_single(row_start, row_end):
        result = {}
        for row in rows[row_start - 1: row_end]:
            label = str(row[0]).strip() if row[0] else ""
            if not label or label == "—":
                continue
            if any(s in label.upper() for s in
                   ["META", "PRICE:", "DERIVED:", "COMPANY NAME",
                    "LATEST VERSION", "CURRENT VERSION"]):
                continue
            for v in row[1:]:
                fv = parse_float(v)
                if fv is not None:
                    result[label] = fv
                    break
        return result

    pl = read_section(16, 39)
    bs = read_section(55, 79)
    cf = read_section(80, 89)
    meta = read_single(5, 14)
    derived = read_single(92, 93)

    return {"pl": pl, "bs": bs, "cf": cf, "meta": meta, "derived": derived}


def extract_fundamentals(ticker: str, filepath: str, btype: str) -> dict:
    """
    Extract and validate fundamental metrics from a Screener Excel file.
    Returns a rich dict of financial KPIs.
    """
    penalties = []

    try:
        s = parse_screener_annual(filepath)
    except Exception as e:
        return {"error": str(e), "confidence": 0, "ticker": ticker, "btype": btype}

    pl, bs, cf, meta, derived = s["pl"], s["bs"], s["cf"], s["meta"], s["derived"]

    def last_direct(section, *keys):
        for key in keys:
            vals = [v for v in section.get(key, []) if v is not None]
            if vals:
                return vals[-1]
        return None

    def cagr_calc(section, *keys, years=3):
        for key in keys:
            vals = [v for v in section.get(key, []) if v is not None]
            if vals and len(vals) >= 2:
                n = min(years, len(vals) - 1)
                st, en = vals[-(n + 1)], vals[-1]
                if st and st > 0 and en:
                    return round(((en / st) ** (1 / n) - 1) * 100, 1)
        return None

    def validate(val, lo, hi, name, penalty=8):
        if val is None:
            penalties.append((name, penalty, "missing"))
            return None
        if lo <= val <= hi:
            return round(val, 3)
        penalties.append((name, penalty, f"out_of_range:{val:.2f}"))
        return None

    # P&L
    revenue = last_direct(pl, "Sales")
    pat = last_direct(pl, "Net profit", "Net Profit")
    pbt = last_direct(pl, "Profit before tax")
    tax = last_direct(pl, "Tax") or 0
    interest = last_direct(pl, "Interest") or 0
    deprec = last_direct(pl, "Depreciation") or 0
    other_inc = last_direct(pl, "Other Income") or 0
    op_profit = last_direct(pl, "Operating Profit")

    if revenue is None:
        penalties.append(("Revenue", 20, "missing"))
    if pat is None:
        penalties.append(("PAT", 15, "missing"))

    # EBITDA reconstruction
    ebitda = None
    if pbt is not None and revenue:
        ebitda = pbt + tax + interest + deprec - other_inc
    if (ebitda is None or ebitda <= 0) and op_profit and op_profit > 0:
        ebitda = op_profit
    if btype == "EXCHANGE_PLATFORM" and pbt is not None:
        ebitda = pbt + tax + interest + deprec

    # Balance Sheet
    eq_cap = last_direct(bs, "Equity Share Capital") or 0
    reserves = last_direct(bs, "Reserves") or 0
    borrowings = last_direct(bs, "Borrowings") or 0
    cash = last_direct(bs, "Cash & Bank") or 0
    net_block = last_direct(bs, "Net Block") or 0
    cwip = last_direct(bs, "Capital Work in Progress") or 0
    other_lib = last_direct(bs, "Other Liabilities") or 0
    total_assets = last_direct(bs, "Total") or 0

    # Cash Flow
    cfo = last_direct(cf, "Cash from Operating Activity", "Cash from Operating Activ")
    capex = last_direct(cf, "Cash from Investing Activity", "Cash from Investing Activ")

    # Meta
    mcap_screener = meta.get("Market Capitalization") or meta.get("Market Cap")
    price_screener = meta.get("Current Price")

    # Shares
    shares_cr = (
        derived.get("Adjusted Equity Shares in Cr")
        or derived.get("Adjusted Equity Shares in")
    )
    if not shares_cr:
        shares_abs = last_direct(bs, "No. of Equity Shares")
        if shares_abs and shares_abs > 1e5:
            shares_cr = round(shares_abs / 1e7, 4)
    if not shares_cr and price_screener and mcap_screener and price_screener > 0:
        shares_cr = round(mcap_screener / price_screener, 4)

    # Derived
    total_equity = max(eq_cap + reserves, 1)
    net_debt = borrowings - cash

    rev_base = (revenue or 1)
    if btype == "EXCHANGE_PLATFORM":
        rev_base = (revenue or 0) + max(other_inc, 0)

    # EBITDA margin validation
    ebitda_margin = None
    if ebitda is not None and rev_base > 0:
        raw_m = ebitda / rev_base * 100
        if btype in IS_FINANCIAL:
            ebitda = None
        elif btype == "EXCHANGE_PLATFORM":
            ebitda_margin = validate(raw_m, 40, 95, "EBITDA_margin", 0)
        elif btype == "COMMODITY_METAL":
            ebitda_margin = validate(raw_m, 5, 75, "EBITDA_margin", 5)
        elif btype == "POWER_IPP":
            ebitda_margin = validate(raw_m, 10, 80, "EBITDA_margin", 5)
        elif btype == "TELECOM_INFRA":
            ebitda_margin = validate(raw_m, 20, 75, "EBITDA_margin", 5)
        else:
            ebitda_margin = validate(raw_m, 2, 65, "EBITDA_margin", 5)
        if ebitda_margin is None:
            ebitda = None

    # PAT margin
    pat_margin = None
    if pat is not None and revenue and revenue > 0:
        raw_pm = pat / revenue * 100
        if btype in IS_FINANCIAL:
            pat_margin = validate(raw_pm, 0, 55, "PAT_margin", 0)
        else:
            pat_margin = validate(raw_pm, -60, 60, "PAT_margin", 0)

    # ROCE
    roce = ebit = None
    if btype not in IS_FINANCIAL and ebitda is not None and deprec is not None:
        ebit = ebitda - deprec
        nwc = max(total_assets - net_block - cwip - cash - borrowings - other_lib, 0)
        ce = net_block + cwip + nwc
        if ce <= 0:
            ce = max(total_equity + borrowings, 1)
        raw_roce = ebit / ce * 100
        roce_ceil = 200 if btype in IS_ASSET_LIGHT else 70
        roce = validate(raw_roce, -30, roce_ceil, "ROCE", 5)

    # ROE
    roe = None
    if pat is not None and total_equity > 0:
        raw_roe = pat / total_equity * 100
        roe = validate(raw_roe, -80, 80, "ROE", 3)

    # D/E
    de_ratio = None
    if btype not in IS_FINANCIAL and total_equity > 0:
        de_raw = borrowings / total_equity
        de_ratio = round(de_raw, 3) if de_raw <= 30 else None

    # ND/EBITDA
    nd_ebitda = None
    if ebitda and ebitda > 0 and btype not in IS_FINANCIAL:
        nd_e = net_debt / ebitda
        nd_ebitda = round(nd_e, 2) if -15 < nd_e < 30 else None

    # FCF
    fcf = round(cfo + capex, 2) if (cfo is not None and capex is not None) else None

    # Growth CAGRs
    rev_cagr = cagr_calc(pl, "Sales", years=3)
    pat_cagr = cagr_calc(pl, "Net profit", "Net Profit", years=3)

    # BVPS
    bvps = round(total_equity / shares_cr, 2) if (shares_cr and shares_cr > 0) else None

    # Accounting flags
    accounting_flags = []
    if cfo is not None and pat is not None and pat > 0:
        conv = cfo / pat
        if conv < 0.5:
            accounting_flags.append(f"Weak CFO/PAT ({conv:.1f}x) — earnings quality risk")
    if nd_ebitda is not None and nd_ebitda > 4:
        accounting_flags.append(f"High leverage ND/EBITDA {nd_ebitda:.1f}x")
    if other_inc and pat is not None and pat > 0 and other_inc > 0:
        ratio = other_inc / pat
        if ratio > 0.4:
            accounting_flags.append(f"Other income = {ratio*100:.0f}% of PAT")
    if btype == "COMMODITY_METAL" and ebitda_margin and ebitda_margin > 45:
        accounting_flags.append(f"Margins {ebitda_margin:.0f}% — near cyclical peak")

    confidence = max(0, min(100, 100 - sum(p for _, p, _ in penalties)))

    return {
        "ticker": ticker, "btype": btype,
        "confidence": confidence,
        "accounting_flags": accounting_flags,
        "data_issues": penalties,
        # P&L
        "revenue": revenue, "rev_base": rev_base,
        "ebitda": ebitda, "ebitda_margin": ebitda_margin,
        "pat": pat, "pat_margin": pat_margin,
        "interest": interest, "depreciation": deprec,
        "other_income": other_inc, "pbt": pbt, "op_profit": op_profit,
        # Balance Sheet
        "total_equity": round(total_equity, 2),
        "borrowings": borrowings, "cash": cash,
        "net_debt": round(net_debt, 2),
        "total_assets": total_assets,
        "net_block": net_block, "cwip": cwip,
        # Cash Flow
        "cfo": cfo, "capex": capex, "fcf": fcf,
        # Returns
        "roce": roce, "roe": roe,
        "de_ratio": de_ratio, "nd_ebitda": nd_ebitda,
        # Growth
        "rev_cagr_3y": rev_cagr, "pat_cagr_3y": pat_cagr,
        # Per-share
        "shares_cr": shares_cr, "bvps": bvps,
        # Market data (from screener)
        "mcap_screener": mcap_screener,
        "price_screener": price_screener,
        # Live (filled by enrich_price)
        "live_price": None, "live_mcap": None,
        "pe": None, "pb": None, "ev_ebitda": None, "ev": None,
    }


def enrich_price(d: dict, ticker: str) -> dict:
    """Fetch live price and compute market multiples."""
    price = 0
    try:
        fi = yf.Ticker(ticker).fast_info
        price = float(fi.get("last_price") or fi.get("previous_close") or 0)
    except Exception:
        pass

    if price <= 0:
        price = d.get("price_screener") or 0

    shares = d.get("shares_cr")
    mcap = (
        round(price * shares, 2)
        if (price > 0 and shares and shares > 0)
        else d.get("mcap_screener")
    )

    d["live_price"] = round(price, 2) if price else None
    d["live_mcap"] = round(mcap, 2) if mcap else None

    btype = d["btype"]
    pat = d.get("pat")
    ebitda = d.get("ebitda")
    equity = d.get("total_equity", 1)
    nd = d.get("net_debt", 0) or 0
    ev = round(mcap + nd, 2) if mcap else None
    d["ev"] = ev

    if mcap and pat and pat > 0:
        pe_raw = mcap / pat
        d["pe"] = round(pe_raw, 1) if 0 < pe_raw < 500 else None
    else:
        d["pe"] = None

    if mcap and equity and equity > 0:
        pb_raw = mcap / equity
        d["pb"] = round(pb_raw, 2) if 0 < pb_raw < 100 else None
    else:
        d["pb"] = None

    if btype in IS_FINANCIAL:
        d["ev_ebitda"] = None
    elif ev and ebitda and ebitda > 0:
        ev_e = ev / ebitda
        d["ev_ebitda"] = round(ev_e, 1) if 0 < ev_e < 200 else None
    else:
        d["ev_ebitda"] = None

    return d


def fetch_fundamentals_yfinance(ticker: str, btype: str, bname: str) -> dict:
    """
    Auto-fetch fundamentals from yfinance.info when no Screener Excel is available.
    Returns the same dict format as extract_fundamentals() so downstream is identical.
    Quality: less accurate than Screener for Indian stocks, but works without uploads.
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}

    def _f(key, fallback=None):
        val = info.get(key, fallback)
        try:
            return float(val) if val is not None else None
        except Exception:
            return None

    revenue    = _f("totalRevenue")
    ebitda     = _f("ebitda")
    pat        = _f("netIncomeToCommon")
    total_debt = _f("totalDebt") or 0
    cash       = _f("totalCash") or 0
    mcap       = _f("marketCap")
    roe        = _f("returnOnEquity")
    roa        = _f("returnOnAssets")
    pe         = _f("trailingPE")
    pb         = _f("priceToBook")
    ev         = _f("enterpriseValue")
    rev_growth = _f("revenueGrowth")
    shares     = _f("sharesOutstanding")

    # Derived
    net_debt   = total_debt - cash
    ev_ebitda  = round(ev / ebitda, 1) if (ev and ebitda and ebitda > 0 and btype not in IS_FINANCIAL) else None
    ebitda_m   = round(ebitda / revenue * 100, 1) if (ebitda and revenue and revenue > 0 and btype not in IS_FINANCIAL) else None
    pat_m      = round(pat / revenue * 100, 1) if (pat and revenue and revenue > 0) else None
    roe_pct    = round(roe * 100, 1) if roe is not None else None
    shares_cr  = round(shares / 1e7, 2) if shares else None
    rev_cagr   = round(rev_growth * 100, 1) if rev_growth is not None else None

    # Convert revenue from absolute to Cr (divide by 1e7 for Indian stocks reported in units)
    # yfinance returns revenue in the currency unit (INR) — divide by 1e7 to get Crores
    if revenue and revenue > 1e9:   # already in INR units
        revenue_cr = round(revenue / 1e7, 2)
    elif revenue:
        revenue_cr = round(revenue, 2)  # already in Cr
    else:
        revenue_cr = None

    if ebitda and ebitda > 1e9:
        ebitda_cr = round(ebitda / 1e7, 2)
    elif ebitda:
        ebitda_cr = round(ebitda, 2)
    else:
        ebitda_cr = None

    if pat and abs(pat) > 1e9:
        pat_cr = round(pat / 1e7, 2)
    elif pat:
        pat_cr = round(pat, 2)
    else:
        pat_cr = None

    if mcap and mcap > 1e9:
        mcap_cr = round(mcap / 1e7, 2)
    else:
        mcap_cr = mcap

    # Bank-specific manual metrics
    bank_metrics = MANUAL_BANK_METRICS.get(ticker, {})

    # Confidence: lower since yfinance data quality is variable for Indian stocks
    confidence = 60 if (revenue_cr and pat_cr) else 35

    result = {
        "ticker": ticker, "btype": btype, "bname": bname,
        "confidence": confidence,
        "data_source": "yfinance",
        "accounting_flags": ["Data from yfinance.info — upload Screener Excel for higher accuracy"],
        "data_issues": [],
        # P&L
        "revenue": revenue_cr, "rev_base": revenue_cr or 1,
        "ebitda": ebitda_cr, "ebitda_margin": ebitda_m,
        "pat": pat_cr, "pat_margin": pat_m,
        "interest": None, "depreciation": None, "other_income": None,
        "pbt": None, "op_profit": None,
        # Balance Sheet
        "total_equity": None, "borrowings": round(total_debt / 1e7, 2) if total_debt > 1e6 else total_debt,
        "cash": round(cash / 1e7, 2) if cash > 1e6 else cash,
        "net_debt": round(net_debt / 1e7, 2) if abs(net_debt) > 1e6 else net_debt,
        "total_assets": None, "net_block": None, "cwip": None,
        # Cash Flow
        "cfo": None, "capex": None, "fcf": None,
        # Returns
        "roce": None, "roe": roe_pct,
        "de_ratio": None, "nd_ebitda": None,
        # Growth
        "rev_cagr_3y": rev_cagr, "pat_cagr_3y": None,
        # Per-share
        "shares_cr": shares_cr, "bvps": None,
        # Market
        "mcap_screener": mcap_cr, "price_screener": _f("currentPrice") or _f("previousClose"),
        # Live (filled by enrich_price)
        "live_price": None, "live_mcap": None,
        "pe": round(pe, 1) if pe and 0 < pe < 500 else None,
        "pb": round(pb, 2) if pb and 0 < pb < 100 else None,
        "ev_ebitda": ev_ebitda, "ev": round(ev / 1e7, 2) if ev and ev > 1e6 else ev,
        # Bank-specific
        "gnpa_pct":  bank_metrics.get("gnpa_pct"),
        "nim_pct":   bank_metrics.get("nim_pct"),
        "casa_pct":  bank_metrics.get("casa_pct"),
    }
    return result


def run_data_layer(
    holdings_df,
    screener_files: dict,
    progress_callback=None,
) -> dict:
    """
    Main entry point: iterate all tickers, parse Excel, enrich prices.
    screener_files: {ticker: filepath_or_None}
    Returns clean_data: {ticker: fundamentals_dict}
    """
    clean_data = {}

    for ticker, fpath in screener_files.items():
        btype, bname = BUSINESS_MODEL.get(ticker, ("UNKNOWN", ticker))
        row = holdings_df[holdings_df["Ticker"] == ticker]
        company = row["Company"].iloc[0][:28] if not row.empty else ticker

        if fpath is None:
            if btype == "ETF":
                # ETFs have no fundamentals — skip entirely
                clean_data[ticker] = {
                    "ticker": ticker, "btype": btype, "confidence": 100,
                    "accounting_flags": [], "data_issues": [],
                    "live_price": None, "live_mcap": None,
                    "pe": None, "pb": None, "ev_ebitda": None,
                    "revenue": None, "ebitda": None, "pat": None,
                    "bname": bname, "data_source": "none",
                    "gnpa_pct": None, "nim_pct": None, "casa_pct": None,
                }
                if progress_callback:
                    progress_callback(ticker, company, "ETF — skipped")
                continue
            else:
                # No Screener file — auto-fetch from yfinance.info as fallback
                if progress_callback:
                    progress_callback(ticker, company, "fetching via yfinance...")
                d = fetch_fundamentals_yfinance(ticker, btype, bname)
                d = enrich_price(d, ticker)
                clean_data[ticker] = d
                if progress_callback:
                    src = "yfinance"
                    pe_str = f"PE:{d['pe']}" if d.get("pe") else ""
                    progress_callback(ticker, company, f"ok [{src}] {pe_str}")
                continue

        d = extract_fundamentals(ticker, fpath, btype)

        if "error" in d:
            if progress_callback:
                progress_callback(ticker, company, f"error: {d['error'][:40]}")
            clean_data[ticker] = None
            continue

        d = enrich_price(d, ticker)
        d["bname"] = bname
        clean_data[ticker] = d

        if progress_callback:
            pe_str = f"PE:{d['pe']}" if d["pe"] else ""
            pb_str = f"PB:{d['pb']}" if d["pb"] else ""
            progress_callback(ticker, company, f"ok {pe_str} {pb_str}")

    return clean_data
