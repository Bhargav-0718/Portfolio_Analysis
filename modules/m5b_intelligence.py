# =============================================================================
# MODULE 5B — INTELLIGENCE LAYER
# Peer comparison, relative valuation, composite scoring, thesis seeds
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

IS_FINANCIAL = {"PSU_BANK", "SFB", "NBFC"}
IS_ASSET_LIGHT = {"EXCHANGE_PLATFORM"}

C = {
    "bg": "#FFFFFF", "panel": "#F8F9FC", "border": "#DDE3EE",
    "text": "#0F172A", "text2": "#475569", "text3": "#94A3B8",
    "blue": "#1D4ED8", "green": "#16A34A", "red": "#DC2626",
    "gold": "#D97706", "grid": "#E2E8F0",
}

# =============================================================================
# PEER MAP
# =============================================================================
PEER_MAP = {
    "HINDCOPPER.NS": {"primary": ["HINDALCO.NS", "VEDL.NS"], "secondary": ["NMDC.NS"]},
    "HINDZINC.NS":   {"primary": ["VEDL.NS", "HINDALCO.NS"], "secondary": ["NMDC.NS"]},
    "TATASTEEL.NS":  {"primary": ["JSWSTEEL.NS", "SAIL.NS"], "secondary": ["HINDALCO.NS"]},
    "TATAMOTORS.NS": {"primary": ["M&M.NS", "ASHOKLEY.NS"], "secondary": ["MARUTI.NS"]},
    "GABRIEL.NS":    {"primary": ["SUPRAJIT.NS", "ENDURANCE.NS"], "secondary": ["MOTHERSON.NS"]},
    "PNB.NS":        {"primary": ["CANBK.NS", "UNIONBANK.NS"], "secondary": ["BANKBARODA.NS"]},
    "BANKINDIA.NS":  {"primary": ["CANBK.NS", "MAHABANK.NS"], "secondary": ["UNIONBANK.NS"]},
    "IDFCFIRSTB.NS": {"primary": ["AUBANK.NS", "UJJIVANSFB.NS"], "secondary": ["EQUITASBNK.NS"]},
    "EQUITASBNK.NS": {"primary": ["UJJIVANSFB.NS", "SURYODAY.NS"], "secondary": ["AUBANK.NS"]},
    "JIOFIN.NS":     {"primary": ["BAJFINANCE.NS", "CHOLAFIN.NS"], "secondary": ["BAJAJFINSV.NS"]},
    "BBG.NS":        {"primary": ["CAMS.NS", "KFINTECH.NS"], "secondary": ["CDSL.NS"]},
    "ADANIPOWER.NS": {"primary": ["NTPC.NS", "JSWENERGY.NS"], "secondary": ["TATAPOWER.NS"]},
    "BORORENEW.NS":  {"primary": ["WEBSOL.NS", "WAAREEENE.NS"], "secondary": ["BOROSIL.NS"]},
    "IEX.NS":        {"primary": ["BSE.NS", "MCX.NS"], "secondary": ["CDSL.NS"]},
    "INDUSTOWER.NS": {"primary": ["BHARTIARTL.NS", "TATACOMM.NS"], "secondary": ["RAILTEL.NS"]},
    "MON100.NS":     {"primary": [], "secondary": []},
}

# =============================================================================
# METRIC FRAMEWORKS — what to measure per business type
# =============================================================================
METRIC_FRAMEWORK = {
    "COMMODITY_METAL":   {"primary": "EV/EBITDA", "use_ev_ebitda": True,  "use_pb": False, "use_pe": True,  "cheap": 5.0,  "fair": 8.0,  "exp": 12.0, "bank": False},
    "AUTO_OEM_GLOBAL":   {"primary": "EV/EBITDA", "use_ev_ebitda": True,  "use_pb": False, "use_pe": True,  "cheap": 6.0,  "fair": 10.0, "exp": 16.0, "bank": False},
    "AUTO_ANCILLARY":    {"primary": "PE",         "use_ev_ebitda": True,  "use_pb": False, "use_pe": True,  "cheap": 15.0, "fair": 25.0, "exp": 40.0, "bank": False},
    "PSU_BANK":          {"primary": "P/B",        "use_ev_ebitda": False, "use_pb": True,  "use_pe": True,  "cheap": 0.4,  "fair": 0.9,  "exp": 1.4,  "bank": True},
    "SFB":               {"primary": "P/B",        "use_ev_ebitda": False, "use_pb": True,  "use_pe": True,  "cheap": 0.8,  "fair": 1.8,  "exp": 3.0,  "bank": True},
    "NBFC":              {"primary": "P/B",        "use_ev_ebitda": False, "use_pb": True,  "use_pe": True,  "cheap": 1.0,  "fair": 3.0,  "exp": 6.0,  "bank": True},
    "POWER_IPP":         {"primary": "EV/EBITDA",  "use_ev_ebitda": True,  "use_pb": True,  "use_pe": False, "cheap": 5.0,  "fair": 9.0,  "exp": 14.0, "bank": False},
    "RENEWABLE_SOLAR":   {"primary": "EV/EBITDA",  "use_ev_ebitda": True,  "use_pb": False, "use_pe": False, "cheap": 8.0,  "fair": 15.0, "exp": 25.0, "bank": False},
    "EXCHANGE_PLATFORM": {"primary": "PE",         "use_ev_ebitda": True,  "use_pb": False, "use_pe": True,  "cheap": 20.0, "fair": 35.0, "exp": 55.0, "bank": False},
    "TELECOM_INFRA":     {"primary": "EV/EBITDA",  "use_ev_ebitda": True,  "use_pb": False, "use_pe": False, "cheap": 5.0,  "fair": 9.0,  "exp": 13.0, "bank": False},
    "ETF":               {"primary": "NAV",        "use_ev_ebitda": False, "use_pb": False, "use_pe": False, "cheap": 0,    "fair": 0,    "exp": 0,    "bank": False},
    "UNKNOWN":           {"primary": "PE",         "use_ev_ebitda": False, "use_pb": False, "use_pe": True,  "cheap": 15.0, "fair": 25.0, "exp": 40.0, "bank": False},
}


def fetch_peer_multiples(ticker: str) -> dict:
    """Fetch validated multiples for a peer stock."""
    try:
        info = yf.Ticker(ticker).info
        pe   = info.get("trailingPE")
        pb   = info.get("priceToBook")
        ev_e = info.get("enterpriseToEbitda")
        roe  = info.get("returnOnEquity")

        if pe  and not (0 < pe  < 400): pe  = None
        if pb  and not (0 < pb  < 80):  pb  = None
        if ev_e and not (0 < ev_e < 150): ev_e = None
        if roe: roe = round(roe * 100, 1)

        return {
            "pe":       round(pe,  1) if pe  else None,
            "pb":       round(pb,  2) if pb  else None,
            "ev_ebitda": round(ev_e, 1) if ev_e else None,
            "roe":      roe,
        }
    except Exception:
        return None


def get_peer_medians(ticker: str) -> dict:
    """Fetch primary peers and return median multiples."""
    peer_list = PEER_MAP.get(ticker, {}).get("primary", [])
    all_pe, all_pb, all_ev, all_roe = [], [], [], []

    for peer in peer_list:
        m = fetch_peer_multiples(peer)
        if m:
            if m.get("pe"):  all_pe.append(m["pe"])
            if m.get("pb"):  all_pb.append(m["pb"])
            if m.get("ev_ebitda"): all_ev.append(m["ev_ebitda"])
            if m.get("roe"): all_roe.append(m["roe"])

    def med(lst):
        return round(float(np.median(lst)), 2) if lst else None

    return {
        "pe": med(all_pe), "pb": med(all_pb),
        "ev_ebitda": med(all_ev), "roe": med(all_roe),
        "peers_used": len(peer_list),
    }


def compute_score(d: dict, fw: dict, peer_med: dict) -> tuple:
    """
    5-factor composite score (0-100).
    Valuation(30) + Quality(25) + Growth(20) + BalanceSheet(15) + Profitability(10)
    Returns (score, label, breakdown_list)
    """
    primary = fw["primary"]
    btype = d["btype"]
    scores = []

    def score_multiple(val, peer_val, cheap_t, exp_t, max_pts=30):
        if val is None:
            return max_pts * 0.5
        if peer_val and peer_val > 0:
            disc = (peer_val - val) / peer_val
            return min(max_pts, max(0, max_pts * 0.5 + disc * max_pts * 0.8))
        mid = (cheap_t + exp_t) / 2
        if val <= cheap_t:    return max_pts * 0.93
        elif val <= mid:      return max_pts * 0.68
        elif val <= exp_t:    return max_pts * 0.38
        else:                 return max_pts * 0.12

    cheap, fair, exp = fw["cheap"], fw["fair"], fw["exp"]

    if primary == "EV/EBITDA":
        v_score = score_multiple(d.get("ev_ebitda"), peer_med.get("ev_ebitda"), cheap, exp)
    elif primary == "P/B":
        roe_adj = 0
        if d.get("roe") and peer_med.get("roe"):
            roe_adj = min(5, max(-5, (d["roe"] - peer_med["roe"]) * 0.25))
        v_score = score_multiple(d.get("pb"), peer_med.get("pb"), cheap, exp) + roe_adj
        v_score = min(30, max(0, v_score))
    elif primary == "PE":
        v_score = score_multiple(d.get("pe"), peer_med.get("pe"), cheap, exp)
    else:
        v_score = 15
    scores.append(("Valuation", round(v_score), 30))

    # Quality
    roce, roe = d.get("roce"), d.get("roe")
    if fw["bank"]:
        q_score = 23 if roe and roe >= 15 else (18 if roe and roe >= 12 else (12 if roe and roe >= 8 else (6 if roe and roe >= 4 else 2)))
    else:
        if roce is not None:
            n = min(roce, 200 if btype in IS_ASSET_LIGHT else 60)
            q_score = 24 if n >= 25 else (19 if n >= 18 else (13 if n >= 12 else (7 if n >= 6 else 3)))
        else:
            q_score = 10
    scores.append(("Quality", q_score, 25))

    # Growth
    rc = d.get("rev_cagr_3y") or 0
    pc = d.get("pat_cagr_3y") or 0
    g = (rc + pc) / 2 if (rc and pc) else max(rc, pc, 0)
    g_score = 19 if g >= 25 else (15 if g >= 18 else (11 if g >= 10 else (7 if g >= 5 else (4 if g >= 0 else 1))))
    scores.append(("Growth", g_score, 20))

    # Balance Sheet
    de = d.get("de_ratio") or 0
    nd_e = d.get("nd_ebitda")
    if fw["bank"]:
        bs_score = 9
    else:
        bs_score = 14 if de < 0.2 else (11 if de < 0.5 else (7 if de < 1.5 else (4 if de < 3.0 else 1)))
        if nd_e is not None:
            if nd_e < 0:   bs_score = min(15, bs_score + 3)
            elif nd_e > 4: bs_score = max(0, bs_score - 3)
            elif nd_e > 6: bs_score = max(0, bs_score - 5)
    scores.append(("Balance Sheet", bs_score, 15))

    # Profitability
    em = d.get("ebitda_margin") or 0
    pr_score = 7 if fw["bank"] else (9 if em >= 30 else (7 if em >= 20 else (5 if em >= 12 else (3 if em >= 5 else 1))))
    scores.append(("Profitability", pr_score, 10))

    total = sum(s for _, s, _ in scores)
    maxpts = sum(m for _, _, m in scores)
    comp = round(total / maxpts * 100, 1)

    conf = d.get("confidence", 100)
    if conf < 50:  comp = round(comp * 0.85, 1)
    elif conf < 70: comp = round(comp * 0.95, 1)

    label = (
        "DEEP VALUE" if comp >= 75 else
        "ATTRACTIVE" if comp >= 60 else
        "FAIR"       if comp >= 45 else
        "RICH"       if comp >= 30 else
        "EXPENSIVE"
    )
    return comp, label, scores


def generate_flags(d: dict, fw: dict, peer_med: dict, btype: str) -> list:
    """Generate investment signal flags as thesis seeds."""
    flags = []
    primary = fw["primary"]

    ev_e = d.get("ev_ebitda");   pe = d.get("pe");   pb = d.get("pb")
    peer_ev = peer_med.get("ev_ebitda"); peer_pe = peer_med.get("pe"); peer_pb = peer_med.get("pb")

    # Valuation flags
    if primary == "EV/EBITDA" and ev_e and peer_ev:
        disc = (peer_ev - ev_e) / peer_ev * 100
        if disc > 30:   flags.append(f"Trading {disc:.0f}% below peer EV/EBITDA median")
        elif disc > 15: flags.append(f"Moderate {disc:.0f}% discount to peer EV/EBITDA")
        elif disc < -20: flags.append(f"Trading {abs(disc):.0f}% PREMIUM to peer EV/EBITDA")

    if primary == "P/B" and pb and peer_pb:
        disc = (peer_pb - pb) / peer_pb * 100
        if disc > 25: flags.append(f"P/B {disc:.0f}% below sector peers")
        elif disc < -20: flags.append(f"P/B {abs(disc):.0f}% premium to peers")

    if primary == "PE" and pe and peer_pe:
        disc = (peer_pe - pe) / peer_pe * 100
        if disc > 25: flags.append(f"PE at {disc:.0f}% discount to peers")

    # Quality flags
    roce = d.get("roce"); roe = d.get("roe")
    if fw["bank"]:
        if roe and roe >= 15: flags.append(f"Strong ROE {roe:.1f}% — above 15% threshold")
        elif roe and roe < 8: flags.append(f"Weak ROE {roe:.1f}% — below cost of equity")
    else:
        if roce and roce >= 20: flags.append(f"High ROCE {roce:.1f}% — above-average capital efficiency")
        elif roce and roce < 8: flags.append(f"Low ROCE {roce:.1f}% — capital allocation concern")

    # Growth flags
    rc = d.get("rev_cagr_3y"); pc = d.get("pat_cagr_3y")
    if rc and rc > 15: flags.append(f"Revenue CAGR {rc:.1f}% (3yr) — above sector growth")
    if pc and pc > 20: flags.append(f"PAT CAGR {pc:.1f}% (3yr) — earnings momentum strong")
    if pc and pc < 0:  flags.append(f"PAT declining {pc:.1f}% CAGR — earnings quality concern")

    # Balance sheet flags
    nd_e = d.get("nd_ebitda"); de = d.get("de_ratio")
    if nd_e is not None and nd_e < 0: flags.append("Net cash position — balance sheet strength")
    if nd_e and nd_e > 5: flags.append(f"High leverage ND/EBITDA {nd_e:.1f}x — debt overhang risk")
    if de and de > 2.0:   flags.append(f"High D/E {de:.1f}x — elevated financial risk")

    # Accounting flags
    flags.extend(d.get("accounting_flags", []))

    return flags[:8]  # cap at 8 flags


def run_intelligence_layer(
    clean_data: dict,
    progress_callback=None,
) -> dict:
    """
    Main entry: score + flag every holding.
    Returns verdicts: {ticker: {score, label, breakdown, flags, peer_med}}
    """
    verdicts = {}

    for ticker, d in clean_data.items():
        if d is None:
            verdicts[ticker] = None
            continue

        btype = d.get("btype", "UNKNOWN")
        fw = METRIC_FRAMEWORK.get(btype, METRIC_FRAMEWORK["UNKNOWN"])

        if btype == "ETF":
            verdicts[ticker] = {
                "score": None, "label": "ETF",
                "breakdown": [], "flags": ["ETF — no fundamental analysis applicable"],
                "peer_med": {}, "fw": fw,
            }
            continue

        peer_med = get_peer_medians(ticker)
        score, label, breakdown = compute_score(d, fw, peer_med)
        flags = generate_flags(d, fw, peer_med, btype)

        verdicts[ticker] = {
            "score": score, "label": label,
            "breakdown": breakdown, "flags": flags,
            "peer_med": peer_med, "fw": fw,
        }

        if progress_callback:
            progress_callback(ticker, score, label)

    return verdicts


def plot_valuation_dashboard(
    clean_data: dict,
    verdicts: dict,
    holdings_df,
    report_date: str,
) -> plt.Figure:
    """
    Valuation summary chart: composite score bars + key multiples table.
    """
    # Build display rows
    rows = []
    for _, row in holdings_df.iterrows():
        ticker = row["Ticker"]
        d = clean_data.get(ticker)
        v = verdicts.get(ticker)
        if d is None or v is None:
            continue

        rows.append({
            "Company": row["Company"][:22],
            "Sector": row["Sector"][:18],
            "Score": v.get("score"),
            "Label": v.get("label", "—"),
            "PE": d.get("pe"),
            "PB": d.get("pb"),
            "EV/EBITDA": d.get("ev_ebitda"),
            "ROCE": d.get("roce"),
            "ROE": d.get("roe"),
            "Rev CAGR": d.get("rev_cagr_3y"),
        })

    if not rows:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "No valuation data available\n(upload Screener Excel files in sidebar)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11, color=C["text3"])
        return fig

    df = pd.DataFrame(rows).dropna(subset=["Score"]).sort_values("Score", ascending=True)

    fig = plt.figure(figsize=(18, 10), facecolor=C["bg"])
    fig.suptitle(f"Valuation & Scoring Dashboard  |  {report_date}",
                 fontsize=13, fontweight="bold", color=C["text"], y=0.98)
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.5)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    def style_ax(ax):
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["text2"], labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(C["border"])
        ax.spines["bottom"].set_color(C["border"])
        ax.set_axisbelow(True)

    # --- Left: Composite score bars ---
    style_ax(ax1)
    label_color = {
        "DEEP VALUE": "#15803D", "ATTRACTIVE": "#1D4ED8",
        "FAIR": "#D97706", "RICH": "#DC2626", "EXPENSIVE": "#7F1D1D", "ETF": "#64748B",
    }
    bar_colors = [label_color.get(row["Label"], C["blue"]) for _, row in df.iterrows()]
    ax1.barh(df["Company"], df["Score"], color=bar_colors, alpha=0.85, zorder=3)
    ax1.set_xlim(0, 105)
    for i, (_, row) in enumerate(df.iterrows()):
        ax1.text(row["Score"] + 1, i, f'{row["Label"]}  {row["Score"]:.0f}',
                 va="center", fontsize=7.5, color=C["text"])
    ax1.xaxis.grid(True, color=C["grid"], lw=0.5, zorder=0)
    ax1.yaxis.grid(False)
    ax1.set_xlabel("Composite Score (0–100)", fontsize=8)
    ax1.set_title("Investment Attractiveness Score", fontsize=9, color=C["text"])

    # Legend
    patches = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85, label=l)
               for l, c in label_color.items()]
    ax1.legend(handles=patches, fontsize=7, loc="lower right", framealpha=0.7)

    # --- Right: Multiples table ---
    ax2.set_facecolor(C["panel"])
    ax2.axis("off")
    ax2.set_title("Key Multiples", fontsize=9, color=C["text"], pad=8)

    col_labels = ["Company", "PE", "PB", "EV/EBITDA", "ROCE%", "RevCAGR%"]
    table_data = []
    for _, row in df.sort_values("Score", ascending=False).iterrows():
        table_data.append([
            row["Company"][:20],
            f"{row['PE']:.1f}" if row["PE"] else "—",
            f"{row['PB']:.2f}" if row["PB"] else "—",
            f"{row['EV/EBITDA']:.1f}" if row["EV/EBITDA"] else "—",
            f"{row['ROCE']:.1f}" if row["ROCE"] else "—",
            f"{row['Rev CAGR']:.1f}%" if row["Rev CAGR"] else "—",
        ])

    if table_data:
        tbl = ax2.table(
            cellText=table_data, colLabels=col_labels,
            cellLoc="center", loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.5)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor(C["border"])
            if r == 0:
                cell.set_facecolor(C["blue"])
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor(C["bg"] if r % 2 == 0 else C["panel"])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
