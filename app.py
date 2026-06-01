# =============================================================================
# PORTFOLIO INTELLIGENCE PLATFORM — FULL INSTITUTIONAL BUILD
# GPT-4o mini AI | 8 Tabs | 18-page PDF | Modules 1–7
# =============================================================================
# Run: streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tempfile, os
from datetime import datetime, date

# Load .env if present (allows pre-filling API key without re-entering each session)
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)
except ImportError:
    pass  # python-dotenv not installed — sidebar entry still works

_ENV_API_KEY       = os.environ.get("OPENAI_API_KEY", "")
_ENV_PORTFOLIO_NAME = os.environ.get("PORTFOLIO_NAME", "Portfolio Intelligence Report")

# ── Core modules ──────────────────────────────────────────────────────────────
from modules.m1_loader     import load_portfolio, fetch_prices, calculate_metrics, portfolio_summary
from modules.m2_sector     import aggregate_sectors, plot_sector_overview, plot_cumulative_performance
from modules.m3_attribution import fetch_sector_benchmark_returns, bhb_attribution, plot_attribution_charts
from modules.m4_risk       import calculate_risk_metrics, generate_risk_flags, fetch_correlation_data, plot_risk_dashboard
from modules.m5a_data      import run_data_layer, BUSINESS_MODEL
from modules.m5b_intelligence import run_intelligence_layer, plot_valuation_dashboard
from modules.m5c_scenario  import run_scenario_engine
from modules.m5d_action    import run_action_engine, portfolio_action_summary
from modules.m5e_context   import build_context_json
from modules.m6a_research  import run_research_ingestion
from modules.m6b_thesis    import generate_institutional_report, DEFAULT_PHILOSOPHY
from utils.pdf_export      import build_institutional_pdf

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Portfolio Intelligence Platform",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main { background-color: #FAFBFC; }
.stTabs [data-baseweb="tab"] { padding:8px 18px; font-weight:600; font-size:13px; }
.section-header {
    font-size:17px; font-weight:700; color:#0F172A;
    border-bottom:2px solid #1D4ED8; padding-bottom:6px; margin-bottom:14px;
}
.kpi-card { background:white; border:1px solid #DDE3EE; border-radius:8px; padding:14px 18px; text-align:center; }
.signal-acc  { background:#DCFCE7; color:#15803D; font-weight:700; border-radius:6px; padding:3px 10px; }
.signal-hold { background:#DBEAFE; color:#1D4ED8; font-weight:700; border-radius:6px; padding:3px 10px; }
.signal-trim { background:#FEF3C7; color:#D97706; font-weight:700; border-radius:6px; padding:3px 10px; }
.signal-exit { background:#FEE2E2; color:#DC2626; font-weight:700; border-radius:6px; padding:3px 10px; }
.signal-watch{ background:#F1F5F9; color:#475569; font-weight:700; border-radius:6px; padding:3px 10px; }
div[data-testid="stExpander"] { border:1px solid #DDE3EE; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================
for key in [
    "holdings", "portfolio_stats", "sector_df", "risk_metrics", "risk_flags",
    "attribution_summary", "attrib_df", "clean_data", "verdicts",
    "scenarios", "actions", "context_json", "research_data",
    "institutional_report", "analysis_date",
    "figs_overview", "figs_sector", "figs_attribution", "figs_risk", "figs_valuation",
    "price_history_data",
]:
    if key not in st.session_state:
        st.session_state[key] = None

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # ── Portfolio CSV ─────────────────────────────────────────────────────────
    st.markdown("### 📁 Portfolio")
    portfolio_file = st.file_uploader("Upload portfolio.csv", type=["csv"])

    analysis_date_input = st.date_input("Analysis Date", value=date(2026, 1, 5))
    analysis_date = datetime.combine(analysis_date_input, datetime.min.time())

    st.markdown("### 🔧 Manual Prices")
    st.caption("For tickers that fail yfinance")
    tatamotors_price = st.number_input("TATAMOTORS.NS", value=420.50, step=0.5)
    bbg_price        = st.number_input("BBG.NS",        value=159.05, step=0.5)
    manual_prices    = {"TATAMOTORS.NS": tatamotors_price, "BBG.NS": bbg_price}

    # ── Screener Files ────────────────────────────────────────────────────────
    st.markdown("### 📊 Screener Files")
    st.caption("Screener.in Excel export per holding")
    ticker_to_company = {
        "HINDCOPPER.NS": "Hindustan Copper", "TATAMOTORS.NS": "Tata Motors",
        "HINDZINC.NS": "Hindustan Zinc",     "TATASTEEL.NS": "Tata Steel",
        "IDFCFIRSTB.NS": "IDFC First Bank",  "PNB.NS": "Punjab Natl Bank",
        "EQUITASBNK.NS": "Equitas SFB",      "BANKINDIA.NS": "Bank of India",
        "INDUSTOWER.NS": "Indus Towers",     "GABRIEL.NS": "Gabriel India",
        "BORORENEW.NS": "Borosil Renew.",    "JIOFIN.NS": "Jio Financial",
        "BBG.NS": "BillionBrains",           "ADANIPOWER.NS": "Adani Power",
        "IEX.NS": "IEX",
    }
    screener_uploads = {}
    with st.expander("Upload Screener Excels", expanded=False):
        for ticker, company in ticker_to_company.items():
            f = st.file_uploader(company, type=["xlsx"], key=f"sc_{ticker}")
            if f:
                screener_uploads[ticker] = f

    st.markdown("---")

    # ── Investment Philosophy ─────────────────────────────────────────────────
    st.markdown("### 💡 Investment Philosophy")
    st.caption("The AI learns your investment style — edit as needed")
    philosophy = st.text_area(
        "Philosophy",
        value=DEFAULT_PHILOSOPHY,
        height=180,
        label_visibility="collapsed",
    )

    # ── Concall Uploads ───────────────────────────────────────────────────────
    st.markdown("### 📋 Research — Concall PDFs")
    st.caption("Upload latest concall transcript per holding (PDF)")
    concall_uploads_raw = {}
    with st.expander("Upload Concall PDFs", expanded=False):
        for ticker, company in ticker_to_company.items():
            cf = st.file_uploader(f"{company} concall", type=["pdf"], key=f"cc_{ticker}")
            if cf:
                concall_uploads_raw[ticker] = cf.read()

    st.markdown("---")

    # ── API Key (loaded from .env — no sidebar input) ────────────────────────
    api_key = _ENV_API_KEY
    if api_key:
        st.success("🤖 GPT-4o mini ready ✅")
    else:
        st.warning("⚠️ Set OPENAI_API_KEY in .env to enable AI modules")

    st.markdown("---")

    # ── Run button ────────────────────────────────────────────────────────────
    run_btn = st.button("▶  Run Full Analysis", type="primary",
                        use_container_width=True, disabled=(portfolio_file is None))
    if portfolio_file is None:
        st.info("Upload portfolio.csv to begin")

# =============================================================================
# MAIN ANALYSIS EXECUTION
# =============================================================================
if run_btn and portfolio_file:
    report_date = analysis_date.strftime("%d %b %Y")
    st.markdown("---")
    st.markdown("### ⏳ Running Analysis Pipeline...")

    # M1 — Portfolio Loader
    with st.status("Module 1 — Portfolio loading + live prices...", expanded=True) as s1:
        holdings = load_portfolio(portfolio_file)
        prices, price_results = fetch_prices(holdings, analysis_date, manual_prices)
        for ticker, company, price, status in price_results:
            icon = "✅" if status == "live" else ("⚙️" if status == "manual" else "⚠️")
            st.write(f"{icon} {company[:28]}: ₹{price:.2f}  [{status}]")
        holdings = calculate_metrics(holdings, prices)
        stats    = portfolio_summary(holdings)
        st.session_state.update({"holdings": holdings, "portfolio_stats": stats, "analysis_date": analysis_date})
        s1.update(label="Module 1 ✅ — Portfolio loaded", state="complete")

    # M2 — Sector Analysis
    with st.status("Module 2 — Sector analysis...", expanded=False) as s2:
        sector_df    = aggregate_sectors(holdings)
        fig_overview = plot_sector_overview(sector_df, holdings, report_date)
        fig_sector   = plot_cumulative_performance(holdings, analysis_date)
        st.session_state.update({
            "sector_df": sector_df,
            "figs_overview": fig_overview,
            "figs_sector": fig_sector,
        })
        s2.update(label="Module 2 ✅ — Sector charts ready", state="complete")

    # M3 — Attribution
    with st.status("Module 3 — BHB attribution...", expanded=False) as s3:
        unique_sectors = sector_df[sector_df["Port_Weight"] > 0]["Sector"].tolist()
        bench_returns, total_bench = fetch_sector_benchmark_returns(unique_sectors, analysis_date)
        attrib_df, attr_summary, holdings_contrib = bhb_attribution(
            holdings, sector_df, bench_returns, total_bench
        )
        fig_attr = plot_attribution_charts(attrib_df, holdings_contrib, attr_summary, report_date)
        st.session_state.update({
            "attrib_df": attrib_df, "attribution_summary": attr_summary,
            "figs_attribution": fig_attr,
        })
        s3.update(label="Module 3 ✅ — Attribution complete", state="complete")

    # M4 — Risk
    with st.status("Module 4 — Risk analytics...", expanded=False) as s4:
        risk_metrics = calculate_risk_metrics(holdings, sector_df)
        risk_flags   = generate_risk_flags(risk_metrics, sector_df)
        with st.spinner("Fetching price history for correlation..."):
            corr_returns = fetch_correlation_data(holdings, analysis_date)
        fig_risk = plot_risk_dashboard(holdings, sector_df, risk_metrics, corr_returns, report_date)
        st.session_state.update({
            "risk_metrics": risk_metrics, "risk_flags": risk_flags, "figs_risk": fig_risk,
        })
        s4.update(label="Module 4 ✅ — Risk analytics complete", state="complete")

    # M5 — Valuation
    with st.status("Module 5 — Fundamental data & valuation...", expanded=True) as s5:
        tmp_dir = tempfile.mkdtemp()
        screener_paths = {}
        for ticker, uploaded in screener_uploads.items():
            tmp_path = os.path.join(tmp_dir, f"{ticker.replace('.', '_')}.xlsx")
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())
            screener_paths[ticker] = tmp_path

        all_screener_files = {}
        for t in holdings["Ticker"].tolist():
            btype, _ = BUSINESS_MODEL.get(t, ("UNKNOWN", t))
            all_screener_files[t] = screener_paths.get(t) if btype != "ETF" else None

        def sp_cb(_ticker, company, status): st.write(f"  {company}: {status}")
        clean_data = run_data_layer(holdings, all_screener_files, sp_cb)
        verdicts   = run_intelligence_layer(clean_data)
        fig_val    = plot_valuation_dashboard(clean_data, verdicts, holdings, report_date)
        st.session_state.update({
            "clean_data": clean_data, "verdicts": verdicts, "figs_valuation": fig_val,
        })
        s5.update(label="Module 5 ✅ — Valuation complete", state="complete")

    # M5C — Scenario Engine
    with st.status("Module 5C — Scenario engine...", expanded=False) as s5c:
        scenarios = run_scenario_engine(clean_data, verdicts, holdings)
        st.session_state["scenarios"] = scenarios
        s5c.update(label="Module 5C ✅ — Scenarios computed", state="complete")

    # M5D — Action Engine
    with st.status("Module 5D — Action engine...", expanded=False) as s5d:
        actions = run_action_engine(verdicts, scenarios, holdings)
        st.session_state["actions"] = actions
        action_sum = portfolio_action_summary(actions, holdings)
        counts = action_sum.get("signal_counts", {})
        st.write(f"  ACCUMULATE: {counts.get('ACCUMULATE',0)} | HOLD: {counts.get('HOLD',0)} | "
                 f"TRIM: {counts.get('TRIM',0)} | EXIT: {counts.get('EXIT',0)}")
        s5d.update(label="Module 5D ✅ — Action signals generated", state="complete")

    # M5E — Context Consolidation
    with st.status("Module 5E — Context consolidation (M6.1)...", expanded=False) as s5e:
        context_json = build_context_json(
            holdings_df=holdings, portfolio_stats=stats, sector_df=sector_df,
            attribution_summary=attr_summary, risk_metrics=risk_metrics,
            risk_flags=risk_flags, clean_data=clean_data, verdicts=verdicts,
            scenarios=scenarios, actions=actions, analysis_date=report_date,
        )
        st.session_state["context_json"] = context_json
        s5e.update(label="Module 5E ✅ — Context JSON ready", state="complete")

    st.success("✅ Full analysis complete! Navigate tabs to explore results.")

# =============================================================================
# MAIN CONTENT — 8 TABS
# =============================================================================
if st.session_state["holdings"] is not None:
    holdings      = st.session_state["holdings"]
    stats         = st.session_state["portfolio_stats"]
    sector_df     = st.session_state["sector_df"]
    risk_metrics  = st.session_state["risk_metrics"]
    risk_flags    = st.session_state["risk_flags"]
    clean_data    = st.session_state["clean_data"] or {}
    verdicts      = st.session_state["verdicts"] or {}
    scenarios     = st.session_state["scenarios"] or {}
    actions       = st.session_state["actions"] or {}
    context_json  = st.session_state["context_json"]
    attr_summary  = st.session_state["attribution_summary"]
    attrib_df     = st.session_state["attrib_df"]
    inst_report   = st.session_state["institutional_report"] or {}
    analysis_date = st.session_state["analysis_date"]
    report_date   = analysis_date.strftime("%d %b %Y") if analysis_date else "—"

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Overview",
        "🏭 Sector Analysis",
        "📈 Attribution",
        "⚠️ Risk",
        "💹 Valuation",
        "🎯 Scenarios & Actions",
        "📰 Research",
        "🏛️ Institutional Report",
    ])

    # =========================================================================
    # TAB 1 — PORTFOLIO OVERVIEW
    # =========================================================================
    with tab1:
        st.markdown(f'<div class="section-header">Portfolio Overview — {report_date}</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Invested",      f"₹{stats['total_invested']:,.0f}")
        c2.metric("Current Value", f"₹{stats['total_current']:,.0f}")
        c3.metric("Unrealised P&L",f"₹{stats['total_pnl']:+,.0f}", delta=f"{stats['total_return']:+.2f}%")
        c4.metric("Winners / Losers", f"{stats['num_winners']} / {stats['num_losers']}")
        c5.metric("Active Share",  f"{(risk_metrics or {}).get('active_share', 0):.1f}%")

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Best",  stats["best_performer"][:22],  f"{stats['best_return']:+.1f}%")
        c2.metric("Worst", stats["worst_performer"][:22], f"{stats['worst_return']:+.1f}%")
        c3.metric("Largest Position", stats["largest_position"][:22], f"{stats['largest_wt']:.1f}%")

        st.markdown("---")
        st.markdown("#### Holdings Table")

        display_df = holdings[[
            "Company", "Sector", "Avg Buy Price", "Current Price",
            "Invested Value", "Current Value", "P&L", "Return %",
            "Portfolio Wt%", "Benchmark Wt%", "Active Bet%",
        ]].copy()

        # Add action signal column if available
        if actions:
            display_df["Signal"] = display_df.index.map(
                lambda i: actions.get(holdings.loc[i, "Ticker"], {}).get("signal", "—") if i in holdings.index else "—"
            )
            # Re-index based on Ticker
            ticker_signal = {t: a.get("signal", "—") for t, a in actions.items()}
            display_df["Signal"] = holdings["Ticker"].map(ticker_signal)

        def color_ret(val):
            return "color:#15803D; font-weight:bold" if val >= 0 else "color:#DC2626; font-weight:bold"

        styled = display_df.style.applymap(color_ret, subset=["Return %", "P&L", "Active Bet%"])
        styled = styled.format({
            "Avg Buy Price": "₹{:.2f}", "Current Price": "₹{:.2f}",
            "Invested Value": "₹{:,.0f}", "Current Value": "₹{:,.0f}",
            "P&L": "₹{:+,.0f}", "Return %": "{:+.2f}%",
            "Portfolio Wt%": "{:.2f}%", "Benchmark Wt%": "{:.2f}%",
            "Active Bet%": "{:+.2f}%",
        }, na_rep="—")
        st.dataframe(styled, use_container_width=True, height=520)

    # =========================================================================
    # TAB 2 — SECTOR ANALYSIS
    # =========================================================================
    with tab2:
        st.markdown('<div class="section-header">Sector Allocation & Performance</div>', unsafe_allow_html=True)
        if sector_df is not None:
            sec_display = sector_df[sector_df["Port_Weight"] > 0][[
                "Sector", "Port_Weight", "Bench_Weight", "Active_Bet", "Sector_Return%", "Num_Stocks"
            ]].copy()
            sec_display.columns = ["Sector", "Port %", "Bench %", "Active Bet %", "Return %", "# Stocks"]
            def color_bet(v): return "color:#15803D;font-weight:bold" if v > 1 else ("color:#DC2626;font-weight:bold" if v < -1 else "")
            st.dataframe(sec_display.style.applymap(color_bet, subset=["Active Bet %"]).format({
                "Port %": "{:.2f}%", "Bench %": "{:.2f}%",
                "Active Bet %": "{:+.2f}%", "Return %": "{:+.2f}%",
            }), use_container_width=True)
        st.markdown("---")
        if st.session_state["figs_overview"]:
            st.pyplot(st.session_state["figs_overview"], use_container_width=True)
        st.markdown("---")
        st.markdown("#### 5-Year Cumulative Return vs Benchmark")
        if st.session_state["figs_sector"]:
            st.pyplot(st.session_state["figs_sector"], use_container_width=True)

    # =========================================================================
    # TAB 3 — ATTRIBUTION
    # =========================================================================
    with tab3:
        st.markdown('<div class="section-header">Brinson-Hood-Beebower Attribution</div>', unsafe_allow_html=True)
        if attr_summary:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Portfolio Return",  f"{attr_summary['total_port_return']:+.2f}%")
            c2.metric("Benchmark Return",  f"{attr_summary['total_bench_return']:+.2f}%")
            c3.metric("Total Alpha",       f"{attr_summary['total_alpha']:+.2f}%")
            c4.metric("Hit Rate",          f"{(risk_metrics or {}).get('hit_rate', 0):.1f}%")
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Allocation Effect",  f"{attr_summary['allocation_effect']:+.4f}%")
            c2.metric("Selection Effect",   f"{attr_summary['selection_effect']:+.4f}%")
            c3.metric("Interaction Effect", f"{attr_summary['interaction_effect']:+.4f}%")
            alloc, sel = attr_summary["allocation_effect"], attr_summary["selection_effect"]
            st.info(f"📌 Alpha driven by **{'sector allocation' if abs(alloc)>abs(sel) else 'stock selection'}** "
                    f"({(alloc if abs(alloc)>abs(sel) else sel):+.4f}%)")
        if attrib_df is not None:
            st.markdown("#### BHB Attribution Table")
            st.dataframe(attrib_df.style.format({
                "Port_Wt%": "{:.2f}%", "Bench_Wt%": "{:.2f}%",
                "Active_Bet%": "{:+.2f}%", "Allocation": "{:+.4f}%",
                "Selection": "{:+.4f}%", "Active_Return": "{:+.4f}%",
            }), use_container_width=True)
        if st.session_state["figs_attribution"]:
            st.markdown("---")
            st.pyplot(st.session_state["figs_attribution"], use_container_width=True)

    # =========================================================================
    # TAB 4 — RISK
    # =========================================================================
    with tab4:
        st.markdown('<div class="section-header">Portfolio Risk Analytics</div>', unsafe_allow_html=True)
        if risk_metrics:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Active Share",     f"{risk_metrics['active_share']:.1f}%")
            c2.metric("HHI",              f"{risk_metrics['hhi']:.0f}")
            c3.metric("Effective N",      f"{risk_metrics['effective_n']:.1f}")
            c4.metric("Hit Rate",         f"{risk_metrics['hit_rate']:.1f}%")
            c5.metric("Win/Loss",         f"{risk_metrics['win_loss_ratio']:.2f}x")
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Top 1 Wt",  f"{risk_metrics['top1_wt']:.1f}%")
            c2.metric("Top 5 Wt",  f"{risk_metrics['top5_wt']:.1f}%")
            c3.metric("Top 10 Wt", f"{risk_metrics['top10_wt']:.1f}%")
            st.markdown("---")
            st.markdown("#### Risk Flags")
            for flag in (risk_flags or []):
                lvl = flag.get("level", str(flag)) if isinstance(flag, dict) else str(flag)
                if "⚠️" in lvl: st.error(lvl)
                elif "🟡" in lvl: st.warning(lvl)
                elif "✅" in lvl: st.success(lvl)
                else: st.info(lvl)
        if st.session_state["figs_risk"]:
            st.markdown("---")
            st.pyplot(st.session_state["figs_risk"], use_container_width=True)

    # =========================================================================
    # TAB 5 — VALUATION
    # =========================================================================
    with tab5:
        st.markdown('<div class="section-header">Valuation & Peer Analysis</div>', unsafe_allow_html=True)
        if not verdicts:
            st.warning("Upload Screener Excel files in sidebar to run fundamental analysis.")
        else:
            valid_scores = [v["score"] for v in verdicts.values() if v and v.get("score")]
            avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Analysed", len(valid_scores))
            c2.metric("Avg Score", f"{avg:.1f}/100")
            c3.metric("Deep Value + Attractive",
                      sum(1 for v in verdicts.values() if v and v.get("label") in ("DEEP VALUE","ATTRACTIVE")))
            c4.metric("Rich + Expensive",
                      sum(1 for v in verdicts.values() if v and v.get("label") in ("RICH","EXPENSIVE")))
            st.markdown("---")

            val_rows = []
            for _, row in holdings.iterrows():
                ticker = row["Ticker"]
                d = clean_data.get(ticker) or {}
                v = verdicts.get(ticker) or {}
                sc = scenarios.get(ticker) or {}
                ac = actions.get(ticker) or {}
                val_rows.append({
                    "Company": row["Company"][:22], "Sector": row["Sector"][:16],
                    "Score": v.get("score"), "Verdict": v.get("label","—"),
                    "PE": d.get("pe"), "PB": d.get("pb"), "EV/EBITDA": d.get("ev_ebitda"),
                    "ROCE%": d.get("roce"), "ROE%": d.get("roe"), "RevCAGR%": d.get("rev_cagr_3y"),
                    "Bear %": (sc.get("bear") or {}).get("return_pct"),
                    "Bull %": (sc.get("bull") or {}).get("return_pct"),
                    "Asymmetry": sc.get("asymmetry_ratio"),
                    "Signal": ac.get("signal","—"),
                })
            val_df = pd.DataFrame(val_rows)
            st.dataframe(val_df.style.format({
                "Score": "{:.0f}", "PE": "{:.1f}", "PB": "{:.2f}",
                "EV/EBITDA": "{:.1f}", "ROCE%": "{:.1f}", "ROE%": "{:.1f}",
                "RevCAGR%": "{:.1f}", "Bear %": "{:+.1f}%", "Bull %": "{:+.1f}%",
                "Asymmetry": "{:.2f}x",
            }, na_rep="—"), use_container_width=True)

            st.markdown("---")
            for _, row in holdings.iterrows():
                ticker = row["Ticker"]
                d = clean_data.get(ticker) or {}
                v = verdicts.get(ticker) or {}
                if not d: continue
                with st.expander(f"{row['Company']}  |  {v.get('label','—')}  ({v.get('score') or 'N/A':.0f if v.get('score') else 'N/A'}/100)"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(f"Revenue: ₹{d.get('revenue',0) or 0:,.0f} Cr | EBITDA: {d.get('ebitda_margin') or '—'}%")
                        st.write(f"ROCE: {d.get('roce') or '—'}% | ROE: {d.get('roe') or '—'}%")
                        st.write(f"D/E: {d.get('de_ratio') or '—'}x | ND/EBITDA: {d.get('nd_ebitda') or '—'}x")
                    with c2:
                        st.write(f"Rev CAGR 3yr: {d.get('rev_cagr_3y') or '—'}%")
                        st.write(f"PAT CAGR 3yr: {d.get('pat_cagr_3y') or '—'}%")
                        if v.get("flags"):
                            for fl in v["flags"][:4]: st.write(f"• {fl}")

        if st.session_state["figs_valuation"]:
            st.markdown("---")
            st.pyplot(st.session_state["figs_valuation"], use_container_width=True)

    # =========================================================================
    # TAB 6 — SCENARIOS & ACTIONS
    # =========================================================================
    with tab6:
        st.markdown('<div class="section-header">🎯 Scenarios & Action Engine</div>', unsafe_allow_html=True)

        if not scenarios:
            st.warning("Run Full Analysis first to generate scenarios and action signals.")
        else:
            # Portfolio action summary
            action_sum = portfolio_action_summary(actions, holdings)
            counts = action_sum.get("signal_counts", {})
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("ACCUMULATE", counts.get("ACCUMULATE", 0))
            c2.metric("HOLD",       counts.get("HOLD", 0))
            c3.metric("TRIM",       counts.get("TRIM", 0))
            c4.metric("EXIT",       counts.get("EXIT", 0))
            c5.metric("Net Bias",   action_sum.get("net_bias", "—"))

            if action_sum.get("accumulate_names"):
                st.success(f"🟢 **Add to:** {', '.join(action_sum['accumulate_names'])}")
            if action_sum.get("trim_names"):
                st.warning(f"🟡 **Trim:** {', '.join(action_sum['trim_names'])}")
            if action_sum.get("exit_names"):
                st.error(f"🔴 **Exit:** {', '.join(action_sum['exit_names'])}")

            st.markdown("---")
            st.markdown("#### Per-Holding Scenarios & Signals")

            for _, row in holdings.iterrows():
                ticker  = row["Ticker"]
                company = row["Company"]
                sc      = scenarios.get(ticker) or {}
                ac      = actions.get(ticker) or {}
                signal  = ac.get("signal", "WATCH")
                signal_css = f"signal-{signal.lower()}"

                with st.expander(
                    f"{company}  |  {row['Portfolio Wt%']:.1f}%  |  Role: {sc.get('portfolio_role','?')}  |  {signal}"
                ):
                    c1, c2, c3 = st.columns(3)
                    bear = sc.get("bear", {})
                    bull = sc.get("bull", {})
                    c1.metric("Bear Case",  f"{bear.get('return_pct', '—'):+.1f}%" if isinstance(bear.get('return_pct'), float) else "—",
                              delta="downside", delta_color="inverse")
                    c2.metric("Base Case",  "+0.0%")
                    c3.metric("Bull Case",  f"{bull.get('return_pct', '—'):+.1f}%" if isinstance(bull.get('return_pct'), float) else "—",
                              delta="upside")

                    st.write(f"**Asymmetry:** {sc.get('asymmetry_ratio', 0):.1f}x  |  "
                             f"**Portfolio Role:** {sc.get('portfolio_role', '?')}")
                    st.write(f"**Bear catalyst:** {bear.get('catalyst', '?')}")
                    st.write(f"**Bull catalyst:** {bull.get('catalyst', '?')}")
                    st.markdown("---")
                    st.write(f"**Action Signal:** `{signal}`  |  **Conviction:** {ac.get('conviction', '?')}")
                    st.write(f"**Sizing:** {ac.get('target_wt_lo', 0):.0f}–{ac.get('target_wt_hi', 0):.0f}% target "
                             f"(current: {ac.get('current_wt', 0):.1f}%)")
                    st.write(f"**Rationale:** {ac.get('reason', '—')}")

    # =========================================================================
    # TAB 7 — RESEARCH
    # =========================================================================
    with tab7:
        st.markdown('<div class="section-header">📰 Research Data Ingestion (M6.2)</div>', unsafe_allow_html=True)

        if not api_key:
            st.warning("⚠️ Set OPENAI_API_KEY in the `.env` file to enable research ingestion.")
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("Processes uploaded concall PDFs + auto-fetches latest news per holding via yfinance.")
            with col2:
                run_research_btn = st.button("🔍 Run Research Ingestion", type="primary", use_container_width=True)

            if run_research_btn:
                progress_bar = st.progress(0)
                status_text  = st.empty()
                total = len(holdings)
                done  = [0]

                def research_cb(_ticker, msg):
                    done[0] += 1
                    progress_bar.progress(min(done[0] / total, 1.0))
                    status_text.text(f"⏳ {msg}")

                with st.spinner("Running research ingestion..."):
                    research_data = run_research_ingestion(
                        holdings, concall_uploads_raw, api_key, research_cb
                    )
                st.session_state["research_data"] = research_data
                # Update context JSON with research
                if st.session_state["context_json"]:
                    ctx = st.session_state["context_json"]
                    for ticker, res in research_data.items():
                        if ticker in ctx.get("holdings", {}):
                            ctx["holdings"][ticker]["research"] = res
                    st.session_state["context_json"] = ctx
                progress_bar.progress(1.0)
                status_text.text("✅ Research ingestion complete!")
                st.success("Research data ready!")

        research_data = st.session_state.get("research_data") or {}
        if research_data:
            st.markdown("---")
            for _, row in holdings.iterrows():
                ticker  = row["Ticker"]
                company = row["Company"]
                res     = research_data.get(ticker) or {}
                if not res: continue

                has_cc = res.get("has_concall", False)
                with st.expander(f"{company}  |  {'✅ Concall + News' if has_cc else '📰 News only'}"):
                    if has_cc and res.get("concall"):
                        cc = res["concall"]
                        st.markdown("**Concall Insights:**")
                        cols = st.columns(2)
                        with cols[0]:
                            if cc.get("growth_guidance"): st.write(f"📈 **Growth:** {cc['growth_guidance']}")
                            if cc.get("capex_plans"):     st.write(f"🏗️ **Capex:** {cc['capex_plans']}")
                            if cc.get("management_tone"): st.write(f"🎙️ **Tone:** `{cc['management_tone']}`")
                        with cols[1]:
                            if cc.get("margin_guidance"): st.write(f"📊 **Margins:** {cc['margin_guidance']}")
                            risks = cc.get("risk_acknowledgements", [])
                            if risks: st.write(f"⚠️ **Risks:** {'; '.join(str(r) for r in risks[:2])}")
                        st.markdown("---")

                    news = res.get("news", {})
                    if news:
                        sentiment = news.get("sentiment", "NEUTRAL")
                        color = "🟢" if sentiment == "POSITIVE" else ("🔴" if sentiment == "NEGATIVE" else "⚪")
                        st.write(f"{color} **News Sentiment:** {sentiment}")
                        if news.get("sentiment_rationale"): st.write(f"  {news['sentiment_rationale']}")
                        events = news.get("key_events", [])
                        if events: st.write(f"📌 **Events:** {'; '.join(str(e) for e in events[:3])}")

                    articles = res.get("articles", [])
                    if articles:
                        st.markdown("**Latest Headlines:**")
                        for art in articles[:3]:
                            st.write(f"• {art.get('title', '')} *(_{art.get('publisher', '')}_)*")

    # =========================================================================
    # TAB 8 — INSTITUTIONAL REPORT
    # =========================================================================
    with tab8:
        st.markdown('<div class="section-header">🏛️ Institutional Report — AI Thesis Engine (M6.3)</div>', unsafe_allow_html=True)

        if not api_key:
            st.warning("⚠️ Set OPENAI_API_KEY in the `.env` file to enable AI report generation.")
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(
                    "Runs the 7-prompt AI chain per holding: Core Thesis → Mispricing → Conviction → "
                    "Contrarian → Sector → Structuring → Validation. Generates 12-point institutional thesis + PM Commentary."
                )
            with col2:
                gen_btn = st.button("🏛️ Generate Institutional Report", type="primary", use_container_width=True)

            if gen_btn:
                if not context_json:
                    st.error("Run Full Analysis first — context JSON not available.")
                else:
                    progress_bar = st.progress(0)
                    status_text  = st.empty()
                    total = len(holdings) + 2
                    done  = [0]

                    def thesis_cb(_ticker, msg):
                        done[0] += 1
                        progress_bar.progress(min(done[0] / total, 1.0))
                        status_text.text(f"⏳ {msg}")

                    with st.spinner("Generating institutional report (7-prompt chain per stock)..."):
                        inst_report = generate_institutional_report(
                            context_json=context_json,
                            philosophy=philosophy,
                            api_key=api_key,
                            progress_callback=thesis_cb,
                        )
                    st.session_state["institutional_report"] = inst_report
                    progress_bar.progress(1.0)
                    status_text.text("✅ Institutional report complete!")
                    st.success("Institutional report generated!")
                    st.rerun()

        inst_report = st.session_state.get("institutional_report") or {}
        if inst_report:
            # Portfolio Identity + Executive Summary
            st.markdown(f"### {inst_report.get('portfolio_identity', 'Portfolio')}")
            if inst_report.get("executive_summary"):
                st.markdown(inst_report["executive_summary"])
            st.markdown("---")

            # Per-stock theses
            st.markdown("### Individual Investment Theses")
            stock_theses = inst_report.get("stock_theses", {})
            for _, row in holdings.iterrows():
                ticker  = row["Ticker"]
                company = row["Company"]
                th      = stock_theses.get(ticker) or {}
                ac      = actions.get(ticker) or {}
                sc      = scenarios.get(ticker) or {}

                action_lbl  = th.get("action", ac.get("signal", "WATCH"))
                conv_lbl    = th.get("conviction", ac.get("conviction", "LOW"))
                role_lbl    = th.get("portfolio_role", sc.get("portfolio_role", "?"))
                score_str   = f"{(verdicts.get(ticker) or {}).get('score') or 'N/A':.0f}" if isinstance((verdicts.get(ticker) or {}).get("score"), float) else "N/A"

                with st.expander(
                    f"**{company}**  ·  {row['Portfolio Wt%']:.1f}%  ·  {action_lbl}  ·  {role_lbl}  ·  Score {score_str}/100"
                ):
                    if th.get("investment_thesis"):
                        st.markdown(f"**Investment Thesis**\n\n{th['investment_thesis']}")
                    if th.get("market_expectation_vs_reality"):
                        st.markdown(f"**Market Expectation vs Reality**\n\n{th['market_expectation_vs_reality']}")
                    if th.get("why_market_wrong"):
                        st.markdown(f"**Why Market May Be Wrong**\n\n{th['why_market_wrong']}")
                    c1, c2 = st.columns(2)
                    with c1:
                        if th.get("bull_case"):
                            st.markdown("**Bull Case**")
                            for b in th["bull_case"]: st.write(f"• {b}")
                        if th.get("key_triggers"):
                            st.markdown("**Key Triggers**")
                            for t_ in th["key_triggers"]: st.write(f"• {t_}")
                    with c2:
                        if th.get("bear_case"):
                            st.markdown("**Bear Case**")
                            for b in th["bear_case"]: st.write(f"• {b}")
                        if th.get("risk_factors"):
                            st.markdown("**Risk Factors**")
                            for r in th["risk_factors"]: st.write(f"• {r}")
                    if th.get("capital_allocation_quality"):
                        st.markdown(f"**Capital Allocation:** {th['capital_allocation_quality']}")
                    if th.get("management_assessment"):
                        st.markdown(f"**Management:** {th['management_assessment']}")
                    if th.get("position_sizing_logic"):
                        st.markdown(f"**Position Sizing:** {th['position_sizing_logic']}")
                    if th.get("long_term_outlook"):
                        st.markdown(f"**Long-term Outlook:** {th['long_term_outlook']}")
                    if th.get("monitoring_framework"):
                        st.markdown("**Monitoring KPIs:** " + " | ".join(str(m) for m in th["monitoring_framework"][:4]))

            st.markdown("---")
            st.markdown("### PM Commentary")
            if inst_report.get("pm_commentary"):
                st.markdown(inst_report["pm_commentary"])

        # ── PDF Download ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📄 Download 18-Page Institutional PDF Report")
        pdf_btn = st.button("Generate PDF Report", use_container_width=True)
        if pdf_btn:
            if not st.session_state["portfolio_stats"]:
                st.error("Run analysis first.")
            else:
                with st.spinner("Assembling 18-page institutional report..."):
                    try:
                        pdf_bytes = build_institutional_pdf(
                            report_date=report_date,
                            portfolio_stats=stats,
                            risk_metrics=risk_metrics or {},
                            risk_flags=risk_flags or [],
                            attribution_summary=attr_summary,
                            attrib_df=attrib_df,
                            holdings_df=holdings,
                            sector_df=sector_df,
                            clean_data=clean_data,
                            verdicts=verdicts,
                            scenarios=scenarios,
                            actions=actions,
                            institutional_report=inst_report,
                            figures={},
                            price_history=st.session_state.get("price_history_data") or {},
                        )
                        fname = f"institutional_report_{analysis_date.strftime('%Y%m%d')}.pdf"
                        st.download_button(
                            "⬇️  Download PDF (18 pages)",
                            data=pdf_bytes, file_name=fname,
                            mime="application/pdf", use_container_width=True,
                        )
                        st.success(f"✅ {fname} ready for download.")
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")

# =============================================================================
# EMPTY STATE
# =============================================================================
else:
    st.markdown("---")
    st.markdown("""
<div style="text-align:center; padding:60px 20px;">
<h2 style="color:#1D4ED8;">🏛️ Portfolio Intelligence Platform</h2>
<p style="color:#475569; font-size:16px;">
Upload your <code>portfolio.csv</code> in the sidebar and click <b>▶ Run Full Analysis</b>
</p><br/>
<table style="margin:auto; border-collapse:collapse; text-align:left;">
<tr><td style="padding:8px 16px"><b>M5A–5B</b></td><td style="padding:8px;color:#475569">Financial extraction → Scoring → Peer comparison</td></tr>
<tr style="background:#F8F9FC"><td style="padding:8px 16px"><b>M5C</b></td><td style="padding:8px;color:#475569">Scenario Engine — Bull/Base/Bear + Asymmetry ratio</td></tr>
<tr><td style="padding:8px 16px"><b>M5D</b></td><td style="padding:8px;color:#475569">Action Engine — ACCUMULATE/HOLD/TRIM/EXIT signals</td></tr>
<tr style="background:#F8F9FC"><td style="padding:8px 16px"><b>M5E</b></td><td style="padding:8px;color:#475569">Context Consolidation — unified JSON for LLM (M6.1)</td></tr>
<tr><td style="padding:8px 16px"><b>M6A</b></td><td style="padding:8px;color:#475569">Research Ingestion — Concall PDF + News (M6.2)</td></tr>
<tr style="background:#F8F9FC"><td style="padding:8px 16px"><b>M6B</b></td><td style="padding:8px;color:#475569">AI Thesis Engine — 7-prompt chain, 12-point output (M6.3)</td></tr>
<tr><td style="padding:8px 16px"><b>M7</b></td><td style="padding:8px;color:#475569">18-page Goldman-style Institutional PDF Report</td></tr>
</table></div>""", unsafe_allow_html=True)
