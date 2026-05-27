# =============================================================================
# PORTFOLIO INTELLIGENCE PLATFORM
# A Streamlit app for full portfolio analysis + GPT-4o mini AI report generation
# =============================================================================
# Run: streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tempfile
import os
import io
from datetime import datetime, date

# ── Project modules ───────────────────────────────────────────────────────────
from modules.m1_loader import load_portfolio, fetch_prices, calculate_metrics, portfolio_summary
from modules.m2_sector import aggregate_sectors, plot_sector_overview, plot_cumulative_performance
from modules.m3_attribution import fetch_sector_benchmark_returns, bhb_attribution, plot_attribution_charts
from modules.m4_risk import calculate_risk_metrics, generate_risk_flags, fetch_correlation_data, plot_risk_dashboard
from modules.m5a_data import run_data_layer, BUSINESS_MODEL
from modules.m5b_intelligence import run_intelligence_layer, plot_valuation_dashboard
from modules.m6_ai_report import generate_full_report
from utils.pdf_export import build_report_pdf

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Portfolio Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #FAFBFC; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
        font-size: 14px;
    }
    .metric-card {
        background: white;
        border: 1px solid #DDE3EE;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .verdict-deep-value   { color: #15803D; font-weight: 700; }
    .verdict-attractive   { color: #1D4ED8; font-weight: 700; }
    .verdict-fair         { color: #D97706; font-weight: 700; }
    .verdict-rich         { color: #EA580C; font-weight: 700; }
    .verdict-expensive    { color: #DC2626; font-weight: 700; }
    .section-header {
        font-size: 18px; font-weight: 700; color: #0F172A;
        border-bottom: 2px solid #1D4ED8; padding-bottom: 6px;
        margin-bottom: 16px;
    }
    div[data-testid="stExpander"] { border: 1px solid #DDE3EE; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INIT
# =============================================================================
for key in [
    "holdings", "portfolio_stats", "sector_df", "risk_metrics",
    "risk_flags", "attribution_summary", "attrib_df",
    "clean_data", "verdicts", "ai_report", "analysis_date",
    "figs_overview", "figs_sector", "figs_attribution", "figs_risk", "figs_valuation",
]:
    if key not in st.session_state:
        st.session_state[key] = None


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/64/stock-share.png", width=50)
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # ── Portfolio CSV ────────────────────────────────────────────────────────
    st.markdown("### 📁 Portfolio Data")
    portfolio_file = st.file_uploader(
        "Upload portfolio.csv", type=["csv"],
        help="Same format as Sudhanva's portfolio.csv",
    )

    # ── Analysis Date ─────────────────────────────────────────────────────────
    analysis_date_input = st.date_input(
        "Analysis Date",
        value=date(2026, 1, 5),
        help="Date for price fetch and analysis snapshot",
    )
    analysis_date = datetime.combine(analysis_date_input, datetime.min.time())

    # ── Manual Prices ─────────────────────────────────────────────────────────
    st.markdown("### 🔧 Manual Prices")
    st.caption("For tickers that fail yfinance")
    tatamotors_price = st.number_input("TATAMOTORS.NS", value=420.50, step=0.5)
    bbg_price = st.number_input("BBG.NS", value=159.05, step=0.5)
    manual_prices = {"TATAMOTORS.NS": tatamotors_price, "BBG.NS": bbg_price}

    # ── Screener Excel Files ─────────────────────────────────────────────────
    st.markdown("### 📊 Screener Files")
    st.caption("Upload one Excel per holding (Screener.in export)")

    screener_uploads = {}
    ticker_to_company = {
        "HINDCOPPER.NS": "Hindustan Copper",
        "TATAMOTORS.NS": "Tata Motors",
        "HINDZINC.NS":   "Hindustan Zinc",
        "TATASTEEL.NS":  "Tata Steel",
        "IDFCFIRSTB.NS": "IDFC First Bank",
        "PNB.NS":        "Punjab National Bank",
        "EQUITASBNK.NS": "Equitas SFB",
        "BANKINDIA.NS":  "Bank of India",
        "INDUSTOWER.NS": "Indus Towers",
        "GABRIEL.NS":    "Gabriel India",
        "BORORENEW.NS":  "Borosil Renewables",
        "JIOFIN.NS":     "Jio Financial",
        "BBG.NS":        "BillionBrains",
        "ADANIPOWER.NS": "Adani Power",
        "IEX.NS":        "IEX",
    }

    with st.expander("Upload Screener Excel files", expanded=False):
        for ticker, company in ticker_to_company.items():
            f = st.file_uploader(f"{company}", type=["xlsx"], key=f"sc_{ticker}")
            if f:
                screener_uploads[ticker] = f

    st.markdown("---")

    # ── OpenAI API Key ────────────────────────────────────────────────────────
    st.markdown("### 🤖 GPT-4o mini (Module 6)")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Required for AI thesis generation (GPT-4o mini)",
        placeholder="sk-...",
    )

    st.markdown("---")

    # ── Run Analysis Button ───────────────────────────────────────────────────
    run_btn = st.button(
        "▶  Run Full Analysis",
        type="primary",
        use_container_width=True,
        disabled=(portfolio_file is None),
    )

    if portfolio_file is None:
        st.info("Upload portfolio.csv to begin")


# =============================================================================
# ANALYSIS EXECUTION
# =============================================================================
if run_btn and portfolio_file:
    report_date = analysis_date.strftime("%d %b %Y")

    progress_container = st.container()
    with progress_container:
        st.markdown("---")
        st.markdown("### ⏳ Running Analysis...")

        # ── MODULE 1: Load + Prices ───────────────────────────────────────────
        with st.status("Module 1 — Loading portfolio & fetching prices...", expanded=True) as s1:
            holdings = load_portfolio(portfolio_file)
            prices, price_results = fetch_prices(holdings, analysis_date, manual_prices)
            for ticker, company, price, status in price_results:
                icon = "✅" if status == "live" else ("⚙️" if status == "manual" else "⚠️")
                st.write(f"{icon} {company[:28]}: ₹{price:.2f}  [{status}]")
            holdings = calculate_metrics(holdings, prices)
            stats = portfolio_summary(holdings)
            st.session_state["holdings"] = holdings
            st.session_state["portfolio_stats"] = stats
            st.session_state["analysis_date"] = analysis_date
            s1.update(label="Module 1 ✅ — Portfolio loaded", state="complete")

        # ── MODULE 2: Sector Analysis ─────────────────────────────────────────
        with st.status("Module 2 — Sector allocation analysis...", expanded=False) as s2:
            sector_df = aggregate_sectors(holdings)
            st.session_state["sector_df"] = sector_df
            fig_overview = plot_sector_overview(sector_df, holdings, report_date)
            fig_sector = plot_cumulative_performance(holdings, analysis_date)
            st.session_state["figs_overview"] = fig_overview
            st.session_state["figs_sector"] = fig_sector
            s2.update(label="Module 2 ✅ — Sector charts ready", state="complete")

        # ── MODULE 3: Attribution ─────────────────────────────────────────────
        with st.status("Module 3 — Return attribution analysis...", expanded=False) as s3:
            unique_sectors = sector_df[sector_df["Port_Weight"] > 0]["Sector"].tolist()
            bench_returns, total_bench = fetch_sector_benchmark_returns(unique_sectors, analysis_date)
            attrib_df, attr_summary, holdings_w_contrib = bhb_attribution(
                holdings, sector_df, bench_returns, total_bench
            )
            fig_attr = plot_attribution_charts(attrib_df, holdings_w_contrib, attr_summary, report_date)
            st.session_state["attrib_df"] = attrib_df
            st.session_state["attribution_summary"] = attr_summary
            st.session_state["figs_attribution"] = fig_attr
            s3.update(label="Module 3 ✅ — Attribution complete", state="complete")

        # ── MODULE 4: Risk ────────────────────────────────────────────────────
        with st.status("Module 4 — Risk analytics...", expanded=False) as s4:
            risk_metrics = calculate_risk_metrics(holdings, sector_df)
            risk_flags = generate_risk_flags(risk_metrics, sector_df)
            with st.spinner("Fetching price history for correlation..."):
                corr_returns = fetch_correlation_data(holdings, analysis_date)
            fig_risk = plot_risk_dashboard(holdings, sector_df, risk_metrics, corr_returns, report_date)
            st.session_state["risk_metrics"] = risk_metrics
            st.session_state["risk_flags"] = risk_flags
            st.session_state["figs_risk"] = fig_risk
            s4.update(label="Module 4 ✅ — Risk analytics complete", state="complete")

        # ── MODULE 5: Valuation (if files uploaded) ───────────────────────────
        with st.status("Module 5 — Fundamental data & valuation...", expanded=True) as s5:
            # Save uploaded Screener files to temp dir
            screener_paths = {}
            tmp_dir = tempfile.mkdtemp()
            for ticker, uploaded in screener_uploads.items():
                ext = ".xlsx"
                tmp_path = os.path.join(tmp_dir, f"{ticker.replace('.', '_')}{ext}")
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.read())
                screener_paths[ticker] = tmp_path

            # Fill None for tickers without files
            all_tickers = holdings["Ticker"].tolist()
            all_screener_files = {}
            for t in all_tickers:
                btype, _ = BUSINESS_MODEL.get(t, ("UNKNOWN", t))
                if btype == "ETF":
                    all_screener_files[t] = None
                elif t in screener_paths:
                    all_screener_files[t] = screener_paths[t]
                else:
                    all_screener_files[t] = None

            def screener_progress(ticker, company, status):
                st.write(f"  {company}: {status}")

            clean_data = run_data_layer(holdings, all_screener_files, screener_progress)

            if any(v is not None for v in clean_data.values()):
                verdicts = run_intelligence_layer(clean_data)
                fig_val = plot_valuation_dashboard(clean_data, verdicts, holdings, report_date)
                st.session_state["clean_data"] = clean_data
                st.session_state["verdicts"] = verdicts
                st.session_state["figs_valuation"] = fig_val
                s5.update(label="Module 5 ✅ — Valuation complete", state="complete")
            else:
                st.session_state["clean_data"] = clean_data
                st.session_state["verdicts"] = {}
                s5.update(
                    label="Module 5 ⚠️ — No Screener files uploaded (valuation skipped)",
                    state="complete",
                )

        st.success("✅ Analysis complete! View results in the tabs below.")


# =============================================================================
# MAIN CONTENT AREA — TABS
# =============================================================================
if st.session_state["holdings"] is not None:
    holdings = st.session_state["holdings"]
    stats = st.session_state["portfolio_stats"]
    sector_df = st.session_state["sector_df"]
    risk_metrics = st.session_state["risk_metrics"]
    risk_flags = st.session_state["risk_flags"]
    clean_data = st.session_state["clean_data"] or {}
    verdicts = st.session_state["verdicts"] or {}
    analysis_date = st.session_state["analysis_date"]
    report_date = analysis_date.strftime("%d %b %Y") if analysis_date else "—"
    attr_summary = st.session_state["attribution_summary"]

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "🏭 Sector Analysis",
        "📈 Attribution",
        "⚠️ Risk",
        "💹 Valuation",
        "🤖 AI Report",
    ])

    # =========================================================================
    # TAB 1: PORTFOLIO OVERVIEW
    # =========================================================================
    with tab1:
        st.markdown(f'<div class="section-header">Portfolio Overview — {report_date}</div>', unsafe_allow_html=True)

        # KPI Cards row
        c1, c2, c3, c4, c5 = st.columns(5)
        ret_color = "green" if stats["total_return"] >= 0 else "red"
        pnl_color = "green" if stats["total_pnl"] >= 0 else "red"

        c1.metric("Invested", f"₹{stats['total_invested']:,.0f}")
        c2.metric("Current Value", f"₹{stats['total_current']:,.0f}")
        c3.metric("Unrealised P&L", f"₹{stats['total_pnl']:+,.0f}",
                  delta=f"{stats['total_return']:+.2f}%")
        c4.metric("Winners / Losers", f"{stats['num_winners']} / {stats['num_losers']}")
        c5.metric("Active Share", f"{risk_metrics['active_share']:.1f}%")

        st.markdown("---")

        # Quick stats
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Performer", stats["best_performer"][:22], f"{stats['best_return']:+.1f}%")
        c2.metric("Worst Performer", stats["worst_performer"][:22], f"{stats['worst_return']:+.1f}%")
        c3.metric("Largest Position", stats["largest_position"][:22], f"{stats['largest_wt']:.1f}%")

        st.markdown("---")
        st.markdown("#### Holdings Table")

        # Display table
        display_df = holdings[[
            "Company", "Sector", "Avg Buy Price", "Current Price",
            "Invested Value", "Current Value", "P&L", "Return %",
            "Portfolio Wt%", "Benchmark Wt%", "Active Bet%",
        ]].copy()

        # Color-code return column via Styler
        def color_return(val):
            color = "#16A34A" if val >= 0 else "#DC2626"
            return f"color: {color}; font-weight: bold"

        styled = display_df.style.applymap(color_return, subset=["Return %", "P&L", "Active Bet%"])
        styled = styled.format({
            "Avg Buy Price": "₹{:.2f}", "Current Price": "₹{:.2f}",
            "Invested Value": "₹{:,.0f}", "Current Value": "₹{:,.0f}",
            "P&L": "₹{:+,.0f}", "Return %": "{:+.2f}%",
            "Portfolio Wt%": "{:.2f}%", "Benchmark Wt%": "{:.2f}%",
            "Active Bet%": "{:+.2f}%",
        })
        st.dataframe(styled, use_container_width=True, height=520)

    # =========================================================================
    # TAB 2: SECTOR ANALYSIS
    # =========================================================================
    with tab2:
        st.markdown(f'<div class="section-header">Sector Allocation & Performance</div>', unsafe_allow_html=True)

        if sector_df is not None:
            # Sector table
            st.markdown("#### Sector Breakdown vs Nifty 500")
            sec_display = sector_df[sector_df["Port_Weight"] > 0][[
                "Sector", "Port_Weight", "Bench_Weight", "Active_Bet",
                "Sector_Return%", "Num_Stocks"
            ]].copy()
            sec_display.columns = ["Sector", "Port %", "Bench %", "Active Bet %", "Return %", "# Stocks"]

            def color_active_bet(val):
                color = "#16A34A" if val > 1 else ("#DC2626" if val < -1 else "#0F172A")
                return f"color: {color}; font-weight: bold"

            styled_sec = sec_display.style.applymap(color_active_bet, subset=["Active Bet %"])
            styled_sec = styled_sec.format({
                "Port %": "{:.2f}%", "Bench %": "{:.2f}%",
                "Active Bet %": "{:+.2f}%", "Return %": "{:+.2f}%",
            })
            st.dataframe(styled_sec, use_container_width=True)

        st.markdown("---")

        if st.session_state["figs_overview"]:
            st.pyplot(st.session_state["figs_overview"], use_container_width=True)

        st.markdown("---")
        st.markdown("#### 5-Year Cumulative Return vs Benchmark")
        if st.session_state["figs_sector"]:
            st.pyplot(st.session_state["figs_sector"], use_container_width=True)

    # =========================================================================
    # TAB 3: ATTRIBUTION
    # =========================================================================
    with tab3:
        st.markdown(f'<div class="section-header">Brinson-Hood-Beebower Attribution</div>', unsafe_allow_html=True)

        if attr_summary:
            c1, c2, c3, c4 = st.columns(4)
            delta_color = "normal" if attr_summary["total_alpha"] >= 0 else "inverse"
            c1.metric("Portfolio Return", f"{attr_summary['total_port_return']:+.2f}%")
            c2.metric("Benchmark Return", f"{attr_summary['total_bench_return']:+.2f}%")
            c3.metric("Total Alpha", f"{attr_summary['total_alpha']:+.2f}%",
                      delta=f"vs benchmark")
            c4.metric("Hit Rate", f"{risk_metrics['hit_rate']:.1f}%")

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Allocation Effect", f"{attr_summary['allocation_effect']:+.4f}%",
                      help="Skill in choosing right sectors")
            c2.metric("Selection Effect", f"{attr_summary['selection_effect']:+.4f}%",
                      help="Skill in picking stocks within sectors")
            c3.metric("Interaction Effect", f"{attr_summary['interaction_effect']:+.4f}%",
                      help="Combined sector timing + stock picking")

            st.markdown("---")
            # Attribution interpretation
            alloc = attr_summary["allocation_effect"]
            sel = attr_summary["selection_effect"]
            if abs(alloc) > abs(sel):
                insight = f"📌 Alpha driven by **sector allocation** ({alloc:+.4f}%) — macro/thematic skill dominates"
            else:
                insight = f"📌 Alpha driven by **stock selection** ({sel:+.4f}%) — fundamental research is key edge"
            st.info(insight)

        if st.session_state["attrib_df"] is not None:
            st.markdown("#### BHB Attribution Table")
            attrib_df = st.session_state["attrib_df"]
            st.dataframe(
                attrib_df.style.format({
                    "Port_Wt%": "{:.2f}%", "Bench_Wt%": "{:.2f}%",
                    "Active_Bet%": "{:+.2f}%",
                    "Allocation": "{:+.4f}%", "Selection": "{:+.4f}%",
                    "Active_Return": "{:+.4f}%",
                }),
                use_container_width=True,
            )

        if st.session_state["figs_attribution"]:
            st.markdown("---")
            st.pyplot(st.session_state["figs_attribution"], use_container_width=True)

    # =========================================================================
    # TAB 4: RISK ANALYTICS
    # =========================================================================
    with tab4:
        st.markdown(f'<div class="section-header">Portfolio Risk Analytics</div>', unsafe_allow_html=True)

        if risk_metrics:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Active Share", f"{risk_metrics['active_share']:.1f}%",
                      help="0% = pure index | 100% = fully active")
            c2.metric("HHI", f"{risk_metrics['hhi']:.0f}",
                      help="Concentration index. < 1000 = diversified")
            c3.metric("Effective N", f"{risk_metrics['effective_n']:.1f}",
                      help="Equivalent equal-weighted holdings")
            c4.metric("Hit Rate", f"{risk_metrics['hit_rate']:.1f}%")
            c5.metric("Win/Loss", f"{risk_metrics['win_loss_ratio']:.2f}x")

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Top 1 Weight", f"{risk_metrics['top1_wt']:.1f}%")
            c2.metric("Top 5 Weight", f"{risk_metrics['top5_wt']:.1f}%")
            c3.metric("Top 10 Weight", f"{risk_metrics['top10_wt']:.1f}%")

            st.markdown("---")
            st.markdown("#### Risk Flags")
            if risk_flags:
                for flag in risk_flags:
                    level = flag["level"]
                    if "⚠️" in level:
                        st.error(level)
                    elif "🟡" in level:
                        st.warning(level)
                    elif "✅" in level:
                        st.success(level)
                    else:
                        st.info(level)

        if st.session_state["figs_risk"]:
            st.markdown("---")
            st.pyplot(st.session_state["figs_risk"], use_container_width=True)

    # =========================================================================
    # TAB 5: VALUATION & PEER ANALYSIS
    # =========================================================================
    with tab5:
        st.markdown(f'<div class="section-header">Valuation & Peer Analysis (Module 5)</div>', unsafe_allow_html=True)

        if not verdicts:
            st.warning(
                "⚠️ No Screener Excel files uploaded. "
                "Upload them in the sidebar to run fundamental analysis."
            )
        else:
            # Summary scores
            scores_list = [(t, v["score"], v["label"]) for t, v in verdicts.items() if v and v.get("score")]
            if scores_list:
                avg = sum(s for _, s, _ in scores_list) / len(scores_list)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Holdings Analysed", len(scores_list))
                c2.metric("Avg Composite Score", f"{avg:.1f}/100")
                deep_val = sum(1 for _, _, l in scores_list if l == "DEEP VALUE")
                attractive = sum(1 for _, _, l in scores_list if l == "ATTRACTIVE")
                c3.metric("Deep Value + Attractive", f"{deep_val + attractive}")
                expensive = sum(1 for _, _, l in scores_list if l in ("RICH", "EXPENSIVE"))
                c4.metric("Rich + Expensive", f"{expensive}")

            st.markdown("---")

            # Valuation table
            st.markdown("#### Composite Scores & Multiples")
            val_rows = []
            for _, row in holdings.iterrows():
                ticker = row["Ticker"]
                d = clean_data.get(ticker) or {}
                v = verdicts.get(ticker) or {}
                val_rows.append({
                    "Company":   row["Company"][:24],
                    "Sector":    row["Sector"][:18],
                    "Score":     v.get("score"),
                    "Verdict":   v.get("label", "—"),
                    "PE":        d.get("pe"),
                    "PB":        d.get("pb"),
                    "EV/EBITDA": d.get("ev_ebitda"),
                    "ROCE%":     d.get("roce"),
                    "ROE%":      d.get("roe"),
                    "RevCAGR3y": d.get("rev_cagr_3y"),
                    "Confidence":d.get("confidence"),
                })

            val_df = pd.DataFrame(val_rows)
            st.dataframe(
                val_df.style.format({
                    "Score":     "{:.0f}",
                    "PE":        "{:.1f}",
                    "PB":        "{:.2f}",
                    "EV/EBITDA": "{:.1f}",
                    "ROCE%":     "{:.1f}",
                    "ROE%":      "{:.1f}",
                    "RevCAGR3y": "{:.1f}",
                    "Confidence":"{:.0f}",
                }, na_rep="—"),
                use_container_width=True,
            )

            # Per-holding expanders
            st.markdown("---")
            st.markdown("#### Individual Holding Details")
            for _, row in holdings.iterrows():
                ticker = row["Ticker"]
                d = clean_data.get(ticker) or {}
                v = verdicts.get(ticker) or {}
                if not d:
                    continue

                verdict_lbl = v.get("label", "—")
                score_str = f"{v.get('score'):.0f}" if v.get("score") else "—"

                with st.expander(f"{row['Company']}  |  {verdict_lbl} ({score_str}/100)"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**P&L Metrics**")
                        st.write(f"Revenue: ₹{d.get('revenue', 0) or 0:,.0f} Cr")
                        st.write(f"EBITDA Margin: {d.get('ebitda_margin') or '—'}%")
                        st.write(f"PAT Margin: {d.get('pat_margin') or '—'}%")
                        st.write(f"Rev CAGR 3yr: {d.get('rev_cagr_3y') or '—'}%")
                        st.write(f"PAT CAGR 3yr: {d.get('pat_cagr_3y') or '—'}%")
                    with c2:
                        st.markdown("**Returns & Balance Sheet**")
                        st.write(f"ROCE: {d.get('roce') or '—'}%")
                        st.write(f"ROE: {d.get('roe') or '—'}%")
                        st.write(f"D/E: {d.get('de_ratio') or '—'}x")
                        st.write(f"ND/EBITDA: {d.get('nd_ebitda') or '—'}x")
                        st.write(f"FCF: ₹{d.get('fcf', 0) or 0:,.0f} Cr")

                    if v.get("flags"):
                        st.markdown("**Investment Signals**")
                        for f in v["flags"]:
                            st.write(f"• {f}")

                    if v.get("breakdown"):
                        st.markdown("**Score Breakdown**")
                        for name, score, max_pts in v["breakdown"]:
                            pct = score / max_pts if max_pts > 0 else 0
                            st.progress(pct, text=f"{name}: {score}/{max_pts}")

        if st.session_state["figs_valuation"]:
            st.markdown("---")
            st.pyplot(st.session_state["figs_valuation"], use_container_width=True)

    # =========================================================================
    # TAB 6: AI REPORT
    # =========================================================================
    with tab6:
        st.markdown(f'<div class="section-header">🤖 GPT-4o mini Investment Report</div>', unsafe_allow_html=True)

        if not api_key:
            st.warning("⚠️ Enter your OpenAI API key in the sidebar to generate AI theses.")
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(
                    "Generates a per-holding investment thesis (bull/bear/valuation/portfolio rationale) "
                    "using GPT-4o mini, plus an executive portfolio summary."
                )
            with col2:
                gen_btn = st.button("🤖 Generate AI Report", type="primary", use_container_width=True)

            if gen_btn:
                if st.session_state["portfolio_stats"] is None:
                    st.error("Run the full analysis first (click 'Run Full Analysis' in sidebar)")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    total = len(holdings)
                    done = [0]

                    def ai_progress(ticker, msg):
                        done[0] += 1
                        progress_bar.progress(min(done[0] / (total + 1), 1.0))
                        status_text.text(f"⏳ {msg}")

                    with st.spinner("Generating AI report..."):
                        ai_report = generate_full_report(
                            holdings_df=holdings,
                            portfolio_stats=st.session_state["portfolio_stats"],
                            risk_metrics=st.session_state["risk_metrics"],
                            clean_data=clean_data,
                            verdicts=verdicts,
                            attribution_summary=attr_summary,
                            api_key=api_key,
                            progress_callback=ai_progress,
                        )
                    st.session_state["ai_report"] = ai_report
                    progress_bar.progress(1.0)
                    status_text.text("✅ AI report generation complete!")
                    st.success("Report generated! See below.")

        # Display AI report
        ai_report = st.session_state.get("ai_report")
        if ai_report:
            st.markdown("---")
            st.markdown("### Executive Summary")
            st.markdown(ai_report.get("summary", "*Not generated*"))

            st.markdown("---")
            st.markdown("### Individual Investment Theses")

            for _, row in holdings.iterrows():
                ticker = row["Ticker"]
                company = row["Company"]
                v = verdicts.get(ticker) or {}
                verdict_lbl = v.get("label", "")
                score_str = f"{v.get('score'):.0f}/100" if v.get("score") else ""

                with st.expander(
                    f"**{company}**  ·  {ticker}  ·  {verdict_lbl} {score_str}",
                    expanded=False,
                ):
                    thesis = ai_report.get("theses", {}).get(ticker, "")
                    if thesis:
                        st.markdown(thesis)
                    else:
                        st.write("*Thesis not generated*")

        # PDF Download
        st.markdown("---")
        if st.button("📄 Generate & Download PDF Report", use_container_width=True):
            if st.session_state["portfolio_stats"] is None:
                st.error("Run analysis first")
            else:
                with st.spinner("Building PDF report..."):
                    figures = {}
                    if st.session_state["figs_overview"]:   figures["overview"] = st.session_state["figs_overview"]
                    if st.session_state["figs_sector"]:     figures["sector"] = st.session_state["figs_sector"]
                    if st.session_state["figs_attribution"]:figures["attribution"] = st.session_state["figs_attribution"]
                    if st.session_state["figs_risk"]:        figures["risk"] = st.session_state["figs_risk"]
                    if st.session_state["figs_valuation"]:  figures["valuation"] = st.session_state["figs_valuation"]

                    pdf_bytes = build_report_pdf(
                        report_date=analysis_date.strftime("%d %b %Y"),
                        portfolio_stats=st.session_state["portfolio_stats"],
                        risk_metrics=st.session_state["risk_metrics"] or {},
                        risk_flags=st.session_state["risk_flags"] or [],
                        attribution_summary=attr_summary,
                        holdings_df=holdings,
                        clean_data=clean_data,
                        verdicts=verdicts,
                        ai_report=st.session_state.get("ai_report") or {"summary": "", "theses": {}},
                        figures=figures,
                    )

                fname = f"portfolio_report_{analysis_date.strftime('%Y%m%d')}.pdf"
                st.download_button(
                    label="⬇️  Download PDF",
                    data=pdf_bytes,
                    file_name=fname,
                    mime="application/pdf",
                    use_container_width=True,
                )
                st.success(f"PDF ready: {fname}")

# =============================================================================
# EMPTY STATE
# =============================================================================
else:
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; padding: 60px 20px;">
            <h2 style="color:#1D4ED8;">📊 Portfolio Intelligence Platform</h2>
            <p style="color:#475569; font-size:16px;">
                Upload your <code>portfolio.csv</code> in the sidebar and click
                <b>Run Full Analysis</b> to begin.
            </p>
            <br/>
            <table style="margin:auto; border-collapse:collapse; text-align:left;">
                <tr><td style="padding:8px 16px; color:#0F172A;"><b>Module 1</b></td>
                    <td style="padding:8px; color:#475569;">Live prices, P&L, performance table</td></tr>
                <tr style="background:#F8F9FC;"><td style="padding:8px 16px;"><b>Module 2</b></td>
                    <td style="padding:8px; color:#475569;">Sector allocation vs Nifty 500 benchmark</td></tr>
                <tr><td style="padding:8px 16px;"><b>Module 3</b></td>
                    <td style="padding:8px; color:#475569;">Brinson-Hood-Beebower attribution (allocation vs selection)</td></tr>
                <tr style="background:#F8F9FC;"><td style="padding:8px 16px;"><b>Module 4</b></td>
                    <td style="padding:8px; color:#475569;">Risk analytics — active share, HHI, correlation, flags</td></tr>
                <tr><td style="padding:8px 16px;"><b>Module 5</b></td>
                    <td style="padding:8px; color:#475569;">Fundamental analysis, peer multiples, composite scoring</td></tr>
                <tr style="background:#F8F9FC;"><td style="padding:8px 16px;"><b>Module 6</b></td>

                
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
