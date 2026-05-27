# =============================================================================
# MODULE 6 — GPT-4o MINI AI REPORT GENERATION
# Generates per-holding investment theses + portfolio summary
# Uses OpenAI GPT-4o mini
# =============================================================================

from openai import OpenAI
from typing import Optional


SYSTEM_PROMPT = """You are a CFA-level senior portfolio analyst specializing in Indian equity markets.
Your task is to write concise, data-driven investment memos.
- Use crisp, professional language. No fluff.
- Lead with the investment verdict, back with data.
- Be specific about numbers (cite the exact metrics provided).
- Acknowledge risks honestly — do not oversell.
- Each thesis: max 350 words."""


def generate_holding_thesis(
    company: str,
    ticker: str,
    sector: str,
    portfolio_wt: float,
    current_return: float,
    fundamentals: dict,
    verdict: dict,
    client: OpenAI,
) -> str:
    """
    Generate investment thesis for one holding using GPT-4o mini.
    Returns markdown-formatted thesis string.
    """
    d = fundamentals or {}
    v = verdict or {}

    flags_text = "\n".join(f"  - {f}" for f in v.get("flags", []))
    breakdown_text = "\n".join(
        f"  - {name}: {score}/{max_pts}" for name, score, max_pts in v.get("breakdown", [])
    )

    multiples = []
    if d.get("pe"):        multiples.append(f"PE: {d['pe']:.1f}x")
    if d.get("pb"):        multiples.append(f"P/B: {d['pb']:.2f}x")
    if d.get("ev_ebitda"): multiples.append(f"EV/EBITDA: {d['ev_ebitda']:.1f}x")

    pm = d.get("peer_med", {}) or v.get("peer_med", {})
    peer_multiples = []
    if pm.get("pe"):        peer_multiples.append(f"Peer PE: {pm['pe']:.1f}x")
    if pm.get("pb"):        peer_multiples.append(f"Peer P/B: {pm['pb']:.2f}x")
    if pm.get("ev_ebitda"): peer_multiples.append(f"Peer EV/EBITDA: {pm['ev_ebitda']:.1f}x")

    user_prompt = f"""Write an investment thesis for {company} ({ticker}).

HOLDING DATA:
- Sector: {sector}
- Portfolio Weight: {portfolio_wt:.1f}%
- Current Return (from cost): {current_return:+.1f}%
- Business Model: {d.get('bname', sector)}

FINANCIAL KPIs (latest annual):
- Revenue: ₹{d.get('revenue', 0) or 0:,.0f} Cr  |  Rev CAGR 3yr: {d.get('rev_cagr_3y') or 'N/A'}%
- EBITDA Margin: {d.get('ebitda_margin') or 'N/A'}%  |  PAT Margin: {d.get('pat_margin') or 'N/A'}%
- ROCE: {d.get('roce') or 'N/A'}%  |  ROE: {d.get('roe') or 'N/A'}%
- D/E: {d.get('de_ratio') or 'N/A'}x  |  ND/EBITDA: {d.get('nd_ebitda') or 'N/A'}x
- PAT CAGR 3yr: {d.get('pat_cagr_3y') or 'N/A'}%
- FCF: ₹{d.get('fcf', 0) or 0:,.0f} Cr

VALUATION:
- {' | '.join(multiples) if multiples else 'Limited data'}
- {' | '.join(peer_multiples) if peer_multiples else 'No peer data'}
- Composite Score: {v.get('score') or 'N/A'}/100 → {v.get('label', 'UNRATED')}

SCORE BREAKDOWN:
{breakdown_text if breakdown_text else '  - No breakdown available'}

INVESTMENT FLAGS / SIGNALS:
{flags_text if flags_text else '  - No specific flags generated'}

DATA CONFIDENCE: {d.get('confidence', 0)}%

Write the thesis in this structure:
**INVESTMENT VERDICT** (1 sentence: buy/hold/reduce + why)

**BULL CASE** (2-3 bullets: what needs to go right)

**BEAR CASE / RISKS** (2-3 bullets: key risks and what could go wrong)

**VALUATION CONTEXT** (1-2 sentences: cheap/fair/expensive vs peers, what multiple expansion needs)

**PORTFOLIO RATIONALE** (1 sentence: why this belongs in the portfolio)"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=600,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"*Thesis generation failed: {str(e)}*"


def generate_portfolio_summary(
    portfolio_stats: dict,
    risk_metrics: dict,
    all_verdicts: dict,
    attribution_summary: Optional[dict],
    client: OpenAI,
) -> str:
    """Generate executive summary of the entire portfolio."""
    scores = [v["score"] for v in all_verdicts.values() if v and v.get("score") is not None]
    avg_score = sum(scores) / len(scores) if scores else 0

    label_counts = {}
    for v in all_verdicts.values():
        if v:
            lbl = v.get("label", "UNRATED")
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
    label_str = ", ".join(f"{k}: {v}" for k, v in sorted(label_counts.items()))

    attr_text = ""
    if attribution_summary:
        attr_text = f"""ATTRIBUTION (BHB):
- Total Alpha vs Benchmark: {attribution_summary.get('total_alpha', 0):+.2f}%
- Allocation Effect: {attribution_summary.get('allocation_effect', 0):+.4f}% (sector timing)
- Selection Effect:  {attribution_summary.get('selection_effect', 0):+.4f}% (stock picking)"""

    prompt = f"""Write a 250-word executive summary for this Indian equity portfolio.

PORTFOLIO PERFORMANCE:
- Total Invested: ₹{portfolio_stats.get('total_invested', 0):,.0f}
- Current Value:  ₹{portfolio_stats.get('total_current', 0):,.0f}
- Unrealised P&L: ₹{portfolio_stats.get('total_pnl', 0):+,.0f}
- Total Return:   {portfolio_stats.get('total_return', 0):+.2f}%
- Best performer: {portfolio_stats.get('best_performer', '—')} ({portfolio_stats.get('best_return', 0):+.1f}%)
- Worst performer: {portfolio_stats.get('worst_performer', '—')} ({portfolio_stats.get('worst_return', 0):+.1f}%)

RISK METRICS:
- Active Share: {risk_metrics.get('active_share', 0):.1f}%
- Effective Holdings: {risk_metrics.get('effective_n', 0):.1f}
- HHI: {risk_metrics.get('hhi', 0):.0f}
- Hit Rate: {risk_metrics.get('hit_rate', 0):.1f}%
- Win/Loss Ratio: {risk_metrics.get('win_loss_ratio', 0):.2f}x

VALUATION LANDSCAPE:
- Average Composite Score: {avg_score:.1f}/100
- Verdict Distribution: {label_str}

{attr_text}

Write a professional executive summary covering:
1. Overall portfolio performance and positioning
2. Key strengths (what's working)
3. Key risks and concerns
4. Portfolio construction quality (concentration, diversification)
5. Forward outlook (1-2 sentences)

Use a confident, analytical tone. Be specific with numbers."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=500,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"*Executive summary generation failed: {str(e)}*"


def generate_full_report(
    holdings_df,
    portfolio_stats: dict,
    risk_metrics: dict,
    clean_data: dict,
    verdicts: dict,
    attribution_summary: Optional[dict],
    api_key: str,
    progress_callback=None,
) -> dict:
    """
    Generate all AI content: executive summary + per-holding theses.
    Returns dict with 'summary' and 'theses' keys.
    """
    client = OpenAI(api_key=api_key)
    report = {"summary": "", "theses": {}, "errors": []}

    # Executive summary first
    if progress_callback:
        progress_callback("portfolio", "Generating executive summary...")
    report["summary"] = generate_portfolio_summary(
        portfolio_stats, risk_metrics, verdicts, attribution_summary, client
    )

    # Per-holding theses
    for _, row in holdings_df.iterrows():
        ticker  = row["Ticker"]
        company = row["Company"]
        d       = clean_data.get(ticker, {}) or {}

        if d.get("btype") == "ETF":
            report["theses"][ticker] = (
                f"**{company} (ETF)**\n\n"
                "This is an ETF tracking the NASDAQ 100 index. "
                "Performance is driven by the underlying index constituents. "
                "No fundamental analysis applicable — evaluate based on index performance vs fee."
            )
            continue

        if progress_callback:
            progress_callback(ticker, f"Writing thesis for {company}...")

        thesis = generate_holding_thesis(
            company=company,
            ticker=ticker,
            sector=row["Sector"],
            portfolio_wt=row["Portfolio Wt%"],
            current_return=row["Return %"],
            fundamentals=d,
            verdict=verdicts.get(ticker, {}),
            client=client,
        )
        report["theses"][ticker] = thesis

    return report
