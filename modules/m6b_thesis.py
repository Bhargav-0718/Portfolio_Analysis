# =============================================================================
# MODULE 6B — AI INSTITUTIONAL THESIS ENGINE (M6.3)
# 7-prompt chain → 12-point institutional output per stock
# Custom investment philosophy personality layer
# =============================================================================

import json
from openai import OpenAI
from modules.m5e_context import context_portfolio_brief
from modules.m6a_research import research_to_text
from typing import Optional

# =============================================================================
# DEFAULT INVESTMENT PHILOSOPHY (from roadmap — user can override in sidebar)
# =============================================================================
DEFAULT_PHILOSOPHY = """My investment philosophy:
- Focus on asymmetric payoffs — I want 3:1 upside:downside minimum
- Look for scalable businesses where incremental returns exceed cost of capital
- Value cyclical rerating carefully — buy cyclicals only at trough earnings, not peak
- Avoid weak governance — promoter integrity is non-negotiable
- Prioritize operating leverage — I want businesses where revenue growth beats cost growth
- Look for capital efficiency — ROCE > 15% consistently is the quality bar
- Reward strong management commentary — clarity, honesty, and delivery track record matter
- Avoid low-quality balance sheets — D/E > 2x or ND/EBITDA > 5x is a red flag"""

# =============================================================================
# THE 7-PROMPT SYSTEM (from prompt_AI.pdf)
# Applied sequentially per stock — each builds on the prior output
# =============================================================================

def _p1_core_thesis(stock_context: str, philosophy: str) -> str:
    return f"""You are a senior buy-side equity research analyst writing internal hedge fund memos.
Your goal is NOT to describe the company. Your goal is to generate an INVESTMENT DECISION.

{philosophy}

You always think in terms of:
- market expectations vs reality
- earnings durability (structural vs cyclical)
- capital efficiency
- asymmetry of outcomes
- risk of permanent capital loss

STOCK DATA:
{stock_context}

Generate the CORE INVESTMENT THESIS in 3-4 sentences.
Every sentence must answer: "So what does this mean for the stock?"
Write like a portfolio manager, not an analyst.
Be decisive. No neutral language.
Output: plain text only."""


def _p2_insight_generator(stock_context: str, core_thesis: str) -> str:
    return f"""You are analyzing this company ONLY to identify mispricing between market expectations and business reality.

STOCK DATA:
{stock_context}

CORE THESIS (already established):
{core_thesis}

Now identify the specific mispricing. Output ONLY valid JSON:
{{
  "market_expectation": "What the market is currently pricing in (be specific)",
  "reality_from_data": "What the actual fundamental data shows",
  "mispricing_thesis": "The specific gap between market expectation and reality",
  "what_changes_it": "The specific catalyst or event that will close this gap"
}}

Rules:
- Do NOT summarize the business
- Do NOT list generic positives/negatives
- Focus ONLY on expectation mismatches
- Be forward-looking"""


def _p3_conviction_builder(stock_context: str, score: float, asymmetry: float) -> str:
    return f"""You are a portfolio manager deciding whether to increase, hold, or reduce exposure.

STOCK DATA:
{stock_context}

Composite Score: {score}/100 | Asymmetry Ratio: {asymmetry:.1f}x

Evaluate across these 5 factors:
1. Earnings durability (structural vs cyclical)
2. Quality of capital allocation
3. Competitive positioning strength
4. Risk asymmetry (downside vs upside)
5. Catalyst visibility

Then output ONLY valid JSON:
{{
  "conviction": "HIGH or MEDIUM or LOW",
  "action": "ACCUMULATE or HOLD or TRIM or EXIT",
  "action_rationale": "2-3 high-impact sentences only. No generic language. Take a firm stance.",
  "capital_allocation_quality": "Specific assessment of how management deploys capital",
  "earnings_durability": "STRUCTURAL or CYCLICAL or MIXED — with 1-sentence justification"
}}"""


def _p4_contrarian_lens(stock_context: str, core_thesis: str) -> str:
    return f"""You are a contrarian equity research analyst.
Your job is to actively challenge the bullish narrative.

STOCK DATA:
{stock_context}

ESTABLISHED BULL THESIS:
{core_thesis}

Output ONLY valid JSON:
{{
  "bull_narrative": "The strongest version of the bull case currently in the market",
  "weakness_in_bull_case": "Specific assumption in the bull case that is questionable",
  "hidden_risks": ["risk ignored by consensus 1", "risk ignored by consensus 2"],
  "bear_catalyst": "The specific event that would break the thesis",
  "permanent_loss_risk": "LOW or MEDIUM or HIGH — with 1-sentence justification"
}}

Rules:
- Be intellectually aggressive but data-grounded
- Focus more on what could break the thesis than validating it"""


def _p5_sector_analysis(stock_context: str, btype: str, sector: str) -> str:
    return f"""You are analyzing this company within its sector ecosystem.
Business type: {btype} | Sector: {sector}

STOCK DATA:
{stock_context}

Output ONLY valid JSON:
{{
  "sector_positioning": "Where does this company sit in the sector hierarchy (leader/laggard/disruptor)?",
  "competitive_advantage": "Specific moat or lack thereof — be precise",
  "peer_comparison_insight": "How does it compare to peers on the key metrics?",
  "sector_tailwind": "Current sector tailwind benefiting this stock",
  "sector_headwind": "Current sector headwind threatening this stock"
}}"""


def _p6_structuring_prompt(
    company: str, ticker: str,
    core_thesis: str, insight: dict, conviction: dict,
    contrarian: dict, sector: dict,
    scenario_bear: float, scenario_bull: float, scenario_base: float,
    monitoring_kpis: list,
    portfolio_role: str,
) -> str:
    return f"""Convert the following analysis into a hedge fund style investment memo for {company} ({ticker}).

ESTABLISHED ANALYSIS:
Core Thesis: {core_thesis}
Market Expectation vs Reality: {insight.get('market_expectation', '')} → {insight.get('reality_from_data', '')}
Mispricing: {insight.get('mispricing_thesis', '')}
Catalyst: {insight.get('what_changes_it', '')}
Conviction: {conviction.get('conviction', '')} | Action: {conviction.get('action', '')}
Contrarian: Hidden risks: {contrarian.get('hidden_risks', [])} | Bear catalyst: {contrarian.get('bear_catalyst', '')}
Sector: {sector.get('sector_positioning', '')} | Advantage: {sector.get('competitive_advantage', '')}
Scenarios: Bear {scenario_bear:+.1f}% | Base {scenario_base:+.0f}% | Bull {scenario_bull:+.1f}%

Requirements:
- No descriptive writing
- No repetition of data
- Only investment-relevant interpretation
- Every sentence must impact investment understanding

Output ONLY valid JSON (no markdown, no code blocks):
{{
  "investment_thesis": "Edge-based 2-3 sentence thesis. Not descriptive. What is mispriced and why.",
  "market_expectation_vs_reality": "Specific gap between what market prices in vs fundamental reality",
  "why_market_wrong": "Where consensus is incorrect or incomplete",
  "bull_case": ["bullet 1", "bullet 2", "bullet 3"],
  "bear_case": ["bullet 1", "bullet 2"],
  "key_triggers": ["trigger 1", "trigger 2", "trigger 3"],
  "risk_factors": ["risk 1", "risk 2", "risk 3"],
  "capital_allocation_quality": "Specific management capital deployment assessment",
  "management_assessment": "Credibility, execution track record, communication quality",
  "portfolio_role": "{portfolio_role}",
  "position_sizing_logic": "Why weight is high/medium/low based on conviction + asymmetry + risk",
  "monitoring_framework": {json.dumps(monitoring_kpis if monitoring_kpis else ["Quarterly earnings", "Margin trajectory", "Debt levels"])},
  "long_term_outlook": "2-3 year forward view on business quality and value creation"
}}"""


def _p7_validation_filter(thesis_json: dict, company: str) -> tuple:
    """
    Non-generic filter: validates thesis quality.
    Returns (is_valid: bool, issues: list)
    """
    issues = []
    it = thesis_json.get("investment_thesis", "")
    mevr = thesis_json.get("market_expectation_vs_reality", "")
    wmw = thesis_json.get("why_market_wrong", "")

    # Check for generic AI language
    generic_phrases = [
        "strong fundamentals", "well-positioned", "significant opportunities",
        "faces challenges", "management is focused", "going forward",
        "it is important to note", "overall",
    ]
    full_text = f"{it} {mevr} {wmw}".lower()
    for phrase in generic_phrases:
        if phrase in full_text:
            issues.append(f"Generic language detected: '{phrase}'")

    # Must mention a specific number or metric
    import re
    has_number = bool(re.search(r'\d+\.?\d*[x%]|\b\d{2,}\b', f"{it} {mevr}"))
    if not has_number:
        issues.append("No specific metric or number in core thesis")

    # Must take a stance in investment_thesis
    neutral_words = ["may", "could potentially", "might", "unclear", "uncertain"]
    for word in neutral_words:
        if word in it.lower():
            issues.append(f"Neutral/indecisive language: '{word}'")

    is_valid = len(issues) == 0
    return is_valid, issues


# =============================================================================
# MAIN THESIS GENERATOR
# =============================================================================

def generate_stock_thesis(
    ticker: str,
    company: str,
    context_holding: dict,
    philosophy: str,
    client: OpenAI,
    max_retries: int = 1,
) -> dict:
    """
    Run the 7-prompt chain for one stock.
    Returns the 12-point institutional thesis dict.
    """
    d        = context_holding.get("fundamentals", {})
    v        = context_holding.get("valuation", {})
    sc       = context_holding.get("scenarios", {})
    ac       = context_holding.get("action", {})
    res      = context_holding.get("research", {})
    pos      = context_holding.get("position", {})

    # Build compact stock context string
    research_text = research_to_text(res, company) if res else ""
    btype = d.get("btype", "UNKNOWN")
    sector = context_holding.get("sector", "")

    stock_context = f"""Company: {company} ({ticker})
Sector: {sector} | Business: {d.get('bname', btype)}
Portfolio Weight: {pos.get('portfolio_wt', 0):.1f}% | Return: {pos.get('return_pct', 0):+.1f}%

FINANCIALS (latest annual):
Revenue: ₹{d.get('revenue', 0) or 0:,.0f} Cr | EBITDA Margin: {d.get('ebitda_margin') or 'N/A'}%
PAT: ₹{d.get('pat', 0) or 0:,.0f} Cr | PAT Margin: {d.get('pat_margin') or 'N/A'}%
ROCE: {d.get('roce') or 'N/A'}% | ROE: {d.get('roe') or 'N/A'}%
D/E: {d.get('de_ratio') or 'N/A'}x | ND/EBITDA: {d.get('nd_ebitda') or 'N/A'}x
Rev CAGR 3yr: {d.get('rev_cagr_3y') or 'N/A'}% | PAT CAGR 3yr: {d.get('pat_cagr_3y') or 'N/A'}%

VALUATION:
PE: {v.get('pe') or 'N/A'}x | PB: {v.get('pb') or 'N/A'}x | EV/EBITDA: {v.get('ev_ebitda') or 'N/A'}x
Composite Score: {v.get('score') or 'N/A'}/100 → {v.get('label', '')}
Peer median EV/EBITDA: {(v.get('peer_med') or {}).get('ev_ebitda') or 'N/A'}x
Investment flags: {'; '.join(v.get('flags', [])[:4])}

SCENARIOS: Bear {(sc.get('bear') or {}).get('return_pct', 'N/A')}% | Base 0% | Bull {(sc.get('bull') or {}).get('return_pct', 'N/A')}%
Asymmetry: {sc.get('asymmetry_ratio', 'N/A')}x | Role: {sc.get('portfolio_role', 'N/A')}
Action Signal: {ac.get('signal', 'N/A')} (conviction: {ac.get('conviction', 'N/A')})

{research_text}"""

    score     = v.get("score") or 50
    asymmetry = sc.get("asymmetry_ratio") or 1.0
    bear_ret  = (sc.get("bear") or {}).get("return_pct", -20)
    bull_ret  = (sc.get("bull") or {}).get("return_pct", +30)
    role      = sc.get("portfolio_role", "TACTICAL")
    monitoring_kpis = [
        f"{d.get('btype', '')} margins trajectory",
        "Quarterly revenue growth vs guidance",
        f"{'NIM / credit cost' if btype in ('PSU_BANK','SFB','NBFC') else 'ROCE / FCF'}",
        "Management guidance delivery",
        "Debt reduction / leverage",
    ]

    def _call(prompt: str, json_mode: bool = False, max_tokens: int = 600) -> str:
        kwargs = dict(
            model="gpt-4o-mini",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": f"You are an institutional equity analyst. {philosophy}"},
                {"role": "user",   "content": prompt},
            ],
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    for attempt in range(max_retries + 1):
        try:
            # P1 — Core thesis
            core_thesis = _call(_p1_core_thesis(stock_context, philosophy), max_tokens=300)

            # P2 — Insight / mispricing
            insight_raw = _call(_p2_insight_generator(stock_context, core_thesis), json_mode=True, max_tokens=400)
            insight     = json.loads(insight_raw)

            # P3 — Conviction
            conv_raw  = _call(_p3_conviction_builder(stock_context, score, asymmetry), json_mode=True, max_tokens=400)
            conviction = json.loads(conv_raw)

            # P4 — Contrarian
            contra_raw  = _call(_p4_contrarian_lens(stock_context, core_thesis), json_mode=True, max_tokens=400)
            contrarian  = json.loads(contra_raw)

            # P5 — Sector analysis
            sector_raw  = _call(_p5_sector_analysis(stock_context, btype, sector), json_mode=True, max_tokens=350)
            sector_anal = json.loads(sector_raw)

            # P6 — Structuring into 12-point memo
            struct_prompt = _p6_structuring_prompt(
                company, ticker, core_thesis, insight, conviction, contrarian, sector_anal,
                bear_ret, bull_ret, 0.0, monitoring_kpis, role,
            )
            struct_raw = _call(struct_prompt, json_mode=True, max_tokens=1000)
            thesis     = json.loads(struct_raw)

            # P7 — Validation filter
            is_valid, issues = _p7_validation_filter(thesis, company)

            # Add derived fields
            thesis["action"]    = conviction.get("action", ac.get("signal", "HOLD"))
            thesis["conviction"] = conviction.get("conviction", ac.get("conviction", "MEDIUM"))
            thesis["portfolio_role"] = role
            thesis["validation_issues"] = issues if not is_valid else []
            thesis["bear_return_pct"]  = bear_ret
            thesis["bull_return_pct"]  = bull_ret
            thesis["asymmetry_ratio"]  = asymmetry

            return thesis

        except Exception as e:
            if attempt == max_retries:
                return {
                    "investment_thesis":           f"Thesis generation failed: {e}",
                    "market_expectation_vs_reality": "",
                    "why_market_wrong":            "",
                    "bull_case":                   [],
                    "bear_case":                   [],
                    "key_triggers":                [],
                    "risk_factors":                v.get("flags", []),
                    "capital_allocation_quality":  "",
                    "management_assessment":       "",
                    "portfolio_role":              role,
                    "position_sizing_logic":       ac.get("sizing_rationale", ""),
                    "monitoring_framework":        monitoring_kpis,
                    "long_term_outlook":           "",
                    "action":                      ac.get("signal", "HOLD"),
                    "conviction":                  ac.get("conviction", "MEDIUM"),
                    "error":                       str(e),
                }


# =============================================================================
# PORTFOLIO-LEVEL OUTPUTS
# =============================================================================

def generate_pm_commentary(
    context: dict,
    stock_theses: dict,
    philosophy: str,
    client: OpenAI,
) -> str:
    """Generate final PM Commentary (Page 18 of report)."""
    brief = context_portfolio_brief(context)
    action_summary = context.get("action_summary", {})
    counts = action_summary.get("signal_counts", {})

    acc  = ", ".join(action_summary.get("accumulate_names", [])[:3]) or "none"
    trim = ", ".join(action_summary.get("trim_names", [])[:3]) or "none"
    ext  = ", ".join(action_summary.get("exit_names", [])[:2]) or "none"

    # Gather roles
    roles = {}
    for ticker, th in stock_theses.items():
        role = th.get("portfolio_role", "TACTICAL")
        roles.setdefault(role, []).append(ticker)

    role_summary = " | ".join(f"{r}: {len(t)}" for r, t in roles.items())

    prompt = f"""You are a senior portfolio manager writing your final PM Commentary for an investment committee report.

PORTFOLIO BRIEF:
{brief}

ACTION SIGNALS: Accumulate: {counts.get('ACCUMULATE', 0)} | Hold: {counts.get('HOLD', 0)} | Trim: {counts.get('TRIM', 0)} | Exit: {counts.get('EXIT', 0)}
Top adds: {acc}
Trim candidates: {trim}
Exit: {ext}
Portfolio roles: {role_summary}

PHILOSOPHY:
{philosophy}

Write a 200-word PM Commentary covering:
1. What changed this cycle (1-2 sentences)
2. Key conviction shifts (which positions changed conviction + why)
3. Forward outlook (macro + positioning)
4. Final positioning statement (net bias + key risk to watch)

Write in first person. Be decisive. No generic language. Every sentence must affect investment understanding."""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=400,
            messages=[
                {"role": "system", "content": "You are a senior portfolio manager at a hedge fund."},
                {"role": "user",   "content": prompt},
            ],
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"PM Commentary generation failed: {e}"


def generate_executive_summary(
    context: dict,
    stock_theses: dict,
    philosophy: str,
    client: OpenAI,
) -> str:
    """Generate the executive portfolio summary."""
    brief = context_portfolio_brief(context)
    ps    = context.get("portfolio_summary", {})

    scores = [th.get("conviction", "LOW") for th in stock_theses.values()]
    high_conv = scores.count("HIGH")
    med_conv  = scores.count("MEDIUM")

    # Portfolio identity from roles
    roles = [th.get("portfolio_role", "") for th in stock_theses.values()]
    compounders = roles.count("COMPOUNDER")
    cyclicals   = roles.count("CYCLICAL")
    portfolio_type = (
        "quality-growth" if compounders > cyclicals
        else "cyclical-value" if cyclicals > compounders
        else "hybrid"
    )

    prompt = f"""Write a 200-word institutional executive summary for this portfolio.

{brief}

Portfolio type: {portfolio_type}
High conviction positions: {high_conv} | Medium: {med_conv}
Total return: {ps.get('total_return', 0):+.2f}%
Best performer: {ps.get('best_performer', '?')} ({ps.get('best_return', 0):+.1f}%)
Worst performer: {ps.get('worst_performer', '?')} ({ps.get('worst_return', 0):+.1f}%)

Write like a hedge fund quarterly letter. Cover:
1. Portfolio identity and what it is trying to do
2. Performance drivers (what worked, what didn't)
3. Portfolio quality today (conviction, risk, positioning)
4. Key risks to the portfolio
5. One forward-looking statement

No generic language. Be specific."""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=400,
            messages=[
                {"role": "system", "content": "You are writing an institutional investment committee memo."},
                {"role": "user",   "content": prompt},
            ],
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Executive summary generation failed: {e}"


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def generate_institutional_report(
    context_json: dict,
    philosophy: str,
    api_key: str,
    progress_callback=None,
) -> dict:
    """
    Generate full institutional report:
    - Per-stock 12-point thesis (7-prompt chain)
    - PM Commentary
    - Executive Summary
    - Portfolio identity

    Returns: {
        "stock_theses": {ticker: thesis_dict},
        "pm_commentary": str,
        "executive_summary": str,
        "portfolio_identity": str,
    }
    """
    client     = OpenAI(api_key=api_key)
    philosophy = philosophy or DEFAULT_PHILOSOPHY
    holdings   = context_json.get("holdings", {})
    report     = {"stock_theses": {}, "pm_commentary": "", "executive_summary": "", "portfolio_identity": ""}

    # Per-stock theses
    for ticker, holding_ctx in holdings.items():
        company = holding_ctx.get("company", ticker)
        if progress_callback:
            progress_callback(ticker, f"Generating institutional thesis for {company}...")

        btype = holding_ctx.get("fundamentals", {}).get("btype", "UNKNOWN")
        if btype == "ETF":
            report["stock_theses"][ticker] = {
                "investment_thesis": "ETF tracking NASDAQ 100. No fundamental thesis applicable.",
                "portfolio_role":    "TACTICAL",
                "action":            "HOLD",
                "conviction":        "MEDIUM",
                "bull_case":         ["NASDAQ 100 continues AI-driven earnings growth"],
                "bear_case":         ["USD/INR depreciation", "Tech valuation compression"],
                "key_triggers":      ["US Fed policy", "Big Tech earnings"],
                "risk_factors":      ["Currency risk", "Tracking error"],
                "monitoring_framework": ["NASDAQ 100 returns", "USD/INR"],
            }
            continue

        thesis = generate_stock_thesis(ticker, company, holding_ctx, philosophy, client)
        report["stock_theses"][ticker] = thesis

    # Portfolio-level
    if progress_callback:
        progress_callback("portfolio", "Generating executive summary...")
    report["executive_summary"] = generate_executive_summary(
        context_json, report["stock_theses"], philosophy, client
    )

    if progress_callback:
        progress_callback("portfolio", "Generating PM commentary...")
    report["pm_commentary"] = generate_pm_commentary(
        context_json, report["stock_theses"], philosophy, client
    )

    # Portfolio identity label
    roles = [th.get("portfolio_role", "") for th in report["stock_theses"].values()]
    compounders = roles.count("COMPOUNDER")
    cyclicals   = roles.count("CYCLICAL")
    report["portfolio_identity"] = (
        "Quality-Growth Portfolio"      if compounders >= cyclicals * 2 else
        "Cyclical-Value Portfolio"      if cyclicals >= compounders * 2 else
        "Hybrid (Quality + Cyclical)"   if compounders > 0 and cyclicals > 0 else
        "Turnaround / Special Situations"
    )

    return report
