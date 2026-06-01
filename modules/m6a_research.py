# =============================================================================
# MODULE 6A — RESEARCH DATA INGESTION ENGINE (M6.2)
# Concall PDF processing + auto news fetch per holding
# =============================================================================

import yfinance as yf
import warnings
import logging
from openai import OpenAI

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# =============================================================================
# CONCALL ANALYST PERSONA (from prompt_AI.pdf)
# =============================================================================
CONCALL_SYSTEM_PROMPT = """You are an experienced equity research analyst with 10+ years in Indian markets.
Your expertise: evaluating quarterly results and concall transcripts in a structured manner.
Extract ONLY what is explicitly stated in the transcript — do not infer or hallucinate.
Output strictly as JSON. No explanations outside the JSON."""

NEWS_SYSTEM_PROMPT = """You are a senior equity research analyst.
Extract investment-relevant signals from news headlines.
Focus on: earnings surprises, management changes, regulatory events, capex announcements, credit events.
Output strictly as JSON. No explanations outside the JSON."""

# =============================================================================
# CONCALL PDF PROCESSOR
# =============================================================================

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract raw text from a PDF file (bytes)."""
    try:
        import pypdf
        import io
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        return "\n\n".join(pages)
    except Exception as e:
        return f"PDF extraction failed: {e}"


def process_concall(
    pdf_bytes: bytes,
    company: str,
    ticker: str,
    client: OpenAI,
) -> dict:
    """
    Extract structured insights from a concall transcript PDF.
    Returns structured dict matching the M6.2 spec.
    """
    raw_text = extract_text_from_pdf(pdf_bytes)

    if not raw_text or "PDF extraction failed" in raw_text:
        return {"error": "PDF text extraction failed", "raw": raw_text}

    # Truncate to ~8000 chars to stay within token limits
    text_chunk = raw_text[:8000]

    prompt = f"""Analyse the following concall transcript for {company} ({ticker}).

TRANSCRIPT:
{text_chunk}

Extract and return ONLY the following as valid JSON:
{{
  "growth_guidance": "Management's stated revenue/volume/growth targets for next 1-2 years",
  "capex_plans": "Specific capex amounts, projects, or timelines mentioned",
  "margin_guidance": "Any margin targets or commentary on cost structure",
  "risk_acknowledgements": ["risk 1", "risk 2", "risk 3"],
  "management_tone": "CONFIDENT or CAUTIOUS or DEFENSIVE — based on language used",
  "promises_made": ["commitment 1", "commitment 2"],
  "key_metrics_guidance": {{
    "revenue_target": "...",
    "volume_target": "...",
    "margin_target": "..."
  }},
  "analyst_qna_signals": "Key demand/pricing/cost signals from analyst Q&A section",
  "red_flags": ["any concerning statements or evasions"]
}}

If a field is not mentioned in the transcript, use null."""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=800,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": CONCALL_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        import json
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"error": str(e), "raw_text_length": len(raw_text)}


# =============================================================================
# NEWS FETCHER
# =============================================================================

def fetch_news(ticker: str, max_articles: int = 5) -> list:
    """Fetch latest news for a ticker via yfinance (free, no API key)."""
    try:
        news = yf.Ticker(ticker).news
        articles = []
        for item in (news or [])[:max_articles]:
            articles.append({
                "title":     item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "link":      item.get("link", ""),
            })
        return articles
    except Exception:
        return []


def process_news(
    articles: list,
    company: str,
    ticker: str,
    client: OpenAI,
) -> dict:
    """GPT-4o mini extracts investment signals from news headlines."""
    if not articles:
        return {"sentiment": "NEUTRAL", "key_events": [], "macro_signals": []}

    headlines = "\n".join(
        f"- {a['title']} ({a['publisher']})"
        for a in articles
    )

    prompt = f"""Analyse these recent news headlines for {company} ({ticker}):

{headlines}

Return ONLY valid JSON:
{{
  "sentiment": "POSITIVE or NEGATIVE or NEUTRAL",
  "sentiment_rationale": "1 sentence explaining the sentiment",
  "key_events": ["event 1", "event 2"],
  "macro_signals": ["macro/sector signal 1"],
  "earnings_surprise": null,
  "regulatory_events": [],
  "management_changes": null,
  "capex_news": null
}}"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=400,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": NEWS_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        import json
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"sentiment": "NEUTRAL", "key_events": [], "error": str(e)}


# =============================================================================
# MAIN RESEARCH INGESTION RUNNER
# =============================================================================

def run_research_ingestion(
    holdings_df,
    concall_uploads: dict,        # {ticker: pdf_bytes_or_None}
    api_key: str,
    progress_callback=None,
) -> dict:
    """
    For every holding:
    1. Process concall PDF if uploaded
    2. Fetch + process news (always)

    Returns: {ticker: {concall: {...}|None, news: {...}, articles: [...]}}
    """
    client   = OpenAI(api_key=api_key)
    research = {}

    for _, row in holdings_df.iterrows():
        ticker  = row["Ticker"]
        company = row["Company"]

        if progress_callback:
            progress_callback(ticker, f"Fetching research for {company}...")

        # ── News (always) ─────────────────────────────────────────────────
        articles   = fetch_news(ticker)
        news_data  = process_news(articles, company, ticker, client)

        # ── Concall PDF (if uploaded) ─────────────────────────────────────
        concall_data = None
        pdf_bytes = concall_uploads.get(ticker)
        if pdf_bytes:
            if progress_callback:
                progress_callback(ticker, f"Processing concall PDF for {company}...")
            concall_data = process_concall(pdf_bytes, company, ticker, client)

        research[ticker] = {
            "concall":   concall_data,
            "news":      news_data,
            "articles":  articles,
            "has_concall": concall_data is not None and "error" not in concall_data,
        }

        if progress_callback:
            concall_status = "concall ✅" if research[ticker]["has_concall"] else "news only"
            progress_callback(ticker, f"{company}: {concall_status}")

    return research


# =============================================================================
# RESEARCH BRIEF FOR LLM CONTEXT
# =============================================================================

def research_to_text(research_entry: dict, company: str) -> str:
    """Convert a holding's research dict to concise text for LLM consumption."""
    lines = []

    concall = research_entry.get("concall")
    if concall and "error" not in concall:
        lines.append(f"CONCALL INSIGHTS ({company}):")
        if concall.get("growth_guidance"):
            lines.append(f"  Growth guidance: {concall['growth_guidance']}")
        if concall.get("margin_guidance"):
            lines.append(f"  Margin guidance: {concall['margin_guidance']}")
        if concall.get("capex_plans"):
            lines.append(f"  Capex: {concall['capex_plans']}")
        if concall.get("management_tone"):
            lines.append(f"  Management tone: {concall['management_tone']}")
        risks = concall.get("risk_acknowledgements", [])
        if risks:
            lines.append(f"  Risks acknowledged: {'; '.join(str(r) for r in risks[:3])}")
        red = concall.get("red_flags", [])
        if red:
            lines.append(f"  Red flags: {'; '.join(str(r) for r in red[:2])}")

    news = research_entry.get("news", {})
    if news:
        lines.append(f"NEWS SENTIMENT: {news.get('sentiment', 'NEUTRAL')}")
        if news.get("sentiment_rationale"):
            lines.append(f"  {news['sentiment_rationale']}")
        events = news.get("key_events", [])
        if events:
            lines.append(f"  Key events: {'; '.join(str(e) for e in events[:3])}")

    return "\n".join(lines) if lines else "No research data available."
