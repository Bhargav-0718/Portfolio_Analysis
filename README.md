# 📊 Portfolio Intelligence Platform

A Streamlit application for comprehensive Indian equity portfolio analysis with
GPT-4o mini-powered investment report generation.

---

## What It Does

| Module | What you get |
|--------|-------------|
| **1 — Portfolio Loader** | Live prices via yfinance, P&L, returns, portfolio weights vs benchmark |
| **2 — Sector Analysis** | Sector allocation vs Nifty 500, 5-year cumulative return, top/bottom performers |
| **3 — Attribution (BHB)** | Brinson-Hood-Beebower return attribution — allocation effect vs selection skill |
| **4 — Risk Analytics** | Active share, HHI, effective N, correlation heatmap, automated risk flags |
| **5 — Valuation & Peers** | Screener.in fundamental parsing, peer multiples, 5-factor composite scoring |
| **6 — AI Report** | GPT-4o mini writes per-holding investment theses + executive summary → PDF download |

---

## Setup

```bash
# 1. Navigate to the folder
cd portfolio_platform

# 2. Create & activate venv
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate      # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## Input Files

### Required
- **`portfolio.csv`** — your holdings file

  Columns: `No., Ticker, Company, Sector, Quantity, Avg Buy Price, MarketCap, Benchmark Wt%`

### Optional (enables Module 5 valuation)
- **Screener.in Excel exports** — one per holding  
  Download: screener.in → search company → **Excel** button (top right)  
  Supported: all 16 holdings. `Hindustan Copper.xlsx` is bundled for testing.

### Sidebar configuration
| Setting | Where |
|---------|-------|
| Analysis Date | Sidebar → date picker |
| Manual Prices | TATAMOTORS.NS + BBG.NS (yfinance broken for these) |
| OpenAI API Key | Sidebar → Module 6 section |
| Screener Excel files | Sidebar → expandable upload section |

---

## Demo Flow (5 minutes)

1. Upload `portfolio.csv` → click **▶ Run Full Analysis**
2. Watch all 6 modules execute with live progress
3. Browse tabs: **Overview → Sector → Attribution → Risk → Valuation**
4. Enter OpenAI API key → click **🤖 Generate AI Report**
5. Review executive summary + per-holding investment theses
6. Click **📄 Generate & Download PDF Report** → professional investment memo

---

## Architecture

```
portfolio_platform/
├── app.py                      # Main Streamlit app (6 tabs + sidebar)
├── requirements.txt
├── modules/
│   ├── m1_loader.py            # Portfolio loading + live price fetch (yfinance)
│   ├── m2_sector.py            # Sector allocation + 5yr cumulative return charts
│   ├── m3_attribution.py       # Brinson-Hood-Beebower attribution analysis
│   ├── m4_risk.py              # Risk metrics, correlation heatmap, flags
│   ├── m5a_data.py             # Screener.in Excel parser + fundamental extraction
│   ├── m5b_intelligence.py     # Peer multiples, 5-factor scoring, investment flags
│   └── m6_ai_report.py         # GPT-4o mini thesis + executive summary generation
└── utils/
    └── pdf_export.py           # ReportLab PDF assembly (charts + tables + theses)
```

---

## Screener.in Excel Format

The parser reads Screener's **Data Sheet** tab with fixed row boundaries:

| Section | Rows |
|---------|------|
| META (price, mcap) | 5–14 |
| Annual P&L | 16–39 |
| Balance Sheet | 55–79 |
| Cash Flow | 80–89 |
| Derived (adj. shares) | 92–93 |

---

## Business Model Awareness

The platform uses the right valuation metric per company type:

| Business Type | Primary Metric | Threshold (cheap / fair / exp) |
|---------------|---------------|-------------------------------|
| Commodity Metal (Cu/Zn/Steel) | EV/EBITDA | 5x / 8x / 12x |
| Auto OEM | EV/EBITDA | 6x / 10x / 16x |
| Auto Ancillary | PE | 15x / 25x / 40x |
| PSU Bank | P/B | 0.4x / 0.9x / 1.4x |
| Small Finance Bank | P/B | 0.8x / 1.8x / 3.0x |
| NBFC | P/B | 1.0x / 3.0x / 6.0x |
| Power IPP | EV/EBITDA | 5x / 9x / 14x |
| Exchange Platform | PE | 20x / 35x / 55x |
| Telecom Infra | EV/EBITDA | 5x / 9x / 13x |

---

## Cost Reference (Module 6 — GPT-4o mini)

| Run | Approx cost |
|-----|------------|
| Executive summary only | ~$0.001 |
| Full 16-stock portfolio | ~$0.02–0.04 |

Pricing: $0.15 / 1M input tokens · $0.60 / 1M output tokens

---

## Notes

- **yfinance failures**: TATAMOTORS.NS and BBG.NS require manual prices in sidebar
- **Correlation heatmap**: needs 2 years of price history (fetched live, ~30s)
- **Attribution**: uses 3-year lookback for sector benchmark returns
- **Notebook**: `portfolio_automation.ipynb` also has Module 6 as Cell 6  
  (runs in Colab — auto-installs `openai`, prompts for API key via `getpass`)

---

*Built from Sudhanva's `portfolio_automation.ipynb` — Modules 1–6 extracted and productionized*
