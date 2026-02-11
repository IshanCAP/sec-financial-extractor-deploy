# SEC Financial Data Extractor

A Streamlit application that extracts cash position and quarterly burn rate from SEC 10-Q filings using GPT-4o.

## Features

### Cash Position Analysis (from most recent 10-Q)
- **Cash and Cash Equivalents** - From Balance Sheet
- **Marketable Securities** - From Balance Sheet
- **Total Cash at Hand** - Sum of both components
- Date-stamped with period end date

### Quarterly Burn Analysis (from 4 most recent 10-Qs)
- **Net Cash from Operating Activities** - From Cash Flow Statement
- **Capital Expenditure** - From Cash Flow Statement
- **Free Cash Flow** - OCF minus CapEx
- **Quarter Selection** - Select/deselect quarters for average calculation
- **Manual Override** - Override calculated average with custom value
- **Runway Analysis** - Quarters until cash depleted

### Data Quality Features
- 📊 **Source Links** - Direct links to SEC EDGAR filings
- 📝 **Evidence Snippets** - Exact text from filings showing extracted values
- 🏷️ **Row Labels** - The exact table row labels for each data point
- ✅ **Confidence Scores** - GPT-4o confidence in extraction accuracy

## Quick Start

### 1. Setup Environment

```powershell
cd sec-financial-extractor
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and add your keys:

```env
OPENAI_API_KEY=your-openai-api-key
SEC_USER_AGENT=YourApp your-email@example.com
```

### 3. Run the App

```powershell
streamlit run app.py
```

## Usage

1. **Enter Ticker** - Type a stock ticker (e.g., AAPL, TSLA, SANA)
2. **Find Company** - Click to search SEC EDGAR for 10-Q filings
3. **Extract Data** - Click to download and analyze filings with GPT-4o
4. **View Cash Position** - See total cash with evidence links
5. **Analyze Burn Rate** - Select quarters, calculate average, view runway

## Architecture

```
sec-financial-extractor/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variable template
├── README.md             # This file
├── lib/
│   ├── __init__.py       # Package exports
│   ├── models.py         # Data classes (Filing, CashPosition, etc.)
│   ├── sec_client.py     # SEC EDGAR API client with rate limiting
│   ├── extractor.py      # GPT-4o financial data extraction
│   ├── cash_calculator.py # Cash position calculation
│   └── burn_calculator.py # FCF and burn rate calculation
├── docs/
│   └── SEC_SEARCH_WORKFLOW.md  # Detailed documentation
└── tests/
    └── test_extraction.py      # Test suite
```

## API Rate Limiting

The SEC requires rate limiting to 10 requests per second. This is automatically handled by the `RateLimiter` class in `sec_client.py`.

## Extraction Logic

### Balance Sheet Items
- **Cash and Cash Equivalents**: Found in Current Assets section
- **Marketable Securities**: May also be labeled "Short-term investments"

### Cash Flow Items
- **Operating Cash Flow**: Total/subtotal of Operating Activities section
- **Capital Expenditure**: Found in Investing Activities, usually negative

### Unit Handling
GPT-4o automatically detects and handles:
- Numbers in thousands (multiplies by 1,000)
- Numbers in millions (multiplies by 1,000,000)
- Raw numbers (no multiplication)

## Testing

Test with these tickers:
- **AAPL** - Large cap, clean financials
- **TSLA** - High volume, complex structure
- **SANA** - Biotech, frequent filer
- **BLUE** - Biotech with negative cash flow

## Troubleshooting

### "No OPENAI_API_KEY found"
Add your OpenAI API key to `.env` or enter it in the sidebar.

### "Could not find company"
- Check ticker spelling
- Try the CIK number directly
- Some ADRs/foreign companies may not be in SEC database

### "Extraction failed"
- Check if filing is in standard format
- Some older or unusual filings may not parse correctly
- Try re-running extraction

### Rate limit errors
The app automatically rate-limits to 10 req/sec. If you still see errors, wait a moment and retry.

## License

MIT License
