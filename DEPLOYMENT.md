# SEC Financial Extractor - Deployment Guide

This folder contains a production-ready version of the SEC Financial Data Extractor Streamlit application.

## Quick Start

### Local Deployment

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

3. **Access the app:**
   Open your browser to http://localhost:8501

### Environment Variables

The app requires an OpenAI API key. You can provide it in one of two ways:

**Option 1: In-app configuration (recommended)**
- Enter your API key directly in the sidebar when running the app
- The key is stored in session state (not persisted)

**Option 2: Environment variable**
- Set `OPENAI_API_KEY` environment variable before running

### 🔒 Password Protection (Optional)

The app includes optional password authentication. To enable:

**Local:**
```bash
export APP_PASSWORD="your-secure-password"
streamlit run app.py
```

**Streamlit Cloud:**
Add to secrets:
```toml
APP_PASSWORD = "your-secure-password"
```

**To disable:** Simply don't set the `APP_PASSWORD` variable.

📖 See [AUTHENTICATION.md](AUTHENTICATION.md) for detailed setup instructions.

## Deployment to Cloud

### Streamlit Community Cloud

1. Push this folder to a GitHub repository
2. Go to https://share.streamlit.io/
3. Connect your GitHub repository
4. Select `app.py` as the main file
5. Add `OPENAI_API_KEY` in the secrets section
6. Click "Deploy"

### Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

Build and run:
```bash
docker build -t sec-extractor .
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key_here sec-extractor
```

### Heroku Deployment

1. Create `setup.sh`:
   ```bash
   mkdir -p ~/.streamlit/
   echo "[server]
   headless = true
   port = $PORT
   enableCORS = false
   " > ~/.streamlit/config.toml
   ```

2. Create `Procfile`:
   ```
   web: sh setup.sh && streamlit run app.py
   ```

3. Deploy:
   ```bash
   heroku create your-app-name
   heroku config:set OPENAI_API_KEY=your_key_here
   git push heroku main
   ```

## Features

- **Cash Position Analysis**: Extract and analyze cash position from 10-Q filings
- **Quarterly Burn Rate**: Calculate free cash flow and burn metrics
- **FDSO Analysis**: AI-powered analysis of fully diluted shares from 10-K filings
- **Interactive Visualizations**: Charts and graphs for FDSO vs stock price
- **Real-time Stock Price Fetching**: Automatic price fetching via Yahoo Finance

## Requirements

- Python 3.11+
- OpenAI API key (for AI-powered analysis)
- Internet connection (for SEC EDGAR and stock price data)

## File Structure

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── DEPLOYMENT.md         # This file
├── .gitignore            # Git ignore rules
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── lib/
│   ├── __init__.py
│   ├── sec_client.py     # SEC EDGAR API client
│   ├── extractor.py      # Financial data extraction
│   ├── cash_calculator.py # Cash position calculations
│   ├── burn_calculator.py # Burn rate calculations
│   ├── models.py         # Data models
│   └── fdso_ai_analyzer.py # AI-powered FDSO analysis
└── prompts/
    └── *.txt             # AI prompts for analysis
```

## Support

For issues or questions, refer to the main README.md file.
