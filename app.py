"""
SEC Financial Data Extractor - Streamlit App
Single-page flow: Enter ticker → Extract → View Results
All configuration in-app (no .env required)
"""

import streamlit as st
import logging
import sys
import os
from datetime import datetime

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from lib.sec_client import SECClient
from lib.extractor import FinancialExtractor
from lib.cash_calculator import calculate_cash_position
from lib.burn_calculator import BurnCalculator, calculate_quarterly_from_ytd
from lib.models import format_currency
from lib.fdso_ai_analyzer import FDSOAIAnalyzer

# Try to import yfinance for stock price fetching
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed - stock price auto-fetch unavailable")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="SEC Financial Data Extractor",
    page_icon="📊",
    layout="wide"
)

# =============================================================================
# AUTHENTICATION
# =============================================================================
def check_password():
    """Returns True if the user has entered the correct password."""

    # Get password from environment variable or Streamlit secrets
    # For local: set APP_PASSWORD environment variable
    # For Streamlit Cloud: add APP_PASSWORD to secrets.toml
    correct_password = os.getenv("APP_PASSWORD") or st.secrets.get("APP_PASSWORD", None)

    # If no password is set, allow access (authentication disabled)
    if not correct_password:
        return True

    # Initialize authentication state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # If already authenticated, return True
    if st.session_state.authenticated:
        return True

    # Show login form
    st.markdown("# 🔒 SEC Financial Data Extractor")
    st.markdown("### Please enter the password to access the application")

    password = st.text_input("Password", type="password", key="password_input")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Login", use_container_width=True):
            if password == correct_password:
                st.session_state.authenticated = True
                st.success("✅ Authentication successful!")
                st.rerun()
            else:
                st.error("❌ Incorrect password")

    # Show info about password setup
    with st.expander("ℹ️ How to set up password protection"):
        st.markdown("""
        **For Local Development:**
        ```bash
        # Set environment variable before running
        export APP_PASSWORD="your-secure-password"
        streamlit run app.py
        ```

        **For Streamlit Cloud:**
        1. Go to your app settings
        2. Click "Secrets" tab
        3. Add:
        ```toml
        APP_PASSWORD = "your-secure-password"
        ```

        **To Disable Authentication:**
        - Simply don't set the `APP_PASSWORD` variable
        """)

    st.stop()  # Stop execution until authenticated

# Check authentication before showing the app
if not check_password():
    st.stop()

# Initialize session state
# Check for API key in environment or secrets
env_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

defaults = {
    "openai_api_key": env_api_key,
    "company_info": None,
    "filings": [],
    "ticker": "",
    "extraction_results": [],
    "cash_position": None,
    "quarterly_fcf": [],
    "burn_metrics": None,
    "extraction_complete": False,
    "manual_burn_override": None,
    # 10-K filing
    "tenk_filings": [],
    "fdso_analysis": None,  # AI analysis results
    "fdso_analysis_error": None,  # Analysis error message
    # Section toggles (all enabled by default)
    "process_cash_position": True,
    "process_quarterly_burn": True,
    "process_fdso": True,  # FDSO enabled by default
}

for key, default in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def fetch_stock_price(ticker: str) -> float:
    """
    Fetch current stock price using Yahoo Finance.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Current stock price (float) or None if unavailable
    """
    if not YFINANCE_AVAILABLE:
        logger.warning("yfinance not available for price fetching")
        return None

    try:
        logger.info(f"Attempting to fetch stock price for ticker: {ticker}")
        stock = yf.Ticker(ticker)

        # Try method 1: fast_info (fastest)
        try:
            info = stock.fast_info
            price = info.get('lastPrice') or info.get('last_price')
            if price and price > 0:
                logger.info(f"✓ Fetched stock price for {ticker} via fast_info: ${price:.2f}")
                return float(price)
        except Exception as e:
            logger.info(f"fast_info failed for {ticker}: {e}")

        # Try method 2: info dict
        try:
            info_dict = stock.info
            price = info_dict.get('currentPrice') or info_dict.get('regularMarketPrice')
            if price and price > 0:
                logger.info(f"✓ Fetched stock price for {ticker} via info: ${price:.2f}")
                return float(price)
        except Exception as e:
            logger.info(f"info dict failed for {ticker}: {e}")

        # Try method 3: history (most reliable but slower)
        try:
            hist = stock.history(period="1d")
            if not hist.empty:
                price = hist['Close'].iloc[-1]
                if price and price > 0:
                    logger.info(f"✓ Fetched stock price for {ticker} via history: ${price:.2f}")
                    return float(price)
        except Exception as e:
            logger.info(f"history failed for {ticker}: {e}")

        logger.warning(f"❌ No valid price found for {ticker} after all methods")
        return None

    except Exception as e:
        logger.error(f"❌ Error fetching stock price for {ticker}: {e}")
        return None

# =============================================================================
# SIDEBAR - Configuration
# =============================================================================
st.sidebar.header("⚙️ Configuration")

# API Key (stored in session state)
st.sidebar.markdown("**OpenAI API Key**")

# Check if API key is pre-configured in secrets/environment
preconfigured_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

if preconfigured_key:
    # API key is pre-configured - show status message
    st.sidebar.info("🔑 API key pre-configured")
    st.session_state.openai_api_key = preconfigured_key
else:
    # No pre-configured key - show input field
    api_key_input = st.sidebar.text_input(
        "API Key",
        value=st.session_state.openai_api_key,
        type="password",
        placeholder="sk-...",
        label_visibility="collapsed"
    )
    if api_key_input:
        st.session_state.openai_api_key = api_key_input
        st.sidebar.success("✅ API Key saved")
    else:
        st.sidebar.warning("⚠️ Enter your OpenAI API key above")

# Model selection for FDSO analysis
st.sidebar.markdown("**FDSO Analysis Model**")
model_options = ["gpt-5.2", "gpt-4o", "gpt-4-turbo", "gpt-4"]
selected_model = st.sidebar.selectbox(
    "Select AI Model",
    options=model_options,
    index=0,
    help="Choose the OpenAI model for FDSO analysis. gpt-5.2 uses URL-based analysis.",
    label_visibility="collapsed"
)
st.session_state.fdso_model = selected_model

st.sidebar.markdown("---")

# =============================================================================
# MAIN CONTENT
# =============================================================================
st.title("📊 SEC Financial Data Extractor")
st.markdown("Extract **Cash Position** and **Quarterly Burn** from SEC filings")

# -----------------------------------------------------------------------------
# STEP 1: Ticker Input
# -----------------------------------------------------------------------------
st.markdown("---")
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    ticker_input = st.text_input(
        "Enter Ticker Symbol",
        value=st.session_state.ticker,
        placeholder="AAPL, TSLA, SANA, MRNA...",
        key="ticker_input"
    ).upper().strip()

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_clicked = st.button("🔍 Extract Financial Data", type="primary", use_container_width=True)

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    reset_clicked = st.button("🔄 Reset", use_container_width=True)

# Section selection checkboxes
st.markdown("**Select sections to process:**")
section_col1, section_col2, section_col3 = st.columns(3)

with section_col1:
    process_cash = st.checkbox(
        "💰 Cash Position",
        value=st.session_state.process_cash_position,
        key="cb_cash_position"
    )
    st.session_state.process_cash_position = process_cash

with section_col2:
    process_burn = st.checkbox(
        "🔥 Quarterly Burn",
        value=st.session_state.process_quarterly_burn,
        key="cb_quarterly_burn"
    )
    st.session_state.process_quarterly_burn = process_burn

with section_col3:
    process_fdso = st.checkbox(
        "📈 Fully Diluted Shares (10-K)",
        value=st.session_state.process_fdso,
        key="cb_fdso"
    )
    st.session_state.process_fdso = process_fdso

if reset_clicked:
    # Clear all extraction results and state
    for key in ["company_info", "filings", "extraction_results", "cash_position", "quarterly_fcf", "burn_metrics", "extraction_complete", "tenk_filings", "fdso_sections", "fdso_instruments", "fdso_empty_sections"]:
        if key in defaults:
            st.session_state[key] = defaults[key]
    # Clear FDSO-related state
    for key in ["fdso_result", "fdso_auto_result", "current_stock_price", "auto_fetched_price", "auto_price_fetched", "fdso_calc_price", "fdso_needs_recalc", "fdso_analysis"]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.ticker = ""
    st.rerun()

# -----------------------------------------------------------------------------
# STEP 2: Search & Extract (Single Action)
# -----------------------------------------------------------------------------
if search_clicked and ticker_input:
    if not st.session_state.openai_api_key:
        st.error("⚠️ Please enter your OpenAI API key in the sidebar first")
    else:
        st.session_state.ticker = ticker_input

        # Reset previous results
        st.session_state.extraction_complete = False
        st.session_state.extraction_results = []
        st.session_state.cash_position = None
        st.session_state.quarterly_fcf = []
        st.session_state.burn_metrics = None
        st.session_state.tenk_filings = []

        # Reset FDSO-related state to fetch fresh data for new ticker
        for key in ["fdso_result", "fdso_auto_result", "current_stock_price", "auto_fetched_price", "auto_price_fetched", "fdso_calc_price", "fdso_needs_recalc", "fdso_analysis"]:
            if key in st.session_state:
                del st.session_state[key]

        # Check if any section is selected
        if not st.session_state.process_cash_position and not st.session_state.process_quarterly_burn and not st.session_state.process_fdso:
            st.warning("⚠️ Please select at least one section to process")
        else:
            sections_to_process = []
            if st.session_state.process_cash_position:
                sections_to_process.append("Cash Position")
            if st.session_state.process_quarterly_burn:
                sections_to_process.append("Quarterly Burn")
            if st.session_state.process_fdso:
                sections_to_process.append("Fully Diluted Shares")

            with st.status(f"Extracting {', '.join(sections_to_process)} for {ticker_input}...", expanded=True) as status:
                try:
                    # Initialize clients
                    sec_client = SECClient()

                    # =========================================================================
                    # CASH POSITION ONLY - Process only if selected
                    # =========================================================================
                    if st.session_state.process_cash_position and not st.session_state.process_quarterly_burn:
                        st.write("💰 Processing Cash Position (most recent 10-Q)...")
                        extractor = FinancialExtractor(api_key=st.session_state.openai_api_key)

                        # Get company info and most recent 10-Q only
                        company_info, filings_10q = sec_client.get_10q_filings(ticker=ticker_input, num_quarters=1)

                        if not company_info:
                            st.error(f"Could not find company for ticker: {ticker_input}")
                            status.update(label="Search failed", state="error")
                        else:
                            st.session_state.company_info = company_info
                            st.session_state.filings = filings_10q
                            st.write(f"✅ Found {company_info.get('name')}")

                            if not filings_10q:
                                st.warning("No 10-Q filings found for this company")
                                status.update(label="No filings found", state="error")
                            else:
                                filing = filings_10q[0]
                                st.write(f"📄 Processing {filing.form_type or '10-Q'} ({filing.period_of_report})...")

                                content = sec_client.download_filing(filing, max_chars=200000)

                                if content:
                                    result = extractor.extract_all(content)
                                    result["filing"] = filing
                                    st.session_state.extraction_results = [result]

                                    if result.get("success") and result.get("balance_sheet"):
                                        cash_pos = calculate_cash_position(result["balance_sheet"])
                                        st.session_state.cash_position = cash_pos
                                        st.write(f"   ✅ Cash position calculated: {cash_pos.total_formatted}")
                                    else:
                                        st.write(f"   ⚠️ Extraction issue: {result.get('notes', 'Unknown')}")
                                else:
                                    st.write(f"   ❌ Failed to download filing")

                    # =========================================================================
                    # QUARTERLY BURN ONLY - Process only if selected
                    # =========================================================================
                    elif st.session_state.process_quarterly_burn and not st.session_state.process_cash_position:
                        st.write("🔥 Processing Quarterly Burn (3-4 most recent filings)...")
                        extractor = FinancialExtractor(api_key=st.session_state.openai_api_key)
                        burn_calc = BurnCalculator()

                        # Get company info and filings (10-Q and 10-K for Q4)
                        company_info, filings_10q = sec_client.get_10q_filings(ticker=ticker_input, num_quarters=4)
                        _, filings_10k = sec_client.get_10k_filings(ticker=ticker_input, num_years=1)

                        if not company_info:
                            st.error(f"Could not find company for ticker: {ticker_input}")
                            status.update(label="Search failed", state="error")
                        else:
                            filings = filings_10q + filings_10k
                            st.session_state.company_info = company_info
                            st.session_state.filings = filings
                            filing_summary = f"{len(filings_10q)} 10-Q" + (f" + {len(filings_10k)} 10-K" if filings_10k else "")
                            st.write(f"✅ Found {company_info.get('name')} ({filing_summary} filings)")

                            if not filings:
                                st.warning("No filings found for this company")
                                status.update(label="No filings found", state="error")
                            else:
                                results = []
                                cash_flows_raw = []

                                for filing in filings:
                                    form_type = filing.form_type or "10-Q"
                                    st.write(f"📄 Processing {form_type} ({filing.period_of_report})...")

                                    max_chars = 300000 if form_type == "10-K" else 200000
                                    content = sec_client.download_filing(filing, max_chars=max_chars)

                                    if content:
                                        result = extractor.extract_all(content)
                                        result["filing"] = filing
                                        results.append(result)

                                        if result.get("success"):
                                            period_type = f"{result['cash_flow'].period_months}-month" if result.get("cash_flow") else ""
                                            st.write(f"   ✅ Extracted ({period_type}, confidence: {result.get('confidence', 'N/A')}/5)")

                                            if result.get("cash_flow"):
                                                cash_flows_raw.append(result["cash_flow"])
                                        else:
                                            st.write(f"   ⚠️ Extraction issue: {result.get('notes', 'Unknown')}")
                                    else:
                                        st.write(f"   ❌ Failed to download filing")
                                        results.append({"success": False, "error": "Download failed", "filing": filing})

                                st.session_state.extraction_results = results

                                # Convert YTD to quarterly
                                has_ytd = any(cf.period_months != 3 for cf in cash_flows_raw)
                                if has_ytd and len(cash_flows_raw) > 1:
                                    st.write("🔄 Converting YTD data to quarterly values...")
                                    quarterly_cash_flows = calculate_quarterly_from_ytd(cash_flows_raw)
                                    st.write(f"   ✅ Calculated {len(quarterly_cash_flows)} quarterly values")
                                else:
                                    quarterly_cash_flows = cash_flows_raw

                                # Create QuarterlyFCF objects
                                quarterly_fcf_list = []
                                for cf in quarterly_cash_flows:
                                    qfcf = burn_calc.create_quarterly_fcf(cf)
                                    if qfcf:
                                        quarterly_fcf_list.append(qfcf)

                                st.session_state.quarterly_fcf = quarterly_fcf_list

                                # Calculate burn metrics
                                if quarterly_fcf_list:
                                    burn_metrics = burn_calc.calculate_average_burn(
                                        quarterly_fcf_list,
                                        manual_override=st.session_state.manual_burn_override,
                                        cash_position=None
                                    )
                                    st.session_state.burn_metrics = burn_metrics
                                    st.write(f"   ✅ Quarterly burn calculated: {burn_metrics.burn_formatted}")

                    # =========================================================================
                    # BOTH CASH POSITION AND QUARTERLY BURN - Process together
                    # =========================================================================
                    elif st.session_state.process_cash_position and st.session_state.process_quarterly_burn:
                        st.write("💰🔥 Processing Cash Position + Quarterly Burn (3-4 most recent filings)...")
                        extractor = FinancialExtractor(api_key=st.session_state.openai_api_key)
                        burn_calc = BurnCalculator()

                        # Get company info and filings (10-Q and 10-K for Q4)
                        company_info, filings_10q = sec_client.get_10q_filings(ticker=ticker_input, num_quarters=4)
                        _, filings_10k = sec_client.get_10k_filings(ticker=ticker_input, num_years=1)

                        if not company_info:
                            st.error(f"Could not find company for ticker: {ticker_input}")
                            status.update(label="Search failed", state="error")
                        else:
                            filings = filings_10q + filings_10k
                            st.session_state.company_info = company_info
                            st.session_state.filings = filings
                            filing_summary = f"{len(filings_10q)} 10-Q" + (f" + {len(filings_10k)} 10-K" if filings_10k else "")
                            st.write(f"✅ Found {company_info.get('name')} ({filing_summary} filings)")

                            if not filings:
                                st.warning("No filings found for this company")
                                status.update(label="No filings found", state="error")
                            else:
                                results = []
                                cash_flows_raw = []

                                for filing in filings:
                                    form_type = filing.form_type or "10-Q"
                                    st.write(f"📄 Processing {form_type} ({filing.period_of_report})...")

                                    max_chars = 300000 if form_type == "10-K" else 200000
                                    content = sec_client.download_filing(filing, max_chars=max_chars)

                                    if content:
                                        result = extractor.extract_all(content)
                                        result["filing"] = filing
                                        results.append(result)

                                        if result.get("success"):
                                            period_type = f"{result['cash_flow'].period_months}-month" if result.get("cash_flow") else ""
                                            st.write(f"   ✅ Extracted ({period_type}, confidence: {result.get('confidence', 'N/A')}/5)")

                                            if result.get("cash_flow"):
                                                cash_flows_raw.append(result["cash_flow"])
                                        else:
                                            st.write(f"   ⚠️ Extraction issue: {result.get('notes', 'Unknown')}")
                                    else:
                                        st.write(f"   ❌ Failed to download filing")
                                        results.append({"success": False, "error": "Download failed", "filing": filing})

                                st.session_state.extraction_results = results

                                # Calculate cash position from most recent
                                if results and results[0].get("success") and results[0].get("balance_sheet"):
                                    cash_pos = calculate_cash_position(results[0]["balance_sheet"])
                                    st.session_state.cash_position = cash_pos
                                    st.write(f"   ✅ Cash position calculated: {cash_pos.total_formatted}")

                                # Convert YTD to quarterly
                                has_ytd = any(cf.period_months != 3 for cf in cash_flows_raw)
                                if has_ytd and len(cash_flows_raw) > 1:
                                    st.write("🔄 Converting YTD data to quarterly values...")
                                    quarterly_cash_flows = calculate_quarterly_from_ytd(cash_flows_raw)
                                    st.write(f"   ✅ Calculated {len(quarterly_cash_flows)} quarterly values")
                                else:
                                    quarterly_cash_flows = cash_flows_raw

                                # Create QuarterlyFCF objects
                                quarterly_fcf_list = []
                                for cf in quarterly_cash_flows:
                                    qfcf = burn_calc.create_quarterly_fcf(cf)
                                    if qfcf:
                                        quarterly_fcf_list.append(qfcf)

                                st.session_state.quarterly_fcf = quarterly_fcf_list

                                # Calculate burn metrics
                                if quarterly_fcf_list:
                                    burn_metrics = burn_calc.calculate_average_burn(
                                        quarterly_fcf_list,
                                        manual_override=st.session_state.manual_burn_override,
                                        cash_position=st.session_state.cash_position
                                    )
                                    st.session_state.burn_metrics = burn_metrics
                                    st.write(f"   ✅ Quarterly burn calculated: {burn_metrics.burn_formatted}")

                    # Process Fully Diluted Shares if selected (needs 10-K)
                    if st.session_state.process_fdso:
                        st.write("📈 Retrieving 10-K filing...")

                        # Get 10-K filing
                        _, tenk_filings = sec_client.get_10k_filings(ticker=ticker_input, num_years=1)

                        if tenk_filings:
                            st.session_state.tenk_filings = tenk_filings
                            filing = tenk_filings[0]
                            st.write(f"   ✅ Retrieved 10-K ({filing.period_of_report})")

                            # Download 10-K content (full document - no limit)
                            st.write("   📄 Downloading full 10-K content...")
                            filing_content_obj = sec_client.download_filing(filing, max_chars=None)

                            if filing_content_obj:
                                st.write(f"   ✅ Downloaded {filing_content_obj.char_count:,} characters")

                                # Run AI analysis with full content
                                st.write("   🤖 Analyzing 10-K with AI...")
                                fdso_analyzer = FDSOAIAnalyzer(
                                    api_key=st.session_state.openai_api_key,
                                    model=st.session_state.get("fdso_model", "gpt-5.2")
                                )

                                analysis_result = fdso_analyzer.analyze_10k(
                                    filing_content=filing_content_obj.plain_text,
                                    filing_url=filing.filing_url,
                                    ticker=ticker_input,
                                    filing_date=str(filing.filing_date) if filing.filing_date else None
                                )

                                if analysis_result.get("success"):
                                    st.session_state.fdso_analysis = analysis_result.get("data")
                                    st.session_state.fdso_analysis_error = None
                                    st.write("   ✅ FDSO analysis complete")

                                    # Show preview of what was found
                                    data = analysis_result.get("data", {})
                                    summary = data.get("summary", {})
                                    if summary.get("notes"):
                                        st.success(f"✅ Found dilutive securities! Scroll down to see full FDSO analysis.")
                                        st.info(f"📊 Preview: {summary['notes'][:200]}...")
                                else:
                                    st.session_state.fdso_analysis = None
                                    st.session_state.fdso_analysis_error = analysis_result.get("error")
                                    st.write(f"   ⚠️ FDSO analysis failed: {analysis_result.get('error')}")
                            else:
                                st.write("   ⚠️ Failed to download 10-K content")
                                st.session_state.fdso_analysis_error = "Failed to download 10-K"
                        else:
                            st.write("   ⚠️ No 10-K filings found")
                            st.session_state.fdso_analysis_error = "No 10-K filings found"

                    st.session_state.extraction_complete = True
                    status.update(label="Extraction complete!", state="complete")

                except Exception as e:
                    logger.error(f"Extraction error: {e}", exc_info=True)
                    st.error(f"Error: {e}")
                    status.update(label="Extraction failed", state="error")

# -----------------------------------------------------------------------------
# Company Header (Show once at top if any data is available)
# -----------------------------------------------------------------------------
if st.session_state.company_info:
    st.markdown("---")
    company_name = st.session_state.company_info.get("name", "Unknown")
    st.title(f"{company_name} ({st.session_state.ticker})")
    st.caption(f"CIK: {st.session_state.company_info.get('cik', 'N/A')}")

    # =========================================================================
    # SUMMARY SECTION - Key Metrics at a Glance
    # =========================================================================
    st.markdown("---")
    st.header("📊 Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    # Cash Position Summary
    with summary_col1:
        if st.session_state.get("cash_position"):
            cash_pos = st.session_state.cash_position
            net_value = cash_pos.net_cash_position
            if net_value >= 0:
                # Positive - Green
                st.markdown(f'<div style="text-align: left;"><p style="color: #0F9D58; font-size: 20px; font-weight: bold; margin: 0;">💰 Net Cash Position</p><p style="color: #0F9D58; font-size: 32px; font-weight: bold; margin: 0;">{cash_pos.net_cash_formatted}</p></div>', unsafe_allow_html=True)
            else:
                # Negative - Red
                st.markdown(f'<div style="text-align: left;"><p style="color: #DB4437; font-size: 20px; font-weight: bold; margin: 0;">💰 Net Cash Position</p><p style="color: #DB4437; font-size: 32px; font-weight: bold; margin: 0;">{cash_pos.net_cash_formatted}</p></div>', unsafe_allow_html=True)
        else:
            st.metric("💰 Net Cash Position", "N/A")

    # Burn Rate Summary
    with summary_col2:
        if st.session_state.get("burn_metrics"):
            metrics = st.session_state.burn_metrics
            avg_value = metrics.average_quarterly_burn
            if avg_value < 0:
                # Negative (burning cash) - Red
                st.markdown(f'<div style="text-align: left;"><p style="color: #DB4437; font-size: 20px; font-weight: bold; margin: 0;">🔥 Avg Quarterly Burn</p><p style="color: #DB4437; font-size: 32px; font-weight: bold; margin: 0;">{metrics.burn_formatted}</p></div>', unsafe_allow_html=True)
            else:
                # Positive (cash flow positive) - Green
                st.markdown(f'<div style="text-align: left;"><p style="color: #0F9D58; font-size: 20px; font-weight: bold; margin: 0;">🔥 Avg Quarterly FCF</p><p style="color: #0F9D58; font-size: 32px; font-weight: bold; margin: 0;">{metrics.burn_formatted}</p></div>', unsafe_allow_html=True)
        else:
            st.metric("🔥 Avg Quarterly Burn", "N/A")

    # FDSO Summary (using auto-fetched current price)
    with summary_col3:
        try:
            if hasattr(st.session_state, 'fdso_auto_result') and st.session_state.fdso_auto_result:
                result = st.session_state.fdso_auto_result
                st.metric(
                    "📈 FDSO",
                    f"{result['fdso']:,.0f}",
                    delta=f"{result['dilution_pct']:.2f}% dilution",
                    delta_color="inverse",
                    help=f"At current price ${result['price_used']:.2f}"
                )
            else:
                st.metric("📈 FDSO", "N/A", help="Calculate FDSO in section below")
        except Exception as e:
            st.metric("📈 FDSO", "Error", help=str(e))

# -----------------------------------------------------------------------------
# STEP 3: Display Results (All on one page)
# -----------------------------------------------------------------------------
if st.session_state.extraction_complete and st.session_state.company_info:

    # =========================================================================
    # CASH POSITION SECTION (only show if selected)
    # =========================================================================
    if st.session_state.process_cash_position:
        try:
            st.markdown("---")
            st.header("💰 Cash Position")

            if st.session_state.cash_position:
                cash_pos = st.session_state.cash_position

                # Helper to get evidence for a field
                def get_evidence_text(field_name):
                    for ev in cash_pos.evidence:
                        if ev.field_name == field_name and ev.source_text:
                            return ev.source_text
                    return None

                st.markdown(f"**As of {cash_pos.as_of_date}** (from most recent 10-Q)")

                # Net Cash Position - Show first, aligned right, with color
                if cash_pos.total_debt is not None:
                    net_col1, net_col2, net_col3 = st.columns([1, 1, 1])
                    with net_col3:
                        net_value = cash_pos.net_cash_position
                        if net_value >= 0:
                            # Positive - Green
                            st.markdown(f'<div style="text-align: right;"><p style="color: #0F9D58; font-size: 24px; font-weight: bold; margin: 0;">Net Cash Position</p><p style="color: #0F9D58; font-size: 36px; font-weight: bold; margin: 0;">{cash_pos.net_cash_formatted}</p><p style="font-size: 12px; color: #666;">Total Cash at Hand - Total Debt</p></div>', unsafe_allow_html=True)
                        else:
                            # Negative - Red
                            st.markdown(f'<div style="text-align: right;"><p style="color: #DB4437; font-size: 24px; font-weight: bold; margin: 0;">Net Cash Position</p><p style="color: #DB4437; font-size: 36px; font-weight: bold; margin: 0;">{cash_pos.net_cash_formatted}</p><p style="font-size: 12px; color: #666;">Total Cash at Hand - Total Debt</p></div>', unsafe_allow_html=True)
                    st.markdown("")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Cash & Equivalents", cash_pos.cash_formatted)
                    ev_text = get_evidence_text("cash_and_equivalents")
                    if ev_text:
                        st.markdown(f'<p style="font-size: 12px; color: #666; font-family: monospace; margin-top: 4px;">{ev_text}</p>', unsafe_allow_html=True)
                with col2:
                    st.metric("Marketable Securities", cash_pos.securities_formatted)
                    ev_text = get_evidence_text("marketable_securities")
                    if ev_text:
                        st.markdown(f'<p style="font-size: 12px; color: #666; font-family: monospace; margin-top: 4px;">{ev_text}</p>', unsafe_allow_html=True)
                with col3:
                    st.metric("**Total Cash at Hand**", cash_pos.total_formatted)

                # Debt - Always show this section
                st.markdown("")
                st.markdown("**Debt**")
                dcol1, dcol2, dcol3 = st.columns(3)
                with dcol1:
                    st.metric("Long-Term Debt", cash_pos.long_term_debt_formatted)
                    ev_text = get_evidence_text("long_term_debt")
                    if ev_text:
                        st.markdown(f'<p style="font-size: 12px; color: #666; font-family: monospace; margin-top: 4px;">{ev_text}</p>', unsafe_allow_html=True)
                with dcol2:
                    st.metric("Short-Term Debt", cash_pos.short_term_debt_formatted)
                    ev_text = get_evidence_text("short_term_debt")
                    if ev_text:
                        st.markdown(f'<p style="font-size: 12px; color: #666; font-family: monospace; margin-top: 4px;">{ev_text}</p>', unsafe_allow_html=True)
                with dcol3:
                    st.metric("Total Debt", cash_pos.total_debt_formatted)

                # Rerun button for cash position
                if st.button("🔄 Re-extract Cash Position", key="rerun_cash"):
                    with st.spinner("Re-extracting cash position..."):
                        try:
                            sec_client = SECClient()
                            extractor = FinancialExtractor(api_key=st.session_state.openai_api_key)
                            # Get most recent filing
                            if st.session_state.filings:
                                filing = st.session_state.filings[0]
                                content = sec_client.download_filing(filing, max_chars=200000)
                                if content:
                                    result = extractor.extract_all(content)
                                    if result.get("success") and result.get("balance_sheet"):
                                        cash_pos = calculate_cash_position(result["balance_sheet"])
                                        st.session_state.cash_position = cash_pos
                                        st.session_state.extraction_results[0] = result
                                        st.success("Cash position re-extracted!")
                                        st.rerun()
                        except Exception as e:
                            st.error(f"Re-extraction failed: {e}")

                # Evidence expander
                with st.expander("📝 Source Evidence"):
                    for ev in cash_pos.evidence:
                        if ev.source_text and ev.field_name != "total_cash_at_hand":
                            st.markdown(f"**{ev.field_name.replace('_', ' ').title()}**")
                            st.markdown(f"- Row: _{ev.source_text}_")
                            st.markdown(f"- Value: `{ev.table_label}`")
                            st.markdown(f"- [View Filing]({ev.filing_url})")
                            st.markdown("")
            else:
                st.warning("Cash position data not available")
        except Exception as e:
            st.error(f"Error displaying Cash Position section: {str(e)}")
            logger.error(f"Cash position display error: {e}", exc_info=True)

    # =========================================================================
    # QUARTERLY BURN SECTION (only show if selected)
    # =========================================================================
    if st.session_state.process_quarterly_burn:
        try:
            st.markdown("---")
            st.header("🔥 Quarterly Burn Rate")

            if st.session_state.quarterly_fcf:
                # Display burn metrics at the TOP
                if st.session_state.burn_metrics:
                    metrics = st.session_state.burn_metrics

                    # Burn Analysis - Top section with color-coded values
                    st.markdown("### Burn Analysis")

                    burn_col1, burn_col2, burn_col3 = st.columns([1, 1, 1])

                    # AVG Quarterly Cash Flow - Colored (left/center)
                    with burn_col1:
                        avg_value = metrics.average_quarterly_burn
                        if avg_value < 0:
                            # Negative (burning cash) - Red
                            st.markdown(f'<div style="text-align: left;"><p style="color: #DB4437; font-size: 20px; font-weight: bold; margin: 0;">Avg Quarterly Burn</p><p style="color: #DB4437; font-size: 32px; font-weight: bold; margin: 0;">{metrics.burn_formatted}</p></div>', unsafe_allow_html=True)
                        else:
                            # Positive (cash flow positive) - Green
                            st.markdown(f'<div style="text-align: left;"><p style="color: #0F9D58; font-size: 20px; font-weight: bold; margin: 0;">Avg Quarterly FCF</p><p style="color: #0F9D58; font-size: 32px; font-weight: bold; margin: 0;">{metrics.burn_formatted}</p></div>', unsafe_allow_html=True)

                        # Compact override form
                        with st.form(key="override_form", clear_on_submit=False, border=False):
                            ocol1, ocol2, ocol3 = st.columns([2, 1, 1])
                            with ocol1:
                                override_value = st.number_input(
                                    "Override",
                                    value=st.session_state.manual_burn_override if st.session_state.manual_burn_override is not None else 0.0,
                                    format="%.0f",
                                    label_visibility="collapsed"
                                )
                            with ocol2:
                                submitted = st.form_submit_button("Set", use_container_width=True)
                            with ocol3:
                                cleared = st.form_submit_button("Clear", use_container_width=True)

                            if submitted:
                                st.session_state.manual_burn_override = override_value
                                burn_calc = BurnCalculator()
                                new_metrics = burn_calc.calculate_average_burn(
                                    st.session_state.quarterly_fcf,
                                    manual_override=st.session_state.manual_burn_override,
                                    cash_position=st.session_state.cash_position
                                )
                                st.session_state.burn_metrics = new_metrics
                                st.rerun()

                            if cleared:
                                st.session_state.manual_burn_override = None
                                burn_calc = BurnCalculator()
                                new_metrics = burn_calc.calculate_average_burn(
                                    st.session_state.quarterly_fcf,
                                    manual_override=None,
                                    cash_position=st.session_state.cash_position
                                )
                                st.session_state.burn_metrics = new_metrics
                                st.rerun()

                        if metrics.is_manual_override:
                            st.caption("📝 Override active")

                    # Quarters in calculation (center)
                    with burn_col2:
                        st.metric("Quarters in Calculation", len(metrics.quarters_included))
                        st.caption(f"Quarters: {', '.join(metrics.quarters_included)}")

                    # Runway - Colored (right)
                    with burn_col3:
                        if metrics.runway_quarters is not None:
                            runway_value = metrics.runway_quarters
                            if runway_value == float('inf'):
                                st.markdown(f'<div style="text-align: right;"><p style="color: #0F9D58; font-size: 20px; font-weight: bold; margin: 0;">Runway</p><p style="color: #0F9D58; font-size: 32px; font-weight: bold; margin: 0;">∞</p><p style="font-size: 12px; color: #666;">Infinite (cash flow positive)</p></div>', unsafe_allow_html=True)
                            elif runway_value < 8:
                                # Less than 8 quarters - Red (warning)
                                st.markdown(f'<div style="text-align: right;"><p style="color: #DB4437; font-size: 20px; font-weight: bold; margin: 0;">Runway</p><p style="color: #DB4437; font-size: 32px; font-weight: bold; margin: 0;">{metrics.runway_formatted}</p>', unsafe_allow_html=True)
                                if metrics.runway_end_date:
                                    st.markdown(f'<p style="font-size: 12px; color: #DB4437; text-align: right;">Ends ~{metrics.runway_end_date.strftime("%B %Y")}</p></div>', unsafe_allow_html=True)
                                else:
                                    st.markdown('</div>', unsafe_allow_html=True)
                            else:
                                # 8+ quarters - Green (healthy)
                                st.markdown(f'<div style="text-align: right;"><p style="color: #0F9D58; font-size: 20px; font-weight: bold; margin: 0;">Runway</p><p style="color: #0F9D58; font-size: 32px; font-weight: bold; margin: 0;">{metrics.runway_formatted}</p>', unsafe_allow_html=True)
                                if metrics.runway_end_date:
                                    st.markdown(f'<p style="font-size: 12px; color: #0F9D58; text-align: right;">Extends to ~{metrics.runway_end_date.strftime("%B %Y")}</p></div>', unsafe_allow_html=True)
                                else:
                                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown("---")

                st.markdown("**Free Cash Flow by Quarter** (select quarters to include in average)")

                # Quarter selection with checkboxes
                selected_indices = []

                for i, qfcf in enumerate(st.session_state.quarterly_fcf):
                    col1, col2, col3, col4, col5, col6 = st.columns([0.5, 1.5, 1.5, 1.5, 1.5, 1.5])

                    with col1:
                        is_selected = st.checkbox(
                            "Include",
                            value=qfcf.is_selected,
                            key=f"q_select_{i}",
                            label_visibility="collapsed"
                        )
                        if is_selected:
                            selected_indices.append(i)
                            qfcf.is_selected = True
                        else:
                            qfcf.is_selected = False

                    col2.markdown(f"**{qfcf.quarter_label}**")

                    # Get evidence for this quarter
                    ocf_ev = None
                    capex_ev = None
                    for ev in qfcf.evidence:
                        if ev.field_name == "operating_cash_flow":
                            ocf_ev = ev.source_text
                        elif ev.field_name == "capital_expenditure":
                            capex_ev = ev.source_text

                    with col3:
                        if qfcf.operating_cash_flow is not None:
                            color = "green" if qfcf.operating_cash_flow >= 0 else "red"
                            st.markdown(f"OCF: :{color}[{qfcf.ocf_formatted}]")
                            if ocf_ev:
                                st.caption(f"_{ocf_ev}_")
                        else:
                            st.markdown("OCF: N/A")

                    with col4:
                        if qfcf.capital_expenditure is not None:
                            st.markdown(f"CapEx: {qfcf.capex_formatted}")
                            if capex_ev:
                                st.caption(f"_{capex_ev}_")
                        else:
                            st.markdown("CapEx: N/A")

                    with col5:
                        if qfcf.free_cash_flow is not None:
                            color = "green" if qfcf.free_cash_flow >= 0 else "red"
                            st.markdown(f"**FCF: :{color}[{qfcf.fcf_formatted}]**")
                        else:
                            st.markdown("FCF: N/A")

                    if qfcf.filing:
                        col6.markdown(f"[Source]({qfcf.filing.filing_url})")

                    # Show calculation methodology if available
                    if qfcf.extraction_notes:
                        st.caption(f"📝 _{qfcf.extraction_notes}_")

                st.markdown("")

                # Buttons row
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    # Recalculate button
                    if st.button("📊 Recalculate with Selected Quarters"):
                        burn_calc = BurnCalculator()
                        burn_metrics = burn_calc.calculate_average_burn(
                            st.session_state.quarterly_fcf,
                            selected_indices=selected_indices if selected_indices else None,
                            manual_override=st.session_state.manual_burn_override,
                            cash_position=st.session_state.cash_position
                        )
                        st.session_state.burn_metrics = burn_metrics
                        st.rerun()

                with btn_col2:
                    # Rerun extraction button
                    if st.button("🔄 Re-extract All Quarters", key="rerun_quarters"):
                        with st.spinner("Re-extracting quarterly data from all filings..."):
                            try:
                                sec_client = SECClient()
                                extractor = FinancialExtractor(api_key=st.session_state.openai_api_key)
                                burn_calc = BurnCalculator()

                                results = []
                                cash_flows_raw = []

                                for i, filing in enumerate(st.session_state.filings):
                                    form_type = filing.form_type or "10-Q"
                                    max_chars = 300000 if form_type == "10-K" else 200000
                                    content = sec_client.download_filing(filing, max_chars=max_chars)
                                    if content:
                                        result = extractor.extract_all(content)
                                        result["filing"] = filing
                                        results.append(result)

                                        if result.get("success") and result.get("cash_flow"):
                                            cash_flows_raw.append(result["cash_flow"])

                                # Convert YTD to quarterly
                                has_ytd = any(cf.period_months != 3 for cf in cash_flows_raw)
                                if has_ytd and len(cash_flows_raw) > 1:
                                    quarterly_cash_flows = calculate_quarterly_from_ytd(cash_flows_raw)
                                else:
                                    quarterly_cash_flows = cash_flows_raw

                                # Create QuarterlyFCF objects
                                quarterly_fcf_list = []
                                for cf in quarterly_cash_flows:
                                    qfcf = burn_calc.create_quarterly_fcf(cf)
                                    if qfcf:
                                        quarterly_fcf_list.append(qfcf)

                                st.session_state.extraction_results = results
                                st.session_state.quarterly_fcf = quarterly_fcf_list

                                # Recalculate burn metrics
                                if quarterly_fcf_list:
                                    burn_metrics = burn_calc.calculate_average_burn(
                                        quarterly_fcf_list,
                                        manual_override=st.session_state.manual_burn_override,
                                        cash_position=st.session_state.cash_position
                                    )
                                    st.session_state.burn_metrics = burn_metrics

                                st.success("Quarterly data re-extracted!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Re-extraction failed: {e}")

            else:
                st.warning("Quarterly FCF data not available")
        except Exception as e:
            st.error(f"Error displaying Quarterly Burn section: {str(e)}")
            logger.error(f"Burn rate display error: {e}", exc_info=True)

# =============================================================================
# FILING DETAILS (Expandable) - Show if any section was processed
# =============================================================================
if st.session_state.extraction_complete:
    st.markdown("---")
    with st.expander("📋 Filing Details & Extraction Results"):
        for result in st.session_state.extraction_results:
            filing = result.get("filing")
            st.markdown(f"### {filing.period_of_report} ({filing.filing_date})")
            st.markdown(f"[View on SEC EDGAR]({filing.filing_url})")

            if result.get("success"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Balance Sheet:**")
                    bs = result.get("balance_sheet")
                    if bs:
                        st.markdown(f"- Cash & Equivalents: {bs.cash_formatted}")
                        st.markdown(f"- Marketable Securities: {bs.securities_formatted}")

                with col2:
                    st.markdown("**Cash Flow:**")
                    cf = result.get("cash_flow")
                    if cf:
                        st.markdown(f"- Operating CF: {cf.ocf_formatted}")
                        st.markdown(f"- CapEx: {cf.capex_formatted}")
                        st.markdown(f"- FCF: {cf.fcf_formatted}")

                if result.get("notes"):
                    st.caption(f"Notes: {result['notes']}")
            else:
                st.error(f"Extraction failed: {result.get('error', 'Unknown')}")

            st.markdown("---")

# -----------------------------------------------------------------------------
# GETTING STARTED (shown when no data)
# -----------------------------------------------------------------------------
elif not st.session_state.extraction_complete:
    st.markdown("""
    ### How to Use

    1. **Enter your OpenAI API key** in the sidebar (required for GPT-4o-mini extraction)
    2. **Enter a ticker symbol** above (e.g., AAPL, TSLA, SANA, MRNA)
    3. **Click "Extract Financial Data"** - the app will automatically:
       - Find the company on SEC EDGAR
       - Download the most recent 10-Q filings
       - Extract cash position and FCF data using GPT-4o-mini
       - Calculate burn rate and runway

    ### What Gets Extracted

    **Cash Position** (from most recent 10-Q):
    - Cash and Cash Equivalents
    - Marketable Securities
    - Total = Cash at Hand

    **Quarterly Burn** (from 3 most recent 10-Qs):
    - Net Cash from Operating Activities
    - Capital Expenditure
    - Free Cash Flow = OCF - CapEx
    - Average quarterly burn (with quarter selection)
    - Cash runway projection

    **Fully Diluted Shares Outstanding (FDSO)** (from most recent 10-K):
    - AI-powered extraction of dilutive securities
    - Stock options, RSUs, PSUs, warrants, convertible debt
    - Treasury Stock Method (TSM) calculations
    - Interactive FDSO calculator at different stock prices
    - Visual graph showing dilution vs price
    - Automatic calculation at current market price

    ### Features
    - 🔗 Direct links to SEC filings
    - 📝 Evidence snippets showing source data
    - ✅ Select/deselect quarters for burn calculation
    - 🔧 Manual override option for burn rate
    - 📊 Interactive FDSO calculator and dilution analysis
    - 📈 Real-time stock price fetching via Yahoo Finance
    """)

# =============================================================================
# FDSO - AI Analysis Results (Independent Section)
# =============================================================================
if st.session_state.extraction_complete and st.session_state.process_fdso:
    st.markdown("---")
    try:
        st.header("📈 Fully Diluted Shares Outstanding (FDSO)")
        st.caption("AI-Powered Analysis from 10-K Filing")

        # Check if we have the 10-K filing
        if not st.session_state.tenk_filings:
            st.warning("⚠️ No 10-K filing found. Please run the extraction first.")
        else:
            filing = st.session_state.tenk_filings[0]

            # Debug: Check if analysis data exists
            st.caption(f"📊 Analysis status: {'✅ Results available' if st.session_state.fdso_analysis else '⚠️ No results'}")

            # Show analysis results if available
            if st.session_state.fdso_analysis:
                data = st.session_state.fdso_analysis

                # Header with filing link
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"[View 10-K Filing]({filing.filing_url})")
                    st.caption(f"Filing Date: {filing.filing_date} | Period: {filing.period_of_report}")
                with col2:
                    if data.get("summary", {}).get("confidence"):
                        confidence = data["summary"]["confidence"]
                        st.metric("Confidence", confidence)

                # Basic Shares Outstanding
                if data.get("basic_shares_outstanding"):
                    basic = data["basic_shares_outstanding"]
                    st.markdown("### Basic Shares Outstanding")
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric("Shares", f"{basic.get('shares') or 0:,}")
                    with col2:
                        st.caption(f"**As of:** {basic.get('as_of_date', 'N/A')}")
                        st.caption(f"**Source:** {basic.get('source', 'N/A')}")

                # ============================================================
                # DILUTIVE SECURITIES SUMMARY TABLE
                # ============================================================
                st.markdown("---")
                st.markdown("### Dilutive Securities Summary")

                # ============================================================
                # CURRENT PRICE & FDSO CALCULATOR
                # ============================================================
                st.markdown("#### Current Stock Price & FDSO Calculator")

                # Debug: Show ticker and yfinance status
                debug_ticker = st.session_state.get('ticker', 'NOT SET')
                debug_yf = "✅ Available" if YFINANCE_AVAILABLE else "❌ Not installed"
                with st.expander("🔍 Debug Info", expanded=False):
                    st.write(f"**Ticker:** {debug_ticker}")
                    st.write(f"**yfinance:** {debug_yf}")
                    st.write(f"**Auto-fetched price:** {st.session_state.get('auto_fetched_price', 'None')}")
                    st.write(f"**Current stock price:** {st.session_state.get('current_stock_price', 'None')}")
                    st.write(f"**Fetch attempted:** {st.session_state.get('auto_price_fetched', False)}")

                # Initialize default price in session state and auto-fetch
                if 'current_stock_price' not in st.session_state:
                    st.session_state.current_stock_price = None
                if 'auto_fetched_price' not in st.session_state:
                    st.session_state.auto_fetched_price = None

                # Auto-fetch stock price if not already set and ticker is available
                if 'auto_price_fetched' not in st.session_state:
                    st.session_state.auto_price_fetched = False

                # Always try to fetch when FDSO section first loads with a ticker
                ticker = st.session_state.get('ticker', None)
                if ticker and not st.session_state.auto_price_fetched:
                    with st.spinner(f"Fetching current stock price for {ticker}..."):
                        fetched_price = fetch_stock_price(ticker)
                        if fetched_price:
                            st.session_state.current_stock_price = fetched_price
                            st.session_state.auto_fetched_price = fetched_price
                            st.session_state.auto_price_fetched = True
                            # Auto-trigger calculation with fetched price
                            st.session_state.fdso_needs_recalc = True
                        else:
                            st.session_state.auto_fetched_price = None
                            st.session_state.auto_price_fetched = True

                # Price input form with results display
                input_left, results_right = st.columns([1, 1])

                with input_left:
                    with st.form(key="price_input_form", clear_on_submit=False):
                        form_col1, form_col2, form_col3 = st.columns([3, 1, 1])

                        with form_col1:
                            current_price = st.number_input(
                                "Stock Price ($)",
                                value=st.session_state.current_stock_price if st.session_state.current_stock_price else 0.0,
                                min_value=0.0,
                                step=0.01,
                                format="%.2f",
                                help="Enter current stock price to calculate FDSO"
                            )

                        with form_col2:
                            set_price = st.form_submit_button("Calculate", use_container_width=True)

                        with form_col3:
                            refresh_price = st.form_submit_button("🔄 Refresh", use_container_width=True)

                        if set_price and current_price > 0:
                            st.session_state.current_stock_price = current_price
                            st.session_state.fdso_needs_recalc = True  # Flag to trigger display update
                        elif refresh_price:
                            # Fetch fresh stock price from Yahoo Finance
                            if ticker:
                                fetched_price = fetch_stock_price(ticker)
                                if fetched_price:
                                    st.session_state.current_stock_price = fetched_price
                                    st.session_state.auto_fetched_price = fetched_price
                                    st.session_state.auto_price_fetched = True
                                    st.session_state.fdso_needs_recalc = True  # Recalculate with new price
                                    st.rerun()
                                else:
                                    st.error(f"Failed to fetch price for {ticker}")
                            else:
                                st.error("No ticker available to fetch price")

                    # Display fetched price info below the buttons
                    if st.session_state.auto_fetched_price:
                        st.caption(f"📊 Current Market Price: **${st.session_state.auto_fetched_price:.2f}** (auto-fetched from Yahoo Finance)")
                    elif not YFINANCE_AVAILABLE:
                        st.caption("⚠️ yfinance not installed. Run `pip install yfinance` to enable auto-fetch.")
                    else:
                        st.caption(f"⚠️ Could not auto-fetch stock price for {ticker if ticker else 'N/A'}. Click 🔄 Refresh to try again.")

                # Display FDSO results on the right if available
                with results_right:
                    if hasattr(st.session_state, 'fdso_result') and st.session_state.fdso_result:
                        result = st.session_state.fdso_result
                        # Display metrics side-by-side: FDSO on left, Dilution on right
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            st.metric("FDSO", f"{result['fdso']:,.0f} shares")
                        with metric_col2:
                            st.metric("Dilution", f"{result['dilution_pct']:.2f}%")
                        # Show calculation context
                        if 'price_used' in result and 'included_count' in result:
                            st.caption(f"At ${result['price_used']:.2f} | {result['included_count']} securities included")
                    elif st.session_state.current_stock_price and st.session_state.current_stock_price > 0:
                        st.info("Calculating FDSO...")
                    else:
                        st.info("Enter price to calculate FDSO")

                # Store price for calculation
                if st.session_state.current_stock_price and st.session_state.current_stock_price > 0:
                    st.session_state.fdso_calc_price = st.session_state.current_stock_price
                else:
                    st.session_state.fdso_calc_price = None

                st.markdown("---")

                # Function to determine if TSM should be used
                def should_use_tsm(security_type, exercise_price):
                    """
                    Determine if Treasury Stock Method should be used.

                    TSM is used for:
                    - Stock options with exercise price > $0.01
                    - Warrants with exercise price > $0.01
                    - SARs with exercise price > $0.01

                    TSM is NOT used for:
                    - RSUs (no exercise price)
                    - PSUs (no exercise price)
                    - Convertible debt (uses if-converted method)
                    - Convertible preferred (uses if-converted method)
                    - ESPP reserves
                    - Pre-funded warrants (exercise price ≤ $0.01)
                    """
                    # Securities that can use TSM if exercise price is not nominal
                    tsm_eligible = ["Stock Options", "Warrants", "SARs"]

                    if any(eligible in security_type for eligible in tsm_eligible):
                        # Check if exercise price is non-nominal
                        if exercise_price is not None and exercise_price > 0.01:
                            return "TRUE"

                    return "FALSE"

                # Build table data
                table_data = []

                # Stock Options - Vested
                if data.get("stock_options", {}).get("found"):
                    opts = data["stock_options"]
                    if opts.get("vested_exercisable"):
                        table_data.append({
                            "Security Type": "Stock Options",
                            "Amount": f"{opts.get('vested_exercisable') or 0:,}",
                            "Weighted Avg Cost": f"${opts.get('vested_waep'):.2f}" if opts.get('vested_waep') else "N/A",
                            "Schedule": "Vested",
                            "TSM": should_use_tsm("Stock Options", opts.get('vested_waep'))
                        })

                    # Stock Options - Unvested
                    if opts.get("unvested"):
                        table_data.append({
                            "Security Type": "Stock Options",
                            "Amount": f"{opts.get('unvested') or 0:,}",
                            "Weighted Avg Cost": f"${opts.get('unvested_waep'):.2f}" if opts.get('unvested_waep') else "N/A",
                            "Schedule": "Unvested",
                            "TSM": should_use_tsm("Stock Options", opts.get('unvested_waep'))
                        })

                # RSUs - Unvested (RSUs are always unvested until they vest and convert)
                if data.get("rsus", {}).get("found"):
                    rsus = data["rsus"]
                    if rsus.get("outstanding_unvested"):
                        table_data.append({
                            "Security Type": "RSUs",
                            "Amount": f"{rsus.get('outstanding_unvested') or 0:,}",
                            "Weighted Avg Cost": "N/A",
                            "Schedule": "Unvested",
                            "TSM": "FALSE"
                        })

                # PSUs - Target and Maximum
                if data.get("psus", {}).get("found"):
                    psus = data["psus"]
                    if psus.get("target_payout"):
                        table_data.append({
                            "Security Type": "PSUs (Target)",
                            "Amount": f"{psus.get('target_payout') or 0:,}",
                            "Weighted Avg Cost": "N/A",
                            "Schedule": "Unvested",
                            "TSM": "FALSE"
                        })
                    if psus.get("maximum_payout"):
                        table_data.append({
                            "Security Type": "PSUs (Maximum)",
                            "Amount": f"{psus.get('maximum_payout') or 0:,}",
                            "Weighted Avg Cost": "N/A",
                            "Schedule": "Unvested",
                            "TSM": "FALSE"
                        })

                # Warrants - Each tranche separately
                if data.get("warrants") and len(data["warrants"]) > 0:
                    for warrant in data["warrants"]:
                        exercise_price = warrant.get('exercise_price', 0)
                        warrant_desc = warrant.get('description', 'Warrant')

                        table_data.append({
                            "Security Type": warrant_desc,
                            "Amount": f"{warrant.get('shares') or 0:,}",
                            "Weighted Avg Cost": f"${exercise_price:.4f}" if exercise_price is not None else "N/A",
                            "Schedule": "Vested",  # Warrants are typically vested/exercisable
                            "TSM": should_use_tsm("Warrants", exercise_price)
                        })

                # Convertible Debt
                if data.get("convertible_debt") and len(data["convertible_debt"]) > 0:
                    for debt in data["convertible_debt"]:
                        table_data.append({
                            "Security Type": debt.get('description', 'Convertible Debt'),
                            "Amount": f"{debt.get('shares_issuable') or 0:,}",
                            "Weighted Avg Cost": f"${debt.get('conversion_price'):.2f}" if debt.get('conversion_price') else "N/A",
                            "Schedule": "Vested",
                            "TSM": "FALSE"  # Convertible debt uses if-converted method
                        })

                # ESPP
                if data.get("espp", {}).get("found"):
                    espp = data["espp"]
                    if espp.get("shares_available"):
                        table_data.append({
                            "Security Type": "ESPP Reserves",
                            "Amount": f"{espp.get('shares_available') or 0:,}",
                            "Weighted Avg Cost": "N/A",
                            "Schedule": "Unvested",
                            "TSM": "FALSE"
                        })

                # SARs
                if data.get("sars", {}).get("found"):
                    sars = data["sars"]
                    if sars.get("outstanding"):
                        table_data.append({
                            "Security Type": "SARs",
                            "Amount": f"{sars.get('outstanding') or 0:,}",
                            "Weighted Avg Cost": f"${sars.get('waep'):.2f}" if sars.get('waep') else "N/A",
                            "Schedule": "Vested",  # Assuming vested, adjust if data provides this
                            "TSM": should_use_tsm("SARs", sars.get('waep'))
                        })

                # Display table with checkboxes and conditional styling
                if table_data:
                    import pandas as pd

                    # Build dataframe with Exclude column
                    df = pd.DataFrame(table_data)
                    df.insert(0, "Exclude", False)

                    # Display editable table
                    edited_df = st.data_editor(
                        df,
                        column_config={
                            "Exclude": st.column_config.CheckboxColumn(
                                "Exclude",
                                help="Check to exclude this security from calculations",
                                default=False,
                            )
                        },
                        disabled=["Security Type", "Amount", "Weighted Avg Cost", "Schedule", "TSM"],
                        hide_index=True,
                        use_container_width=True,
                        key="fdso_securities_table"
                    )

                    # ============================================================
                    # FDSO CALCULATION AT SPECIFIC PRICE
                    # ============================================================
                    # Always recalculate when there's a price set, using current table state
                    if st.session_state.fdso_calc_price and st.session_state.fdso_calc_price > 0:
                        # Get BSO from data
                        calc_bso = data.get("basic_shares_outstanding", {}).get("shares", 0)

                        # Filter to included securities only (current state of Exclude checkboxes)
                        included_calc_df = edited_df[edited_df["Exclude"] == False].copy()

                        if len(included_calc_df) > 0 and calc_bso > 0:
                            import pandas as pd

                            # Parse amounts and prices
                            def parse_amount(amt_str):
                                if isinstance(amt_str, str):
                                    return int(amt_str.replace(',', ''))
                                return int(amt_str) if amt_str else 0

                            def parse_price(price_str):
                                if isinstance(price_str, str) and price_str != "N/A":
                                    return float(price_str.replace('$', '').replace(',', ''))
                                return None

                            included_calc_df['amount_num'] = included_calc_df['Amount'].apply(parse_amount)
                            included_calc_df['price_num'] = included_calc_df['Weighted Avg Cost'].apply(parse_price)

                            # Calculate FDSO at the specified price
                            def calculate_fdso_at_price(price, schedule_filter=None):
                                """
                                Calculate FDSO at a given price for specified schedule type.

                                Formula: Incremental Shares = N × MAX(Px - K, 0) / Px
                                Where:
                                - N = Number of underlying shares
                                - Px = Current stock price
                                - K = Strike/exercise price (0 for RSUs, PSUs, etc.)
                                - If Px ≤ K, then incremental shares = 0 (no dilution)
                                """
                                fdso = calc_bso
                                df_filter = included_calc_df if schedule_filter is None else included_calc_df[included_calc_df['Schedule'] == schedule_filter]

                                for _, row in df_filter.iterrows():
                                    amount = row['amount_num']
                                    strike = row['price_num']
                                    use_tsm = row['TSM'] == 'TRUE'

                                    # If TSM is FALSE or strike is N/A, add full amount (price-independent)
                                    if not use_tsm or strike is None:
                                        fdso += amount
                                    # If strike is very small (≈$0), treat as price-independent
                                    elif strike <= 0.01:
                                        fdso += amount
                                    # Otherwise apply TSM formula
                                    elif price > 0:
                                        incremental = amount * max(price - strike, 0) / price
                                        fdso += incremental

                                return fdso

                            # Calculate for each category using CURRENT table state
                            fdso_vested_calc = calculate_fdso_at_price(st.session_state.fdso_calc_price, 'Vested')
                            fdso_unvested_calc = calculate_fdso_at_price(st.session_state.fdso_calc_price, 'Unvested')
                            fdso_combined_calc = calculate_fdso_at_price(st.session_state.fdso_calc_price)

                            # Store results in session state for display above table
                            dilution_pct = ((fdso_combined_calc - calc_bso) / calc_bso * 100) if calc_bso > 0 else 0

                            # Store with timestamp to track when calculation was done
                            st.session_state.fdso_result = {
                                "fdso": fdso_combined_calc,
                                "bso": calc_bso,
                                "dilution_pct": dilution_pct,
                                "fdso_vested": fdso_vested_calc,
                                "fdso_unvested": fdso_unvested_calc,
                                "price_used": st.session_state.fdso_calc_price,
                                "included_count": len(included_calc_df)
                            }

                            # Also calculate FDSO at auto-fetched price for summary display
                            if st.session_state.get('auto_fetched_price') and st.session_state.auto_fetched_price > 0:
                                fdso_auto_calc = calculate_fdso_at_price(st.session_state.auto_fetched_price)
                                dilution_auto_pct = ((fdso_auto_calc - calc_bso) / calc_bso * 100) if calc_bso > 0 else 0
                                st.session_state.fdso_auto_result = {
                                    "fdso": fdso_auto_calc,
                                    "bso": calc_bso,
                                    "dilution_pct": dilution_auto_pct,
                                    "price_used": st.session_state.auto_fetched_price,
                                    "included_count": len(included_calc_df)
                                }
                            else:
                                # No auto-fetched price, use manual result for summary too
                                st.session_state.fdso_auto_result = st.session_state.fdso_result

                            # If Calculate button was just clicked, rerun to show results immediately
                            if st.session_state.get('fdso_needs_recalc', False):
                                st.session_state.fdso_needs_recalc = False
                                st.rerun()

                        else:
                            st.session_state.fdso_result = None
                    else:
                        # No price set, clear results
                        if not st.session_state.get('fdso_calc_price'):
                            st.session_state.fdso_result = None

                    # ============================================================
                    # FDSO vs STOCK PRICE GRAPH
                    # ============================================================
                    st.markdown("---")
                    st.markdown("### FDSO vs Stock Price")

                    # Get basic shares outstanding
                    bso = data.get("basic_shares_outstanding", {}).get("shares", 0)

                    if bso and len(edited_df) > 0:
                        import numpy as np
                        import plotly.graph_objects as go

                        # Filter to included securities only
                        included_df = edited_df[edited_df["Exclude"] == False].copy()

                        if len(included_df) > 0:
                            # Parse amounts (remove commas)
                            def parse_amount(amt_str):
                                if isinstance(amt_str, str):
                                    return int(amt_str.replace(',', ''))
                                return int(amt_str) if amt_str else 0

                            # Parse prices (remove $ and commas)
                            def parse_price(price_str):
                                if isinstance(price_str, str) and price_str != "N/A":
                                    return float(price_str.replace('$', '').replace(',', ''))
                                return None

                            included_df['amount_num'] = included_df['Amount'].apply(parse_amount)
                            included_df['price_num'] = included_df['Weighted Avg Cost'].apply(parse_price)

                            # Smart x-axis range: extend beyond max strike to show TSM curve flattening
                            # Find all valid strike prices
                            valid_strikes = included_df[included_df['price_num'].notna()]['price_num']

                            if len(valid_strikes) > 0:
                                max_strike = valid_strikes.max()
                                min_strike = valid_strikes.min()
                                # Extend to 2.5× max strike to show asymptotic behavior
                                x_max = max_strike * 2.5
                                # Start from 10% of min strike or $1, whichever is lower
                                x_min = min(1.0, min_strike * 0.1)
                            else:
                                # No TSM securities, use default range
                                x_min = 1.0
                                x_max = 100.0
                                max_strike = 50.0

                            # Create price range with denser sampling near strikes
                            prices = np.linspace(x_min, x_max, 150)

                            def calculate_fdso(price, schedule_filter=None):
                                """
                                Calculate FDSO at a given price for specified schedule type.

                                Formula: Incremental Shares = N × MAX(Px - K, 0) / Px
                                Where:
                                - N = Number of underlying shares
                                - Px = Current stock price
                                - K = Strike/exercise price (0 for RSUs, PSUs, etc.)
                                - If Px ≤ K, then incremental shares = 0 (no dilution)
                                """
                                fdso = bso
                                df_filter = included_df if schedule_filter is None else included_df[included_df['Schedule'] == schedule_filter]

                                for _, row in df_filter.iterrows():
                                    amount = row['amount_num']
                                    strike = row['price_num']
                                    use_tsm = row['TSM'] == 'TRUE'

                                    # If TSM is FALSE or strike is N/A, add full amount (price-independent)
                                    if not use_tsm or strike is None:
                                        fdso += amount
                                    # If strike is very small (≈$0), treat as price-independent
                                    elif strike <= 0.01:
                                        fdso += amount
                                    # Otherwise apply TSM formula
                                    elif price > 0:
                                        incremental = amount * max(price - strike, 0) / price
                                        fdso += incremental

                                return fdso

                            # VALIDATION: Check schedule distribution
                            vested_count = len(included_df[included_df['Schedule'] == 'Vested'])
                            unvested_count = len(included_df[included_df['Schedule'] == 'Unvested'])
                            total_included = len(included_df)

                            # Check for any securities with unexpected Schedule values
                            unique_schedules = included_df['Schedule'].unique()
                            unexpected_schedules = [s for s in unique_schedules if s not in ['Vested', 'Unvested']]

                            # Calculate total shares in each category
                            vested_shares = included_df[included_df['Schedule'] == 'Vested']['amount_num'].sum()
                            unvested_shares = included_df[included_df['Schedule'] == 'Unvested']['amount_num'].sum()
                            total_shares = included_df['amount_num'].sum()

                            # Calculate FDSO curves
                            fdso_vested = [calculate_fdso(p, 'Vested') for p in prices]
                            fdso_unvested = [calculate_fdso(p, 'Unvested') for p in prices]
                            fdso_combined = [calculate_fdso(p) for p in prices]

                            # Floor = BSO only (no price-independent additions)
                            # Vested options still use TSM and only contribute when in-the-money
                            floor = bso

                            # Calculate price-independent dilution (for informational purposes)
                            price_independent = sum(
                                row['amount_num'] for _, row in included_df.iterrows()
                                if row['TSM'] == 'FALSE' or row['price_num'] is None or row['price_num'] <= 0.01
                            )

                            # Smart y-axis range: find min/max across all curves
                            all_fdso_values = fdso_vested + fdso_unvested + fdso_combined
                            y_min = min(all_fdso_values)
                            y_max = max(all_fdso_values)
                            y_range = y_max - y_min
                            # Add 10% padding, but ensure y_axis_min doesn't go below BSO
                            y_axis_min = max(bso * 0.95, y_min - (y_range * 0.1))
                            y_axis_max = y_max + (y_range * 0.1)

                            # Create plot
                            fig = go.Figure()

                            # Add FDSO curves
                            fig.add_trace(go.Scatter(
                                x=prices, y=fdso_vested,
                                mode='lines',
                                name='Vested Only',
                                line=dict(color='#0F9D58', width=2)
                            ))

                            fig.add_trace(go.Scatter(
                                x=prices, y=fdso_unvested,
                                mode='lines',
                                name='Unvested Only',
                                line=dict(color='#F4B400', width=2)
                            ))

                            fig.add_trace(go.Scatter(
                                x=prices, y=fdso_combined,
                                mode='lines',
                                name='Vested + Unvested',
                                line=dict(color='#DB4437', width=3)
                            ))

                            # Add floor reference line
                            fig.add_hline(
                                y=floor,
                                line_dash="dash",
                                line_color="gray",
                                annotation_text=f"Floor: {floor:,.0f}",
                                annotation_position="right"
                            )

                            # Add vertical lines at key strike prices
                            strike_prices = included_df[included_df['price_num'].notna()]['price_num'].unique()
                            for strike in sorted(strike_prices)[:3]:  # Show up to 3 key strikes
                                fig.add_vline(
                                    x=strike,
                                    line_dash="dot",
                                    line_color="lightgray",
                                    opacity=0.5,
                                    annotation_text=f"${strike:.2f}",
                                    annotation_position="top"
                                )

                            # Update layout with smart axis ranges
                            fig.update_layout(
                                title="Fully Diluted Shares Outstanding vs Stock Price",
                                xaxis_title="Stock Price ($)",
                                yaxis_title="Fully Diluted Shares Outstanding",
                                hovermode='x unified',
                                template='plotly_white',
                                height=500,
                                xaxis=dict(range=[x_min, x_max]),
                                yaxis=dict(range=[y_axis_min, y_axis_max])
                            )

                            # Format axes
                            fig.update_yaxes(tickformat=',')
                            fig.update_xaxes(tickformat='$,.0f')

                            # Display plot
                            st.plotly_chart(fig, use_container_width=True)

                            # Show calculation details
                            with st.expander("📊 Calculation Details & Formulas"):
                                st.markdown("### Summary")
                                st.markdown(f"**Basic Shares Outstanding (BSO):** {bso:,}")
                                st.markdown(f"**Floor (BSO only):** {floor:,}")
                                st.markdown(f"**Price-Independent Securities:** {price_independent:,}")
                                st.markdown(f"**Included Securities:** {len(included_df)} of {len(edited_df)}")
                                st.markdown(f"**Price Range:** ${x_min:.2f} - ${x_max:.2f}")
                                if len(valid_strikes) > 0:
                                    st.markdown(f"**Strike Price Range:** ${min_strike:.2f} - ${max_strike:.2f}")

                                st.markdown("---")
                                st.markdown("### Schedule Breakdown")
                                st.markdown(f"**Vested Securities:** {vested_count} ({vested_shares:,} shares)")
                                st.markdown(f"**Unvested Securities:** {unvested_count} ({unvested_shares:,} shares)")
                                st.markdown(f"**Total:** {total_included} ({total_shares:,} shares)")

                                # Show validation check
                                if vested_count + unvested_count == total_included:
                                    st.success("✓ All securities properly categorized as Vested or Unvested")
                                else:
                                    st.warning(f"⚠ {total_included - vested_count - unvested_count} securities with unexpected Schedule values")

                                if unexpected_schedules:
                                    st.error(f"Unexpected Schedule values found: {', '.join(unexpected_schedules)}")
                                    st.markdown("**Securities with unexpected schedules:**")
                                    unexpected_df = included_df[~included_df['Schedule'].isin(['Vested', 'Unvested'])]
                                    st.dataframe(unexpected_df[['Security Type', 'Amount', 'Schedule']])

                                # Sample calculation at a test price
                                if len(valid_strikes) > 0:
                                    test_price = max_strike * 1.5
                                else:
                                    test_price = 50.0

                                st.markdown("---")
                                st.markdown(f"### Sample Calculation at ${test_price:.2f}")

                                sample_calc_data = []
                                total_incremental = 0
                                for idx, row in included_df.iterrows():
                                    amount = row['amount_num']
                                    strike = row['price_num']
                                    use_tsm = row['TSM'] == 'TRUE'

                                    # Calculate incremental shares using same logic as main calculation
                                    if not use_tsm or strike is None:
                                        # Price-independent: add full amount
                                        incremental = amount
                                        method = "Direct Add (TSM=FALSE or N/A)"
                                        in_money = "N/A"
                                    elif strike <= 0.01:
                                        # Very small strike: treat as price-independent
                                        incremental = amount
                                        method = "Direct Add (K≈$0)"
                                        in_money = "N/A"
                                    elif test_price > 0:
                                        # Apply TSM formula
                                        incremental = amount * max(test_price - strike, 0) / test_price
                                        method = "TSM"
                                        in_money = "Yes" if test_price > strike else "No"
                                    else:
                                        incremental = 0
                                        method = "TSM"
                                        in_money = "No"

                                    total_incremental += incremental

                                    sample_calc_data.append({
                                        "Security": row['Security Type'],
                                        "Schedule": row['Schedule'],
                                        "Shares (N)": f"{amount:,}",
                                        "Strike (K)": f"${strike:.4f}" if strike and strike > 0 else "N/A",
                                        "Method": method,
                                        "In the Money?": in_money,
                                        "Incremental Shares": f"{incremental:,.0f}"
                                    })

                                import pandas as pd
                                sample_df = pd.DataFrame(sample_calc_data)
                                st.dataframe(sample_df, use_container_width=True)

                                st.markdown(f"**FDSO at ${test_price:.2f}:** {bso:,} (BSO) + {total_incremental:,.0f} (incremental) = **{bso + total_incremental:,.0f}**")

                                st.markdown("---")
                                st.markdown("### FDSO Calculation Formula")
                                st.markdown("""
**Universal Formula Applied to ALL Securities:**

`FDSO(Px) = BSO + Σ(N × MAX(Px - K, 0) / Px)`

Where the sum is over all non-excluded securities.
                                """)

                                st.markdown("---")
                                st.markdown("### Treasury Stock Method (TSM)")
                                st.markdown("""
**Formula:** `Incremental Shares = N × MAX(Px - K, 0) / Px`

Where:
- **N** = Number of underlying shares
- **Px** = Current stock price
- **K** = Strike/exercise price (or 0 for RSUs, PSUs)
- **MAX(Px - K, 0)** = Zero if out-of-the-money

**Key Rule:** If Px ≤ K, then incremental shares = 0 (no dilution)

**For K = 0** (RSUs, PSUs, pre-funded warrants):
- Formula becomes: N × Px / Px = N (always fully dilutive)
                                """)

                                st.markdown("---")
                                st.markdown("### How Securities Behave")

                                st.markdown("**1. Stock Options (Vested & Unvested)**")
                                st.markdown("- **Strike:** Real exercise price (e.g., $10, $25)")
                                st.markdown("- **Dilution:** Only when Px > Strike Price")
                                st.markdown("- **Example:** Option at $10 strike contributes 0 below $10, partial shares above $10")

                                st.markdown("**2. Warrants (with real strikes)**")
                                st.markdown("- **Strike:** Real exercise price")
                                st.markdown("- **Dilution:** Only when Px > Strike Price")
                                st.markdown("- **Example:** Warrant with $20 strike contributes zero when price is $15")

                                st.markdown("**3. RSUs (Restricted Stock Units)**")
                                st.markdown("- **Strike:** K = 0 (no exercise price)")
                                st.markdown("- **Dilution:** N × (Px - 0) / Px = N (always fully dilutive)")
                                st.markdown("- **Result:** Full share count added at all prices")

                                st.markdown("**4. PSUs (Performance Stock Units)**")
                                st.markdown("- **Strike:** K = 0 (no exercise price)")
                                st.markdown("- **Dilution:** Always fully dilutive (target or maximum payout)")
                                st.markdown("- **Result:** Full share count added at all prices")

                                st.markdown("**5. Pre-Funded Warrants**")
                                st.markdown("- **Strike:** K ≈ $0.0001 (treated as 0)")
                                st.markdown("- **Dilution:** N × (Px - 0) / Px = N (always fully dilutive)")
                                st.markdown("- **Result:** Full share count added at all prices")

                                st.markdown("**6. Convertible Debt**")
                                st.markdown("- **Strike:** Conversion price (or treated as K = 0)")
                                st.markdown("- **Dilution:** Shares issuable upon conversion")

                                st.markdown("**7. ESPP Reserves**")
                                st.markdown("- **Strike:** K = 0 (future issuance)")
                                st.markdown("- **Dilution:** Full share count added at all prices")

                                st.markdown("---")
                                st.markdown("### Important Notes")
                                st.markdown("""
- **Floor = BSO only** - No dilutive securities included
- **All securities use the same formula** - Difference is only in the strike price K
- **Vested vs Unvested:** Only determines which curve, not the calculation method
- **Excluded securities:** Rows with "Exclude" checked are not included in calculations
- **When K = 0:** Security is always fully dilutive regardless of price
                                """)
                        else:
                            st.info("No securities included. Uncheck 'Exclude' boxes to see the graph.")
                    else:
                        st.info("Basic shares outstanding data not available for graphing.")

                else:
                    st.info("No dilutive securities found to display in summary table.")

                st.markdown("---")
                st.markdown("### Dilutive Securities")

                # Stock Options
                if data.get("stock_options", {}).get("found"):
                    with st.expander("📊 Stock Options", expanded=True):
                        opts = data["stock_options"]

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Outstanding", f"{opts.get('total_outstanding') or 0:,}")
                        with col2:
                            st.metric("Vested/Exercisable", f"{opts.get('vested_exercisable') or 0:,}")
                            if opts.get('vested_waep'):
                                st.caption(f"WAEP: ${opts.get('vested_waep'):.2f}")
                        with col3:
                            st.metric("Unvested", f"{opts.get('unvested') or 0:,}")
                            if opts.get('unvested_waep'):
                                st.caption(f"WAEP: ${opts.get('unvested_waep'):.2f}")

                        st.markdown("**Source:** " + opts.get("source", "N/A"))
                        if opts.get("evidence"):
                            st.markdown("**Evidence:**")
                            st.info(opts["evidence"])
                        if opts.get("vesting_schedule"):
                            st.caption(f"Vesting: {opts['vesting_schedule']}")

                # RSUs
                if data.get("rsus", {}).get("found"):
                    with st.expander("🎁 Restricted Stock Units (RSUs)", expanded=True):
                        rsus = data["rsus"]

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Outstanding Unvested", f"{rsus.get('outstanding_unvested') or 0:,}")
                        with col2:
                            st.metric("Expected to Vest", f"{rsus.get('expected_to_vest') or 0:,}")

                        st.markdown("**Source:** " + rsus.get("source", "N/A"))
                        if rsus.get("evidence"):
                            st.markdown("**Evidence:**")
                            st.info(rsus["evidence"])

                # PSUs
                if data.get("psus", {}).get("found"):
                    with st.expander("🎯 Performance Stock Units (PSUs)", expanded=True):
                        psus = data["psus"]

                        col1, col2 = st.columns(2)
                        with col1:
                            target = psus.get('target_payout') or 0
                            st.metric("Target Payout", f"{target:,}")
                        with col2:
                            maximum = psus.get('maximum_payout')
                            st.metric("Maximum Payout", f"{maximum:,}" if maximum else "N/A")

                        st.markdown("**Source:** " + psus.get("source", "N/A"))
                        if psus.get("evidence"):
                            st.markdown("**Evidence:**")
                            st.info(psus["evidence"])

                # Warrants
                if data.get("warrants") and len(data["warrants"]) > 0:
                    with st.expander("⚡ Warrants", expanded=True):
                        for i, warrant in enumerate(data["warrants"]):
                            st.markdown(f"**{warrant.get('description', f'Warrant {i+1}')}**")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Shares", f"{warrant.get('shares') or 0:,}")
                            with col2:
                                st.metric("Exercise Price", f"${warrant.get('exercise_price') or 0:.4f}")
                            with col3:
                                exp = warrant.get('expiration', 'N/A')
                                st.caption(f"**Expiration:** {exp}")

                            st.caption(f"**Source:** {warrant.get('source', 'N/A')}")
                            if warrant.get("evidence"):
                                st.info(warrant["evidence"])

                            if i < len(data["warrants"]) - 1:
                                st.markdown("---")

                # Convertible Debt
                if data.get("convertible_debt") and len(data["convertible_debt"]) > 0:
                    with st.expander("💰 Convertible Debt", expanded=True):
                        for i, debt in enumerate(data["convertible_debt"]):
                            st.markdown(f"**{debt.get('description', f'Convertible Note {i+1}')}**")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                principal = debt.get('principal_amount') or 0
                                st.metric("Principal", f"${principal:,}")
                            with col2:
                                st.metric("Conversion Price", f"${debt.get('conversion_price') or 0:.2f}")
                            with col3:
                                st.metric("Shares Issuable", f"{debt.get('shares_issuable') or 0:,}")

                            st.caption(f"**Source:** {debt.get('source', 'N/A')}")
                            if debt.get("evidence"):
                                st.info(debt["evidence"])

                # ESPP
                if data.get("espp", {}).get("found"):
                    with st.expander("👥 Employee Stock Purchase Plan (ESPP)"):
                        espp = data["espp"]

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Shares Reserved", f"{espp.get('shares_reserved') or 0:,}")
                        with col2:
                            st.metric("Shares Available", f"{espp.get('shares_available') or 0:,}")

                        st.markdown("**Source:** " + espp.get("source", "N/A"))
                        if espp.get("evidence"):
                            st.info(espp["evidence"])

                # SARs
                if data.get("sars", {}).get("found"):
                    with st.expander("📈 Stock Appreciation Rights (SARs)"):
                        sars = data["sars"]
                        st.metric("Outstanding", f"{sars.get('outstanding') or 0:,}")
                        if sars.get('waep'):
                            st.caption(f"WAEP: ${sars.get('waep'):.2f}")
                        st.markdown("**Source:** " + sars.get("source", "N/A"))

                # Totals
                if data.get("summary"):
                    st.markdown("---")
                    st.markdown("### Totals")
                    summary = data["summary"]

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        vested = summary.get('total_dilutive_securities_vested') or 0
                        st.metric("Total Vested Dilution", f"{vested:,}")
                        st.caption("Currently dilutive")
                    with col2:
                        unvested = summary.get('total_dilutive_securities_unvested') or 0
                        st.metric("Total Unvested Dilution", f"{unvested:,}")
                        st.caption("Potentially dilutive")
                    with col3:
                        total = summary.get('total_potential_dilution') or 0
                        st.metric("Total Potential Dilution", f"{total:,}")
                        st.caption("Vested + Unvested")

                    if summary.get("notes"):
                        st.info(summary["notes"])

                # Anti-dilutive securities
                if data.get("anti_dilutive_excluded") and len(data["anti_dilutive_excluded"]) > 0:
                    with st.expander("⚠️ Anti-Dilutive Securities Excluded"):
                        st.caption("These securities were excluded from diluted EPS calculations:")
                        for item in data["anti_dilutive_excluded"]:
                            st.markdown(f"- **{item.get('type')}**: {item.get('shares') or 0:,} shares")
                            st.caption(f"  Reason: {item.get('reason', 'N/A')}")

                # Debug: Show raw JSON data
                with st.expander("🔍 Debug: Raw AI Response (JSON)"):
                    st.json(data)

            # Show error if analysis failed
            elif st.session_state.fdso_analysis_error:
                st.error(f"FDSO Analysis Failed: {st.session_state.fdso_analysis_error}")
                st.markdown(f"[View 10-K Filing Manually]({filing.filing_url})")
                st.caption(f"Filing Date: {filing.filing_date} | Period: {filing.period_of_report}")

            # Fallback to link only
            else:
                st.markdown(f"[View 10-K Filing]({filing.filing_url})")
                st.caption(f"Filing Date: {filing.filing_date} | Period: {filing.period_of_report}")

    except Exception as e:
        st.error(f"Error displaying FDSO section: {str(e)}")
        logger.error(f"FDSO display error: {e}", exc_info=True)

# Footer
st.markdown("---")
st.caption("Data from SEC EDGAR | Extraction powered by GPT-4o-mini | Rate limited to 10 req/sec")
