"""
SEC EDGAR API client with rate limiting.
Handles CIK lookup, filings retrieval, and document downloading.
"""

import requests
import logging
import time
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date, timedelta
from threading import Lock
import os
from bs4 import BeautifulSoup

try:
    from .models import Filing, FilingContent
except ImportError:
    from models import Filing, FilingContent

logger = logging.getLogger(__name__)

# SEC requires User-Agent identification
DEFAULT_USER_AGENT = os.getenv("SEC_USER_AGENT", "FinancialExtractor support@example.com")


class RateLimiter:
    """Thread-safe rate limiter for SEC API compliance (max 10 req/sec)."""

    def __init__(self, max_per_second: int = 10):
        self.max_per_second = max_per_second
        self.min_interval = 1.0 / max_per_second
        self.last_request_time = 0.0
        self.lock = Lock()

    def wait(self):
        """Wait if needed to respect rate limit."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)
            self.last_request_time = time.time()


# Global rate limiter instance
_rate_limiter = RateLimiter(max_per_second=10)


class SECClient:
    """Client for interacting with SEC EDGAR API."""

    BASE_URL = "https://data.sec.gov"
    ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

    def __init__(self, user_agent: str = None):
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self._company_tickers_cache = None

    def _make_request(self, url: str, timeout: int = 30) -> requests.Response:
        """Make rate-limited request to SEC."""
        _rate_limiter.wait()

        # Use session with standard headers (requests handles Host automatically)
        response = self.session.get(url, timeout=timeout)
        response.raise_for_status()
        return response

    def _get_company_tickers(self) -> Dict:
        """Fetch and cache company tickers JSON."""
        if self._company_tickers_cache is None:
            # Use www.sec.gov for company tickers (not data.sec.gov)
            url = "https://www.sec.gov/files/company_tickers.json"
            response = self._make_request(url)
            self._company_tickers_cache = response.json()
        return self._company_tickers_cache

    def ticker_to_cik(self, ticker: str) -> Optional[str]:
        """
        Convert ticker to CIK using SEC company tickers JSON.
        Returns CIK with leading zeros (10 digits).
        """
        try:
            companies = self._get_company_tickers()
            ticker_upper = ticker.upper().strip()

            for company_data in companies.values():
                if company_data.get("ticker", "").upper() == ticker_upper:
                    cik = str(company_data["cik_str"]).zfill(10)
                    logger.info(f"Found CIK {cik} for ticker {ticker}")
                    return cik

            logger.warning(f"No CIK found for ticker {ticker}")
            return None

        except Exception as e:
            logger.error(f"Error looking up ticker {ticker}: {e}")
            return None

    def get_company_info(self, cik: str) -> Optional[Dict]:
        """
        Fetch company info and submissions from SEC.
        Returns company name, ticker, and filings data.
        """
        try:
            url = f"{self.BASE_URL}/submissions/CIK{cik.zfill(10)}.json"
            logger.info(f"Fetching company info for CIK {cik}")
            response = self._make_request(url)
            return response.json()

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.error(f"CIK {cik} not found")
            else:
                logger.error(f"HTTP error fetching company info: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching company info for CIK {cik}: {e}")
            return None

    def get_10q_filings(
        self,
        ticker: str = None,
        cik: str = None,
        num_quarters: int = 4
    ) -> Tuple[Optional[Dict], List[Filing]]:
        """
        Get the most recent 10-Q filings for a company.

        Returns:
            (company_info, list of Filing objects)
        """
        # Resolve CIK
        if not cik and ticker:
            cik = self.ticker_to_cik(ticker)
            if not cik:
                return None, []
        elif not cik:
            logger.error("Must provide ticker or CIK")
            return None, []

        # Get company submissions
        company_info = self.get_company_info(cik)
        if not company_info:
            return None, []

        company_name = company_info.get("name", "Unknown")
        company_ticker = company_info.get("tickers", [ticker or ""])[0] if company_info.get("tickers") else (ticker or "")

        # Extract recent filings
        recent = company_info.get("filings", {}).get("recent", {})
        if not recent:
            logger.warning("No recent filings found")
            return company_info, []

        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accession_numbers = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        report_dates = recent.get("reportDate", [])

        # Filter for 10-Q filings
        filings = []
        for i in range(len(forms)):
            if forms[i] == "10-Q":
                try:
                    filing_date = datetime.strptime(filing_dates[i], "%Y-%m-%d").date()
                    report_date = datetime.strptime(report_dates[i], "%Y-%m-%d").date() if report_dates[i] else filing_date

                    accession = accession_numbers[i]
                    accession_clean = accession.replace("-", "")
                    cik_clean = cik.lstrip("0") or "0"

                    filing_url = f"{self.ARCHIVES_URL}/{cik_clean}/{accession_clean}/{primary_docs[i]}"

                    filing = Filing(
                        cik=cik,
                        company_name=company_name,
                        ticker=company_ticker,
                        accession_number=accession,
                        form_type="10-Q",
                        filing_date=filing_date,
                        period_of_report=report_date,
                        primary_document=primary_docs[i],
                        filing_url=filing_url
                    )
                    filings.append(filing)

                    if len(filings) >= num_quarters:
                        break

                except Exception as e:
                    logger.warning(f"Error parsing filing {i}: {e}")
                    continue

        logger.info(f"Found {len(filings)} 10-Q filings for {company_ticker}")
        return company_info, filings

    def get_10k_filings(
        self,
        ticker: str = None,
        cik: str = None,
        num_years: int = 1
    ) -> Tuple[Optional[Dict], List[Filing]]:
        """
        Get the most recent 10-K filings for a company.

        10-K filings contain comprehensive equity compensation disclosures
        needed for fully diluted share calculations.

        Args:
            ticker: Stock ticker symbol
            cik: CIK number (alternative to ticker)
            num_years: Number of annual filings to retrieve

        Returns:
            (company_info dict, list of Filing objects)
        """
        # Resolve CIK
        if not cik and ticker:
            cik = self.ticker_to_cik(ticker)
            if not cik:
                return None, []
        elif not cik:
            logger.error("Must provide ticker or CIK")
            return None, []

        # Get company submissions
        company_info = self.get_company_info(cik)
        if not company_info:
            return None, []

        company_name = company_info.get("name", "Unknown")
        company_ticker = company_info.get("tickers", [ticker or ""])[0] if company_info.get("tickers") else (ticker or "")

        # Extract recent filings
        recent = company_info.get("filings", {}).get("recent", {})
        if not recent:
            logger.warning("No recent filings found")
            return company_info, []

        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accession_numbers = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        report_dates = recent.get("reportDate", [])

        # Filter for 10-K filings (exclude 10-K/A amendments)
        filings = []
        for i in range(len(forms)):
            if forms[i] == "10-K":  # Exact match, not 10-K/A
                try:
                    filing_date = datetime.strptime(filing_dates[i], "%Y-%m-%d").date()
                    report_date = datetime.strptime(report_dates[i], "%Y-%m-%d").date() if report_dates[i] else filing_date

                    accession = accession_numbers[i]
                    accession_clean = accession.replace("-", "")
                    cik_clean = cik.lstrip("0") or "0"

                    filing_url = f"{self.ARCHIVES_URL}/{cik_clean}/{accession_clean}/{primary_docs[i]}"

                    filing = Filing(
                        cik=cik,
                        company_name=company_name,
                        ticker=company_ticker,
                        accession_number=accession,
                        form_type="10-K",
                        filing_date=filing_date,
                        period_of_report=report_date,
                        primary_document=primary_docs[i],
                        filing_url=filing_url
                    )
                    filings.append(filing)

                    if len(filings) >= num_years:
                        break

                except Exception as e:
                    logger.warning(f"Error parsing 10-K filing {i}: {e}")
                    continue

        logger.info(f"Found {len(filings)} 10-K filings for {company_ticker}")
        return company_info, filings

    def download_filing(self, filing: Filing, max_chars: int = 500000) -> Optional[FilingContent]:
        """
        Download and parse a filing document.

        Returns:
            FilingContent with HTML and plain text
        """
        try:
            logger.info(f"Downloading filing from {filing.filing_url}")
            response = self._make_request(filing.filing_url, timeout=60)
            html = response.text
            logger.info(f"Downloaded {len(html):,} chars of HTML")

            # Convert to plain text FIRST (before truncating)
            # This ensures we process the full document to find financial data
            plain_text = self._html_to_text(html)
            logger.info(f"Extracted {len(plain_text):,} chars of text")

            # Now truncate the plain text if needed (only if max_chars is specified)
            if max_chars is not None and len(plain_text) > max_chars:
                logger.info(f"Truncating text from {len(plain_text)} to {max_chars} chars")
                plain_text = plain_text[:max_chars]

            return FilingContent(
                filing=filing,
                html_text=html[:max_chars] if max_chars is not None and len(html) > max_chars else html,
                plain_text=plain_text,
                char_count=len(plain_text)
            )

        except Exception as e:
            logger.error(f"Error downloading filing: {e}")
            return None

    def _html_to_text(self, html: str) -> str:
        """
        Extract clean text from HTML using BeautifulSoup.

        IMPORTANT: Preserves table structure by converting tables to
        pipe-separated format so that row/column relationships are maintained.
        This is critical for financial tables.

        Based on working implementation from clinicaltrials/sec_filings.py
        """
        try:
            # Use html.parser (more reliable for SEC filings than lxml)
            soup = BeautifulSoup(html, 'html.parser')

            # Remove script, style, and other non-content tags
            for tag in soup(['script', 'style', 'meta', 'link', 'header', 'footer', 'nav']):
                tag.decompose()

            # Remove hidden XBRL header sections (display:none divs at the start)
            # These contain XBRL context data that pollutes the text extraction
            for div in soup.find_all('div', style=lambda s: s and 'display:none' in s.lower() if s else False):
                div.decompose()

            # Also remove ix:header elements directly (XBRL inline header)
            for ix_header in soup.find_all(['ix:header', 'ix:hidden']):
                ix_header.decompose()

            # Process tables to preserve row/column structure
            for table in soup.find_all('table'):
                table_text_lines = []
                for row in table.find_all('tr'):
                    cells = row.find_all(['td', 'th'])
                    if cells:
                        # Extract text from each cell, preserving values
                        cell_texts = []
                        for cell in cells:
                            cell_text = cell.get_text(strip=True)
                            if cell_text:  # Only include non-empty cells
                                cell_texts.append(cell_text)
                        if cell_texts:
                            # Create pipe-separated row preserving column relationships
                            row_text = ' | '.join(cell_texts)
                            table_text_lines.append(f"| {row_text} |")

                # Replace table with formatted text
                if table_text_lines:
                    table_str = '\n'.join(table_text_lines)
                    table.replace_with(soup.new_string(f"\n\n{table_str}\n\n"))
                else:
                    table.decompose()

            # Get text with newlines preserved between elements
            text = soup.get_text(separator='\n', strip=True)

            # Clean up excessive whitespace while preserving structure
            text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
            text = re.sub(r'[ \t]+', ' ', text)  # Collapse spaces

            return text.strip()

        except Exception as e:
            logger.error(f"Error converting HTML to text: {e}")
            # Fallback: simple strip
            return BeautifulSoup(html, 'html.parser').get_text()


def fetch_10q_filings(
    ticker: str = None,
    cik: str = None,
    num_quarters: int = 4,
    user_agent: str = None
) -> Tuple[Optional[Dict], List[Filing]]:
    """
    Convenience function to fetch 10-Q filings.

    Returns:
        (company_info dict, list of Filing objects)
    """
    client = SECClient(user_agent=user_agent)
    return client.get_10q_filings(ticker=ticker, cik=cik, num_quarters=num_quarters)


def download_filing_content(
    filing: Filing,
    max_chars: int = 150000,
    user_agent: str = None
) -> Optional[FilingContent]:
    """
    Convenience function to download filing content.
    """
    client = SECClient(user_agent=user_agent)
    return client.download_filing(filing, max_chars=max_chars)


def fetch_10k_filings(
    ticker: str = None,
    cik: str = None,
    num_years: int = 1,
    user_agent: str = None
) -> Tuple[Optional[Dict], List[Filing]]:
    """
    Convenience function to fetch 10-K filings.

    Args:
        ticker: Stock ticker symbol
        cik: CIK number (alternative to ticker)
        num_years: Number of annual filings to retrieve
        user_agent: Custom user agent string

    Returns:
        (company_info dict, list of Filing objects)
    """
    client = SECClient(user_agent=user_agent)
    return client.get_10k_filings(ticker=ticker, cik=cik, num_years=num_years)
