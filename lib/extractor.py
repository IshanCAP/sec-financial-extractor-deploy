"""
GPT-5-mini powered financial data extraction from SEC filings.
Extracts specific line items with evidence snippets and source citations.

Based on the working pattern from clinicaltrials/sec_filings.py
Enhanced with Bay Bridge Bio's structured prompt methodology.
"""

import json
import logging
import os
import re
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, date

# Import openai the same way as sec_filings.py
import openai

try:
    from .models import (
        FilingContent, BalanceSheetData, CashFlowData,
        EvidenceSnippet, Filing
    )
except ImportError:
    from models import (
        FilingContent, BalanceSheetData, CashFlowData,
        EvidenceSnippet, Filing
    )

logger = logging.getLogger(__name__)

# Extraction prompt - Handles both quarterly and YTD cash flow statements
EXTRACTION_PROMPT = """You are a financial data extraction specialist analyzing SEC 10-Q filings.

================================================================================
FILING INFORMATION
================================================================================
Company: {company_name}
Ticker: {ticker}
Form Type: 10-Q
Filing Date: {filing_date}
Period of Report: {period_of_report}
================================================================================

TASK: Extract financial metrics from this 10-Q filing.

################################################################################
#  IMPORTANT: CASH FLOW STATEMENT STRUCTURE                                    #
################################################################################

10-Q filings have DIFFERENT cash flow statement formats depending on the quarter:

- Q1 FILINGS (e.g., March 31): Have "Three Months Ended" columns -> EXTRACT THIS
- Q2 FILINGS (e.g., June 30): May ONLY have "Six Months Ended" columns -> EXTRACT YTD
- Q3 FILINGS (e.g., September 30): May ONLY have "Nine Months Ended" columns -> EXTRACT YTD

*** HOW TO EXTRACT ***

1. FIRST: Check if the cash flow statement has a "Three Months Ended" column
   - If YES: Extract from that column (period_months = 3)
   - If NO: The filing only has YTD data, extract that instead

2. If only YTD data available (Six Months or Nine Months):
   - Set period_months to the actual period (6 or 9)
   - Extract from the YTD column
   - We will calculate quarterly data from multiple filings

EXAMPLE 1 - Q1 Filing with quarterly data:
| Three Months Ended March 31, |
| 2025 | 2024 |
Net cash used in operating activities: (1,037) | (989)
-> Extract: value = -1037000000 (assuming in millions), period_months = 3

EXAMPLE 2 - Q3 Filing with only YTD data:
| Nine Months Ended September 30, |
| 2025 | 2024 |
Net cash used in operating activities: (2,803) | (3,829)
-> Extract: value = -2803000000 (assuming in millions), period_months = 9

BALANCE SHEET EXTRACTIONS (from Condensed Balance Sheet):

1. Cash and Cash Equivalents - from Current Assets section
   Extract the total cash and cash equivalents line item

2. Marketable Securities - COMBINE ALL of the following from Current Assets AND Non-Current Assets:
   - Short-term marketable securities
   - Long-term marketable securities / Long-term investments
   - Available-for-sale securities (both short and long term)
   - Trading securities
   SUM all these items together. List ALL line items found in the evidence field.

3. Long-Term Debt - COMBINE ALL of the following from Non-Current Liabilities:
   - Long-term debt (non-current portion)
   - Non-current operating lease liabilities
   - Long-term borrowings
   - Senior notes / Senior debt (long-term)
   - Convertible debt (non-current portion)
   - Long-term notes payable
   SUM all these items together. List ALL line items found in the evidence field.

4. Short-Term Debt - COMBINE ALL of the following from Current Liabilities:
   - Short-term debt
   - Current portion of long-term debt
   - Current maturities of long-term debt
   - Current operating lease liabilities
   - Short-term borrowings
   - Notes payable (current)
   - Commercial paper
   - Current portion of convertible debt
   SUM all these items together. List ALL line items found in the evidence field.

CASH FLOW EXTRACTIONS (from Statement of Cash Flows):

4. Operating Cash Flow - the subtotal line for Operating Activities section
   Look for: "Net cash used in operating activities" or "Net cash provided by operating activities"
   This is the TOTAL of the operating section, NOT individual line items

   IMPORTANT: Set period_months based on what's ACTUALLY in the cash flow statement:
   - "Three Months Ended" -> period_months = 3
   - "Six Months Ended" -> period_months = 6
   - "Nine Months Ended" -> period_months = 9
   - "Year Ended" or "Twelve Months Ended" -> period_months = 12 (for 10-K annual filings)

5. Capital Expenditure (OPTIONAL) - from Investing Activities section
   Look for: "Purchases of property and equipment", "Capital expenditures", "Additions to property, plant and equipment"
   DO NOT use: "Purchases of marketable securities" or investment-related items
   If no CapEx line exists, use null

RESPONSE FORMAT - Return ONLY valid JSON (no markdown, no code fences):
{{
  "balance_sheet": {{
    "cash_and_equivalents": {{
      "value": <number in actual dollars>,
      "raw_value": <number as shown in filing>,
      "unit_multiplier": <1, 1000, or 1000000>,
      "evidence": "<exact row label from filing>",
      "value_text": "<exact value as displayed>"
    }},
    "marketable_securities": {{
      "value": <number or null - SUM of all marketable securities: short-term + long-term + available-for-sale>,
      "raw_value": <number or null - total sum as calculated>,
      "unit_multiplier": <1, 1000, or 1000000>,
      "evidence": "<list ALL line items found and their values, e.g., 'Short-term marketable securities: $XXX + Long-term investments: $YYY = $Total'>",
      "value_text": "<breakdown of all values added>"
    }},
    "long_term_debt": {{
      "value": <number or null - SUM of all long-term debt components: long-term debt + non-current operating leases + convertible debt (long-term) + other long-term borrowings>,
      "raw_value": <number or null - total sum as calculated>,
      "unit_multiplier": <1, 1000, or 1000000>,
      "evidence": "<list ALL line items found and their values, e.g., 'Long-term debt: $XXX + Non-current operating lease liabilities: $YYY + Senior notes: $ZZZ = $Total'>",
      "value_text": "<breakdown of all values added>"
    }},
    "short_term_debt": {{
      "value": <number or null - SUM of all short-term debt components: short-term debt + current portion of long-term debt + current operating leases + short-term borrowings>,
      "raw_value": <number or null - total sum as calculated>,
      "unit_multiplier": <1, 1000, or 1000000>,
      "evidence": "<list ALL line items found and their values, e.g., 'Short-term debt: $XXX + Current portion of long-term debt: $YYY + Current operating lease liabilities: $ZZZ = $Total'>",
      "value_text": "<breakdown of all values added>"
    }},
    "period_end_date": "<YYYY-MM-DD>"
  }},
  "cash_flow": {{
    "period_type": "<'three_months', 'six_months', 'nine_months', or 'twelve_months' based on what's in the filing>",
    "column_header": "<copy EXACT text from cash flow table header, e.g., 'Three Months Ended March 31,' or 'Nine Months Ended September 30,'>",
    "operating_cash_flow": {{
      "value": <number in actual dollars>,
      "raw_value": <number as shown>,
      "unit_multiplier": <1, 1000, or 1000000>,
      "evidence": "<exact row label - should be subtotal like 'Net cash used in operating activities'>",
      "value_text": "<exact value>"
    }},
    "capital_expenditure": {{
      "value": <number or null if not found - keep negative sign>,
      "raw_value": <number or null>,
      "unit_multiplier": <1, 1000, or 1000000>,
      "evidence": "<exact row label or empty>",
      "value_text": "<exact value or empty>"
    }},
    "period_months": <3, 6, 9, or 12 - MUST match what's in the cash flow statement header>,
    "period_start_date": "<YYYY-MM-DD>",
    "period_end_date": "<YYYY-MM-DD>"
  }},
  "extraction_confidence": <1-5>,
  "notes": "<any issues or notes about the extraction>"
}}

################################################################################
#  VALIDATION RULES                                                            #
################################################################################

1. period_months MUST match the cash flow statement header (3, 6, 9, or 12)
2. period_start_date to period_end_date MUST match period_months (~90, ~180, ~270, or ~365 days)
3. column_header MUST reflect what's actually in the filing
4. Operating cash flow evidence should mention "operating activities" subtotal
5. CapEx should NOT include marketable securities purchases
6. If table says "(in thousands)" multiply by 1000; "(in millions)" multiply by 1000000
7. Use null for any field not found - do not guess or estimate
8. For marketable securities, long-term debt, and short-term debt: LIST ALL COMPONENTS in evidence field with their individual values and show the calculation (e.g., "$100 + $200 = $300")

BEFORE SUBMITTING, VERIFY:
- Did I extract from the correct column matching period_months?
- Does the period length (in days) match period_months?
- If only YTD data available, did I set period_months to 6 or 9 accordingly?

================================================================================
DOCUMENT TEXT
================================================================================
{filing_text}"""


def _clean_llm_json_response(response_text: str) -> str:
    """
    Clean LLM response to extract valid JSON.

    LLMs sometimes wrap JSON in markdown code blocks. This function handles that.
    (Same approach used by Bay Bridge Bio / sec_filings.py)
    """
    cleaned = response_text.strip()

    # Remove markdown code block wrapper if present
    if cleaned.startswith('```'):
        # Find the end of the first line (```json or just ```)
        first_newline = cleaned.find('\n')
        if first_newline > 0:
            cleaned = cleaned[first_newline + 1:]

        # Remove trailing ```
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]

    # Also handle ```json at the start (if newline wasn't found)
    if cleaned.startswith('json'):
        cleaned = cleaned[4:]

    return cleaned.strip()


def _extract_financial_sections(text: str, max_total_chars: int = 80000) -> str:
    """
    Intelligently extract financial statement sections from SEC filing text.

    This function finds and extracts the Balance Sheet and Cash Flow Statement
    sections specifically, ensuring they're included even in very long documents.

    Args:
        text: Full filing text
        max_total_chars: Maximum characters to return

    Returns:
        Text containing balance sheet and cash flow sections (with context)
    """
    text_lower = text.lower()

    # Patterns to find cash flow statement section - look for the section header
    cash_flow_patterns = [
        "condensed consolidated statements of cash flows",
        "consolidated statements of cash flows",
        "statements of cash flows",
        "statement of cash flows",
    ]

    # Patterns to find balance sheet section
    balance_sheet_patterns = [
        "condensed consolidated balance sheets",
        "consolidated balance sheets",
        "balance sheets",
        "condensed balance sheets",
    ]

    # Find cash flow statement position
    cash_flow_start = -1
    for pattern in cash_flow_patterns:
        idx = text_lower.find(pattern)
        if idx != -1:
            # Found cash flow section header
            cash_flow_start = max(0, idx - 200)  # Small context before
            logger.info(f"Found cash flow section at position {idx} with pattern: '{pattern}'")
            break

    # Find balance sheet position
    balance_sheet_start = -1
    for pattern in balance_sheet_patterns:
        idx = text_lower.find(pattern)
        if idx != -1:
            balance_sheet_start = max(0, idx - 200)
            logger.info(f"Found balance sheet section at position {idx} with pattern: '{pattern}'")
            break

    # If no specific sections found, fall back to using more of the document
    if cash_flow_start == -1 and balance_sheet_start == -1:
        logger.warning("Could not find specific financial sections, using full text truncation")
        return text[:max_total_chars]

    # Build extraction ranges - we need to include the actual tables, not just headers
    # Tables typically follow the headers and can be 15-25k chars each
    extracted_parts = []

    # Include header/metadata section (first 8k)
    header_end = 8000
    extracted_parts.append(text[:header_end])
    logger.info(f"Included header section: 0-{header_end} chars")

    # Find the earliest financial section
    sections = []
    if balance_sheet_start != -1:
        sections.append(("balance_sheet", balance_sheet_start))
    if cash_flow_start != -1:
        sections.append(("cash_flow", cash_flow_start))

    sections.sort(key=lambda x: x[1])  # Sort by position

    # Extract financial sections with enough content to capture the tables
    for section_name, start_pos in sections:
        # Make sure we don't overlap with what we already have
        actual_start = max(start_pos, header_end)
        if actual_start < start_pos + 100:  # Section header is after our last extract
            actual_start = start_pos

        # Extract a large chunk to capture the full table
        # Balance sheets can be 10-15k, cash flow statements 15-25k
        section_length = 30000 if section_name == "cash_flow" else 20000
        end_pos = min(actual_start + section_length, len(text))

        section_text = text[actual_start:end_pos]
        extracted_parts.append(f"\n\n--- {section_name.upper().replace('_', ' ')} SECTION ---\n\n{section_text}")
        logger.info(f"Included {section_name} section: {actual_start}-{end_pos} chars ({end_pos - actual_start} chars)")

    # Combine extracted parts
    combined = "\n".join(extracted_parts)

    # Ensure we don't exceed max chars
    if len(combined) > max_total_chars:
        logger.info(f"Combined sections too long ({len(combined)}), truncating to {max_total_chars}")
        combined = combined[:max_total_chars]

    logger.info(f"Smart extraction: {len(text)} chars -> {len(combined)} chars")
    return combined


def _validate_extraction(data: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate extracted data - accepts both quarterly (3-month) and YTD (6/9-month) data.

    Returns:
        (is_valid, list of warning messages)
    """
    warnings = []
    is_valid = True

    # Check balance sheet data
    bs = data.get("balance_sheet", {})
    cash_data = bs.get("cash_and_equivalents", {})
    cash_value = cash_data.get("value") if isinstance(cash_data, dict) else cash_data

    if cash_value is not None:
        # Cash should be positive
        if cash_value < 0:
            warnings.append(f"Cash value is negative: {cash_value}")
        # Sanity check: cash over $1 trillion is suspicious
        if cash_value > 1_000_000_000_000:
            warnings.append(f"Cash value suspiciously high: {cash_value}")

    # Check cash flow data
    cf = data.get("cash_flow", {})

    # =========================================================================
    # VALIDATION #1: Verify period_months is valid (3, 6, 9, or 12)
    # =========================================================================
    period_months = cf.get("period_months")
    if period_months is not None:
        if period_months not in [3, 6, 9, 12]:
            warnings.append(f"Invalid period_months: {period_months}. Expected 3, 6, 9, or 12.")
            is_valid = False
        elif period_months == 3:
            logger.info("Quarterly (3-month) cash flow data extracted")
        elif period_months == 12:
            logger.info("Annual (12-month) cash flow data extracted from 10-K")
            warnings.append("ANNUAL_DATA: 12-month data extracted (Q4 calculation requires 9-month YTD from prior 10-Q)")
        else:
            logger.info(f"YTD ({period_months}-month) cash flow data extracted - will need prior period for quarterly calculation")
            warnings.append(f"YTD_DATA: {period_months}-month data extracted (quarterly calculation requires prior period)")

    # =========================================================================
    # VALIDATION #2: Check column header matches period_months
    # =========================================================================
    column_header_raw = cf.get("column_header") or ""
    column_header = column_header_raw.lower() if column_header_raw else ""
    if column_header and period_months:
        has_three = "three" in column_header or "3 month" in column_header
        has_six = "six" in column_header or "6 month" in column_header
        has_nine = "nine" in column_header or "9 month" in column_header

        # Verify column header matches period_months
        if period_months == 3 and not has_three:
            warnings.append(f"Column header mismatch: period_months=3 but header='{column_header_raw}'")
        elif period_months == 6 and not has_six:
            warnings.append(f"Column header mismatch: period_months=6 but header='{column_header_raw}'")
        elif period_months == 9 and not has_nine:
            warnings.append(f"Column header mismatch: period_months=9 but header='{column_header_raw}'")
    elif not column_header:
        warnings.append("WARNING: No column header provided")

    # =========================================================================
    # VALIDATION #3: Date-based period length check
    # =========================================================================
    period_start_str = cf.get("period_start_date", "")
    period_end_str = cf.get("period_end_date", "")

    if period_start_str and period_end_str and period_months:
        try:
            period_start = datetime.strptime(period_start_str, "%Y-%m-%d").date()
            period_end = datetime.strptime(period_end_str, "%Y-%m-%d").date()
            period_days = (period_end - period_start).days

            # Validate period length matches period_months
            expected_days_map = {3: (80, 100), 6: (170, 190), 9: (260, 280)}
            min_days, max_days = expected_days_map.get(period_months, (0, 365))

            if period_days < min_days or period_days > max_days:
                warnings.append(f"Period length mismatch: {period_days} days doesn't match period_months={period_months} (expected {min_days}-{max_days} days)")
            else:
                logger.info(f"Period validation PASSED: {period_days} days ({period_start} to {period_end}) for {period_months}-month period")
        except ValueError as e:
            warnings.append(f"Could not parse period dates: {e}")
    else:
        warnings.append("WARNING: Missing period dates or period_months - cannot validate period length")

    ocf_data = cf.get("operating_cash_flow", {})
    ocf_value = ocf_data.get("value") if isinstance(ocf_data, dict) else ocf_data

    capex_data = cf.get("capital_expenditure", {})
    capex_value = capex_data.get("value") if isinstance(capex_data, dict) else capex_data

    # =========================================================================
    # VALIDATION #4: Verify period_type matches period_months
    # =========================================================================
    period_type = cf.get("period_type", "")
    valid_period_types = {
        3: "three_months",
        6: "six_months",
        9: "nine_months"
    }
    if period_months and period_type:
        expected_type = valid_period_types.get(period_months)
        if expected_type and period_type != expected_type:
            warnings.append(f"Period type mismatch: period_months={period_months} but period_type='{period_type}'")

    # =========================================================================
    # VALIDATION #5: OCF evidence should mention operating activities
    # =========================================================================
    ocf_evidence = ""
    if isinstance(ocf_data, dict):
        ocf_evidence = ocf_data.get("evidence", "").lower()
    if ocf_evidence:
        valid_ocf_terms = ["operating activities", "operating", "cash used in", "cash provided by"]
        has_valid_ocf_evidence = any(term in ocf_evidence for term in valid_ocf_terms)
        if not has_valid_ocf_evidence:
            warnings.append(f"OCF evidence may not be from operating activities section: '{ocf_data.get('evidence', '')}'")

    # =========================================================================
    # VALIDATION #6: CapEx should NOT be marketable securities
    # =========================================================================
    capex_evidence = ""
    if isinstance(capex_data, dict):
        capex_evidence = capex_data.get("evidence", "").lower()
    if capex_evidence:
        invalid_capex_terms = ["marketable securities", "investment", "short-term investment", "securities purchased"]
        has_invalid_capex = any(term in capex_evidence for term in invalid_capex_terms)
        if has_invalid_capex:
            warnings.append(f"WRONG CAPEX: Evidence mentions investments/securities (not capital expenditure): '{capex_data.get('evidence', '')}'. REJECTED.")
            is_valid = False

        # Valid CapEx should mention property/equipment
        valid_capex_terms = ["property", "equipment", "capital expenditure", "ppe", "plant", "leasehold"]
        has_valid_capex = any(term in capex_evidence for term in valid_capex_terms)
        if not has_valid_capex and capex_value is not None:
            warnings.append(f"CapEx evidence may not be actual capital expenditure: '{capex_data.get('evidence', '')}'")

    # CapEx should typically be negative (cash outflow) - but it's optional
    if capex_value is not None and capex_value > 0:
        warnings.append(f"CapEx is positive (unusual): {capex_value}")

    # Check confidence level
    confidence = data.get("extraction_confidence", 0)
    if confidence < 3:
        warnings.append(f"Low extraction confidence: {confidence}/5")

    # Check for required fields (CapEx is now optional)
    if cash_value is None:
        warnings.append("Cash and equivalents not found")
    if ocf_value is None:
        warnings.append("Operating cash flow not found")
        is_valid = False

    # CapEx is optional - just note if missing, don't invalidate
    if capex_value is None:
        warnings.append("CapEx not found (optional - may not have property purchases)")

    # =========================================================================
    # VALIDATION #7: Sanity check OCF magnitude based on period
    # =========================================================================
    if ocf_value is not None and period_months:
        # Scale expected magnitude by period length
        quarterly_max = 500_000_000  # $500M max for quarterly
        period_max = quarterly_max * (period_months / 3)
        if abs(ocf_value) > period_max:
            warnings.append(f"OCF magnitude large for {period_months}-month period: ${ocf_value:,.0f}")

    # Log final validation status
    if is_valid:
        logger.info("Extraction validation PASSED - 3-month data confirmed")
    else:
        logger.error("Extraction validation FAILED - wrong period detected")

    return is_valid, warnings


class FinancialExtractor:
    """GPT-5-mini powered financial data extraction."""

    def __init__(self, api_key: str = None, model: str = "gpt-5-mini"):
        """
        Initialize the extractor.

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-5-mini for improved accuracy)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None

        if self.api_key:
            # Validate API key format
            if len(self.api_key) < 10:
                logger.error("Invalid OpenAI API key format")
            else:
                try:
                    # Initialize client exactly like sec_filings.py
                    self.client = openai.OpenAI(api_key=self.api_key)
                    logger.info(f"OpenAI client initialized with model {model}")
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAI client: {e}")
        else:
            logger.warning("No OPENAI_API_KEY found, GPT-5-mini extraction unavailable")

    def is_available(self) -> bool:
        """Check if extraction is available."""
        return self.client is not None

    def extract_all(
        self,
        content: FilingContent,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Extract all financial data from a filing.

        Returns dict with:
        - balance_sheet: BalanceSheetData
        - cash_flow: CashFlowData
        - raw_response: The raw GPT response
        - success: bool
        """
        if not self.is_available():
            logger.error("GPT-5-mini extraction not available (no API key)")
            return {"success": False, "error": "No API key"}

        # Use smart extraction to find financial statement sections
        # This ensures cash flow statement is included even in long documents
        text = content.plain_text
        max_chars = 80000  # Allow up to 80k chars (~20k tokens)

        if len(text) > max_chars:
            logger.info(f"Document is {len(text)} chars, using smart section extraction...")
            text = _extract_financial_sections(text, max_chars)
        else:
            logger.info(f"Document is {len(text)} chars, using full text")

        # Build prompt with filing metadata context (Bay Bridge Bio pattern)
        filing = content.filing
        prompt = EXTRACTION_PROMPT.format(
            company_name=filing.company_name or "Unknown",
            ticker=filing.ticker or "Unknown",
            filing_date=str(filing.filing_date) if filing.filing_date else "Unknown",
            period_of_report=str(filing.period_of_report) if filing.period_of_report else "Unknown",
            filing_text=text
        )

        for attempt in range(max_retries + 1):
            try:
                logger.info(f"GPT-5-mini extraction attempt {attempt + 1}/{max_retries + 1}")
                logger.info(f"Sending {len(text):,} chars to OpenAI ({self.model})...")

                # API call - extract available cash flow data (quarterly or YTD)
                system_message = """You are a financial data extraction specialist. Return only valid JSON.

################################################################################
# CASH FLOW EXTRACTION RULES
################################################################################

For cash flow data (OCF and CapEx) from 10-Q filings:

IMPORTANT: Different quarters have different cash flow statement formats:
- Q1 filings (Jan-Mar): Have "Three Months Ended" -> period_months = 3
- Q2 filings (Apr-Jun): Often only have "Six Months Ended" YTD -> period_months = 6
- Q3 filings (Jul-Sep): Often only have "Nine Months Ended" YTD -> period_months = 9

1. Extract from whatever period is available in the cash flow statement
2. Set period_months to match the actual column header (3, 6, or 9)
3. Set period_type to match: "three_months", "six_months", or "nine_months"

4. OCF must be the "Net cash used/provided by operating activities" subtotal
   - NOT individual line items

5. CapEx must be actual capital expenditure (property/equipment purchases)
   - NOT "Purchases of marketable securities"
   - NOT investment purchases

CRITICAL: If the filing only has YTD data, extract that - don't return null!
We will calculate quarterly values from consecutive filings.
"""
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=4096,  # GPT-5 uses max_completion_tokens instead of max_tokens
                    timeout=90
                )

                raw_response = response.choices[0].message.content
                logger.info(f"Got response from OpenAI ({len(raw_response) if raw_response else 0} chars)")

                if not raw_response:
                    logger.warning(f"Empty response on attempt {attempt + 1}")
                    if attempt < max_retries:
                        time.sleep(2)  # Wait before retry
                        continue
                    return {"success": False, "error": "Empty response from API"}

                # Clean and parse JSON (using proven pattern from sec_filings.py)
                cleaned = _clean_llm_json_response(raw_response)

                # Parse JSON
                data = json.loads(cleaned)

                # Log period verification for debugging
                cf = data.get("cash_flow", {})
                period_months = cf.get("period_months")
                column_header = cf.get("column_header", "")
                logger.info(f"Period verification: {period_months} months, column: '{column_header}'")

                # Validate extraction with strict 3-month enforcement
                is_valid, validation_warnings = _validate_extraction(data)
                if validation_warnings:
                    for warning in validation_warnings:
                        logger.warning(f"Validation: {warning}")

                # If any critical validation failed (WRONG CAPEX, etc.), retry
                # Note: YTD data is now acceptable, so we don't reject based on period alone
                critical_failures = [w for w in validation_warnings if "REJECTED" in w or "WRONG CAPEX" in w]
                if not is_valid and critical_failures:
                    logger.error(f"Extraction failed with critical errors: {critical_failures}")
                    if attempt < max_retries:
                        logger.info(f"Retrying extraction (attempt {attempt + 2}/{max_retries + 1})...")
                        time.sleep(2)
                        continue
                    return {"success": False, "error": f"Extraction failed. Errors: {'; '.join(critical_failures)}"}

                # Convert to data models
                balance_sheet = self._parse_balance_sheet(data, content.filing)
                cash_flow = self._parse_cash_flow(data, content.filing)

                logger.info(f"Extraction successful (confidence: {data.get('extraction_confidence', 'unknown')})")

                # Combine notes with validation warnings
                notes = data.get("notes", "")
                if validation_warnings:
                    notes = f"{notes} | Warnings: {'; '.join(validation_warnings)}" if notes else f"Warnings: {'; '.join(validation_warnings)}"

                return {
                    "success": True,
                    "balance_sheet": balance_sheet,
                    "cash_flow": cash_flow,
                    "raw_response": data,
                    "confidence": data.get("extraction_confidence", 0),
                    "notes": notes,
                    "validation_warnings": validation_warnings
                }

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}")
                logger.warning(f"Raw response preview: {raw_response[:500] if 'raw_response' in dir() and raw_response else 'None'}...")
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                return {"success": False, "error": f"JSON parse error: {e}"}

            except openai.APIConnectionError as e:
                # Connection error - retry with backoff
                logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    wait_time = (attempt + 1) * 3  # 3, 6, 9 seconds
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return {"success": False, "error": f"Connection error: Could not reach OpenAI API. Check your internet connection."}

            except openai.RateLimitError as e:
                logger.warning(f"Rate limit error on attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                    logger.info(f"Rate limited. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return {"success": False, "error": "Rate limit exceeded. Please wait and try again."}

            except openai.AuthenticationError as e:
                logger.error(f"Authentication error: {e}")
                return {"success": False, "error": "Invalid API key. Please check your OpenAI API key."}

            except openai.APITimeoutError as e:
                logger.warning(f"Timeout on attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                return {"success": False, "error": "Request timed out. OpenAI may be slow. Try again."}

            except Exception as e:
                error_msg = str(e).lower()
                logger.error(f"Extraction error: {e}")

                # Check for specific error types in message
                if "connection" in error_msg:
                    if attempt < max_retries:
                        time.sleep(3)
                        continue
                    return {"success": False, "error": "Connection error. Check internet connection."}
                elif "timeout" in error_msg:
                    if attempt < max_retries:
                        time.sleep(2)
                        continue
                    return {"success": False, "error": "Request timed out."}
                else:
                    return {"success": False, "error": str(e)}

        return {"success": False, "error": "Max retries exceeded"}

    def _parse_balance_sheet(self, data: Dict, filing: Filing) -> BalanceSheetData:
        """Parse balance sheet data from GPT response."""
        bs = data.get("balance_sheet", {})

        # Handle both nested and flat structures
        cash_data = bs.get("cash_and_equivalents", {})
        if isinstance(cash_data, (int, float)):
            cash_data = {"value": cash_data}

        securities_data = bs.get("marketable_securities", {})
        if isinstance(securities_data, (int, float)):
            securities_data = {"value": securities_data}

        evidence = []

        # Cash evidence
        cash_value = cash_data.get("value") if isinstance(cash_data, dict) else cash_data
        if cash_value is not None:
            evidence.append(EvidenceSnippet(
                field_name="cash_and_equivalents",
                extracted_value=cash_value,
                source_text=cash_data.get("evidence", "") if isinstance(cash_data, dict) else "",
                table_label=cash_data.get("value_text", "") if isinstance(cash_data, dict) else str(cash_value),
                confidence=0.9 if isinstance(cash_data, dict) and cash_data.get("evidence") else 0.5,
                filing_url=filing.filing_url
            ))

        # Securities evidence
        securities_value = securities_data.get("value") if isinstance(securities_data, dict) else securities_data
        if securities_value is not None:
            evidence.append(EvidenceSnippet(
                field_name="marketable_securities",
                extracted_value=securities_value,
                source_text=securities_data.get("evidence", "") if isinstance(securities_data, dict) else "",
                table_label=securities_data.get("value_text", "") if isinstance(securities_data, dict) else str(securities_value),
                confidence=0.9 if isinstance(securities_data, dict) and securities_data.get("evidence") else 0.5,
                filing_url=filing.filing_url
            ))

        # Long-term debt
        long_term_data = bs.get("long_term_debt", {})
        if isinstance(long_term_data, (int, float)):
            long_term_data = {"value": long_term_data}

        long_term_value = long_term_data.get("value") if isinstance(long_term_data, dict) else long_term_data
        if long_term_value is not None:
            evidence.append(EvidenceSnippet(
                field_name="long_term_debt",
                extracted_value=long_term_value,
                source_text=long_term_data.get("evidence", "") if isinstance(long_term_data, dict) else "",
                table_label=long_term_data.get("value_text", "") if isinstance(long_term_data, dict) else str(long_term_value),
                confidence=0.9 if isinstance(long_term_data, dict) and long_term_data.get("evidence") else 0.5,
                filing_url=filing.filing_url
            ))

        # Short-term debt
        short_term_data = bs.get("short_term_debt", {})
        if isinstance(short_term_data, (int, float)):
            short_term_data = {"value": short_term_data}

        short_term_value = short_term_data.get("value") if isinstance(short_term_data, dict) else short_term_data
        if short_term_value is not None:
            evidence.append(EvidenceSnippet(
                field_name="short_term_debt",
                extracted_value=short_term_value,
                source_text=short_term_data.get("evidence", "") if isinstance(short_term_data, dict) else "",
                table_label=short_term_data.get("value_text", "") if isinstance(short_term_data, dict) else str(short_term_value),
                confidence=0.9 if isinstance(short_term_data, dict) and short_term_data.get("evidence") else 0.5,
                filing_url=filing.filing_url
            ))

        # Parse period end date
        period_end = None
        if bs.get("period_end_date"):
            try:
                period_end = datetime.strptime(bs["period_end_date"], "%Y-%m-%d").date()
            except:
                pass

        # Extract unit multipliers from filing source
        cash_unit = cash_data.get("unit_multiplier") if isinstance(cash_data, dict) else None
        securities_unit = securities_data.get("unit_multiplier") if isinstance(securities_data, dict) else None
        debt_unit = long_term_data.get("unit_multiplier") if isinstance(long_term_data, dict) else None
        # Use short-term debt unit if long-term not available
        if debt_unit is None:
            debt_unit = short_term_data.get("unit_multiplier") if isinstance(short_term_data, dict) else None

        return BalanceSheetData(
            cash_and_equivalents=cash_value,
            marketable_securities=securities_value,
            long_term_debt=long_term_value,
            short_term_debt=short_term_value,
            period_end_date=period_end,
            filing=filing,
            evidence=evidence,
            extraction_notes=data.get("notes", ""),
            cash_unit=cash_unit,
            securities_unit=securities_unit,
            debt_unit=debt_unit
        )

    def _parse_cash_flow(self, data: Dict, filing: Filing) -> CashFlowData:
        """Parse cash flow data from GPT response with 3-month period validation."""
        cf = data.get("cash_flow", {})

        # Handle both nested and flat structures
        ocf_data = cf.get("operating_cash_flow", {})
        if isinstance(ocf_data, (int, float)):
            ocf_data = {"value": ocf_data}

        capex_data = cf.get("capital_expenditure", {})
        if isinstance(capex_data, (int, float)):
            capex_data = {"value": capex_data}

        evidence = []

        # OCF evidence
        ocf_value = ocf_data.get("value") if isinstance(ocf_data, dict) else ocf_data
        if ocf_value is not None:
            # Include column header in evidence for transparency
            column_header = cf.get("column_header", "")
            evidence_text = ocf_data.get("evidence", "") if isinstance(ocf_data, dict) else ""
            if column_header:
                evidence_text = f"{evidence_text} [Column: {column_header}]"

            evidence.append(EvidenceSnippet(
                field_name="operating_cash_flow",
                extracted_value=ocf_value,
                source_text=evidence_text,
                table_label=ocf_data.get("value_text", "") if isinstance(ocf_data, dict) else str(ocf_value),
                confidence=0.9 if isinstance(ocf_data, dict) and ocf_data.get("evidence") else 0.5,
                filing_url=filing.filing_url
            ))

        # CapEx evidence
        capex_value = capex_data.get("value") if isinstance(capex_data, dict) else capex_data
        if capex_value is not None:
            # Include column header in CapEx evidence for transparency
            capex_evidence_text = capex_data.get("evidence", "") if isinstance(capex_data, dict) else ""
            if column_header:
                capex_evidence_text = f"{capex_evidence_text} [Column: {column_header}]"

            evidence.append(EvidenceSnippet(
                field_name="capital_expenditure",
                extracted_value=capex_value,
                source_text=capex_evidence_text,
                table_label=capex_data.get("value_text", "") if isinstance(capex_data, dict) else str(capex_value),
                confidence=0.9 if isinstance(capex_data, dict) and capex_data.get("evidence") else 0.5,
                filing_url=filing.filing_url
            ))

        # Parse dates
        period_start = None
        period_end = None
        if cf.get("period_start_date"):
            try:
                period_start = datetime.strptime(cf["period_start_date"], "%Y-%m-%d").date()
            except:
                pass
        if cf.get("period_end_date"):
            try:
                period_end = datetime.strptime(cf["period_end_date"], "%Y-%m-%d").date()
            except:
                pass

        # =========================================================================
        # CHECKPOINT: Log period length at parse time
        # =========================================================================
        if period_start and period_end:
            period_days = (period_end - period_start).days
            period_months = cf.get("period_months", 3)
            logger.info(f"Parsed cash flow: {period_days} days ({period_months}-month period) from {period_start} to {period_end}")

        # Get CapEx as absolute value (stored positive)
        if capex_value is not None:
            capex_value = abs(capex_value)

        # Extract unit multipliers from filing source
        ocf_unit = ocf_data.get("unit_multiplier") if isinstance(ocf_data, dict) else None
        capex_unit = capex_data.get("unit_multiplier") if isinstance(capex_data, dict) else None

        # Get period_months from extracted data (defaults to 3 if not specified)
        period_months = cf.get("period_months", 3)
        if period_months not in [3, 6, 9, 12]:
            logger.warning(f"Invalid period_months {period_months}, defaulting to 3")
            period_months = 3

        return CashFlowData(
            operating_cash_flow=ocf_value,
            capital_expenditure=capex_value,
            period_start_date=period_start,
            period_end_date=period_end,
            filing=filing,
            evidence=evidence,
            extraction_notes=data.get("notes", ""),
            ocf_unit=ocf_unit,
            capex_unit=capex_unit,
            period_months=period_months
        )


def extract_financial_data(
    content: FilingContent,
    api_key: str = None
) -> Dict[str, Any]:
    """
    Convenience function to extract financial data.

    Returns dict with balance_sheet, cash_flow, and metadata.
    """
    extractor = FinancialExtractor(api_key=api_key)
    return extractor.extract_all(content)
