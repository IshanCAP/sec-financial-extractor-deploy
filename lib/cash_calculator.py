"""
Cash position calculator.
Calculates total cash at hand from balance sheet data.
"""

import logging
from datetime import date
from typing import Optional

try:
    from .models import BalanceSheetData, CashPosition, EvidenceSnippet
except ImportError:
    from models import BalanceSheetData, CashPosition, EvidenceSnippet

logger = logging.getLogger(__name__)


def calculate_cash_position(balance_sheet: BalanceSheetData) -> Optional[CashPosition]:
    """
    Calculate total cash position from balance sheet data.

    Formula:
        Cash at Hand = Cash and Cash Equivalents + Marketable Securities

    Args:
        balance_sheet: Extracted balance sheet data

    Returns:
        CashPosition with total cash and evidence, or None if no data
    """
    if balance_sheet is None:
        logger.warning("No balance sheet data provided")
        return None

    cash = balance_sheet.cash_and_equivalents or 0
    securities = balance_sheet.marketable_securities or 0

    # If both are zero/None, we don't have valid data
    if cash == 0 and securities == 0 and balance_sheet.cash_and_equivalents is None:
        logger.warning("No cash data available in balance sheet")
        return None

    total = cash + securities

    # Determine as-of date
    as_of_date = balance_sheet.period_end_date
    if as_of_date is None and balance_sheet.filing:
        as_of_date = balance_sheet.filing.period_of_report

    if as_of_date is None:
        logger.warning("No period date available, using filing date")
        if balance_sheet.filing:
            as_of_date = balance_sheet.filing.filing_date

    # Copy evidence
    evidence = list(balance_sheet.evidence)

    # Add total cash evidence
    evidence.append(EvidenceSnippet(
        field_name="total_cash_at_hand",
        extracted_value=total,
        source_text=f"Calculated: Cash ({cash:,.0f}) + Marketable Securities ({securities:,.0f})",
        table_label="Total Cash Position",
        confidence=0.95,
        filing_url=balance_sheet.filing.filing_url if balance_sheet.filing else ""
    ))

    logger.info(f"Cash position calculated: ${total:,.0f} as of {as_of_date}")

    # Calculate total debt if components exist
    long_term = balance_sheet.long_term_debt or 0
    short_term = balance_sheet.short_term_debt or 0
    total_debt = long_term + short_term if (balance_sheet.long_term_debt is not None or balance_sheet.short_term_debt is not None) else None

    # Calculate net cash position (Total Cash - Total Debt)
    net_cash = None
    if total_debt is not None:
        net_cash = total - total_debt
        logger.info(f"Net cash position: ${net_cash:,.0f} (Cash: ${total:,.0f} - Debt: ${total_debt:,.0f})")

    return CashPosition(
        cash_and_equivalents=cash,
        marketable_securities=securities,
        total_cash_at_hand=total,
        as_of_date=as_of_date,
        filing=balance_sheet.filing,
        evidence=evidence,
        source_unit=balance_sheet.cash_unit,  # Pass source unit from filing
        long_term_debt=balance_sheet.long_term_debt,
        short_term_debt=balance_sheet.short_term_debt,
        total_debt=total_debt,
        debt_unit=balance_sheet.debt_unit,
        net_cash_position=net_cash
    )
