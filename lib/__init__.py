# SEC Financial Extractor Library
# Modules for extracting financial data from SEC EDGAR filings

from .sec_client import SECClient, fetch_10q_filings
from .extractor import FinancialExtractor, extract_financial_data
from .models import (
    Filing, FilingContent, EvidenceSnippet,
    BalanceSheetData, CashFlowData, QuarterlyFCF,
    CashPosition, BurnMetrics
)
from .cash_calculator import calculate_cash_position
from .burn_calculator import BurnCalculator

__all__ = [
    'SECClient', 'fetch_10q_filings',
    'FinancialExtractor', 'extract_financial_data',
    'Filing', 'FilingContent', 'EvidenceSnippet',
    'BalanceSheetData', 'CashFlowData', 'QuarterlyFCF',
    'CashPosition', 'BurnMetrics',
    'calculate_cash_position', 'BurnCalculator'
]
