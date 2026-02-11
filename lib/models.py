"""
Data models for SEC financial extraction.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import List, Optional, Dict, Any


def format_currency(
    value: Optional[float],
    show_sign: bool = False,
    source_unit: Optional[int] = None
) -> str:
    """
    Format a currency value with K (thousands) or MM (millions) notation.

    Args:
        value: The value in actual dollars
        show_sign: Whether to show +/- sign for positive/negative values
        source_unit: The unit multiplier from the filing (1000 for "in thousands",
                    1000000 for "in millions"). If provided, format uses this unit.
                    If None, format is based on magnitude.

    Returns:
        Formatted string like "$72.2MM", "$500K", "-$102.6MM"

    Examples (with source_unit):
        72,200,000 with source_unit=1000 -> "$72,200K" (filing said "in thousands")
        72,200,000 with source_unit=1000000 -> "$72.2MM" (filing said "in millions")

    Examples (without source_unit - magnitude based):
        72,200,000 -> "$72.2MM"
        500,000 -> "$500K"
        -102,600,000 -> "-$102.6MM"
    """
    if value is None:
        return "N/A"

    # Determine sign
    sign = ""
    if value < 0:
        sign = "-"
        value = abs(value)
    elif show_sign and value > 0:
        sign = "+"

    # If source_unit is provided, use it for formatting
    if source_unit is not None:
        if source_unit == 1_000_000:  # Filing said "in millions"
            display_value = value / 1_000_000
            if display_value >= 1000:
                # Very large - show as billions
                formatted = f"{sign}${display_value / 1000:.2f}B"
            elif display_value == int(display_value):
                formatted = f"{sign}${int(display_value):,}MM"
            else:
                formatted = f"{sign}${display_value:,.1f}MM"
        elif source_unit == 1_000:  # Filing said "in thousands"
            display_value = value / 1_000
            if display_value >= 1_000_000:
                # Very large - show as millions for readability
                formatted = f"{sign}${display_value / 1000:,.1f}MM"
            elif display_value == int(display_value):
                formatted = f"{sign}${int(display_value):,}K"
            else:
                formatted = f"{sign}${display_value:,.1f}K"
        else:  # No multiplier (source_unit == 1)
            formatted = f"{sign}${value:,.0f}"
    else:
        # No source_unit - format based on magnitude
        if value >= 1_000_000_000:  # Billions
            formatted = f"{sign}${value / 1_000_000_000:.2f}B"
        elif value >= 1_000_000:  # Millions
            formatted = f"{sign}${value / 1_000_000:.1f}MM"
        elif value >= 1_000:  # Thousands
            formatted = f"{sign}${value / 1_000:.1f}K"
        else:
            formatted = f"{sign}${value:,.0f}"

    # Clean up trailing zeros (e.g., "$72.0MM" -> "$72MM")
    formatted = formatted.replace('.0MM', 'MM').replace('.0K', 'K').replace('.0B', 'B')

    return formatted


@dataclass
class EvidenceSnippet:
    """Source evidence for an extracted data point."""
    field_name: str
    extracted_value: Optional[float]
    source_text: str
    table_label: str
    confidence: float  # 0-1
    filing_url: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "extracted_value": self.extracted_value,
            "source_text": self.source_text,
            "table_label": self.table_label,
            "confidence": self.confidence,
            "filing_url": self.filing_url
        }


@dataclass
class Filing:
    """SEC filing metadata."""
    cik: str
    company_name: str
    ticker: str
    accession_number: str
    form_type: str
    filing_date: date
    period_of_report: date
    primary_document: str
    filing_url: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cik": self.cik,
            "company_name": self.company_name,
            "ticker": self.ticker,
            "accession_number": self.accession_number,
            "form_type": self.form_type,
            "filing_date": self.filing_date.isoformat() if self.filing_date else None,
            "period_of_report": self.period_of_report.isoformat() if self.period_of_report else None,
            "primary_document": self.primary_document,
            "filing_url": self.filing_url
        }


@dataclass
class FilingContent:
    """Downloaded filing content."""
    filing: Filing
    html_text: str
    plain_text: str
    char_count: int


@dataclass
class BalanceSheetData:
    """Extracted balance sheet items."""
    cash_and_equivalents: Optional[float] = None
    marketable_securities: Optional[float] = None
    total_cash_position: Optional[float] = None
    long_term_debt: Optional[float] = None
    short_term_debt: Optional[float] = None
    total_debt: Optional[float] = None
    period_end_date: Optional[date] = None
    filing: Optional[Filing] = None
    evidence: List[EvidenceSnippet] = field(default_factory=list)
    extraction_notes: str = ""
    # Source units from filing (1000 = "in thousands", 1000000 = "in millions")
    cash_unit: Optional[int] = None
    securities_unit: Optional[int] = None
    debt_unit: Optional[int] = None

    def __post_init__(self):
        # Calculate total cash if components exist
        if self.cash_and_equivalents is not None or self.marketable_securities is not None:
            cash = self.cash_and_equivalents or 0
            securities = self.marketable_securities or 0
            self.total_cash_position = cash + securities

        # Calculate total debt if components exist
        if self.long_term_debt is not None or self.short_term_debt is not None:
            long_term = self.long_term_debt or 0
            short_term = self.short_term_debt or 0
            self.total_debt = long_term + short_term

    @property
    def cash_formatted(self) -> str:
        """Cash and equivalents formatted with K/MM notation based on filing source."""
        return format_currency(self.cash_and_equivalents, source_unit=self.cash_unit)

    @property
    def securities_formatted(self) -> str:
        """Marketable securities formatted with K/MM notation based on filing source."""
        return format_currency(self.marketable_securities, source_unit=self.securities_unit)

    @property
    def total_formatted(self) -> str:
        """Total cash position formatted with K/MM notation."""
        # Use cash_unit for total since it's usually from the same table
        return format_currency(self.total_cash_position, source_unit=self.cash_unit)

    @property
    def long_term_debt_formatted(self) -> str:
        """Long-term debt formatted with K/MM notation based on filing source."""
        return format_currency(self.long_term_debt, source_unit=self.debt_unit)

    @property
    def short_term_debt_formatted(self) -> str:
        """Short-term debt formatted with K/MM notation based on filing source."""
        return format_currency(self.short_term_debt, source_unit=self.debt_unit)

    @property
    def total_debt_formatted(self) -> str:
        """Total debt formatted with K/MM notation based on filing source."""
        return format_currency(self.total_debt, source_unit=self.debt_unit)

    @property
    def source_unit_label(self) -> str:
        """Human-readable label for the source unit."""
        if self.cash_unit == 1_000_000:
            return "in millions"
        elif self.cash_unit == 1_000:
            return "in thousands"
        return "in dollars"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cash_and_equivalents": self.cash_and_equivalents,
            "cash_formatted": self.cash_formatted,
            "marketable_securities": self.marketable_securities,
            "securities_formatted": self.securities_formatted,
            "total_cash_position": self.total_cash_position,
            "total_formatted": self.total_formatted,
            "long_term_debt": self.long_term_debt,
            "long_term_debt_formatted": self.long_term_debt_formatted,
            "short_term_debt": self.short_term_debt,
            "short_term_debt_formatted": self.short_term_debt_formatted,
            "total_debt": self.total_debt,
            "total_debt_formatted": self.total_debt_formatted,
            "source_unit_label": self.source_unit_label,
            "period_end_date": self.period_end_date.isoformat() if self.period_end_date else None,
            "filing": self.filing.to_dict() if self.filing else None,
            "evidence": [e.to_dict() for e in self.evidence],
            "extraction_notes": self.extraction_notes
        }


@dataclass
class CashFlowData:
    """Extracted cash flow statement items."""
    operating_cash_flow: Optional[float] = None
    capital_expenditure: Optional[float] = None  # Stored as positive (absolute value)
    free_cash_flow: Optional[float] = None
    period_start_date: Optional[date] = None
    period_end_date: Optional[date] = None
    filing: Optional[Filing] = None
    evidence: List[EvidenceSnippet] = field(default_factory=list)
    extraction_notes: str = ""
    # Source units from filing (1000 = "in thousands", 1000000 = "in millions")
    ocf_unit: Optional[int] = None
    capex_unit: Optional[int] = None
    # Period length in months (3 = quarterly, 6 = 6-month YTD, 9 = 9-month YTD, 12 = annual from 10-K)
    period_months: int = 3

    def __post_init__(self):
        # Calculate FCF if OCF exists
        if self.operating_cash_flow is not None:
            if self.capital_expenditure is not None:
                # CapEx is stored as positive, subtract from OCF
                self.free_cash_flow = self.operating_cash_flow - abs(self.capital_expenditure)
            else:
                # No CapEx data - use OCF as FCF (conservative estimate)
                self.free_cash_flow = self.operating_cash_flow

    @property
    def ocf_formatted(self) -> str:
        """Operating cash flow formatted with K/MM notation based on filing source."""
        return format_currency(self.operating_cash_flow, show_sign=True, source_unit=self.ocf_unit)

    @property
    def capex_formatted(self) -> str:
        """Capital expenditure formatted with K/MM notation based on filing source."""
        return format_currency(self.capital_expenditure, source_unit=self.capex_unit)

    @property
    def fcf_formatted(self) -> str:
        """Free cash flow formatted with K/MM notation based on filing source."""
        return format_currency(self.free_cash_flow, show_sign=True, source_unit=self.ocf_unit)

    @property
    def source_unit_label(self) -> str:
        """Human-readable label for the source unit."""
        if self.ocf_unit == 1_000_000:
            return "in millions"
        elif self.ocf_unit == 1_000:
            return "in thousands"
        return "in dollars"

    @property
    def is_quarterly(self) -> bool:
        """Whether this is quarterly (3-month) data vs YTD."""
        return self.period_months == 3

    @property
    def period_label(self) -> str:
        """Human-readable label for the period type."""
        if self.period_months == 3:
            return "3-month (quarterly)"
        elif self.period_months == 6:
            return "6-month (YTD)"
        elif self.period_months == 9:
            return "9-month (YTD)"
        elif self.period_months == 12:
            return "12-month (annual)"
        return f"{self.period_months}-month"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operating_cash_flow": self.operating_cash_flow,
            "ocf_formatted": self.ocf_formatted,
            "capital_expenditure": self.capital_expenditure,
            "capex_formatted": self.capex_formatted,
            "free_cash_flow": self.free_cash_flow,
            "fcf_formatted": self.fcf_formatted,
            "source_unit_label": self.source_unit_label,
            "period_months": self.period_months,
            "is_quarterly": self.is_quarterly,
            "period_label": self.period_label,
            "period_start_date": self.period_start_date.isoformat() if self.period_start_date else None,
            "period_end_date": self.period_end_date.isoformat() if self.period_end_date else None,
            "filing": self.filing.to_dict() if self.filing else None,
            "evidence": [e.to_dict() for e in self.evidence],
            "extraction_notes": self.extraction_notes
        }


@dataclass
class QuarterlyFCF:
    """Free cash flow data for one quarter."""
    quarter_label: str  # e.g., "Q3 2025"
    fiscal_quarter: int  # 1-4
    fiscal_year: int
    period_start: Optional[date]
    period_end: Optional[date]
    operating_cash_flow: Optional[float]
    capital_expenditure: Optional[float]
    free_cash_flow: Optional[float]
    is_selected: bool = True  # For UI selection
    filing: Optional[Filing] = None
    evidence: List[EvidenceSnippet] = field(default_factory=list)
    # Source units from filing
    source_unit: Optional[int] = None
    # Calculation notes (e.g., "Calculated: 9mo YTD - 6mo YTD")
    extraction_notes: str = ""

    @property
    def ocf_formatted(self) -> str:
        """Operating cash flow formatted with K/MM notation based on filing source."""
        return format_currency(self.operating_cash_flow, show_sign=True, source_unit=self.source_unit)

    @property
    def capex_formatted(self) -> str:
        """Capital expenditure formatted with K/MM notation based on filing source."""
        return format_currency(self.capital_expenditure, source_unit=self.source_unit)

    @property
    def fcf_formatted(self) -> str:
        """Free cash flow formatted with K/MM notation based on filing source."""
        return format_currency(self.free_cash_flow, show_sign=True, source_unit=self.source_unit)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quarter_label": self.quarter_label,
            "fiscal_quarter": self.fiscal_quarter,
            "fiscal_year": self.fiscal_year,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "operating_cash_flow": self.operating_cash_flow,
            "ocf_formatted": self.ocf_formatted,
            "capital_expenditure": self.capital_expenditure,
            "capex_formatted": self.capex_formatted,
            "free_cash_flow": self.free_cash_flow,
            "fcf_formatted": self.fcf_formatted,
            "is_selected": self.is_selected,
            "filing": self.filing.to_dict() if self.filing else None,
            "evidence": [e.to_dict() for e in self.evidence]
        }


@dataclass
class CashPosition:
    """Current cash position summary."""
    cash_and_equivalents: float
    marketable_securities: float
    total_cash_at_hand: float
    as_of_date: date
    filing: Filing
    evidence: List[EvidenceSnippet]
    # Source unit from filing
    source_unit: Optional[int] = None
    # Debt
    long_term_debt: Optional[float] = None
    short_term_debt: Optional[float] = None
    total_debt: Optional[float] = None
    debt_unit: Optional[int] = None
    # Net position (Total Cash - Total Debt)
    net_cash_position: Optional[float] = None

    @property
    def cash_formatted(self) -> str:
        """Cash and equivalents formatted with K/MM notation based on filing source."""
        return format_currency(self.cash_and_equivalents, source_unit=self.source_unit)

    @property
    def securities_formatted(self) -> str:
        """Marketable securities formatted with K/MM notation based on filing source."""
        return format_currency(self.marketable_securities, source_unit=self.source_unit)

    @property
    def total_formatted(self) -> str:
        """Total cash at hand formatted with K/MM notation based on filing source."""
        return format_currency(self.total_cash_at_hand, source_unit=self.source_unit)

    @property
    def long_term_debt_formatted(self) -> str:
        """Long-term debt formatted with K/MM notation based on filing source."""
        return format_currency(self.long_term_debt, source_unit=self.debt_unit)

    @property
    def short_term_debt_formatted(self) -> str:
        """Short-term debt formatted with K/MM notation based on filing source."""
        return format_currency(self.short_term_debt, source_unit=self.debt_unit)

    @property
    def total_debt_formatted(self) -> str:
        """Total debt formatted with K/MM notation based on filing source."""
        return format_currency(self.total_debt, source_unit=self.debt_unit)

    @property
    def net_cash_formatted(self) -> str:
        """Net cash position (Total Cash - Total Debt) formatted with K/MM notation."""
        return format_currency(self.net_cash_position, show_sign=True, source_unit=self.source_unit)

    @property
    def source_unit_label(self) -> str:
        """Human-readable label for the source unit."""
        if self.source_unit == 1_000_000:
            return "in millions"
        elif self.source_unit == 1_000:
            return "in thousands"
        return "in dollars"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cash_and_equivalents": self.cash_and_equivalents,
            "cash_formatted": self.cash_formatted,
            "marketable_securities": self.marketable_securities,
            "securities_formatted": self.securities_formatted,
            "total_cash_at_hand": self.total_cash_at_hand,
            "total_formatted": self.total_formatted,
            "operating_lease_liabilities": self.operating_lease_liabilities,
            "lease_liabilities_formatted": self.lease_liabilities_formatted,
            "source_unit_label": self.source_unit_label,
            "as_of_date": self.as_of_date.isoformat() if self.as_of_date else None,
            "filing": self.filing.to_dict() if self.filing else None,
            "evidence": [e.to_dict() for e in self.evidence]
        }


@dataclass
class BurnMetrics:
    """Burn rate analysis results."""
    quarters_analyzed: List[QuarterlyFCF]
    quarters_included: List[str]
    average_quarterly_burn: float
    is_manual_override: bool
    manual_override_value: Optional[float]
    cash_position: Optional[CashPosition]
    runway_quarters: Optional[float]
    runway_end_date: Optional[date]
    # Source unit from quarterly data (for consistent formatting)
    source_unit: Optional[int] = None

    @property
    def burn_formatted(self) -> str:
        """Average quarterly burn formatted with K/MM notation based on filing source."""
        # Use source_unit from quarters for consistent formatting
        unit = self.source_unit
        if unit is None and self.quarters_analyzed:
            # Get source_unit from first quarter that has one
            for q in self.quarters_analyzed:
                if q.source_unit is not None:
                    unit = q.source_unit
                    break
        return format_currency(self.average_quarterly_burn, show_sign=True, source_unit=unit)

    @property
    def runway_formatted(self) -> str:
        """Runway formatted as quarters and date."""
        if self.runway_quarters is None or self.runway_quarters == float('inf'):
            return "Cash flow positive"
        return f"{self.runway_quarters:.1f} quarters"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quarters_analyzed": [q.to_dict() for q in self.quarters_analyzed],
            "quarters_included": self.quarters_included,
            "average_quarterly_burn": self.average_quarterly_burn,
            "burn_formatted": self.burn_formatted,
            "is_manual_override": self.is_manual_override,
            "manual_override_value": self.manual_override_value,
            "cash_position": self.cash_position.to_dict() if self.cash_position else None,
            "runway_quarters": self.runway_quarters,
            "runway_formatted": self.runway_formatted,
            "runway_end_date": self.runway_end_date.isoformat() if self.runway_end_date else None
        }


# =============================================================================
# FULLY DILUTED SHARES MODELS
# =============================================================================

class DilutiveSecurityType(Enum):
    """Classification of dilutive securities by calculation method."""
    # TSM Instruments (have exercise price)
    STOCK_OPTION = "stock_option"
    WARRANT = "warrant"
    PREFUNDED_WARRANT = "prefunded_warrant"
    SAR = "sar"  # Stock Appreciation Rights
    ESPP = "espp"  # Employee Stock Purchase Plan

    # Add-the-Shares (no exercise price, K=$0)
    RSU = "rsu"  # Restricted Stock Units
    RESTRICTED_STOCK = "restricted_stock"
    PSU = "psu"  # Performance Stock Units

    # If-Converted
    CONVERTIBLE_DEBT = "convertible_debt"
    CONVERTIBLE_PREFERRED = "convertible_preferred"
    SAFE = "safe"

    # Other
    ATM_PROGRAM = "atm_program"
    EARNOUT = "earnout"
    OTHER = "other"


class VestingStatus(Enum):
    """Vesting status for equity awards."""
    VESTED = "vested"
    UNVESTED = "unvested"
    EXPECTED_TO_VEST = "expected_to_vest"
    ALL = "all"  # Combined


def format_shares(value: Optional[float], source_unit: Optional[int] = None) -> str:
    """
    Format share count with K/M/B notation.

    Args:
        value: The number of shares
        source_unit: The unit multiplier from filing (1000 or 1000000)

    Returns:
        Formatted string like "15.2M", "500K", "1.2B"
    """
    if value is None:
        return "N/A"

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K"
    else:
        return f"{value:,.0f}"


@dataclass
class DilutiveSecurity:
    """Individual dilutive security with extraction evidence."""
    security_type: DilutiveSecurityType
    description: str  # e.g., "Incentive Stock Options", "Series A Warrants"

    # Quantities
    shares_count: Optional[float] = None  # Number of shares/units
    shares_vested: Optional[float] = None
    shares_unvested: Optional[float] = None
    shares_expected_to_vest: Optional[float] = None

    # Exercise/Conversion terms
    exercise_price: Optional[float] = None  # WAEP for options/warrants
    conversion_price: Optional[float] = None  # For convertibles
    conversion_ratio: Optional[float] = None  # Shares per unit

    # For convertible debt
    principal_amount: Optional[float] = None

    # Metadata
    expiry_date: Optional[date] = None
    grant_date: Optional[date] = None
    weighted_avg_remaining_life: Optional[float] = None  # Years

    # Evidence tracking (following existing pattern)
    evidence: List[EvidenceSnippet] = field(default_factory=list)
    source_unit: Optional[int] = None  # 1000 or 1000000
    extraction_notes: str = ""
    filing: Optional[Filing] = None

    # Calculation results (populated by calculator)
    incremental_shares: Optional[float] = None
    is_dilutive: bool = True  # False if anti-dilutive
    calculation_method: str = ""  # "TSM", "Add-the-Shares", "If-Converted"
    is_included: bool = True  # User can toggle

    @property
    def waep_formatted(self) -> str:
        """Format weighted average exercise price."""
        if self.exercise_price is None:
            return "N/A"
        return f"${self.exercise_price:.2f}"

    @property
    def shares_formatted(self) -> str:
        """Format share count."""
        return format_shares(self.shares_count, self.source_unit)

    @property
    def incremental_formatted(self) -> str:
        """Format incremental shares from dilution calculation."""
        return format_shares(self.incremental_shares)

    @property
    def principal_formatted(self) -> str:
        """Format principal amount for convertible debt."""
        return format_currency(self.principal_amount, source_unit=self.source_unit)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "security_type": self.security_type.value,
            "description": self.description,
            "shares_count": self.shares_count,
            "shares_formatted": self.shares_formatted,
            "shares_vested": self.shares_vested,
            "shares_unvested": self.shares_unvested,
            "shares_expected_to_vest": self.shares_expected_to_vest,
            "exercise_price": self.exercise_price,
            "waep_formatted": self.waep_formatted,
            "conversion_price": self.conversion_price,
            "conversion_ratio": self.conversion_ratio,
            "principal_amount": self.principal_amount,
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None,
            "incremental_shares": self.incremental_shares,
            "incremental_formatted": self.incremental_formatted,
            "is_dilutive": self.is_dilutive,
            "calculation_method": self.calculation_method,
            "is_included": self.is_included,
            "evidence": [e.to_dict() for e in self.evidence],
            "extraction_notes": self.extraction_notes,
        }


@dataclass
class EquityCompensationData:
    """Extracted equity compensation data from 10-K filing."""
    # Basic shares
    common_shares_outstanding: Optional[float] = None
    weighted_avg_basic: Optional[float] = None
    weighted_avg_diluted: Optional[float] = None

    # Stock Options - comprehensive breakdown
    options_outstanding: Optional[float] = None
    options_vested: Optional[float] = None  # Exercisable options
    options_unvested: Optional[float] = None
    options_waep: Optional[float] = None  # Weighted Average Exercise Price (all outstanding)
    options_vested_waep: Optional[float] = None  # WAEP for vested options
    options_unvested_waep: Optional[float] = None  # WAEP for unvested options
    options_intrinsic_value: Optional[float] = None  # Aggregate intrinsic value
    options_avg_remaining_life: Optional[float] = None  # Weighted avg remaining life (years)

    # RSUs
    rsus_outstanding: Optional[float] = None
    rsus_vested: Optional[float] = None  # Vested during period (informational)
    rsus_unvested: Optional[float] = None  # Usually equals outstanding
    rsus_expected_to_vest: Optional[float] = None  # Forfeiture-adjusted

    # Restricted Stock Awards (separate from RSUs)
    restricted_stock_outstanding: Optional[float] = None
    restricted_stock_unvested: Optional[float] = None

    # PSUs
    psus_outstanding: Optional[float] = None
    psus_target_payout: Optional[float] = None  # Shares at target performance
    psus_max_payout: Optional[float] = None  # Shares at max performance (often 150-200%)

    # Stock Appreciation Rights (SARs)
    sars_outstanding: Optional[float] = None
    sars_vested: Optional[float] = None
    sars_waep: Optional[float] = None  # Base price
    sars_settles_in_stock: Optional[bool] = None  # True if stock-settled, False if cash

    # Warrants (may be multiple tranches)
    warrants: List[DilutiveSecurity] = field(default_factory=list)

    # Convertible instruments
    convertible_debt: List[DilutiveSecurity] = field(default_factory=list)
    convertible_preferred: List[DilutiveSecurity] = field(default_factory=list)

    # ESPP
    espp_shares_available: Optional[float] = None
    espp_discount_percent: Optional[float] = None  # e.g., 15 for 15% discount

    # ATM Program
    atm_shares_remaining: Optional[float] = None
    atm_total_capacity: Optional[float] = None

    # Earnouts
    earnout_shares: Optional[float] = None  # Total potential earnout shares

    # Filing metadata
    period_end_date: Optional[date] = None
    filing: Optional[Filing] = None
    evidence: List[EvidenceSnippet] = field(default_factory=list)
    extraction_notes: str = ""
    source_unit: Optional[int] = None
    extraction_confidence: int = 0  # 1-5

    @property
    def basic_shares_formatted(self) -> str:
        """Format basic shares outstanding."""
        return format_shares(self.common_shares_outstanding)

    @property
    def options_formatted(self) -> str:
        """Format options outstanding."""
        return format_shares(self.options_outstanding)

    @property
    def rsus_formatted(self) -> str:
        """Format RSUs outstanding."""
        return format_shares(self.rsus_outstanding)

    def to_dict(self) -> Dict[str, Any]:
        return {
            # Basic shares
            "common_shares_outstanding": self.common_shares_outstanding,
            "basic_shares_formatted": self.basic_shares_formatted,
            "weighted_avg_basic": self.weighted_avg_basic,
            "weighted_avg_diluted": self.weighted_avg_diluted,

            # Stock options - comprehensive
            "options_outstanding": self.options_outstanding,
            "options_formatted": self.options_formatted,
            "options_waep": self.options_waep,
            "options_vested": self.options_vested,
            "options_vested_waep": self.options_vested_waep,
            "options_unvested": self.options_unvested,
            "options_unvested_waep": self.options_unvested_waep,
            "options_intrinsic_value": self.options_intrinsic_value,
            "options_avg_remaining_life": self.options_avg_remaining_life,

            # RSUs
            "rsus_outstanding": self.rsus_outstanding,
            "rsus_formatted": self.rsus_formatted,
            "rsus_vested": self.rsus_vested,
            "rsus_unvested": self.rsus_unvested,
            "rsus_expected_to_vest": self.rsus_expected_to_vest,

            # Restricted stock
            "restricted_stock_outstanding": self.restricted_stock_outstanding,
            "restricted_stock_unvested": self.restricted_stock_unvested,

            # PSUs
            "psus_outstanding": self.psus_outstanding,
            "psus_target_payout": self.psus_target_payout,
            "psus_max_payout": self.psus_max_payout,

            # SARs
            "sars_outstanding": self.sars_outstanding,
            "sars_vested": self.sars_vested,
            "sars_waep": self.sars_waep,
            "sars_settles_in_stock": self.sars_settles_in_stock,

            # Securities lists
            "warrants": [w.to_dict() for w in self.warrants],
            "convertible_debt": [c.to_dict() for c in self.convertible_debt],
            "convertible_preferred": [c.to_dict() for c in self.convertible_preferred],

            # ESPP
            "espp_shares_available": self.espp_shares_available,
            "espp_discount_percent": self.espp_discount_percent,

            # ATM
            "atm_shares_remaining": self.atm_shares_remaining,
            "atm_total_capacity": self.atm_total_capacity,

            # Earnouts
            "earnout_shares": self.earnout_shares,

            # Metadata
            "period_end_date": self.period_end_date.isoformat() if self.period_end_date else None,
            "extraction_confidence": self.extraction_confidence,
            "extraction_notes": self.extraction_notes,
            "evidence": [e.to_dict() for e in self.evidence],
        }


@dataclass
class FullyDilutedSummary:
    """Fully diluted shares calculation summary."""
    # Input: Basic shares
    basic_shares_outstanding: float
    stock_price: float  # Used for TSM calculation

    # TSM Calculations
    tsm_incremental_shares: float = 0.0
    tsm_securities: List[DilutiveSecurity] = field(default_factory=list)

    # Add-the-Shares
    add_shares_total: float = 0.0
    add_shares_securities: List[DilutiveSecurity] = field(default_factory=list)

    # If-Converted
    if_converted_shares: float = 0.0
    if_converted_securities: List[DilutiveSecurity] = field(default_factory=list)

    # Anti-dilutive (excluded from calculation)
    anti_dilutive_shares: float = 0.0
    anti_dilutive_securities: List[DilutiveSecurity] = field(default_factory=list)

    # Totals
    total_dilutive_shares: float = 0.0
    fully_diluted_shares: float = 0.0
    dilution_percentage: float = 0.0

    # Methodology tracking
    include_vested: bool = True
    include_unvested: bool = True
    psu_payout_assumption: str = "target"  # "target" or "max"

    # Source data
    equity_data: Optional[EquityCompensationData] = None
    as_of_date: Optional[date] = None
    filing: Optional[Filing] = None

    @property
    def basic_formatted(self) -> str:
        """Format basic shares."""
        return format_shares(self.basic_shares_outstanding)

    @property
    def fully_diluted_formatted(self) -> str:
        """Format fully diluted shares."""
        return format_shares(self.fully_diluted_shares)

    @property
    def dilution_formatted(self) -> str:
        """Format dilution percentage."""
        return f"{self.dilution_percentage:.1f}%"

    @property
    def tsm_formatted(self) -> str:
        """Format TSM incremental shares."""
        return format_shares(self.tsm_incremental_shares)

    @property
    def add_shares_formatted(self) -> str:
        """Format add-the-shares total."""
        return format_shares(self.add_shares_total)

    @property
    def if_converted_formatted(self) -> str:
        """Format if-converted shares."""
        return format_shares(self.if_converted_shares)

    @property
    def anti_dilutive_formatted(self) -> str:
        """Format anti-dilutive shares (excluded)."""
        return format_shares(self.anti_dilutive_shares)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "basic_shares_outstanding": self.basic_shares_outstanding,
            "basic_formatted": self.basic_formatted,
            "stock_price": self.stock_price,
            "tsm_incremental_shares": self.tsm_incremental_shares,
            "tsm_formatted": self.tsm_formatted,
            "tsm_securities": [s.to_dict() for s in self.tsm_securities],
            "add_shares_total": self.add_shares_total,
            "add_shares_formatted": self.add_shares_formatted,
            "add_shares_securities": [s.to_dict() for s in self.add_shares_securities],
            "if_converted_shares": self.if_converted_shares,
            "if_converted_formatted": self.if_converted_formatted,
            "if_converted_securities": [s.to_dict() for s in self.if_converted_securities],
            "anti_dilutive_shares": self.anti_dilutive_shares,
            "anti_dilutive_formatted": self.anti_dilutive_formatted,
            "anti_dilutive_securities": [s.to_dict() for s in self.anti_dilutive_securities],
            "total_dilutive_shares": self.total_dilutive_shares,
            "fully_diluted_shares": self.fully_diluted_shares,
            "fully_diluted_formatted": self.fully_diluted_formatted,
            "dilution_percentage": self.dilution_percentage,
            "dilution_formatted": self.dilution_formatted,
            "include_vested": self.include_vested,
            "include_unvested": self.include_unvested,
            "psu_payout_assumption": self.psu_payout_assumption,
            "as_of_date": self.as_of_date.isoformat() if self.as_of_date else None,
        }
