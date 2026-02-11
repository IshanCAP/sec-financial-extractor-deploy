"""
Free Cash Flow and Quarterly Burn Rate Calculator.
Supports quarter selection and manual override.
"""

import logging
from datetime import date, timedelta
from typing import List, Optional, Dict, Any

try:
    from .models import (
        CashFlowData, QuarterlyFCF, CashPosition, BurnMetrics, EvidenceSnippet
    )
except ImportError:
    from models import (
        CashFlowData, QuarterlyFCF, CashPosition, BurnMetrics, EvidenceSnippet
    )

logger = logging.getLogger(__name__)


def calculate_quarterly_from_ytd(
    cash_flows: List[CashFlowData]
) -> List[CashFlowData]:
    """
    Calculate quarterly cash flow values from YTD (6-month, 9-month, and 12-month) data.

    SEC filings have different cash flow formats:
    - Q1 (10-Q): 3-month data (quarterly) - use directly
    - Q2 (10-Q): 6-month YTD data - calculate: Q2_quarter = Q2_6month - Q1_3month
    - Q3 (10-Q): 9-month YTD data - calculate: Q3_quarter = Q3_9month - Q2_6month
    - Q4 (10-K): 12-month annual data - calculate: Q4_quarter = 10K_12month - Q3_9month

    IMPORTANT: Uses date-based matching for fiscal year continuity.
    This handles non-calendar fiscal years (e.g., AAPL ending in September).
    Prior period is matched by finding the filing whose period_end_date is
    approximately 3 months before the current period.

    Args:
        cash_flows: List of CashFlowData

    Returns:
        List of CashFlowData with quarterly values calculated
    """
    if not cash_flows:
        return []

    # Sort by period end date (oldest first for proper calculation)
    sorted_flows = sorted(cash_flows, key=lambda x: x.period_end_date or date.min)

    # Build lookup by period_months and period_end_date for matching
    # Key: (period_months, period_end_date) -> CashFlowData
    period_lookup: Dict[int, List[CashFlowData]] = {3: [], 6: [], 9: [], 12: []}
    for cf in sorted_flows:
        if cf.period_months in period_lookup:
            period_lookup[cf.period_months].append(cf)

    def find_prior_period(current_cf: CashFlowData, prior_months: int) -> Optional[CashFlowData]:
        """
        Find the prior period filing that matches this fiscal year.
        For a 6-month YTD, find the 3-month filing ~3 months before.
        For a 9-month YTD, find the 6-month filing ~3 months before.
        For a 12-month annual, find the 9-month filing ~3 months before.
        """
        if current_cf.period_end_date is None:
            return None

        candidates = period_lookup.get(prior_months, [])
        if not candidates:
            return None

        # Find the closest match within a reasonable window (60-120 days before)
        best_match = None
        best_diff = float('inf')

        for candidate in candidates:
            if candidate.period_end_date is None:
                continue

            days_diff = (current_cf.period_end_date - candidate.period_end_date).days

            # Prior period should be 60-120 days before (roughly 2-4 months)
            # This handles slight variations in fiscal quarter lengths
            if 60 <= days_diff <= 120:
                diff_from_expected = abs(days_diff - 90)
                if diff_from_expected < best_diff:
                    best_diff = diff_from_expected
                    best_match = candidate

        return best_match

    result = []

    # Process 3-month (Q1) filings - use directly
    for cf in period_lookup[3]:
        period_end_str = cf.period_end_date.strftime("%b %d, %Y") if cf.period_end_date else "Unknown"
        calc_note = f"Direct from 3-month filing (period ending {period_end_str})"

        cf_copy = CashFlowData(
            operating_cash_flow=cf.operating_cash_flow,
            capital_expenditure=cf.capital_expenditure,
            period_start_date=cf.period_start_date,
            period_end_date=cf.period_end_date,
            filing=cf.filing,
            evidence=cf.evidence,
            extraction_notes=calc_note,
            ocf_unit=cf.ocf_unit,
            capex_unit=cf.capex_unit,
            period_months=3
        )
        result.append(cf_copy)
        logger.info(f"Q1: Direct 3-month OCF = ${cf.operating_cash_flow:,.0f} ({period_end_str})" if cf.operating_cash_flow else f"Q1: No OCF ({period_end_str})")

    # Process 6-month YTD filings - calculate Q2
    for cf in period_lookup[6]:
        period_end_str = cf.period_end_date.strftime("%b %d, %Y") if cf.period_end_date else "Unknown"
        prior_cf = find_prior_period(cf, prior_months=3)

        if prior_cf and prior_cf.operating_cash_flow is not None:
            prev_ocf = prior_cf.operating_cash_flow
            prev_capex = prior_cf.capital_expenditure or 0
            prev_date = prior_cf.period_end_date.strftime("%b %d, %Y") if prior_cf.period_end_date else "Unknown"

            q2_ocf = (cf.operating_cash_flow or 0) - prev_ocf if cf.operating_cash_flow is not None else None
            q2_capex = (cf.capital_expenditure or 0) - prev_capex if cf.capital_expenditure is not None else None

            ocf_6mo = cf.operating_cash_flow or 0
            capex_6mo = cf.capital_expenditure or 0
            calc_note = (
                f"Calculated Q2: OCF: 6mo YTD (${ocf_6mo/1e6:,.1f}M) - 3mo (${prev_ocf/1e6:,.1f}M) = ${q2_ocf/1e6:,.1f}M | "
                f"CapEx: 6mo YTD (${capex_6mo/1e6:,.1f}M) - 3mo (${prev_capex/1e6:,.1f}M) = ${q2_capex/1e6:,.1f}M | "
                f"Sources: 6mo from {period_end_str}, 3mo from {prev_date}"
            )

            # Calculate period start (approximately 3 months before period end)
            period_start = cf.period_end_date - timedelta(days=90) if cf.period_end_date else None

            quarterly_cf = CashFlowData(
                operating_cash_flow=q2_ocf,
                capital_expenditure=q2_capex,
                period_start_date=period_start,
                period_end_date=cf.period_end_date,
                filing=cf.filing,
                evidence=cf.evidence,
                extraction_notes=calc_note,
                ocf_unit=cf.ocf_unit,
                capex_unit=cf.capex_unit,
                period_months=3
            )
            result.append(quarterly_cf)
            logger.info(f"Q2: Calculated OCF = ${q2_ocf:,.0f} (6mo {period_end_str} - 3mo {prev_date})")
        else:
            logger.warning(f"Q2: Cannot calculate for {period_end_str} - no matching 3-month prior period found")

    # Process 9-month YTD filings - calculate Q3
    for cf in period_lookup[9]:
        period_end_str = cf.period_end_date.strftime("%b %d, %Y") if cf.period_end_date else "Unknown"
        prior_cf = find_prior_period(cf, prior_months=6)

        if prior_cf and prior_cf.operating_cash_flow is not None:
            prev_ocf = prior_cf.operating_cash_flow
            prev_capex = prior_cf.capital_expenditure or 0
            prev_date = prior_cf.period_end_date.strftime("%b %d, %Y") if prior_cf.period_end_date else "Unknown"

            q3_ocf = (cf.operating_cash_flow or 0) - prev_ocf if cf.operating_cash_flow is not None else None
            q3_capex = (cf.capital_expenditure or 0) - prev_capex if cf.capital_expenditure is not None else None

            ocf_9mo = cf.operating_cash_flow or 0
            capex_9mo = cf.capital_expenditure or 0
            calc_note = (
                f"Calculated Q3: OCF: 9mo YTD (${ocf_9mo/1e6:,.1f}M) - 6mo YTD (${prev_ocf/1e6:,.1f}M) = ${q3_ocf/1e6:,.1f}M | "
                f"CapEx: 9mo YTD (${capex_9mo/1e6:,.1f}M) - 6mo YTD (${prev_capex/1e6:,.1f}M) = ${q3_capex/1e6:,.1f}M | "
                f"Sources: 9mo from {period_end_str}, 6mo from {prev_date}"
            )

            period_start = cf.period_end_date - timedelta(days=90) if cf.period_end_date else None

            quarterly_cf = CashFlowData(
                operating_cash_flow=q3_ocf,
                capital_expenditure=q3_capex,
                period_start_date=period_start,
                period_end_date=cf.period_end_date,
                filing=cf.filing,
                evidence=cf.evidence,
                extraction_notes=calc_note,
                ocf_unit=cf.ocf_unit,
                capex_unit=cf.capex_unit,
                period_months=3
            )
            result.append(quarterly_cf)
            logger.info(f"Q3: Calculated OCF = ${q3_ocf:,.0f} (9mo {period_end_str} - 6mo {prev_date})")
        else:
            logger.warning(f"Q3: Cannot calculate for {period_end_str} - no matching 6-month prior period found")

    # Process 12-month annual filings (10-K) - calculate Q4
    for cf in period_lookup[12]:
        period_end_str = cf.period_end_date.strftime("%b %d, %Y") if cf.period_end_date else "Unknown"

        if cf.operating_cash_flow is None:
            logger.warning(f"Q4: Cannot calculate for {period_end_str} - 10-K OCF data not available")
            continue

        prior_cf = find_prior_period(cf, prior_months=9)

        if prior_cf and prior_cf.operating_cash_flow is not None:
            prev_ocf = prior_cf.operating_cash_flow
            prev_capex = prior_cf.capital_expenditure or 0
            prev_date = prior_cf.period_end_date.strftime("%b %d, %Y") if prior_cf.period_end_date else "Unknown"

            q4_ocf = cf.operating_cash_flow - prev_ocf
            q4_capex = (cf.capital_expenditure or 0) - prev_capex if cf.capital_expenditure is not None else None

            ocf_12mo = cf.operating_cash_flow
            capex_12mo = cf.capital_expenditure or 0
            calc_note = (
                f"Calculated Q4: OCF: 12mo Annual (${ocf_12mo/1e6:,.1f}M) - 9mo YTD (${prev_ocf/1e6:,.1f}M) = ${q4_ocf/1e6:,.1f}M | "
                f"CapEx: 12mo Annual (${capex_12mo/1e6:,.1f}M) - 9mo YTD (${prev_capex/1e6:,.1f}M) = ${q4_capex/1e6:,.1f}M | "
                f"Sources: 12mo from {period_end_str} (10-K), 9mo from {prev_date}"
            )

            period_start = cf.period_end_date - timedelta(days=90) if cf.period_end_date else None

            quarterly_cf = CashFlowData(
                operating_cash_flow=q4_ocf,
                capital_expenditure=q4_capex,
                period_start_date=period_start,
                period_end_date=cf.period_end_date,
                filing=cf.filing,
                evidence=cf.evidence,
                extraction_notes=calc_note,
                ocf_unit=cf.ocf_unit,
                capex_unit=cf.capex_unit,
                period_months=3
            )
            result.append(quarterly_cf)
            logger.info(f"Q4: Calculated OCF = ${q4_ocf:,.0f} (12mo {period_end_str} - 9mo {prev_date})")
        else:
            logger.warning(f"Q4: Cannot calculate for {period_end_str} - no matching 9-month prior period found")

    # Sort result by period_end_date (newest first for display)
    result.sort(key=lambda x: x.period_end_date or date.min, reverse=True)

    return result


class BurnCalculator:
    """
    Quarterly burn rate calculator with selection and override support.
    """

    def __init__(self):
        pass

    def create_quarterly_fcf(self, cash_flow: CashFlowData) -> Optional[QuarterlyFCF]:
        """
        Create a QuarterlyFCF object from CashFlowData.

        Args:
            cash_flow: Extracted cash flow data for a quarter

        Returns:
            QuarterlyFCF object or None if insufficient data
        """
        if cash_flow is None:
            return None

        # Determine quarter label from period end date
        period_end = cash_flow.period_end_date
        if period_end is None and cash_flow.filing:
            period_end = cash_flow.filing.period_of_report

        if period_end is None:
            logger.warning("No period date available for FCF calculation")
            return None

        # Determine fiscal quarter
        quarter = self._get_fiscal_quarter(period_end)
        year = period_end.year
        quarter_label = f"Q{quarter} {year}"

        # Calculate FCF
        ocf = cash_flow.operating_cash_flow
        capex = cash_flow.capital_expenditure  # Already stored as positive

        fcf = None
        if ocf is not None and capex is not None:
            fcf = ocf - capex
        elif ocf is not None:
            fcf = ocf  # If no CapEx, FCF = OCF

        return QuarterlyFCF(
            quarter_label=quarter_label,
            fiscal_quarter=quarter,
            fiscal_year=year,
            period_start=cash_flow.period_start_date,
            period_end=period_end,
            operating_cash_flow=ocf,
            capital_expenditure=capex,
            free_cash_flow=fcf,
            is_selected=True,  # Default to selected
            filing=cash_flow.filing,
            evidence=cash_flow.evidence,
            source_unit=cash_flow.ocf_unit,  # Pass source unit from filing
            extraction_notes=cash_flow.extraction_notes  # Pass calculation notes
        )

    def _get_fiscal_quarter(self, period_end: date) -> int:
        """
        Determine fiscal quarter from period end date.
        Assumes calendar year fiscal quarters.
        """
        month = period_end.month
        if month <= 3:
            return 1
        elif month <= 6:
            return 2
        elif month <= 9:
            return 3
        else:
            return 4

    def calculate_average_burn(
        self,
        quarterly_fcf_list: List[QuarterlyFCF],
        selected_indices: Optional[List[int]] = None,
        manual_override: Optional[float] = None,
        cash_position: Optional[CashPosition] = None
    ) -> BurnMetrics:
        """
        Calculate average quarterly burn rate.

        Args:
            quarterly_fcf_list: List of QuarterlyFCF data
            selected_indices: Indices of quarters to include (None = all selected)
            manual_override: If provided, use this value instead of calculated average
            cash_position: Current cash position for runway calculation

        Returns:
            BurnMetrics with average burn and runway analysis
        """
        if not quarterly_fcf_list:
            return BurnMetrics(
                quarters_analyzed=[],
                quarters_included=[],
                average_quarterly_burn=0,
                is_manual_override=manual_override is not None,
                manual_override_value=manual_override,
                cash_position=cash_position,
                runway_quarters=None,
                runway_end_date=None
            )

        # Determine which quarters are selected
        if selected_indices is not None:
            # Update selection based on provided indices
            for i, qfcf in enumerate(quarterly_fcf_list):
                qfcf.is_selected = i in selected_indices
        else:
            # Use existing is_selected flags
            pass

        # Get selected quarters
        selected_quarters = [q for q in quarterly_fcf_list if q.is_selected]
        selected_labels = [q.quarter_label for q in selected_quarters]

        # Calculate average burn
        if manual_override is not None:
            avg_burn = manual_override
            is_override = True
            logger.info(f"Using manual override for quarterly burn: ${manual_override:,.0f}")
        else:
            # Calculate from selected quarters
            fcf_values = [q.free_cash_flow for q in selected_quarters if q.free_cash_flow is not None]

            if not fcf_values:
                avg_burn = 0
            else:
                avg_burn = sum(fcf_values) / len(fcf_values)

            is_override = False
            logger.info(f"Calculated average FCF from {len(fcf_values)} quarters: ${avg_burn:,.0f}")

        # Calculate runway if we have cash position and burn is negative (using cash)
        runway_quarters = None
        runway_end_date = None

        if cash_position and avg_burn < 0:
            # Burn is negative (using cash), calculate runway
            runway_quarters = cash_position.total_cash_at_hand / abs(avg_burn)

            # Calculate end date
            if cash_position.as_of_date:
                months_runway = runway_quarters * 3  # Quarters to months
                runway_end_date = cash_position.as_of_date + timedelta(days=int(months_runway * 30))

            logger.info(f"Runway: {runway_quarters:.1f} quarters ({runway_end_date})")
        elif cash_position and avg_burn >= 0:
            # Company is cash flow positive
            runway_quarters = float('inf')
            logger.info("Company is cash flow positive - infinite runway")

        # Get source_unit from quarters for consistent formatting
        source_unit = None
        for q in quarterly_fcf_list:
            if q.source_unit is not None:
                source_unit = q.source_unit
                break

        return BurnMetrics(
            quarters_analyzed=quarterly_fcf_list,
            quarters_included=selected_labels,
            average_quarterly_burn=avg_burn,
            is_manual_override=is_override,
            manual_override_value=manual_override,
            cash_position=cash_position,
            runway_quarters=runway_quarters,
            runway_end_date=runway_end_date,
            source_unit=source_unit
        )

    def update_selection(
        self,
        quarterly_fcf_list: List[QuarterlyFCF],
        quarter_label: str,
        is_selected: bool
    ) -> List[QuarterlyFCF]:
        """
        Update selection status for a specific quarter.

        Args:
            quarterly_fcf_list: List of QuarterlyFCF data
            quarter_label: Label of quarter to update (e.g., "Q3 2025")
            is_selected: New selection status

        Returns:
            Updated list
        """
        for qfcf in quarterly_fcf_list:
            if qfcf.quarter_label == quarter_label:
                qfcf.is_selected = is_selected
                logger.info(f"Updated {quarter_label} selection to {is_selected}")
                break

        return quarterly_fcf_list


def calculate_burn_metrics(
    cash_flows: List[CashFlowData],
    selected_indices: Optional[List[int]] = None,
    manual_override: Optional[float] = None,
    cash_position: Optional[CashPosition] = None
) -> BurnMetrics:
    """
    Convenience function to calculate burn metrics from cash flow data.

    Args:
        cash_flows: List of CashFlowData from 10-Q filings
        selected_indices: Indices of quarters to include
        manual_override: Manual quarterly burn override
        cash_position: Current cash position

    Returns:
        BurnMetrics
    """
    calculator = BurnCalculator()

    # Convert to QuarterlyFCF
    quarterly_fcf = []
    for cf in cash_flows:
        qfcf = calculator.create_quarterly_fcf(cf)
        if qfcf:
            quarterly_fcf.append(qfcf)

    # Calculate metrics
    return calculator.calculate_average_burn(
        quarterly_fcf,
        selected_indices=selected_indices,
        manual_override=manual_override,
        cash_position=cash_position
    )
