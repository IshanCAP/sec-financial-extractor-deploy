"""
AI-powered FDSO (Fully Diluted Shares Outstanding) analyzer.
Uses OpenAI GPT to identify and analyze all dilutive securities from 10-K filings.
"""

import json
import logging
import os
import time
from typing import Dict, Any, Optional
from pathlib import Path
import openai

try:
    from .models import Filing
except ImportError:
    from models import Filing

logger = logging.getLogger(__name__)

# Path to prompts directory
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def load_prompt(prompt_name: str) -> str:
    """Load a prompt from the prompts directory."""
    prompt_path = PROMPTS_DIR / f"{prompt_name}.txt"
    if prompt_path.exists():
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        logger.warning(f"Prompt file not found: {prompt_path}")
        return ""


def _clean_llm_json_response(response_text: str) -> str:
    """
    Clean LLM response to extract valid JSON.
    Handles markdown code blocks that GPT sometimes adds.
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

    # Also handle ```json at the start
    if cleaned.startswith('json'):
        cleaned = cleaned[4:]

    return cleaned.strip()


class FDSOAIAnalyzer:
    """
    AI-powered analyzer for Fully Diluted Shares Outstanding.

    Uses OpenAI GPT models to independently analyze 10-K filings and identify
    all dilutive securities with vested/unvested breakdowns.
    """

    def __init__(self, api_key: str = None, model: str = "gpt-5.2"):
        """
        Initialize the FDSO analyzer.

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-5.2, supports: gpt-5.2, gpt-4o, gpt-4-turbo, gpt-4)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None

        if self.api_key:
            try:
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info(f"FDSO Analyzer initialized with model {model}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        else:
            logger.warning("No OPENAI_API_KEY found, FDSO analysis unavailable")

    def is_available(self) -> bool:
        """Check if analysis is available."""
        return self.client is not None

    def analyze_10k(
        self,
        filing_content: str,
        filing_url: str,
        ticker: str,
        filing_date: str = None,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Analyze a 10-K filing to identify all dilutive securities.

        Args:
            filing_content: Plain text content of the 10-K filing
            filing_url: URL to the filing on SEC EDGAR
            ticker: Company ticker symbol
            filing_date: Filing date (YYYY-MM-DD format)
            max_retries: Number of retry attempts on failure

        Returns:
            Dict with:
            - success: bool
            - data: Parsed FDSO analysis (if successful)
            - raw_response: Raw GPT response
            - error: Error message (if failed)
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "OpenAI API key not configured"
            }

        # Load prompt template
        prompt_template = load_prompt("fdso_ai_analysis_prompt")
        if not prompt_template:
            return {
                "success": False,
                "error": "FDSO analysis prompt not found"
            }

        # Send the full content to GPT-5.2 (no truncation)
        logger.info(f"Sending full 10-K content: {len(filing_content):,} characters")

        # Build prompt
        prompt = prompt_template.format(
            ticker=ticker,
            filing_date=filing_date or "N/A",
            filing_url=filing_url,
            filing_content=filing_content
        )

        for attempt in range(max_retries + 1):
            try:
                logger.info(f"FDSO analysis attempt {attempt + 1}/{max_retries + 1}")
                logger.info(f"Sending {len(filing_content):,} chars to OpenAI ({self.model})...")

                # API call
                system_message = """You are an expert financial analyst specializing in dilutive securities analysis.
Your task is to comprehensively analyze 10-K filings and identify ALL securities that could dilute common shareholders.

Key principles:
1. Be thorough - examine the entire document
2. Look for tables in equity compensation notes (typically Note 10-15)
3. Distinguish between vested (currently dilutive) and unvested (potentially dilutive)
4. Provide evidence snippets for all findings
5. Return only valid JSON, no markdown formatting"""

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=8000,  # Allow longer response for comprehensive analysis
                    temperature=0.1,  # Low temperature for factual extraction
                    timeout=120  # 2 minute timeout for large documents
                )

                raw_response = response.choices[0].message.content
                logger.info(f"Got response from OpenAI ({len(raw_response) if raw_response else 0} chars)")

                if not raw_response:
                    if attempt < max_retries:
                        time.sleep(2)
                        continue
                    return {"success": False, "error": "Empty response from API"}

                # Clean and parse JSON
                cleaned = _clean_llm_json_response(raw_response)
                data = json.loads(cleaned)

                # Validate response structure
                if not isinstance(data, dict):
                    raise ValueError("Response is not a JSON object")

                # Log summary
                if data.get("summary"):
                    logger.info(f"FDSO Analysis complete: {data['summary'].get('notes', 'Done')}")

                return {
                    "success": True,
                    "data": data,
                    "raw_response": raw_response
                }

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                return {"success": False, "error": f"Failed to parse AI response: {e}"}

            except openai.APIConnectionError as e:
                logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    wait_time = (attempt + 1) * 3
                    time.sleep(wait_time)
                    continue
                return {"success": False, "error": "Could not reach OpenAI API"}

            except openai.RateLimitError as e:
                logger.warning(f"Rate limit error on attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    wait_time = (attempt + 1) * 5
                    time.sleep(wait_time)
                    continue
                return {"success": False, "error": "Rate limit exceeded"}

            except openai.AuthenticationError as e:
                logger.error(f"Authentication error: {e}")
                return {"success": False, "error": "Invalid API key"}

            except Exception as e:
                logger.error(f"FDSO analysis error: {e}")
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                return {"success": False, "error": str(e)}

        return {"success": False, "error": "Max retries exceeded"}

    def analyze_10k_from_url(
        self,
        filing_url: str,
        ticker: str,
        filing_date: str = None,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Analyze a 10-K filing by URL - let the AI access it independently.

        Args:
            filing_url: URL to the filing on SEC EDGAR
            ticker: Company ticker symbol
            filing_date: Filing date (YYYY-MM-DD format)
            max_retries: Number of retry attempts on failure

        Returns:
            Dict with:
            - success: bool
            - data: Parsed FDSO analysis (if successful)
            - raw_response: Raw GPT response
            - error: Error message (if failed)
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "OpenAI API key not configured"
            }

        # Load prompt template
        prompt_template = load_prompt("fdso_ai_analysis_prompt")
        if not prompt_template:
            return {
                "success": False,
                "error": "FDSO analysis prompt not found"
            }

        # Build prompt with URL only
        prompt = prompt_template.format(
            filing_url=filing_url,
            ticker=ticker,
            filing_date=filing_date or "N/A"
        )

        for attempt in range(max_retries + 1):
            try:
                logger.info(f"FDSO URL analysis attempt {attempt + 1}/{max_retries + 1}")
                logger.info(f"Asking {self.model} to independently access: {filing_url}")

                # API call - ask AI to access the URL
                system_message = """You are an expert financial analyst specializing in dilutive securities analysis with web search capabilities.

Your task is to:
1. Use web search to access and retrieve the 10-K filing from the provided URL
2. If the direct URL doesn't work, search the web for the company's 10-K filing
3. Read and analyze the ENTIRE 10-K document thoroughly
4. Identify ALL dilutive securities (stock options, RSUs, PSUs, warrants, convertible debt, etc.)
5. Distinguish between vested (currently dilutive) and unvested (potentially dilutive) securities
6. Extract evidence snippets for every finding
7. Return your analysis in valid JSON format (no markdown)

You MUST use your web browsing/search capability to access the filing. Make your own independent determination about what securities exist in the document."""

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=8000,
                    temperature=0.1,
                    timeout=180  # 3 minute timeout for URL access + analysis
                )

                raw_response = response.choices[0].message.content
                logger.info(f"Got response from OpenAI ({len(raw_response) if raw_response else 0} chars)")

                if not raw_response:
                    if attempt < max_retries:
                        time.sleep(2)
                        continue
                    return {"success": False, "error": "Empty response from API"}

                # Clean and parse JSON
                cleaned = _clean_llm_json_response(raw_response)
                data = json.loads(cleaned)

                # Validate response structure
                if not isinstance(data, dict):
                    raise ValueError("Response is not a JSON object")

                # Log summary
                if data.get("summary"):
                    logger.info(f"FDSO Analysis complete: {data['summary'].get('notes', 'Done')}")

                return {
                    "success": True,
                    "data": data,
                    "raw_response": raw_response
                }

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                return {"success": False, "error": f"Failed to parse AI response: {e}"}

            except openai.APIConnectionError as e:
                logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    wait_time = (attempt + 1) * 3
                    time.sleep(wait_time)
                    continue
                return {"success": False, "error": "Could not reach OpenAI API"}

            except openai.RateLimitError as e:
                logger.warning(f"Rate limit error on attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    wait_time = (attempt + 1) * 5
                    time.sleep(wait_time)
                    continue
                return {"success": False, "error": "Rate limit exceeded"}

            except openai.AuthenticationError as e:
                logger.error(f"Authentication error: {e}")
                return {"success": False, "error": "Invalid API key"}

            except Exception as e:
                logger.error(f"FDSO URL analysis error: {e}")
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                return {"success": False, "error": str(e)}

        return {"success": False, "error": "Max retries exceeded"}


def analyze_10k_fdso(
    filing_content: str,
    filing_url: str,
    ticker: str,
    filing_date: str = None,
    api_key: str = None,
    model: str = "gpt-4o"
) -> Dict[str, Any]:
    """
    Convenience function to analyze 10-K for FDSO.

    Args:
        filing_content: Plain text content of 10-K
        filing_url: URL to filing
        ticker: Company ticker
        filing_date: Filing date
        api_key: OpenAI API key
        model: Model to use (gpt-4o, gpt-5.2-pro, etc.)

    Returns:
        Analysis results dict
    """
    analyzer = FDSOAIAnalyzer(api_key=api_key, model=model)
    return analyzer.analyze_10k(
        filing_content=filing_content,
        filing_url=filing_url,
        ticker=ticker,
        filing_date=filing_date
    )
