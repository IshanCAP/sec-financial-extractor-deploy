#!/usr/bin/env python3
"""
Verification script for deployment package.
Run this to ensure all dependencies and files are properly configured.
"""

import sys
from pathlib import Path

def verify_deployment():
    """Verify deployment package is complete and functional."""
    print("[*] Verifying SEC Financial Extractor Deployment Package...\n")

    errors = []
    warnings = []

    # Check required files
    print("1. Checking required files...")
    required_files = [
        "app.py",
        "requirements.txt",
        "README.md",
        "DEPLOYMENT.md",
        ".gitignore",
        ".streamlit/config.toml",
    ]

    for file in required_files:
        if Path(file).exists():
            print(f"   [+] {file}")
        else:
            errors.append(f"Missing file: {file}")
            print(f"   [-] {file}")

    # Check lib directory
    print("\n2. Checking lib directory...")
    lib_files = [
        "lib/__init__.py",
        "lib/sec_client.py",
        "lib/extractor.py",
        "lib/cash_calculator.py",
        "lib/burn_calculator.py",
        "lib/models.py",
        "lib/fdso_ai_analyzer.py",
    ]

    for file in lib_files:
        if Path(file).exists():
            print(f"   [+] {file}")
        else:
            errors.append(f"Missing lib file: {file}")
            print(f"   [-] {file}")

    # Check prompts directory
    print("\n3. Checking prompts directory...")
    prompts_dir = Path("prompts")
    if prompts_dir.exists():
        prompt_files = list(prompts_dir.glob("*.txt"))
        print(f"   [+] Found {len(prompt_files)} prompt files")
        for pf in prompt_files:
            print(f"      - {pf.name}")
    else:
        errors.append("Missing prompts directory")
        print("   [-] prompts directory not found")

    # Check Python imports
    print("\n4. Testing Python imports...")
    try:
        from lib.sec_client import SECClient
        print("   [+] lib.sec_client")
    except ImportError as e:
        errors.append(f"Import error: lib.sec_client - {e}")
        print(f"   [-] lib.sec_client - {e}")

    try:
        from lib.extractor import FinancialExtractor
        print("   [+] lib.extractor")
    except ImportError as e:
        errors.append(f"Import error: lib.extractor - {e}")
        print(f"   [-] lib.extractor - {e}")

    try:
        from lib.cash_calculator import calculate_cash_position
        print("   [+] lib.cash_calculator")
    except ImportError as e:
        errors.append(f"Import error: lib.cash_calculator - {e}")
        print(f"   [-] lib.cash_calculator - {e}")

    try:
        from lib.burn_calculator import BurnCalculator
        print("   [+] lib.burn_calculator")
    except ImportError as e:
        errors.append(f"Import error: lib.burn_calculator - {e}")
        print(f"   [-] lib.burn_calculator - {e}")

    try:
        from lib.models import format_currency
        print("   [+] lib.models")
    except ImportError as e:
        errors.append(f"Import error: lib.models - {e}")
        print(f"   [-] lib.models - {e}")

    try:
        from lib.fdso_ai_analyzer import FDSOAIAnalyzer
        print("   [+] lib.fdso_ai_analyzer")
    except ImportError as e:
        errors.append(f"Import error: lib.fdso_ai_analyzer - {e}")
        print(f"   [-] lib.fdso_ai_analyzer - {e}")

    # Check for development artifacts
    print("\n5. Checking for development artifacts...")
    unwanted_patterns = [
        "test_*.py",
        "debug_*.py",
        "__pycache__",
        "*.pyc",
        ".pytest_cache",
        "venv",
        ".venv",
    ]

    found_artifacts = []
    for pattern in unwanted_patterns:
        matches = list(Path(".").rglob(pattern))
        if matches:
            found_artifacts.extend(matches)

    if found_artifacts:
        warnings.append("Found development artifacts (should be cleaned)")
        print(f"   [!] Found {len(found_artifacts)} development artifacts")
        for artifact in found_artifacts[:5]:  # Show first 5
            print(f"      - {artifact}")
    else:
        print("   [+] No development artifacts found")

    # Summary
    print("\n" + "="*60)
    if errors:
        print(f"[FAIL] VERIFICATION FAILED with {len(errors)} error(s):")
        for error in errors:
            print(f"   - {error}")
        return False
    elif warnings:
        print(f"[WARN]  VERIFICATION PASSED with {len(warnings)} warning(s):")
        for warning in warnings:
            print(f"   - {warning}")
        print("\n[OK] Deployment package is functional but could be improved")
        return True
    else:
        print("[OK] VERIFICATION PASSED - Deployment package is ready!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run the app: streamlit run app.py")
        print("  3. Or push to GitHub and deploy to Streamlit Cloud")
        return True

if __name__ == "__main__":
    success = verify_deployment()
    sys.exit(0 if success else 1)
