#!/usr/bin/env python3
"""
Pre-flight Kaggle Environment & Credentials Doctor for Amazon ML Challenge 2026.

Verifies:
1. Local kaggle.json presence or KAGGLE_USERNAME / KAGGLE_KEY environment variables.
2. kaggle Python package and CLI accessibility.
3. Authenticated ping to Kaggle API without exposing raw credentials.
4. Actionable setup tips if credentials or CLI are missing.
"""

import os
import sys
import json
import argparse
from pathlib import Path


def mask_string(val: str, show_chars: int = 3) -> str:
    """Mask sensitive string leaving only initial characters visible."""
    if not val:
        return "Not Set"
    if len(val) <= show_chars:
        return "***"
    return val[:show_chars] + "*" * (len(val) - show_chars)


def check_kaggle_credentials() -> dict:
    """Check credentials file or environment variables."""
    home_kaggle = Path.home() / ".kaggle" / "kaggle.json"
    env_username = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")

    has_file = home_kaggle.is_file()
    has_env = bool(env_username and env_key)
    username = None

    if has_file:
        try:
            with open(home_kaggle, "r", encoding="utf-8") as f:
                data = json.load(f)
                username = data.get("username")
        except Exception:
            pass
    elif has_env:
        username = env_username

    return {
        "has_credentials_file": has_file,
        "credentials_path": str(home_kaggle) if has_file else None,
        "has_env_credentials": has_env,
        "username": username,
        "masked_username": mask_string(username, 4) if username else None
    }


def verify_kaggle_api() -> dict:
    """Attempt live authentication ping with Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        # Ping API by checking active username
        user = api.get_config_value("username") or api.config_values.get("username")
        return {
            "api_installed": True,
            "authenticated": True,
            "api_user": user,
            "error": None
        }
    except ImportError:
        return {
            "api_installed": False,
            "authenticated": False,
            "api_user": None,
            "error": "The 'kaggle' Python package is not installed."
        }
    except Exception as e:
        return {
            "api_installed": True,
            "authenticated": False,
            "api_user": None,
            "error": str(e)
        }


def run_doctor(check_only: bool = False, json_mode: bool = False) -> int:
    cred_info = check_kaggle_credentials()
    api_info = verify_kaggle_api()

    success = cred_info["has_credentials_file"] or cred_info["has_env_credentials"]
    api_success = api_info["authenticated"]

    report = {
        "status": "PASS" if (success and api_success) else "FAIL",
        "credentials": cred_info,
        "api": api_info
    }

    if json_mode:
        print(json.dumps(report, indent=2))
        return 0 if (success and api_success) or check_only else 1

    print("=======================================================")
    print("   AMAZON ML CHALLENGE 2026: KAGGLE PRE-FLIGHT DOCTOR  ")
    print("=======================================================")

    if cred_info["has_credentials_file"]:
        print(f"[PASS] Kaggle credentials file detected: {cred_info['credentials_path']}")
        print(f"       Account Username: {cred_info['masked_username']}")
    elif cred_info["has_env_credentials"]:
        print(f"[PASS] Kaggle credentials detected from environment variables")
        print(f"       Account Username: {cred_info['masked_username']}")
    else:
        print("[FAIL] No Kaggle credentials found in ~/.kaggle/kaggle.json or env vars.")

    if api_info["authenticated"]:
        print("[PASS] Kaggle API authentication succeeded.")
    else:
        print(f"[FAIL] Kaggle API authentication failed: {api_info['error']}")

    if not (success and api_success):
        print("\n--- Troubleshooting Guide ---")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Scroll down to the 'API' section.")
        print("3. Click 'Create New Token' to download kaggle.json.")
        print(f"4. Move the downloaded kaggle.json to: {Path.home() / '.kaggle' / 'kaggle.json'}")
        print("5. Or export environment variables: KAGGLE_USERNAME and KAGGLE_KEY.")
        print("6. Run 'pip install kaggle' if the CLI is not yet installed.")
        print("=======================================================\n")
        return 0 if check_only else 1

    print("=======================================================")
    print(" [READY] Remote Kaggle execution pipeline is authenticated!")
    print("=======================================================\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Kaggle Pre-Flight Credentials & API Doctor")
    parser.add_argument("--check-only", action="store_true", help="Report status without exiting with error code on failure")
    parser.add_argument("--json", action="store_true", help="Output diagnostic in JSON format")
    args = parser.parse_args()

    sys.exit(run_doctor(check_only=args.check_only, json_mode=args.json))


if __name__ == "__main__":
    main()
