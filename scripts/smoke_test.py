#!/usr/bin/env python3
"""Smoke test script for verifying API availability and health endpoint.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --url http://127.0.0.1:8000
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def check_endpoint(url: str, description: str) -> dict:
    """Perform HTTP GET request and return parsed JSON or raise error."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "DL-CP-SmokeTest/1.0", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            status_code = response.getcode()
            body = response.read().decode("utf-8")
            data = json.loads(body)
            print(f"[OK] {description} ({url}) returned HTTP {status_code}")
            return data
    except urllib.error.HTTPError as e:
        print(f"[FAIL] {description} ({url}) failed with HTTP {e.code}: {e.reason}", file=sys.stderr)
        raise
    except urllib.error.URLError as e:
        print(f"[FAIL] {description} ({url}) could not connect: {e.reason}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"[FAIL] {description} ({url}) unexpected error: {e}", file=sys.stderr)
        raise


def run_smoke_test(base_url: str) -> bool:
    """Execute smoke tests against root and /health endpoints."""
    base_url = base_url.rstrip("/")
    print("=" * 60)
    print(f"Starting Multimodal RAG API Smoke Test against: {base_url}")
    print("=" * 60)

    try:
        # Check root endpoint
        root_data = check_endpoint(f"{base_url}/", "Root Service Info")
        app_name = root_data.get("app_name", "Unknown")
        version = root_data.get("version", "Unknown")
        print(f"     Service: {app_name} | Version: {version}")

        # Check health endpoint
        health_data = check_endpoint(f"{base_url}/health", "Health Check")
        status_val = health_data.get("status")
        dependencies = health_data.get("dependencies", {})
        print(f"     Health Status: {status_val}")
        print("     Reported Dependencies:")
        for dep, state in dependencies.items():
            print(f"       - {dep}: {state}")

        if status_val != "ok":
            print(f"\n[ERROR] Health check status was '{status_val}', expected 'ok'", file=sys.stderr)
            return False

        print("=" * 60)
        print("[SUCCESS] All smoke test assertions passed successfully!")
        print("=" * 60)
        return True

    except Exception:
        print("=" * 60)
        print("[FAILED] Smoke tests failed to complete. Is the API server running?", file=sys.stderr)
        print("Tip: Start the dev server in another terminal using: make dev", file=sys.stderr)
        print("=" * 60)
        return False


def main():
    default_host = os.getenv("API_HOST", "127.0.0.1")
    default_port = os.getenv("API_PORT", "8000")
    default_url = f"http://{default_host}:{default_port}"

    parser = argparse.ArgumentParser(description="Multimodal RAG API Smoke Test")
    parser.add_argument(
        "--url",
        default=default_url,
        help=f"Base URL for target API service (default: {default_url})",
    )
    args = parser.parse_args()

    success = run_smoke_test(args.url)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
