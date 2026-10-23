#!/usr/bin/env python3
"""Smoke test script for verifying API availability, health, documents, and query endpoints.

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


def check_get_endpoint(url: str, description: str) -> dict:
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
        print(
            f"[FAIL] {description} ({url}) failed with HTTP {e.code}: {e.reason}", file=sys.stderr
        )
        raise
    except urllib.error.URLError as e:
        print(f"[FAIL] {description} ({url}) could not connect: {e.reason}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"[FAIL] {description} ({url}) unexpected error: {e}", file=sys.stderr)
        raise


def check_post_endpoint(url: str, payload: dict, description: str) -> dict:
    """Perform HTTP POST request and return parsed JSON or raise error."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "User-Agent": "DL-CP-SmokeTest/1.0",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            status_code = response.getcode()
            body = response.read().decode("utf-8")
            data = json.loads(body)
            print(f"[OK] {description} ({url}) returned HTTP {status_code}")
            return data
    except urllib.error.HTTPError as e:
        print(
            f"[FAIL] {description} ({url}) failed with HTTP {e.code}: {e.reason}", file=sys.stderr
        )
        raise
    except urllib.error.URLError as e:
        print(f"[FAIL] {description} ({url}) could not connect: {e.reason}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"[FAIL] {description} ({url}) unexpected error: {e}", file=sys.stderr)
        raise


def run_smoke_test(base_url: str) -> bool:
    """Execute smoke tests against root, /health, /documents, and /query endpoints."""
    base_url = base_url.rstrip("/")
    print("=" * 65)
    print(f"Starting Multimodal RAG API Smoke Test against: {base_url}")
    print("=" * 65)

    try:
        # 1. Check root endpoint
        root_data = check_get_endpoint(f"{base_url}/", "Root Service Info")
        app_name = root_data.get("app_name", "Unknown")
        version = root_data.get("version", "Unknown")
        print(f"     Service: {app_name} | Version: {version}")

        # 2. Check health endpoint
        health_data = check_get_endpoint(f"{base_url}/health", "Health Check")
        status_val = health_data.get("status")
        dependencies = health_data.get("dependencies", {})
        print(f"     Health Status: {status_val}")
        print("     Reported Dependencies:")
        for dep, state in dependencies.items():
            print(f"       - {dep}: {state}")

        if status_val != "ok":
            print(
                f"\n[ERROR] Health check status was '{status_val}', expected 'ok'", file=sys.stderr
            )
            return False

        # 3. Check documents listing endpoint
        docs_data = check_get_endpoint(f"{base_url}/documents", "Document Library")
        print(f"     Indexed Documents Count: {len(docs_data)}")

        # 4. Check query endpoint
        query_payload = {"query": "What are transformer attention layers?", "top_k": 3}
        query_data = check_post_endpoint(f"{base_url}/query", query_payload, "Research QA Query")
        timing = query_data.get("timing", {})
        print(f"     Query Status: Answer synthesized in {timing.get('total_ms', 0):.1f}ms")
        print(f"     Insufficient Evidence Flag: {query_data.get('insufficient_evidence')}")

        print("=" * 65)
        print("[SUCCESS] All smoke test assertions passed successfully!")
        print("=" * 65)
        return True

    except Exception as e:
        print("=" * 65)
        print(f"[FAILED] Smoke tests failed to complete: {e}", file=sys.stderr)
        print("Tip: Start the dev server in another terminal using: make dev", file=sys.stderr)
        print("=" * 65)
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
