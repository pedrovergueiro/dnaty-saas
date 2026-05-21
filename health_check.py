"""
Smoke-test the deployed API.
Usage:
    python health_check.py                        # local (default)
    python health_check.py https://your-app.up.railway.app
"""
import sys
import urllib.request
import urllib.error
import json

BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"

CHECKS = [
    ("GET", "/health",   200, lambda b: b.get("status") == "ok"),
    ("GET", "/docs",     200, None),
    ("GET", "/api/v1/status/nonexistent-id", 404, None),
]


def request(method: str, path: str):
    req = urllib.request.Request(BASE_URL + path, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read()) if resp.headers.get_content_type() == "application/json" else {}
            return resp.status, body
    except urllib.error.HTTPError as e:
        return e.code, {}


def main():
    print(f"Target: {BASE_URL}\n")
    passed = failed = 0

    for method, path, expected_status, validator in CHECKS:
        status, body = request(method, path)
        ok = status == expected_status and (validator is None or validator(body))
        icon = "PASS" if ok else "FAIL"
        print(f"  [{icon}]  {method} {path}  →  {status} (expected {expected_status})")
        if ok:
            passed += 1
        else:
            failed += 1
            if body:
                print(f"         body: {body}")

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
