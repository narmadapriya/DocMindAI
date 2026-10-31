"""DocMindAI read-only source/API verification and live response-time audit.

Safe to add under tests/: NO edits to source, UI, RAG/LLM logic, schema, or
existing configuration. The script calls existing authenticated APIs against
seven existing user-owned, RAG-ready sample documents.

This measures backend HTTP round-trip only, NOT browser rendering. It does NOT
calculate RAGAS faithfulness, retrieval context recall, or context precision;
those metrics require labeled retrieved chunks and evaluation traces.

Usage (Windows PowerShell, from DocMindAI root):
    python tests/docmindai_live_accuracy_latency.py --runs 3 --output test_results.json

Login credentials are prompted and NEVER written to the output file.
"""
from __future__ import annotations

import argparse
import getpass
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FILES = {
    "pdf": "apex_motors_market_report.pdf",
    "docx": "apex_motors_annual_report.docx",
    "txt": "apex_motors.txt",
    "monthly_csv": "apex_motors_monthly_sales_2025.csv",
    "data_xlsx": "apex_motors_data.xlsx",
    "fy25_csv": "DocMindAI_Comparison_Test_A_FY2025.csv",
    "fy26_xlsx": "DocMindAI_Comparison_Test_B_FY2026.xlsx",
}

# Ground-truth checks are confined to this testing script, NEVER application code.
# Multiple regexes allow natural variations in currency/number formatting.
CHECKS = {
    "pdf_prices": {
        "apex_price": r"(?:38[,.]?100|38100)",
        "horizon_price": r"(?:41[,.]?500|41500)",
        "apex_csat": r"4[.]6",
        "horizon_csat": r"4[.]4",
        "price_gap": r"(?:3[,.]?400|3400)",
    },
    "docx_quarters": {
        "fy_revenue": r"(?:612[.]4|612[,.]?400[,.]?000)",
        "fy_units": r"16[,.]?070",
        "q1_revenue": r"(?:128[.]5|128[,.]?500[,.]?000)",
        "q1_units": r"3[,.]?420",
        "q4_revenue": r"(?:181[.]1|181[,.]?100[,.]?000)",
        "q4_units": r"4[,.]?610",
    },
    "txt_plans": {
        "sales_growth_target": r"12\s*%",
        "volt_x_capacity": r"3[,.]?000",
        "service_centers": r"(?:two|2)\s+(?:new\s+)?(?:regional\s+)?service\s+centers",
        "east_expansion": r"\beast\b",
        "customer_target": r"5[.]0",
    },
    "csv_monthly": {
        "ten_month_units": r"12[,.]?940",
        "ten_month_revenue": r"488[,.]?538[,.]?000|488[.]538\s*million",
        "highest_month": r"\boct(?:ober)?\b",
        "oct_revenue": r"57[,.]?128[,.]?000|57[.]128\s*million",
    },
    "xlsx_executive": {
        "ten_month_units": r"12[,.]?940",
        "ten_month_revenue": r"488[,.]?538[,.]?000|488[.]538\s*million",
        "volt_x_stock": r"\b265\b",
        "volt_x_production": r"\b310\b",
        "manufacturing_count": r"\b520\b",
        "workforce_context": r"\b(?:manufacturing|workforce)\b",
    },
}

SCENARIOS = [
    {
        "id": "pdf_prices", "kind": "chat", "files": ["pdf"],
        "query": "What were the average transaction prices and customer satisfaction scores of Apex Motors and Horizon Motor Co. in FY2025, and what was their price difference?",
    },
    {
        "id": "docx_quarters", "kind": "chat", "files": ["docx"],
        "query": "What were Apex Motors' total FY2025 revenue and vehicles sold, and how did Q4 performance compare with Q1? Include both Q1 and Q4 figures.",
    },
    {
        "id": "txt_plans", "kind": "chat", "files": ["txt"],
        "query": "What are Apex Motors' major FY2026 growth targets and expansion plans, including EV production, service centers, and customer satisfaction?",
    },
    {
        "id": "csv_monthly", "kind": "chat", "files": ["monthly_csv"],
        "query": "Calculate total vehicles sold and total revenue from January to October 2025. Which month recorded the highest revenue and how much?",
    },
    {
        "id": "xlsx_executive", "kind": "summary", "files": ["data_xlsx"],
        "instructions": "Summarize FY2025 sales performance, January–October revenue and units, vehicle inventory, and workforce distribution. Include exact numerical figures and key findings.",
    },
    {
        "id": "pdf_docx_comparison", "kind": "compare", "files": ["pdf", "docx"],
        "metrics": ["Apex SUV Trailblazer FY2025 units sold", "Apex SUV Trailblazer starting price"],
        "expected": [("5640", "5640", "0"), ("41200", "41200", "0")],
    },
    {
        "id": "fy25_fy26_comparison", "kind": "compare", "files": ["fy25_csv", "fy26_xlsx"],
        "metrics": ["Revenue", "Total vehicles sold", "Operating margin", "EV share", "Regional service centers"],
        "expected": [
            ("612400000", "694800000", "+82400000"),
            ("16070", "17840", "+1770"),
            ("8.4", "9.1", "+0.7 pp"),
            ("11.5", "16.2", "+4.7 pp"),
            ("6", "8", "+2"),
        ],
    },
]


def http_json(base: str, path: str, *, token: str | None = None,
              data: dict[str, Any] | None = None, form: bool = False,
              timeout: float = 180) -> tuple[Any, float]:
    body = None
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if data is not None:
        if form:
            body = urllib.parse.urlencode(data).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
    request = urllib.request.Request(base.rstrip("/") + path, data=body, headers=headers)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        detail = exc.read(4000).decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code} on {path}: {detail[:600]}") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Connection/timeout on {path}: {exc}") from None
    elapsed = time.perf_counter() - started
    if status < 200 or status >= 300:
        raise RuntimeError(f"HTTP status {status} on {path}")
    try:
        return json.loads(raw.decode("utf-8")), elapsed
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError(f"Non-JSON response from {path}") from None


def select_documents(items: list[dict[str, Any]]) -> dict[str, str]:
    result = {}
    for key, name in FILES.items():
        matches = [doc for doc in items if str(doc.get("original_filename") or doc.get("filename") or "").casefold() == name.casefold()]
        if not matches:
            raise RuntimeError(f"RAG-ready file missing from uploaded list: {name}")
        matches.sort(key=lambda d: str(d.get("created_at") or ""), reverse=True)
        if len(matches) > 1:
            print(f"WARNING: {len(matches)} files named {name}; using most recently uploaded", file=sys.stderr)
        result[key] = str(matches[0]["id"])
    return result


def accuracy_check(scenario: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    key = scenario["id"]
    if scenario["kind"] != "compare":
        field = "summary" if scenario["kind"] == "summary" else "answer"
        text = str(response.get(field) or "")
        matches = {label: bool(re.search(pattern, text, re.IGNORECASE))
                   for label, pattern in CHECKS[key].items()}
        return {"covered_markers": matches,
                "marker_coverage": sum(matches.values()) / len(matches),
                "citations_present": bool(response.get("citations")),
                "note": "Lexical factual-marker coverage only; NOT faithfulness/context-precision proof."}
    rows = response.get("comparison") or []
    expected = scenario["expected"]
    checks = []
    for index, triplet in enumerate(expected):
        item = rows[index] if index < len(rows) else {}
        values = tuple(str(item.get(k, "")).strip() for k in ("document_a", "document_b", "change"))
        checks.append(values == triplet)
    return {"correct_rows": checks, "metric_row_coverage": sum(checks) / len(checks),
            "citations_present": bool(response.get("citations")),
            "note": "Strict ordered metric check; inspect citations/evidence manually."}


def do_scenario(base: str, token: str, user_id: str,
                doc_ids: dict[str, str], scenario: dict[str, Any], timeout: float):
    ids = [doc_ids[key] for key in scenario["files"]]
    common = {"user_id": user_id, "document_ids": ids}
    if scenario["kind"] == "chat":
        path = "/api/v1/chat"
        payload = {**common, "query": scenario["query"], "history": [], "top_k": 5}
    elif scenario["kind"] == "summary":
        path = "/api/v1/summary"
        payload = {**common, "scope": "executive", "scope_instructions": scenario["instructions"], "top_k": 8}
    else:
        path = "/api/v1/compare"
        payload = {**common, "metrics": scenario["metrics"], "top_k": 10}
    response, elapsed = http_json(base, path, token=token, data=payload, timeout=timeout)
    return response, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--runs", type=int, default=3, help="Measured rounds per scenario, default 3")
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured rounds before measurements, default 1")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, default=Path("docmindai_live_evaluation.json"))
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0:
        parser.error("--runs must be >= 1 and --warmup >= 0")

    username = input("DocMindAI login email: ").strip()
    password = getpass.getpass("DocMindAI password (hidden): ")
    login, _ = http_json(args.base_url, "/api/auth/login", data={"username": username, "password": password}, form=True, timeout=args.timeout)
    del password
    token = str(login.get("access_token") or "")
    if not token:
        raise RuntimeError("Login response did not provide access_token")
    user, _ = http_json(args.base_url, "/me", token=token, timeout=args.timeout)
    user_id = str(user["id"])
    docs, _ = http_json(args.base_url, "/api/upload/documents", token=token, timeout=args.timeout)
    doc_ids = select_documents(list(docs.get("documents") or []))
    del login

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": "October 26 frozen + locally installed October 31 comparison/summary agent replacements (verify your working copy)",
        "method": "authenticated existing backend HTTP APIs; localhost; one warm-up per test by default",
        "latency_scope": "HTTP request-to-response; excludes browser paint/render and model cold load if warmup used",
        "metric_limitations": "Automatic checks are expected-fact marker coverage, not measured faithfulness, context precision, or context recall",
        "runs_per_test": args.runs,
        "warmup_per_test": args.warmup,
        "results": [],
    }
    for scenario in SCENARIOS:
        result = {"id": scenario["id"], "module": scenario["kind"],
                  "files": [FILES[k] for k in scenario["files"]], "runs": []}
        print(f"Testing {scenario['id']}...", flush=True)
        for n in range(args.warmup + args.runs):
            is_warmup = n < args.warmup
            try:
                response, elapsed = do_scenario(args.base_url, token, user_id, doc_ids, scenario, args.timeout)
                check = accuracy_check(scenario, response)
                row = {"seconds": round(elapsed, 3), "status": "success", "validation": check,
                       "response": response}
                print(f"  {'warmup' if is_warmup else 'measured'}: {elapsed:.3f}s, coverage={check.get('marker_coverage', check.get('metric_row_coverage'))}")
            except Exception as exc:
                row = {"status": "failed", "error": str(exc)}
                print(f"  {'warmup' if is_warmup else 'measured'}: FAILED: {exc}", file=sys.stderr)
            if not is_warmup:
                result["runs"].append(row)
        samples = [x["seconds"] for x in result["runs"] if x.get("status") == "success"]
        result["successful_runs"] = len(samples)
        if samples:
            samples_sorted = sorted(samples)
            result["median_seconds"] = round(statistics.median(samples), 3)
            rank = max(0, (95 * len(samples) + 99) // 100 - 1)
            result["p95_nearest_rank_seconds"] = round(samples_sorted[rank], 3)
            result["all_successful_under_10s"] = len(samples) == args.runs and max(samples) <= 10.0
        report["results"].append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved full report: {args.output}")
    print("WARNING: This report contains test answers; keep it private. No credentials or tokens were saved.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("Cancelled", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
