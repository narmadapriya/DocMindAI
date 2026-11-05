r"""Read-only latency audit for the 10 previously passed DocMindAI tests.

Run from the DocMindAI project root:
    python .\tests\docmindai_10_performance_only.py --runs 3 --warmup 1

No application source, schema, documents or expected answers are changed
by the harness itself. It calls authenticated RAG endpoints; the Chat & Ask
endpoint may write ordinary conversation records as part of existing behavior. Responses are NOT re-rated for semantic accuracy.

NOTE: Chat & Ask may persist conversation history using existing app behavior.
The script itself has no direct DB write, and records no credentials or tokens.
"""
from __future__ import annotations

import argparse
import csv
import getpass
import json
import math
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

FILE_PROMPT = (
    "Generate a concise, accurate File Summary of the selected document. "
    "Include its overview, key topics, exact facts and figures, main findings, "
    "conclusion, and source citations. Do not invent missing details."
)
EXEC_PROMPT = (
    "Generate an Executive Summary containing an overview, objective and scope, "
    "key highlights, critical KPIs, findings, supported risks, recommendations "
    "if justified, conclusion, and source citations. Preserve all numerical "
    "values exactly."
)

# Answer keys are preserved for human verification; this script never changes
# them or uses them to adjust the results or introduce benchmark-specific answers.
TESTS = [
    {
        "id": "Q10", "module": "Chat & Ask", "kind": "chat",
        "files": ["monthly_csv"],
        "question": "Calculate total vehicles sold and total revenue from January to October 2025.",
        "expected": "12,940 vehicles; $488,538,000 revenue; cite CSV January-October rows.",
    },
    {
        "id": "FY25_CSV_File", "module": "Summaries - File", "kind": "summary",
        "files": ["fy25_csv"], "scope": "document", "question": FILE_PROMPT,
        "expected": "FY2025 revenue $612.4M, 16,070 vehicles, average price $38,100, operating margin 8.4%, CSAT 4.6/5, EV share 11.5%, 1,240 employees, 6 regional service centers; South 5,150 vehicles and East 3,180; cite CSV metric rows.",
    },
    {
        "id": "FY25_CSV_Executive", "module": "Summaries - Executive", "kind": "summary",
        "files": ["fy25_csv"], "scope": "executive", "question": EXEC_PROMPT,
        "expected": "FY2025 $612.4M revenue, 16,070 vehicles, operating margin 8.4%, EV share 11.5%, CSAT 4.6/5, 6 service centers, Trailblazer 5,640 and Volt-X 1,850 vehicles; cite CSV metric rows.",
    },
    {
        "id": "FY26_XLSX_File", "module": "Summaries - File", "kind": "summary",
        "files": ["fy26_xlsx"], "scope": "document", "question": FILE_PROMPT,
        "expected": "FY2026 revenue $694.8M, 17,840 vehicles, average price $38,950, margin 9.1%, CSAT 4.7/5, EV share 16.2%, 1,325 employees, 8 service centers; South 5,620 and East 3,810 vehicles; cite workbook sheets.",
    },
    {
        "id": "FY26_XLSX_Executive", "module": "Summaries - Executive", "kind": "summary",
        "files": ["fy26_xlsx"], "scope": "executive", "question": EXEC_PROMPT,
        "expected": "FY2026 $694.8M, 17,840 vehicles, 9.1% margin, 16.2% EV share, 4.7/5 CSAT, 8 service centers, 1,325 employees, Trailblazer 5,980, Volt-X 2,890, South 5,620 vehicles; cite comparison and regional sheets.",
    },
    {
        "id": "C2", "module": "Comparisons", "kind": "compare",
        "files": ["docx", "txt"],
        "question": "Compare FY2025 annual revenue, annual vehicles sold, Q1 revenue, and Q4 revenue.",
        "metrics": ["FY2025 annual revenue", "Annual vehicles sold", "Q1 revenue", "Q4 revenue"],
        "expected": "DOCX = TXT: $612.4M FY2025 revenue, 16,070 annual vehicles, Q1 $128.5M revenue, Q4 $181.1M revenue; differences all zero; cite both documents.",
    },
    {
        "id": "C3", "module": "Comparisons", "kind": "compare",
        "files": ["monthly_csv", "data_xlsx"],
        "question": "Compare January–October total vehicles sold and total revenue.",
        "metrics": ["January–October total vehicles sold", "January–October total revenue"],
        "expected": "CSV = XLSX: January–October total vehicles 12,940 and revenue $488,538,000; differences zero; cite monthly CSV rows and XLSX Sales Data.",
    },
    {
        "id": "C4", "module": "Comparisons", "kind": "compare",
        "files": ["txt", "data_xlsx"],
        "question": "Compare Volt-X current stock, units in production, and manufacturing workforce.",
        "metrics": ["Volt-X current stock", "Volt-X units in production", "Manufacturing workforce"],
        "expected": "TXT = XLSX: Volt-X current stock 265, in production 310, manufacturing workforce 520; zero difference; cite TXT sections 7–8 and XLSX Inventory & Workforce.",
    },
    {
        "id": "C6", "module": "Comparisons", "kind": "compare",
        "files": ["pdf", "txt"],
        "question": "Compare Apex Motors' and Horizon Motor Co.'s average transaction prices and CSAT scores across the reports.",
        "metrics": [
            "Apex Motors average transaction price", "Apex Motors CSAT",
            "Horizon Motor Co. average transaction price", "Horizon Motor Co. CSAT",
        ],
        "expected": "PDF = TXT: Apex $38,100 and 4.6/5 CSAT; Horizon $41,500 and 4.4/5 CSAT; differences zero between documents; cite PDF page 4 and TXT section 10.",
    },
    {
        "id": "C7", "module": "Comparisons", "kind": "compare",
        "files": ["pdf", "docx", "txt"],
        "question": "Do all three documents agree on the FY2025 Trailblazer starting price and sales volume?",
        "metrics": ["FY2025 Trailblazer starting price", "FY2025 Trailblazer units sold"],
        "expected": "PDF, DOCX and TXT all agree: FY2025 Trailblazer starting price $41,200; 5,640 vehicles sold; differences zero; cite PDF page 3, DOCX page 4, TXT section 6.",
    },
]


def request_json(base: str, path: str, *, token: str | None = None,
                 data: dict[str, Any] | None = None, form: bool = False,
                 timeout: float = 180.0) -> tuple[Any, float]:
    body = None
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    if data is not None:
        if form:
            body = urllib.parse.urlencode(data).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base.rstrip("/") + path, data=body, headers=headers)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = response.read()
            code = response.status
    except urllib.error.HTTPError as exc:
        detail = exc.read(1000).decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code} on {path}: {detail[:300]}") from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise RuntimeError(f"Connection/timeout on {path}: {exc}") from None
    elapsed = time.perf_counter() - start
    if not 200 <= code < 300:
        raise RuntimeError(f"HTTP {code} on {path}")
    try:
        return json.loads(payload.decode("utf-8")), elapsed
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError(f"Non-JSON response on {path}") from None


def select_documents(documents: list[dict[str, Any]]) -> dict[str, str]:
    ids = {}
    for key, filename in FILES.items():
        matches = [item for item in documents if
                   str(item.get("original_filename") or item.get("filename") or "").casefold()
                   == filename.casefold()]
        if not matches:
            raise RuntimeError(f"Missing selected document in upload list: {filename}")
        matches.sort(key=lambda d: str(d.get("created_at") or ""), reverse=True)
        if len(matches) > 1:
            print(f"WARNING: multiple documents named {filename}; using newest", file=sys.stderr)
        ids[key] = str(matches[0]["id"])
    return ids


def build_request(case: dict[str, Any], user_id: str,
                  doc_ids: dict[str, str]) -> tuple[str, dict[str, Any]]:
    common = {"user_id": user_id,
              "document_ids": [doc_ids[name] for name in case["files"]]}
    if case["kind"] == "chat":
        return "/api/v1/chat", {**common, "query": case["question"], "history": [], "top_k": 5}
    if case["kind"] == "summary":
        return "/api/v1/summary", {
            **common, "scope": case["scope"], "scope_instructions": case["question"], "top_k": 8,
        }
    return "/api/v1/compare", {**common, "metrics": case["metrics"], "top_k": 10}


def evaluate_response(response: Any, kind: str) -> tuple[bool, bool, str]:
    if not isinstance(response, dict):
        return False, False, "Non-object response"
    field = "summary" if kind == "summary" else "answer"
    answer = str(response.get(field) or "").strip()
    citations = bool(response.get("citations"))
    if not answer:
        return False, citations, "Empty answer"
    if any(term in answer.lower() for term in (
        "local model request failed", "i could not produce a sufficiently verified answer",
    )):
        return False, citations, "Model/error text returned in answer"
    return True, citations, "Unchanged answer key still needs manual confirmation"


def run_case(base: str, token: str, user_id: str, doc_ids: dict[str, str],
             case: dict[str, Any], runs: int, warmup: int,
             timeout: float, target: float) -> dict[str, Any]:
    path, payload = build_request(case, user_id, doc_ids)
    result = {
        "id": case["id"], "module": case["module"],
        "filenames": [FILES[name] for name in case["files"]],
        "original_question": case["question"],
        "expected_answer_unchanged": case["expected"],
        "api_metric_fields": case.get("metrics"),
        "measured": [], "warmup": [],
    }
    for index in range(warmup + runs):
        is_warmup = index < warmup
        label = "warmup" if is_warmup else "measured"
        started = time.perf_counter()
        try:
            response, elapsed = request_json(base, path, token=token,
                                             data=payload, timeout=timeout)
            answer_ok, citations, note = evaluate_response(response, case["kind"])
            one = {"seconds": round(elapsed, 3), "api_ok": True,
                   "answer_present": answer_ok, "citations_present": citations,
                   "note": note}
            print(f"  {label}: {elapsed:.3f}s | HTTP OK | "
                  f"answer={'YES' if answer_ok else 'NO'} | "
                  f"citations={'YES' if citations else 'NO'}")
        except Exception as exc:
            one = {"seconds": round(time.perf_counter() - started, 3),
                   "api_ok": False, "answer_present": False,
                   "citations_present": False, "error": str(exc)}
            print(f"  {label}: ERROR after {one['seconds']:.3f}s - {str(exc)[:180]}")
        result["warmup" if is_warmup else "measured"].append(one)

    successful = [r["seconds"] for r in result["measured"]
                  if r["api_ok"] and r["answer_present"]]
    result["successful_measured_runs"] = len(successful)
    result["all_measured_successful"] = len(successful) == runs
    result["within_target_all_runs"] = (len(successful) == runs and
                                        all(r["seconds"] <= target for r in result["measured"]))
    if successful:
        ordered = sorted(successful)
        result.update({
            "min_seconds": round(min(successful), 3),
            "median_seconds": round(statistics.median(successful), 3),
            "mean_seconds": round(statistics.mean(successful), 3),
            "p95_nearest_rank_seconds": round(ordered[math.ceil(0.95 * len(ordered)) - 1], 3),
            "max_seconds": round(max(successful), 3),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--target", type=float, default=10.0,
                        help="Maximum acceptable API response time in seconds")
    parser.add_argument("--output", type=Path, default=Path("docmindai_10_performance_results.json"))
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0 or args.timeout <= 0 or args.target <= 0:
        parser.error("Invalid runs, warmup, timeout or target")
    print("Source-code-safe latency audit; authenticating with existing API.")
    email = input("DocMindAI login email: ").strip()
    password = getpass.getpass("DocMindAI password (hidden): ")
    try:
        login, _ = request_json(args.base_url, "/api/auth/login",
                                data={"username": email, "password": password}, form=True,
                                timeout=args.timeout)
    finally:
        del password
    token = str(login.get("access_token") or "")
    if not token:
        raise RuntimeError("Login response has no access_token")
    user, _ = request_json(args.base_url, "/me", token=token, timeout=args.timeout)
    documents, _ = request_json(args.base_url, "/api/upload/documents",
                                token=token, timeout=args.timeout)
    doc_ids = select_documents(list(documents.get("documents") or []))
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "performance-only; no original expected answers modified",
        "latency_scope": "Authenticated HTTP request-to-response only; excludes browser rendering",
        "cache_note": "Repeated requests may be served from cache; warmup=0 is not a guarantee of cold-start",
        "target_seconds": args.target,
        "runs": args.runs, "warmup": args.warmup,
        "tests": [],
    }
    for test in TESTS:
        print(f"Testing {test['id']} | {test['module']}", flush=True)
        report["tests"].append(run_case(args.base_url, token, str(user["id"]),
                                        doc_ids, test, args.runs, args.warmup,
                                        args.timeout, args.target))
    all_within = all(item["within_target_all_runs"] for item in report["tests"])
    report["all_10_within_target"] = all_within
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["test_id", "module", "successful_runs", "median_seconds",
                         "p95_seconds", "max_seconds", "all_within_target"])
        for item in report["tests"]:
            writer.writerow([item["id"], item["module"], item["successful_measured_runs"],
                             item.get("median_seconds", ""), item.get("p95_nearest_rank_seconds", ""),
                             item.get("max_seconds", ""), item["within_target_all_runs"]])
    print(f"\nSaved report: {args.output}")
    print(f"Saved timing CSV: {csv_path}")
    print(f"All 10 within {args.target:g}s: {'YES' if all_within else 'NO'}")
    print("Answers were NOT re-rated; check existing expected facts/citations separately.")
    return 0 if all_within else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("Cancelled", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
