r"""Latency audit of the 13 original Chat & Ask and Summaries retests.

Run from the DocMindAI project root:
    python .\tests\docmindai_13_live_latency.py --runs 3 --warmup 1

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

TESTS = [
    {"id":"Q6", "module":"Chat & Ask", "kind":"chat", "files":["docx"],
     "question":"Which vehicle model sold the most units in FY2025, and which showed the fastest year-over-year growth?",
     "expected":"Trailblazer top seller with 5,640 FY2025 units; Volt-X fastest growth with 1,850 FY2025 units; no exact Volt-X growth rate asserted."},
    {"id":"Q7", "module":"Chat & Ask", "kind":"chat", "files":["txt"],
     "question":"What are Apex Motors' major FY2026 growth targets and expansion plans?",
     "expected":"12% sales growth, 3,000 Volt-X capacity, two service centers, East region expansion, CSAT 4.6 to 5.0."},
    {"id":"Q9", "module":"Chat & Ask", "kind":"chat", "files":["txt"],
     "question":"Compare the average prices and customer satisfaction of Apex Motors and Veltrix Auto.",
     "expected":"Apex $38,100/CSAT 4.6; Veltrix $36,900/CSAT 4.2; correct relative comparison and sources."},
]
for code,fmt in (("PDF","pdf"),("DOCX","docx"),("TXT","txt"),("Monthly_CSV","monthly_csv"),("Data_XLSX","data_xlsx")):
    for label,scope,prompt in (("File","document",FILE_PROMPT),("Executive","executive",EXEC_PROMPT)):
        TESTS.append({
            "id":f"{code}_{label}","module":"Summaries - "+label,
            "kind":"summary","files":[fmt],"scope":scope,
            "question":prompt,
            "expected":"Compare against the original 47-test answer key for this selected document; document-grounded facts, complete sections, and citations.",
        })


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
    parser.add_argument("--output", type=Path, default=Path("docmindai_13_latency_results.json"))
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
    report["all_13_within_target"] = all_within
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
    print(f"All 13 within {args.target:g}s: {'YES' if all_within else 'NO'}")
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
