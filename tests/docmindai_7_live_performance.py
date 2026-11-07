#!/usr/bin/env python3
r"""DocMindAI: live HTTP performance audit for ONLY 7 previously partial cases.

Tests: Chat & Ask Q9; File and Executive Summaries for the original PDF,
DOCX and TXT documents. No other tests or Comparison fixtures are included.

Run from DocMindAI project root (PowerShell):
    python .\tests\docmindai_7_live_performance.py `
      --base-url http://127.0.0.1:8000 `
      --runs 3 --warmup 1 --target 10 --timeout 30 `
      --output .\docmindai_7_live_results.json

Uses standard library only. Does NOT modify project code, APIs, DB schema,
documents or their expected answers. It calls authenticated existing APIs;
Chat & Ask may store ordinary conversation history as usual. Repeated
requests may benefit from caches; warmup=0 is not proof of a cold start.

Measures server HTTP request-to-response time including JSON transfer,
NOT browser rendering or user-perceived end-to-end latency.
Does NOT automatically certify factual correctness or formal RAG metrics.
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

FILE_SUMMARY_PROMPT = (
    "Generate a concise, accurate File Summary of the selected document. "
    "Include its overview, key topics, exact facts and figures, main findings, "
    "conclusion, and source citations. Do not invent missing details."
)
EXECUTIVE_SUMMARY_PROMPT = (
    "Generate an Executive Summary containing an overview, objective and scope, "
    "key highlights, critical KPIs, findings, supported risks, recommendations "
    "if justified, conclusion, and source citations. Preserve all numerical "
    "values exactly."
)

DOCUMENTS = {
    "txt": "apex_motors.txt",
    "pdf": "apex_motors_market_report.pdf",
    "docx": "apex_motors_annual_report.docx",
}

# Test identifiers and original prompts are unchanged. Reference facts below
# are informational for HUMAN review only, never inserted into an API request.
TESTS: list[dict[str, Any]] = [
    {
        "id": "Q9", "module": "Chat & Ask", "type": "chat", "document": "txt",
        "prompt": "Compare the average prices and customer satisfaction of Apex Motors and Veltrix Auto.",
        "reference": "Apex $38,100 / CSAT 4.6/5; Veltrix $36,900 / CSAT 4.2/5; $1,200 and 0.4-point differences; TXT section 10.",
    },
    {
        "id": "PDF_File_Summary", "module": "Summaries - File", "type": "summary",
        "document": "pdf", "scope": "document", "prompt": FILE_SUMMARY_PROMPT,
        "reference": "Market/SUV/EV trends; Trailblazer 5,640 and $41,200; Apex $38,100 and 4.6/5; inventory/outlook; PDF pages 2-5.",
    },
    {
        "id": "PDF_Executive_Summary", "module": "Summaries - Executive", "type": "summary",
        "document": "pdf", "scope": "executive", "prompt": EXECUTIVE_SUMMARY_PROMPT,
        "reference": "Apex $38,100 / 4.6; Horizon $41,500 / 4.4; $3,400 and 0.2-point deltas; Trailblazer 5,640; PDF pages 2-5.",
    },
    {
        "id": "DOCX_File_Summary", "module": "Summaries - File", "type": "summary",
        "document": "docx", "scope": "document", "prompt": FILE_SUMMARY_PROMPT,
        "reference": "FY2025 $612.4M / 16,070; Q1 $128.5M / 3,420; Q4 $181.1M / 4,610; South 5,150; East 3,180; Trailblazer 5,640; workforce 1,240.",
    },
    {
        "id": "DOCX_Executive_Summary", "module": "Summaries - Executive", "type": "summary",
        "document": "docx", "scope": "executive", "prompt": EXECUTIVE_SUMMARY_PROMPT,
        "reference": "FY2025 $612.4M / 16,070; Q1/Q4; Trailblazer 5,640; South 5,150; workforce 1,240; FY2026 +12%, +3,000 Volt-X, 2 centers, East expansion.",
    },
    {
        "id": "TXT_File_Summary", "module": "Summaries - File", "type": "summary",
        "document": "txt", "scope": "document", "prompt": FILE_SUMMARY_PROMPT,
        "reference": "FY2025 $612.4M / 16,070; separate January-October $488,538,000 / 12,940; inventory, workforce 1,240 and FY2026 plans; TXT sections 2-11.",
    },
    {
        "id": "TXT_Executive_Summary", "module": "Summaries - Executive", "type": "summary",
        "document": "txt", "scope": "executive", "prompt": EXECUTIVE_SUMMARY_PROMPT,
        "reference": "FY2025 $612.4M / 16,070; South/Trailblazer; FY2026 +12%, +3,000 Volt-X, 2 centers, East, CSAT 4.6 to 5.0; TXT sections 2,5-7,10-11.",
    },
]

ERROR_MESSAGES = (
    "local model request failed",
    "i could not produce a sufficiently verified answer",
    "i could not find sufficient verified evidence",
    "internal server error",
)


def request_json(base_url: str, path: str, *, token: str | None = None,
                 data: dict[str, Any] | None = None, form: bool = False,
                 timeout: float = 30.0) -> tuple[Any, float]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = None
    if data is not None:
        if form:
            payload = urllib.parse.urlencode(data).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base_url.rstrip("/") + path, data=payload, headers=headers
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code} on {path}: {detail[:240]}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Connection/timeout on {path}: {exc}") from None
    seconds = time.perf_counter() - started
    if not 200 <= status < 300:
        raise RuntimeError(f"HTTP {status} on {path}")
    try:
        return json.loads(raw.decode("utf-8")), seconds
    except (UnicodeError, json.JSONDecodeError):
        raise RuntimeError(f"Non-JSON response on {path}") from None


def document_ids(doc_list: list[dict[str, Any]]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for key, filename in DOCUMENTS.items():
        matches = [item for item in doc_list if
                   str(item.get("original_filename") or item.get("filename") or "").casefold()
                   == filename.casefold()]
        if not matches:
            raise RuntimeError(f"Document not found in uploads: {filename}")
        matches.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        if len(matches) > 1:
            print(f"WARNING: Multiple uploads named {filename}; using most recent.", file=sys.stderr)
        selected[key] = str(matches[0]["id"])
    return selected


def request_for(test: dict[str, Any], user_id: str,
                selected: dict[str, str]) -> tuple[str, dict[str, Any]]:
    common = {"user_id": user_id, "document_ids": [selected[test["document"]]]}
    if test["type"] == "chat":
        return "/api/v1/chat", {**common, "query": test["prompt"], "history": [], "top_k": 5}
    return "/api/v1/summary", {
        **common, "scope": test["scope"],
        "scope_instructions": test["prompt"], "top_k": 8,
    }


def inspect_response(response: Any, test_type: str) -> tuple[str, bool, bool, str]:
    if not isinstance(response, dict):
        return "", False, False, "Response was not a JSON object"
    field = "answer" if test_type == "chat" else "summary"
    answer = str(response.get(field) or "").strip()
    citations = bool(response.get("citations"))
    if not answer:
        return "", False, citations, "Empty answer"
    if any(message in answer.lower() for message in ERROR_MESSAGES):
        return answer, False, citations, "Answer contains model/error or refusal message"
    return answer, True, citations, "Factual accuracy requires manual review"


def run_case(test: dict[str, Any], *, base_url: str, token: str,
             user_id: str, selected: dict[str, str], runs: int, warmup: int,
             timeout: float, target: float, save_answers: bool) -> dict[str, Any]:
    path, payload = request_for(test, user_id, selected)
    result: dict[str, Any] = {
        "test_id": test["id"], "module": test["module"],
        "document": DOCUMENTS[test["document"]],
        "original_prompt": test["prompt"], "manual_reference": test["reference"],
        "warmup": [], "measured": [],
    }
    for index in range(warmup + runs):
        is_warmup = index < warmup
        label = "warmup" if is_warmup else "measured"
        started = time.perf_counter()
        try:
            response, seconds = request_json(
                base_url, path, token=token, data=payload, timeout=timeout
            )
            answer, answer_ok, has_citations, note = inspect_response(response, test["type"])
            entry: dict[str, Any] = {
                "seconds": round(seconds, 3), "api_ok": True,
                "answer_ok": answer_ok, "citations_present": has_citations,
                "under_target": seconds <= target,
                "note": note,
            }
            if save_answers:
                entry["answer"] = answer
                entry["citations"] = response.get("citations", [])
            print(f"  {label}: {seconds:.3f}s | HTTP OK | "
                  f"answer={'YES' if answer_ok else 'NO'} | "
                  f"citations={'YES' if has_citations else 'NO'} | "
                  f"<= {target:g}s: {'YES' if seconds <= target else 'NO'}", flush=True)
        except Exception as exc:
            entry = {
                "seconds": round(time.perf_counter() - started, 3),
                "api_ok": False, "answer_ok": False,
                "citations_present": False, "under_target": False,
                "error": str(exc),
            }
            print(f"  {label}: FAILED at {entry['seconds']:.3f}s - "
                  f"{str(exc)[:180]}", flush=True)
        result[label].append(entry)

    successful = [item["seconds"] for item in result["measured"]
                  if item["api_ok"] and item["answer_ok"]]
    result["successful_measured_runs"] = len(successful)
    result["all_measured_http_and_answer_ok"] = len(successful) == runs
    result["all_measured_citations_present"] = all(
        item["citations_present"] for item in result["measured"]
    )
    result["all_measured_under_target"] = (
        len(successful) == runs
        and all(item["under_target"] for item in result["measured"])
    )
    result["all_measured_pass_basic_checks"] = (
        result["all_measured_under_target"]
        and result["all_measured_citations_present"]
    )
    if successful:
        ordered = sorted(successful)
        result.update({
            "min_seconds": round(min(successful), 3),
            "median_seconds": round(statistics.median(successful), 3),
            "mean_seconds": round(statistics.mean(successful), 3),
            "p95_nearest_rank_seconds": round(
                ordered[math.ceil(0.95 * len(ordered)) - 1], 3
            ),
            "max_seconds": round(max(successful), 3),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--target", type=float, default=10.0,
                        help="Maximum allowed HTTP response seconds")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Per-request HTTP timeout (seconds)")
    parser.add_argument("--output", type=Path,
                        default=Path("docmindai_7_live_performance_results.json"))
    parser.add_argument("--save-answers", action="store_true",
                        help="Store returned answer text and citation objects in JSON (private)")
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0 or args.target <= 0 or args.timeout <= 0:
        parser.error("--runs must be >=1, --warmup >=0, --target and --timeout >0")
    if args.timeout < args.target:
        parser.error("--timeout should be >= --target")

    print("DocMindAI: auditing ONLY 7 partial Chat & Ask / Summaries tests.")
    print("API calls use existing authentication; no source files will be changed.")
    email = input("DocMindAI login email: ").strip()
    password = getpass.getpass("DocMindAI password (hidden): ")
    try:
        login, _ = request_json(
            args.base_url, "/api/auth/login", data={"username": email,
            "password": password}, form=True, timeout=args.timeout
        )
    finally:
        del password
    token = str(login.get("access_token") or "")
    if not token:
        raise RuntimeError("Login response had no access_token")
    me, _ = request_json(args.base_url, "/me", token=token, timeout=args.timeout)
    documents, _ = request_json(
        args.base_url, "/api/upload/documents", token=token, timeout=args.timeout
    )
    selected = document_ids(list(documents.get("documents") or []))

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "7-partial-case live latency and basic response audit",
        "latency_definition": "Authenticated HTTP request-to-JSON response, not browser rendering",
        "accuracy_note": "Not an automatic factual accuracy score; compare original answer keys manually",
        "cache_note": "Warmups/repeated requests can be cached; warmup=0 is not guaranteed cold",
        "security_note": "No login password or access token saved; --save-answers may save document content",
        "runs": args.runs, "warmup": args.warmup,
        "target_seconds": args.target, "timeout_seconds": args.timeout,
        "tests": [],
    }
    for test in TESTS:
        print(f"Testing {test['id']} | {test['module']} | {DOCUMENTS[test['document']]}", flush=True)
        report["tests"].append(run_case(
            test, base_url=args.base_url, token=token,
            user_id=str(me["id"]), selected=selected,
            runs=args.runs, warmup=args.warmup, timeout=args.timeout,
            target=args.target, save_answers=args.save_answers,
        ))

    report["all_7_meet_latency_and_basic_checks"] = all(
        test["all_measured_pass_basic_checks"] for test in report["tests"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "test_id", "module", "document", "successful_measured_runs",
            "median_seconds", "p95_nearest_rank_seconds", "max_seconds",
            "all_citations_present", "all_under_10s_target",
            "all_basic_checks_passed",
        ])
        for case in report["tests"]:
            writer.writerow([
                case["test_id"], case["module"], case["document"],
                f"{case['successful_measured_runs']}/{args.runs}",
                case.get("median_seconds", ""),
                case.get("p95_nearest_rank_seconds", ""),
                case.get("max_seconds", ""),
                case["all_measured_citations_present"],
                case["all_measured_under_target"],
                case["all_measured_pass_basic_checks"],
            ])
    print(f"\nSaved JSON report: {args.output}")
    print(f"Saved CSV report:  {csv_path}")
    print("All 7 within target, with valid answers and citations: "
          f"{'YES' if report['all_7_meet_latency_and_basic_checks'] else 'NO'}")
    print("Reminder: API and citation checks do NOT prove factual accuracy.")
    return 0 if report["all_7_meet_latency_and_basic_checks"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("Cancelled.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
