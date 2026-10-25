#!/usr/bin/env python3
"""
Script to validate Item 1A parsing on real SEC EDGAR 10-K filings (W1).
Fetches 10-Ks for 20 companies across the last 3 fiscal years.
Evaluates Item 1A section parser without modification and logs all failure modes.
Outputs results to results/real_data_validation.json.
"""
from __future__ import annotations
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.edgar import EDGARClient
from data.sections import Item1AParser

EASTERN = ZoneInfo("America/New_York")


def load_tickers(tickers_path: Path) -> List[str]:
    tickers = []
    with open(tickers_path, "r", encoding="utf-8") as f:
        for line in f:
            cleaned = line.strip()
            if cleaned and not cleaned.startswith("#"):
                tickers.append(cleaned.upper())
    return tickers


def run_validation():
    repo_root = Path(__file__).resolve().parent.parent
    tickers_file = repo_root / "config" / "tickers.txt"
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "real_data_validation.json"

    tickers = load_tickers(tickers_file)
    print(f"[INFO] Loaded {len(tickers)} tickers from {tickers_file}")

    user_agent = os.environ.get("SEC_USER_AGENT") or os.environ.get("EDGAR_USER_AGENT")
    client = EDGARClient(user_agent=user_agent, cache_dir=repo_root / "data" / "raw" / "edgar")
    parser = Item1AParser()

    print("[INFO] Fetching SEC ticker-to-CIK mapping...")
    ticker_to_cik = client.fetch_ticker_to_cik_map()

    filing_records: List[Dict[str, Any]] = []
    odd_timestamp_records: List[Dict[str, Any]] = []

    for ticker in tickers:
        cik = ticker_to_cik.get(ticker)
        if not cik:
            print(f"[WARN] No CIK found for {ticker}")
            continue

        print(f"[INFO] Fetching submissions for {ticker} (CIK {cik})...")
        try:
            subs = client.fetch_company_submissions(cik)
        except Exception as e:
            print(f"[ERROR] Failed to fetch submissions for {ticker}: {e}")
            continue

        recent = subs.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        acceptances = recent.get("acceptanceDateTime", [])
        primary_docs = recent.get("primaryDocument", [])

        # Find 10-K filings (skip 10-K/A amendments to inspect primary annual reports)
        ten_k_indices = [idx for idx, f in enumerate(forms) if f.upper() == "10-K"]
        target_indices = ten_k_indices[:3]  # Last 3 fiscal years

        print(f"  * {ticker}: Found {len(ten_k_indices)} total 10-Ks, evaluating latest {len(target_indices)}")

        for idx in target_indices:
            acc_num = accessions[idx]
            filing_date = filing_dates[idx] if idx < len(filing_dates) else "UNKNOWN"
            acc_raw = acceptances[idx] if idx < len(acceptances) else ""
            primary_doc = primary_docs[idx] if idx < len(primary_docs) else ""

            # Check timestamp validity & tz-awareness
            is_tz_aware = False
            parsed_dt_iso: Optional[str] = None
            ts_failure = None

            if not acc_raw:
                ts_failure = "Missing acceptance timestamp in submissions metadata"
            else:
                try:
                    dt = parser.parse_acceptance_timestamp(acc_raw)
                    is_tz_aware = dt.tzinfo is not None
                    parsed_dt_iso = dt.isoformat()
                    # Check for odd timestamp (e.g. year < 2000 or year > 2030)
                    if dt.year < 2000 or dt.year > 2030:
                        ts_failure = f"Odd year in timestamp: {dt.year}"
                except Exception as e:
                    ts_failure = f"Failed to parse timestamp '{acc_raw}': {e}"

            if ts_failure:
                odd_timestamp_records.append({
                    "ticker": ticker,
                    "accession_number": acc_num,
                    "raw_timestamp": acc_raw,
                    "reason": ts_failure,
                })

            # Fetch document text
            failure_reason = ts_failure
            section_found = False
            section_length = 0
            section_text_preview = ""

            try:
                raw_text = client.fetch_filing_document(ticker, cik, acc_num, primary_doc)
            except Exception as e:
                failure_reason = f"Failed to fetch filing document: {e}"
                raw_text = ""

            if raw_text and not failure_reason:
                try:
                    section = parser.extract_item_1a(
                        ticker=ticker,
                        accession_number=acc_num,
                        raw_text=raw_text,
                        accepted_at_str=acc_raw,
                    )
                    if section is not None and section.text and len(section.text.strip()) > 100:
                        section_found = True
                        section_length = len(section.text)
                        section_text_preview = section.text[:120].replace("\n", " ")
                    else:
                        section_found = False
                        if section is None:
                            failure_reason = "Item 1A header boundary pattern not matched"
                        else:
                            failure_reason = f"Extracted section text too short ({len(section.text) if section else 0} chars)"
                except Exception as e:
                    section_found = False
                    failure_reason = f"Parser exception: {e}"

            record = {
                "ticker": ticker,
                "accession_number": acc_num,
                "filing_date": filing_date,
                "accepted_at": parsed_dt_iso,
                "raw_acceptance_str": acc_raw,
                "is_tz_aware": is_tz_aware,
                "primary_document": primary_doc,
                "section_found": section_found,
                "section_length": section_length,
                "failure_reason": failure_reason,
                "preview": section_text_preview,
            }
            filing_records.append(record)

    # Compute Totals & Failure groupings
    total_filings = len(filing_records)
    successful_parses = sum(1 for r in filing_records if r["section_found"])
    success_rate = (successful_parses / total_filings) if total_filings > 0 else 0.0

    failures = [r for r in filing_records if not r["section_found"]]
    failure_reasons = [r["failure_reason"] or "Unknown" for r in failures]
    failures_grouped = dict(Counter(failure_reasons))

    output_payload = {
        "timestamp_utc": datetime.now(ZoneInfo("UTC")).isoformat(),
        "totals": {
            "total_filings": total_filings,
            "successful_parses": successful_parses,
            "failed_parses": len(failures),
            "parse_success_rate": round(success_rate, 4),
            "filings_with_missing_or_odd_timestamps": len(odd_timestamp_records),
        },
        "failures_grouped_by_reason": failures_grouped,
        "odd_timestamps": odd_timestamp_records,
        "filings": filing_records,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[OK] Validation results written to {out_file}")
    print("=" * 60)
    print("REAL DATA VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Filings Evaluated: {total_filings}")
    print(f"Successful Parses:       {successful_parses} ({success_rate:.1%})")
    print(f"Failed Parses:           {len(failures)} ({1.0 - success_rate:.1%})")
    print(f"Odd / Missing Timestamps: {len(odd_timestamp_records)}")
    print("\nFailures Grouped by Reason:")
    for reason, count in failures_grouped.items():
        print(f"  - {reason}: {count}")

    # Top 3 failure modes with 2 examples each
    top_failures = Counter(failure_reasons).most_common(3)
    print("\nTop Failure Modes with Example Filings:")
    for reason, count in top_failures:
        print(f"\nFailure Mode: '{reason}' (Count: {count})")
        examples = [f for f in failures if f["failure_reason"] == reason][:2]
        for ex in examples:
            print(f"  * {ex['ticker']} (Acc: {ex['accession_number']}, Date: {ex['filing_date']}, Doc: {ex['primary_document']})")


if __name__ == "__main__":
    run_validation()
