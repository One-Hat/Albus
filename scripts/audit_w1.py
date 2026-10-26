"""
Comprehensive audit script for W1 Real-Data Validation.
Audits all 54 parsed filings and reconciles all 60 expected filings.
DO NOT modify parser or thresholds. Report only.
"""
from __future__ import annotations
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from src.data.sections import Item1AParser


def run_audit():
    # ---------------------------------------------------------
    # Load tickers and validation data
    # ---------------------------------------------------------
    tickers_file = Path("config/tickers.txt")
    with open(tickers_file, "r", encoding="utf-8") as f:
        tickers = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]

    val_file = Path("results/real_data_validation.json")
    with open(val_file, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    filings = val_data["filings"]
    filings_by_ticker = defaultdict(list)
    for f in filings:
        filings_by_ticker[f["ticker"]].append(f)

    print("=" * 80)
    print("SECTION 1: RECONCILIATION OF FILING COUNTS (EXPECTED 60 VS EVALUATED 54)")
    print("=" * 80)
    print(f"Total Tickers Requested: {len(tickers)}")
    print(f"Expected Filings (3 years * 20 tickers): 60")
    print(f"Total Filings Evaluated & Parsed: {len(filings)}")
    print(f"Total Missing Filings: {60 - len(filings)}\n")

    cik_map_file = Path("data/raw/edgar/company_tickers.json")
    cik_map = {}
    if cik_map_file.exists():
        raw_m = json.loads(cik_map_file.read_text(encoding="utf-8"))
        for v in raw_m.values():
            cik_map[v["ticker"].upper()] = (str(v["cik_str"]).zfill(10), v["title"])

    print(f"{'Ticker':<7} | {'Cap Size':<10} | {'CIK':<10} | {'EDGAR Entity Title':<35} | {'Listed':<7} | {'Parsed':<7} | {'Missing':<7}")
    print("-" * 95)

    small_caps = {"CALM", "BOOT", "SHOO", "KFY", "MED", "PLCE", "TITN", "PATK"}
    missing_breakdown = []

    for t in tickers:
        cap = "Small Cap" if t in small_caps else "Large Cap"
        cik, title = cik_map.get(t, ("UNKNOWN", "UNKNOWN"))
        sub_file = Path(f"data/raw/edgar/CIK{cik}_submissions.json")
        total_10k_listed = 0
        if sub_file.exists():
            try:
                sub_data = json.loads(sub_file.read_text(encoding="utf-8"))
                forms = sub_data.get("filings", {}).get("recent", {}).get("form", [])
                total_10k_listed = forms.count("10-K")
            except Exception:
                pass

        parsed_count = len(filings_by_ticker.get(t, []))
        missing_count = 3 - parsed_count
        print(f"{t:<7} | {cap:<10} | {cik:<10} | {title[:35]:<35} | {total_10k_listed:<7} | {parsed_count:<7} | {missing_count:<7}")

        if missing_count > 0:
            missing_breakdown.append((t, cap, cik, title, total_10k_listed, parsed_count, missing_count))

    print("\nDETAILED EXPLANATION OF EVERY MISSING FILING:")
    print("-" * 80)
    for t, cap, cik, title, listed, parsed, missing in missing_breakdown:
        print(f"\n* Ticker: {t} ({cap}) -- CIK {cik} ({title})")
        print(f"  Expected: 3 filings | Found in EDGAR submissions.recent: {listed} | Parsed: {parsed} | Missing: {missing}")
        if t == "XOM":
            print("  Reason: Entity CIK Ambiguity in SEC company_tickers.json.")
            print("          'XOM' mapped to 'ExxonMobil Holdings Corp' (CIK 0002115436, newly incorporated entity).")
            print("          The primary submissions JSON for CIK 0002115436 contains 0 historical 10-K filings.")
            print("          Historical operating entity Exxon Mobil Corp is registered under CIK 0000034088.")
            print("          Missing Fiscal Years: FY2023, FY2024, FY2025 (3 filings).")
        elif t == "JPM":
            sub_file = Path(f"data/raw/edgar/CIK{cik}_submissions.json")
            sub_data = json.loads(sub_file.read_text(encoding="utf-8"))
            num_recent = len(sub_data["filings"]["recent"]["form"])
            older_files = len(sub_data["filings"].get("files", []))
            print(f"  Reason: Mega-Filer Sub-Submission Partitioning.")
            print(f"          JPM has {num_recent:,} recent filings in the primary submissions file (mostly debt/structured notes).")
            print(f"          The primary submissions.json recent chunk only extends back to late 2024 / early 2025.")
            print(f"          Older 10-Ks are partitioned across {older_files} secondary shard files (CIK0000019617-submissions-*.json).")
            print(f"          Only 1 10-K was within the primary 'recent' array (Accession: 0000019617-25-000293, FY2024).")
            print(f"          Missing Fiscal Years: FY2023, FY2022 (2 filings).")
        elif t == "META":
            print("  Reason: Primary submissions.recent truncated to last 1,000 filings.")
            print("          Meta Platforms had 2 10-Ks in recent array: FY2025 (2026-01-29) and FY2024 (2025-01-30).")
            print("          FY2023 10-K was beyond the 1,000 filing cutoff in submissions.json.")
            print("          Missing Fiscal Year: FY2023 (1 filing).")

    # ---------------------------------------------------------
    # SECTION 2: SECTION LENGTH STATS & UNDER 5,000 CHARACTERS
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("SECTION 2: SECTION LENGTH STATS OVER ALL 54 FILINGS")
    print("=" * 80)
    lengths = np.array([f["section_length"] for f in filings])
    min_len = int(np.min(lengths))
    p10_len = float(np.percentile(lengths, 10))
    med_len = float(np.median(lengths))
    p90_len = float(np.percentile(lengths, 90))
    max_len = int(np.max(lengths))

    print(f"Sample Size: {len(lengths)}")
    print(f"Min:         {min_len:,} characters")
    print(f"P10:         {p10_len:,.1f} characters")
    print(f"Median:      {med_len:,.1f} characters")
    print(f"P90:         {p90_len:,.1f} characters")
    print(f"Max:         {max_len:,} characters")
    print(f"Mean:        {float(np.mean(lengths)):,.1f} characters")
    print(f"Std Dev:     {float(np.std(lengths)):,.1f} characters")

    short_filings = [f for f in filings if f["section_length"] < 5000]
    print(f"\nFILINGS UNDER 5,000 CHARACTERS (COUNT: {len(short_filings)}):")
    print("-" * 80)
    for sf in short_filings:
        ticker = sf["ticker"]
        acc = sf["accession_number"]
        fdate = sf["filing_date"]
        raw_cache = Path(f"data/raw/edgar/{ticker}_{acc}.txt")
        # Extract full extracted text from cache
        if raw_cache.exists():
            parser = Item1AParser()
            section = parser.extract_item_1a(ticker, acc, raw_cache.read_text(encoding="utf-8", errors="ignore"), sf["accepted_at"])
            extracted_text = section.text if section else ""
        else:
            extracted_text = sf.get("preview", "")

        first_300 = repr(extracted_text[:300])
        print(f"\n* Ticker: {ticker} | Accession: {acc} | Filing Date: {fdate} | Length: {sf['section_length']} chars")
        print(f"  First 300 chars: {first_300}")

    # ---------------------------------------------------------
    # SECTION 3: AUDIT OF EVERY FILING (ALL 54)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("SECTION 3: DETAILED AUDIT FOR EVERY FILING (ALL 54)")
    print("=" * 80)

    # Keywords regex
    kw_risk = re.compile(r'\brisk\b', re.IGNORECASE)
    kw_adverse = re.compile(r'\badverse\b', re.IGNORECASE)
    kw_may = re.compile(r'\bmay\b', re.IGNORECASE)
    kw_could = re.compile(r'\bcould\b', re.IGNORECASE)

    # Table of contents patterns
    toc_regex = re.compile(r'(?i)(table\s+of\s+contents|\.{4,}|item\s+1a[^\n\r]{0,25}\.{3,})')

    audit_records = []
    parser = Item1AParser()

    for idx, f in enumerate(filings, 1):
        ticker = f["ticker"]
        acc = f["accession_number"]
        fdate = f["filing_date"]
        accepted_at = f["accepted_at"]
        raw_cache = Path(f"data/raw/edgar/{ticker}_{acc}.txt")

        if raw_cache.exists():
            raw_text = raw_cache.read_text(encoding="utf-8", errors="ignore")
            section = parser.extract_item_1a(ticker, acc, raw_text, accepted_at)
            extracted = section.text if section else ""
        else:
            extracted = f.get("preview", "")

        # 1. Starts with Item 1A or Risk Factors?
        first_100 = extracted[:100].strip()
        starts_item_1a_or_rf = bool(re.match(r'(?i)^(item\s+1a|risk\s+factors)', first_100))

        # 2. Ends before Item 1B or Item 2?
        # Check if extracted text contains substantive Item 1B or Item 2 header inside it
        contains_item_1b_body = bool(re.search(r'(?i)\bitem\s+1b[\.\:\s\-\—]+unresolved', extracted[500:]))
        contains_item_2_body = bool(re.search(r'(?i)\bitem\s+2[\.\:\s\-\—]+properties', extracted[500:]))
        ends_before_1b_or_2 = not (contains_item_1b_body or contains_item_2_body)

        # 3. Contains TOC pattern?
        has_toc_pattern = bool(toc_regex.search(extracted))

        # 4. Keyword counts
        c_risk = len(kw_risk.findall(extracted))
        c_adverse = len(kw_adverse.findall(extracted))
        c_may = len(kw_may.findall(extracted))
        c_could = len(kw_could.findall(extracted))
        c_total_kw = c_risk + c_adverse + c_may + c_could

        audit_records.append({
            "idx": idx,
            "ticker": ticker,
            "accession": acc,
            "filing_date": fdate,
            "length": len(extracted),
            "starts_1a_or_rf": starts_item_1a_or_rf,
            "ends_before_1b_2": ends_before_1b_or_2,
            "has_toc_pattern": has_toc_pattern,
            "kw_risk": c_risk,
            "kw_adverse": c_adverse,
            "kw_may": c_may,
            "kw_could": c_could,
            "kw_total": c_total_kw,
            "first_60": extracted[:60].replace("\n", " ").strip(),
        })

    # Print markdown table of all 54
    print(f"{'#':<3} | {'Ticker':<6} | {'Accession':<22} | {'Date':<10} | {'Chars':<8} | {'Start 1A?':<10} | {'End <1B/2?':<11} | {'Has TOC?':<9} | {'Risk':<5} | {'Adv':<5} | {'May':<5} | {'Could':<5} | {'Tot KW':<6}")
    print("-" * 125)
    for r in audit_records:
        print(
            f"{r['idx']:<3} | {r['ticker']:<6} | {r['accession']:<22} | {r['filing_date']:<10} | {r['length']:<8} | "
            f"{str(r['starts_1a_or_rf']):<10} | {str(r['ends_before_1b_2']):<11} | {str(r['has_toc_pattern']):<9} | "
            f"{r['kw_risk']:<5} | {r['kw_adverse']:<5} | {r['kw_may']:<5} | {r['kw_could']:<5} | {r['kw_total']:<6}"
        )

    # ---------------------------------------------------------
    # SECTION 4: SMALL-CAP FILINGS (ITEM 1A NOT REQUIRED / NOT APPLICABLE)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("SECTION 4: SMALL-CAP FILINGS AUDIT (ITEM 1A NOT REQUIRED / NOT APPLICABLE)")
    print("=" * 80)

    small_cap_filings = [f for f in filings if f["ticker"] in small_caps]
    print(f"Total Small-Cap Filings Evaluated: {len(small_cap_filings)} across {len(small_caps)} tickers")

    not_required_pattern = re.compile(
        r'(?i)(item\s+1a[^\n\r]{0,30}(not\s+applicable|not\s+required|none)|smaller\s+reporting\s+companies\s+are\s+not\s+required)'
    )

    found_not_required = []
    for sf in small_cap_filings:
        ticker = sf["ticker"]
        acc = sf["accession_number"]
        raw_cache = Path(f"data/raw/edgar/{ticker}_{acc}.txt")
        if raw_cache.exists():
            raw_text = raw_cache.read_text(encoding="utf-8", errors="ignore")
            section = parser.extract_item_1a(ticker, acc, raw_text, sf["accepted_at"])
            text_to_check = section.text if section else ""
            # Also check raw text around item 1a
            match = not_required_pattern.search(text_to_check)
            if not match:
                match = not_required_pattern.search(raw_text[:50000])
            if match:
                found_not_required.append((ticker, acc, sf["filing_date"], match.group(0)))

    if not found_not_required:
        print("RESULT: None. No small-cap filings stated that Item 1A is not required or not applicable.")
        print("All 8 small-cap companies in this universe (CALM, BOOT, SHOO, KFY, MED, PLCE, TITN, PATK)")
        print("provide Item 1A Risk Factors disclosures.")
    else:
        print(f"Found {len(found_not_required)} small-cap filings stating Item 1A not required / not applicable:")
        for t, acc, dt, matched_txt in found_not_required:
            print(f"- {t} ({acc}, {dt}): Matched '{matched_txt}'")

    # ---------------------------------------------------------
    # SECTION 5: TIMESTAMPS AUDIT (10 SAMPLE + MIDNIGHT COUNT)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("SECTION 5: TIMESTAMPS AUDIT (10 SAMPLES + MIDNIGHT COUNT)")
    print("=" * 80)

    # Count how many are exactly midnight
    midnight_filings = []
    for f in filings:
        accepted_str = f["accepted_at"]
        raw_str = f.get("raw_acceptance_str", "")
        # Check time component in accepted_at and raw_str
        if "T00:00:00" in accepted_str or " 00:00:00" in accepted_str:
            midnight_filings.append(f)
        elif len(raw_str.strip()) == 14 and raw_str.strip().endswith("000000"):
            midnight_filings.append(f)

    print(f"Filings with timestamp exactly at Midnight (00:00:00): {len(midnight_filings)} of {len(filings)}")

    # Deterministic sample of 10 filings (fixed seed for reproducibility)
    rng = random.Random(42)
    sample_10 = rng.sample(filings, 10)

    print(f"\n{'#':<3} | {'Ticker':<6} | {'Accession':<22} | {'accepted_at (tz-aware)':<28} | {'raw_acceptance_str'}")
    print("-" * 90)
    for idx, f in enumerate(sample_10, 1):
        print(f"{idx:<3} | {f['ticker']:<6} | {f['accession_number']:<22} | {f['accepted_at']:<28} | {f.get('raw_acceptance_str')}")

    # ---------------------------------------------------------
    # SECTION 6: ACCESSION NUMBER & SOURCE URL FOR 5 SAMPLE FILINGS
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("SECTION 6: ACCESSION NUMBER AND SOURCE URL FOR 5 RANDOM FILINGS")
    print("=" * 80)
    sample_5 = rng.sample(filings, 5)

    for idx, f in enumerate(sample_5, 1):
        ticker = f["ticker"]
        acc = f["accession_number"]
        cik, title = cik_map.get(ticker, ("UNKNOWN", "UNKNOWN"))
        clean_cik = str(int(cik)) if cik.isdigit() else cik
        acc_nodash = acc.replace("-", "")
        pdoc = f.get("primary_document", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{clean_cik}/{acc_nodash}/{pdoc}"
        print(f"{idx}. Ticker: {ticker} ({title})")
        print(f"   Accession Number: {acc}")
        print(f"   Primary Document: {pdoc}")
        print(f"   Filing Date:      {f['filing_date']}")
        print(f"   Accepted At (ET): {f['accepted_at']}")
        print(f"   SEC EDGAR URL:    {url}\n")


if __name__ == "__main__":
    run_audit()
