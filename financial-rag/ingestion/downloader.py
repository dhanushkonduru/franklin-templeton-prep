"""
SEC EDGAR downloader
Fetches 10-K filings for any ticker, saves PDFs/HTMLs to data/raw/
Rate limit: 10 req/sec per SEC policy — we stay at ~5 req/sec to be safe.
"""

import os
import time
import json
import requests
from pathlib import Path
from loguru import logger
from tqdm import tqdm

HEADERS = {
    "User-Agent": "FinancialRAG research@example.com",  # SEC requires a user-agent
    "Accept-Encoding": "gzip, deflate",
}
BASE_URL = "https://data.sec.gov"
EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
RAW_DIR = Path("./data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

SLEEP = 0.2  # 5 req/sec


def _get(url: str, **kwargs) -> requests.Response:
    time.sleep(SLEEP)
    r = requests.get(url, headers=HEADERS, timeout=30, **kwargs)
    r.raise_for_status()
    return r


def get_cik(ticker: str) -> str:
    """Resolve ticker → CIK (SEC company identifier)."""
    url = "https://www.sec.gov/files/company_tickers.json"
    data = _get(url).json()
    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry["ticker"].upper() == ticker_upper:
            cik = str(entry["cik_str"]).zfill(10)
            logger.info(f"{ticker} → CIK {cik}")
            return cik
    raise ValueError(f"Ticker '{ticker}' not found on SEC EDGAR")


def get_10k_filings(cik: str, count: int = 2) -> list[dict]:
    """Return metadata for the most recent `count` 10-K filings."""
    url = f"{BASE_URL}/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K&dateb=&owner=include&count={count}&search_text=&output=atom"
    # Use submissions endpoint — more reliable
    url = f"{BASE_URL}/submissions/CIK{cik}.json"
    data = _get(url).json()

    filings = []
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form == "10-K" and len(filings) < count:
            filings.append({
                "accession": accessions[i].replace("-", ""),
                "date": dates[i],
                "primary_doc": primary_docs[i],
                "cik": cik,
            })

    logger.info(f"Found {len(filings)} 10-K filings for CIK {cik}")
    return filings


def download_filing(filing: dict, ticker: str) -> Path | None:
    """Download the primary document (PDF or HTML) for a filing."""
    cik = filing["cik"]
    acc = filing["accession"]
    doc = filing["primary_doc"]
    date = filing["date"]

    # Build filing index URL
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}"
    doc_url = f"{base}/{doc}"

    out_name = f"{ticker}_{date}_{doc}"
    out_path = RAW_DIR / out_name

    if out_path.exists():
        logger.info(f"Already downloaded: {out_path.name}")
        return out_path

    logger.info(f"Downloading {doc_url}")
    try:
        r = _get(doc_url)
        out_path.write_bytes(r.content)
        logger.success(f"Saved → {out_path.name} ({len(r.content)//1024} KB)")
        return out_path
    except Exception as e:
        logger.error(f"Failed to download {doc_url}: {e}")
        return None


def fetch_tickers(tickers: list[str], filings_per_ticker: int = 2) -> list[Path]:
    """Main entry point — fetch 10-K filings for a list of tickers."""
    all_paths: list[Path] = []
    for ticker in tqdm(tickers, desc="Tickers"):
        try:
            cik = get_cik(ticker)
            filings = get_10k_filings(cik, count=filings_per_ticker)
            for filing in filings:
                path = download_filing(filing, ticker)
                if path:
                    all_paths.append(path)
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
    return all_paths


if __name__ == "__main__":
    # Quick test: download Apple + Microsoft 10-Ks (2 years each)
    paths = fetch_tickers(["AAPL", "MSFT", "TSLA"], filings_per_ticker=2)
    print(f"\nDownloaded {len(paths)} filings:")
    for p in paths:
        print(f"  {p}")
