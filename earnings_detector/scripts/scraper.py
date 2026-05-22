"""
TRANSCRIPT SCRAPER
------------------
In production, you need 200+ transcripts across 20 companies.

Strategy:
1. Motley Fool: publicly accessible, clean HTML structure
2. Seeking Alpha: requires account but best coverage
3. Company IR pages: most reliable, usually PDFs

WHY 200+ TRANSCRIPTS?
- Per company: 4 quarters = ~5 years of data  
- Across 20 companies: enough cross-sectional variation to see patterns
- Rule of thumb: need 30+ data points per company for meaningful stats

RATE LIMITING: Always respect robots.txt and add delays between requests.
We don't want to get IP banned. For production: use a scraping service
(ScrapingBee, Apify) or buy the data from a vendor.
"""

import re
import time
import requests
from datetime import datetime
from typing import Optional


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (research project, contact: your@email.com)',
    'Accept': 'text/html,application/xhtml+xml',
}


def scrape_motley_fool_transcript(url: str, delay: float = 2.0) -> Optional[str]:
    """
    Scrape a Motley Fool earnings call transcript.
    
    Motley Fool URL pattern:
    https://www.fool.com/earnings/call-transcripts/{year}/{month}/{day}/{company-name-ticker}/
    
    WHY MOTLEY FOOL: Their transcript format is consistent, well-tagged,
    and publicly accessible. Structure:
    - <article class="article-content"> contains the full text
    - Speaker names in <strong> tags
    - Sections divided by headers
    """
    try:
        time.sleep(delay)  # be polite
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Motley Fool puts transcript in article content div
        article = soup.find('div', class_='article-content')
        if not article:
            article = soup.find('article')
        
        if not article:
            return None
        
        # Extract text while preserving structure
        paragraphs = []
        for elem in article.find_all(['p', 'h2', 'h3', 'strong']):
            text = elem.get_text(strip=True)
            if text and len(text) > 10:
                paragraphs.append(text)
        
        return '\n\n'.join(paragraphs)
    
    except Exception as e:
        print(f"Scrape failed for {url}: {e}")
        return None


def scrape_from_ir_page(ticker: str, ir_url: str) -> Optional[str]:
    """
    Many companies post transcript PDFs on their IR pages.
    Example: investor.apple.com → Earnings → Q4 2024
    
    For PDFs: use pdfplumber or PyMuPDF to extract text.
    For HTML: standard BeautifulSoup extraction.
    """
    try:
        response = requests.get(ir_url, headers=HEADERS, timeout=15)
        content_type = response.headers.get('content-type', '')
        
        if 'pdf' in content_type.lower() or ir_url.endswith('.pdf'):
            return _extract_pdf_text(response.content)
        else:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            # Remove script/style
            for tag in soup(['script', 'style', 'nav', 'footer']):
                tag.decompose()
            return soup.get_text(separator='\n', strip=True)
    
    except Exception as e:
        print(f"IR scrape failed: {e}")
        return None


def _extract_pdf_text(pdf_bytes: bytes) -> Optional[str]:
    """Extract text from PDF bytes. Requires pdfplumber."""
    try:
        import pdfplumber
        import io
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return '\n\n'.join(pages)
    except ImportError:
        print("Install pdfplumber: pip install pdfplumber")
        return None


def build_transcript_dataset(
    companies: list,
    quarters: list,
    source: str = "motley_fool"
) -> list:
    """
    Build a dataset of transcripts across companies and quarters.
    
    Args:
        companies: list of {ticker, name, sector, ir_base_url}
        quarters: list of {quarter, year, month, day}
        source: "motley_fool" | "ir_page"
    
    Returns: list of transcript dicts ready for run_pipeline()
    
    PRODUCTION CHECKLIST:
    □ Add error handling + retry logic (tenacity library)
    □ Persist raw HTML alongside extracted text (for debugging)
    □ Add a seen-URLs cache (don't re-scrape what you have)
    □ Consider async scraping (httpx + asyncio) for speed
    □ Store raw transcripts in S3/GCS before processing
    """
    transcripts = []
    
    for company in companies:
        for quarter in quarters:
            print(f"Fetching {company['ticker']} {quarter['quarter']}...")
            
            if source == "motley_fool":
                # Build MF URL (this pattern works for many companies)
                slug = company['name'].lower().replace(' ', '-').replace(',', '')
                url = (f"https://www.fool.com/earnings/call-transcripts/"
                       f"{quarter['year']}/{quarter['month']:02d}/{quarter['day']:02d}/"
                       f"{slug}-{company['ticker'].lower()}/")
                
                raw_text = scrape_motley_fool_transcript(url)
            
            elif source == "ir_page":
                raw_text = scrape_from_ir_page(
                    company['ticker'],
                    company['ir_base_url']
                )
            else:
                raw_text = None
            
            if raw_text:
                transcripts.append({
                    'ticker': company['ticker'],
                    'company_name': company['name'],
                    'sector': company['sector'],
                    'earnings_date': datetime(
                        quarter['year'], quarter['month'], quarter['day']
                    ),
                    'quarter': quarter['quarter'],
                    'transcript': raw_text,
                    'source_url': url if source == 'motley_fool' else company.get('ir_base_url', ''),
                })
            else:
                print(f"  → No transcript found")
            
            time.sleep(2)  # rate limit
    
    return transcripts


# ── Example target companies for a 20-company dataset ───────────────────────

SAMPLE_COMPANIES = [
    {"ticker": "AAPL", "name": "Apple",     "sector": "Technology"},
    {"ticker": "MSFT", "name": "Microsoft", "sector": "Technology"},
    {"ticker": "GOOGL","name": "Alphabet",  "sector": "Technology"},
    {"ticker": "AMZN", "name": "Amazon",    "sector": "Consumer Discretionary"},
    {"ticker": "META", "name": "Meta",      "sector": "Communication Services"},
    {"ticker": "NVDA", "name": "NVIDIA",    "sector": "Technology"},
    {"ticker": "JPM",  "name": "JPMorgan",  "sector": "Financials"},
    {"ticker": "GS",   "name": "Goldman Sachs","sector": "Financials"},
    {"ticker": "JNJ",  "name": "Johnson & Johnson","sector": "Healthcare"},
    {"ticker": "XOM",  "name": "Exxon Mobil","sector": "Energy"},
]

SAMPLE_QUARTERS = [
    {"quarter": "Q1 2024", "year": 2024, "month": 2, "day": 1},
    {"quarter": "Q2 2024", "year": 2024, "month": 5, "day": 2},
    {"quarter": "Q3 2024", "year": 2024, "month": 8, "day": 1},
    {"quarter": "Q4 2024", "year": 2024, "month": 11, "day": 1},
]


if __name__ == "__main__":
    # Demo: show what a scrape attempt looks like
    print("Sample transcript fetch (Motley Fool):")
    print("URL pattern: https://www.fool.com/earnings/call-transcripts/...")
    print("In a real run with ~200 transcripts, this takes ~10 minutes with 2s delays")
    print()
    print("Companies configured:", len(SAMPLE_COMPANIES))
    print("Quarters configured:", len(SAMPLE_QUARTERS))
    print("Total calls to fetch:", len(SAMPLE_COMPANIES) * len(SAMPLE_QUARTERS))
