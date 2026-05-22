# ESG Analyzer

Production-style NLP pipeline for sustainability reports.

## Problem Statement

Sustainability reports are long, inconsistent, and hard to compare at scale. The goal of this project is to ingest ESG PDFs, extract structured metrics, classify disclosures into ESG taxonomy buckets, detect greenwashing signals, and produce a consistent scorecard that can be compared across companies and years.

## What We Built

- PDF ingestion and text cleaning for sustainability reports
- ESG segmentation into Environmental, Social, and Governance sections
- Regex-based metric extraction for common ESG signals
- Groq-backed LLM fallback for structured extraction when rules are not enough
- SASB/GRI-style taxonomy classification
- Greenwashing risk detection from claims vs. measurable disclosures
- ESG scoring engine with weighted components
- Batch processing for multiple companies
- Streamlit dashboard for review and comparison
- JSON output for downstream analysis

## How It Works

1. Parse the report PDF with PyMuPDF.
2. Clean and chunk the extracted text.
3. Segment the document into ESG-related sections.
4. Extract metrics with regex patterns first.
5. Use Groq LLM extraction as a fallback for missing structured fields.
6. Classify the disclosure against taxonomy keywords.
7. Detect greenwashing patterns using claim/disclosure balance.
8. Combine the results into a final ESG score and export JSON.

## What We Used

- Python 3.11+
- PyMuPDF for PDF parsing
- Pydantic for data models and validation
- Groq API with `llama-3.3-70b-versatile` for LLM extraction
- spaCy-compatible NLP utilities and regex patterns
- pandas for tabular analysis
- Streamlit for the dashboard
- Plotly for visualizations
- pytest for testing

## Repository Layout

```text
esg-analyzer/
├── main.py
├── config.py
├── utils.py
├── pipeline/
│   ├── pdf_parser.py
│   ├── segmenter.py
│   ├── extractor.py
│   └── llm_extractor.py
├── taxonomy/
│   └── classifier.py
├── analysis/
│   ├── greenwashing.py
│   └── esg_score.py
├── models/
│   └── schemas.py
├── dashboard/
│   └── dashboard.py
├── tests/
│   └── test_esg_analyzer.py
└── data/reports/
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Create a `.env` file:

```bash
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
LOG_LEVEL=INFO
USE_LLM_EXTRACTION=true
SAVE_INTERMEDIATE_RESULTS=true
```

## Run

Process all reports in `data/reports/`:

```bash
python main.py
```

Launch the dashboard:

```bash
streamlit run dashboard/dashboard.py
```

## Output

The pipeline writes structured JSON to `outputs/` and a summary report that includes:

- Company name and report year
- Extracted ESG metrics
- Taxonomy classification scores
- Greenwashing risk level
- Final ESG score and component breakdown

## Notes

- Generated artifacts such as `outputs/`, `logs/`, `.env`, and `__pycache__/` are intentionally excluded from version control.
- The codebase is built to be extensible, so additional metrics or taxonomy rules can be added without changing the orchestration layer.
