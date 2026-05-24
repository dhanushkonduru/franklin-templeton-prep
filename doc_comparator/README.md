# Long-Context Document Comparator

This project compares two long documents, such as one year's 10-K and the previous year's, and turns the raw text into a short analyst report.

It does three things:

1. Splits each document into 10-K sections.
2. Compares matching sections to find what was added, removed, or changed.
3. Classifies each change so an analyst can see what matters first.

## Why we built it

Analysts do not read a 10-K line by line just to see every tiny edit. They want to know what changed, whether the change matters, and what to review next. This tool answers that question in one place.

## What we used

- Python for the document pipeline.
- PyMuPDF for PDF and text extraction.
- A diff step to compare sections directly.
- A Groq-powered semantic analysis step for the changes that need more than a text diff.
- React in a single HTML file for the analyst dashboard.

We used both text diff and semantic analysis because they solve different problems. Text diff is fast and good for exact wording changes. Semantic analysis is better when the meaning changed but the wording did not make that obvious.

## How it works

The pipeline follows a simple flow:

1. Parse the two documents into sections.
2. Compare the sections.
3. Score and classify each change.
4. Save a final JSON report.
5. Load that report into the dashboard.

The dashboard then shows:

- the document pair,
- the total findings,
- a risk delta badge,
- severity counts,
- and expandable finding cards with old text, new text, and analyst action notes.

## Token cost management

Long documents can be expensive to process in one shot, so the system works section by section. That keeps the context smaller and makes the comparison easier to control. It also lets us use semantic analysis only where it adds value.

## Files to run

- `run_phase1_2.py` for parsing and diffing.
- `run_phase1_2_3.py` for parsing, diffing, and semantic analysis.
- `run_all.py` for the full pipeline through the final report.
- `index.html` for the analyst dashboard.

## Setup

```bash
pip install -r requirements.txt
```

## Run the pipeline

```bash
python run_all.py
```

## Use the dashboard

Run the pipeline first so it writes `output/final_report.json`, then open `index.html` in a browser and load that file, or click the demo data button to view the interface right away.

## Repo layout

```text
doc_comparator/
├── input/        source documents
├── modules/      parser, diff, semantic analysis, classifier
├── output/       generated reports
├── index.html    React analyst dashboard
├── run_all.py    full pipeline runner
└── requirements.txt
```
