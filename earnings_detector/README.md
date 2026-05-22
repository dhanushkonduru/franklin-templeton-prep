# Earnings Call Sentiment and Surprise Detector

This project answers a simple question: can we read earnings call transcripts and tell whether the market agrees with management’s tone?

The idea is to take an earnings call, break it into sentences, score the language, store the results, and compare the text signal with the stock’s reaction around the earnings date.

## What the project does

The pipeline does four main things:

1. Gets transcript text from a source or sample data.
2. Splits the transcript into sentences and separates prepared remarks from Q&A.
3. Scores each sentence for sentiment, hedging, and forward-looking language.
4. Compares the transcript signal with stock movement using an event study.

The most interesting case is when the language and the price disagree. For example, management may sound positive, but the stock falls. That kind of mismatch is the “surprise” signal.

## What question we are trying to answer

The project is trying to answer this:

- Does positive or negative language in earnings calls line up with the stock’s reaction?
- Do uncertain or cautious words matter?
- Can we spot cases where the market does not believe what management says?

## What we used

- Python for the main code.
- FinBERT for financial sentiment.
- Rule-based matching for hedging language.
- A simple forward-looking vs historical classifier.
- SQLAlchemy and SQLite for storage.
- Pandas, NumPy, SciPy, and yfinance for the event study.

## How it works

1. `scripts/scraper.py` gets transcript text.
2. `models/parser.py` breaks the transcript into sentences and labels sections like MD&A and Q&A.
3. `models/scorer.py` scores each sentence.
4. `models/database.py` stores companies, transcripts, sentences, and scores.
5. `analytics/event_study.py` checks the stock’s abnormal return around earnings.
6. `scripts/run_pipeline.py` connects everything and writes the final results.

## What the outputs mean

The output tells you:

- whether the transcript tone was positive, neutral, or negative,
- how much hedging language was used,
- how much of the call was forward-looking,
- whether the stock went up or down around earnings,
- whether the language and price reaction were aligned or divergent.

## Why this project matters

This is a finance and NLP project. It shows how text can become a signal, not just a summary.

It is useful because it combines:

- natural language processing,
- financial analysis,
- database storage,
- and event-study style market measurement.

## Simple interview answer

If someone asks what you built, you can say:

"I built a pipeline that reads earnings call transcripts, scores the language with finance-aware NLP, stores the results, and compares the tone of the call with the stock’s reaction after earnings. The goal is to find cases where the language and price disagree, because those are the most interesting signals."

## Current repo state

The repo now focuses on source code and documentation. Generated files like the local database and exported results are not part of the long-term source.
