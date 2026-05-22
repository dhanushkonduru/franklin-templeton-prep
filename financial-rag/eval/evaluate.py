"""
RAGAS Evaluation Harness
Measures: faithfulness, answer_relevancy, context_precision, context_recall

Usage:
  python eval/evaluate.py                   # run eval on built-in test set
  python eval/evaluate.py --out results.json
"""

import json
import argparse
from pathlib import Path
from loguru import logger

# Built-in test dataset (question + ground truth pairs)
# Add your own after reviewing actual document content
TEST_DATASET = [
    {
        "question": "What was Apple's total net sales in the most recent fiscal year?",
        "ground_truth": "Apple's total net sales were $383.3 billion in fiscal year 2023.",
    },
    {
        "question": "What are the primary risk factors Apple discloses in its 10-K?",
        "ground_truth": "Apple discloses risks including global economic conditions, supply chain disruptions, intense competition, regulatory changes, cybersecurity threats, and dependence on key personnel.",
    },
    {
        "question": "How does Apple describe its competitive position?",
        "ground_truth": "Apple describes competing in highly competitive markets and believes the principal competitive factors include price, quality, innovation, and customer service.",
    },
    {
        "question": "What is Microsoft's revenue from its cloud segment?",
        "ground_truth": "Microsoft's Intelligent Cloud segment generated approximately $87.9 billion in revenue in fiscal year 2023.",
    },
    {
        "question": "What does Tesla disclose about its manufacturing risks?",
        "ground_truth": "Tesla discloses risks related to manufacturing ramp-up challenges, quality control, supply chain disruptions, and the complexity of scaling production of electric vehicles.",
    },
]


def run_evaluation(out_path: str | None = None) -> dict:
    """Run RAGAS evaluation and return metrics dict."""

    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError:
        logger.error("RAGAS not installed. Run: pip install ragas datasets")
        return {}

    from retrieval.chain import query

    logger.info(f"Running RAGAS evaluation on {len(TEST_DATASET)} questions…")

    rows = []
    for item in TEST_DATASET:
        q = item["question"]
        logger.info(f"  Evaluating: {q[:60]}…")

        result = query(q)

        rows.append({
            "question": q,
            "answer": result.answer,
            "contexts": [s.excerpt for s in result.sources],
            "ground_truth": item["ground_truth"],
        })

    dataset = Dataset.from_list(rows)

    logger.info("Computing RAGAS metrics (may take 1–2 min)…")
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    metrics = {
        "faithfulness":       round(scores["faithfulness"], 4),
        "answer_relevancy":   round(scores["answer_relevancy"], 4),
        "context_precision":  round(scores["context_precision"], 4),
        "context_recall":     round(scores["context_recall"], 4),
    }
    metrics["mean_score"] = round(sum(metrics.values()) / len(metrics), 4)

    # Print report
    logger.info("\n" + "=" * 50)
    logger.success("RAGAS Evaluation Results")
    logger.info("=" * 50)
    for k, v in metrics.items():
        bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
        logger.info(f"  {k:<22} {bar}  {v:.4f}")
    logger.info("=" * 50)
    logger.info("Target scores for production: all > 0.75")

    if out_path:
        out = {"metrics": metrics, "per_question": rows}
        Path(out_path).write_text(json.dumps(out, indent=2))
        logger.success(f"Results saved → {out_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="Save results to JSON file")
    args = parser.parse_args()
    run_evaluation(out_path=args.out)
