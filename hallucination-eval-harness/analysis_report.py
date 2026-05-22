import csv
import json
from pathlib import Path

from sklearn.metrics import precision_score, recall_score, f1_score

RESULTS_FILE = Path("results/final_eval.json")
SUMMARY_TXT = Path("results/summary_report.txt")
SUMMARY_CSV = Path("results/summary_metrics.csv")


def to_bool(x):
    return bool(x)


def nli_vote(item):
    """Binary vote from NLI evaluator for hallucination."""
    label = str(item.get("nli_label", "")).lower()
    if label == "contradiction":
        return 1
    if label == "entailment":
        return 0
    # Treat neutral as potentially unsupported
    return 1


def heuristic_vote(item):
    """Binary vote from heuristic metrics for hallucination.

    Conservative rule: low lexical overlap and low semantic similarity.
    """
    rouge = float(item.get("rouge", 0.0) or 0.0)
    emb = float(item.get("embedding_similarity", 0.0) or 0.0)
    return 1 if (rouge < 0.10 and emb < 0.55) else 0


def classifier_vote(item):
    return 1 if to_bool(item.get("hallucinated", False)) else 0


def agreement_percentage(items):
    if not items:
        return 0.0

    agree = 0
    for item in items:
        votes = [nli_vote(item), heuristic_vote(item), classifier_vote(item)]
        if votes[0] == votes[1] == votes[2]:
            agree += 1

    return (agree / len(items)) * 100.0


def disagreement_items(items):
    out = []
    for item in items:
        votes = {
            "nli": nli_vote(item),
            "heuristic": heuristic_vote(item),
            "classifier": classifier_vote(item),
        }
        if len(set(votes.values())) > 1:
            score = abs(votes["nli"] - votes["heuristic"]) + abs(votes["nli"] - votes["classifier"]) + abs(votes["heuristic"] - votes["classifier"])
            out.append((score, item, votes))

    out.sort(key=lambda x: (x[0], float(x[1].get("nli_score", 0.0) or 0.0)), reverse=True)
    return out


def worst_hallucinations(items, k=5):
    rows = [x for x in items if to_bool(x.get("hallucinated", False))]
    rows.sort(
        key=lambda x: (
            float(x.get("rouge", 0.0) or 0.0),
            float(x.get("embedding_similarity", 0.0) or 0.0),
            -float(x.get("nli_score", 0.0) or 0.0),
        )
    )
    return rows[:k]


def best_grounded(items, k=5):
    rows = [x for x in items if not to_bool(x.get("hallucinated", False))]

    def quality(x):
        rouge = float(x.get("rouge", 0.0) or 0.0)
        emb = float(x.get("embedding_similarity", 0.0) or 0.0)
        faith = x.get("ragas_faithfulness")
        faith = float(faith) if faith is not None else 0.0
        nli = float(x.get("nli_score", 0.0) or 0.0)
        return rouge + emb + faith + nli

    rows.sort(key=quality, reverse=True)
    return rows[:k]


def compute_metrics(items):
    total = len(items)

    total_h = sum(1 for x in items if to_bool(x.get("hallucinated", False)))
    intrinsic = sum(1 for x in items if str(x.get("hallucination_type", "")).lower() == "intrinsic")
    extrinsic = sum(1 for x in items if str(x.get("hallucination_type", "")).lower() == "extrinsic")

    # Ground truth proxy: NLI says unsupported if not entailment.
    y_true = [nli_vote(x) for x in items]
    y_pred = [classifier_vote(x) for x in items]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    agreement = agreement_percentage(items)

    return {
        "total_samples": total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "evaluator_agreement_percentage": agreement,
        "total_hallucinations": total_h,
        "intrinsic_hallucinations": intrinsic,
        "extrinsic_hallucinations": extrinsic,
    }


def write_summary_csv(metrics):
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def format_item(item):
    return (
        f"question: {item.get('question', '')}\n"
        f"answer: {item.get('answer', '')}\n"
        f"nli_label: {item.get('nli_label', '')} | nli_score: {item.get('nli_score', '')}\n"
        f"rouge: {item.get('rouge', '')} | embedding_similarity: {item.get('embedding_similarity', '')}\n"
        f"hallucination_type: {item.get('hallucination_type', '')} | hallucinated: {item.get('hallucinated', '')}\n"
    )


def write_summary_report(metrics, top_disagreements, worst_h, best_g):
    lines = []
    lines.append("Summary Metrics")
    lines.append("=" * 40)
    for k, v in metrics.items():
        lines.append(f"{k}: {v}")

    lines.append("\nTop 5 Disagreements")
    lines.append("=" * 40)
    if not top_disagreements:
        lines.append("No evaluator disagreements found.")
    else:
        for i, (_, item, votes) in enumerate(top_disagreements[:5], start=1):
            lines.append(f"\n{i}. votes={votes}")
            lines.append(format_item(item))

    lines.append("\nWorst Hallucination Examples")
    lines.append("=" * 40)
    if not worst_h:
        lines.append("No hallucination examples found.")
    else:
        for i, item in enumerate(worst_h, start=1):
            lines.append(f"\n{i}.")
            lines.append(format_item(item))

    lines.append("\nBest Grounded Answers")
    lines.append("=" * 40)
    if not best_g:
        lines.append("No grounded examples found.")
    else:
        for i, item in enumerate(best_g, start=1):
            lines.append(f"\n{i}.")
            lines.append(format_item(item))

    SUMMARY_TXT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")


def main():
    if not RESULTS_FILE.exists():
        raise SystemExit("results/final_eval.json not found. Run compare_evaluators.py first.")

    items = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))

    metrics = compute_metrics(items)
    top_dis = disagreement_items(items)[:5]
    worst_h = worst_hallucinations(items, k=5)
    best_g = best_grounded(items, k=5)

    write_summary_csv(metrics)
    write_summary_report(metrics, top_dis, worst_h, best_g)

    print("Top 5 disagreements")
    for i, (_, item, votes) in enumerate(top_dis, start=1):
        print(f"{i}. votes={votes} | question={item.get('question', '')}")

    print("\nWorst hallucination examples")
    for i, item in enumerate(worst_h, start=1):
        print(f"{i}. {item.get('question', '')} | type={item.get('hallucination_type', '')} | rouge={item.get('rouge', 0)}")

    print("\nBest grounded answers")
    for i, item in enumerate(best_g, start=1):
        print(f"{i}. {item.get('question', '')} | nli={item.get('nli_label', '')} | rouge={item.get('rouge', 0)}")

    print(f"\nWrote {SUMMARY_TXT}")
    print(f"Wrote {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
