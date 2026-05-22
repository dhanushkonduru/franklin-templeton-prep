import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from pathlib import Path

from tqdm import tqdm

from evaluators import heuristic_eval
from evaluators import ragas_eval
from evaluation.hallucination import classify_hallucination

OUT_DIR = Path("results")
OUT_FILE = OUT_DIR / "final_eval.json"
DATA_FILE = Path("data/qa_dataset.json")


def safe_run_ragas(samples):
    if not samples:
        return []

    print(samples[0])

    for sample in samples:
        if not isinstance(sample.get("contexts"), list):
            raise TypeError("RAGAS sample contexts must be a list of strings")

    results = []
    for index, sample in enumerate(samples):
        try:
            single_result = ragas_eval.run_ragas([sample])

            faith = None
            relev = None
            if isinstance(single_result, dict):
                faith = single_result.get("faithfulness") or single_result.get("faithfulness_score")
                relev = single_result.get("answer_relevancy") or single_result.get("relevancy")

            if isinstance(faith, list):
                faith = faith[0] if faith else None
            if isinstance(relev, list):
                relev = relev[0] if relev else None

            results.append({
                "index": index,
                "ragas_faithfulness": float(faith) if faith is not None else None,
                "ragas_relevancy": float(relev) if relev is not None else None,
            })
        except Exception as e:
            print("RAGAS evaluation failed for sample:", sample.get("question", ""))
            print(e)
            continue

    return results


def main():
    if not DATA_FILE.exists():
        raise SystemExit("Dataset not found. Run generate_dataset.py first.")

    # Ensure GROQ API key is available for generation
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY not set in environment. Set it before running compare_evaluators.py")

    # Import run_rag here to avoid importing generator at module import time
    from rag.pipeline import run_rag

    qa = json.loads(DATA_FILE.read_text(encoding="utf-8"))

    results = []
    ragas_samples = []

    for item in tqdm(qa, desc="Running RAG and heuristics"):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")

        # Run RAG pipeline (retrieval + generation + verification)
        rag_result = run_rag(question)

        answer = rag_result.get("answer", "")
        contexts = rag_result.get("contexts", [])
        verification = rag_result.get("verification", {})

        # Heuristic metrics per-sample
        bleu = heuristic_eval.compute_bleu(ground_truth, answer)
        rouge = heuristic_eval.compute_rouge(ground_truth, answer)
        emb_sim = heuristic_eval.compute_embedding_similarity(ground_truth, answer)

        # NLI label/score
        nli_label = verification.get("label") if isinstance(verification, dict) else None
        nli_score = verification.get("score") if isinstance(verification, dict) else None

        # Hallucination classification
        hal = classify_hallucination(answer, contexts[0] if contexts else "", verification)

        res = {
            "question": question,
            "answer": answer,
            "ragas_faithfulness": None,
            "ragas_relevancy": None,
            "bleu": bleu,
            "rouge": rouge,
            "embedding_similarity": emb_sim,
            "nli_label": nli_label,
            "nli_score": nli_score,
            "hallucination_type": hal.get("type"),
            "hallucinated": hal.get("hallucinated", False)
        }

        results.append(res)

        # Prepare ragas batch sample
        ragas_samples.append({
            "question": question,
            "answer": answer,
            "contexts": contexts if isinstance(contexts, list) else [str(contexts)],
            "ground_truth": ground_truth
        })

    # Run RAGAS safely per sample and continue if any single sample fails.
    ragas_results = safe_run_ragas(ragas_samples)
    for item in ragas_results:
        idx = item["index"]
        results[idx]["ragas_faithfulness"] = item["ragas_faithfulness"]
        results[idx]["ragas_relevancy"] = item["ragas_relevancy"]

    # Write per-question results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Summary statistics
    total = len(results)
    hallucinated = sum(1 for r in results if r.get("hallucinated"))
    entailment = sum(1 for r in results if (r.get("nli_label") and str(r.get("nli_label")).lower() == "entailment"))
    contradiction = sum(1 for r in results if (r.get("nli_label") and str(r.get("nli_label")).lower() == "contradiction"))

    avg_ragas_faith = None
    ragas_vals = [r["ragas_faithfulness"] for r in results if r["ragas_faithfulness"] is not None]
    if ragas_vals:
        avg_ragas_faith = sum(ragas_vals) / len(ragas_vals)

    avg_rouge = sum(r["rouge"] for r in results) / total if total else 0.0

    summary = {
        "total": total,
        "hallucination_rate": hallucinated / total if total else 0.0,
        "entailment_rate": entailment / total if total else 0.0,
        "contradiction_rate": contradiction / total if total else 0.0,
        "average_ragas_faithfulness": avg_ragas_faith,
        "average_rouge": avg_rouge
    }

    print("Evaluation summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
