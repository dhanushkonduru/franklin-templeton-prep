# Financial RAG Hallucination Evaluation Harness

This project measures hallucinations in a financial Retrieval-Augmented Generation (RAG) system. It uses annual report PDFs, retrieves supporting text, generates answers with Groq, and checks whether the answer is grounded in the source material.

## 1. What Problem This Project Solves

Financial questions often need exact answers, such as revenue, net income, assets, liabilities, and segment results. A wrong number is more harmful than a vague answer.

This project helps answer a practical question:

> Does the RAG system give answers that are supported by the annual report, or does it make things up?

## 2. Why Hallucination Detection Matters in Finance

Hallucinations are risky in finance because they can change the meaning of a report, distort performance analysis, or invent unsupported figures. A system that sounds confident but gives the wrong number is dangerous.

In finance, we care about:
- exact values,
- traceability to source text,
- clear evidence for every answer,
- safe handling of missing information.

## 3. What RAG Is

RAG means Retrieval-Augmented Generation.

The model does not answer from memory alone. Instead, it:
1. retrieves relevant text from documents,
2. gives that text to the LLM,
3. generates an answer from the retrieved context.

This is a better fit for financial documents because answers should come from the filing, not from general model memory.

## 4. What We Built

We built a financial QA and hallucination analysis harness with these parts:
- document ingestion from financial PDFs,
- text chunking and embeddings,
- Qdrant vector storage,
- retrieval and reranking,
- Groq-based answer generation,
- NLI verification,
- RAGAS, heuristic, and hallucination analysis,
- a Streamlit app for interactive testing.

## 5. Step-by-Step Architecture

Simple flow:

```text
PDFs
↓
Text extraction / processed files
↓
Chunking
↓
Embeddings
↓
Qdrant
↓
Retriever
↓
Reranker
↓
Groq LLM
↓
NLI verification
↓
Evaluation + reports
```

How the main pieces fit together:
- `rag/ingest.py` builds the Qdrant index from processed text files.
- `rag/retriever.py` finds relevant chunks using embeddings and metadata.
- `rag/reranker.py` reorders the retrieved chunks for better relevance.
- `rag/generator.py` sends the query and contexts to Groq.
- `rag/verifier.py` checks whether the answer is supported by the context.
- `compare_evaluators.py` runs the evaluator comparison.
- `analysis_report.py` summarizes the saved evaluation results.

## 6. Technologies Used

- Python
- LlamaIndex
- Qdrant local vector database
- HuggingFace embeddings: `BAAI/bge-small-en-v1.5`
- SentenceTransformers cross-encoder reranker
- Groq Llama model for generation
- Transformers NLI model for verification
- RAGAS for evaluation
- BLEU and ROUGE for heuristic comparison
- Streamlit for the UI

## 7. Evaluation Metrics Used

### RAGAS

RAGAS measures how well the answer matches the context and whether the retrieved context is useful.

Main ideas:
- faithfulness: does the answer stay within the evidence?
- answer relevancy: does the answer address the question?
- context precision: are the retrieved chunks relevant?
- context recall: did retrieval capture the needed evidence?

### BLEU and ROUGE

These are overlap-based metrics.
- BLEU checks word overlap.
- ROUGE checks sequence overlap.

They are useful as quick signals, but they are limited:
- a correct answer can score low if it is paraphrased,
- a wrong answer can score high if it shares similar words.

### NLI Verification

The verifier uses a natural language inference model.
- entailment means the context supports the answer,
- neutral means the context does not clearly support or reject it,
- contradiction means the answer conflicts with the context.

This is the most direct hallucination check in the project.

## 8. Hallucination Types

We use three simple categories:

- **Faithful**: the answer is supported by the context.
- **Intrinsic hallucination**: the answer conflicts with the context.
- **Extrinsic hallucination**: the answer adds unsupported facts, often numbers, that are not in the context.

Examples from this project:
- a contradiction triggers the NLI-based hallucination warning,
- an unsupported revenue number can be classified as extrinsic,
- a context-supported financial statement answer is faithful.

## 9. Key Findings From Experiments

The saved experiment artifacts in `results/` show that evaluator disagreement is real and important.

From the current saved summary files:
- precision: 1.00
- recall: 0.67
- F1: 0.80
- evaluator agreement: 59%
- total hallucinations: 50
- intrinsic hallucinations: 25
- extrinsic hallucinations: 25

What this means:
- the NLI-based checker is useful for catching unsupported answers,
- heuristics alone can miss hallucinations,
- evaluator disagreement is expected and should be analyzed, not ignored.

## 10. How to Run the Project

Create or activate your virtual environment, then run:

```bash
python rag/ingest.py
python generate_dataset.py
python compare_evaluators.py
python analysis_report.py
streamlit run app.py
```

To validate dataset and hallucination logic, you can also run the small test scripts if they are present in your workspace.

## 11. How Streamlit Works

`app.py` is the UI entry point.

It does three things:
1. takes a user question,
2. runs the RAG pipeline,
3. shows the answer, verification result, and retrieved context.

This gives a simple way to inspect what the system retrieved and how the verifier judged the output.

## 12. Example Outputs

The project writes evaluation outputs to `results/`.

Useful files:
- `results/final_eval.json`
- `results/summary_report.txt`
- `results/summary_metrics.csv`

These files show:
- the generated answer,
- heuristic scores,
- NLI label and score,
- hallucination type,
- evaluator disagreement summaries.

## 13. Future Improvements

- Improve retrieval ranking for tables and numeric rows.
- Add more company-year coverage.
- Add more precise citation-level evaluation.
- Improve RAGAS stability on small local datasets.
- Expand evaluator comparison to include more failure analysis.

## Repository Layout

```text
hallucination-eval-harness/
├── data/
├── rag/
│   ├── retriever.py
│   ├── reranker.py
│   ├── generator.py
│   ├── verifier.py
│   ├── pipeline.py
│   └── ingest.py
├── evaluation/
│   ├── nli_eval.py
│   ├── hallucination.py
│   └── utils.py
├── evaluators/
├── results/
├── app.py
├── compare_evaluators.py
├── analysis_report.py
├── generate_dataset.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Notes

- This project is intentionally simple and local.
- It is focused on measuring hallucinations, not on adding new features.
- The goal is to make the answer generation process more observable and easier to trust.