from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

# Embedding model
embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

# ROUGE scorer
rouge = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True
)

def compute_bleu(reference, prediction):

    reference_tokens = [reference.split()]
    prediction_tokens = prediction.split()

    return sentence_bleu(
        reference_tokens,
        prediction_tokens
    )

def compute_rouge(reference, prediction):

    scores = rouge.score(reference, prediction)

    return scores["rougeL"].fmeasure

def compute_embedding_similarity(reference, prediction):

    embeddings = embedding_model.encode(
        [reference, prediction]
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return float(similarity)

def run_heuristic_eval(samples):

    bleu_scores = []
    rouge_scores = []
    embedding_scores = []

    for sample in samples:

        reference = sample["ground_truth"]
        prediction = sample["answer"]

        bleu = compute_bleu(
            reference,
            prediction
        )

        rouge_l = compute_rouge(
            reference,
            prediction
        )

        embedding_sim = compute_embedding_similarity(
            reference,
            prediction
        )

        bleu_scores.append(bleu)
        rouge_scores.append(rouge_l)
        embedding_scores.append(embedding_sim)

    return {
        "bleu": np.mean(bleu_scores),
        "rouge_l": np.mean(rouge_scores),
        "embedding_similarity": np.mean(embedding_scores)
    }