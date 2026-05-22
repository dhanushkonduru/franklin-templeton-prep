import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from transformers import pipeline

# Load NLI model
nli_pipeline = pipeline(
    "text-classification",
    model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    device=-1
)

def check_entailment(premise, hypothesis):
    result = nli_pipeline(
        f"{premise} </s></s> {hypothesis}"
    )

    return result


def run_nli_eval(samples):
    results = []

    for sample in samples:
        result = check_entailment(
            sample["context"],
            sample["answer"]
        )

        results.append({
            "question": sample["question"],
            "result": result
        })

    return results