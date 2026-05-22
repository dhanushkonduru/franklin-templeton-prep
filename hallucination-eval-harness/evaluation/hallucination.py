import re


NUM_RE = re.compile(r"[\$€£]?\d[\d,\.\%]*")


def _numbers_in_text(text: str):
    if not text:
        return []
    return NUM_RE.findall(text)


def classify_hallucination(answer, context, verification):
    label = "" if not verification else str(verification.get("label", "")).lower()

    if label == "contradiction":
        return {"type": "intrinsic", "hallucinated": True}

    if label == "neutral":
        nums = _numbers_in_text(answer)
        if nums:
            # check if any numeric claim is missing from context
            missing = []
            ctx = context or ""
            for n in nums:
                if n not in ctx:
                    # also check digits-only variant
                    digits = re.sub(r"[^0-9]", "", n)
                    if digits and digits not in ctx:
                        missing.append(n)
            if missing:
                return {"type": "extrinsic", "hallucinated": True}

    if label == "entailment":
        return {"type": "faithful", "hallucinated": False}

    return {"type": "unknown", "hallucinated": False}
