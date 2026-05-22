import json
import re
from pathlib import Path

DATA_DIR = Path("data/processed")
OUT_FILE = Path("data/qa_dataset.json")
MIN_PER_COMPANY = 20
MAX_PER_COMPANY = 30

MONEY_RE = re.compile(
    r"(?<!\w)(?:\$\s*)?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion|bn|mn|m|b|%))?(?!\w)",
    flags=re.IGNORECASE,
)
STRICT_VALUE_RE = re.compile(
    r"^\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|trillion|bn|mn|m|b|%)?$",
    flags=re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b(20\d{2})\b")
SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
COMPARATIVE_RE = re.compile(
    r"\b(increased|decreased|change|growth|decline|improvement|reduction|from prior year)\b",
    flags=re.IGNORECASE,
)
EXPLICIT_DIRECT_RE = re.compile(
    r"\b(was|total|reported|equaled|amounted\s+to)\b",
    flags=re.IGNORECASE,
)
MIN_FINANCIAL_CONFIDENCE = 4

SEGMENT_NAMES = [
    "investment management",
    "global markets",
    "wealth management",
    "asset management",
    "platform solutions",
    "investment banking",
    "institutional securities",
    "consumer and community banking",
    "commercial banking",
    "corporate and investment bank",
]

GEOGRAPHY_NAMES = [
    "americas",
    "emea",
    "asia",
    "asia pacific",
    "europe",
    "north america",
    "united states",
    "u.s.",
]

FACT_SPECS = [
    {"type": "total_revenue", "patterns": [r"\btotal revenue\b", r"\bnet revenues\b", r"\brevenues\b"], "question": "What was {company} total revenue in {year}?"},
    {"type": "net_income", "patterns": [r"\bnet income\b", r"\bnet earnings\b", r"\bnet loss\b", r"\bincome \(loss\)\b"], "question": "What was {company} net income in {year}?"},
    {"type": "operating_income", "patterns": [r"\boperating income\b", r"\bincome from operations\b", r"\boperating profit\b"], "question": "What was {company} operating income in {year}?"},
    {"type": "assets", "patterns": [r"\btotal assets\b"], "question": "What were {company} total assets in {year}?"},
    {"type": "liabilities", "patterns": [r"\btotal liabilities\b"], "question": "What were {company} total liabilities in {year}?"},
    {"type": "aum", "patterns": [r"\bassets under management\b", r"\baum\b"], "question": "What was {company} assets under management (AUM) in {year}?"},
    {"type": "segment_revenue", "patterns": [r"\bsegment\b", r"\bsegment revenue\b", r"\brevenues\b"], "question": "What was {segment} segment revenue for {company} in {year}?"},
    {"type": "technology_revenue", "patterns": [r"\btechnology\b", r"\btechnology services\b", r"\bplatform\b", r"\bdigital\b"], "question": "What technology-related revenue did {company} report in {year}?"},
    {"type": "geographic_revenue", "patterns": [r"\bemea\b", r"\bamericas\b", r"\basia\b", r"\bgeographic\b", r"\bregion\b"], "question": "What was {geography} revenue for {company} in {year}?"},
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_company_and_default_year(filename: str):
    stem = Path(filename).stem
    parts = stem.split("_")
    company = parts[0].replace("_", " ").title()
    default_year = parts[-1] if parts and parts[-1].isdigit() else "2024"
    return company, default_year


def split_sentences(text: str):
    raw = SPLIT_RE.split(text)
    out = []
    for s in raw:
        s = normalize_text(s)
        if len(s) < 30:
            continue
        if len(s) > 700:
            continue
        out.append(s)
    return out


def pick_best_numeric(sentence: str, keyword_positions):
    matches = list(MONEY_RE.finditer(sentence))
    if not matches:
        return None

    if not keyword_positions:
        return matches[0].group(0)

    best = None
    best_dist = 10**9
    for m in matches:
        m_pos = m.start()
        dist = min(abs(m_pos - kp) for kp in keyword_positions)
        if dist < best_dist:
            best_dist = dist
            best = m.group(0)

    if best_dist > 120:
        return None
    return best


def has_statement_structure(sentence: str, fact_type: str, answer: str) -> bool:
    s = normalize_text(sentence)

    patterns = {
        "total_revenue": [r"\b(total revenue|net revenues|revenues)\b"],
        "net_income": [r"\b(net income|net earnings|net loss|income \(loss\))\b"],
        "operating_income": [r"\b(operating income|income from operations|operating profit)\b"],
        "assets": [r"\btotal assets\b"],
        "liabilities": [r"\btotal liabilities\b"],
        "aum": [r"\b(assets under management|aum)\b"],
        "segment_revenue": [r"\bsegment\b", r"\brevenues?\b"],
        "technology_revenue": [r"\b(technology|technology services|platform|digital)\b", r"\brevenues?\b"],
        "geographic_revenue": [r"\b(emea|americas|asia|region|geographic)\b", r"\brevenues?\b"],
    }

    keys = patterns.get(fact_type, [])
    if not keys:
        return False

    # Statement-like lines: keyword + connector + value OR keyword directly followed by value token.
    connector = r"(?:was|were|reported|equaled|amounted\s+to|:|=)"
    for k in keys:
        if re.search(k, s, flags=re.IGNORECASE):
            if re.search(k + r".{0,60}" + connector + r".{0,40}" + re.escape(answer), s, flags=re.IGNORECASE):
                return True
            if re.search(k + r".{0,40}" + re.escape(answer), s, flags=re.IGNORECASE):
                return True

    return False


def financial_fact_confidence(sentence: str, fact_type: str, answer: str) -> int:
    s = sentence.lower()
    score = 0

    # Strong signal: statement row structure.
    if has_statement_structure(sentence, fact_type, answer):
        score += 3

    # Exact financial value formatting.
    if STRICT_VALUE_RE.match(answer.strip()):
        score += 2

    # Explicit direct assertion words.
    if EXPLICIT_DIRECT_RE.search(sentence):
        score += 1

    # Penalize comparative/change language unless direct assertion exists.
    if COMPARATIVE_RE.search(sentence) and not EXPLICIT_DIRECT_RE.search(sentence):
        score -= 3

    # Prefer core financial statement facts.
    if fact_type in {"total_revenue", "net_income", "operating_income", "assets", "liabilities", "aum"}:
        score += 1

    # Temporal anchoring typically indicates a report fact.
    if YEAR_RE.search(s):
        score += 1

    return score


def preferred_financial_value(answer: str, fact_type: str) -> bool:
    a = answer.lower().strip()

    # For this dataset, prioritize monetary/absolute values over percentages.
    if a.endswith("%"):
        return False

    monetary_tokens = ("$", "million", "billion", "trillion", "bn", "mn", " m", " b")
    if any(t in a for t in monetary_tokens):
        return True

    # Also allow large comma-formatted values like 177,556.
    if re.search(r"\d{1,3}(?:,\d{3})+", a):
        return True

    return False


def extract_year(sentence: str, default_year: str):
    years = YEAR_RE.findall(sentence)
    if years:
        return years[0]
    return default_year


def extract_entity(sentence: str, values):
    s = sentence.lower()
    for v in values:
        if v in s:
            return v.title()
    return None


def build_question(fact_type: str, template: str, company: str, year: str, sentence: str):
    segment = extract_entity(sentence, SEGMENT_NAMES)
    geography = extract_entity(sentence, GEOGRAPHY_NAMES)

    if fact_type == "segment_revenue":
        if not segment:
            return None
        return template.format(segment=segment, company=company, year=year)

    if fact_type == "geographic_revenue":
        if not geography:
            return None
        return template.format(geography=geography, company=company, year=year)

    return template.format(company=company, year=year)


def sentence_fact_candidates(sentence: str, company: str, default_year: str):
    lower = sentence.lower()
    results = []

    for spec in FACT_SPECS:
        positions = []
        for pat in spec["patterns"]:
            for m in re.finditer(pat, lower):
                positions.append(m.start())

        # Avoid false operating-income matches from "nonoperating income".
        if spec["type"] == "operating_income" and "nonoperating income" in lower and " operating income" not in lower:
            continue

        if not positions:
            continue

        answer = pick_best_numeric(sentence, positions)
        if answer is None:
            continue

        if not re.search(r"\d", answer):
            continue

        # Reject comparative/change statements unless explicitly assertive.
        if COMPARATIVE_RE.search(sentence) and not EXPLICIT_DIRECT_RE.search(sentence):
            continue

        # Require value to look like an exact financial figure.
        if not STRICT_VALUE_RE.match(answer.strip()):
            continue

        # Require statement-like structure for higher factual precision.
        if not has_statement_structure(sentence, spec["type"], answer):
            continue

        # Prefer direct financial values (monetary/absolute), not ratio percentages.
        if not preferred_financial_value(answer, spec["type"]):
            continue

        year = extract_year(sentence, default_year)
        question = build_question(spec["type"], spec["question"], company, year, sentence)
        if not question:
            continue

        confidence = financial_fact_confidence(sentence, spec["type"], answer)
        if confidence < MIN_FINANCIAL_CONFIDENCE:
            continue

        results.append({
            "question": question,
            "ground_truth": normalize_text(answer),
            "supporting_context": sentence,
            "fact_type": spec["type"],
            "company": company,
            "year": year,
            "segment": extract_entity(sentence, SEGMENT_NAMES),
            "geography": extract_entity(sentence, GEOGRAPHY_NAMES),
            "financial_fact_confidence": confidence,
        })

    return results


def dedupe_samples(samples):
    seen = set()
    out = []
    for s in samples:
        q = normalize_text(s["question"]).lower()
        a = normalize_text(s["ground_truth"]).lower()
        c = normalize_text(s["supporting_context"]).lower()
        key = (q, a)
        strict_key = (q, a, c)

        if key in seen or strict_key in seen:
            continue

        seen.add(key)
        seen.add(strict_key)
        out.append(s)
    return out


def dedupe_by_question_best(samples):
    best = {}
    for s in samples:
        q = normalize_text(s["question"]).lower()
        rank = (s.get("financial_fact_confidence", 0), len(s.get("supporting_context", "")), len(s.get("ground_truth", "")))
        if q not in best or rank > best[q][0]:
            best[q] = (rank, s)

    out = [v[1] for v in best.values()]
    out.sort(key=lambda x: (x.get("financial_fact_confidence", 0), len(x.get("supporting_context", ""))), reverse=True)
    return out


def question_variants(sample):
    company = sample.get("company", "Company")
    year = sample.get("year", "2024")
    segment = sample.get("segment")
    geography = sample.get("geography")
    fact_type = sample.get("fact_type")

    variants = {
        "total_revenue": [
            f"What was {company} total revenue in {year}?",
            f"How much total revenue did {company} report in {year}?",
            f"What total revenues were reported by {company} in {year}?",
            f"In {year}, what was {company} revenue?",
            f"State {company}'s total revenue for {year}.",
        ],
        "net_income": [
            f"What was {company} net income in {year}?",
            f"How much net income did {company} report in {year}?",
            f"What net earnings did {company} report in {year}?",
            f"In {year}, what was {company}'s net income?",
            f"State {company}'s net income for {year}.",
        ],
        "operating_income": [
            f"What was {company} operating income in {year}?",
            f"How much operating income did {company} report in {year}?",
            f"What income from operations did {company} report in {year}?",
            f"In {year}, what was {company}'s operating income?",
            f"State {company}'s operating income for {year}.",
        ],
        "assets": [
            f"What were {company} total assets in {year}?",
            f"How much in total assets did {company} report in {year}?",
            f"In {year}, what were {company}'s total assets?",
            f"State the total assets of {company} for {year}.",
            f"What total asset value did {company} disclose in {year}?",
        ],
        "liabilities": [
            f"What were {company} total liabilities in {year}?",
            f"How much in total liabilities did {company} report in {year}?",
            f"In {year}, what were {company}'s total liabilities?",
            f"State the total liabilities of {company} for {year}.",
            f"What total liability value did {company} disclose in {year}?",
        ],
        "aum": [
            f"What was {company} assets under management (AUM) in {year}?",
            f"How much AUM did {company} report in {year}?",
            f"In {year}, what assets under management did {company} disclose?",
            f"State {company}'s AUM for {year}.",
            f"What was the AUM of {company} in {year}?",
        ],
        "technology_revenue": [
            f"What technology-related revenue did {company} report in {year}?",
            f"How much technology revenue did {company} disclose in {year}?",
            f"In {year}, what technology services revenue was reported by {company}?",
            f"State {company}'s technology revenue in {year}.",
            f"What revenue from technology activities did {company} report in {year}?",
        ],
    }

    if fact_type == "segment_revenue" and segment:
        return [
            f"What was {segment} segment revenue for {company} in {year}?",
            f"How much revenue did the {segment} segment generate for {company} in {year}?",
            f"In {year}, what was revenue for {company}'s {segment} segment?",
            f"State {segment} segment revenue for {company} in {year}.",
            f"What revenue figure is reported for the {segment} segment of {company} in {year}?",
        ]

    if fact_type == "geographic_revenue" and geography:
        return [
            f"What was {geography} revenue for {company} in {year}?",
            f"How much revenue did {company} report in {geography} in {year}?",
            f"In {year}, what revenue is disclosed for {company} in {geography}?",
            f"State {company}'s {geography} revenue in {year}.",
            f"What revenue figure did {company} report for {geography} in {year}?",
        ]

    return variants.get(fact_type, [sample.get("question", "")])


def generate_for_document(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    sentences = split_sentences(text)
    company, default_year = parse_company_and_default_year(path.name)

    candidates = []
    for sentence in sentences:
        facts = sentence_fact_candidates(sentence, company, default_year)
        for f in facts:
            f["document"] = path.name
            candidates.append(f)

    candidates = dedupe_samples(candidates)

    expanded = []
    for c in candidates:
        for q in question_variants(c):
            if not q:
                continue
            row = dict(c)
            row["question"] = q
            expanded.append(row)

    expanded = dedupe_by_question_best(expanded)

    # prioritize high precision facts; keep only requested fields
    selected = expanded[:MAX_PER_COMPANY]
    if len(selected) < MIN_PER_COMPANY:
        print(f"Warning: {path.name} produced {len(selected)} high-quality facts (< {MIN_PER_COMPANY}).")

    final = []
    for s in selected:
        final.append({
            "question": s["question"],
            "ground_truth": s["ground_truth"],
            "supporting_context": s["supporting_context"],
            "document": s["document"],
            "financial_fact_confidence": s["financial_fact_confidence"],
        })
    return final


def main():
    docs = sorted(DATA_DIR.glob("*.txt"))
    all_samples = []

    for doc in docs:
        company_samples = generate_for_document(doc)
        print(f"{doc.name}: {len(company_samples)} factual QA pairs")
        all_samples.extend(company_samples)

    all_samples = dedupe_samples(all_samples)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(all_samples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(all_samples)} samples to {OUT_FILE}")


if __name__ == "__main__":
    main()
