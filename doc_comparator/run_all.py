"""
run_all.py — Master runner for all 4 backend phases.
Usage: python run_all.py
Reads GROQ_API_KEY from .env automatically.
"""

import os
import sys
import json
import inspect
from dotenv import load_dotenv

load_dotenv()

MODULES_DIR = os.path.join(os.path.dirname(__file__), "modules")
sys.path.insert(0, MODULES_DIR)

from parser     import parse_document
from differ     import run_diff
from analyser   import run_semantic_analysis
from classifier import run_classifier

INPUT_DIR  = "input"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_input_files():
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith((".txt", ".pdf"))])
    if len(files) < 2:
        print(f"❌ Need 2 files in {INPUT_DIR}/. Found: {files}")
        sys.exit(1)
    return files[0], files[1]


def _attr(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _sections(doc):
    if isinstance(doc, list):
        return doc
    return _attr(doc, "sections", default=[])


def _call_run_diff(doc_a, doc_b):
    sig = inspect.signature(run_diff)
    if len(sig.parameters) >= 3:
        return run_diff(doc_a, doc_b, OUTPUT_DIR)
    return run_diff(doc_a, doc_b)


def _to_path(result, fallback_filename):
    """Turn any module return value into a file path string."""
    if isinstance(result, str) and os.path.exists(result):
        return result
    for attr in ("output_path", "path", "report_path", "filename"):
        val = getattr(result, attr, None)
        if val and isinstance(val, str) and os.path.exists(val):
            return val
    fallback = os.path.join(OUTPUT_DIR, fallback_filename)
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(
        f"Cannot find output file. Tried fallback: {fallback}\n"
        f"Return value was: {result!r}"
    )


def _call_analyser(diff_result, groq_key):
    """
    run_semantic_analysis first param might want a DiffReport object or a file path.
    Try the object first; if it blows up with an AttributeError fall back to path.
    """
    try:
        return run_semantic_analysis(diff_result, groq_key, OUTPUT_DIR)
    except AttributeError:
        path = _to_path(diff_result, "diff_report.json")
        return run_semantic_analysis(path, groq_key, OUTPUT_DIR)


def _call_classifier(semantic_result):
    """
    run_classifier first param might want a SemanticReport object or a file path.
    Try path first (most common); fall back to object.
    """
    path = _to_path(semantic_result, "semantic_report.json")
    try:
        return run_classifier(path, OUTPUT_DIR)
    except TypeError:
        return run_classifier(semantic_result, OUTPUT_DIR)


def main():
    print("=" * 60)
    print("  10-K Document Comparator — All Phases (1–4)")
    print("=" * 60)

    file_a, file_b = find_input_files()
    path_a = os.path.join(INPUT_DIR, file_a)
    path_b = os.path.join(INPUT_DIR, file_b)

    # ── Phase 1: Parse ────────────────────────────────────────────────────────
    parsed_docs = {}
    for label, path in [(file_a, path_a), (file_b, path_b)]:
        doc      = parse_document(path)
        stem     = os.path.splitext(label)[0]
        out_path = os.path.join(OUTPUT_DIR, f"{stem}_parsed.json")
        with open(out_path, "w") as f:
            json.dump(doc, f, indent=2,
                      default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o))
        sections    = _sections(doc)
        total_words = sum(_attr(s, "word_count", 0) for s in sections)
        print(f"   ✅ Found {len(sections)} sections | {total_words} words total")
        for s in sections:
            print(f"      Item {_attr(s,'item_id','?'):>3} │ {_attr(s,'title',''):<45} │ {_attr(s,'word_count',0):>5} words")
        print(f"   💾 Saved → {out_path}")
        parsed_docs[label] = doc

    # ── Phase 2: Diff ─────────────────────────────────────────────────────────
    diff_result = _call_run_diff(parsed_docs[file_a], parsed_docs[file_b])

    # ── Phase 3: Semantic analysis via Groq ───────────────────────────────────
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("\n❌ GROQ_API_KEY not found in .env — skipping Phase 3 & 4.")
        sys.exit(1)

    print(f"\n🤖 Running Groq semantic analysis...")
    semantic_result = _call_analyser(diff_result, groq_key)

    # ── Phase 4: Classify + final report ──────────────────────────────────────
    final_result = _call_classifier(semantic_result)
    final_path   = _to_path(final_result, "final_report.json")

    print("\n" + "=" * 60)
    print("  ✅ All done!")
    print(f"  Final report: {final_path}")
    print("  Load final_report.json into the Phase 5 React UI.")
    print("=" * 60)


if __name__ == "__main__":
    main()