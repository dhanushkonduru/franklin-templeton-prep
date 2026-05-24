"""
Phase 2: Textual Diff Engine
==============================
Runs a section-aware textual diff between two parsed 10-K documents.
Produces structured hunk objects per section — the raw material for
Phase 3 (semantic analysis) and Phase 4 (classification).

Key decisions made here:
  - difflib.SequenceMatcher (Python's Myers-family implementation)
  - Diff at SENTENCE level (not line/word) — better signal for financial prose
  - Sections matched by item_id; missing sections flagged as fully added/removed
  - Change ratio computed per section to gate Phase 3 LLM calls (skip trivial)

Output: diff_report.json  (one DiffReport object)
"""

import re
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from difflib import SequenceMatcher
from typing import Literal

# bring in Phase 1 models
sys.path.insert(0, str(Path(__file__).parent))
from parser import ParsedDocument, Section, load_parsed


# ── Data models ───────────────────────────────────────────────────────────────

ChangeKind = Literal["added", "removed", "modified", "unchanged"]

@dataclass
class Hunk:
    """A single contiguous block of change within a section."""
    kind: ChangeKind          # added | removed | modified | unchanged
    old_text: str             # text in the OLD document (empty if added)
    new_text: str             # text in the NEW document (empty if removed)
    old_index: int            # sentence index in old section (-1 if added)
    new_index: int            # sentence index in new section (-1 if removed)

@dataclass
class SectionDiff:
    """Diff result for one matched section pair."""
    item_id: str
    title: str
    status: ChangeKind        # overall: added | removed | modified | unchanged
    change_ratio: float       # 0.0 (identical) → 1.0 (completely different)
    old_word_count: int
    new_word_count: int
    word_delta: int           # new - old (negative = shrunk)
    hunks: list[Hunk]
    needs_semantic: bool      # True → send to Groq in Phase 3

@dataclass
class DiffReport:
    """Top-level output of Phase 2."""
    old_filename: str
    new_filename: str
    total_sections_old: int
    total_sections_new: int
    sections_added: list[str]      # item_ids only in new doc
    sections_removed: list[str]    # item_ids only in old doc
    sections_compared: int
    sections_changed: int
    sections_needing_semantic: int
    section_diffs: list[SectionDiff]


# ── Sentence tokeniser ────────────────────────────────────────────────────────

# Split on sentence boundaries while keeping financial abbreviations intact.
# We avoid splitting on "approx.", "U.S.", "No.", "$12.4", decimals, etc.
_SENT_END = re.compile(
    r'(?<!\bMr)(?<!\bMs)(?<!\bMrs)(?<!\bDr)(?<!\bNo)(?<!\bvs)'
    r'(?<!\bapprox)(?<!\bU\.S)(?<!\bFig)(?<!\bSec)'
    r'(?<=[.!?])\s+(?=[A-Z\'\"(])'
)

def _to_sentences(text: str) -> list[str]:
    """
    Split a section's text into sentence-level units.
    Financial documents often have one idea per sentence — diffing at this
    granularity surfaces meaningful changes better than line-level diff.
    Empty/whitespace-only strings are dropped.
    """
    # Normalise whitespace first
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    raw = _SENT_END.split(text.strip())
    return [s.strip() for s in raw if s.strip()]


# ── Core diff logic ───────────────────────────────────────────────────────────

# If a section's change_ratio is BELOW this threshold, skip the Groq call —
# it's either cosmetic whitespace or identical. Tune this as needed.
SEMANTIC_THRESHOLD = 0.08   # 8% or more of sentences changed → worth analysing

def _diff_section_text(old_text: str, new_text: str) -> tuple[list[Hunk], float]:
    """
    Core algorithm: sentence-level SequenceMatcher diff.

    SequenceMatcher uses the Ratcliff/Obershelp algorithm (related to Myers diff)
    to find the longest common subsequence. It returns "opcodes" — instructions
    for transforming the old sequence into the new one:

      'equal'   → both sides have the same sentence (no change)
      'insert'  → sentence exists only in new (added)
      'delete'  → sentence exists only in old (removed)
      'replace' → a block in old is replaced by a block in new (modified)

    We map these opcodes → Hunk objects.
    """
    old_sents = _to_sentences(old_text)
    new_sents = _to_sentences(new_text)

    if not old_sents and not new_sents:
        return [], 0.0

    # SequenceMatcher junk function: ignore very short sentences (headers,
    # standalone numbers) which match trivially across sections
    def is_junk(s):
        return len(s.strip()) < 8

    matcher = SequenceMatcher(
        isjunk=is_junk,
        a=old_sents,
        b=new_sents,
        autojunk=False,   # disable autojunk — financial docs have repetitive boilerplate
    )

    hunks: list[Hunk] = []
    changed_sentences = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                hunks.append(Hunk(
                    kind="unchanged",
                    old_text=old_sents[i1 + k],
                    new_text=new_sents[j1 + k],
                    old_index=i1 + k,
                    new_index=j1 + k,
                ))
        elif tag == "insert":
            for k in range(j2 - j1):
                hunks.append(Hunk(
                    kind="added",
                    old_text="",
                    new_text=new_sents[j1 + k],
                    old_index=-1,
                    new_index=j1 + k,
                ))
            changed_sentences += (j2 - j1)
        elif tag == "delete":
            for k in range(i2 - i1):
                hunks.append(Hunk(
                    kind="removed",
                    old_text=old_sents[i1 + k],
                    new_text="",
                    old_index=i1 + k,
                    new_index=-1,
                ))
            changed_sentences += (i2 - i1)
        elif tag == "replace":
            # Pair up old/new sentences to create "modified" hunks.
            # If counts differ, surplus sentences become pure added/removed.
            old_block = old_sents[i1:i2]
            new_block = new_sents[j1:j2]
            pairs = min(len(old_block), len(new_block))
            for k in range(pairs):
                hunks.append(Hunk(
                    kind="modified",
                    old_text=old_block[k],
                    new_text=new_block[k],
                    old_index=i1 + k,
                    new_index=j1 + k,
                ))
            # Handle surplus
            for k in range(pairs, len(old_block)):
                hunks.append(Hunk(kind="removed", old_text=old_block[k],
                                  new_text="", old_index=i1+k, new_index=-1))
            for k in range(pairs, len(new_block)):
                hunks.append(Hunk(kind="added", old_text="",
                                  new_text=new_block[k], old_index=-1, new_index=j1+k))
            changed_sentences += max(len(old_block), len(new_block))

    total = max(len(old_sents), len(new_sents), 1)
    change_ratio = round(changed_sentences / total, 4)
    return hunks, change_ratio


# ── Section matching ──────────────────────────────────────────────────────────

def _match_sections(
    old_doc: ParsedDocument,
    new_doc: ParsedDocument,
) -> tuple[list[str], list[str], list[tuple[Section, Section]]]:
    """
    Align sections between two documents by item_id.
    Returns:
      added_ids   — item_ids present only in new doc
      removed_ids — item_ids present only in old doc
      pairs       — list of (old_section, new_section) tuples
    """
    old_map = {s.item_id: s for s in old_doc.sections}
    new_map = {s.item_id: s for s in new_doc.sections}

    all_ids = sorted(
        set(old_map) | set(new_map),
        key=lambda x: (int(re.sub(r'[A-Z]', '', x)) if re.sub(r'[A-Z]', '', x).isdigit() else 99, x)
    )

    added   = [i for i in all_ids if i not in old_map]
    removed = [i for i in all_ids if i not in new_map]
    pairs   = [(old_map[i], new_map[i]) for i in all_ids if i in old_map and i in new_map]

    return added, removed, pairs


# ── Public API ────────────────────────────────────────────────────────────────

def run_diff(old_doc: ParsedDocument, new_doc: ParsedDocument) -> DiffReport:
    """
    Main entry point for Phase 2.
    Diffs every matched section pair and assembles a DiffReport.
    """
    print(f"\n🔍 Diffing: {old_doc.filename}  →  {new_doc.filename}")

    added_ids, removed_ids, pairs = _match_sections(old_doc, new_doc)

    section_diffs: list[SectionDiff] = []

    # Fully added sections (only in new doc)
    new_map = {s.item_id: s for s in new_doc.sections}
    for item_id in added_ids:
        sec = new_map[item_id]
        sents = _to_sentences(sec.content)
        hunks = [Hunk("added", "", s, -1, i) for i, s in enumerate(sents)]
        section_diffs.append(SectionDiff(
            item_id=item_id, title=sec.title, status="added",
            change_ratio=1.0, old_word_count=0, new_word_count=sec.word_count,
            word_delta=sec.word_count, hunks=hunks, needs_semantic=True,
        ))

    # Fully removed sections (only in old doc)
    old_map = {s.item_id: s for s in old_doc.sections}
    for item_id in removed_ids:
        sec = old_map[item_id]
        sents = _to_sentences(sec.content)
        hunks = [Hunk("removed", s, "", i, -1) for i, s in enumerate(sents)]
        section_diffs.append(SectionDiff(
            item_id=item_id, title=sec.title, status="removed",
            change_ratio=1.0, old_word_count=sec.word_count, new_word_count=0,
            word_delta=-sec.word_count, hunks=hunks, needs_semantic=True,
        ))

    # Matched section pairs — run the diff
    for old_sec, new_sec in pairs:
        hunks, change_ratio = _diff_section_text(old_sec.content, new_sec.content)

        if change_ratio == 0.0:
            status = "unchanged"
        else:
            status = "modified"

        needs_semantic = change_ratio >= SEMANTIC_THRESHOLD

        section_diffs.append(SectionDiff(
            item_id=old_sec.item_id,
            title=new_sec.title,
            status=status,
            change_ratio=change_ratio,
            old_word_count=old_sec.word_count,
            new_word_count=new_sec.word_count,
            word_delta=new_sec.word_count - old_sec.word_count,
            hunks=hunks,
            needs_semantic=needs_semantic,
        ))

    # Sort by item_id for consistent output
    section_diffs.sort(
        key=lambda d: (int(re.sub(r'[A-Z]', '', d.item_id))
                       if re.sub(r'[A-Z]', '', d.item_id).isdigit() else 99, d.item_id)
    )

    changed      = sum(1 for d in section_diffs if d.status != "unchanged")
    need_sem     = sum(1 for d in section_diffs if d.needs_semantic)

    report = DiffReport(
        old_filename=old_doc.filename,
        new_filename=new_doc.filename,
        total_sections_old=len(old_doc.sections),
        total_sections_new=len(new_doc.sections),
        sections_added=added_ids,
        sections_removed=removed_ids,
        sections_compared=len(pairs),
        sections_changed=changed,
        sections_needing_semantic=need_sem,
        section_diffs=section_diffs,
    )

    # Pretty print summary
    print(f"\n{'─'*60}")
    print(f"  {'ITEM':<6} {'TITLE':<40} {'RATIO':>6}  STATUS")
    print(f"{'─'*60}")
    for d in section_diffs:
        flag = "🔴" if d.change_ratio > 0.4 else "🟡" if d.change_ratio > 0.08 else "🟢"
        sem  = " [→Groq]" if d.needs_semantic else ""
        print(f"  {d.item_id:<6} {d.title[:38]:<40} {d.change_ratio:>5.0%}  {flag} {d.status}{sem}")
    print(f"{'─'*60}")
    print(f"  {changed}/{len(section_diffs)} sections changed  |  {need_sem} queued for Groq\n")

    return report


def save_diff(report: DiffReport, output_dir: str = "output") -> str:
    out_path = Path(output_dir) / "diff_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False))
    print(f"💾 Saved diff report → {out_path}")
    return str(out_path)


def load_diff(json_path: str) -> DiffReport:
    data = json.loads(Path(json_path).read_text())
    diffs = []
    for sd in data["section_diffs"]:
        sd["hunks"] = [Hunk(**h) for h in sd["hunks"]]
        diffs.append(SectionDiff(**sd))
    data["section_diffs"] = diffs
    return DiffReport(**data)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python differ.py <old_parsed.json> <new_parsed.json>")
        sys.exit(1)

    old_doc = load_parsed(sys.argv[1])
    new_doc = load_parsed(sys.argv[2])
    report  = run_diff(old_doc, new_doc)
    save_diff(report)
