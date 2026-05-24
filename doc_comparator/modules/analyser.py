"""
Phase 3: Semantic Analysis via Groq
=====================================
Takes the diff report from Phase 2 and sends each changed section
to Groq's LLM to extract *meaning-level* changes — not just what
text was swapped, but what the company is actually saying differently.

Key design decisions:
  - Only sections flagged needs_semantic=True are sent (cost gate from Phase 2)
  - Chunking strategy: sections under TOKEN_LIMIT go single-shot;
    larger sections are split into chunks and reassembled
  - Structured JSON output enforced via prompt — no free-form prose
  - Rate limiting built in to stay within Groq's free tier limits
  - Results merged back into the diff report for Phase 4

Output: semantic_report.json
"""

import os
import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field

from groq import Groq
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from differ import load_diff, DiffReport, SectionDiff


# ── Config ────────────────────────────────────────────────────────────────────

MODEL            = "llama-3.3-70b-versatile"   # best Groq model for analysis
TOKEN_LIMIT      = 6000    # chars per section before we switch to chunked mode
                           # (~1500 tokens; keeps each call well under context limits)
RATE_LIMIT_DELAY = 1.5     # seconds between API calls (Groq free tier: ~30 req/min)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class SemanticChange:
    """One atomic meaning-level change identified by the LLM."""
    change_type: str        # "new_disclosure" | "risk_escalation" | "risk_de-escalation"
                            # "financial_revision" | "strategic_shift" | "removed_disclosure"
                            # "policy_change" | "cosmetic"
    summary: str            # one-sentence plain-English summary of the change
    old_stance: str         # what the old document said / implied (empty if new)
    new_stance: str         # what the new document says / implies (empty if removed)
    significance: str       # "high" | "medium" | "low"
    analyst_note: str       # why an analyst should care (or "" if cosmetic)

@dataclass
class SectionSemanticResult:
    """Semantic analysis result for one section."""
    item_id: str
    title: str
    overall_summary: str          # 2-3 sentence summary of all changes in this section
    changes: list[SemanticChange]
    groq_model: str
    tokens_used: int
    chunked: bool                 # True if section was too long and chunked

@dataclass
class SemanticReport:
    """Top-level output of Phase 3."""
    old_filename: str
    new_filename: str
    sections_analysed: int
    sections_skipped: int         # unchanged sections not sent to Groq
    total_tokens_used: int
    results: list[SectionSemanticResult]


# ── Prompt engineering ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior financial analyst specialising in SEC filings.
Your job is to identify MEANINGFUL changes between two versions of a 10-K section —
not formatting or cosmetic rewording, but genuine shifts in what the company is disclosing,
admitting, claiming, or omitting.

You must respond with ONLY a valid JSON object — no preamble, no markdown fences.
The JSON must exactly follow the schema provided in each user message."""


def _build_user_prompt(item_id: str, title: str, old_text: str, new_text: str) -> str:
    return f"""Compare these two versions of Item {item_id} ({title}) from a company's 10-K filing.

=== OLD VERSION (prior year) ===
{old_text}

=== NEW VERSION (current year) ===
{new_text}

Identify all meaningful semantic changes. Ignore pure formatting or whitespace differences.

Respond with ONLY this JSON structure (no markdown, no extra text):
{{
  "overall_summary": "<2-3 sentences summarising what changed in this section overall>",
  "changes": [
    {{
      "change_type": "<one of: new_disclosure | risk_escalation | risk_de-escalation | financial_revision | strategic_shift | removed_disclosure | policy_change | cosmetic>",
      "summary": "<one sentence: what changed>",
      "old_stance": "<what the old doc said or implied — empty string if this is a new addition>",
      "new_stance": "<what the new doc says or implies — empty string if this was removed>",
      "significance": "<high | medium | low>",
      "analyst_note": "<why an analyst should care — empty string if cosmetic>"
    }}
  ]
}}

Focus on changes that matter to investors: new risks, changed financial guidance,
new legal exposure, shifts in business strategy, and changes in tone around uncertainty."""


def _build_chunk_prompt(item_id: str, title: str, chunk_idx: int,
                         total_chunks: int, old_chunk: str, new_chunk: str) -> str:
    return f"""Compare chunk {chunk_idx+1} of {total_chunks} for Item {item_id} ({title}).

=== OLD VERSION — chunk {chunk_idx+1}/{total_chunks} ===
{old_chunk}

=== NEW VERSION — chunk {chunk_idx+1}/{total_chunks} ===
{new_chunk}

Respond with ONLY this JSON structure:
{{
  "overall_summary": "<summary of changes in this chunk only>",
  "changes": [
    {{
      "change_type": "<new_disclosure|risk_escalation|risk_de-escalation|financial_revision|strategic_shift|removed_disclosure|policy_change|cosmetic>",
      "summary": "<one sentence>",
      "old_stance": "<old text or empty>",
      "new_stance": "<new text or empty>",
      "significance": "<high|medium|low>",
      "analyst_note": "<why it matters or empty>"
    }}
  ]
}}"""


# ── Chunking strategy ─────────────────────────────────────────────────────────

def _chunk_text(text: str, max_chars: int) -> list[str]:
    """
    Split text into chunks at paragraph boundaries.
    Never cuts mid-sentence — always looks for a double-newline near the limit.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    remaining = text
    while len(remaining) > max_chars:
        # Find last paragraph break before the limit
        cut = remaining[:max_chars].rfind('\n\n')
        if cut == -1:
            # No paragraph break — fall back to last sentence boundary
            cut = remaining[:max_chars].rfind('. ')
        if cut == -1:
            cut = max_chars  # last resort: hard cut
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()

    if remaining:
        chunks.append(remaining)
    return chunks


# ── Groq API calls ────────────────────────────────────────────────────────────

def _call_groq(client: Groq, user_prompt: str) -> tuple[dict, int]:
    """
    Single Groq API call. Returns (parsed_json, tokens_used).
    Raises on API error or JSON parse failure.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.1,      # low temperature = consistent, factual output
        max_tokens=2048,
    )

    raw = response.choices[0].message.content.strip()
    tokens = response.usage.total_tokens

    # Strip markdown fences if the model adds them despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\n---\n{raw[:300]}")

    return data, tokens


def _analyse_section_single(
    client: Groq, sd: SectionDiff, old_text: str, new_text: str
) -> SectionSemanticResult:
    """Single-shot analysis for sections within token limit."""
    prompt = _build_user_prompt(sd.item_id, sd.title, old_text, new_text)
    data, tokens = _call_groq(client, prompt)

    changes = [SemanticChange(**c) for c in data.get("changes", [])]
    return SectionSemanticResult(
        item_id=sd.item_id,
        title=sd.title,
        overall_summary=data.get("overall_summary", ""),
        changes=changes,
        groq_model=MODEL,
        tokens_used=tokens,
        chunked=False,
    )


def _analyse_section_chunked(
    client: Groq, sd: SectionDiff, old_text: str, new_text: str
) -> SectionSemanticResult:
    """
    Chunked analysis for large sections.
    Strategy: chunk both old and new independently at the same chunk_size,
    then diff chunk[i] vs chunk[i] in parallel calls, reassemble results.
    """
    old_chunks = _chunk_text(old_text, TOKEN_LIMIT)
    new_chunks = _chunk_text(new_text, TOKEN_LIMIT)
    total = max(len(old_chunks), len(new_chunks))

    all_changes: list[SemanticChange] = []
    all_summaries: list[str] = []
    total_tokens = 0

    for i in range(total):
        old_c = old_chunks[i] if i < len(old_chunks) else "(no content in prior year)"
        new_c = new_chunks[i] if i < len(new_chunks) else "(no content in current year)"

        prompt = _build_chunk_prompt(sd.item_id, sd.title, i, total, old_c, new_c)
        data, tokens = _call_groq(client, prompt)

        all_changes.extend([SemanticChange(**c) for c in data.get("changes", [])])
        all_summaries.append(data.get("overall_summary", ""))
        total_tokens += tokens

        if i < total - 1:
            time.sleep(RATE_LIMIT_DELAY)

    # Merge chunk summaries into one with a final Groq call
    if len(all_summaries) > 1:
        merge_prompt = f"""These are summaries of changes found in different chunks of Item {sd.item_id} ({sd.title}):

{chr(10).join(f'Chunk {i+1}: {s}' for i, s in enumerate(all_summaries))}

Write a single 2-3 sentence overall_summary combining these. Respond with ONLY:
{{"overall_summary": "<your combined summary>"}}"""
        merge_data, merge_tokens = _call_groq(client, merge_prompt)
        overall = merge_data.get("overall_summary", " ".join(all_summaries))
        total_tokens += merge_tokens
    else:
        overall = all_summaries[0] if all_summaries else ""

    return SectionSemanticResult(
        item_id=sd.item_id,
        title=sd.title,
        overall_summary=overall,
        changes=all_changes,
        groq_model=MODEL,
        tokens_used=total_tokens,
        chunked=True,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def run_semantic_analysis(
    diff_report: DiffReport,
    groq_api_key = None,
    output_dir: str = "output",
) -> SemanticReport:
    """
    Main entry point for Phase 3.
    Iterates over sections needing semantic analysis and calls Groq.
    """
    api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("No Groq API key found. Set GROQ_API_KEY in .env or pass it directly.")
    client = Groq(api_key=api_key)

    # Rebuild old/new text from hunks so we have clean section text
    # (avoids re-reading the original files)
    def _reconstruct(sd: SectionDiff, side: str) -> str:
        parts = []
        for h in sd.hunks:
            if side == "old" and h.kind in ("unchanged", "removed", "modified"):
                parts.append(h.old_text)
            elif side == "new" and h.kind in ("unchanged", "added", "modified"):
                parts.append(h.new_text)
        return " ".join(parts)

    to_analyse = [sd for sd in diff_report.section_diffs if sd.needs_semantic]
    skipped    = len(diff_report.section_diffs) - len(to_analyse)

    print(f"\n🤖 Groq Semantic Analysis")
    print(f"   Model  : {MODEL}")
    print(f"   Sections to analyse : {len(to_analyse)}")
    print(f"   Sections skipped    : {skipped} (unchanged)\n")

    results: list[SectionSemanticResult] = []
    total_tokens = 0

    for i, sd in enumerate(to_analyse):
        old_text = _reconstruct(sd, "old")
        new_text = _reconstruct(sd, "new")

        # Decide: single-shot or chunked?
        needs_chunking = len(old_text) + len(new_text) > TOKEN_LIMIT * 2
        mode = "chunked" if needs_chunking else "single-shot"

        print(f"  [{i+1}/{len(to_analyse)}] Item {sd.item_id} — {sd.title[:40]}")
        print(f"         {len(old_text):,} → {len(new_text):,} chars | mode: {mode}")

        try:
            if needs_chunking:
                result = _analyse_section_chunked(client, sd, old_text, new_text)
            else:
                result = _analyse_section_single(client, sd, old_text, new_text)

            results.append(result)
            total_tokens += result.tokens_used

            # Show quick preview
            high = sum(1 for c in result.changes if c.significance == "high")
            print(f"         ✅ {len(result.changes)} changes found ({high} high-significance)")
            print(f"         📝 {result.overall_summary[:120]}...")

        except Exception as e:
            print(f"         ❌ Error: {e}")
            # Don't crash the whole run — record a failed result and continue
            results.append(SectionSemanticResult(
                item_id=sd.item_id, title=sd.title,
                overall_summary=f"Analysis failed: {str(e)[:200]}",
                changes=[], groq_model=MODEL, tokens_used=0, chunked=False,
            ))

        # Rate limiting between sections
        if i < len(to_analyse) - 1:
            time.sleep(RATE_LIMIT_DELAY)

    report = SemanticReport(
        old_filename=diff_report.old_filename,
        new_filename=diff_report.new_filename,
        sections_analysed=len(results),
        sections_skipped=skipped,
        total_tokens_used=total_tokens,
        results=results,
    )

    # Save
    out_path = Path(output_dir) / "semantic_report.json"
    out_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False))
    print(f"\n💾 Saved semantic report → {out_path}")
    print(f"   Total tokens used: {total_tokens:,}")

    return report


def load_semantic(json_path: str) -> SemanticReport:
    data = json.loads(Path(json_path).read_text())
    results = []
    for r in data["results"]:
        r["changes"] = [SemanticChange(**c) for c in r["changes"]]
        results.append(SectionSemanticResult(**r))
    data["results"] = results
    return SemanticReport(**data)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyser.py <groq_api_key> [diff_report.json]")
        sys.exit(1)

    api_key    = sys.argv[1]
    diff_path  = sys.argv[2] if len(sys.argv) > 2 else "output/diff_report.json"

    diff = load_diff(diff_path)
    run_semantic_analysis(diff, api_key)