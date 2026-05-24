"""
Phase 4: Classifier & Final Report Assembler
Scores every SemanticChange 1-5 for materiality and builds the final JSON report.
No API calls — pure rule-based logic on top of Phase 3 output.
"""

import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional


# ─── Materiality scoring weights ─────────────────────────────────────────────

# Keywords that indicate HIGH materiality (score +2 each)
HIGH_MATERIALITY_SIGNALS = [
    "material weakness", "restatement", "going concern", "fraud", "breach",
    "regulatory action", "sec investigation", "doj", "criminal", "sanctions",
    "bankruptcy", "default", "covenant violation", "impairment", "write-down",
    "write-off", "goodwill impairment", "layoffs", "restructuring charge",
    "class action", "material adverse", "liquidity risk", "solvency",
    "revenue recognition", "internal control", "icfr", "whistleblower",
    "acquisition", "divestiture", "merger", "spin-off", "significant transaction",
    "data breach", "cyberattack", "ransomware", "customer data",
    "patent infringement", "intellectual property litigation",
]

# Keywords that indicate LOW materiality (score -1 each)
LOW_MATERIALITY_SIGNALS = [
    "minor", "immaterial", "routine", "ordinary course", "no material",
    "no significant", "did not materially", "consistent with prior",
    "similar to", "unchanged", "nominal", "cosmetic", "typographical",
    "formatting", "boilerplate",
]

# Change-type to base score mapping
CHANGE_TYPE_BASE_SCORES = {
    "RISK_ESCALATION":    4,
    "NEW_DISCLOSURE":     4,
    "RISK_REMOVAL":       3,
    "RISK_DOWNGRADE":     3,
    "NUMBERS_CHANGED":    3,
    "TONE_SHIFT":         2,
    "COSMETIC":           1,
    "UNKNOWN":            2,
}

# Significance label from Phase 3 → score modifier
SIGNIFICANCE_MODIFIERS = {
    "high":   +1,
    "medium":  0,
    "low":    -1,
}


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ClassifiedChange:
    section_id: str
    section_title: str
    change_type: str
    materiality_score: int          # 1–5
    priority_tier: str              # CRITICAL / HIGH / MEDIUM / LOW / COSMETIC
    significance: str               # high / medium / low (from Phase 3)
    summary: str
    detail: str
    old_text: Optional[str]
    new_text: Optional[str]
    analyst_note: str               # generated guidance for the analyst


@dataclass
class FinalReport:
    generated_at: str
    doc_a_name: str
    doc_b_name: str
    total_sections_compared: int
    total_changes: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    cosmetic_count: int
    overall_risk_delta: str         # INCREASED / DECREASED / NEUTRAL
    executive_summary: str
    changes: list                   # list of ClassifiedChange dicts
    section_summaries: list         # per-section rollup


# ─── Core classifier ──────────────────────────────────────────────────────────

def _compute_materiality_score(change: dict) -> int:
    """
    Returns a score 1–5.
    Starts with a base from change_type, applies significance modifier,
    then scans summary + detail text for keyword signals.
    """
    change_type = change.get("change_type", "UNKNOWN").upper()
    significance = change.get("significance", "medium").lower()

    base = CHANGE_TYPE_BASE_SCORES.get(change_type, 2)
    modifier = SIGNIFICANCE_MODIFIERS.get(significance, 0)

    # Scan text for keyword signals
    combined_text = (
        (change.get("summary") or "") + " " +
        (change.get("detail") or "") + " " +
        (change.get("new_text") or "")
    ).lower()

    keyword_boost = 0
    for kw in HIGH_MATERIALITY_SIGNALS:
        if kw in combined_text:
            keyword_boost += 1          # cap contribution per keyword
    keyword_boost = min(keyword_boost, 2)   # max +2 from keywords

    keyword_drag = 0
    for kw in LOW_MATERIALITY_SIGNALS:
        if kw in combined_text:
            keyword_drag += 1
    keyword_drag = min(keyword_drag, 1)     # max -1 from downgrade signals

    raw = base + modifier + keyword_boost - keyword_drag
    return max(1, min(5, raw))              # clamp to [1, 5]


def _score_to_tier(score: int) -> str:
    return {
        5: "CRITICAL",
        4: "HIGH",
        3: "MEDIUM",
        2: "LOW",
        1: "COSMETIC",
    }.get(score, "LOW")


def _generate_analyst_note(change: dict, score: int, tier: str) -> str:
    """Generates a one-liner guidance note for the analyst."""
    change_type = change.get("change_type", "UNKNOWN").upper()
    section = change.get("section_title", "this section")

    notes = {
        "RISK_ESCALATION": f"[{tier}] Escalated risk in {section} — verify whether management has disclosed mitigation plans in the MD&A.",
        "NEW_DISCLOSURE":  f"[{tier}] New disclosure in {section} — check if there are related footnotes or legal proceedings that cross-reference this.",
        "RISK_REMOVAL":    f"[{tier}] Risk factor removed from {section} — confirm whether the underlying risk resolved or was quietly dropped.",
        "RISK_DOWNGRADE":  f"[{tier}] Risk language softened in {section} — compare with any quantitative data in Item 8 to check consistency.",
        "NUMBERS_CHANGED": f"[{tier}] Numerical change in {section} — cross-reference with audited financials in Item 8 and earnings release.",
        "TONE_SHIFT":      f"[{tier}] Tone shift in {section} — read in context with the full section to assess direction of confidence.",
        "COSMETIC":        f"[COSMETIC] Formatting or boilerplate change in {section} — no analyst action required.",
    }
    return notes.get(change_type, f"[{tier}] Review change in {section} carefully.")


def _detect_overall_risk_delta(changes: list) -> str:
    """Heuristic: count risk-up vs risk-down signals across all changes."""
    risk_up   = sum(1 for c in changes if c.change_type.upper() in ("RISK_ESCALATION", "NEW_DISCLOSURE"))
    risk_down = sum(1 for c in changes if c.change_type.upper() in ("RISK_REMOVAL", "RISK_DOWNGRADE"))
    if risk_up > risk_down + 1:
        return "INCREASED"
    elif risk_down > risk_up + 1:
        return "DECREASED"
    return "NEUTRAL"


def _build_executive_summary(classified: list, risk_delta: str, doc_a: str, doc_b: str) -> str:
    """Generates a human-readable paragraph-style executive summary."""
    critical = [c for c in classified if c.priority_tier == "CRITICAL"]
    high     = [c for c in classified if c.priority_tier == "HIGH"]
    total    = len(classified)

    risk_phrase = {
        "INCREASED": "overall risk profile has materially increased",
        "DECREASED": "overall risk profile appears to have decreased",
        "NEUTRAL":   "overall risk profile is broadly unchanged",
    }.get(risk_delta, "risk profile has shifted")

    summary = (
        f"Comparing {doc_b} against {doc_a}, {total} semantic changes were identified across "
        f"the analysed sections. The {risk_phrase} year-over-year."
    )

    if critical:
        # Deduplicate by section so the summary doesn't repeat "Item 1A" three times
        seen = set()
        deduped = []
        for c in critical:
            key = (c.section_id, c.change_type)
            if key not in seen:
                seen.add(key)
                deduped.append(c)
        names = ", ".join(f"Item {c.section_id} ({c.change_type})" for c in deduped[:3])
        summary += f" CRITICAL findings requiring immediate analyst review: {names}."
    if high:
        summary += f" {len(high)} HIGH-priority change(s) were also flagged for follow-up."
    if not critical and not high:
        summary += " No critical or high-priority changes were identified; changes appear largely cosmetic or low-impact."

    return summary


def _build_section_summaries(classified: list) -> list:
    """Rolls up per-section stats for the UI sidebar."""
    from collections import defaultdict
    sections = defaultdict(lambda: {"changes": [], "max_score": 0})

    for c in classified:
        key = c.section_id
        sections[key]["section_id"]    = c.section_id
        sections[key]["section_title"] = c.section_title
        sections[key]["changes"].append(c.priority_tier)
        sections[key]["max_score"]     = max(sections[key]["max_score"], c.materiality_score)

    result = []
    for sid, data in sorted(sections.items()):
        tiers = data["changes"]
        result.append({
            "section_id":    data["section_id"],
            "section_title": data["section_title"],
            "change_count":  len(tiers),
            "max_score":     data["max_score"],
            "max_tier":      _score_to_tier(data["max_score"]),
            "tier_breakdown": {
                "CRITICAL": tiers.count("CRITICAL"),
                "HIGH":     tiers.count("HIGH"),
                "MEDIUM":   tiers.count("MEDIUM"),
                "LOW":      tiers.count("LOW"),
                "COSMETIC": tiers.count("COSMETIC"),
            }
        })
    return result


# ─── Main entry point ─────────────────────────────────────────────────────────

def run_classifier(semantic_report_path: str, output_dir: str = "output") -> str:
    """
    Reads the Phase 3 semantic_analysis.json, classifies every change,
    and writes final_report.json to output_dir.
    Returns the path to the final report.
    """
    print("\n🏷️  Phase 4: Classifying and scoring changes...")
    print("─" * 60)

    # Load Phase 3 output
    with open(semantic_report_path, "r") as f:
        semantic = json.load(f)

    print(f"  📂 Semantic report keys: {list(semantic.keys())}")

    doc_a = semantic.get("doc_a", semantic.get("document_a", semantic.get("file_a", "Document A")))
    doc_b = semantic.get("doc_b", semantic.get("document_b", semantic.get("file_b", "Document B")))

    # Try every key name the analyser might use for the sections list
    all_section_analyses = (
        semantic.get("section_analyses")
        or semantic.get("sections")
        or semantic.get("results")
        or semantic.get("analyses")
        or semantic.get("section_results")
        or []
    )

    # If still empty, auto-detect: find the first list-of-dicts value
    if not all_section_analyses:
        print(f"  ⚠️  Keys in semantic report: {list(semantic.keys())}")
        for key, val in semantic.items():
            if isinstance(val, list) and val and isinstance(val[0], dict):
                print(f"  ℹ️  Auto-detected section data under key: '{key}'")
                all_section_analyses = val
                break

    classified_changes: list[ClassifiedChange] = []

    for section_analysis in all_section_analyses:
        # Tolerate different field names the analyser might use
        section_id = (
            section_analysis.get("section_id")
            or section_analysis.get("item_id")
            or section_analysis.get("item")
            or "?"
        )
        section_title = (
            section_analysis.get("section_title")
            or section_analysis.get("title")
            or section_analysis.get("name")
            or "Unknown Section"
        )
        changes = (
            section_analysis.get("changes")
            or section_analysis.get("semantic_changes")
            or section_analysis.get("findings")
            or []
        )

        if not changes:
            print(f"  Item {section_id:>3} — no changes to classify, skipping")
            continue

        section_count = 0
        for change in changes:
            # Inject section info so downstream functions can use it
            change["section_id"]    = section_id
            change["section_title"] = section_title

            score   = _compute_materiality_score(change)
            tier    = _score_to_tier(score)
            note    = _generate_analyst_note(change, score, tier)

            classified_changes.append(ClassifiedChange(
                section_id       = section_id,
                section_title    = section_title,
                change_type      = change.get("change_type", "UNKNOWN").upper(),
                materiality_score= score,
                priority_tier    = tier,
                significance     = change.get("significance", "medium"),
                summary          = change.get("summary", ""),
                detail           = change.get("detail", ""),
                old_text         = change.get("old_text"),
                new_text         = change.get("new_text"),
                analyst_note     = note,
            ))
            section_count += 1

        tier_counts = {}
        for c in classified_changes[-section_count:]:
            tier_counts[c.priority_tier] = tier_counts.get(c.priority_tier, 0) + 1

        tier_str = "  ".join(f"{t}:{n}" for t, n in tier_counts.items())
        print(f"  Item {section_id:>3} — {section_count} changes classified  [{tier_str}]")

    # Sort by materiality score descending
    classified_changes.sort(key=lambda c: c.materiality_score, reverse=True)

    # Build counts
    tier_map = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "COSMETIC": 0}
    for c in classified_changes:
        tier_map[c.priority_tier] = tier_map.get(c.priority_tier, 0) + 1

    risk_delta       = _detect_overall_risk_delta(classified_changes)
    exec_summary     = _build_executive_summary(classified_changes, risk_delta, doc_a, doc_b)
    section_summaries = _build_section_summaries(classified_changes)

    report = FinalReport(
        generated_at              = datetime.utcnow().isoformat() + "Z",
        doc_a_name                = doc_a,
        doc_b_name                = doc_b,
        total_sections_compared   = len(all_section_analyses),
        total_changes             = len(classified_changes),
        critical_count            = tier_map["CRITICAL"],
        high_count                = tier_map["HIGH"],
        medium_count              = tier_map["MEDIUM"],
        low_count                 = tier_map["LOW"],
        cosmetic_count            = tier_map["COSMETIC"],
        overall_risk_delta        = risk_delta,
        executive_summary         = exec_summary,
        changes                   = [asdict(c) for c in classified_changes],
        section_summaries         = section_summaries,
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "final_report.json")
    with open(out_path, "w") as f:
        json.dump(asdict(report), f, indent=2)

    print("─" * 60)
    print(f"\n📊 CLASSIFICATION SUMMARY")
    print(f"   Total changes : {report.total_changes}")
    print(f"   🔴 CRITICAL   : {report.critical_count}")
    print(f"   🟠 HIGH       : {report.high_count}")
    print(f"   🟡 MEDIUM     : {report.medium_count}")
    print(f"   🟢 LOW        : {report.low_count}")
    print(f"   ⚪ COSMETIC   : {report.cosmetic_count}")
    print(f"   Risk delta    : {report.overall_risk_delta}")
    print(f"\n📝 Executive Summary:")
    print(f"   {report.executive_summary}")
    print(f"\n💾 Saved final report → {out_path}")

    return out_path


if __name__ == "__main__":
    import sys
    semantic_path = sys.argv[1] if len(sys.argv) > 1 else "output/semantic_analysis.json"
    run_classifier(semantic_path)