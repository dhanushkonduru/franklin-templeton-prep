"""
TRANSCRIPT PARSER
-----------------
Key concepts taught here:
1. Section detection: MD&A ("prepared remarks") vs Q&A — these have different
   NLP properties. Management rehearses MD&A; Q&A is more spontaneous and
   arguably contains more signal.
2. Speaker tagging: who said what matters. CFO on gross margins > analyst question.
3. Sentence segmentation: we score at sentence level. Paragraph-level averaging
   washes out strong signals buried in hedge-heavy paragraphs.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ParsedSentence:
    text: str
    section: str          # "mda" or "qa"
    speaker: str
    sentence_idx: int


@dataclass
class ParsedTranscript:
    sentences: List[ParsedSentence] = field(default_factory=list)
    mda_text: str = ""
    qa_text: str = ""


# Patterns that signal the Q&A section has started
QA_MARKERS = [
    r"question.and.answer",
    r"q\s*&\s*a\s+session",
    r"open.{0,20}questions",
    r"operator.*instructions",
    r"we.ll now.{0,30}questions",
    r"first question",
]

# Patterns that signal speaker transitions
# Matches: "John Smith - CEO" or "OPERATOR:" or "Analyst:"
SPEAKER_RE = re.compile(
    r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"   # Name (Title Case)
    r"(?:\s*[-–—]\s*(.+?))?$",               # Optional role after dash
    re.MULTILINE
)


def simple_sentence_split(text: str) -> List[str]:
    """
    WHY NOT NLTK/spaCy? We want zero heavy dependencies for this demo.
    In production: use spaCy's sentencizer — it handles "Mr. Smith said..."
    without false splits, and handles financial abbreviations like "EPS.", "vs."
    
    This regex handles the most common cases:
    - End of sentence: . ! ? followed by space+capital
    - But NOT: decimal numbers (3.5%), abbreviations (U.S.), etc.
    """
    # Protect common abbreviations
    text = re.sub(r'\b(Mr|Mrs|Dr|Jr|Sr|vs|etc|U\.S|approx|est)\.',
                  lambda m: m.group(0).replace('.', '<DOT>'), text)
    text = re.sub(r'(\d+)\.(\d+)', r'\1<DOT>\2', text)  # decimals
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Restore protected dots
    sentences = [s.replace('<DOT>', '.').strip() for s in sentences]
    return [s for s in sentences if len(s.split()) > 4]  # filter noise


def detect_section_boundary(text: str) -> int:
    """Returns char index where Q&A section begins. -1 if not found."""
    for pattern in QA_MARKERS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.start()
    return -1


def extract_speaker(line: str) -> Optional[str]:
    """Extract speaker name from a transcript line."""
    # Common patterns in earnings transcripts:
    # "Tim Cook -- Apple CEO" or "Tim Cook:" or "ANALYST:"
    patterns = [
        r'^([A-Z][a-zA-Z\s]+?)(?:\s*[-–—]{1,2}\s*|\s*:\s*)(?:Chief|CEO|CFO|COO|President|Analyst|Operator)',
        r'^([A-Z][a-zA-Z\s]{2,30}):\s',
        r'^(OPERATOR|MODERATOR|ANALYST)',
    ]
    for p in patterns:
        m = re.match(p, line.strip())
        if m:
            return m.group(1).strip()
    return None


def parse_transcript(raw_text: str) -> ParsedTranscript:
    """
    Main parsing function. Returns structured transcript.
    
    DESIGN DECISION: We parse greedily — extract as many labeled sentences
    as possible, fall back to "unknown" speaker rather than dropping sentences.
    Missing speaker labels just mean we can't do speaker-stratified analysis.
    """
    result = ParsedTranscript()
    
    # 1. Find Q&A boundary
    qa_start = detect_section_boundary(raw_text)
    if qa_start == -1:
        # No boundary found — treat all as MDA (conservative)
        result.mda_text = raw_text
        result.qa_text = ""
    else:
        result.mda_text = raw_text[:qa_start]
        result.qa_text = raw_text[qa_start:]
    
    # 2. Parse each section
    idx = 0
    for section_name, section_text in [("mda", result.mda_text), ("qa", result.qa_text)]:
        if not section_text.strip():
            continue
        
        current_speaker = "Management"
        lines = section_text.split('\n')
        current_block = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line introduces a new speaker
            new_speaker = extract_speaker(line)
            if new_speaker:
                # Flush current block
                if current_block:
                    block_text = ' '.join(current_block)
                    for sent in simple_sentence_split(block_text):
                        result.sentences.append(ParsedSentence(
                            text=sent,
                            section=section_name,
                            speaker=current_speaker,
                            sentence_idx=idx
                        ))
                        idx += 1
                current_speaker = new_speaker
                current_block = []
            else:
                current_block.append(line)
        
        # Flush remaining block
        if current_block:
            block_text = ' '.join(current_block)
            for sent in simple_sentence_split(block_text):
                result.sentences.append(ParsedSentence(
                    text=sent,
                    section=section_name,
                    speaker=current_speaker,
                    sentence_idx=idx
                ))
                idx += 1
    
    return result


# ── Minimal demo (no heavy deps) ────────────────────────────────────────────

SAMPLE_TRANSCRIPT = """
Apple Inc (AAPL) Q4 2024 Earnings Call
Date: November 1, 2024

Tim Cook -- CEO
Thank you. We are pleased to report another record quarter. Revenue grew 6% 
year-over-year to $94.9 billion. Services revenue reached an all-time high.

Luca Maestri -- CFO
Gross margin was 46.2%, up 80 basis points from the year-ago quarter.
We believe we are well positioned for continued growth in fiscal 2025,
though we remain cautious about potential headwinds in the macro environment.
We expect some softness in Greater China in the near term.

Questions and Answers

Operator
Our first question comes from Amit Daryanani at Evercore.

Amit Daryanani -- Evercore
Tim, can you talk about AI monetization timeline? When do you expect to see
meaningful revenue contribution from Apple Intelligence features?

Tim Cook -- CEO
We are very early in the AI cycle. We may see some upside in fiscal 2025,
but it's too early to quantify with precision. We think the opportunity
could be substantial over the medium term.
"""


if __name__ == "__main__":
    result = parse_transcript(SAMPLE_TRANSCRIPT)
    print(f"Parsed {len(result.sentences)} sentences")
    print(f"\nMDA sentences: {sum(1 for s in result.sentences if s.section == 'mda')}")
    print(f"Q&A sentences: {sum(1 for s in result.sentences if s.section == 'qa')}")
    print("\nFirst 5 sentences:")
    for s in result.sentences[:5]:
        print(f"  [{s.section.upper()}] {s.speaker}: {s.text[:80]}...")
