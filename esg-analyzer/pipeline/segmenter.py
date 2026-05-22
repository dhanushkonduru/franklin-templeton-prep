"""
ESG Topic Segmentation.
Segments text into Environmental, Social, and Governance categories.
"""

import re
import logging
from typing import Dict, List, Tuple
from config import Config, setup_logging
from models.schemas import DocumentSegmentation

logger = setup_logging(__name__)


class ESGSegmenter:
    """Segment documents by ESG topic."""
    
    def __init__(self):
        """Initialize segmenter."""
        self.config = Config()
        self._build_keyword_patterns()
    
    def _build_keyword_patterns(self):
        """Build regex patterns for ESG keywords."""
        self.environmental_keywords = [
            "carbon", "emissions", "ghg", "co2", "renewable", "energy",
            "water", "climate", "environmental", "sustainability", "green",
            "net zero", "zero emissions", "solar", "wind", "hydroelectric",
            "waste", "recycling", "circular", "biodiversity", "pollution"
        ]
        
        self.social_keywords = [
            "employee", "human rights", "labor", "diversity", "inclusion",
            "community", "social", "supplier", "training", "development",
            "health", "safety", "working conditions", "stakeholder",
            "education", "local", "workforce"
        ]
        
        self.governance_keywords = [
            "governance", "board", "audit", "compliance", "risk", "ethics",
            "policy", "regulation", "management", "leadership", "executive",
            "transparency", "accountability", "control", "oversight",
            "code of conduct", "anti-corruption", "legal"
        ]
        
        # Build regex patterns (case-insensitive)
        self.env_pattern = self._build_pattern(self.environmental_keywords)
        self.social_pattern = self._build_pattern(self.social_keywords)
        self.gov_pattern = self._build_pattern(self.governance_keywords)
    
    def _build_pattern(self, keywords: List[str]) -> re.Pattern:
        """Build regex pattern from keywords."""
        escaped_keywords = [re.escape(kw) for kw in keywords]
        pattern_str = r'\b(' + '|'.join(escaped_keywords) + r')\b'
        return re.compile(pattern_str, re.IGNORECASE)
    
    def segment_text(self, text: str) -> DocumentSegmentation:
        """
        Segment text by ESG category.
        
        Args:
            text: Document text to segment
            
        Returns:
            DocumentSegmentation with segmented text
        """
        logger.info(f"Segmenting text ({len(text)} chars)")
        
        sentences = self._split_into_sentences(text)
        
        environmental_text = []
        social_text = []
        governance_text = []
        other_text = []
        
        for sentence in sentences:
            category = self._classify_sentence(sentence)
            
            if category == "environmental":
                environmental_text.append(sentence)
            elif category == "social":
                social_text.append(sentence)
            elif category == "governance":
                governance_text.append(sentence)
            else:
                other_text.append(sentence)
        
        segmentation = DocumentSegmentation(
            environmental_text="\n".join(environmental_text),
            social_text="\n".join(social_text),
            governance_text="\n".join(governance_text),
            other_text="\n".join(other_text),
            document_length=len(text)
        )
        
        logger.info(
            f"Segmentation complete: E={len(environmental_text)} sentences, "
            f"S={len(social_text)}, G={len(governance_text)}, O={len(other_text)}"
        )
        
        return segmentation
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting on periods, exclamation marks, question marks
        sentences = re.split(r'[.!?]+', text)
        
        # Filter out very short sentences and strip whitespace
        sentences = [s.strip() for s in sentences if len(s.strip()) > self.config.MIN_SENTENCE_LENGTH]
        
        return sentences
    
    def _classify_sentence(self, sentence: str) -> str:
        """
        Classify a sentence into ESG category.
        
        Args:
            sentence: Sentence to classify
            
        Returns:
            Category: 'environmental', 'social', 'governance', or 'other'
        """
        env_matches = len(self.env_pattern.findall(sentence))
        social_matches = len(self.social_pattern.findall(sentence))
        gov_matches = len(self.gov_pattern.findall(sentence))
        
        # Get the category with most matches
        if env_matches >= social_matches and env_matches >= gov_matches and env_matches > 0:
            return "environmental"
        elif social_matches >= gov_matches and social_matches > 0:
            return "social"
        elif gov_matches > 0:
            return "governance"
        else:
            return "other"
    
    def get_category_stats(self, segmentation: DocumentSegmentation) -> Dict[str, int]:
        """
        Get statistics on segmentation.
        
        Args:
            segmentation: Segmentation result
            
        Returns:
            Dictionary with sentence counts per category
        """
        return {
            "environmental": len(self._split_into_sentences(segmentation.environmental_text)),
            "social": len(self._split_into_sentences(segmentation.social_text)),
            "governance": len(self._split_into_sentences(segmentation.governance_text)),
            "other": len(self._split_into_sentences(segmentation.other_text)),
        }