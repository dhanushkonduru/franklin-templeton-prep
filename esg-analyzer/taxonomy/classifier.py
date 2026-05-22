"""
SASB/GRI Taxonomy Classification.
Classifies ESG disclosures into standard taxonomies.
"""

import re
import logging
from typing import Dict, List, Tuple
from config import Config, setup_logging
from models.schemas import TaxonomyClassification, TaxonomyScore

logger = setup_logging(__name__)


class TaxonomyClassifier:
    """Classify ESG disclosures by SASB/GRI taxonomy."""
    
    def __init__(self):
        """Initialize classifier."""
        self.config = Config()
        self.keywords = self.config.TAXONOMY_KEYWORDS
    
    def classify(self, text: str) -> TaxonomyClassification:
        """
        Classify text by ESG taxonomy.
        
        Args:
            text: Document text to classify
            
        Returns:
            TaxonomyClassification with scores for each category
        """
        logger.info(f"Classifying text ({len(text)} chars)")
        
        classification = TaxonomyClassification()
        lower_text = text.lower()
        
        # Classify Environmental subcategories
        for subcategory, keywords in self.keywords["Environmental"].items():
            score, matches, confidence = self._score_subcategory(lower_text, keywords)
            classification.Environmental[subcategory] = TaxonomyScore(
                score=min(score, 10),
                matched_terms=matches,
                confidence=confidence
            )
        
        # Classify Social subcategories
        for subcategory, keywords in self.keywords["Social"].items():
            score, matches, confidence = self._score_subcategory(lower_text, keywords)
            classification.Social[subcategory] = TaxonomyScore(
                score=min(score, 10),
                matched_terms=matches,
                confidence=confidence
            )
        
        # Classify Governance subcategories
        for subcategory, keywords in self.keywords["Governance"].items():
            score, matches, confidence = self._score_subcategory(lower_text, keywords)
            classification.Governance[subcategory] = TaxonomyScore(
                score=min(score, 10),
                matched_terms=matches,
                confidence=confidence
            )
        
        logger.info("Classification complete")
        return classification
    
    def _score_subcategory(
        self,
        text: str,
        keywords: List[str]
    ) -> Tuple[float, List[str], float]:
        """
        Score a subcategory based on keyword matches.
        
        Args:
            text: Text to search
            keywords: List of keywords to search for
            
        Returns:
            Tuple of (score, matched_terms, confidence)
        """
        matches = []
        total_occurrences = 0
        
        for keyword in keywords:
            # Use word boundaries for more accurate matching
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches_found = re.findall(pattern, text, re.IGNORECASE)
            
            if matches_found:
                matches.append(keyword)
                total_occurrences += len(matches_found)
        
        # Score: 1 point per keyword match, plus occurrence count bonus
        score = len(matches) + min(total_occurrences / 5, 3)  # Cap bonus at 3
        
        # Confidence based on match count
        max_possible = len(keywords) + 3
        confidence = min(score / max_possible, 1.0)
        
        return score, matches, confidence
    
    def get_category_scores(self, classification: TaxonomyClassification) -> Dict[str, float]:
        """
        Get aggregate score for each main category.
        
        Args:
            classification: Taxonomy classification
            
        Returns:
            Dictionary with Environmental, Social, Governance scores (0-10)
        """
        env_score = sum(v.score for v in classification.Environmental.values()) / len(classification.Environmental)
        social_score = sum(v.score for v in classification.Social.values()) / len(classification.Social)
        gov_score = sum(v.score for v in classification.Governance.values()) / len(classification.Governance)
        
        return {
            "Environmental": min(env_score, 10),
            "Social": min(social_score, 10),
            "Governance": min(gov_score, 10),
        }
    
    def get_top_subcategories(
        self,
        classification: TaxonomyClassification,
        top_n: int = 3
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top N subcategories by score for each main category.
        
        Args:
            classification: Taxonomy classification
            top_n: Number of top items to return
            
        Returns:
            Dictionary with top subcategories per category
        """
        result = {}
        
        # Environmental
        env_items = sorted(
            [(k, v.score) for k, v in classification.Environmental.items()],
            key=lambda x: x[1],
            reverse=True
        )
        result["Environmental"] = env_items[:top_n]
        
        # Social
        social_items = sorted(
            [(k, v.score) for k, v in classification.Social.items()],
            key=lambda x: x[1],
            reverse=True
        )
        result["Social"] = social_items[:top_n]
        
        # Governance
        gov_items = sorted(
            [(k, v.score) for k, v in classification.Governance.items()],
            key=lambda x: x[1],
            reverse=True
        )
        result["Governance"] = gov_items[:top_n]
        
        return result
    
    def get_matched_terms_by_category(
        self,
        classification: TaxonomyClassification
    ) -> Dict[str, List[str]]:
        """
        Get all matched terms grouped by main category.
        
        Args:
            classification: Taxonomy classification
            
        Returns:
            Dictionary with matched terms per category
        """
        result = {
            "Environmental": [],
            "Social": [],
            "Governance": []
        }
        
        for subcategory, score_obj in classification.Environmental.items():
            result["Environmental"].extend(score_obj.matched_terms)
        
        for subcategory, score_obj in classification.Social.items():
            result["Social"].extend(score_obj.matched_terms)
        
        for subcategory, score_obj in classification.Governance.items():
            result["Governance"].extend(score_obj.matched_terms)
        
        # Remove duplicates
        result["Environmental"] = list(set(result["Environmental"]))
        result["Social"] = list(set(result["Social"]))
        result["Governance"] = list(set(result["Governance"]))
        
        return result