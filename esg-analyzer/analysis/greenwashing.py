"""
Greenwashing Detection.
Analyzes sustainability claims vs measurable disclosures.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from config import Config, setup_logging
from models.schemas import GreenwashingAnalysis

logger = setup_logging(__name__)


class GreenwashingDetector:
    """Detect potential greenwashing in sustainability reports."""
    
    def __init__(self):
        """Initialize detector."""
        self.config = Config()
        self.green_claim_keywords = self.config.GREEN_CLAIM_KEYWORDS
        self._build_patterns()
    
    def _build_patterns(self):
        """Build regex patterns for claims and disclosures."""
        # Sustainability claims (aspirational language)
        self.claim_pattern = r'\b(' + '|'.join(re.escape(kw) for kw in self.green_claim_keywords) + r')\b'
        
        # Measurable disclosures (specific metrics)
        self.disclosure_keywords = [
            "percent", "%", "reduction", "metric ton", "tonnes", "mw", "megawatt",
            "audit", "verified", "third-party", "independent", "certification",
            "data", "measurement", "target", "goal", "2030", "2035", "2040", "2050",
            "scope 1", "scope 2", "scope 3", "emissions"
        ]
        self.disclosure_pattern = r'\b(' + '|'.join(re.escape(kw) for kw in self.disclosure_keywords) + r')\b'
    
    def analyze(self, text: str) -> GreenwashingAnalysis:
        """
        Analyze text for greenwashing indicators.
        
        Args:
            text: Document text to analyze
            
        Returns:
            GreenwashingAnalysis with risk assessment
        """
        logger.info(f"Analyzing text for greenwashing ({len(text)} chars)")
        
        lower_text = text.lower()
        
        # Count claims and disclosures
        claims = len(re.findall(self.claim_pattern, lower_text, re.IGNORECASE))
        disclosures = len(re.findall(self.disclosure_pattern, lower_text, re.IGNORECASE))
        
        # Extract indicators
        risk_indicators = self._identify_risk_indicators(text, lower_text, claims, disclosures)
        
        # Calculate risk level
        ratio = claims / max(disclosures, 1)
        risk_level = self._assess_risk(claims, disclosures, ratio)
        
        analysis = GreenwashingAnalysis(
            green_claims_count=claims,
            measurable_disclosures_count=disclosures,
            greenwashing_risk=risk_level,
            claim_disclosure_ratio=ratio,
            risk_indicators=risk_indicators
        )
        
        logger.info(
            f"Greenwashing analysis: {claims} claims, {disclosures} disclosures, "
            f"ratio={ratio:.2f}, risk={risk_level}"
        )
        
        return analysis
    
    def _identify_risk_indicators(
        self,
        text: str,
        lower_text: str,
        claims: int,
        disclosures: int
    ) -> List[str]:
        """
        Identify specific greenwashing risk indicators.
        
        Args:
            text: Original text
            lower_text: Lowercase text
            claims: Number of claims found
            disclosures: Number of disclosures found
            
        Returns:
            List of risk indicators
        """
        indicators = []
        
        # High claims-to-disclosure ratio
        if claims > 0 and disclosures == 0:
            indicators.append("No measurable disclosures despite green claims")
        elif claims > 0 and claims / max(disclosures, 1) > 2:
            indicators.append("High ratio of claims to measurable disclosures")
        
        # Vague language patterns
        vague_patterns = [
            r'commitment to|dedicated to|working towards',  # Non-specific commitments
            r'we aim|we plan|we will try',  # Weak language
            r'considering|exploring|investigating'  # Exploratory language
        ]
        
        vague_count = sum(
            len(re.findall(pattern, lower_text, re.IGNORECASE))
            for pattern in vague_patterns
        )
        
        if vague_count > 5:
            indicators.append("Excessive use of vague commitment language")
        
        # Missing target dates
        if "net zero" in lower_text or "carbon neutral" in lower_text:
            if not re.search(r'\b(202[0-9]|203[0-9]|204[0-9]|205[0-9])\b', text):
                indicators.append("Net zero/carbon neutral claim without specific target year")
        
        # Lack of verification language
        verification_keywords = ["third-party", "verified", "audited", "certified", "independent"]
        verification_mentions = sum(
            lower_text.count(kw) for kw in verification_keywords
        )
        if verification_mentions == 0 and claims > 5:
            indicators.append("No verification or audit mentions despite multiple claims")
        
        # Scope hiding (mentioning only Scope 1/2, hiding Scope 3)
        if "scope 1" in lower_text or "scope 2" in lower_text:
            if "scope 3" not in lower_text:
                indicators.append("Mentions Scope 1/2 emissions but not Scope 3")
        
        return indicators
    
    def _assess_risk(self, claims: int, disclosures: int, ratio: float) -> str:
        """
        Assess greenwashing risk level.
        
        Args:
            claims: Number of claims
            disclosures: Number of disclosures
            ratio: Claim-to-disclosure ratio
            
        Returns:
            Risk level: LOW, MEDIUM, or HIGH
        """
        risk_score = 0
        
        # No claims = low risk
        if claims == 0:
            return "LOW"
        
        # No disclosures with claims = high risk
        if disclosures == 0:
            return "HIGH"
        
        # High ratio = risk (more claims than disclosures)
        if ratio > 3:
            risk_score += 3
        elif ratio > 2:
            risk_score += 2
        elif ratio > 1:
            risk_score += 1
        
        # High absolute number of claims without corresponding disclosures
        if claims > 20 and disclosures < 10:
            risk_score += 2
        
        # Sufficient balance = low risk
        if risk_score <= 1:
            return "LOW"
        elif risk_score <= 3:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def get_claim_context(self, text: str, context_window: int = 150) -> List[Dict[str, str]]:
        """
        Extract context around green claims.
        
        Args:
            text: Document text
            context_window: Characters before/after to include
            
        Returns:
            List of claim contexts
        """
        contexts = []
        
        for match in re.finditer(self.claim_pattern, text, re.IGNORECASE):
            start = max(0, match.start() - context_window)
            end = min(len(text), match.end() + context_window)
            
            snippet = text[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."
            
            contexts.append({
                "claim": match.group(1),
                "context": snippet
            })
        
        return contexts
    
    def compare_reports(
        self,
        text1: str,
        text2: str,
        label1: str = "Report 1",
        label2: str = "Report 2"
    ) -> Dict[str, any]:
        """
        Compare greenwashing risk between two reports.
        
        Args:
            text1: First report text
            text2: Second report text
            label1: Label for first report
            label2: Label for second report
            
        Returns:
            Comparison dictionary
        """
        analysis1 = self.analyze(text1)
        analysis2 = self.analyze(text2)
        
        return {
            label1: {
                "claims": analysis1.green_claims_count,
                "disclosures": analysis1.measurable_disclosures_count,
                "risk": analysis1.greenwashing_risk,
                "ratio": analysis1.claim_disclosure_ratio
            },
            label2: {
                "claims": analysis2.green_claims_count,
                "disclosures": analysis2.measurable_disclosures_count,
                "risk": analysis2.greenwashing_risk,
                "ratio": analysis2.claim_disclosure_ratio
            },
            "improvement": analysis1.greenwashing_risk != analysis2.greenwashing_risk
        }