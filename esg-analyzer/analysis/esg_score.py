"""
ESG Score Engine.
Calculates comprehensive ESG scores based on metrics and greenwashing analysis.
"""

import logging
from typing import Dict, Optional
from datetime import datetime

from config import Config, setup_logging
from models.schemas import (
    ESGScore, MetricExtraction, GreenwashingAnalysis, TaxonomyClassification
)

logger = setup_logging(__name__)


class ESGScoreEngine:
    """Calculate ESG scores from extracted metrics and analysis."""
    
    def __init__(self):
        """Initialize score engine."""
        self.config = Config()
        self.weights = self.config.ESG_WEIGHTS
    
    def calculate_score(
        self,
        metrics: MetricExtraction,
        taxonomy: TaxonomyClassification,
        greenwashing: GreenwashingAnalysis
    ) -> ESGScore:
        """
        Calculate comprehensive ESG score.
        
        Args:
            metrics: Extracted metrics
            taxonomy: Taxonomy classification
            greenwashing: Greenwashing analysis
            
        Returns:
            ESGScore with overall and component scores
        """
        logger.info("Calculating ESG score")
        
        # Calculate component scores
        environmental_score = self._calculate_environmental_score(metrics, taxonomy)
        social_score = self._calculate_social_score(metrics, taxonomy)
        governance_score = self._calculate_governance_score(taxonomy)
        
        # Calculate greenwashing adjustment
        greenwashing_adjustment = self._calculate_greenwashing_adjustment(greenwashing)
        
        # Calculate weighted overall score
        overall_score = (
            environmental_score * self.weights["carbon_metrics"] +
            social_score * self.weights["governance_metrics"] +
            governance_score * self.weights["governance_metrics"] +
            greenwashing_adjustment * self.weights["greenwashing_adjustment"]
        )
        
        # Clamp to 0-100
        overall_score = max(0, min(100, overall_score))
        
        # Component breakdown
        score_components = {
            "carbon_metrics_contribution": environmental_score * self.weights["carbon_metrics"],
            "renewable_metrics_contribution": environmental_score * self.weights["renewable_metrics"],
            "water_metrics_contribution": environmental_score * self.weights["water_metrics"],
            "governance_metrics_contribution": governance_score * self.weights["governance_metrics"],
            "greenwashing_adjustment_contribution": greenwashing_adjustment * self.weights["greenwashing_adjustment"],
        }
        
        score = ESGScore(
            overall_score=overall_score,
            environmental_score=environmental_score,
            social_score=social_score,
            governance_score=governance_score,
            greenwashing_adjustment=greenwashing_adjustment,
            score_components=score_components,
            score_date=datetime.now()
        )
        
        logger.info(
            f"ESG Score calculated: Overall={overall_score:.1f}, "
            f"E={environmental_score:.1f}, S={social_score:.1f}, "
            f"G={governance_score:.1f}"
        )
        
        return score
    
    def _calculate_environmental_score(
        self,
        metrics: MetricExtraction,
        taxonomy: TaxonomyClassification
    ) -> float:
        """Calculate environmental score."""
        score = 0.0
        weight_count = 0
        
        # Carbon reduction
        if metrics.carbon_reduction_percent is not None:
            carbon_score = (metrics.carbon_reduction_percent / 100) * 25
            score += carbon_score
            weight_count += 1
        
        # Renewable energy
        if metrics.renewable_energy_percent is not None:
            renewable_score = (metrics.renewable_energy_percent / 100) * 25
            score += renewable_score
            weight_count += 1
        
        # Water reduction
        if metrics.water_reduction_percent is not None:
            water_score = (metrics.water_reduction_percent / 100) * 20
            score += water_score
            weight_count += 1
        
        # Emissions reduction
        if metrics.emissions_reduction_percent is not None:
            emissions_score = (metrics.emissions_reduction_percent / 100) * 15
            score += emissions_score
            weight_count += 1
        
        # Waste reduction
        if metrics.waste_reduction_percent is not None:
            waste_score = (metrics.waste_reduction_percent / 100) * 15
            score += waste_score
            weight_count += 1
        
        # Net zero targets
        if metrics.net_zero_target_year is not None:
            # Score based on how soon target is (sooner = higher score)
            years_to_target = metrics.net_zero_target_year - datetime.now().year
            if years_to_target > 0:
                target_score = max(0, 20 - (years_to_target / 3))
                score += target_score
                weight_count += 1
        
        # Taxonomy contribution
        env_taxonomy_score = sum(v.score for v in taxonomy.Environmental.values()) / len(taxonomy.Environmental)
        score += env_taxonomy_score * 5
        weight_count += 0.5
        
        return score / weight_count if weight_count > 0 else 0.0
    
    def _calculate_social_score(
        self,
        metrics: MetricExtraction,
        taxonomy: TaxonomyClassification
    ) -> float:
        """Calculate social score."""
        score = 0.0
        weight_count = 0
        
        # Employee count (normalized, companies with more employees may have more programs)
        if metrics.employee_count is not None:
            # Normalize: companies with 50k+ employees get full points
            employee_score = min(100, (metrics.employee_count / 50000) * 100) / 4
            score += employee_score
            weight_count += 1
        
        # Supplier audits
        if metrics.supplier_audited_count is not None:
            supplier_score = min(100, (metrics.supplier_audited_count / 500) * 100) / 4
            score += supplier_score
            weight_count += 1
        
        # Taxonomy contribution
        social_taxonomy_score = sum(v.score for v in taxonomy.Social.values()) / len(taxonomy.Social)
        score += social_taxonomy_score * 25
        weight_count += 2.5
        
        return score / weight_count if weight_count > 0 else 0.0
    
    def _calculate_governance_score(
        self,
        taxonomy: TaxonomyClassification
    ) -> float:
        """Calculate governance score."""
        gov_taxonomy_score = sum(v.score for v in taxonomy.Governance.values()) / len(taxonomy.Governance)
        
        # Scale taxonomy score to 0-100
        return gov_taxonomy_score * 10
    
    def _calculate_greenwashing_adjustment(
        self,
        greenwashing: GreenwashingAnalysis
    ) -> float:
        """Calculate greenwashing adjustment to overall score."""
        if greenwashing.greenwashing_risk == "LOW":
            return 10  # Bonus for low risk
        elif greenwashing.greenwashing_risk == "MEDIUM":
            return 0  # No adjustment
        else:  # HIGH
            return -10  # Penalty for high risk
    
    def score_to_grade(self, score: float) -> str:
        """
        Convert score to letter grade.
        
        Args:
            score: ESG score (0-100)
            
        Returns:
            Letter grade (A+, A, B+, B, C, D, F)
        """
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 80:
            return "B+"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        else:
            return "F"
    
    def calculate_improvement(
        self,
        previous_score: ESGScore,
        current_score: ESGScore
    ) -> Dict[str, float]:
        """
        Calculate year-over-year improvement.
        
        Args:
            previous_score: Previous ESG score
            current_score: Current ESG score
            
        Returns:
            Dictionary with improvement metrics
        """
        overall_change = current_score.overall_score - previous_score.overall_score
        env_change = current_score.environmental_score - previous_score.environmental_score
        social_change = current_score.social_score - previous_score.social_score
        gov_change = current_score.governance_score - previous_score.governance_score
        
        return {
            "overall_change": overall_change,
            "environmental_change": env_change,
            "social_change": social_change,
            "governance_change": gov_change,
            "improving": overall_change > 0,
            "improvement_pct": (overall_change / max(abs(previous_score.overall_score), 1)) * 100
        }
    
    def get_score_breakdown(self, score: ESGScore) -> Dict[str, any]:
        """
        Get detailed score breakdown for visualization.
        
        Args:
            score: ESG score
            
        Returns:
            Dictionary with breakdown
        """
        return {
            "overall": {
                "score": score.overall_score,
                "grade": self.score_to_grade(score.overall_score)
            },
            "environmental": {
                "score": score.environmental_score,
                "grade": self.score_to_grade(score.environmental_score),
                "weight": self.weights["carbon_metrics"] + self.weights["renewable_metrics"] + self.weights["water_metrics"]
            },
            "social": {
                "score": score.social_score,
                "grade": self.score_to_grade(score.social_score),
                "weight": 0  # Added if needed
            },
            "governance": {
                "score": score.governance_score,
                "grade": self.score_to_grade(score.governance_score),
                "weight": self.weights["governance_metrics"]
            },
            "greenwashing_adjustment": score.greenwashing_adjustment,
            "components": score.score_components
        }
