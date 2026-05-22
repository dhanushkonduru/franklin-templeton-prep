"""
Pydantic models and schemas for ESG Analyzer.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class ESGCategory(str, Enum):
    """ESG categories."""
    ENVIRONMENTAL = "Environmental"
    SOCIAL = "Social"
    GOVERNANCE = "Governance"


class TaxonomyScore(BaseModel):
    """Score for a single taxonomy classification."""
    score: float = Field(..., ge=0, le=10, description="Score from 0-10")
    matched_terms: List[str] = Field(default_factory=list, description="Matched keywords")
    confidence: float = Field(default=1.0, ge=0, le=1, description="Confidence score")


class TaxonomyClassification(BaseModel):
    """Taxonomy classification results."""
    Environmental: Dict[str, TaxonomyScore] = Field(
        default_factory=lambda: {
            "carbon": TaxonomyScore(score=0, matched_terms=[]),
            "emissions": TaxonomyScore(score=0, matched_terms=[]),
            "water": TaxonomyScore(score=0, matched_terms=[]),
            "renewable": TaxonomyScore(score=0, matched_terms=[]),
            "waste": TaxonomyScore(score=0, matched_terms=[]),
        }
    )
    Social: Dict[str, TaxonomyScore] = Field(
        default_factory=lambda: {
            "supplier": TaxonomyScore(score=0, matched_terms=[]),
            "training": TaxonomyScore(score=0, matched_terms=[]),
            "employee": TaxonomyScore(score=0, matched_terms=[]),
            "community": TaxonomyScore(score=0, matched_terms=[]),
        }
    )
    Governance: Dict[str, TaxonomyScore] = Field(
        default_factory=lambda: {
            "audit": TaxonomyScore(score=0, matched_terms=[]),
            "policy": TaxonomyScore(score=0, matched_terms=[]),
            "board": TaxonomyScore(score=0, matched_terms=[]),
            "compliance": TaxonomyScore(score=0, matched_terms=[]),
        }
    )


class MetricExtraction(BaseModel):
    """Extracted ESG metrics."""
    carbon_reduction_percent: Optional[float] = Field(None, ge=0, le=100)
    renewable_energy_percent: Optional[float] = Field(None, ge=0, le=100)
    water_reduction_percent: Optional[float] = Field(None, ge=0, le=100)
    net_zero_target_year: Optional[int] = Field(None, ge=2020)
    emissions_reduction_percent: Optional[float] = Field(None, ge=0, le=100)
    emissions_target_year: Optional[int] = Field(None, ge=2020)
    waste_reduction_percent: Optional[float] = Field(None, ge=0, le=100)
    renewable_capacity_mw: Optional[float] = Field(None, ge=0)
    employee_count: Optional[int] = Field(None, ge=0)
    supplier_audited_count: Optional[int] = Field(None, ge=0)
    custom_metrics: Dict[str, Any] = Field(default_factory=dict)


class GreenwashingAnalysis(BaseModel):
    """Greenwashing detection results."""
    green_claims_count: int = Field(ge=0)
    measurable_disclosures_count: int = Field(ge=0)
    greenwashing_risk: str = Field(..., description="LOW, MEDIUM, HIGH")
    claim_disclosure_ratio: float = Field(ge=0)
    risk_indicators: List[str] = Field(default_factory=list)
    
    @validator("greenwashing_risk")
    def validate_risk(cls, v):
        if v not in ["LOW", "MEDIUM", "HIGH"]:
            raise ValueError("Risk must be LOW, MEDIUM, or HIGH")
        return v


class ESGScore(BaseModel):
    """Overall ESG score."""
    overall_score: float = Field(..., ge=0, le=100)
    environmental_score: float = Field(..., ge=0, le=100)
    social_score: float = Field(..., ge=0, le=100)
    governance_score: float = Field(..., ge=0, le=100)
    greenwashing_adjustment: float = Field(default=0, ge=-10, le=10)
    score_components: Dict[str, float] = Field(default_factory=dict)
    score_date: datetime = Field(default_factory=datetime.now)


class DocumentSegmentation(BaseModel):
    """Segmented document by ESG category."""
    environmental_text: str = ""
    social_text: str = ""
    governance_text: str = ""
    other_text: str = ""
    document_length: int = 0


class ESGAnalysisResult(BaseModel):
    """Complete ESG analysis result for a company."""
    company_name: str
    report_year: int
    report_file: str
    extraction_date: datetime = Field(default_factory=datetime.now)
    metrics: MetricExtraction = Field(default_factory=MetricExtraction)
    taxonomy: TaxonomyClassification = Field(default_factory=TaxonomyClassification)
    greenwashing: GreenwashingAnalysis = Field(default_factory=lambda: GreenwashingAnalysis(
        green_claims_count=0,
        measurable_disclosures_count=0,
        greenwashing_risk="MEDIUM",
        claim_disclosure_ratio=0.0,
        risk_indicators=[]
    ))
    score: ESGScore = Field(default_factory=lambda: ESGScore(
        overall_score=0,
        environmental_score=0,
        social_score=0,
        governance_score=0
    ))
    raw_text: Optional[str] = Field(None, description="Original extracted text")
    segmentation: DocumentSegmentation = Field(default_factory=DocumentSegmentation)
    extraction_confidence: float = Field(default=0.0, ge=0, le=1)
    processing_notes: List[str] = Field(default_factory=list)


class CompanyReport(BaseModel):
    """Metadata for a company report."""
    company_name: str
    report_year: int
    file_path: str
    file_size_mb: float
    pages: int = 0
    extraction_status: str = "pending"  # pending, processing, completed, failed
    error_message: Optional[str] = None


class TimeSeries(BaseModel):
    """Time series ESG scores for a company."""
    company_name: str
    years: List[int]
    scores: List[float]
    metrics_history: Dict[int, MetricExtraction] = Field(default_factory=dict)


class ComparisonMetrics(BaseModel):
    """Multi-company comparison metrics."""
    companies: List[str]
    overall_scores: Dict[str, float]
    environmental_scores: Dict[str, float]
    social_scores: Dict[str, float]
    governance_scores: Dict[str, float]
    greenwashing_risks: Dict[str, str]
    key_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    comparison_date: datetime = Field(default_factory=datetime.now)
