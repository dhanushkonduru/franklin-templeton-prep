"""
Test suite for ESG Analyzer.
Unit and integration tests for core modules.
"""

import pytest
from pathlib import Path
from datetime import datetime

from pipeline.pdf_parser import PDFParser
from pipeline.segmenter import ESGSegmenter
from pipeline.extractor import ESGMetricExtractor
from taxonomy.classifier import TaxonomyClassifier
from analysis.greenwashing import GreenwashingDetector
from analysis.esg_score import ESGScoreEngine
from models.schemas import MetricExtraction, GreenwashingAnalysis, TaxonomyClassification


class TestESGSegmenter:
    """Test ESG segmentation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.segmenter = ESGSegmenter()
        self.sample_text = """
        We reduced carbon emissions by 50%.
        Our workforce includes employees from diverse backgrounds.
        We have strong governance policies.
        """
    
    def test_segment_text(self):
        """Test text segmentation."""
        result = self.segmenter.segment_text(self.sample_text)
        
        assert result.environmental_text
        assert result.social_text
        assert result.governance_text
    
    def test_category_stats(self):
        """Test category statistics."""
        result = self.segmenter.segment_text(self.sample_text)
        stats = self.segmenter.get_category_stats(result)
        
        assert "environmental" in stats
        assert "social" in stats
        assert "governance" in stats


class TestESGMetricExtractor:
    """Test metric extraction."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.extractor = ESGMetricExtractor()
    
    def test_extract_carbon_reduction(self):
        """Test carbon reduction extraction."""
        text = "We have reduced overall emissions by 60 percent since 2020."
        metrics = self.extractor.extract_metrics(text)
        
        assert metrics.carbon_reduction_percent == 60.0
    
    def test_extract_renewable_energy(self):
        """Test renewable energy extraction."""
        text = "Our facility is powered by 100% renewable energy."
        metrics = self.extractor.extract_metrics(text)
        
        assert metrics.renewable_energy_percent == 100.0
    
    def test_extract_year(self):
        """Test year extraction."""
        text = "We commit to achieving net zero emissions by 2030."
        metrics = self.extractor.extract_metrics(text)
        
        assert metrics.net_zero_target_year == 2030
    
    def test_extraction_quality(self):
        """Test extraction quality score."""
        text = """
        Carbon reduction: 75%
        Renewable energy: 85%
        Water reduction: 50%
        Net zero target: 2040
        """
        metrics = self.extractor.extract_metrics(text)
        quality = self.extractor.get_extraction_quality(metrics)
        
        assert 0 <= quality <= 1
        assert quality > 0.5  # High quality for this text


class TestTaxonomyClassifier:
    """Test taxonomy classification."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.classifier = TaxonomyClassifier()
        self.sample_text = """
        Carbon emissions reduction through renewable energy investment.
        Employee training programs and community engagement.
        Board audit and compliance measures.
        """
    
    def test_classify_text(self):
        """Test text classification."""
        result = self.classifier.classify(self.sample_text)
        
        assert isinstance(result, TaxonomyClassification)
        assert len(result.Environmental) > 0
        assert len(result.Social) > 0
        assert len(result.Governance) > 0
    
    def test_category_scores(self):
        """Test category score calculation."""
        result = self.classifier.classify(self.sample_text)
        scores = self.classifier.get_category_scores(result)
        
        assert "Environmental" in scores
        assert "Social" in scores
        assert "Governance" in scores
        assert all(0 <= v <= 10 for v in scores.values())
    
    def test_top_subcategories(self):
        """Test getting top subcategories."""
        result = self.classifier.classify(self.sample_text)
        top = self.classifier.get_top_subcategories(result, top_n=2)
        
        assert len(top["Environmental"]) <= 2
        assert len(top["Social"]) <= 2
        assert len(top["Governance"]) <= 2


class TestGreenwashingDetector:
    """Test greenwashing detection."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.detector = GreenwashingDetector()
    
    def test_low_risk(self):
        """Test low greenwashing risk."""
        text = """
        We achieved 75% emissions reduction by 2023.
        Verified by third-party audit.
        Scope 1, 2, and 3 emissions reported: 500,000 tonnes reduced.
        """
        analysis = self.detector.analyze(text)
        
        assert analysis.greenwashing_risk == "LOW"
        assert analysis.measurable_disclosures_count > 0
    
    def test_high_risk(self):
        """Test high greenwashing risk."""
        text = """
        We are committed to sustainability.
        We are a leader in environmental protection.
        We pursue green initiatives.
        We are eco-friendly.
        """
        analysis = self.detector.analyze(text)
        
        # Should have low disclosure count
        assert analysis.measurable_disclosures_count < analysis.green_claims_count
    
    def test_claim_context(self):
        """Test extracting claim context."""
        text = """
        We are committed to carbon reduction.
        We are dedicated to sustainability.
        """
        contexts = self.detector.get_claim_context(text)
        
        assert len(contexts) > 0
        assert all("claim" in c and "context" in c for c in contexts)


class TestESGScoreEngine:
    """Test ESG scoring."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.engine = ESGScoreEngine()
        
        # Create sample metrics
        self.metrics = MetricExtraction(
            carbon_reduction_percent=75.0,
            renewable_energy_percent=100.0,
            water_reduction_percent=50.0,
            emissions_reduction_percent=60.0,
            employee_count=50000,
            supplier_audited_count=200
        )
        
        # Create sample taxonomy
        self.taxonomy = TaxonomyClassification()
        
        # Create sample greenwashing analysis
        self.greenwashing = GreenwashingAnalysis(
            green_claims_count=30,
            measurable_disclosures_count=25,
            greenwashing_risk="LOW"
        )
    
    def test_calculate_score(self):
        """Test score calculation."""
        score = self.engine.calculate_score(self.metrics, self.taxonomy, self.greenwashing)
        
        assert 0 <= score.overall_score <= 100
        assert 0 <= score.environmental_score <= 100
        assert 0 <= score.social_score <= 100
        assert 0 <= score.governance_score <= 100
    
    def test_score_to_grade(self):
        """Test score to grade conversion."""
        assert self.engine.score_to_grade(95) == "A+"
        assert self.engine.score_to_grade(90) == "A"
        assert self.engine.score_to_grade(80) == "B+"
        assert self.engine.score_to_grade(70) == "B"
        assert self.engine.score_to_grade(50) == "D"
    
    def test_greenwashing_adjustment(self):
        """Test greenwashing adjustment."""
        # Low risk should add points
        low_risk = GreenwashingAnalysis(
            green_claims_count=10,
            measurable_disclosures_count=10,
            greenwashing_risk="LOW"
        )
        adjustment = self.engine._calculate_greenwashing_adjustment(low_risk)
        assert adjustment == 10
        
        # High risk should subtract points
        high_risk = GreenwashingAnalysis(
            green_claims_count=50,
            measurable_disclosures_count=5,
            greenwashing_risk="HIGH"
        )
        adjustment = self.engine._calculate_greenwashing_adjustment(high_risk)
        assert adjustment == -10


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_analysis(self):
        """Test complete analysis pipeline (without PDF)."""
        sample_text = """
        Company ABC ESG Report 2023
        
        Environmental:
        We have reduced carbon emissions by 65% since 2020.
        Our renewable energy capacity increased to 500 MW.
        We achieved 100% renewable electricity at our headquarters.
        
        Social:
        Our workforce of 75,000 employees includes diverse talent.
        We audited 300 suppliers for sustainability standards.
        We invested $50M in employee training and development.
        
        Governance:
        Our board of directors underwent independent audit.
        We have a comprehensive ethics policy.
        We comply with all regulatory requirements.
        
        Targets:
        Net zero emissions by 2035.
        """
        
        # Segment
        segmenter = ESGSegmenter()
        segmentation = segmenter.segment_text(sample_text)
        assert segmentation.environmental_text
        
        # Extract metrics
        extractor = ESGMetricExtractor()
        metrics = extractor.extract_metrics(sample_text)
        assert metrics.carbon_reduction_percent == 65.0
        assert metrics.renewable_energy_percent == 100.0
        
        # Classify
        classifier = TaxonomyClassifier()
        taxonomy = classifier.classify(sample_text)
        assert taxonomy.Environmental
        
        # Detect greenwashing
        detector = GreenwashingDetector()
        greenwashing = detector.analyze(sample_text)
        assert greenwashing.measurable_disclosures_count > 0
        
        # Score
        engine = ESGScoreEngine()
        score = engine.calculate_score(metrics, taxonomy, greenwashing)
        assert 0 <= score.overall_score <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
