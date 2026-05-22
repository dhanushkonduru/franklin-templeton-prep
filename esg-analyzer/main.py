"""
ESG Analyzer Main Orchestration.
Main entry point for processing ESG reports and generating analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from config import Config, setup_logging
from models.schemas import ESGAnalysisResult, CompanyReport
from pipeline.pdf_parser import PDFParser
from pipeline.segmenter import ESGSegmenter
from pipeline.extractor import ESGMetricExtractor
from pipeline.llm_extractor import LLMExtractor
from taxonomy.classifier import TaxonomyClassifier
from analysis.greenwashing import GreenwashingDetector
from analysis.esg_score import ESGScoreEngine

logger = setup_logging(__name__)


class ESGAnalyzer:
    """Main ESG analysis orchestrator."""
    
    def __init__(self):
        """Initialize analyzer with all components."""
        self.config = Config()
        self.pdf_parser = PDFParser()
        self.segmenter = ESGSegmenter()
        self.regex_extractor = ESGMetricExtractor()
        self.llm_extractor = LLMExtractor()
        self.taxonomy_classifier = TaxonomyClassifier()
        self.greenwashing_detector = GreenwashingDetector()
        self.score_engine = ESGScoreEngine()
        
        logger.info("ESG Analyzer initialized")
    
    def analyze_report(
        self,
        pdf_path: str | Path,
        company_name: Optional[str] = None,
        report_year: Optional[int] = None
    ) -> ESGAnalysisResult:
        """
        Analyze a single ESG report.
        
        Args:
            pdf_path: Path to PDF report
            company_name: Company name
            report_year: Report year
            
        Returns:
            ESGAnalysisResult with complete analysis
        """
        pdf_path = Path(pdf_path)
        logger.info(f"Analyzing report: {pdf_path.name}")
        
        try:
            # Extract PDF text
            text, metadata = self.pdf_parser.parse_pdf(pdf_path)
            
            # Auto-detect company name from filename if not provided
            if not company_name:
                company_name = pdf_path.stem.replace("_", " ").title()
            
            # Auto-detect year from filename if not provided
            if not report_year:
                import re
                year_match = re.search(r'(\d{4})', pdf_path.name)
                report_year = int(year_match.group(1)) if year_match else datetime.now().year
            
            # Create result object
            result = ESGAnalysisResult(
                company_name=company_name,
                report_year=report_year,
                report_file=str(pdf_path)
            )
            
            # Segment text by ESG category
            result.segmentation = self.segmenter.segment_text(text)
            
            # Extract metrics using regex
            result.metrics = self.regex_extractor.extract_metrics(text)
            
            # Try LLM extraction for improved results
            if self.config.USE_LLM_EXTRACTION:
                try:
                    llm_metrics = self.llm_extractor.extract_metrics(text, company_name)
                    if llm_metrics:
                        # Merge LLM results with regex results (prefer LLM if available)
                        result.metrics = self._merge_metrics(result.metrics, llm_metrics)
                except Exception as e:
                    logger.warning(f"LLM extraction failed: {str(e)}, using regex results only")
            
            # Classify by taxonomy
            result.taxonomy = self.taxonomy_classifier.classify(text)
            
            # Detect greenwashing
            result.greenwashing = self.greenwashing_detector.analyze(text)
            
            # Calculate ESG score
            result.score = self.score_engine.calculate_score(
                result.metrics,
                result.taxonomy,
                result.greenwashing
            )
            
            # Store raw text if configured
            if self.config.SAVE_INTERMEDIATE_RESULTS:
                result.raw_text = text
            
            # Calculate extraction confidence
            result.extraction_confidence = self.regex_extractor.get_extraction_quality(result.metrics)
            
            logger.info(f"Analysis complete for {company_name}: Score={result.score.overall_score:.1f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing report {pdf_path.name}: {str(e)}")
            raise
    
    def _merge_metrics(self, regex_metrics, llm_metrics):
        """Merge regex and LLM extraction results."""
        merged = regex_metrics.model_copy()
        
        # Prefer LLM results where available
        for field, value in llm_metrics.model_dump(exclude_none=True).items():
            if value is not None:
                setattr(merged, field, value)
        
        return merged
    
    def process_directory(
        self,
        reports_dir: Optional[str | Path] = None
    ) -> Dict[str, ESGAnalysisResult]:
        """
        Process all PDFs in a directory.
        
        Args:
            reports_dir: Directory containing PDF reports
            
        Returns:
            Dictionary mapping company names to results
        """
        if reports_dir is None:
            reports_dir = self.config.REPORTS_DIR
        else:
            reports_dir = Path(reports_dir)
        logger.info(f"Processing directory: {reports_dir}")
        
        if not reports_dir.exists():
            logger.error(f"Reports directory not found: {reports_dir}")
            return {}
        
        results = {}
        pdf_files = list(reports_dir.glob("**/*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            try:
                result = self.analyze_report(pdf_file)
                results[result.company_name] = result
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {str(e)}")
                continue
        
        logger.info(f"Processed {len(results)} reports successfully")
        return results
    
    def save_results(
        self,
        results: Dict[str, ESGAnalysisResult],
        output_dir: Optional[str | Path] = None
    ) -> None:
        """
        Save analysis results to JSON files.
        
        Args:
            results: Dictionary of analysis results
            output_dir: Output directory for JSON files
        """
        if output_dir is None:
            output_dir = self.config.OUTPUTS_DIR
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving results to {output_dir}")
        
        for company_name, result in results.items():
            filename = output_dir / f"{company_name.lower().replace(' ', '_')}.json"
            
            try:
                with open(filename, "w") as f:
                    json.dump(result.model_dump(), f, indent=2, default=str)
                logger.info(f"Saved: {filename}")
            except Exception as e:
                logger.error(f"Error saving {filename}: {str(e)}")
    
    def generate_summary_report(
        self,
        results: Dict[str, ESGAnalysisResult]
    ) -> Dict[str, any]:
        """
        Generate a summary report across all companies.
        
        Args:
            results: Dictionary of analysis results
            
        Returns:
            Summary report dictionary
        """
        if not results:
            return {}
        
        companies = list(results.keys())
        scores = [r.score.overall_score for r in results.values()]
        env_scores = [r.score.environmental_score for r in results.values()]
        social_scores = [r.score.social_score for r in results.values()]
        gov_scores = [r.score.governance_score for r in results.values()]
        
        summary = {
            "report_generated": datetime.now().isoformat(),
            "total_companies": len(results),
            "companies": companies,
            "overall_scores": {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "median": sorted(scores)[len(scores) // 2]
            },
            "environmental_scores": {
                "mean": sum(env_scores) / len(env_scores),
                "min": min(env_scores),
                "max": max(env_scores)
            },
            "social_scores": {
                "mean": sum(social_scores) / len(social_scores),
                "min": min(social_scores),
                "max": max(social_scores)
            },
            "governance_scores": {
                "mean": sum(gov_scores) / len(gov_scores),
                "min": min(gov_scores),
                "max": max(gov_scores)
            },
            "company_rankings": sorted(
                [(name, result.score.overall_score) for name, result in results.items()],
                key=lambda x: x[1],
                reverse=True
            ),
            "greenwashing_risks": {
                name: result.greenwashing.greenwashing_risk
                for name, result in results.items()
            }
        }
        
        return summary


def main():
    """Main entry point."""
    logger.info("=" * 80)
    logger.info("ESG ANALYZER - Starting analysis")
    logger.info("=" * 80)
    
    # Validate configuration
    Config.validate()
    
    # Initialize analyzer
    analyzer = ESGAnalyzer()
    
    # Process reports
    logger.info("Processing reports...")
    results = analyzer.process_directory()
    
    if results:
        # Save results
        analyzer.save_results(results)
        
        # Generate summary
        summary = analyzer.generate_summary_report(results)
        
        # Save summary
        summary_file = analyzer.config.OUTPUTS_DIR / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Summary saved to {summary_file}")
        
        # Print summary
        logger.info("=" * 80)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Companies analyzed: {summary['total_companies']}")
        logger.info(f"Overall ESG Score (mean): {summary['overall_scores']['mean']:.1f}")
        logger.info("\nTop Companies:")
        for rank, (company, score) in enumerate(summary['company_rankings'][:3], 1):
            logger.info(f"  {rank}. {company}: {score:.1f}")
    else:
        logger.warning("No reports processed")
    
    logger.info("=" * 80)
    logger.info("Analysis complete")


if __name__ == "__main__":
    main()