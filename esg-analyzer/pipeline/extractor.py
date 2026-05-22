"""
Regex-based ESG metric extraction.
Extracts structured ESG metrics from text using pattern matching.
"""

import re
import logging
from typing import Dict, Optional, List, Any, Tuple
from config import Config, setup_logging
from models.schemas import MetricExtraction

logger = setup_logging(__name__)


class ESGMetricExtractor:
    """Extract ESG metrics using regex patterns."""
    
    def __init__(self):
        """Initialize extractor with regex patterns."""
        self.config = Config()
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile all regex patterns."""
        self.patterns = {
            "carbon_reduction": [
                r'(?:overall\s+)?emissions?\s+(?:reduction|cut|reduced|reduction)\s+(?:by|of)?\s*(\d+\.?\d*)\s*%',
                r'(\d+\.?\d*)\s*%\s+(?:overall\s+)?emissions?\s+reduction',
                r'reduced\s+(?:overall\s+)?emissions?\s+(?:by|to)?\s*(\d+\.?\d*)\s*%',
                r'cutting\s+emissions?\s+(?:by|to)?\s*(\d+\.?\d*)\s*%'
            ],
            "renewable_energy": [
                r'(\d+\.?\d*)\s*%\s+(?:of\s+)?(?:total\s+)?(?:energy|electricity).*?renewable',
                r'renewable\s+(?:energy|electricity).*?(\d+\.?\d*)\s*%',
                r'(\d+\.?\d*)\s*%\s+renewable\s+(?:energy|electricity)',
                r'renewable\s+(?:energy|electricity).*?(\d+\.?\d*)\s*percent'
            ],
            "water_reduction": [
                r'water\s+(?:consumption|use|withdrawal).*?(?:reduction|reduced|cut)\s*(?:by|of)?\s*(\d+\.?\d*)\s*%',
                r'(\d+\.?\d*)\s*%\s+(?:reduction|reduction)\s+(?:in|of)?\s+water',
                r'water.*?(\d+\.?\d*)\s*%\s+reduction',
                r'replenish\s+all\s+corporate\s+freshwater\s+withdrawals'
            ],
            "net_zero_year": [
                r'net\s+zero\s+(?:by|in|target|goal|commitment)\s*(?:of\s+)?(\d{4})',
                r'(\d{4})\s+net\s+zero',
                r'carbon\s+neutral\s+(?:by|in)?\s*(\d{4})',
                r'achieve\s+net\s+zero\s+(?:by|in)?\s*(\d{4})'
            ],
            "emissions_reduction": [
                r'emissions?\s+(?:reduction|target|reduction)\s+(?:by|of|to)?\s*(\d+\.?\d*)\s*%',
                r'(\d+\.?\d*)\s*%\s+emissions?\s+reduction',
                r'reduce\s+emissions?\s+by\s*(\d+\.?\d*)\s*%'
            ],
            "emissions_target_year": [
                r'emissions?\s+(?:target|reduction|cut)\s+(?:by|of)?\s+(\d{4})',
                r'by\s+(\d{4})\s+(?:reduce|cut)\s+emissions?',
                r'(\d{4})\s+emissions?\s+(?:target|reduction|goal)'
            ],
            "waste_reduction": [
                r'waste\s+(?:reduction|reduced|cut)\s+(?:by|of)?\s*(\d+\.?\d*)\s*%',
                r'(\d+\.?\d*)\s*%\s+(?:waste\s+)?(?:reduction|reduced)',
                r'landfill\s+diversion\s+(?:rate|of)?\s*(\d+\.?\d*)\s*%'
            ],
            "renewable_capacity": [
                r'(\d+\.?\d*)\s*(?:GW|MW)\s+(?:of\s+)?renewable',
                r'renewable\s+capacity\s+(?:of\s+)?(\d+\.?\d*)\s*(?:GW|MW)',
                r'(\d+)\s*megawatt'
            ],
            "employee_count": [
                r'(?:total\s+)?(?:employees|workforce|staff)\s+(?:of|is|are)\s*(?:approximately\s+)?(\d+,?\d*)',
                r'(\d+,?\d*)\s+(?:employees|team\s+members|staff)',
                r'employ[^.]*?(\d+,?\d*)\s+(?:people|employees)'
            ],
            "supplier_audited": [
                r'(?:audited|assessed|evaluated)\s+(?:suppliers?|vendors?).*?(\d+,?\d*)',
                r'(\d+,?\d*)\s+(?:suppliers?|vendors?).*?(?:audited|assessed)',
                r'supplier\s+(?:audits?|assessments?)\s+.*?(\d+,?\d*)'
            ]
        }
    
    def extract_metrics(self, text: str) -> MetricExtraction:
        """
        Extract ESG metrics from text.
        
        Args:
            text: Document text to extract from
            
        Returns:
            MetricExtraction with extracted metrics
        """
        logger.info(f"Extracting metrics from text ({len(text)} chars)")
        
        lower_text = text.lower()
        metrics = MetricExtraction()
        
        # Extract each metric
        metrics.carbon_reduction_percent = self._extract_percentage(lower_text, "carbon_reduction")
        metrics.renewable_energy_percent = self._extract_percentage(lower_text, "renewable_energy")
        metrics.water_reduction_percent = self._extract_percentage(lower_text, "water_reduction")
        metrics.net_zero_target_year = self._extract_year(lower_text, "net_zero_year")
        metrics.emissions_reduction_percent = self._extract_percentage(lower_text, "emissions_reduction")
        metrics.emissions_target_year = self._extract_year(lower_text, "emissions_target_year")
        metrics.waste_reduction_percent = self._extract_percentage(lower_text, "waste_reduction")
        metrics.renewable_capacity_mw = self._extract_capacity(lower_text, "renewable_capacity")
        metrics.employee_count = self._extract_count(lower_text, "employee_count")
        metrics.supplier_audited_count = self._extract_count(lower_text, "supplier_audited")
        
        logger.info(f"Extracted metrics: {metrics.dict(exclude_none=True)}")
        return metrics
    
    def _extract_percentage(self, text: str, metric_key: str) -> Optional[float]:
        """
        Extract percentage value from text.
        
        Args:
            text: Text to search
            metric_key: Key to patterns dictionary
            
        Returns:
            Percentage value or None
        """
        if metric_key not in self.patterns:
            return None
        
        for pattern in self.patterns[metric_key]:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    value = float(match.group(1))
                    if 0 <= value <= 100:
                        logger.debug(f"Extracted {metric_key}: {value}%")
                        return value
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _extract_year(self, text: str, metric_key: str) -> Optional[int]:
        """
        Extract year value from text.
        
        Args:
            text: Text to search
            metric_key: Key to patterns dictionary
            
        Returns:
            Year value or None
        """
        if metric_key not in self.patterns:
            return None
        
        for pattern in self.patterns[metric_key]:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    year = int(match.group(1))
                    if 2020 <= year <= 2100:
                        logger.debug(f"Extracted {metric_key}: {year}")
                        return year
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _extract_capacity(self, text: str, metric_key: str) -> Optional[float]:
        """
        Extract capacity value from text (converts to MW).
        
        Args:
            text: Text to search
            metric_key: Key to patterns dictionary
            
        Returns:
            Capacity in MW or None
        """
        if metric_key not in self.patterns:
            return None
        
        for pattern in self.patterns[metric_key]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    # Check if it's in GW (convert to MW)
                    if "GW" in match.group(0).upper():
                        value *= 1000
                    logger.debug(f"Extracted {metric_key}: {value} MW")
                    return value
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _extract_count(self, text: str, metric_key: str) -> Optional[int]:
        """
        Extract count value from text.
        
        Args:
            text: Text to search
            metric_key: Key to patterns dictionary
            
        Returns:
            Count value or None
        """
        if metric_key not in self.patterns:
            return None
        
        for pattern in self.patterns[metric_key]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    # Remove commas and convert to int
                    count_str = match.group(1).replace(',', '')
                    count = int(count_str)
                    if count > 0:
                        logger.debug(f"Extracted {metric_key}: {count}")
                        return count
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def extract_context(self, text: str, search_term: str, context_window: int = 200) -> List[str]:
        """
        Extract context snippets around search terms.
        
        Args:
            text: Full text
            search_term: Term to search for
            context_window: Characters before/after to include
            
        Returns:
            List of context snippets
        """
        snippets = []
        pattern = re.compile(re.escape(search_term), re.IGNORECASE)
        
        for match in pattern.finditer(text):
            start = max(0, match.start() - context_window)
            end = min(len(text), match.end() + context_window)
            snippet = text[start:end]
            
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."
            
            snippets.append(snippet)
        
        return snippets
    
    def get_extraction_quality(self, metrics: MetricExtraction) -> float:
        """
        Estimate extraction quality (0-1).
        
        Args:
            metrics: Extracted metrics
            
        Returns:
            Quality score
        """
        extracted_fields = sum(1 for v in metrics.dict(exclude_none=True).values() if v is not None)
        total_fields = len(metrics.dict())
        quality = extracted_fields / total_fields if total_fields > 0 else 0
        return quality