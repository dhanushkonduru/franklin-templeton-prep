"""
LLM-based ESG metric extraction using Groq.
Uses structured output for reliable JSON extraction.
"""

import json
import logging
from typing import Optional, Dict, Any
import time

from config import Config, setup_logging
from models.schemas import MetricExtraction

logger = setup_logging(__name__)


class LLMExtractor:
    """Extract ESG metrics using Groq API with structured output."""
    
    def __init__(self):
        """Initialize LLM extractor."""
        self.config = Config()
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize Groq client."""
        try:
            from groq import Groq
            if self.config.GROQ_API_KEY:
                self.client = Groq(api_key=self.config.GROQ_API_KEY)
                logger.info("Groq client initialized")
            else:
                logger.warning("Groq API key not configured")
        except ImportError:
            logger.warning("Groq library not installed. LLM extraction disabled.")
    
    def extract_metrics(self, text: str, company_name: str = "") -> Optional[MetricExtraction]:
        """
        Extract ESG metrics using Groq.
        
        Args:
            text: Document text to extract from
            company_name: Company name for context
            
        Returns:
            MetricExtraction or None if API fails
        """
        if not self.client:
            logger.warning("LLM extractor not available, returning None")
            return None
        
        # Limit text length to avoid token limits
        text = text[:5000]
        
        prompt = self._build_prompt(text, company_name)
        
        try:
            logger.info("Calling Groq API for metric extraction")
            response = self.client.chat.completions.create(
                model=self.config.GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an ESG data extraction expert. Extract structured ESG metrics from sustainability reports."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.config.GROQ_TEMPERATURE,
                max_tokens=self.config.GROQ_MAX_TOKENS,
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            metrics_dict = self._parse_response(response_text)
            
            if metrics_dict:
                metrics = MetricExtraction(**metrics_dict)
                logger.info("Successfully extracted metrics using LLM")
                return metrics
            else:
                logger.warning("Failed to parse LLM response")
                return None
                
        except Exception as e:
            logger.error(f"Error calling Groq API: {str(e)}")
            return None
    
    def _build_prompt(self, text: str, company_name: str) -> str:
        """
        Build extraction prompt.
        
        Args:
            text: Document text
            company_name: Company name
            
        Returns:
            Prompt string
        """
        prompt = f"""Extract ESG metrics from the following sustainability report excerpt.

Company: {company_name if company_name else 'Unknown'}

Report excerpt:
{text[:3000]}

Please extract the following metrics and return as JSON (use null for missing values):
{{
  "carbon_reduction_percent": <number 0-100 or null>,
  "renewable_energy_percent": <number 0-100 or null>,
  "water_reduction_percent": <number 0-100 or null>,
  "net_zero_target_year": <year (2020-2100) or null>,
  "emissions_reduction_percent": <number 0-100 or null>,
  "emissions_target_year": <year (2020-2100) or null>,
  "waste_reduction_percent": <number 0-100 or null>,
  "renewable_capacity_mw": <number or null>,
  "employee_count": <integer or null>,
  "supplier_audited_count": <integer or null>
}}

Guidelines:
- Extract only explicit metrics mentioned in the text
- For percentages, use numbers 0-100
- For years, use format YYYY
- For counts, remove commas and convert to integers
- Use null if metric not found
- Return ONLY valid JSON, no other text"""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse LLM response.
        
        Args:
            response_text: Response from LLM
            
        Returns:
            Dictionary of metrics or None
        """
        try:
            # Try to extract JSON from response
            # Handle case where JSON might be wrapped in markdown code blocks
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()
            
            metrics_dict = json.loads(json_str)
            
            # Validate and clean values
            cleaned = {}
            for key, value in metrics_dict.items():
                if value is None:
                    cleaned[key] = None
                elif key.endswith("_percent"):
                    try:
                        val = float(value)
                        cleaned[key] = val if 0 <= val <= 100 else None
                    except (ValueError, TypeError):
                        cleaned[key] = None
                elif key.endswith("_year"):
                    try:
                        val = int(value)
                        cleaned[key] = val if 2020 <= val <= 2100 else None
                    except (ValueError, TypeError):
                        cleaned[key] = None
                elif key.endswith("_mw"):
                    try:
                        cleaned[key] = float(value)
                    except (ValueError, TypeError):
                        cleaned[key] = None
                elif key.endswith("_count"):
                    try:
                        cleaned[key] = int(value)
                    except (ValueError, TypeError):
                        cleaned[key] = None
                else:
                    cleaned[key] = value
            
            return cleaned
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            return None
    
    def extract_with_retry(
        self,
        text: str,
        company_name: str = "",
        max_retries: int = 3
    ) -> Optional[MetricExtraction]:
        """
        Extract metrics with retry logic.
        
        Args:
            text: Document text
            company_name: Company name
            max_retries: Maximum number of retries
            
        Returns:
            MetricExtraction or None
        """
        for attempt in range(max_retries):
            try:
                metrics = self.extract_metrics(text, company_name)
                if metrics:
                    return metrics
                
                if attempt < max_retries - 1:
                    logger.warning(f"Extraction failed, retrying ({attempt + 1}/{max_retries})")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
            except Exception as e:
                logger.error(f"Extraction attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        logger.error("All extraction attempts failed")
        return None
