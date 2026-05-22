"""
Configuration management for ESG Analyzer.
"""

import os
from pathlib import Path
from typing import Dict, Any
import json
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    """Central configuration for the ESG Analyzer system."""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    REPORTS_DIR = DATA_DIR / "reports"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # Create directories if they don't exist
    for dir_path in [DATA_DIR, REPORTS_DIR, OUTPUTS_DIR, MODELS_DIR, LOGS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Groq Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = "llama-3.3-70b-versatile"
    GROQ_TEMPERATURE = 0.3
    GROQ_MAX_TOKENS = 2000
    
    # PDF Processing
    PDF_CHUNK_SIZE = 1000  # Characters
    PDF_OVERLAP = 100  # Characters
    MAX_PDF_SIZE_MB = 50
    
    # NLP Processing
    SPACY_MODEL = "en_core_web_sm"
    MIN_SENTENCE_LENGTH = 10
    
    # ESG Scoring Weights
    ESG_WEIGHTS = {
        "carbon_metrics": 0.30,
        "renewable_metrics": 0.30,
        "water_metrics": 0.20,
        "governance_metrics": 0.10,
        "greenwashing_adjustment": 0.10,
    }
    
    # Greenwashing Keywords
    GREEN_CLAIM_KEYWORDS = [
        "committed", "leading", "sustainable", "green", "carbon neutral",
        "net zero", "climate action", "environmental steward", "eco-friendly",
        "renewable", "zero waste", "circular economy", "sustainability leader",
        "climate positive", "carbon negative", "zero emissions"
    ]
    
    # SASB/GRI Taxonomy Keywords
    TAXONOMY_KEYWORDS = {
        "Environmental": {
            "carbon": ["carbon", "emissions", "ghg", "co2", "greenhouse gas"],
            "emissions": ["emissions", "scope 1", "scope 2", "scope 3", "emission reduction"],
            "water": ["water", "wastewater", "water consumption", "water quality"],
            "renewable": ["renewable", "solar", "wind", "hydroelectric", "sustainable energy"],
            "waste": ["waste", "recycling", "landfill", "hazardous waste", "circular"]
        },
        "Social": {
            "supplier": ["supplier", "procurement", "supply chain", "vendor", "third-party"],
            "training": ["training", "development", "education", "learning", "skill"],
            "employee": ["employee", "workforce", "staff", "labor", "human capital"],
            "community": ["community", "social", "local", "stakeholder", "outreach"]
        },
        "Governance": {
            "audit": ["audit", "compliance", "control", "assurance", "risk management"],
            "policy": ["policy", "governance", "code of conduct", "ethics", "standards"],
            "board": ["board", "director", "executive", "leadership", "management"],
            "compliance": ["compliance", "regulation", "legal", "requirement", "standards"]
        }
    }
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Feature Flags
    USE_LLM_EXTRACTION = True
    USE_REGEX_EXTRACTION = True
    SAVE_INTERMEDIATE_RESULTS = True
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }
    
    @classmethod
    def validate(cls) -> bool:
        """Validate critical configuration."""
        if cls.USE_LLM_EXTRACTION and not cls.GROQ_API_KEY:
            logger.warning("Groq API key not set. LLM extraction will be disabled.")
            cls.USE_LLM_EXTRACTION = False
        return True


def setup_logging(name: str = __name__) -> logging.Logger:
    """Setup logging for the application."""
    logger = logging.getLogger(name)
    logger.setLevel(Config.LOG_LEVEL)
    
    # Ensure logs directory exists
    Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # File handler
    fh = logging.FileHandler(Config.LOGS_DIR / "esg_analyzer.log")
    fh.setLevel(Config.LOG_LEVEL)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(Config.LOG_LEVEL)
    
    # Formatter
    formatter = logging.Formatter(Config.LOG_FORMAT)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger
