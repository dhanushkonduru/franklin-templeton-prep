"""
Utility functions for ESG Analyzer.
Common utilities used across the application.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def save_json(data: Any, file_path: Path | str, indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save (dict, list, etc.)
        file_path: Path to output file
        indent: JSON indentation
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=indent, default=str)
        logger.info(f"Saved JSON to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {str(e)}")
        raise


def load_json(file_path: Path | str) -> Any:
    """
    Load JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded JSON data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded JSON from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {str(e)}")
        raise


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    return text


def extract_year_from_text(text: str) -> Optional[int]:
    """
    Extract year from text.
    
    Args:
        text: Text to search
        
    Returns:
        Year or None
    """
    import re
    
    match = re.search(r'\b(20\d{2}|19\d{2})\b', text)
    if match:
        return int(match.group(1))
    
    return None


def format_percentage(value: Optional[float], decimals: int = 1) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Value to format (0-100)
        decimals: Decimal places
        
    Returns:
        Formatted percentage string
    """
    if value is None:
        return "N/A"
    
    return f"{value:.{decimals}f}%"


def format_metric(value: Optional[float | int], unit: str = "", decimals: int = 1) -> str:
    """
    Format metric for display.
    
    Args:
        value: Metric value
        unit: Unit string
        decimals: Decimal places
        
    Returns:
        Formatted metric string
    """
    if value is None:
        return "N/A"
    
    if isinstance(value, int):
        return f"{value:,}{unit}"
    
    return f"{value:.{decimals}f}{unit}"


def calculate_percentile(value: float, values: List[float]) -> float:
    """
    Calculate percentile rank.
    
    Args:
        value: Value to rank
        values: List of all values
        
    Returns:
        Percentile (0-100)
    """
    if not values:
        return 0.0
    
    rank = sum(1 for v in values if v <= value)
    return (rank / len(values)) * 100


def get_time_elapsed(start_time: datetime, end_time: Optional[datetime] = None) -> str:
    """
    Get human-readable elapsed time.
    
    Args:
        start_time: Start datetime
        end_time: End datetime (defaults to now)
        
    Returns:
        Human-readable elapsed time
    """
    end_time = end_time or datetime.now()
    elapsed = end_time - start_time
    
    total_seconds = int(elapsed.total_seconds())
    
    if total_seconds < 60:
        return f"{total_seconds}s"
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}m {seconds}s"
    else:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def batch_list(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split list into batches.
    
    Args:
        items: List to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def compare_values(old: float, new: float) -> Dict[str, Any]:
    """
    Compare two values and calculate difference.
    
    Args:
        old: Old value
        new: New value
        
    Returns:
        Dictionary with comparison metrics
    """
    absolute_change = new - old
    percent_change = (absolute_change / abs(old)) * 100 if old != 0 else 0
    improved = new > old
    
    return {
        "old_value": old,
        "new_value": new,
        "absolute_change": absolute_change,
        "percent_change": percent_change,
        "improved": improved,
        "direction": "↑" if improved else "↓"
    }


class ProgressTracker:
    """Track progress of batch operations."""
    
    def __init__(self, total: int, name: str = "Operation"):
        """Initialize tracker."""
        self.total = total
        self.name = name
        self.current = 0
        self.start_time = datetime.now()
    
    def increment(self, count: int = 1):
        """Increment progress."""
        self.current += count
        self._log_progress()
    
    def _log_progress(self):
        """Log current progress."""
        percent = (self.current / self.total) * 100
        elapsed = get_time_elapsed(self.start_time)
        
        remaining = self.total - self.current
        if self.current > 0:
            avg_time_per_item = (datetime.now() - self.start_time).total_seconds() / self.current
            estimated_remaining = remaining * avg_time_per_item
            eta = get_time_elapsed(datetime.now(), 
                                   datetime.now().replace(microsecond=0) + 
                                   __import__('datetime').timedelta(seconds=estimated_remaining))
        else:
            eta = "N/A"
        
        logger.info(
            f"{self.name}: {self.current}/{self.total} ({percent:.1f}%) - "
            f"Elapsed: {elapsed}, ETA: {eta}"
        )
