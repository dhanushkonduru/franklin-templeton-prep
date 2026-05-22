from pydantic import BaseModel
from typing import Optional, List, Dict

class ResearchState(BaseModel):

    ticker: str

    company_name: Optional[str] = None

    stock_price: Optional[float] = None

    news: List[str] = []

    fundamentals: Dict = {}

    sentiment_analysis: Optional[str] = None

    risk_analysis: Optional[str] = None

    final_report: Optional[str] = None

    audit_log: List[str] = []

    critic_review: Optional[str] = None

    revision_count: int = 0
    
    retrieved_context: str = ""
    
    citations: list[str] = []