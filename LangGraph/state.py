from pydantic import BaseModel
from typing import Optional

class ResearchState(BaseModel):
    ticker: str
    stock_price: Optional[float] = None
    company_name: Optional[str] = None
    sentiment: Optional[str] = None
    report: Optional[str] = None