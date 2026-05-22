"""
DATABASE SCHEMA
---------------
Why this schema? We normalize aggressively so analytics queries are fast.
- companies: ticker → company metadata
- transcripts: one row per earnings call (links company + date + raw text)
- sentences: one row per sentence (the unit of NLP analysis)
- scores: one row per sentence×model combination (separates storage from computation)

This lets you re-run just the FinBERT pass without touching the hedging scores.
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    Text, DateTime, ForeignKey, Boolean, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

Base = declarative_base()


class Company(Base):
    __tablename__ = "companies"
    id         = Column(Integer, primary_key=True)
    ticker     = Column(String(10), unique=True, nullable=False)
    name       = Column(String(200))
    sector     = Column(String(100))
    transcripts = relationship("Transcript", back_populates="company")


class Transcript(Base):
    __tablename__ = "transcripts"
    id           = Column(Integer, primary_key=True)
    company_id   = Column(Integer, ForeignKey("companies.id"))
    earnings_date = Column(DateTime, nullable=False)
    quarter      = Column(String(10))   # e.g. "Q3 2024"
    raw_text     = Column(Text)
    source_url   = Column(String(500))
    created_at   = Column(DateTime, default=datetime.utcnow)

    company    = relationship("Company", back_populates="transcripts")
    sentences  = relationship("Sentence", back_populates="transcript")


class Sentence(Base):
    __tablename__ = "sentences"
    id            = Column(Integer, primary_key=True)
    transcript_id = Column(Integer, ForeignKey("transcripts.id"))
    sentence_idx  = Column(Integer)          # position in transcript
    text          = Column(Text)
    section       = Column(String(50))       # "mda" or "qa"
    speaker       = Column(String(200))      # CEO, CFO, Analyst, etc.

    transcript = relationship("Transcript", back_populates="sentences")
    scores     = relationship("Score", back_populates="sentence")


class Score(Base):
    __tablename__ = "scores"
    id          = Column(Integer, primary_key=True)
    sentence_id = Column(Integer, ForeignKey("sentences.id"))
    model       = Column(String(50))     # "finbert", "hedging", "forward_looking"
    label       = Column(String(50))     # e.g. "positive", "hedged", "forward"
    score       = Column(Float)          # confidence 0-1
    raw_output  = Column(Text)           # store full JSON for debugging

    sentence = relationship("Sentence", back_populates="scores")
    __table_args__ = (Index("ix_scores_sentence_model", "sentence_id", "model"),)


def get_engine(db_url: str = "sqlite:///earnings.db"):
    """
    Default to SQLite for local dev. Swap to PostgreSQL in production:
      db_url = "postgresql://user:pass@localhost:5432/earnings"
    
    WHY POSTGRESQL in prod: Parallel writes from multiple scrapers,
    proper JSON support, much faster aggregation queries on millions of rows.
    """
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()
