"""
MAIN PIPELINE
-------------
Ties together: Parser → Scorer → DB → Event Study

Run this file to execute the full pipeline on sample data.
In production you'd call this from a scheduler (cron, Airflow, Prefect)
after earnings season ends.
"""

import sys
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from earnings_detector.models.database import get_engine, get_session, Company, Transcript, Sentence, Score
from earnings_detector.models.parser import parse_transcript, SAMPLE_TRANSCRIPT, ParsedTranscript
from earnings_detector.models.scorer import FinBERTScorer, score_transcript, aggregate_transcript_scores
from earnings_detector.analytics.event_study import compute_car, detect_surprise


# ── Sample data: extend this to 200+ transcripts ────────────────────────────

SAMPLE_CALLS = [
    {
        "ticker": "AAPL",
        "company_name": "Apple Inc",
        "sector": "Technology",
        "earnings_date": datetime(2024, 11, 1),
        "quarter": "Q4 2024",
        "eps_actual": 1.64,
        "eps_estimate": 1.60,
        "transcript": SAMPLE_TRANSCRIPT,
    },
    # Add more here. In production: scrape from Seeking Alpha / Motley Fool
    # Pattern: {ticker, company_name, sector, earnings_date, quarter, transcript}
]


def run_pipeline(calls=None, run_event_study=True):
    """
    Full pipeline run.
    
    Args:
        calls: list of transcript dicts (default: SAMPLE_CALLS)
        run_event_study: if False, skip price data fetch
    """
    if calls is None:
        calls = SAMPLE_CALLS
    
    # 1. Setup database
    engine = get_engine("sqlite:///earnings.db")
    session = get_session(engine)
    
    # 2. Init FinBERT (loads model once, reused across all calls)
    scorer = FinBERTScorer()
    
    all_results = []
    
    print(f"\n{'='*60}")
    print(f"Processing {len(calls)} earnings calls")
    print(f"{'='*60}\n")
    
    for i, call_data in enumerate(calls, 1):
        ticker = call_data['ticker']
        quarter = call_data['quarter']
        print(f"[{i}/{len(calls)}] {ticker} {quarter}")
        
        # ── Step 1: Parse transcript ─────────────────────────────────────
        parsed = parse_transcript(call_data['transcript'])
        print(f"  Parsed: {len(parsed.sentences)} sentences "
              f"({sum(1 for s in parsed.sentences if s.section=='mda')} MDA, "
              f"{sum(1 for s in parsed.sentences if s.section=='qa')} Q&A)")
        
        # ── Step 2: Store company + transcript in DB ─────────────────────
        company = session.query(Company).filter_by(ticker=ticker).first()
        if not company:
            company = Company(
                ticker=ticker,
                name=call_data.get('company_name', ticker),
                sector=call_data.get('sector', 'Unknown')
            )
            session.add(company)
            session.flush()
        
        transcript = Transcript(
            company_id=company.id,
            earnings_date=call_data['earnings_date'],
            quarter=quarter,
            raw_text=call_data['transcript'],
            source_url=call_data.get('source_url', '')
        )
        session.add(transcript)
        session.flush()
        
        # ── Step 3: Score sentences ──────────────────────────────────────
        scores = score_transcript(parsed.sentences, scorer)
        
        # ── Step 4: Store sentences and scores in DB ─────────────────────
        for sent, score_obj in zip(parsed.sentences, scores):
            db_sentence = Sentence(
                transcript_id=transcript.id,
                sentence_idx=sent.sentence_idx,
                text=sent.text,
                section=sent.section,
                speaker=sent.speaker,
            )
            session.add(db_sentence)
            session.flush()
            
            # FinBERT score
            session.add(Score(
                sentence_id=db_sentence.id,
                model="finbert",
                label=score_obj.finbert_label,
                score=score_obj.finbert_scores.get(score_obj.finbert_label, 0),
                raw_output=json.dumps(score_obj.finbert_scores)
            ))
            
            # Hedging score
            session.add(Score(
                sentence_id=db_sentence.id,
                model="hedging",
                label=','.join(score_obj.hedging_types) if score_obj.hedging_types else "none",
                score=score_obj.hedge_score,
                raw_output=json.dumps({"types": score_obj.hedging_types})
            ))
            
            # Forward-looking score
            session.add(Score(
                sentence_id=db_sentence.id,
                model="forward_looking",
                label="forward" if score_obj.is_forward_looking else "historical",
                score=score_obj.forward_score,
                raw_output=json.dumps({"score": score_obj.forward_score})
            ))
        
        session.commit()
        
        # ── Step 5: Aggregate transcript stats ───────────────────────────
        agg = aggregate_transcript_scores(scores)
        overall_sentiment = agg['overall'].get('sentiment_score', 0)
        
        print(f"  Sentiment: {overall_sentiment:+.3f} "
              f"(pos={agg['overall'].get('avg_positive',0):.2f}, "
              f"neg={agg['overall'].get('avg_negative',0):.2f})")
        print(f"  Hedging rate: {agg['overall'].get('avg_hedge_score',0):.2f}, "
              f"Forward-looking: {agg['overall'].get('pct_forward_looking',0)*100:.0f}%")
        
        # ── Step 6: Event study ──────────────────────────────────────────
        result_summary = {
            'ticker': ticker,
            'quarter': quarter,
            'earnings_date': call_data['earnings_date'].strftime('%Y-%m-%d'),
            'sentiment_score': overall_sentiment,
            'section_scores': agg,
            'n_sentences': len(scores),
        }
        
        if run_event_study:
            print(f"  Fetching price data for event study...")
            car_result = compute_car(
                ticker=ticker,
                event_date=call_data['earnings_date'],
                sentiment_score=overall_sentiment,
            )
            result_summary['event_study'] = car_result
            
            if 'car_pct' in car_result:
                print(f"  CAR(-1,+1): {car_result['car_pct']:+.2f}%")
                print(f"  Divergent: {car_result.get('is_divergent', False)}")
            
            # Surprise detection (if EPS data available)
            if 'eps_actual' in call_data and 'eps_estimate' in call_data:
                surprise = detect_surprise(
                    sentiment_score=overall_sentiment,
                    eps_actual=call_data['eps_actual'],
                    eps_estimate=call_data['eps_estimate'],
                    car=car_result.get('car', 0)
                )
                result_summary['surprise'] = surprise
                if surprise['is_divergent']:
                    print(f"  ⚡ SURPRISE: {surprise['divergence_type']}")
        
        all_results.append(result_summary)
        print()
    
    session.close()
    return all_results


def print_analysis_summary(results):
    """Print a clean analysis table."""
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"{'Ticker':8} {'Quarter':12} {'Sentiment':10} {'CAR%':8} {'Divergent':10} {'Type'}")
    print("-"*60)
    
    for r in results:
        es = r.get('event_study', {})
        car_pct = es.get('car_pct', float('nan'))
        divergent = es.get('is_divergent', '-')
        surp_type = r.get('surprise', {}).get('divergence_type', '-') or '-'
        
        print(f"{r['ticker']:8} {r['quarter']:12} "
              f"{r['sentiment_score']:+.3f}     "
              f"{car_pct:+.2f}%  "
              f"{'YES' if divergent else 'no':10} "
              f"{surp_type}")
    
    print("\nINTERVIEW HOOK:")
    n_divergent = sum(1 for r in results 
                      if r.get('event_study', {}).get('is_divergent', False))
    print(f"  {n_divergent}/{len(results)} calls showed sentiment-price divergence.")
    print("  Example: management was upbeat but CAR was negative →")
    print("  'credibility gap' — market didn't believe the guidance.")


if __name__ == "__main__":
    # Full pipeline run
    results = run_pipeline(
        calls=SAMPLE_CALLS,
        run_event_study=True
    )
    
    print_analysis_summary(results)
    
    # Save results to JSON for further analysis
    # Make results serializable
    def make_serializable(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        if isinstance(obj, (bool, int, float, str)) or obj is None:
            return obj
        return str(obj)  # catch numpy bools, np.float64, etc.
    
    with open('results.json', 'w') as f:
        json.dump(make_serializable(results), f, indent=2)
    print("\nResults saved to results.json")
