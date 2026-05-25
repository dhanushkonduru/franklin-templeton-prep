from pathlib import Path

def test_monitoring_exists():

    assert Path(
        "monitoring/drift.py"
    ).exists()


def test_baseline_exists():

    assert Path(
        "monitoring/baseline.csv"
    ).exists()