import pandas as pd

from evidently import Report

from evidently.presets import (
    DataDriftPreset
)

reference = pd.read_csv(
    "monitoring/baseline.csv"
)

current = pd.read_csv(
    "monitoring/current.csv"
)

report = Report(

    metrics=[

        DataDriftPreset()

    ]

)

snapshot = report.run(

    reference_data=reference,

    current_data=current

)

snapshot.save_html(

    "monitoring/drift_report.html"

)

snapshot.save_json(

    "monitoring/drift_report.json"

)

print(

    "Drift report generated"

)