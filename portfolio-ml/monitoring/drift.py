import os
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset


if (
    not os.path.exists(
        "monitoring/baseline.csv"
    )
    or
    not os.path.exists(
        "monitoring/current.csv"
    )
):

    print(
        "Monitoring data missing. Skipping drift."
    )

    exit(0)


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


result = snapshot.dict()


drift_count = \
result["metrics"][0][

    "value"

][

    "count"

]


drift_share = \
result["metrics"][0][

    "value"

][

    "share"

]


print(

    f"\nDrifted Columns: {drift_count}"

)

print(

    f"Drift Share: {drift_share:.2f}"

)


if drift_share > 0.3:

    print(

        "\nALERT DRIFT DETECTED\n"

    )

else:

    print(

        "\nNo significant drift\n"

    )