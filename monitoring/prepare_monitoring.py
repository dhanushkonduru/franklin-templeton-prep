import pandas as pd

dataset = pd.read_csv(
    "output/final_dataset.csv"
)

columns = [

    "return_1d",

    "momentum_3m",

    "momentum_12m",

    "volatility",

    "pe_ratio",

    "roe",

    "debt_to_equity",

    "revenue_growth"

]

dataset = dataset[columns]

split = int(

    len(dataset)*0.8

)

baseline = dataset.iloc[:split]

current = dataset.iloc[split:]

baseline.to_csv(

    "monitoring/baseline.csv",

    index=False

)

current.to_csv(

    "monitoring/current.csv",

    index=False

)

print(

    "Monitoring datasets prepared"

)