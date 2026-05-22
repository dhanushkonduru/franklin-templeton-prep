import torch
import pandas as pd

def predict_mlp(model, X_test):
    
    X_test = X_test.apply(
    pd.to_numeric,
    errors="coerce"
    ).fillna(0)

    X_tensor = torch.tensor(
        X_test.values,
        dtype=torch.float32
    )

    with torch.no_grad():

        predictions = model(
            X_tensor
        ).numpy().flatten()

    return predictions