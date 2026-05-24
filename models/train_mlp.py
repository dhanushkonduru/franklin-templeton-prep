import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from models.mlp_model import (
    MLPRegressor
)

def train_mlp(X_train, y_train):

    X_train = X_train.apply(
        pd.to_numeric,
        errors="coerce"
    ).fillna(0)

    X_tensor = torch.tensor(
        X_train.values.astype(float),
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y_train.values,
        dtype=torch.float32
    ).view(-1, 1)

    model = MLPRegressor(
        X_train.shape[1]
    )

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    for epoch in range(50):

        optimizer.zero_grad()

        outputs = model(X_tensor)

        loss = criterion(
            outputs,
            y_tensor
        )

        loss.backward()

        optimizer.step()

    return model