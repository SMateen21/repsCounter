import torch
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


class FingerTracking(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(63, 63),
            nn.Linear(63, 32),
            nn.ReLU(),
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    df = pd.read_csv('trainData.csv', delimiter=',')

    ds = df.to_numpy()

    y_train = df.iloc[:, 0].to_numpy(copy=True)

    x_train = df.iloc[:, 1:].to_numpy(copy=True)

    X_train = torch.tensor(x_train, dtype=torch.float32)
    Y_train = torch.tensor(y_train, dtype=torch.float32)

    clf = FingerTracking()
    opt = Adam(clf.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    for epoch in range(200):
        yhat = clf(X_train)
        mod_y = np.reshape(Y_train, (205, 1))
        loss = loss_fn(yhat, mod_y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"Epoch: {epoch}, loss = {loss}\n")

    torch.save(clf, "repsCounter.pt")
