import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split

wine_df = pd.read_csv("winequality-red.csv", sep=";")

input_cols_df = list(wine_df.columns)[:-1]

x = wine_df[input_cols_df].to_numpy(dtype=np.float32)
y = wine_df["quality"].to_numpy(dtype=np.float32)
y = y.reshape(1599, 1)

inputs = torch.from_numpy(x).type(torch.float)
target = torch.from_numpy(y).type(torch.float)

dataset = TensorDataset(inputs, target)

train_ds, val_ds = random_split(dataset, [1300, 299])

class WineModel(torch.nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.linear = torch.nn.Linear(input, output)

    def forward(self, x):
        out = self.linear(x)
        return out

model = WineModel(11,1)
epochs = 500

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(epochs, input, label, optimizer, loss):
    for epoch in range(epochs):
        outputs = model(input)
        loss = loss_fn(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'epoch {epoch}, loss {loss.item()}')

train(epochs, inputs, target, optimizer, loss_fn)
