import model
import torch
import pandas as pd
import numpy as np

model = model.Model(1,1)
model.load_state_dict(torch.load("model.pth"))

#x = torch.Tensor([7.8])
#y = torch.Tensor([0.88])
wine_df = pd.read_csv("winequality-red.csv", sep=";")

x = wine_df["fixed acidity"].to_numpy(dtype=np.float32)
y = wine_df["quality"].to_numpy(dtype=np.float32)

x = x.reshape(-1,1)
y = y.reshape(-1,1)

inputs = torch.Tensor(torch.from_numpy(x))
labels = torch.Tensor(torch.from_numpy(y))

model.eval()

with torch.no_grad():
  for i in range(0, x.size):
    pred = model(inputs)
    predicted, real = pred[i], y[i]
    print(f'Prediced: {predicted}, Real: {real}')

