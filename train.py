from re import A
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import Variable

wine_df = pd.read_csv("winequality-red.csv", sep=";")
#wine_df.sort_values(["fixed acidity"],axis=0,inplace=True)

alcohol = wine_df["fixed acidity"].to_numpy()
quality = wine_df["quality"].to_numpy()

x_train = np.array(alcohol, dtype=np.float32)
y_train = np.array(quality, dtype=np.float32)

#x_train, y_train = np.sort(np.array([x_train,y_train]))
x_train = x_train.reshape(-1,1)
y_train = y_train.reshape(-1,1)

# linear regression pytorch model
class Model(torch.nn.Module):
  def __init__(self, input, output):
    super(Model, self).__init__()
    self.linear = torch.nn.Linear(input,output)

  def forward(self, x):
    out = self.linear(x)
    return out

print(x_train.size)
print(y_train.size)

def train(epochs, input, label, optimizer, loss):
  for epoch in range(epochs):
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'epoch {epoch}, loss {loss.item()}')

epochs = 1000

model = Model(1, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

inputs = torch.Tensor(torch.from_numpy(x_train))
labels = torch.Tensor(torch.from_numpy(y_train))

train(epochs, inputs, labels, optimizer, loss_fn)

with torch.no_grad(): # we don't need gradients in the testing phase
    if torch.cuda.is_available():
        predicted = model(torch.Tensor(torch.from_numpy(x_train).cuda())).cpu().data.numpy()
    else:
        predicted = model(torch.Tensor(torch.from_numpy(x_train))).data.numpy()
    print("Predicted value")
    print(predicted)

torch.save(model.state_dict(), "model.pth")
plt.clf()
#plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
plt.plot(x_train, y_train, '--', label="Data", alpha=0.5)
plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
#plt.show()