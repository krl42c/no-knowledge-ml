from sklearn import datasets
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score

iris = datasets.load_iris()

x_data = iris.data
y_data = iris.target

x = torch.FloatTensor(x_data)
y = torch.FloatTensor(y_data)

y = y.type(torch.LongTensor)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear_stack = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.Linear(128, 64),
            torch.nn.Linear(64,3)
        )

    def forward(self,x):
        ret = self.linear_stack(x)
        return ret

model = Model()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 1000
for epoch in range(epochs):
    pred = model(x)
    loss = loss_fn(pred,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Loss: {loss.item()}")
    
pred = model(x)
_, pred_y = torch.max(pred,1)

acc = accuracy_score(y.data,pred_y.data)*100.3
print(f"Accuracy {acc}")

print(pred_y)
print(y)
