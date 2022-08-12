# linear regression pytorch model
import torch
class Model(torch.nn.Module):
  def __init__(self, input, output):
    super(Model, self).__init__()
    self.linear = torch.nn.Linear(input,output)

  def forward(self, x):
    out = self.linear(x)
    return out

