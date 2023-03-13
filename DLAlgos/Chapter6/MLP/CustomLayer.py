import torch
from torch import nn
from IPython.display import display
import warnings
from torch.nn import functional as F
from d2l import torch as d2l

warnings.filterwarnings('ignore')

import torchvision


class CenteredLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


class MyLinear(nn.Module):
    def __init__(self, in_units=20, out_units=5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, out_units))
        self.bias = nn.Parameter(torch.randn(out_units, ))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


if __name__ == "__main__":
    X = torch.randn(5, 20)
    linear = MyLinear()
    display(linear(X))
