import torch
from torch import nn
from torch.nn import functional as F


class MLPModule(nn.Module):
    """
    Lazy Linear, no need to input the in features
    """
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.relu = nn.ReLU()
        self.out = nn.LazyLinear(10)

    def forward(self, X):
        return self.out(self.relu(self.hidden(X)))

class FixedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.LazyLinear(20)
        self.weight = torch.randn((20, 20))

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.matmul(X, self.weight))
        X = self.linear(X)

        while X.abs().sum() > 1:
            X /= 2
        return X

class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.LazyLinear(50), nn.ReLU(),
            nn.LazyLinear(20), nn.ReLU(),
        )

        self.linear = nn.LazyLinear(10)

    def forward(self, X):
        return self.linear(self.seq(X))

# 5 * 10 ^T =>

class TwoLayerMLP(nn.Module):
    def __init__(self):
        self.weight1 = torch.randn((5, 10))
        self.weight2 = torch.randn((5, 5))