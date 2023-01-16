import torch
from d2l import torch as d2l

class LinearRegressionScatch(d2l.Module):
    def __init__(self, num_inputs, lr, sigma = 0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad = True)
        self.b = torch.zeros(1, require_grad = True)

    def forward(self, X):
        return torch.matmul(X, self.w) + b

    def loss(self, y_p, y):
        l = (y_p - y) ** 2 / 2
        return l.mean()

    def configure_optimizer(self):
        return SGD([self.w, self.b], self.lr)


class SGD(d2l.HyperParameters):
    def __init__(self, params, lr):
        self.save_hyperparameters()

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()