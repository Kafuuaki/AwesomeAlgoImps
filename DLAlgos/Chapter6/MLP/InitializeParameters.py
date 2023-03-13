import torch
from torch import nn


# by default, the linear layer provides uniform initialization
"""
    Begin applying function to module, you shall take one forward step
"""

def init_norm(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        nn.init.zeros_(module.bias)


def init_constant(module, constant=1):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, constant)
        nn.init.zeros_(module.bias)


def init_xavier(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)


def my_init(module):
    if type(module) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5

def apply_init(self, inputs, init=None):
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)