import torch
from IPython.core.display_functions import display
from torch import nn
import MLP
import InitializeParameters as INPA
import warnings

warnings.simplefilter('ignore')

from torch.nn import functional as F

# DEFINE AN AFFINE TRANSFORMATION
# net = nn.Sequential(
#     nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10), nn.Sigmoid(),
# )
#
# X = torch.rand(2, 20)
# net(X)

# def init_normal(module):
#     if type(module) == nn.Linear:
#         nn.init.normal_(module.weight, mean=0, std=0.01)
#         nn.init.zeros_(module.bias)

if __name__ == '__main__':
    # shared = nn.LazyLinear(10)
    # 10 * 4 * (4 * 10)^T => 10 * 10
    net = nn.Sequential(nn.LazyLinear(8),
                        nn.ReLU(),
                        nn.LazyLinear(9), nn.ReLU(),
                        nn.LazyLinear(1))
    #
    X = torch.rand(size=(10, 10))
    # net(X)
    # display((net[0].bias != net[0].bias).sum())
    # # display() 2 * 4 * (8 * 4)^T + b
    # display([(name, param.shape) for name, param in net.named_parameters()])
    #
    # nn.init.xavier_normal_()

    net(X)
    net.apply(INPA.my_init)
    print(net[0].weight.data)
