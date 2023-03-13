import torch
from torch import nn
from d2l import torch as d2l
from IPython.display import display


def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


class Conv2d(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


def comp_2d(conv2d, X):
    """
    helper to initialize the weight of convolution network

    :param conv2d:
    :param X:
    :return:
    """

    # (1, 1, x_row, x_col)
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])


def corr_2d_multi_in(X, K):
    """
    :param X:
    :param K: the convolution kernel
    :return:
    """
    return sum([corr2d(x, k) for x, k in zip(X, K)])


def corr_2d_multi_in_out(X, K):
    """

    :param X:
    :param K: the kernel, for every channel of X, it shall be dealt with a kernel K
    :return:
    """
    # torch.stack([corr2d_multi_in(X, k) for k in K], 0)
    return torch.stack([corr_2d_multi_in(X, k) for k in K], 0)


def corr_2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    # to make X a 2d matrix so that K times X is defined
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))

def pool2d(X, pool_size, mode = "max"):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == "max":
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == "avg":
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

if __name__ == "__main__":
    # X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    # K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    # print((X[0:2, 0:2] * K).sum(), end=''), print((X[0:2, 1:3] * K).sum())
    # print((X[0:2, 1:3] * K).sum(), end=''), print((X[1:3, 1:3] * K).sum())
    # print(corr2d(X, K))
    #
    # X = torch.ones((6, 8))
    # X[:, 2:6] = 0
    # display(X), print(X.shape)
    #
    # K = torch.tensor([[1.0, -1.0]])
    # Y = corr2d(X, K)
    # display(Y), print(Y.shape)
    # display(corr2d(X.t(), K)), print(Y.shape)

    # conv_2d = nn.LazyConv2d(1, kernel_size=3, padding=1)
    # X = torch.rand(size=(8, 8))
    # print(X.reshape((1, 1) + X.shape))
    # print(comp_2D(conv_2d, X).shape)
    # conv_2d = nn.LazyConv2d(1, kernel_size=(5, 3), padding=(2, 1))
    # print(comp_2D(conv_2d, X).shape)
    #
    # d2l.corr2d()

    # X = torch.stack([torch.arange(0., 9.).reshape(3, 3), torch.arange(1., 10.).reshape(3, 3)])
    # K = torch.stack([torch.arange(0., 4.).reshape(2, 2),torch.arange(1., 5.).reshape(2, 2)])
    # display(corr_2d_multi_in(X, K))

    # display(corr_2d_multi_in_out(X, K))

    # X = torch.normal(0, 1, (3, 3, 3))
    # K = torch.normal(0, 1, (2, 3, 1, 1))
    # h, w = K.shape
    # print(h), print(w)
    # Y1 = corr_2d_multi_in_out_1x1(X, K)
    # Y2 = corr_2d_multi_in_out(X, K)
    # assert float(torch.abs(Y1 - Y2).sum()) < 1e-6

    X = torch.arange(0.0, 9.0).reshape(3, 3)
    display(X), display(pool2d(X, (2, 2)))
    display(pool2d(X, (2, 2), mode='avg'))

    X = torch.arange(16.0, dtype=torch.float32).reshape(1, 1, 4, 4)
    pool = nn.MaxPool2d((2, 3), padding=(0, 1), stride=(2, 3))
    display(X)
    display(pool(X))

    # help(zip)
