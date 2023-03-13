# class Classifier:

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


from d2l import torch as d2l

# d2l.Module
# d2l.Classifier
# d2l.Trainer
# d2l.FashionMNIST
# d2l.DataModule
# d2l.HyperParameters

class DataModule():
    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)


class FMNIST(DataModule):
    def __init__(self, batch_size=64, resize=(28, 28), root='./data'):
        super().__init__()
        trans = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
        ])
        self.batch_size = batch_size
        self.root = root
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)

    def text_labels(self, indices):
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train):
        """Defined in :numref:`sec_fashion_mnist`"""
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,)


class Trainer(d2l.HyperParameters):
    """Defined in :numref:`subsec_oo-design-models`"""
    def __init__(self, max_epochs , gradient_clip_val=0):
        """Defined in :numref:`sec_use_gpu`"""
        self.max_epochs = max_epochs
        # self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

    def prepare_batch(self, batch):
        """Defined in :numref:`sec_use_gpu`"""
        # if self.gpus:
        #     batch =
        return [a for a in batch]

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        """Defined in :numref:`sec_use_gpu`"""
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        # if self.gpus:
        #     model.to(self.gpus[0])
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()


# d2l.FashionMNIST
class KafuuClassifer(nn.Module):
    # def
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the number of correct predictions."""
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        preds = Y_hat.argmax(axis=1).type(Y.dtype)
        compare = (preds == Y.reshape(-1)).type(torch.float32)
        return compare.mean() if averaged else compare

class LeNet(nn.Module):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.Sigmoid(),
            nn.LazyLinear(84), nn.Sigmoid(),
            nn.LazyLinear(num_classes)
        )

    def forward(self, X):
        return self.net(X)

    # def layer_summary(self, X_shape):
    #     X = torch.randn(*X_shape)
    #     for layer in self.net:
    #         X = layer(X)
    #         print(layer.__class__.__name__, 'output_shape :\t', X.shape)
    #
    # def apply_init(self, inputs, init=None):
    #     """Defined in :numref:`sec_lazy_init`"""
    #     self.forward(*inputs)
    #     if init is not None:
    #         self.net.apply(init)


if __name__ == "__main__":
    # data = FMNIST(batch_size=128)
    # trainer = Trainer(max_epochs=10)
    # model = LeNet(lr=0.1)

    # model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)
    # model.fit(data.train)

    # model = LeNet()
    # model.layer_summary((1, 1, 28, 28))

    # def init_cnn(module):  # @save
    #     """Initialize weights for CNNs."""
    #     if type(module) == nn.Linear or type(module) == nn.Conv2d:
    #         nn.init.xavier_uniform_(module.weight)

    #
    # trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
    # data = d2l.FashionMNIST(batch_size=128)
    # model = d2l.LeNet(lr=0.1)
    # model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)
    # trainer.fit(model, data)
    # torch.save(model, "./lenet.pt")

    data = FMNIST()
    print(len(data.train[0]))

