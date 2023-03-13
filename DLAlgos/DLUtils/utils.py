import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.transforms as transforms


# data prepare
class DataModule(object):
    def __init__(self, batch_size=60, num_workers=-1):
        """
        you shall define training dataset, valuation dataset
        :param batch_size:
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = None
        self.val = None

    # def train_dataloader(self):

    def get_training_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def get_val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=True)

    # def data2tensor(self, data, feature):

    def random2tensor(self, in_data, random_shuffle=True, indices=slice(0, None)):
        tensors = [a[indices] for a in in_data]
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, self.batch_size, shuffle=random_shuffle)


class MnistDataSet(DataModule):
    def __init__(self, root="./root", download=True):
        super().__init__()
        self.graph_shape = (28, 28)
        self.random_pad_size = 2
        self.training_dataset, self.val_dataset = self.set_training_val_data(root=root, train=True, download=download)
        self.test = self.set_test_data(root=root, train=False, download=download)

    def set_training_val_data(self, root, train, download):
        train_val_dataset = torchvision.datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.Compose([
                # transforms.RandomCrop(self.mnist_image_shape,
                #                       self.random_pad_size),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5), (0.5, 0.5))
            ])
        )

        num_train = int(0.5 * len(train_val_dataset))
        num_val = len(train_val_dataset) - num_train

        return torch.utils.data.random_split(train_val_dataset, lengths=[num_train, num_val])

    def set_test_data(self, root, train=False, download=True):
        test = torchvision.datasets.MNIST(
            root=root, train=train, download=download,
            transform=transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5), (0.5, 0.5))
            ])
        )

        return test

    def get_training_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def get_val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=True)

    def get_test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=True)


class NNModule(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def train_step(self, batch):
        train_loss = self.loss(self(*batch[:-1]), batch[-1])
        return train_loss

    def val_step(self, batch):
        val_loss = self.loss(self(*batch[:-1]), batch[-1])

    def config_optimizer(self):
        return torch.optim.SGD(nn.parameter(), lr=self.lr)

    def apply_init__(self, inputs, init=None):
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)


# trainer
class NNTrainer(object):
    def __init__(self, model):
        self.train_dataloader = None
        self.val_dataloader = None
        self.train_batch = None
        self.val_batch = None
        self.model = model
        self.optim = model.config_optimizer()
        self.train_index = 0
        self.val_index = 0

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.train_batch = len(self.train_dataloader)
        self.val_batch = len(self.val_dataloader)

    def fit_epoch(self):
        for batch in self.train_dataloader:
            loss = self.model.train_step(batch)
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                self.optim.step()
            self.train_index += 1

        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.val_step(batch)

            self.val_index += 1


class SingleLayerMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.criterion = None
        self.w = nn.parameter(torch.randn(in_dim, requires_grad=True))
        self.b = nn.parameter(torch.randn([1]))
        # self.criterion =

    def forward(self, x):
        y_pred = torch.matmul(self.w, x) + self.b
        return y_pred

    def set_criterion(self, criterion):
        self.criterion = criterion


if __name__ == "__main__":
    MNIST = MnistDataSet()