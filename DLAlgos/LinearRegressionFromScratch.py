import torch
from d2l import torch as d2l

class LinearRegressionScratch(d2l.Module):
    def __init__(self, num_inputs, lr, sigma = 0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad = True)
        self.b = torch.zeros(1, requires_grad = True)

    def forward(self, X):
        return torch.matmul(X, self.w) + self.b

    def loss(self, y_p, y):
        l = (y_p - y) ** 2 / 2
        return l.mean()

    def configure_optimizers(self):
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

class Train(d2l.Trainer):
    def __init__(self):
        super().__init__()

    def prepare_batch(self, batch):
        return batch

    def fit_epoch(self):
        # what is model.train() (it's in nn.Module) https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
            self.train_batch_idx += 1

        if self.val_data is None:
            return
        self.model.eval()

        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))
                self.val_batch_idx += 1

if __name__ == "__main__":
    model = LinearRegressionScratch(2, lr=0.03)
    data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
    trainer = d2l.Trainer(max_epochs=3)
    trainer.fit(model, data)