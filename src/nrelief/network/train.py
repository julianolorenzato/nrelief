import torch

from .net import Net


def make_dataset():
    # TODO: replace with real data
    x = torch.rand(1024, 2)
    y = torch.rand(1024, 3)
    return x, y


def train(out="checkpoint.pt", epochs=1, lr=1e-3):
    model = Net()
    x, y = make_dataset()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        # TODO: batching, loss, logging
        loss = torch.nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.save({"sizes": model.sizes, "state": model.state_dict()}, out)
