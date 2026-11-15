from torch import nn


class Net(nn.Module):
    """Tiny MLP placeholder: uv (2) -> rgb (3)."""

    def __init__(self, sizes=(2, 16, 16, 3)):
        super().__init__()
        self.sizes = sizes
        layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())  # TODO: choose activation
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
