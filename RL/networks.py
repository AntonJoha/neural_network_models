import torch
import torch.nn as nn
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """
    Generic MLP actor/policy network.

    Config keys:
        input      -- number of input features
        layers     -- list of hidden layer sizes
        output     -- number of output features
        activation -- (optional) activation class; defaults to nn.Sigmoid
    """

    def __init__(self, config=None):
        super(Actor, self).__init__()
        self.config = config
        if self.config is None:
            sys.exit("No config")
        self.make_layers()

    def make_layers(self):
        dims = [self.config["input"]]
        for i in self.config["layers"]:
            dims.append(i)

        self.network = []

        for i in range(len(dims) - 1):
            self.network.append(
                nn.Linear(dims[i], dims[i + 1], dtype=torch.float, device=device)
            )
            if "activation" not in self.config:
                self.network.append(nn.Sigmoid())
            else:
                self.network.append(self.config["activation"]())
        self.network.append(
            nn.Linear(dims[-1], self.config["output"], dtype=torch.float, device=device)
        )

        self.network = nn.ModuleList(self.network)

    def forward(self, data):
        for l in self.network:
            data = l(data)
        return data


class CriticNetwork(nn.Module):
    """
    Critic (Q) network for continuous action spaces.

    Takes a concatenation of (state, action) as input and outputs a scalar value.

    Config keys:
        input      -- number of state features
        output     -- number of action features
        q_layers   -- list of hidden layer sizes
        activation -- (optional) activation class; defaults to nn.Sigmoid
    """

    def __init__(self, config=None):
        self.config = config
        self.network = []
        if self.config is None:
            sys.exit("No config")

        super(CriticNetwork, self).__init__()

        self.make_layers()

    def make_layers(self):
        dims = [self.config["input"] + self.config["output"]]
        for i in self.config["q_layers"]:
            dims.append(i)

        for i in range(len(dims) - 1):
            self.network.append(
                nn.Linear(dims[i], dims[i + 1], dtype=torch.float, device=device)
            )
            if "activation" not in self.config:
                self.network.append(nn.Sigmoid())
            else:
                self.network.append(self.config["activation"]())
        self.network.append(nn.Linear(dims[-1], 1, dtype=torch.float, device=device))

        self.network = nn.ModuleList(self.network)

    def forward(self, data):
        for l in self.network:
            data = l(data)
        return data
