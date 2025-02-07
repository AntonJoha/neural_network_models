import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):

    def __init__(self, config=None):
        self.config = config
        self.network = []
        if self.config is None:
            sys.exit("No config")

        super(QNetwork, self).__init__()
        
        self.make_layers()

    
    def make_layers(self):
        dims = [self.config["input"] + self.config["output"]]
        for i in self.config["q_layers"]:
            dims.append(i)

        for i in range(len(dims) - 1):
            self.network.append(nn.Linear(dims[i], dims[i+1], dtype=torch.float, device=device))
            if "activation" not in self.config:
                self.network.append(nn.Sigmoid())
            else:
                self.network.append(self.config["activation"]())
        self.network.append(nn.Linear(dims[-1], 1, dtype=torch.float, device=device))

        self.network = nn.ModuleList(self.network)
        return

    def forward(self, data):
        for l in self.network:
            data = l(data)
        return data


class Actor(nn.Module):

    def make_layers(self):

        dims = [self.config["input"]]
        for i in self.config["layers"]:
            dims.append(i)

        self.network = []

        for i in range(len(dims) - 1):
            self.network.append(
                    nn.Linear(
                        dims[i],
                        dims[i+1],
                        dtype=torch.float,
                        device=device)
                    )

            if "activation" not in self.config:
                self.network.append(nn.Sigmoid())
            else:
                self.network.append(self.config["activation"]())
        self.network.append(
                nn.Linear(
                    dims[-1],
                    self.config["output"],
                    dtype=torch.float,
                    device=device)
                )

        self.network = nn.ModuleList(self.network)

    def forward(self, data):
        for l in self.network:
            data = l(data)
        return data

    def __init__(self, config=None):
        print("HEERE")
        super(Actor, self).__init__()
        self.config = config
        if self.config is None:
            sys.exit("NO CONFIG")

        self.make_layers()


class DDPG:

    def __init__(self,config=None):

        self.config = config
        if self.config is None:
            sys.exit("NO CONFIG")

        self.critic = QNetwork(config)
        self.optimizer_critic = config["optimizer"](self.critic.parameters(),
                                                    lr=config["critic_lr"],
                                                    config=config)

        if "target_network" in config and config["target_network"]:
            self.target_network = QNetwork(config)

        self.actor = Actor(config)
        self.optimizer_actor = config["optimizer"](self.actor.parameters(),
                                                   lr=config["actor_lr"],
                                                   config=config)
        
    
    def update_lr(self, count):
        sys.exit("DO THIS")

    
    def get_noise_rate(self):
        return {"actor": self.optimizer_actor.noise_rate,
                "critic": self.optimizer_critic}
    
    def select_action(self, state, std=0):
        state_tensor = torch.tensor(state, device=device)
        with torch.no_grad():
            if std==0:
                return self.actor(state_tensor)
            return torch.normal(self.actor(state_tensor), std)



if __name__ == "__main__":

    
    # Need to pass a config file. 
    # This is done to have custom optimizers
    def adam_wrapper(parameters, lr, config):
        return optim.Adam(parameters, lr=lr)

    conf = { "input": 2,
            "output": 1,
            "q_layers": [256,256],
            "layers": [256,256],
            "target_network": True,
            "actor_lr": 0.1,
            "critic_lr": 0.1,
            "discount": 0.99,
            "optimizer": adam_wrapper}

    print(QNetwork(conf).network)
    critic = QNetwork(conf)
    actor = Actor(conf)
    ddpg = DDPG(conf)
 
    import gymnasium as gym
    from collections import deque
    from ReplayBuffer import ReplayBuffer

    env = gym.make("MountainCarContinuous-v0")

    state,_ = env.reset()
    print(state)
    action = ddpg.select_action(state,10000.1)
    print(action)
    next_state, reward, terminated, truncated, info = env.step(action.detach().numpy())

    buffer = ReplayBuffer(1000)

    buffer.add([state, action, reward, next_state])
