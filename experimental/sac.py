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
        if "activation" not in self.config:
            self.entropy = nn.Sequential(nn.Linear(self.config["input"], dims[0]), nn.Sigmoid(), nn.Linear(dims[0],self.config["output"]), nn.Sigmoid())
        else:
            self.entropy = nn.Sequential(nn.Linear(self.config["input"], dims[0]),self.config["activation"],
                                        nn.Linear(dims[0],self.config["output"]),
                                         self.config["activation"]())


    def forward(self, data):
        d = data.detach().clone()
        for l in self.network:
            data = l(data)
        noise = self.entropy(d)
        return data, noise, data + noise

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

        self.make_networks(config)

    def make_networks(self, config):

        self.critic_1 = QNetwork(config)
        self.critic_2 = QNetwork(config)
        self.optimizer_critic_1 = config["optimizer"](self.critic_1.parameters(),
                                                    lr=config["critic_lr"],
                                                    config=config)

        self.optimizer_critic_2 = config["optimizer"](self.critic_2.parameters(),
                                                    lr=config["critic_lr"],
                                                    config=config)

        if "target_network" in config and config["target_network"]:
            self.target_network_1 = QNetwork(config)
            self.target_network_2 = QNetwork(config)

        self.actor = Actor(config)
        self.optimizer_actor = config["optimizer"](self.actor.parameters(),
                                                   lr=config["actor_lr"],
                                                   config=config)

        

    def update_lr(self, count):
        sys.exit("DO THIS")

    
    
    def select_action(self, state, entropy=False):
        state_tensor = torch.tensor(state, device=device)
        with torch.no_grad():
            if entropy:
                return self.actor(state_tensor)[2]
            return self.actor(state_tensor)[0]

    
    def replay(self,replay_buffer,batch_size=128,target_network=True):

 
        if replay_buffer.buffer_size() < batch_size:
            return

        # Sample a batch of experiences from the replay buffer
        states, actions, rewards, next_states = replay_buffer.sample(batch_size)

        # Convert to tensors
        states_tensor = torch.tensor(states,dtype=torch.float,device=device)
        actions_tensor = torch.tensor(actions,dtype=torch.long,device=device).view(-1, 1)
        rewards_tensor = torch.tensor(rewards,dtype=torch.float,device=device).view(-1, 1)
        next_states_tensor = torch.tensor(next_states,dtype=torch.float,device=device)


                # Q-values for the next states (target Q-network)
        if self.config["target_network"]:
            with torch.no_grad():
                next_q_values = self.target_network(next_states_tensor).max(1)[0].unsqueeze(1)
        else:
            with torch.no_grad():
                next_q_values = self.q_network(next_states_tensor).max(1)[0].unsqueeze(1)

        # Calculate target Q-values
        target_q_values = rewards_tensor + self.config["discount"] * next_q_values
        
        self.optimizer.zero_grad()
        # Q-values for the current state-action pairs
        q_values = self.q_network(states_tensor).gather(1,actions_tensor)


        # Update the Q-network
        loss = self.loss_function(q_values, target_q_values)
        loss_r = loss.item()
        loss.backward()
        self.optimizer.step()
        return loss
 

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
    action = ddpg.select_action(state,True)
    print(action)
    next_state, reward, terminated, truncated, info = env.step(action.detach().numpy())

    buffer = ReplayBuffer(1000)

    buffer.add([state, action, reward, next_state])


