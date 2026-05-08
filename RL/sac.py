import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys

from .networks import CriticNetwork, device

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

        self.critic_1 = CriticNetwork(config)
        self.critic_2 = CriticNetwork(config)
        self.optimizer_critic_1 = config["optimizer"](self.critic_1.parameters(),
                                                    lr=config["critic_lr"],
                                                    config=config)

        self.optimizer_critic_2 = config["optimizer"](self.critic_2.parameters(),
                                                    lr=config["critic_lr"],
                                                    config=config)

        if "target_network" in config and config["target_network"]:
            self.target_network_1 = CriticNetwork(config)
            self.target_network_2 = CriticNetwork(config)

        self.actor = Actor(config)
        self.optimizer_actor = config["optimizer"](self.actor.parameters(),
                                                   lr=config["actor_lr"],
                                                   config=config)
        self.loss_function = nn.MSELoss()

        

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
        actions_tensor = torch.tensor(actions,dtype=torch.float,device=device).view(-1, 1)
        rewards_tensor = torch.tensor(rewards,dtype=torch.float,device=device).view(-1, 1)
        next_states_tensor = torch.tensor(next_states,dtype=torch.float,device=device)


        # Q-values for the next states
        if self.config["target_network"] and target_network:
            with torch.no_grad():
                _, _, next_actions = self.actor(next_states_tensor)
                next_q_1 = self.target_network_1(torch.cat((next_states_tensor, next_actions), dim=1))
                next_q_2 = self.target_network_2(torch.cat((next_states_tensor, next_actions), dim=1))
                next_q_values = torch.min(next_q_1, next_q_2)
        else:
            with torch.no_grad():
                _, _, next_actions = self.actor(next_states_tensor)
                next_q_1 = self.critic_1(torch.cat((next_states_tensor, next_actions), dim=1))
                next_q_2 = self.critic_2(torch.cat((next_states_tensor, next_actions), dim=1))
                next_q_values = torch.min(next_q_1, next_q_2)

        # Calculate target Q-values
        target_q_values = rewards_tensor + self.config["discount"] * next_q_values

        self.optimizer_critic_1.zero_grad()
        q_values_1 = self.critic_1(torch.cat((states_tensor, actions_tensor), dim=1))
        loss_1 = self.loss_function(q_values_1, target_q_values)
        loss_1.backward()
        self.optimizer_critic_1.step()

        self.optimizer_critic_2.zero_grad()
        q_values_2 = self.critic_2(torch.cat((states_tensor, actions_tensor), dim=1))
        loss_2 = self.loss_function(q_values_2, target_q_values)
        loss_2.backward()
        self.optimizer_critic_2.step()

        self.optimizer_actor.zero_grad()
        _, _, actor_actions = self.actor(states_tensor)
        actor_loss = -self.critic_1(torch.cat((states_tensor, actor_actions), dim=1)).mean()
        actor_loss.backward()
        self.optimizer_actor.step()

        return (loss_1 + loss_2).detach()
 

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

    print(CriticNetwork(conf).network)
    critic = CriticNetwork(conf)
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
