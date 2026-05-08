import sys

import torch
import torch.nn as nn
import torch.optim as optim

from .networks import Actor, CriticNetwork, device


class DDPG:

    def __init__(self,config=None):

        self.config = config
        if self.config is None:
            sys.exit("NO CONFIG")

        self.critic = CriticNetwork(config)
        self.optimizer_critic = config["optimizer"](self.critic.parameters(),
                                                    lr=config["critic_lr"],
                                                    config=config)

        self.target_network = None
        if "target_network" in config and config["target_network"]:
            self.target_network = CriticNetwork(config)

        self.actor = Actor(config)
        self.optimizer_actor = config["optimizer"](self.actor.parameters(),
                                                   lr=config["actor_lr"],
                                                   config=config)
        self.loss_function = nn.MSELoss()

        

    def update_target_q_network(self):
        if self.target_network is not None:
            self.target_network.load_state_dict(self.critic.state_dict())
    
    def update_lr(self, count):
        sys.exit("Learning rate is to be updated")

    
    def update_noise_rate(self, count):
        sys.exit("UPDATE NOISE RATE")

    def get_noise_rate(self):
        return {"actor": self.optimizer_actor.noise_rate,
                "critic": self.optimizer_critic}
    
    def select_action(self, state, variance=0, with_grad=False):
        if torch.is_tensor(state):
            state_tensor = state.to(device)
        else:
            state_tensor = torch.tensor(state, device=device)
        if with_grad:
            action = self.actor(state_tensor)
            if variance == 0:
                return action
            return torch.normal(action, variance)
        with torch.no_grad():
            if variance==0:
                return self.actor(state_tensor)
            return torch.normal(self.actor(state_tensor), variance)


    def get_q_value(self, state, action, target_network=False):
        if target_network:
            return self.target_network(torch.cat((state,action), dim=1))
        return self.critic(torch.cat((state, action), dim=1))


    
    def train(self, replay_buffer, batch_size=128):

        if replay_buffer.buffer_size() < batch_size:
            return

        

        
        # Sample a batch of experiences from the replay buffer
        states, actions, rewards, next_states = replay_buffer.sample(batch_size)

        # Convert to tensors
        states_tensor = torch.tensor(states,dtype=torch.float,device=device)
        actions_tensor = torch.tensor(actions,dtype=torch.float,device=device).view(-1, self.config["output"])
        rewards_tensor = torch.tensor(rewards,dtype=torch.float,device=device).view(-1, 1)
        next_states_tensor = torch.tensor(next_states,dtype=torch.float,device=device)

        # Q-values for the next states (target Q-network)

        with torch.no_grad():
            next_q_values = self.get_q_value(next_states_tensor, self.select_action(next_states_tensor), self.config["target_network"])

        target_q_values = rewards_tensor + self.config["discount"]*next_q_values


        self.optimizer_critic.zero_grad()
        q_values = self.get_q_value(states_tensor, actions_tensor)

        
        loss = self.loss_function(target_q_values, q_values)
        loss.backward()
        self.optimizer_critic.step()
        

        self.optimizer_actor.zero_grad()
        actor_loss = -self.get_q_value(states_tensor, self.select_action(states_tensor, with_grad=True)).mean()
        actor_loss.backward()
        self.optimizer_actor.step()
        


if __name__ == "__main__":

    
    # Need to pass a config file. 
    # This is done to have custom optimizers
    def adam_wrapper(parameters, lr, config):
        return optim.Adam(parameters, lr=lr)

    conf = { "input": 2,
            "output": 1,
            "q_layers": [256,256],
            "layers": [256,256],
            "target_network": False,
            "actor_lr": 0.1,
            "critic_lr": 0.01,
            "discount": 0.99,
            "optimizer": adam_wrapper}

    print(CriticNetwork(conf).network)
    critic = CriticNetwork(conf)
    actor = Actor(conf)
    ddpg = DDPG(conf)
 

    import gymnasium as gym
    from ReplayBuffer import ReplayBuffer

    env = gym.make("MountainCarContinuous-v0")

    state,_ = env.reset()
    print(state)
    action = ddpg.select_action(state)
    print(action)

    buffer = ReplayBuffer(1000)
    for _i in range(100):
        next_state, reward, terminated, truncated, info = env.step(action.detach().numpy())
        
        print("Reward: ", reward)

        buffer.add([state, action, reward, next_state])
    
        for _i in range(10):
            ddpg.train(buffer, 100)
