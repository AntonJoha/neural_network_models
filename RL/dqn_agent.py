import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import Actor as QNetwork
from .networks import device


class DQNAgent:

    def __init__(self, config=None, loss=nn.MSELoss):
        

        if config is None:
            sys.exit("No config")

        self.config = config
        self.lr = self.config["lr"]   

        self.q_network = QNetwork(self.config).to(device)
        self.target_network = QNetwork(self.config).to(device).eval()
        
        self.optimizer = config["optimizer"](self.q_network.parameters(), lr=self.lr, config=config)
        self.loss_function = loss()
        

    def evaluate_mode(self):
        self.q_network.eval()
    
    def train_mode(self):
        self.q_network.train()

    def get_network_weights(self):
        return self.q_network.state_dict()
    
    def set_network_weights(self,weights):
        self.q_network.load_state_dict(weights)


    def get_noise_rate(self):
        return self.optimizer.noise_rate

    def set_noise_rate(self, noise):
        self.noise_rate = noise
        self.optimizer.noise_rate = noise
        print("NOISE UPDATED: ", noise)

    def get_learning_rate(self):
        return self.lr

    def set_learning_rate(self,lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        print("LEARNING RATE UPDATED: ", self.lr)



    def update_target_q_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


    def select_action(self, state, exploration_prob=0):
        if np.random.rand() < exploration_prob:
            return np.random.choice(self.config["output"])  # Explore
        else:
            state_tensor = torch.tensor(state,device=device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            
            return torch.argmax(q_values).item()  # Exploit
 

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
        loss.item()
        loss.backward()
        self.optimizer.step()
        print(loss)
        return loss
 


if __name__ == "__main__":

    
    # Need to pass a config file. 
    # This is done to have custom optimizers
    def adam_wrapper(parameters, lr, config):
        return optim.Adam(parameters, lr=lr)

    conf = { "input": 6,
            "output": 3,
            "layers": [256,256],
            "target_network": True,
            "lr": 0.1,
            "discount": 0.99,
            "optimizer": adam_wrapper}
    print(DQNAgent(conf))

    a = DQNAgent(conf)
    

    import gymnasium as gym
    from ReplayBuffer import ReplayBuffer

    env = gym.make("Acrobot-v1")

    state, _ = env.reset()

    action = a.select_action(state)
    
    buffer = ReplayBuffer(1000)
    
    for _i in range(100):
        next_state, reward, terminated ,truncated, info = env.step(action)
    
        buffer.add([state, action, reward, next_state])

        a.replay(buffer, 1, True)

    print(QNetwork(conf).network)


