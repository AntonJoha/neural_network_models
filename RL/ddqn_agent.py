import torch

from .dqn_agent import DQNAgent, device


class DoubleDQNAgent(DQNAgent):
    def replay(self, replay_buffer, batch_size=128, target_network=True):
        if replay_buffer.buffer_size() < batch_size:
            return

        states, actions, rewards, next_states = replay_buffer.sample(batch_size)

        states_tensor = torch.tensor(states, dtype=torch.float, device=device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=device).view(
            -1, 1
        )
        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=device).view(
            -1, 1
        )
        next_states_tensor = torch.tensor(next_states, dtype=torch.float, device=device)

        with torch.no_grad():
            next_actions = self.q_network(next_states_tensor).argmax(
                dim=1, keepdim=True
            )
            if self.config["target_network"] and target_network:
                next_q_values = self.target_network(next_states_tensor).gather(
                    1, next_actions
                )
            else:
                next_q_values = self.q_network(next_states_tensor).gather(
                    1, next_actions
                )

        target_q_values = rewards_tensor + self.config["discount"] * next_q_values

        self.optimizer.zero_grad()
        q_values = self.q_network(states_tensor).gather(1, actions_tensor)
        loss = self.loss_function(q_values, target_q_values)
        loss.item()
        loss.backward()
        self.optimizer.step()
        print(loss)
        return loss
