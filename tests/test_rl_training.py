import random
import unittest

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from RL.cddpg_agent import DDPG as CDDPGAgent
from RL.ddpg_agent import DDPG as DDPGAgent
from RL.dqn_agent import DQNAgent
from RL.ReplayBuffer import ReplayBuffer
from RL.sac import DDPG as SACAgent


def adam_wrapper(parameters, lr, config):
    return optim.Adam(parameters, lr=lr)


def clone_parameters(module):
    return [p.detach().clone() for p in module.parameters()]


def parameters_changed(before, module):
    after = list(module.parameters())
    return any(not torch.allclose(b, a.detach()) for b, a in zip(before, after, strict=False))


class TestRLTraining(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    def test_dqn_replay_updates_q_network(self):
        config = {
            "input": 4,
            "output": 3,
            "layers": [16],
            "target_network": True,
            "lr": 1e-2,
            "discount": 0.95,
            "optimizer": adam_wrapper,
        }
        agent = DQNAgent(config)
        replay = ReplayBuffer(256)

        for _ in range(64):
            state = np.random.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)
            action = int(np.random.randint(0, 3))
            reward = float(np.random.uniform(-1.0, 1.0))
            next_state = (state + np.random.normal(0.0, 0.1, size=(4,))).astype(np.float32)
            replay.add([state, action, reward, next_state])

        before = clone_parameters(agent.q_network)
        loss = agent.replay(replay, batch_size=32)
        self.assertIsNotNone(loss)
        self.assertTrue(parameters_changed(before, agent.q_network))

    def test_ddpg_train_updates_actor_and_critic(self):
        config = {
            "input": 3,
            "output": 1,
            "q_layers": [16],
            "layers": [16],
            "target_network": False,
            "actor_lr": 1e-3,
            "critic_lr": 1e-3,
            "discount": 0.95,
            "optimizer": adam_wrapper,
        }
        agent = DDPGAgent(config)
        replay = ReplayBuffer(256)

        for _ in range(64):
            state = np.random.uniform(-1.0, 1.0, size=(3,)).astype(np.float32)
            action = np.random.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
            reward = float(np.random.uniform(-1.0, 1.0))
            next_state = (state + np.random.normal(0.0, 0.1, size=(3,))).astype(np.float32)
            replay.add([state, action, reward, next_state])

        actor_before = clone_parameters(agent.actor)
        critic_before = clone_parameters(agent.critic)
        agent.train(replay, batch_size=32)

        self.assertTrue(parameters_changed(actor_before, agent.actor))
        self.assertTrue(parameters_changed(critic_before, agent.critic))

    def test_cddpg_train_updates_actor_and_critic(self):
        config = {
            "input": 3,
            "output": 1,
            "q_layers": [16],
            "layers": [16],
            "target_network": False,
            "actor_lr": 1e-3,
            "critic_lr": 1e-3,
            "discount": 0.95,
            "optimizer": adam_wrapper,
        }
        agent = CDDPGAgent(config)
        replay = ReplayBuffer(256)

        for _ in range(64):
            state = np.random.uniform(-1.0, 1.0, size=(3,)).astype(np.float32)
            action = np.random.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
            reward = float(np.random.uniform(-1.0, 1.0))
            next_state = (state + np.random.normal(0.0, 0.1, size=(3,))).astype(np.float32)
            replay.add([state, action, reward, next_state])

        actor_before = clone_parameters(agent.actor)
        critic_before = clone_parameters(agent.critic)
        agent.train(replay, batch_size=32)

        self.assertTrue(parameters_changed(actor_before, agent.actor))
        self.assertTrue(parameters_changed(critic_before, agent.critic))

    def test_sac_replay_updates_actor_and_critics(self):
        config = {
            "input": 3,
            "output": 1,
            "q_layers": [16],
            "layers": [16],
            "target_network": False,
            "actor_lr": 1e-3,
            "critic_lr": 1e-3,
            "discount": 0.95,
            "optimizer": adam_wrapper,
        }
        agent = SACAgent(config)
        replay = ReplayBuffer(256)

        for _ in range(64):
            state = np.random.uniform(-1.0, 1.0, size=(3,)).astype(np.float32)
            action = np.random.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
            reward = float(np.random.uniform(-1.0, 1.0))
            next_state = (state + np.random.normal(0.0, 0.1, size=(3,))).astype(np.float32)
            replay.add([state, action, reward, next_state])

        actor_before = clone_parameters(agent.actor)
        critic_1_before = clone_parameters(agent.critic_1)
        critic_2_before = clone_parameters(agent.critic_2)
        loss = agent.replay(replay, batch_size=32)

        self.assertIsNotNone(loss)
        self.assertTrue(parameters_changed(actor_before, agent.actor))
        self.assertTrue(parameters_changed(critic_1_before, agent.critic_1))
        self.assertTrue(parameters_changed(critic_2_before, agent.critic_2))

    def test_dqn_loss_decreases_over_epochs(self):
        # A frozen target network provides fixed bootstrap targets, turning
        # training into a supervised regression task that must converge.
        config = {
            "input": 4,
            "output": 3,
            "layers": [16],
            "target_network": True,
            "lr": 1e-2,
            "discount": 0.95,
            "optimizer": adam_wrapper,
        }
        agent = DQNAgent(config)
        replay = ReplayBuffer(32)

        for _ in range(32):
            state = np.random.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)
            action = int(np.random.randint(0, 3))
            reward = float(np.random.uniform(-1.0, 1.0))
            next_state = (state + np.random.normal(0.0, 0.1, size=(4,))).astype(np.float32)
            replay.add([state, action, reward, next_state])

        early_losses = [agent.replay(replay, batch_size=32).item() for _ in range(10)]
        for _ in range(180):
            agent.replay(replay, batch_size=32)
        late_losses = [agent.replay(replay, batch_size=32).item() for _ in range(10)]
        early_avg = sum(early_losses) / 10
        late_avg = sum(late_losses) / 10
        self.assertLess(late_avg, early_avg)

    def _build_ddpg_replay(self, state_dim, action_dim, n=32):
        states = np.random.uniform(-1.0, 1.0, size=(n, state_dim)).astype(np.float32)
        actions = np.random.uniform(-1.0, 1.0, size=(n, action_dim)).astype(np.float32)
        rewards = np.random.uniform(-1.0, 1.0, size=(n,)).astype(np.float32)
        next_states = (states + np.random.normal(0.0, 0.1, size=(n, state_dim))).astype(np.float32)
        replay = ReplayBuffer(n)
        for i in range(n):
            replay.add([states[i], actions[i], float(rewards[i]), next_states[i]])
        states_t = torch.tensor(states, dtype=torch.float)
        actions_t = torch.tensor(actions, dtype=torch.float).view(-1, action_dim)
        rewards_t = torch.tensor(rewards, dtype=torch.float).view(-1, 1)
        next_states_t = torch.tensor(next_states, dtype=torch.float)
        return replay, states_t, actions_t, rewards_t, next_states_t

    def _ddpg_critic_loss(self, agent, states_t, actions_t, fixed_targets):
        with torch.no_grad():
            q = agent.get_q_value(states_t, actions_t)
        return F.mse_loss(q, fixed_targets).item()

    def test_ddpg_loss_decreases_over_epochs(self):
        # actor_lr=0 freezes the actor; target_network=True freezes the bootstrap
        # critic.  Together they give a fixed supervised regression task for the
        # critic that is guaranteed to converge.
        config = {
            "input": 3,
            "output": 1,
            "q_layers": [16],
            "layers": [16],
            "target_network": True,
            "actor_lr": 0.0,
            "critic_lr": 1e-2,
            "discount": 0.95,
            "optimizer": adam_wrapper,
        }
        agent = DDPGAgent(config)
        replay, states_t, actions_t, rewards_t, next_states_t = self._build_ddpg_replay(3, 1)

        with torch.no_grad():
            next_actions = agent.select_action(next_states_t)
            next_q = agent.get_q_value(next_states_t, next_actions, target_network=True)
            fixed_targets = (rewards_t + config["discount"] * next_q).clone()

        initial_loss = self._ddpg_critic_loss(agent, states_t, actions_t, fixed_targets)
        for _ in range(200):
            agent.train(replay, batch_size=32)
        final_loss = self._ddpg_critic_loss(agent, states_t, actions_t, fixed_targets)
        self.assertLess(final_loss, initial_loss)

    def test_cddpg_loss_decreases_over_epochs(self):
        config = {
            "input": 3,
            "output": 1,
            "q_layers": [16],
            "layers": [16],
            "target_network": True,
            "actor_lr": 0.0,
            "critic_lr": 1e-2,
            "discount": 0.95,
            "optimizer": adam_wrapper,
        }
        agent = CDDPGAgent(config)
        replay, states_t, actions_t, rewards_t, next_states_t = self._build_ddpg_replay(3, 1)

        with torch.no_grad():
            next_actions = agent.select_action(next_states_t)
            next_q = agent.get_q_value(next_states_t, next_actions, target_network=True)
            fixed_targets = (rewards_t + config["discount"] * next_q).clone()

        initial_loss = self._ddpg_critic_loss(agent, states_t, actions_t, fixed_targets)
        for _ in range(200):
            agent.train(replay, batch_size=32)
        final_loss = self._ddpg_critic_loss(agent, states_t, actions_t, fixed_targets)
        self.assertLess(final_loss, initial_loss)

    def test_sac_loss_decreases_over_epochs(self):
        # actor_lr=0 freezes the actor; target_network=True freezes the bootstrap
        # critics.  Together they give a fixed supervised regression task for both
        # critics that is guaranteed to converge.
        config = {
            "input": 3,
            "output": 1,
            "q_layers": [16],
            "layers": [16],
            "target_network": True,
            "actor_lr": 0.0,
            "critic_lr": 1e-2,
            "discount": 0.95,
            "optimizer": adam_wrapper,
        }
        agent = SACAgent(config)
        replay, states_t, actions_t, rewards_t, next_states_t = self._build_ddpg_replay(3, 1)

        with torch.no_grad():
            next_actions = agent.select_action(next_states_t)
            next_q1 = agent.target_network_1(torch.cat((next_states_t, next_actions), dim=1))
            next_q2 = agent.target_network_2(torch.cat((next_states_t, next_actions), dim=1))
            fixed_targets = (rewards_t + config["discount"] * torch.min(next_q1, next_q2)).clone()

        def critic1_loss():
            with torch.no_grad():
                q1 = agent.critic_1(torch.cat((states_t, actions_t), dim=1))
            return F.mse_loss(q1, fixed_targets).item()

        initial_loss = critic1_loss()
        for _ in range(200):
            agent.replay(replay, batch_size=32)
        final_loss = critic1_loss()
        self.assertLess(final_loss, initial_loss)


if __name__ == "__main__":
    unittest.main()
