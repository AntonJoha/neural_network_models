import random
import unittest

import numpy as np
import torch
import torch.optim as optim

from RL.ReplayBuffer import ReplayBuffer
from RL.cddpg_agent import DDPG as CDDPGAgent
from RL.ddpg_agent import DDPG as DDPGAgent
from RL.dqn_agent import DQNAgent
from RL.sac import DDPG as SACAgent


def adam_wrapper(parameters, lr, config):
    return optim.Adam(parameters, lr=lr)


def clone_parameters(module):
    return [p.detach().clone() for p in module.parameters()]


def parameters_changed(before, module):
    after = list(module.parameters())
    return any(not torch.allclose(b, a.detach()) for b, a in zip(before, after))


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
            action = float(np.random.uniform(-1.0, 1.0))
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
            action = float(np.random.uniform(-1.0, 1.0))
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
            action = float(np.random.uniform(-1.0, 1.0))
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


if __name__ == "__main__":
    unittest.main()
