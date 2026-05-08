# neural-network-models

A Python library of neural network models for reinforcement learning and generative modelling.

## Packages

### `RL`

Reinforcement learning agents and utilities:

- **`DQNAgent`** – Deep Q-Network agent
- **`DDPG`** – Deep Deterministic Policy Gradient
- **`CDDPG`** – Constrained DDPG variant
- **`SACAgent`** – Soft Actor-Critic (accessible as `from RL.sac import DDPG`)
- **`ReplayBuffer`** – Experience replay buffer

### `gaussian_models`

Generative models:

- **`tDLGM`** – Temporal Deep Latent Gaussian Model

## Installation

```bash
pip install neural-network-models
```

## Usage

```python
from RL.dqn_agent import DQNAgent
from RL.ReplayBuffer import ReplayBuffer
import torch.optim as optim

def adam_wrapper(parameters, lr, config):
    return optim.Adam(parameters, lr=lr)

config = {
    "input": 6,
    "output": 3,
    "layers": [256, 256],
    "target_network": True,
    "lr": 1e-3,
    "discount": 0.99,
    "optimizer": adam_wrapper,
}

agent = DQNAgent(config)
replay = ReplayBuffer(10000)
```

## License

MIT
