# OrangeRL

OrangeRL is my personal learning agent library with a focus on Deep Reinforcement Learning. 
It is designed to be simple, modular and extremely extensible. Most the agents written by me in this library will support sequence models.

## Installation

```bash
git clone https://github.com/realquantumcookie/OrangeRL
cd OrangeRL
pip install -e .
```

## Usage

```python
import orangerl # Import the library
import orangerl.nn as nn # Import general neural network modules
from nn.imitation_learning.bc_agent import BehavioralCloningAgent # Import a specific agent

import gymnasium as gym
import torch


if __name__ == "__main__":
    env = gym.make("CartPole-v1") # Create an environment
    # Create an agent
    agent = BehavioralCloningAgent(
        nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1) # Output a probability distribution over actions
        ), # BC Network
        torch.optim.Adam(lr=1e-5), # Optimizer
        # TODO: Add a action mapper
    )

    
    
```