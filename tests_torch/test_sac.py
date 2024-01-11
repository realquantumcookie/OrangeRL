from orangerl_torch.actor_critic.sac import SACLearnerAgent
from orangerl_torch.input_mappers.mlp_mapper import MLPInputMapper
from orangerl_torch.critic_mappers.direct_mapper import NNDirectCriticMapper
from orangerl_torch.action_mappers.tanh_bound import NNAgentTanhActionMapper
from orangerl_torch import *
from orangerl import *
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import pytest
from typing import List, Callable, Optional, Union

test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_sac_agent(
    env: gym.Env,
    hidden_dims : List[int],
    activations : Optional[Callable[[int, int], nn.Module]] = lambda x: nn.ReLU(),
    last_layer_activation : Optional[Callable[[int, int], nn.Module]] = None,
    use_layer_norm : bool = False,
    dropout_rate : Optional[float] = None,
    decay_factor : float = 0.99,
    device : Optional[Union[torch.device, str]] = None
) -> SACLearnerAgent:
    input_dim = np.prod(env.observation_space.shape)
    output_dim = np.prod(env.action_space.shape)
    actor_net = MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        activations=activations,
        last_layer_activation=last_layer_activation,
        use_layer_norm=use_layer_norm,
        dropout_rate=dropout_rate
    )
    critic_net = MLP(
        input_dim=input_dim + output_dim,
        output_dim=1,
        hidden_dims=hidden_dims,
        activations=activations,
        last_layer_activation=last_layer_activation,
        use_layer_norm=use_layer_norm,
        dropout_rate=dropout_rate
    )
    input_mapper = MLPInputMapper()
    actor = NNAgentActorImpl(
        actor_input_mapper=input_mapper,
        actor_network=actor_net,
        action_mapper=NNAgentTanhActionMapper(env.action_space),
        is_sequence_model=False,
        empty_state=None
    )
    critic = NNAgentCriticImpl(
        critic_input_mapper=input_mapper,
        critic_network=critic_net,
        critic_mapper=NNDirectCriticMapper(),
        is_sequence_model=False,
        empty_state=None,
        is_discrete=False
    )
    sac_agent = SACLearnerAgent(
        actor=actor,
        actor_optimizer=torch.optim.Adam(actor_net.parameters(), lr=1e-3),
        actor_lr_scheduler=None,
        critic=critic,
        critic_optimizer=torch.optim.Adam(critic_net.parameters(), lr=1e-3),
        critic_lr_scheduler=None,
        replay_buffer=NNReplayBuffer(
            storage=LazyTensorStorage(max_size=10_000_000, device="cpu")
        ),
        target_entropy=-output_dim/2,
        utd_ratio=1,
        actor_delay=1,
        decay_factor=decay_factor
    )
    if device is not None:
        sac_agent.to(device)
    return sac_agent

def evaluate_performance(
    env: gym.Env,
    agent: SACLearnerAgent,
    steps : int
) -> float:
    observation, info = env.reset()
    for i in range(steps):
        action = agent.get_action(observation, state = None).action.cpu().numpy()
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        transition = EnvironmentStep(
            observation=observation,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info = info
        )
        agent.observe_transitions(transition)
        observation = next_observation
        if done:
            observation, info = env.reset()
    