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
print("Test device: ", test_device)

def initialize_sac_agent(
    env: gym.Env,
    hidden_dims : List[int],
    activations : Optional[Callable[[int, int], nn.Module]] = lambda prev_dim, next_dim: nn.ReLU(),
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
        output_dim=output_dim * 2,
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
        action_mapper=NNAgentTanhActionMapper(
            action_min=torch.tensor(env.action_space.low, dtype=torch.float32),
            action_max=torch.tensor(env.action_space.high, dtype=torch.float32)
        ),
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
        actor_optimizer=torch.optim.Adam(actor_net.parameters(), lr=3e-4),
        actor_lr_scheduler=None,
        critic=critic,
        critic_optimizer=torch.optim.Adam(critic_net.parameters(), lr=3e-4),
        critic_lr_scheduler=None,
        replay_buffer=NNReplayBuffer(
            storage=LazyTensorStorage(max_size=1_000_000, device="cpu")
        ),
        target_entropy=float(-output_dim),
        temperature_alpha_lr=3e-4,
        utd_ratio=1,
        actor_delay=1,
        decay_factor=decay_factor
    )
    if device is not None:
        sac_agent = sac_agent.to(device)
    return sac_agent

def eval_agent(
    eval_env : gym.Env,
    agent : SACLearnerAgent,
    num_episodes : int,
) -> float:
    agent.current_stage = AgentStage.EVAL
    total_reward = 0.0
    for i in range(num_episodes):
        observation, info = eval_env.reset()
        done = False
        while not done:
            action = agent.get_action(observation, state = None).action.detach().cpu().numpy()
            next_observation, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            total_reward += reward
            observation = next_observation
    return total_reward / num_episodes

def evaluate_training_performance(
    env: gym.Env,
    agent: SACLearnerAgent,
    steps : int,
    warmup_steps : int,
    eval_episodes : int,
) -> float:
    agent.current_stage = AgentStage.ONLINE
    observation, info = env.reset()
    for i in range(steps):
        action = agent.get_action(observation, state = None).action.detach().cpu().numpy()
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        transition = EnvironmentStep(
            observation=observation,
            next_observation=next_observation,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info = info
        )
        agent.observe_transitions(transition)
        
        if i > warmup_steps:
            agent.update()
        
        observation = next_observation
        if done:
            observation, info = env.reset()
    final_performance = eval_agent(env, agent, eval_episodes)
    return final_performance

def test_ant_env():
    env = gym.make("Ant-v4")
    agent = initialize_sac_agent(
        env,
        hidden_dims=[256, 256],
        use_layer_norm=False,
        dropout_rate=0.0,
        device=test_device
    )
    final_performance = evaluate_training_performance(
        env,
        agent,
        steps=500_000,
        warmup_steps=5000,
        eval_episodes=10
    )
    assert final_performance > 1000.0
