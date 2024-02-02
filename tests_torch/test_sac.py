from orangerl_torch.actor_critic.sac import SACLearnerAgent
from orangerl_torch.network_util import MLP, RNNMLP
from orangerl_torch.network_adaptors.mlp_mapper import MLPNetworkAdaptor
from orangerl_torch.network_adaptors.direct_mapper import DirectNetworkAdaptor
from orangerl_torch.critic_mappers.direct_mapper import NNDirectCriticMapper
from orangerl_torch.action_mappers.tanh_bound import NNAgentTanhActionMapper
from orangerl_torch import *
from orangerl import *
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import pytest
from typing import List, Callable, Optional, Union, Any
import tqdm
from eval_utils import eval_agent, evaluate_training_performance
import tests_config

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
    network_adaptor = MLPNetworkAdaptor()
    actor = NNAgentActorImpl(
        actor_network=actor_net,
        actor_network_adaptor=network_adaptor,
        action_mapper=NNAgentTanhActionMapper(
            action_min=torch.tensor(env.action_space.low, dtype=torch.float32),
            action_max=torch.tensor(env.action_space.high, dtype=torch.float32)
        ),
        is_sequence_model=False,
        empty_state=None
    )
    critic = NNAgentCriticImpl(
        critic_network=critic_net,
        critic_network_adaptor=network_adaptor,
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

def initialize_sac_rnn_agent(
    env: gym.Env,
    rnn_hidden_dim : int,
    rnn_layers : int,
    mlp_hidden_dims : List[int],
    dropout_rate_rnn : Optional[float] = None,
    activations_mlp : Optional[Callable[[int, int], nn.Module]] = lambda prev_dim, next_dim: nn.ReLU(),
    last_layer_activation : Optional[Callable[[int, int], nn.Module]] = None,
    use_layer_norm_mlp : bool = False,
    dropout_rate_mlp : Optional[float] = None,
    decay_factor : float = 0.99,
    device : Optional[Union[torch.device, str]] = None
) -> SACLearnerAgent:
    input_dim = np.prod(env.observation_space.shape)
    output_dim = np.prod(env.action_space.shape)
    actor_net = RNNMLP(
        input_dim=input_dim,
        rnn_hidden_dim=rnn_hidden_dim,
        rnn_layers=rnn_layers,
        output_dim=output_dim * 2,
        mlp_hidden_dims=mlp_hidden_dims,
        dropout_rate_rnn=dropout_rate_rnn,
        activations_mlp=activations_mlp,
        last_layer_activation=last_layer_activation,
        use_layer_norm_mlp=use_layer_norm_mlp,
        dropout_rate_mlp=dropout_rate_mlp
    )
    critic_net = RNNMLP(
        input_dim=input_dim + output_dim,
        rnn_hidden_dim=rnn_hidden_dim,
        rnn_layers=rnn_layers,
        output_dim=1,
        mlp_hidden_dims=mlp_hidden_dims,
        dropout_rate_rnn=dropout_rate_rnn,
        activations_mlp=activations_mlp,
        last_layer_activation=last_layer_activation,
        use_layer_norm_mlp=use_layer_norm_mlp,
        dropout_rate_mlp=dropout_rate_mlp
    )
    network_adaptor = DirectNetworkAdaptor()
    actor = NNAgentActorImpl(
        actor_network=actor_net,
        actor_network_adaptor=network_adaptor,
        action_mapper=NNAgentTanhActionMapper(
            action_min=torch.tensor(env.action_space.low, dtype=torch.float32),
            action_max=torch.tensor(env.action_space.high, dtype=torch.float32)
        ),
        is_sequence_model=True,
        empty_state=actor_net.empty_state
    )
    critic = NNAgentCriticImpl(
        critic_network=critic_net,
        critic_network_adaptor=network_adaptor,
        critic_mapper=NNDirectCriticMapper(),
        is_sequence_model=True,
        empty_state=critic_net.empty_state,
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
            storage=LazyTensorStorage(max_size=1_000_000, device="cpu"),
            sampler=UniformSequencedTransitionSampler(max_sequence_length=10)
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
        eval_episodes=10,
        enable_wandb=tests_config.ENABLE_WANDB,
        wandb_run_name="test_plain_sac_ant"
    )
    assert final_performance > 1000.0

class TransformObservation(gym.ObservationWrapper):
    def __init__(
        self, 
        env : gym.Env, 
        new_obs_space : gym.Space,
        transform_fn : Callable[[Any], Any]
    ):
        super().__init__(env)
        self.transform_fn = transform_fn
        self._observation_space = new_obs_space
    
    def observation(self, obs):
        return self.transform_fn(obs)

def make_pendulum_novel_env():
    env = gym.make("Pendulum-v1")
    env = TransformObservation(
        env,
        gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        lambda obs: obs[:2]
    )
    return env

def test_pendulum_novel_env():
    env = make_pendulum_novel_env()
    agent = initialize_sac_rnn_agent(
        env,
        rnn_hidden_dim=128,
        rnn_layers=2,
        mlp_hidden_dims=[64, 64],
        use_layer_norm_mlp=False,
        dropout_rate_mlp=0.0,
        device=test_device
    )
    final_performance = evaluate_training_performance(
        env,
        agent,
        steps=100_000,
        warmup_steps=1000,
        eval_episodes=10,
        enable_wandb=tests_config.ENABLE_WANDB,
        wandb_run_name="test_plain_sac_pendulum_novel"
    )
    assert final_performance > -300.0

def test_pendulum_novel_env_compiled():
    env = make_pendulum_novel_env()
    agent = initialize_sac_rnn_agent(
        env,
        rnn_hidden_dim=128,
        rnn_layers=2,
        mlp_hidden_dims=[64, 64],
        use_layer_norm_mlp=False,
        dropout_rate_mlp=0.0,
        device=test_device
    )
    agent.actor.compile()
    agent.critic.compile()
    agent.target_critic.compile()
    final_performance = evaluate_training_performance(
        env,
        agent,
        steps=100_000,
        warmup_steps=1000,
        eval_episodes=10,
        enable_wandb=tests_config.ENABLE_WANDB,
        wandb_run_name="test_plain_sac_pendulum_novel_compiled"
    )
    assert final_performance > -300.0