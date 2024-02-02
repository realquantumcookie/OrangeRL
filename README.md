# OrangeRL

Procedual modular reinforcement learning library, designed to support multiple backends while having the same core agent API.
Most the agents written by me in this library will support sequence models.

## Installation

```bash
git clone https://github.com/realquantumcookie/OrangeRL
cd OrangeRL
pip install -r requirements.txt
pip install -e .
```

## Usage

```python
import orangerl # Import the library
from orangerl_torch import * # Import the torch backend
from orangerl_torch.actor_critic.sac import SACLearnerAgent # Import the SAC agent

import gymnasium as gym
import torch
import torch.nn as nn
import tqdm

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

if __name__ == "__main__":
    env = gym.make("CartPole-v1") # Create an environment
    # Create an agent
    agent = initialize_sac_agent(
        env=env,
        hidden_dims=[64, 64],
        decay_factor=0.99,
        device="cuda"
    )
    # Train the agent
    agent.current_stage = AgentStage.ONLINE
    observation, info = env.reset()
    agent_last_state = None
    done = False

    for i in tqdm.trange(100_000):
        agent_output = agent.get_action(observation, state = agent_last_state)
        action = agent_output.action.detach().cpu().numpy()
        agent_last_state = agent_output.state.detach() if agent_output.state is not None else None
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        transition = orangerl.EnvironmentStep(
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
            update_info = agent.update()
        
        observation = next_observation
        if done:
            observation, info = env.reset()
            done = False
            agent_last_state = None
    
    
```