from orangerl import AgentStage, EnvironmentStep
from orangerl_torch import Tensor_Or_Numpy, Tensor_Or_TensorDict, NNAgent, NNBatch, NNReplayBuffer, NNAgentActor, NNAgentCritic, BatchedNNAgentOutput, BatchedNNCriticOutput
from orangerl_torch.action_mappers.tanh_bound import NNAgentTanhActionMapper
from .base import NNActorCriticAgent
from typing import Any, Iterator, Optional, Union, Iterable, Tuple, Dict, Generic, TypeVar, Callable
import gymnasium as gym
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence
import numpy as np
from tensordict import TensorDictBase, TensorDict
import copy

class SACLearnerAgent(NNActorCriticAgent):
    observe_transition_infos : bool = False

    def __init__(
        self,
        actor: NNAgentActor,
        actor_optimizer: torch.optim.Optimizer,
        actor_lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        critic: NNAgentCritic,
        critic_optimizer: torch.optim.Optimizer,
        critic_lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        replay_buffer : NNReplayBuffer,
        target_entropy : Optional[float] = None,
        temperature_alpha : Optional[float] = None,
        utd_ratio: int = 1,
        actor_delay : int = 1,
        decay_factor: float = 0.99, # This param is used to decay the rewards
        target_critic_tau: float = 0.005, # This param is used to update the target critic network
    ):
        NNActorCriticAgent.__init__(
            self,
            is_sequence_model=self.actor.is_sequence_model,
            empty_state=self.actor.empty_state,
            init_stage=AgentStage.ONLINE,
            decay_factor=decay_factor
        )
        
        assert utd_ratio >= 1, "utd_ratio must be at least 1"
        assert actor_delay >= 1, "actor_delay must be at least 1"
        assert replay_buffer.sample_batch_size % actor_delay == 0, "actor_delay must divide sample_batch_size"
        assert target_critic_tau > 0.0 and target_critic_tau < 1.0, "target_critic_tau must be between 0.0 and 1.0"

        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.actor_lr_scheduler = actor_lr_scheduler

        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.critic_lr_scheduler = critic_lr_scheduler

        self.utd_ratio = utd_ratio
        self.actor_delay = actor_delay
        self.target_critic_tau = target_critic_tau

        assert target_entropy is None or temperature_alpha is None, "Only one of target_entropy and temperature_alpha can be specified"
        assert not (target_entropy is None and temperature_alpha is None), "Either target_entropy or temperature_alpha must be specified"
        if temperature_alpha is not None:
            self.temperature_alpha = nn.Parameter(torch.tensor(temperature_alpha), requires_grad=True)
        else:
            self.temperature_alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.target_entropy = target_entropy

        # Create the target critic
        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic.eval()
        self.target_critic.requires_grad_(False)
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(param.data)

    @property
    def has_replay_buffer(self) -> bool:
        return True

    def seed(
        self, 
        seed : Optional[int] = None
    ) -> None:
        pass
    
    @property
    def is_sequence_model(self) -> bool:
        return self.actor.is_sequence_model

    def forward(
        self,
        obs_batch: Tensor_Or_TensorDict,
        masks: Optional[torch.Tensor] = None,
        state: Optional[Tensor_Or_TensorDict] = None,
        is_update = False,
        **kwargs: Any,
    ) -> BatchedNNAgentOutput:
        return self.actor.forward(
            obs_batch,
            masks,
            state,
            is_update,
            **kwargs
        )

    def _observe_transitions(
        self,
        transition : NNBatch
    ) -> None:
        self.replay_buffer.extend(transition)

    def update(self, *args, **kwargs) -> Dict[str, Any]:
        device = next(self.parameters()).device
        for _ in range(self.utd_ratio):
            batch = self.replay_buffer.sample(device)
            minibatch_size = batch.size(0) // self.actor_delay
            
            for i in range(self.actor_delay):
                minibatch = batch[i * minibatch_size : (i + 1) * minibatch_size]
                critic_infos = self.update_critic(minibatch)
            actor_infos = self.update_actor(batch)
        
        return {
            **critic_infos,
            **actor_infos
        }
    
    def update_critic(self, minibatch : NNBatch, **kwargs) -> Dict[str, Any]:
        critic_output = self.critic.forward(
            obs_batch=minibatch.observations,
            act_batch=minibatch.actions,
            masks=minibatch.masks,
            state=None,
            is_update=True
        )
        actor_output = self.actor.forward(
            obs_batch=minibatch.next_observations,
            masks=minibatch.masks,
            state=None,
            is_update=True
        )
        next_critic_output = self.target_critic.forward(
            obs_batch=minibatch.next_observations,
            act_batch=actor_output.actions,
            masks=minibatch.masks,
            state=None,
            is_update=True
        )
        if self.is_sequence_model:
            fetch_idx = torch.sum(minibatch.masks.to(torch.bool), dim=-1) - 1
            gather_mask = fetch_idx.unsqueeze(1)
            next_q_values : torch.Tensor = torch.gather(
                next_critic_output,
                dim=1,
                index=gather_mask
            ).squeeze(1)
            current_q_values : torch.Tensor = torch.gather(
                critic_output,
                dim=1,
                index=gather_mask
            ).squeeze(1)
            current_rewards : torch.Tensor = torch.gather(
                minibatch.rewards,
                dim=1,
                index=gather_mask
            ).squeeze(1)
            
        else:
            current_q_values = critic_output.critic_estimates
            next_q_values = next_critic_output.critic_estimates
            current_rewards = minibatch.rewards
        target_q_values = current_rewards + self.decay_factor * next_q_values
        critic_loss = nn.functional.mse_loss(current_q_values, target_q_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        if self.critic_lr_scheduler is not None:
            self.critic_lr_scheduler.step()
        
        self._sync_target_critic()

        return {
            "critic_loss": critic_loss.item(),
            "q": current_q_values.mean().item()
        }
    
    def update_actor(self, minibatch : NNBatch, **kwargs) -> Dict[str, Any]:
        actor_output = self.actor.forward(
            obs_batch=minibatch.observations,
            masks=minibatch.masks,
            state=None,
            is_update=True
        )
        critic_output = self.critic.forward(
            obs_batch=minibatch.observations,
            act_batch=actor_output.actions,
            masks=minibatch.masks,
            state=None,
            is_update=True
        )
        if self.is_sequence_model:
            fetch_idx = torch.sum(minibatch.masks.to(torch.bool), dim=-1) - 1
            gather_mask = fetch_idx.unsqueeze(1)
            q_values = torch.gather(
                critic_output.critic_estimates,
                dim=1,
                index=gather_mask
            ).squeeze(1)
            log_probs = torch.gather(
                actor_output.log_probs,
                dim=1,
                index=gather_mask
            )
        else:
            q_values = critic_output.critic_estimates
            log_probs = actor_output.log_probs
        actor_loss = (self.temperature_alpha.detach() * log_probs - q_values).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        if self.actor_lr_scheduler is not None:
            self.actor_lr_scheduler.step()
        
        # Update Temperature Alpha
        if self.target_entropy is not None:
            # Negative Log Probability is the actor entropy
            temperature_alpha_loss = (self.temperature_alpha * (- log_probs - self.target_entropy).detach()).mean()
            self.actor_optimizer.zero_grad()
            temperature_alpha_loss.backward()
            self.actor_optimizer.step()
            if self.actor_lr_scheduler is not None:
                self.actor_lr_scheduler.step()

    def _sync_target_critic(self) -> None:
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                (1.0 - self.target_critic_tau) * target_param.data + self.target_critic_tau * param.data,
                non_blocking=True
            )
    

