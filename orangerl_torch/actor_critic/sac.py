from orangerl import AgentStage, EnvironmentStep
from orangerl_torch.data import Tensor_Or_Numpy, NNBatch
from orangerl_torch.replay_buffer import NNReplayBuffer
from orangerl_torch.agent import Tensor_Or_TensorDict, NNAgent, NNAgentWithCritic, NNAgentCriticEstimateOutput, NNAgentNetworkOutput, NNAgentActorNet, NNAgentCriticNet
from orangerl_torch.action_mappers.tanh_bound import NNAgentTanhActionMapper
from typing import Any, Iterator, Optional, Union, Iterable, Tuple, Dict, Generic, TypeVar, Callable
import gymnasium as gym
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence
import numpy as np
from tensordict import TensorDictBase, TensorDict
import copy

class SACLearnerAgent(NNAgent[gym.Space, gym.spaces.Box], NNAgentWithCritic):
    def __init__(
        self,
        actor_net: NNAgentActorNet,
        actor_net_optimizer: torch.optim.Optimizer,
        actor_net_lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        critic_net: NNAgentCriticNet,
        critic_net_optimizer: torch.optim.Optimizer,
        critic_net_lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        observation_space: gym.Space,
        action_space: gym.spaces.Box,
        replay_buffer : NNReplayBuffer,
        target_entropy : Optional[float] = None,
        critic_to_actor_ratio: int = 1,
        is_sequence_model: bool = False,
        empty_state_vector: Optional[Tensor_Or_TensorDict] = None,
        decay_factor: float = 0.99,
        target_critic_tau: float = 0.005,
    ):
        super().__init__(is_sequence_model, empty_state_vector, decay_factor)
        
        assert actor_net_optimizer is not None or actor_net_lr_scheduler is None, "If actor_net_optimizer is None, then actor_net_lr_scheduler must be None"
        assert critic_net_optimizer is not None or critic_net_lr_scheduler is None, "If critic_net_optimizer is None, then critic_net_lr_scheduler must be None"
        assert critic_to_actor_ratio >= 1, "critic_to_actor_ratio must be at least 1"
        assert replay_buffer.sample_batch_size % critic_to_actor_ratio == 0, "critic_to_actor_ratio must divide sample_batch_size"
        assert target_critic_tau > 0.0 and target_critic_tau < 1.0, "target_critic_tau must be between 0.0 and 1.0"

        self.actor_net = actor_net
        self.actor_net_optimizer = actor_net_optimizer
        self.actor_net_lr_scheduler = actor_net_lr_scheduler
        self.critic_net = critic_net
        self.target_critic_net = copy.deepcopy(critic_net)
        self.target_critic_net.eval()
        for param in self.target_critic_net.parameters():
            param.requires_grad = False

        self.critic_net_optimizer = critic_net_optimizer
        self.critic_net_lr_scheduler = critic_net_lr_scheduler
        self.action_mapper = NNAgentTanhActionMapper(action_space)
        self.observation_space = observation_space
        self.replay_buffer = replay_buffer
        self.target_entropy = target_entropy if target_entropy is not None else -(np.prod(action_space.shape) / 2.0)
        self.critic_to_actor_ratio = critic_to_actor_ratio
        self.temperature_alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.target_critic_tau = target_critic_tau

    @property
    def action_space(self) -> gym.spaces.Box:
        return self.action_mapper.action_space

    @property
    def has_replay_buffer(self) -> bool:
        return True

    def seed(
        self, 
        seed : Optional[int] = None
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def evaluate_critic(
        self,
        obs_batch: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        state: Optional[Tensor_Or_TensorDict] = None,
        is_seq = False, # If True, then obs_batch is shaped (batch_size, sequence_length, *observation_shape)
        stage: AgentStage = AgentStage.ONLINE,
        **kwargs: Any,
    ) -> NNAgentCriticEstimateOutput:
        assert is_seq == self.is_sequence_model, "is_seq must match the is_sequence_model of the agent"
        if is_seq and masks is not None:
            assert masks.ndim == 2, "masks must be a 2D tensor"
            assert masks.shape[:2] == obs_batch.shape[:2], "masks must have the same batch size and sequence length as obs_batch"
            assert state is None or (
                state.shape[:1] == obs_batch.shape[:1]
            ), "state must have the same batch size as obs_batch"
        elif not is_seq:
            assert masks is None, "masks must be None if is_seq is False"
            assert state is None, "state must be None if is_seq is False"

        ret = self.critic_net(
            obs_batch,
            masks,
            state,
            is_seq,
            stage,
            **kwargs
        )
        assert isinstance(ret.output, torch.Tensor), "The output of the critic network must be a tensor"
        return ret

    def evaluate_target_critic(
        self,
        obs_batch: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        state: Optional[Tensor_Or_TensorDict] = None,
        is_seq = False, # If True, then obs_batch is shaped (batch_size, sequence_length, *observation_shape)
        stage: AgentStage = AgentStage.ONLINE,
        **kwargs: Any,
    ) -> NNAgentCriticEstimateOutput:
        assert is_seq == self.is_sequence_model, "is_seq must match the is_sequence_model of the agent"
        if is_seq and masks is not None:
            assert masks.ndim == 2, "masks must be a 2D tensor"
            assert masks.shape[:2] == obs_batch.shape[:2], "masks must have the same batch size and sequence length as obs_batch"
            assert state is None or (
                state.shape[:1] == obs_batch.shape[:1]
            ), "state must have the same batch size as obs_batch"
        elif not is_seq:
            assert masks is None, "masks must be None if is_seq is False"
            assert state is None, "state must be None if is_seq is False"

        with torch.no_grad():
            ret = self.target_critic_net(
                obs_batch,
                masks,
                state,
                is_seq,
                stage,
                **kwargs
            )
        assert isinstance(ret.output, torch.Tensor), "The output of the critic network must be a tensor"
        return ret


    def forward(
        self,
        obs_batch: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        state: Optional[Tensor_Or_TensorDict] = None,
        is_seq = False, # If True, then obs_batch is shaped (batch_size, sequence_length, *observation_shape)
        stage: AgentStage = AgentStage.ONLINE,
        **kwargs: Any,
    ) -> NNAgentNetworkOutput:
        assert is_seq == self.is_sequence_model, "is_seq must match the is_sequence_model of the agent"
        if is_seq and masks is not None:
            assert masks.ndim == 2, "masks must be a 2D tensor"
            assert masks.shape[:2] == obs_batch.shape[:2], "masks must have the same batch size and sequence length as obs_batch"
            assert state is None or (
                state.shape[:1] == obs_batch.shape[:1]
            ), "state must have the same batch size as obs_batch"
        elif not is_seq:
            assert masks is None, "masks must be None if is_seq is False"
            assert state is None, "state must be None if is_seq is False"

        return self.actor_net(
            obs_batch,
            masks,
            state,
            is_seq,
            stage,
            **kwargs
        )

    def observe_transitions(
        self, 
        transition : Union[Iterable[EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy]], EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy]],
        stage : AgentStage = AgentStage.ONLINE
    ) -> None:
        self.replay_buffer += transition

    def update(self, stage: AgentStage = AgentStage.ONLINE, *args, **kwargs) -> TensorDict:
        device = next(self.actor_net.parameters()).device
        batch = self.replay_buffer.sample(device)
        minibatch_size = batch.size(0) // self.critic_to_actor_ratio
        critic_infos = []
        for i in range(self.critic_to_actor_ratio):
            minibatch = batch[i * minibatch_size : (i + 1) * minibatch_size]
            critic_infos.append(self.update_critic(minibatch, stage))
        critic_infos = torch.mean(torch.stack(critic_infos))
        actor_infos = self.update_actor_and_temperature(batch, stage)
        return TensorDict({
            **critic_infos,
            **actor_infos
        }, batch_size=())
    
    def _sync_target_critic(self) -> None:
        for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(
                self.target_critic_tau * param.data + (1.0 - self.target_critic_tau) * target_param.data,
                non_blocking=True
            )

    def update_critic(self, batch : NNBatch, stage : AgentStage = AgentStage.ONLINE) -> TensorDict:
        output_batch = self.evaluate_critic(
            batch.observations,
            batch.masks,
            None,
            self.is_sequence_model,
            stage
        )

        critic_each_batch = output_batch.output
        batch_end_sequence_idx = None
        if not self.is_sequence_model:
            critic_each_batch = critic_each_batch.reshape(critic_each_batch.size(0))
            terminations_each_batch = batch.terminations.reshape(batch.terminations.size(0))
            next_critics = self.evaluate_target_critic(
                batch.next_observations,
                batch.masks,
                None,
                self.is_sequence_model,
                stage
            )
        else:
            critic_each_batch = critic_each_batch.reshape(critic_each_batch.size(0), critic_each_batch.size(1))
            batch_next_sequence_idx = torch.sum(batch.masks.to(torch.bool), dim=-1)
            batch_end_sequence_idx = batch_next_sequence_idx - 1
            critic_each_batch = torch.gather(
                critic_each_batch, 1, batch_end_sequence_idx.unsqueeze(-1)
            ).squeeze(-1)
            terminations_each_batch = torch.gather(
                batch.terminations, 1, batch_end_sequence_idx.unsqueeze(-1)
            ).squeeze(-1)
            observations_next = torch.gather(
                batch.next_observations, 1, batch_end_sequence_idx.unsqueeze(-1).expand(-1, -1, batch.next_observations.shape[2:])
            )
            next_critics = self.evaluate_target_critic(
                observations_next,
                None,
                output_batch.state,
                self.is_sequence_model,
                stage
            ).output.squeeze(-1)
        target_critic_batch = batch.rewards + (1.0 - terminations_each_batch) * self.decay_factor * next_critics.output
        target_critic_batch = target_critic_batch.detach()
        critic_loss = nn.functional.mse_loss(critic_each_batch, target_critic_batch)
        self.critic_net_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_net_optimizer.step()
        if self.critic_net_lr_scheduler is not None:
            self.critic_net_lr_scheduler.step()
        self._sync_target_critic()
        with torch.no_grad():
            return TensorDict({
                "critic_loss": critic_loss.item(),
                "q": critic_each_batch.mean()
            }, batch_size=())
    
    def update_actor_and_temperature(self, batch : NNBatch, stage : AgentStage = AgentStage.ONLINE) -> TensorDict:
        output_batch = self.forward(
            batch.observations,
            batch.masks,
            None,
            self.is_sequence_model,
            stage
        )
        output_critic = self.evaluate_critic(
            batch.actions,
            batch.masks,
            None,
            self.is_sequence_model,
            stage
        )

        mapped_action = self.action_mapper.forward(
            self,
            output_batch,
            is_update=True,
            stage=stage
        )


