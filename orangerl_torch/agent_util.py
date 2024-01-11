"""
   Copyright 2023 Yunhao Cao

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from orangerl import AgentStage, EnvironmentStep
from orangerl_torch.agent import Tensor_Or_TensorDict
from .agent import Tensor_Or_TensorDict, NNAgent, BatchedNNAgentOutput
from .data import NNBatch
from typing import TypeVar, Generic, Union, Optional, Iterable, Any, Callable, Dict, List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict import tensorclass, TensorDictBase, TensorDict
# import gymnasium as gym


@dataclass
class NNAgentNetworkOutput:
    """
    NNAgentNetworkOutput is a dataclass that represents the output of a neural network.
    """
    output : Tensor_Or_TensorDict # (batch_size, *action_shape) if not is_seq, (batch_size, sequence_length, *action_shape) if is_seq
    masks: Optional[torch.Tensor] = None # (batch_size, ) if not is_seq, (batch_size, sequence_length) if is_seq
    state : Optional[Tensor_Or_TensorDict] = None # (batch_size, *state_shape) if is_seq
    is_seq : bool = False

# _ActionSpaceMapperT = TypeVar("_ActionSpaceMapperT", bound=gym.Space)
class NNAgentActionMapper(ABC, nn.Module):
    """
    NNAgentActionMapper is a module that maps the output of a neural network to an action.
    """

    @abstractmethod
    def forward(
        self, 
        nn_output : NNAgentNetworkOutput, 
        is_update : bool = False,
        stage : AgentStage = AgentStage.ONLINE
    ) -> BatchedNNAgentOutput:
        """
        Should return a BatchedNNAgentOutput object, which represents the actions taken by the agent.
        """
        ...

    @abstractmethod
    def log_prob(
        self,
        nn_output : NNAgentNetworkOutput,
        actions : torch.Tensor,
        is_update : bool = False,
        stage : AgentStage = AgentStage.ONLINE
    ) -> BatchedNNAgentOutput:
        ...

class NNAgentInputMapper(ABC, nn.Module):

    @abstractmethod
    def forward(
        self, 
        obs_batch: Tensor_Or_TensorDict,
        act_batch: Optional[Tensor_Or_TensorDict], # Only used when mapping critic inputs
        masks: Optional[torch.Tensor] = None,
        state: Optional[Tensor_Or_TensorDict] = None,
        is_seq = False, # If True, then obs_batch is shaped (batch_size, sequence_length, *observation_shape)
        is_update = False,
        stage : AgentStage = AgentStage.ONLINE,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        ...

    @abstractmethod
    def map_net_output(
        self,
        output : Any,
        masks: Optional[torch.Tensor] = None,
        state: Optional[Tensor_Or_TensorDict] = None,
        is_seq = False,
        is_update = False,
        stage : AgentStage = AgentStage.ONLINE,
    ) -> NNAgentNetworkOutput:
        ...

NNAgentActor = NNAgent

class NNAgentActorImpl(NNAgent):
    """
    An convenient actor agent implementation
    """

    def __init__(
        self,
        actor_input_mapper: NNAgentInputMapper,
        actor_network : nn.Module,
        action_mapper : NNAgentActionMapper,
        is_sequence_model : bool,
        empty_state : Optional[Tensor_Or_TensorDict],
        init_stage : AgentStage = AgentStage.ONLINE,
        decay_factor : float = 0.99,
    ):
        NNAgent.__init__(self, is_sequence_model, empty_state, init_stage, decay_factor)
        self.actor_input_mapper = actor_input_mapper
        self.actor_network = actor_network
        self.action_mapper = action_mapper

    def forward(
        self,
        obs_batch: Tensor_Or_TensorDict,
        masks: Optional[torch.Tensor] = None,
        state: Optional[Tensor_Or_TensorDict] = None,
        is_update = False,
        **kwargs: Any,
    ) -> BatchedNNAgentOutput:
        nn_input_args, nn_input_kwargs = self.actor_input_mapper.forward(
            obs_batch,
            None,
            masks,
            state,
            self.is_sequence_model,
            is_update,
            self.current_stage,
        )
        nn_output = self.actor_input_mapper.map_net_output(
            self.actor_network.forward(*nn_input_args, **nn_input_kwargs),
            masks=masks,
            state=state,
            is_seq=self.is_sequence_model,
            is_update=is_update,
            stage=self.current_stage
        )
        mapped_action = self.action_mapper.forward(
            nn_output,
            is_update,
            self.current_stage,
        )
        return mapped_action

    def _observe_transitions(
        self, 
        transition : NNBatch
    ) -> None:
        pass

    def update(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Actor should be updated with the policy")

@dataclass
class BatchedNNCriticOutput:
    critic_estimates : torch.Tensor # (B,[S],[A]) where S and A are only present if is_seq and is_discrete, respectively
    distributions : Optional[torch.distributions.Distribution] # (B, [S], [A])
    log_stds : Optional[torch.Tensor] # (B, [S], [A])
    final_states: Optional[Tensor_Or_TensorDict] = None # (B, S) if is_seq, otherwise None
    masks: Optional[torch.Tensor] = None # (B, [S])
    is_seq : bool = False
    is_discrete : bool = False

class NNAgentCritic(ABC, nn.Module):
    is_sequence_model : bool
    empty_state : Optional[Tensor_Or_TensorDict]
    is_discrete : bool

    @abstractmethod
    def forward(
        self,
        obs_batch: Tensor_Or_TensorDict,
        act_batch: Optional[Tensor_Or_TensorDict],
        masks: Optional[torch.Tensor] = None,
        state: Optional[Tensor_Or_TensorDict] = None,
        is_update = False,
        stage : AgentStage = AgentStage.ONLINE,
        **kwargs: Any,
    ) -> BatchedNNCriticOutput:
        ...

class NNAgentCriticMapper(ABC, nn.Module):
    @abstractmethod
    def forward(
        self, 
        nn_output : NNAgentNetworkOutput, 
        is_update : bool = False,
        is_discrete : bool = False,
        stage : AgentStage = AgentStage.ONLINE
    ) -> BatchedNNCriticOutput:
        ...

class NNAgentCriticImpl(NNAgentCritic):
    def __init__(
        self,
        critic_input_mapper: NNAgentInputMapper,
        critic_network : nn.Module,
        critic_mapper: NNAgentCriticMapper,
        is_sequence_model : bool,
        empty_state : Optional[Tensor_Or_TensorDict],
        is_discrete : bool = False,
    ):
        NNAgentCritic.__init__(self)
        self.critic_input_mapper = critic_input_mapper
        self.critic_network = critic_network
        self.critic_mapper = critic_mapper
        self._is_sequence_model = is_sequence_model
        self.empty_state = empty_state
        self._is_discrete = is_discrete

    @property
    def is_decrete(self) -> bool:
        return self._is_discrete

    @property
    def is_sequence_model(self) -> bool:
        return self._is_sequence_model

    def forward(
        self,
        obs_batch: Tensor_Or_TensorDict,
        act_batch: Optional[Tensor_Or_TensorDict],
        masks: Optional[torch.Tensor] = None,
        state: Optional[Tensor_Or_TensorDict] = None,
        is_update = False,
        stage : AgentStage = AgentStage.ONLINE,
        **kwargs: Any,
    ) -> BatchedNNCriticOutput:
        nn_input_args, nn_input_kwargs = self.critic_input_mapper.forward(
            obs_batch,
            act_batch,
            masks,
            state,
            is_seq=self.is_sequence_model,
            is_update=is_update,
            stage=stage
        )
        nn_output = self.critic_input_mapper.map_net_output(
            self.critic_network.forward(*nn_input_args, **nn_input_kwargs),
            masks=masks,
            state=state,
            is_seq=self.is_sequence_model,
            is_update=is_update,
            stage=stage
        )
        mapped_output = self.critic_mapper.forward(
            nn_output,
            is_update,
            self.is_discrete,
            stage
        )
        return mapped_output
