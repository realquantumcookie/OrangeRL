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
        masks: Optional[torch.Tensor] = None,
        state: Optional[Tensor_Or_TensorDict] = None,
        is_seq = False, # If True, then obs_batch is shaped (batch_size, sequence_length, *observation_shape)
        is_update = False,
        stage : AgentStage = AgentStage.ONLINE,
    ) -> Tuple[List[Any], Dict[str, Any]]:
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
        is_seq = False, # If True, then obs_batch is shaped (batch_size, sequence_length, *observation_shape)
        is_update = False,
        **kwargs: Any,
    ) -> BatchedNNAgentOutput:
        nn_input_args, nn_input_kwargs = self.actor_input_mapper.forward(
            obs_batch,
            masks,
            state,
            is_seq,
            is_update,
            self.current_stage,
        )
        nn_output = self.actor_network.forward(*nn_input_args, **nn_input_kwargs)
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
    critic_estimates : torch.Tensor # (batch_size,) if not is_seq, (batch_size, sequence_length) if is_seq
    distributions : Optional[torch.distributions.Distribution] # (batch_size,) if not is_seq, (batch_size, sequence_length) if is_seq
    log_stds : Optional[torch.Tensor] # (batch_size,) if not is_seq, (batch_size, sequence_length) if is_seq
    final_states: Optional[Tensor_Or_TensorDict] = None # (batch_size, *state_shape) if is_seq
    masks: Optional[torch.Tensor] = None # None if not is_seq, (batch_size, sequence_length) if is_seq
    is_seq : bool = False

class NNAgentCritic(ABC, nn.Module):

    @abstractmethod
    def forward(
        self,
        obs_batch: Tensor_Or_TensorDict,
        masks: Optional[torch.Tensor] = None,
        state: Optional[Tensor_Or_TensorDict] = None,
        is_seq = False, # If True, then obs_batch is shaped (batch_size, sequence_length, *observation_shape)
        is_update = False,
        **kwargs: Any,
    ) -> BatchedNNCriticOutput:
        ...

