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

from orangerl import Agent, AgentOutput, AgentStage, AgentActionType, TransitionBatch, EnvironmentStep
from .data import NNBatch, Tensor_Or_Numpy, transform_any_array_to_numpy, transform_any_array_to_torch
from abc import abstractmethod, ABC
import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDictBase, tensorclass, TensorDict
from typing import Any, Iterator, Optional, Union, Iterable, Tuple, Dict, Generic, TypeVar, Callable
from dataclasses import dataclass
import gymnasium as gym

Tensor_Or_TensorDict = Union[torch.Tensor, TensorDict]
NNAgentOutput = AgentOutput[Tensor_Or_TensorDict, Tensor_Or_TensorDict, torch.Tensor]

@dataclass
class BatchedNNAgentOutput(Iterable[NNAgentOutput]):
    actions : torch.Tensor # (batch_size, *action_shape) if not is_seq, (batch_size, sequence_length, *action_shape) if is_seq
    log_probs : torch.Tensor # (batch_size, ) if not is_seq, (batch_size, sequence_length) if is_seq
    final_states: Optional[Tensor_Or_TensorDict] = None # (batch_size, *state_shape) if is_seq
    masks: Optional[torch.Tensor] = None # (batch_size, ) if not is_seq, (batch_size, sequence_length) if is_seq
    is_seq : bool = False

    def __iter__(self) -> Iterator[NNAgentOutput]:
        if self.is_seq:
            for i in range(self.actions.size(0)):
                i_seq_len = self.actions.size(1) if self.masks is None else int(torch.count_nonzero(self.masks[i]).item())
                for j in range(i_seq_len):
                    yield NNAgentOutput(
                        self.actions[i, j],
                        self.log_probs[i, j],
                        self.final_states[i] if self.final_states is not None and j == i_seq_len - 1 else None
                    )
        else:
            for i in range(self.actions.size(0)):
                if self.masks is not None and self.masks[i] == 0:
                    continue
                yield NNAgentOutput(
                    self.actions[i],
                    self.log_probs[i],
                    self.final_states[i] if self.final_states is not None else None
                )

@tensorclass
class NNAgentNetworkOutput:
    """
    NNAgentNetworkOutput is a dataclass that represents the output of a neural network.
    """
    output : Tensor_Or_TensorDict # (batch_size, *action_shape) if not is_seq, (batch_size, sequence_length, *action_shape) if is_seq
    masks: Optional[torch.Tensor] = None # (batch_size, ) if not is_seq, (batch_size, sequence_length) if is_seq
    state : Optional[Tensor_Or_TensorDict] = None # (batch_size, *state_shape) if is_seq
    is_seq : bool = False

_ActionSpaceMapperT = TypeVar("_ActionSpaceMapperT", bound=gym.Space)
class NNAgentActionMapper(ABC, nn.Module, Generic[_ActionSpaceMapperT]):
    """
    NNAgentActionMapper is a module that maps the output of a neural network to an action.
    """
    action_type : AgentActionType

    def __init__(self, action_space : _ActionSpaceMapperT) -> None:
        super().__init__()
        self.action_space = action_space
    
    @abstractmethod
    def forward(
        self, 
        nn_agent: "NNAgent[_ActionSpaceMapperT]",
        nn_output : NNAgentNetworkOutput, 
        is_update : bool = False,
        stage : AgentStage = AgentStage.ONLINE
    ) -> BatchedNNAgentOutput:
        """
        Should return a BatchedNNAgentOutput object, which represents the actions taken by the agent.
        """

        pass

    @abstractmethod
    def log_prob(
        self,
        nn_agent: "NNAgent[gym.spaces.Box]",
        nn_output : NNAgentNetworkOutput,
        actions : torch.Tensor,
        is_update : bool = False,
        stage : AgentStage = AgentStage.ONLINE
    ) -> BatchedNNAgentOutput:
        pass

class NNAgent(Agent[
    Tensor_Or_Numpy,
    Tensor_Or_Numpy,
    torch.Tensor,
    Tensor_Or_TensorDict,
    Tensor_Or_TensorDict, 
    torch.Tensor
], Generic[_ActionSpaceMapperT], nn.Module, ABC):
    """
    NNAgent is an agent that uses a neural network (written in PyTorch) to map observations to actions.
    """
    action_mapper : NNAgentActionMapper[_ActionSpaceMapperT]
    np_random : Optional[np.random.Generator] = None
    obs_shape : torch.Size

    def __init__(
        self, 
        is_sequence_model : bool,
        empty_state_vector : Tensor_Or_TensorDict
    ) -> None:
        super().__init__()
        self._empty_state_vector = empty_state_vector
        self._is_sequence_model = is_sequence_model

    @property
    def unwrapped(self) -> "NNAgent":
        return self
    
    @property
    def transform_parent(self) -> None:
        return None

    def seed(
        self, 
        seed : Optional[int] = None
    ) -> None:
        pass

    @property
    def action_type(self) -> AgentActionType:
        return self.action_mapper.action_type
    
    @property
    def is_sequence_model(self) -> bool:
        return self._is_sequence_model

    @property
    def empty_state_vector(self) -> Tensor_Or_TensorDict:
        return self._empty_state_vector

    @abstractmethod
    def forward(
        self,
        obs_batch: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        state: Optional[Tensor_Or_TensorDict] = None,
        is_seq = False, # If True, then obs_batch is shaped (batch_size, sequence_length, *observation_shape)
        stage: AgentStage = AgentStage.ONLINE,
        **kwargs: Any,
    ) -> NNAgentNetworkOutput:
        """
        Forward pass of the agent. 
        @param obs_batch: batch of observations, 
            shape (B, *obs_shape) or (B, S, *obs_shape).
        @param state: state of the agent. 
            For sequence models, this is (B, *state_shape) for the initial state of each sequence. 
        @param is_seq: whether the input is a sequence (B, S, *obs_shape) of observations.
        @param stage: stage (Exploration, Online, Offline, Eval) of the agent. 
        @returns: NNAgentNetworkOutput object
        """
        pass

    def add_transitions(
        self, 
        transition : Union[Iterable[EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy]], EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy]],
        stage : AgentStage = AgentStage.ONLINE
    ) -> None:
        pass

    @abstractmethod
    def update(self, stage: AgentStage = AgentStage.ONLINE, *args, **kwargs) -> Dict[str, Any]:
        pass

    @torch.jit.ignore
    def get_action_batch(
        self, 
        observations : Union[Tensor_Or_Numpy, Iterable[Tensor_Or_Numpy]], 
        states : Optional[Iterable[Optional[Union[np.ndarray, torch.Tensor, TensorDictBase]]]] = None, 
        stage : AgentStage = AgentStage.ONLINE,
        device : Optional[Union[torch.device, str, int]] = None,
        dtype : Optional[torch.dtype] = None
    ) -> BatchedNNAgentOutput:
        """
        Use this method to get actions from the agent.
        @param observations: batch of observations, 
            shape (batch_size[0], *observation_shape). 
            Note that this method only supports flattened batch of observations.
            Please do not input any sequence of observations. 
            If you want to input a sequence of observations, use forward() instead.
        @param states: state of the agent.
            For sequence models, this is (batch_size[0], *state_shape) for the initial state of each sequence.
        @param stage: stage (Exploration, Online, Offline, Eval) of the agent.
        @param device: device to put the observations and states on.
        @param dtype: dtype to put the observations and states on.
        @returns: BatchedNNAgentOutput object
        """
        if device is None:
            device = next(self.parameters()).device
        
        input_obs = transform_any_array_to_torch(observations)
        assert input_obs.size()[1:] == self.obs_shape, "Input observation shape does not match the agent's observation shape"
        if self.is_sequence_model:
            input_obs = input_obs.unsqueeze(1)
        input_obs.to(device, dtype=dtype, non_blocking=True)

        if not self.is_sequence_model or states is None:
            input_state = None
        else:
            if isinstance(states, (torch.Tensor, TensorDict)):
                input_state = states
            elif isinstance(states, np.ndarray):
                input_state = transform_any_array_to_torch(states)
            else:
                assert isinstance(states, Iterable), "States must be an iterable of tensors or a single tensor"
                to_stack_state_list = []

                for s in states:
                    if s is None:
                        to_stack_state_list.append(self.empty_state_vector.to(
                            device, dtype=dtype, non_blocking=True
                        ))
                    else:
                        assert isinstance(s, (torch.Tensor, TensorDict, np.ndarray)), "States must be an iterable of tensors or a single tensor"
                        
                        to_append = None
                        if isinstance(s, np.ndarray):
                            to_append = transform_any_array_to_torch(s)
                        else:
                            to_append = s
                        to_stack_state_list.append(to_append.to(
                            device, dtype=dtype, non_blocking=True
                        ))
                input_state = torch.stack(to_stack_state_list, dim=0)
            
            assert input_state.size()[1:] == self.empty_state_vector.size(), "Input state shape does not match the agent's state shape"
            input_state.to(device, dtype=dtype)
        
        output = self.forward(input_obs, state = input_state, is_seq=self.is_sequence_model, stage = stage)
        return self.action_mapper.forward(
            output,
            stage = stage,
            is_update=False
        )
