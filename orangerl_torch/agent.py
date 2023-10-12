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
from .data_util import *
from abc import abstractmethod, ABC
import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict, tensorclass
from typing import Any, Iterator, Optional, Union, Iterable, Tuple, Dict, Generic, TypeVar
import gymnasium as gym
from dataclasses import dataclass

NNAgentState = Union[torch.Tensor, TensorDict]
NNAgentOutput = AgentOutput[torch.Tensor, NNAgentState, torch.Tensor]

@tensorclass
class BatchedNNAgentOutput(Iterable[NNAgentOutput]):
    actions : torch.Tensor # (*batch_size, *action_shape)
    log_probs : torch.Tensor # (*batch_size, 1)
    states: Optional[NNAgentState] = None # (*batch_size, *state_shape)

    def __iter__(self) -> Iterator[NNAgentOutput]:
        flattened_self = self.reshape(-1)
        for i in range(flattened_self.batch_size[0]):
            yield NNAgentOutput(
                action = flattened_self.actions[i],
                log_prob = flattened_self.log_probs[i].flatten()[0],
                state = self.states[i] if self.states is not None else None
            )

class NNAgentActionMapper(ABC, nn.Module):
    """
    NNAgentActionMapper is a module that maps the output of a neural network to an action.
    """
    action_type : AgentActionType

    def __init__(self, action_space : gym.Space) -> None:
        super().__init__()
        self.action_space = action_space
    
    @abstractmethod
    def forward(
        self, 
        output : Union[torch.Tensor, Tuple[torch.Tensor, NNAgentState]], 
        stage : AgentStage = AgentStage.ONLINE
    ) -> BatchedNNAgentOutput:
        """
        Should return a BatchedNNAgentOutput object, which represents the actions taken by the agent.
        @param output: output of the neural network, shaped (*batch_size, *output_shape)
        @param stage: stage (Exploration, Online, Offline, Eval) of the agent.
        @returns: BatchedNNAgentOutput object
        """
        pass

    @abstractmethod
    def log_prob(
        self,
        output : Union[torch.Tensor, Tuple[torch.Tensor, NNAgentState]],
        action : torch.Tensor,
        stage : AgentStage = AgentStage.ONLINE
    ) -> torch.Tensor:
        """
        Returns the log probability of the action given the output of the neural network.
        @param output: output of the neural network, shaped (*batch_size, *output_shape)
        @param action: action taken, shaped (*batch_size, *action_shape)
        @param stage: stage (Exploration, Online, Offline, Eval) of the agent.
        @returns: log probability of the action, shaped (*batch_size, 1)
        """
        pass

class NNAgent(Agent[torch.Tensor, torch.Tensor, NNAgentState, torch.Tensor], nn.Module, ABC):
    """
    NNAgent is an agent that uses a neural network (written in PyTorch) to map observations to actions.
    For sequence models, the batch_size in method annotations would be (episode_length, sequence_length)
    Otherwise, the batch_size in method annotations would be (transition_length, )
    """
    action_mapper : NNAgentActionMapper
    np_random : Optional[np.random.Generator] = None

    def __init__(
        self, 
        is_sequence_model : bool,
        empty_state_vector : NNAgentState
    ) -> None:
        super().__init__()
        self._empty_state_vector = empty_state_vector
        self._is_sequence_model = is_sequence_model

    @property
    def unwrapped(self) -> "NNAgent":
        return self

    @property
    def action_type(self) -> AgentActionType:
        return self.action_mapper.action_type\
    
    @property
    def is_sequence_model(self) -> bool:
        return self._is_sequence_model

    @property
    def empty_state_vector(self) -> NNAgentState:
        return self._empty_state_vector

    @abstractmethod
    def forward(
        self,
        obs_batch: torch.Tensor,
        state: Optional[NNAgentState] = None,
        stage: AgentStage = AgentStage.ONLINE,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, NNAgentState]]:
        """
        Forward pass of the agent. 
        @param obs_batch: batch of observations, 
            shape (*batch_size, *observation_shape). 
        @param state: state of the agent. 
            For sequence models, this is (batch_size[0], *state_shape) for the initial state of each sequence. 
        @param stage: stage (Exploration, Online, Offline, Eval) of the agent. 
        @returns:
            Return actions shaped (*batch_size, *action_shape)
            For sequence models, this returns a tuple with final state of each sequence attached that is sized as (batch_size[0], *state_shape).
            Note that the state output will only give the final state of the sequence model.
        """
        pass

    def action_log_prob(
        self,
        obs_batch: torch.Tensor,
        state: Optional[NNAgentState] = None,
        action: torch.Tensor = None,
        stage: AgentStage = AgentStage.ONLINE,
    ) -> torch.Tensor:
        output = self.forward(obs_batch, state, stage)
        return self.action_mapper.log_prob(output, action, stage)

    @abstractmethod
    def add_transitions(
        self, 
        transition : Union[Iterable[EnvironmentStep[torch.Tensor, torch.Tensor]], EnvironmentStep[torch.Tensor, torch.Tensor]],
        stage : AgentStage = AgentStage.ONLINE
    ):
        pass

    @abstractmethod
    def update(self, batch_size : Optional[int] = None, *args, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def prefetch_update_data(self, batch_size : Optional[int] = None, *args, **kwargs) -> None:
        pass

    def get_action_batch(
        self, 
        observations : Union[torch.Tensor, np.ndarray, Iterable[Union[torch.Tensor, np.ndarray]]], 
        states : Optional[Union[NNAgentState, np.ndarray, Iterable[Optional[Union[NNAgentState, np.ndarray]]]]] = None, 
        stage : AgentStage = AgentStage.ONLINE,
        device : Optional[Union[torch.device, str, int]] = None,
        dtype : Optional[torch.dtype] = None
    ) -> BatchedNNAgentOutput:
        input_obs = transform_any_array_to_torch(observations)
        if self.is_sequence_model:
            input_obs = input_obs.unsqueeze(1)
        input_obs.to(device, dtype=dtype, non_blocking=True)

        if states is None:
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
            input_state.to(device, dtype=dtype)
        
        output = self.forward(input_obs, state = input_state, stage = stage)
        return self.action_mapper.forward(
            output,
            stage = stage
        )
