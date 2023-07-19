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

from ..base.agent import Agent, AgentOutput, AgentStage, AgentActionType
from ..base.data import TransitionBatch, EnvironmentStep
from .data_util import *
from abc import abstractmethod, ABC
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Iterator, Optional, Union, Iterable, Tuple, Dict
from dataclasses import dataclass
import gymnasium as gym

class NNAgentActionMapper(ABC, nn.Module):
    def __init__(self, action_space : gym.Space) -> None:
        super().__init__()
        self.action_space = action_space
    
    @abstractmethod
    def forward(self, output : Union[torch.Tensor, Tuple[torch.Tensor, Any]], stage : AgentStage = AgentStage.ONLINE) -> Union[BatchedNNAgentStochasticOutput, BatchedNNAgentDeterministicOutput]:
        pass

    @property
    def action_type(self) -> AgentActionType:
        pass

    @staticmethod
    def _wrap_state(state : Any) -> Iterable[Any]:
        if isinstance(state, dict):
            return BatchedNNAgentDictStateWrapper(state)
        elif isinstance(state, Iterable):
            return state
        else:
            raise ValueError("NNAgentActionMapper only accepts dict of iterables")


class NNAgent(Agent[Union[np.ndarray,torch.Tensor], Union[np.ndarray, torch.Tensor], Any], nn.Module, ABC):
    def __init__(self, device : Optional[Union[torch.device,str]] = None) -> None:
        super().__init__()
        self.device = device
        self.to(device)
        self._next_update_cache = None

    @property
    def action_type(self) -> AgentActionType:
        return self.action_mapper.action_type

    @property
    def action_mapper(self) -> NNAgentActionMapper:
        raise NotImplementedError

    def map_action_batch(self, forward_output : Union[Any, Tuple[Any, Any]], stage : AgentStage = AgentStage.ONLINE) -> Union[BatchedNNAgentStochasticOutput, BatchedNNAgentDeterministicOutput]:
        return self.action_mapper(forward_output, stage=stage)

    @property
    def is_sequence_model(self) -> bool:
        return False

    """
    Forward pass of the agent. 
    params:
        batch: batch of observations, shape (batch_size, *observation_shape). For sequence models, this is (batch_size, sequence_length, *observation_shape)
        state: state of the agent. For sequence models, this is (batch_size, *state_shape) for the initial state of each sequence. This can also be a dictionary containing different named states with the same shape.
        stage: stage (Exploration, Online, Offline, Eval) of the agent. 
    returns:
        For non-sequence models, returns actions shaped (batch_size, *action_shape)
        For sequence models, returns (actions, states) shaped (batch_size, sequence_length, *action_shape) and (batch_size, *state_shape) respectively
        Note that the state output will only give the final state of the sequence model.
    """
    @abstractmethod
    def forward(
        self,
        batch: torch.Tensor,
        state: Optional[Any] = None,
        stage: AgentStage = AgentStage.ONLINE,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        pass

    def add_transitions(
        self, 
        transition : Union[TransitionBatch[Union[np.ndarray,torch.Tensor], Union[np.ndarray,torch.Tensor]], EnvironmentStep[Union[np.ndarray,torch.Tensor], Union[np.ndarray,torch.Tensor]]],
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
        observations : Union[np.ndarray,torch.Tensor, Iterable[Union[np.ndarray,torch.Tensor]]], 
        states : Optional[Iterable[Optional[Any]]] = None, 
        stage : AgentStage = AgentStage.ONLINE
    ) -> Iterable[AgentOutput[torch.Tensor, Optional[torch.Tensor]]]:
        input_obs = transform_any_array_to_tensor(observations)
        if self.is_sequence_model:
            input_obs = input_obs.unsqueeze(1)
        
        if states is None:
            input_state = None
        else:
            if isinstance(next(iter(states)), dict):
                input_state = {}
                for key in next(iter(states)).keys():
                    input_state[key] = torch.stack([state[key] for state in states])
            else:
                input_state = transform_any_array_to_tensor(states)
            
        output = self.forward(input_obs, state = input_state, stage = stage)
        return self.map_action_batch(output, stage)
