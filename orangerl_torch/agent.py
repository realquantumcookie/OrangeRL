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

from orangerl import Agent, AgentOutput, AgentStage, TransitionBatch, EnvironmentStep
from .data import NNBatch, Tensor_Or_Numpy, transform_any_array_to_numpy, transform_any_array_to_torch, nnbatch_from_transitions
from .replay_buffer import NNReplayBuffer
from abc import abstractmethod, ABC
import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDictBase, tensorclass, TensorDict
from typing import Any, Iterator, Optional, Union, Iterable, Tuple, Dict, Generic, TypeVar, Callable
from dataclasses import dataclass
import gymnasium as gym
from os import PathLike

Tensor_Or_TensorDict = Union[torch.Tensor, TensorDictBase]
Tensor_Or_TensorDict_Or_Numpy_Or_Dict = Union[torch.Tensor, TensorDictBase, np.ndarray, Dict[str, Any]]
NNAgentOutput = AgentOutput[Tensor_Or_TensorDict, Tensor_Or_TensorDict, torch.Tensor]

@dataclass
class BatchedNNAgentOutput(Iterable[NNAgentOutput]):
    actions : Tensor_Or_TensorDict # (batch_size, *action_shape) if not is_seq, (batch_size, sequence_length, *action_shape) if is_seq
    log_probs : torch.Tensor # (batch_size, ) if not is_seq, (batch_size, sequence_length) if is_seq
    final_states: Optional[Tensor_Or_TensorDict] = None # (batch_size, *state_shape) if is_seq
    masks: Optional[torch.Tensor] = None # None if not is_seq, (batch_size, sequence_length) if is_seq
    is_seq : bool = False

    def __iter__(self) -> Iterator[NNAgentOutput]:
        if self.is_seq:
            for i in range(self.actions.size(0)):
                i_seq_len = self.actions.size(1) if self.masks is None else int(torch.count_nonzero(self.masks[i]).item())
                if self.masks is not None:
                    assert self.masks[i, :i_seq_len].all(), "masks must be all True for the first i_seq_len elements"
                
                for j in range(i_seq_len):
                    yield NNAgentOutput(
                        self.actions[i, j],
                        self.log_probs[i, j],
                        self.final_states[i] if self.final_states is not None and j == i_seq_len - 1 else None
                    )
        else:
            assert self.masks is None, "masks must be None if is_seq is False"
            for i in range(self.actions.size(0)):
                # if self.masks is not None and self.masks[i] == 0:
                #     continue
                yield NNAgentOutput(
                    self.actions[i],
                    self.log_probs[i],
                    self.final_states[i] if self.final_states is not None else None
                )

class NNAgent(Agent[
    Tensor_Or_TensorDict,
    Tensor_Or_TensorDict,
    Tensor_Or_TensorDict,
    torch.Tensor
], nn.Module, ABC):
    """
    NNAgent is an agent that uses a neural network (written in PyTorch) to map observations to actions.
    """
    observe_transition_infos : bool

    def __init__(
        self, 
        is_sequence_model : bool,
        empty_state : Optional[Tensor_Or_TensorDict],
        init_stage : AgentStage = AgentStage.ONLINE,
        decay_factor : float = 0.99,
    ) -> None:
        super().__init__()
        assert empty_state is not None or not is_sequence_model, "empty_state must be provided for sequence models"
        assert decay_factor >= 0.0 and decay_factor < 1.0, "decay_factor must be in [0.0, 1.0)"
        self.empty_state = empty_state
        self._is_sequence_model = is_sequence_model
        self.decay_factor = decay_factor
        self._current_stage = init_stage

    @property
    def current_stage(self) -> AgentStage:
        return self._current_stage
    
    @current_stage.setter
    def current_stage(self, stage : AgentStage) -> None:
        if stage == AgentStage.EVAL:
            self.train(False)
        else:
            self.train(True)
        
        self._current_stage = stage

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
    def replay_buffer(self) -> Optional[NNReplayBuffer]:
        return None

    @property
    def is_sequence_model(self) -> bool:
        return self._is_sequence_model

    @abstractmethod
    def forward(
        self,
        obs_batch: Tensor_Or_TensorDict,
        masks: Optional[torch.Tensor] = None,
        state: Optional[Tensor_Or_TensorDict] = None,
        is_update = False,
        **kwargs: Any,
    ) -> BatchedNNAgentOutput:
        """
        Forward pass of the agent. 
        @param obs_batch: batch of observations, 
            shape (B, *obs_shape) or (B, S, *obs_shape).
        @param state: state of the agent. 
            For sequence models, this is (B, *state_shape) for the initial state of each sequence. 
        @param is_seq: whether the input is a sequence (B, S, *obs_shape) of observations.
        @returns: BatchedNNAgentOutput object
        """
        ...

    @abstractmethod
    def observe_transitions(
        self, 
        transition : Union[
            Iterable[EnvironmentStep[Tensor_Or_TensorDict_Or_Numpy_Or_Dict, Tensor_Or_TensorDict_Or_Numpy_Or_Dict]], 
            EnvironmentStep[Tensor_Or_TensorDict_Or_Numpy_Or_Dict, Tensor_Or_TensorDict_Or_Numpy_Or_Dict]
        ]
    ) -> None:
        if isinstance(transition, NNBatch):
            to_observe = transition
        elif isinstance(transition, EnvironmentStep):
            to_observe = nnbatch_from_transitions([transition], save_info=self.observe_transition_infos)
        else:
            to_observe = nnbatch_from_transitions(transition, save_info=self.observe_transition_infos)
        self._observe_transitions(to_observe)

    @abstractmethod
    def _observe_transitions(
        self,
        transition : NNBatch
    ) -> None:
        ...

    @abstractmethod
    def update(self, *args, **kwargs) -> Dict[str, Any]:
        ...

    @torch.jit.ignore
    def get_action_batch(
        self, 
        observations : Union[Tensor_Or_TensorDict, Iterable[Tensor_Or_TensorDict]], 
        states : Optional[Iterable[Optional[Tensor_Or_TensorDict]]] = None, 
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
        @param device: device to put the observations and states on.
        @param dtype: dtype to put the observations and states on.
        @returns: BatchedNNAgentOutput object
        """
        if device is None:
            device = next(self.parameters()).device
        
        if isinstance(observations, (torch.Tensor, TensorDict)):
            input_obs = observations
        else:
            input_obs = torch.stack([transform_any_array_to_torch(obs) for obs in observations], dim=0)
        
        if self.is_sequence_model:
            input_obs = input_obs.unsqueeze(1)
        input_obs.to(device, dtype=dtype, non_blocking=True)

        if not self.is_sequence_model or states is None:
            input_state = None
        else:
            if isinstance(states, (torch.Tensor, TensorDict)):
                input_state = states
            else:
                to_stack_state_list = []

                for s in states:
                    if s is None:
                        to_stack_state_list.append(self.empty_state.to(
                            device, dtype=dtype, non_blocking=True
                        ))
                    else:
                        assert isinstance(s, (torch.Tensor, TensorDict))
                        to_stack_state_list.append(s.to(
                            device, dtype=dtype, non_blocking=True
                        ))
                input_state = torch.stack(to_stack_state_list, dim=0)
            
            input_state = input_state.to(device, dtype=dtype)
        
        output = self.forward(input_obs, state = input_state, is_update=False)
        return output

    def save(self, path : Union[str, PathLike]) -> None:
        torch.save(
            self.state_dict(),
            path
        )

    def load(self, path : Union[str, PathLike]) -> None:
        self.load_state_dict(
            torch.load(path)
        )