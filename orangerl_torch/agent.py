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
from .data import NNBatch, NNBatchSeq
from .data_util import *
from abc import abstractmethod, ABC
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence, unpack_sequence
from tensordict import TensorDict, tensorclass
from typing import Any, Iterator, Optional, Union, Iterable, Tuple, Dict, Generic, TypeVar, Callable
import gymnasium as gym
from dataclasses import dataclass

NNAgentStateInput = Union[np.ndarray, torch.Tensor, TensorDict]
NNAgentState = Union[torch.Tensor, TensorDict]
NNAgentOutput = AgentOutput[torch.Tensor, NNAgentState, torch.Tensor]
NNAgentNetworkInputOutput = Union[torch.Tensor, PackedSequence]

@dataclass
class BatchedNNAgentOutput(Iterable[NNAgentOutput]):
    actions : NNAgentNetworkInputOutput # (*batch_size, *action_shape)
    log_probs : NNAgentNetworkInputOutput # (*batch_size, )
    final_states: Optional[NNAgentState] = None # (batch_size[0], *state_shape)
    is_seq : bool = False

    def __iter__(self) -> Iterator[NNAgentOutput]:
        assert (
            (isinstance(self.actions, PackedSequence) and isinstance(self.log_probs, PackedSequence) and self.is_seq) or
            (isinstance(self.actions, torch.Tensor) and isinstance(self.log_probs, torch.Tensor))
        ), "actions and log_probs must be both PackedSequence or both torch.Tensor"
        
        if isinstance(self.actions, PackedSequence):
            actions_list = unpack_sequence(self.actions)
            log_probs_list = unpack_sequence(self.log_probs)
            for i in range(len(actions_list)):
                i_seq_len = actions_list[i].size(0)
                for j in range(i_seq_len):
                    yield NNAgentOutput(
                        actions_list[i][j],
                        log_probs_list[i][j],
                        self.final_states[i] if self.final_states is not None and j == i_seq_len - 1 else None
                    )
        else:
            if self.is_seq:
                for i in range(self.actions.size(0)):
                    i_seq_len = self.actions.size(1)
                    for j in range(i_seq_len):
                        yield NNAgentOutput(
                            self.actions[i, j],
                            self.log_probs[i, j],
                            self.final_states[i] if self.final_states is not None and j == i_seq_len - 1 else None
                        )
            else:
                for i in range(self.actions.size(0)):
                    yield NNAgentOutput(
                        self.actions[i],
                        self.log_probs[i],
                        self.final_states[i] if self.final_states is not None else None
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
        nn_output : Union[NNAgentNetworkInputOutput, Tuple[NNAgentNetworkInputOutput, NNAgentState]], 
        stage : AgentStage = AgentStage.ONLINE
    ) -> BatchedNNAgentOutput:
        """
        Should return a BatchedNNAgentOutput object, which represents the actions taken by the agent.
        @param nn_output: output of the neural network, can be a tuple of (output, state) for sequence models.
            output should be shaped (*batch_size, *output_shape) or a PackedSequence.
            state should be shaped (batch_size[0], *state_shape) for sequence models.
        @param stage: stage (Exploration, Online, Offline, Eval) of the agent.
        @returns: BatchedNNAgentOutput object
        """
        pass

    @abstractmethod
    def log_prob(
        self,
        nn_output : Union[NNAgentNetworkInputOutput, Tuple[NNAgentNetworkInputOutput, NNAgentState]],
        action : NNAgentNetworkInputOutput,
        stage : AgentStage = AgentStage.ONLINE
    ) -> NNAgentNetworkInputOutput:
        """
        Returns the log probability of the action given the output of the neural network.
        @param nn_output: output of the neural network, can be a tuple of (output, state) for sequence models.
            output should be shaped (*batch_size, *output_shape) or a PackedSequence.
            state should be shaped (batch_size[0], *state_shape) for sequence models.
        @param action: action taken, shaped (*batch_size, *action_shape)
            The action can also be a PackedSequence.
        @param stage: stage (Exploration, Online, Offline, Eval) of the agent.
        @returns: log probability of the action, shaped (*batch_size, )
            If the "nn_output" is a PackedSequence, then the output will also be a PackedSequence.
        """
        pass

class NNAgent(Agent[
    Tensor_Or_Numpy,
    Tensor_Or_Numpy,
    torch.Tensor,
    NNAgentStateInput,
    NNAgentState, 
    torch.Tensor
], nn.Module, ABC):
    """
    NNAgent is an agent that uses a neural network (written in PyTorch) to map observations to actions.
    For sequence models, the batch_size in method annotations would be (episode_length, sequence_length)
    Otherwise, the batch_size in method annotations would be (transition_length, )
    """
    action_mapper : NNAgentActionMapper
    np_random : Optional[np.random.Generator] = None
    obs_shape : torch.Size

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
        return self.action_mapper.action_type
    
    @property
    def is_sequence_model(self) -> bool:
        return self._is_sequence_model

    @property
    def empty_state_vector(self) -> NNAgentState:
        return self._empty_state_vector

    @abstractmethod
    def forward(
        self,
        obs_batch: NNAgentNetworkInputOutput,
        state: Optional[NNAgentState] = None,
        stage: AgentStage = AgentStage.ONLINE,
        **kwargs: Any,
    ) -> Union[NNAgentNetworkInputOutput, Tuple[NNAgentNetworkInputOutput, NNAgentState]]:
        """
        Forward pass of the agent. 
        @param obs_batch: batch of observations, 
            shape (*batch_size, *observation_shape). 
            It can also be a PackedSequence
        @param state: state of the agent. 
            For sequence models, this is (batch_size[0], *state_shape) for the initial state of each sequence. 
        @param stage: stage (Exploration, Online, Offline, Eval) of the agent. 
        @returns:
            Return actions shaped (*batch_size, *action_shape). 
            If input is a PackedSequence, then the output will also be a PackedSequence.
            For sequence models, this returns a tuple with final state of each sequence attached that is sized as (batch_size[0], *state_shape).
        """
        pass

    @torch.jit.export
    def action_log_prob(
        self,
        obs_batch: NNAgentNetworkInputOutput,
        state: Optional[NNAgentState] = None,
        action: torch.Tensor = None,
        stage: AgentStage = AgentStage.ONLINE,
    ) -> NNAgentNetworkInputOutput:
        if isinstance(obs_batch, torch.Tensor):
            assert obs_batch.size()[
                1 if not self.is_sequence_model else 2:
            ] == self.obs_shape, "Input observation shape does not match the agent's observation shape"
        if state is not None:
            assert state.size()[1:] == self.empty_state_vector.size(), "Input state shape does not match the agent's state shape"

        output = self.forward(obs_batch, state, stage)
        return self.action_mapper.log_prob(output, action, stage)

    def add_transitions_relabel(
        transitions: Union[
            Iterable[EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy]],
            EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy]
        ],
        stage: AgentStage = AgentStage.ONLINE
    ) -> Union[
        Iterable[EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy]],
        EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy]
    ]:
        """
        Callable that relabels a transition or a bunch of transitions
            before adding them to the agent's replay buffer or directly training on-policy.
        """
        return transitions

    def add_transitions(
        self, 
        transition : Union[
            Iterable[EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy]], 
            EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy]
        ],
        stage : AgentStage = AgentStage.ONLINE
    ) -> None:
        relabeled_transition = self.add_transitions_relabel(transition, stage)
        self.add_transitions_relabeled(relabeled_transition, stage)

    @abstractmethod
    def add_transitions_relabeled(
        self,
        transition : Union[Iterable[EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy]], EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy]],
        stage : AgentStage = AgentStage.ONLINE
    ) -> None:
        pass

    def update_relabel(
        transition: Union[
            EnvironmentStepInSeq[Tensor_Or_Numpy, Tensor_Or_Numpy],
            EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy]
        ],
        step_until_end: Optional[int] = None,
        stage: AgentStage = AgentStage.ONLINE
    ):
        """
        Callable that relabels a transition.
        If a sequence input is given, the optional int parameter `step_until_end` gives information 
            about the position of the current transition relative to the end of the "sub-episode" sampled.
        For example, if the optional int parameter is 1, then the transition in parameter is 
            the transition exactly before the last transition in the "sub-episode" sampled.
        If a non-sequence input is given, the optional step_until_end parameter is ignored (None).
        """
        return transition

    @abstractmethod
    def update(self, stage: AgentStage = AgentStage.ONLINE, batch_size : Optional[int] = None, *args, **kwargs) -> Dict[str, Any]:
        pass

    def get_action_relabel(
        self, 
        observation : Union[Tensor_Or_Numpy, Iterable[Tensor_Or_Numpy]], 
        states: Optional[Union[NNAgentState, np.ndarray, Iterable[Optional[Union[NNAgentState, np.ndarray]]]]] = None,
        stage : AgentStage = AgentStage.ONLINE,
        **kwargs
    ) -> Tuple[
        Union[Tensor_Or_Numpy, Iterable[Tensor_Or_Numpy]],
        Optional[Union[NNAgentState, np.ndarray, Iterable[Optional[Union[NNAgentState, np.ndarray]]]]]
    ]:
        """
        Callable that relabels an observation or a bunch of observations when getting a batch of actions.
        """
        return observation, states

    @torch.jit.ignore
    def get_action_batch(
        self, 
        observations : Union[Tensor_Or_Numpy, Iterable[Tensor_Or_Numpy]], 
        states : Optional[Union[NNAgentState, np.ndarray, Iterable[Optional[Union[NNAgentState, np.ndarray]]]]] = None, 
        stage : AgentStage = AgentStage.ONLINE,
        relabel_kwargs : Dict[str, Any] = {},
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
        observations, states = self.get_action_relabel(observations, states, stage, **relabel_kwargs)
        
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
        
        output = self.forward(input_obs, state = input_state, stage = stage)
        return self.action_mapper.forward(
            output,
            stage = stage
        )
