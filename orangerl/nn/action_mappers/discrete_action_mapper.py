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

from typing import Union, Tuple, Any
import gymnasium as gym
import torch
import numpy as np
from ...base.agent import AgentStage
from ..agent import NNAgentActionMapper
from ..data_util import BatchedNNAgentDiscreteOutput

class NNAgentDiscreteActionMapper(NNAgentActionMapper):
    def __init__(self, action_space: gym.Space) -> None:
        assert isinstance(action_space, gym.spaces.Discrete), "Action space must be a Discrete"
        super().__init__(action_space)

    """
    Forward pass of the action mapper
    param output: Output from the network, action shaped (batch_size, discrete_action_dimension) or (batch_size, 1, discrete_action_dimension) for sequence models
    The state should be shaped (batch_size, (*state_shape))
    return: The batch of actions, shaped (batch_size, (*action_shape))
    """
    def forward(self, output : Union[torch.Tensor, Tuple[torch.Tensor, Any]], stage : AgentStage = AgentStage.ONLINE) -> BatchedNNAgentStochasticOutput:
        if isinstance(output, tuple):
            batch, states = output
            assert batch.ndim > 2 and batch.shape[1] == 1 # Must be a sequence
            batch = batch.squeeze(1) # Remove the sequence dimension
        else:
            batch = output
            states = None
        
        batch = batch.flatten(1)
        
        if not torch.sum(batch, dim=-1).allclose(1.0):
            batch = torch.softmax(batch, dim=-1)
        
        assert batch.shape[-1] == self.action_space.n
        
        return BatchedNNAgentDiscreteOutput(
            action_probs=batch,
            states=self._wrap_state(states) if states is not None else None,
            stage=stage
        )