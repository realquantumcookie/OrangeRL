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
from ..data_util import BatchedNNAgentStochasticOutput

class NNAgentTanhActionMapper(NNAgentActionMapper):
    def __init__(self, action_space: gym.Space) -> None:
        assert isinstance(action_space, gym.spaces.Box), "Action space must be a Box"
        super().__init__(action_space)
        assert np.all(np.isfinite(action_space.low)) and np.all(np.isfinite(action_space.high)), "Action space must be bounded"

    """
    Forward pass of the action mapper
    param output: Output from the network, action shaped (batch_size, (*action_shape), 2) or (batch_size, 1, (*action_shape), 2) for sequence models
    The state should be shaped (batch_size, (*state_shape))
    return: The batch of actions, shaped (batch_size, (*action_shape))
    """
    def forward(self, output : Union[torch.Tensor, Tuple[torch.Tensor, Any]], stage : AgentStage = AgentStage.ONLINE) -> BatchedNNAgentStochasticOutput:
        if isinstance(output, tuple):
            batch, states = output
            assert batch.ndim > 2 and batch.shape[1] == 1, "The output must be a sequence shaped (batch_size, 1, *action_shape, 2)"
            batch = batch.squeeze(1) # Remove the sequence dimension
        else:
            batch = output
            states = None
        
        assert batch.shape[-1] == 2, "The output must be a sequence shaped (batch_size, 1, *action_shape, 2)"
        mean = torch.tanh(batch[...,0])
        std = torch.exp(batch[...,1])
        dist = torch.distributions.Normal(mean, std)
        
        action_space_mean = (self.action_space.high + self.action_space.low) / 2
        action_space_span = (self.action_space.high - self.action_space.low) / 2
        transforms = [
            torch.distributions.transforms.TanhTransform(),
            torch.distributions.transforms.AffineTransform(loc = action_space_mean[np.newaxis, :], scale = action_space_span[np.newaxis, :])
        ]
        
        transformed_dist = torch.distributions.TransformedDistribution(dist, transforms)
        return BatchedNNAgentStochasticOutput(
            action_dist = transformed_dist,
            states=self._wrap_state(states) if states is not None else None,
            stage=stage
        )