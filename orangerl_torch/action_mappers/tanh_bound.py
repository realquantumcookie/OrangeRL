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
from ..agent import NNAgentActionMapper, NNAgentState, BatchedNNAgentOutput

class NNAgentTanhActionMapper(NNAgentActionMapper):
    def __init__(self, action_space: gym.Space) -> None:
        assert isinstance(action_space, gym.spaces.Box), "Action space must be a Box"
        super().__init__(action_space)
        assert np.all(np.isfinite(action_space.low)) and np.all(np.isfinite(action_space.high)), "Action space must be bounded"

    def forward_distribution(
        self,
        output : Union[torch.Tensor, Tuple[torch.Tensor, NNAgentState]],
        stage : AgentStage = AgentStage.ONLINE
    ):
        if isinstance(output, tuple):
            is_sequence = True
            batch, states = output
            assert batch.ndim > 3, "The output must be a sequence shaped (batch_size, seq_length, *action_shape, 2)"
        else:
            is_sequence = False
            batch = output
            states = None
            assert batch.ndim > 2, "The output must be a sequence shaped (batch_size, *action_shape, 2)"
        assert batch.shape[-1] == 2, "The last dimension of the output must be 2"
        mean = batch[...,0]
        std = torch.exp(batch[...,1])
        action_space_mean = (self.action_space.high + self.action_space.low) / 2
        action_space_span = (self.action_space.high - self.action_space.low) / 2
        action_space_mean = action_space_mean[np.newaxis, :]
        action_space_span = action_space_span[np.newaxis, :]
        if isinstance(output, tuple):
            action_space_mean = action_space_mean[np.newaxis, :]
            action_space_span = action_space_span[np.newaxis, :]
        transforms = [
            torch.distributions.transforms.TanhTransform(),
            torch.distributions.transforms.AffineTransform(loc = torch.from_numpy(action_space_mean), scale = torch.from_numpy(action_space_span))
        ]
        dist = torch.distributions.Normal(mean, std)
        transformed_dist = torch.distributions.TransformedDistribution(dist, transforms)
        return transformed_dist, states, is_sequence

    @staticmethod
    def log_prob_distribution(
        dist: torch.distributions.TransformedDistribution,
        action: torch.Tensor,
        is_sequence: bool
    ) -> torch.Tensor:
        log_prob = dist.log_prob(action)
        sum_dims = log_prob.shape[2:] if is_sequence else log_prob.shape[1:]
        log_prob = torch.sum(log_prob, dim=sum_dims)
        return log_prob

    """
    Forward pass of the action mapper
    param output: Output from the network, action shaped (batch_size, (*action_shape), 2) or (batch_size, 1, (*action_shape), 2) for sequence models
    The state should be shaped (batch_size, (*state_shape))
    return: The batch of actions, shaped (batch_size, (*action_shape))
    """
    def forward(
        self, 
        output : Union[torch.Tensor, Tuple[torch.Tensor, NNAgentState]], 
        stage : AgentStage = AgentStage.ONLINE
    ) -> BatchedNNAgentOutput:
        dist, states, is_sequence = self.forward_distribution(output, stage)
        
        actions = dist.rsample()

        return BatchedNNAgentOutput(
            actions = actions,
            log_probs = __class__.log_prob_distribution(dist, actions, is_sequence),
            states = states,
            is_sequence = is_sequence
        )

    def log_prob(
        self, 
        output: Union[torch.Tensor, Tuple[torch.Tensor, NNAgentState]],
        action: torch.Tensor, 
        stage: AgentStage = AgentStage.ONLINE
    ) -> torch.Tensor:
        dist, _, is_sequence = self.forward_distribution(output, stage)
        return __class__.log_prob_distribution(dist, action, is_sequence)