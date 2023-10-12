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
from ..agent import NNAgentActionMapper, NNAgentState, BatchedNNOutput

class NNAgentDiscreteActionMapper(NNAgentActionMapper):
    def __init__(self, action_space: gym.Space) -> None:
        assert isinstance(action_space, gym.spaces.Discrete), "Action space must be a Discrete"
        super().__init__(action_space)

    def forward_distribution(
        self,
        output : Union[torch.Tensor, Tuple[torch.Tensor, NNAgentState]],
        stage : AgentStage = AgentStage.ONLINE
    ):
        if isinstance(output, tuple):
            is_sequence = True
            batch, states = output
            assert batch.ndim > 2, "The output must be a sequence shaped (batch_size, seq_length, n_actions)"
        else:
            is_sequence = False
            batch = output
            states = None
        
        batch = batch.flatten(start_dim=2) if is_sequence else batch.flatten(start_dim=1)
        
        assert batch.shape[-1] == self.action_space.n
        
        return torch.distributions.Categorical(logits=batch), states, is_sequence

    """
    Forward pass of the action mapper
    param output: Output from the network, action shaped (batch_size, discrete_action_dimension) or (batch_size, 1, discrete_action_dimension) for sequence models
    The state should be shaped (batch_size, (*state_shape))
    return: The batch of actions, shaped (batch_size, (*action_shape))
    """
    def forward(self, output : Union[torch.Tensor, Tuple[torch.Tensor, NNAgentState]], stage : AgentStage = AgentStage.ONLINE) -> BatchedNNAgentStochasticOutput:
        dist, states, is_sequence = self.forward_distribution(output, stage)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return BatchedNNOutput(
            actions = actions,
            log_probs = log_probs,
            states = states,
            is_sequence = is_sequence
        )
    
    def log_prob(
        self, 
        output: Union[torch.Tensor, Tuple[torch.Tensor, NNAgentState]],
        action: torch.Tensor, 
        stage: AgentStage = AgentStage.ONLINE
    ) -> torch.Tensor:
        dist, _, _ = self.forward_distribution(output, stage)
        return dist.log_prob(action)