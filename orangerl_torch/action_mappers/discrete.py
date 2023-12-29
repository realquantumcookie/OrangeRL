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
from tensordict import is_tensor_collection, TensorDictBase
import numpy as np
from orangerl import AgentStage
from ..agent import NNAgent, NNAgentActionMapper, BatchedNNAgentOutput, NNAgentNetworkOutput

class NNAgentDiscreteActionMapper(NNAgentActionMapper[gym.spaces.Discrete]):

    def __init__(self, action_space: gym.spaces.Discrete) -> None:
        assert isinstance(action_space, gym.spaces.Discrete), "Action space must be Discrete"
        super().__init__(action_space)
    
    # @property
    # def action_space(self) -> gym.spaces.Discrete:
    #     return self._action_space
    
    # @action_space.setter
    # def action_space(self, action_space: gym.spaces.Discrete) -> None:
    #     assert isinstance(action_space, gym.spaces.Discrete), "Action space must be Discrete"
    #     self._action_space = action_space
    #     action_space_mean = (action_space.high + action_space.low) / 2
    #     action_space_span = (action_space.high - action_space.low) / 2
    #     self._action_space_mean = torch.from_numpy(action_space_mean)
    #     self._action_space_span = torch.from_numpy(action_space_span)

    def forward_distribution(
        self,
        nn_output : NNAgentNetworkOutput,
        stage : AgentStage = AgentStage.ONLINE
    ):
        assert isinstance(nn_output.output, torch.Tensor), "The output must be a tensor"
        nn_output_tensor_flattened = nn_output.output.flatten(2) if nn_output.is_seq else nn_output.output.flatten(1)
        assert nn_output_tensor_flattened.shape[-1] == self.action_space.n, "The last dimension of the output must match the discrete space"
        
        dist = torch.distributions.Categorical(logits=nn_output_tensor_flattened)
        return dist

    def log_prob_distribution(
        self,
        nn_output : NNAgentNetworkOutput,
        dist: torch.distributions.TransformedDistribution,
        action: torch.Tensor,
        stage: AgentStage = AgentStage.ONLINE
    ) -> torch.Tensor:
        log_prob : torch.Tensor = dist.log_prob(action - self.action_space.start)
        sum_dims = log_prob.shape[2:] if nn_output.is_seq else log_prob.shape[1:]
        log_prob = torch.sum(log_prob, dim=sum_dims)
        return log_prob

    def forward(
        self, 
        nn_output : NNAgentNetworkOutput, 
        is_update : bool = False,
        stage : AgentStage = AgentStage.ONLINE
    ) -> BatchedNNAgentOutput:
        
        dist = self.forward_distribution(
            nn_output, 
            stage
        )

        actions = dist.sample()
        actions += self.action_space.start
        log_probs = self.log_prob_distribution(
            nn_output,
            dist,
            actions,
            stage
        )
        
        return BatchedNNAgentOutput(
            actions = actions,
            log_probs = log_probs,
            final_states = nn_output.state,
            masks=nn_output.masks,
            is_seq=nn_output.is_seq
        )

    def log_prob(
        self, 
        nn_output: NNAgentNetworkOutput,
        actions: torch.Tensor, 
        is_update : bool = False,
        stage: AgentStage = AgentStage.ONLINE
    ) -> torch.Tensor:
        dist = self.forward_distribution(
            nn_output, 
            stage
        )
        return self.log_prob_distribution(
            nn_output,
            dist,
            actions,
            stage
        )