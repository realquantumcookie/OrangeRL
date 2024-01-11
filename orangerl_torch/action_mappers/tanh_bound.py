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
from orangerl_torch import NNAgent, NNAgentActionMapper, BatchedNNAgentOutput, NNAgentNetworkOutput

class NNAgentTanhActionMapper(NNAgentActionMapper):
    def __init__(self, action_space: gym.spaces.Box) -> None:
        assert isinstance(action_space, gym.spaces.Box), "Action space must be a Box"
        assert np.all(np.isfinite(action_space.low)) and np.all(np.isfinite(action_space.high)), "Action space must be bounded"
        super().__init__()
        self.action_space = action_space
    
    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space
    
    @action_space.setter
    def action_space(self, action_space: gym.spaces.Box) -> None:
        assert isinstance(action_space, gym.spaces.Box), "Action space must be a Box"
        assert np.all(np.isfinite(action_space.low)) and np.all(np.isfinite(action_space.high)), "Action space must be bounded"
        assert action_space.dtype == np.float32 or action_space.dtype == np.float64, "Action space must be float"
        self._action_space = action_space
        action_space_mean = (action_space.high + action_space.low) / 2
        action_space_span = (action_space.high - action_space.low) / 2
        self._action_space_mean = torch.from_numpy(action_space_mean)
        self._action_space_span = torch.from_numpy(action_space_span)

    def forward_distribution(
        self,
        nn_output : NNAgentNetworkOutput,
        stage : AgentStage = AgentStage.ONLINE
    ):
        if isinstance(nn_output.output, torch.Tensor):
            assert nn_output.output.shape[-1] == 2, "The last dimension of the output must be 2"
            means_output = nn_output.output[...,0]
            log_stds_output = nn_output.output[...,1]
        else:
            output_keys = nn_output.output.keys()
            assert 'loc' in output_keys and 'log_scale' in output_keys, "The output must have keys 'loc' and 'log_scale'"
            means_output = nn_output.output['loc']
            log_stds_output = nn_output.output['log_scale']
        
        
        action_space_mean = self._action_space_mean.to(means_output.device)
        action_space_span = self._action_space_span.to(means_output.device)
        action_space_mean = action_space_mean.unsqueeze(0)
        action_space_span = action_space_span.unsqueeze(0)
        if nn_output.is_seq:
            action_space_mean = action_space_mean.unsqueeze(1)
            action_space_span = action_space_span.unsqueeze(1)
            means_output = means_output.reshape(means_output.shape[0], means_output.shape[1], -1)
            log_stds_output = log_stds_output.reshape(log_stds_output.shape[0], log_stds_output.shape[1], -1)
        else:
            means_output = means_output.reshape(means_output.shape[0], -1)
            log_stds_output = log_stds_output.reshape(log_stds_output.shape[0], -1)
        
        transforms = [
            torch.distributions.transforms.TanhTransform(),
            torch.distributions.transforms.AffineTransform(loc = action_space_mean, scale = action_space_span)
        ]
        dist = torch.distributions.Normal(means_output, torch.exp(log_stds_output))
        transformed_dist = torch.distributions.TransformedDistribution(dist, transforms)
        return dist, transformed_dist

    @staticmethod
    def log_prob_distribution(
        nn_output : NNAgentNetworkOutput,
        base_dist: torch.distributions.Normal,
        dist: torch.distributions.TransformedDistribution,
        action: torch.Tensor,
        stage: AgentStage = AgentStage.ONLINE
    ) -> torch.Tensor:
        if nn_output.is_seq:
            action = action.reshape(action.shape[0], action.shape[1], -1)
        else:
            action = action.reshape(action.shape[0], -1)
        
        log_prob : torch.Tensor = dist.log_prob(action)
        sum_dims = list(range(2, log_prob.ndim)) if nn_output.is_seq else list(range(1, log_prob.ndim))
        log_prob = torch.sum(log_prob, dim=sum_dims)
        return log_prob

    def forward(
        self, 
        nn_output : NNAgentNetworkOutput, 
        is_update : bool = False,
        stage : AgentStage = AgentStage.ONLINE
    ) -> BatchedNNAgentOutput:
        
        base_dist, dist = self.forward_distribution(
            nn_output, 
            stage
        )

        if stage != AgentStage.EVAL:
            actions : torch.Tensor = dist.rsample()
            log_probs = __class__.log_prob_distribution(
                nn_output,
                base_dist,
                dist,
                actions,
                stage
            )
        else:
            actions : torch.Tensor = base_dist.mean
            for transform in dist.transforms:
                actions = transform(actions)
            log_probs = torch.zeros(actions.shape[:2] if nn_output.is_seq else actions.shape[:1], dtype=actions.dtype, device=actions.device)

        if nn_output.is_seq:
            actions = actions.reshape(actions.shape[0], actions.shape[1], *self._action_space_mean.shape)
        else:
            actions = actions.reshape(actions.shape[0], *self._action_space_mean.shape)
        
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
        base_dist, dist = self.forward_distribution(
            nn_output, 
            stage
        )
        return __class__.log_prob_distribution(
            nn_output,
            base_dist,
            dist,
            actions,
            stage
        )