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
    def __init__(self, action_min : torch.Tensor, action_max : torch.Tensor) -> None:
        assert torch.all(torch.isfinite(action_min)) and torch.all(torch.isfinite(action_max)), "Action space must be bounded"
        assert torch.all(action_min <= action_max), "Action space must be bounded"
        assert action_min.shape == action_max.shape, "Action space must share the same shape"
        assert action_min.ndim == 1, "Action space must be 1D"

        super().__init__()
        self.action_min = action_min
        self.action_max = action_max
        self.__eps = np.finfo(np.float32).eps.item()
        self.tanh_transform = torch.distributions.transforms.TanhTransform(cache_size=1)

    def forward_distribution(
        self,
        nn_output : NNAgentNetworkOutput,
        stage : AgentStage = AgentStage.ONLINE
    ):
        if isinstance(nn_output.output, torch.Tensor):
            if nn_output.is_seq:
                nn_output_flattened = nn_output.output.flatten(start_dim=2)
                action_size = nn_output_flattened.shape[-1] // 2
                means_output = nn_output_flattened[:, :, :action_size]
                log_stds_output = nn_output_flattened[:, :, action_size:]
            else:
                nn_output_flattened = nn_output.output.flatten(start_dim=1)
                action_size = nn_output_flattened.shape[-1] // 2
                means_output = nn_output_flattened[:, :action_size]
                log_stds_output = nn_output_flattened[:, action_size:]
        else:
            output_keys = nn_output.output.keys()
            assert 'loc' in output_keys and 'log_scale' in output_keys, "The output must have keys 'loc' and 'log_scale'"
            means_output = nn_output.output['loc']
            log_stds_output = nn_output.output['log_scale']
            if nn_output.is_seq:
                means_output = means_output.flatten(start_dim=2)
                log_stds_output = log_stds_output.flatten(start_dim=2)
            else:
                means_output = means_output.flatten(start_dim=1)
                log_stds_output = log_stds_output.flatten(start_dim=1)
            action_size = means_output.shape[-1]
        
        dist = torch.distributions.Normal(means_output, torch.exp(log_stds_output))
        return dist

    def forward(
        self, 
        nn_output : NNAgentNetworkOutput, 
        is_update : bool = False,
        stage : AgentStage = AgentStage.ONLINE
    ) -> BatchedNNAgentOutput:
        
        base_dist = self.forward_distribution(
            nn_output, 
            stage
        )

        action_min = self.action_min.to(nn_output.output.device)
        action_max = self.action_max.to(nn_output.output.device)
        if stage != AgentStage.EVAL or is_update:
            raw_actions : torch.Tensor = base_dist.rsample()
        else:
            raw_actions : torch.Tensor = base_dist.mean
        
        squashed_actions : torch.Tensor = self.tanh_transform(raw_actions)
        
        # apply correction for Tanh squashing when computing logprob from Gaussian
        # You can check out the original SAC paper (arXiv 1801.01290): Eq 21.
        # in appendix C to get some understanding of this equation.
        # Taken from Tianshou's source code https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/sac.py
        log_probs = base_dist.log_prob(raw_actions).sum(-1) - torch.log(1 - squashed_actions.pow(2) + self.__eps).sum(
            -1
        )
        
        actions = action_min + (action_max - action_min) * (squashed_actions + 1) / 2

        if nn_output.is_seq:
            actions = actions.reshape(actions.shape[0], actions.shape[1], -1)
        else:
            actions = actions.reshape(actions.shape[0], -1)
        
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
        base_dist = self.forward_distribution(
            nn_output, 
            stage
        )
        actions = actions.flatten(start_dim=1) if not nn_output.is_seq else actions.flatten(start_dim=2)
        squashed_actions = (2 * (actions - self.action_min) / (self.action_max - self.action_min)) - 1
        raw_actions = self.tanh_transform.inv(squashed_actions)
        log_probs = base_dist.log_prob(raw_actions).sum(-1) - torch.log(1 - squashed_actions.pow(2) + self.__eps).sum(
            -1
        )
        return log_probs