from orangerl_torch import NNAgentCriticMapper
from typing import Any, Dict, List, Optional, Tuple
from orangerl import AgentStage
import torch
import torch.nn as nn

from orangerl_torch.agent_util import BatchedNNCriticOutput, NNAgentNetworkOutput

class NNDirectCriticMapper(NNAgentCriticMapper):
    def forward(
        self, 
        nn_output : NNAgentNetworkOutput, 
        is_update : bool = False,
        is_discrete : bool = False,
        stage : AgentStage = AgentStage.ONLINE
    ) -> BatchedNNCriticOutput:
        real_output = nn_output.output
        if not is_discrete:
            if nn_output.is_seq:
                real_output = real_output.flatten(start_dim=1)
            else:
                real_output = real_output.flatten(start_dim=0)
        
        return BatchedNNCriticOutput(
            critic_estimates=real_output,
            distributions=None,
            log_stds=None,
            final_states=nn_output.state,
            masks=nn_output.masks,
            is_seq=nn_output.is_seq,
            is_discrete=is_discrete
        )