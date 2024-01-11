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
        return BatchedNNCriticOutput(
            critic_estimates=nn_output.output,
            distributions=None,
            log_stds=None,
            final_states=nn_output.state,
            masks=nn_output.masks,
            is_seq=nn_output.is_seq,
            is_discrete=is_discrete
        )