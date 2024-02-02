from typing import Any, Dict, List, Optional, Tuple
from orangerl import AgentStage
from orangerl_torch import NNAgentNetworkAdaptor, Tensor_Or_TensorDict
import torch
import torch.nn as nn

from orangerl_torch.agent_util import NNAgentNetworkOutput

class DirectNetworkAdaptor(NNAgentNetworkAdaptor):
    """
    Network adaptor for RNN models. 
    Note that this adaptor requires the RNN layers to be specified with batch_first=True.
    """

    is_seq : bool = True

    def forward(
        self, 
        obs_batch: Tensor_Or_TensorDict, 
        act_batch: Optional[Tensor_Or_TensorDict], 
        masks: Optional[torch.Tensor] = None, 
        state: Optional[Tensor_Or_TensorDict] = None, 
        is_update=False, 
        stage: AgentStage = AgentStage.ONLINE
    ) -> Tuple[List[Any], Dict[str, Any]]:
        return [], {
            "obs_batch": obs_batch,
            "act_batch": act_batch,
            "masks": masks,
            "state": state,
            "is_update": is_update,
            "stage": stage
        }
        
    def map_net_output(
        self,
        output : NNAgentNetworkOutput,
        masks: Optional[torch.Tensor] = None,
        state: Optional[Tensor_Or_TensorDict] = None,
        is_update = False,
        stage : AgentStage = AgentStage.ONLINE,
    ) -> NNAgentNetworkOutput:
        return output