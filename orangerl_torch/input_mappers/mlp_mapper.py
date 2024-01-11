from typing import Any, Dict, List, Optional, Tuple
from orangerl import AgentStage
from orangerl_torch import NNAgentInputMapper, Tensor_Or_TensorDict
import torch
import torch.nn as nn


class MLPInputMapper(NNAgentInputMapper):
    def forward(
        self, 
        obs_batch: Tensor_Or_TensorDict, 
        act_batch: Optional[Tensor_Or_TensorDict], 
        masks: Optional[torch.Tensor] = None, 
        state: Optional[Tensor_Or_TensorDict] = None, 
        is_seq=False, 
        is_update=False, 
        stage: AgentStage = AgentStage.ONLINE
    ) -> Tuple[List[Any], Dict[str, Any]]:
        if is_seq:
            raise NotImplementedError("MLPInputMapper does not support sequence models")
        
        assert masks is None or masks.ndim == 1, "masks must be 1 dimensional"
        assert state is None, "MLPInputMapper does not support stateful models"

        obs_input = obs_batch[masks] if masks is not None else obs_batch
        act_input = act_batch[masks] if masks is not None and act_batch is not None else act_batch
        obs_input = obs_input.flatten(start_dim=1)
        act_input = act_input.flatten(start_dim=1) if act_input is not None else None
        concatenated_input = obs_input if act_input is None else torch.cat([obs_input, act_input], dim=1)
        return [concatenated_input], {}
        