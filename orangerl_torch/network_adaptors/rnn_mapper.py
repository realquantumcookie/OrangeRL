from typing import Any, Dict, List, Optional, Tuple
from orangerl import AgentStage
from orangerl_torch import NNAgentNetworkAdaptor, Tensor_Or_TensorDict
import torch
import torch.nn as nn

from orangerl_torch.agent_util import NNAgentNetworkOutput

class RNNNetworkAdaptor(NNAgentNetworkAdaptor):
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
        assert masks is None or masks.ndim == 2, "masks must be 2 dimensional"

        concatenated_input = obs_batch.flatten(start_dim=2) if act_batch is None else torch.cat([
            obs_batch.flatten(start_dim=2),
            act_batch.flatten(start_dim=2)
        ], dim=2)
        if masks is not None:
            lengths = masks.flatten(start_dim=1).sum(dim=1).int().cpu()
            input = nn.utils.rnn.pack_padded_sequence(
                concatenated_input, 
                lengths, 
                batch_first=True, 
                enforce_sorted=False
            )
        else:
            input = concatenated_input # (N, L, H_in)

        if state is not None:
            state_input = state.transpose(0, 1) # (N, D * num_layers, H_out) => (D * num_layers, N, H_out)
        else:
            state_input = None

        return [input, state_input], {}
        
    def map_net_output(
        self,
        output : Tuple[torch.Tensor, torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        state: Optional[Tensor_Or_TensorDict] = None,
        is_update = False,
        stage : AgentStage = AgentStage.ONLINE,
    ) -> NNAgentNetworkOutput:
        net_out, state_out = output
        if isinstance(net_out, nn.utils.rnn.PackedSequence):
            net_out, lengths = nn.utils.rnn.pad_packed_sequence(net_out, batch_first=True) # (N, L, H_out) and (N, )
        
        state_out = state_out.transpose(0, 1) # (D * num_layers, N, H_out) => (N, D * num_layers, H_out)
        
        return NNAgentNetworkOutput(
            output=net_out,
            masks=masks,
            state=state_out,
            is_seq=True
        )