import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from .agent_util import NNAgentNetworkOutput
from typing import List, Callable, Optional, Union, Tuple
from orangerl import AgentStage

def MLP(
    input_dim: int,
    output_dim : int,
    hidden_dims : List[int],
    activations : Optional[Callable[[int, int], nn.Module]] = lambda prev_dim, next_dim: nn.ReLU(),
    last_layer_activation : Optional[Callable[[int, int], nn.Module]] = None,
    use_layer_norm : bool = False,
    dropout_rate : Optional[float] = None,
):
    next_dims = hidden_dims + [output_dim]
    
    all_layers = [nn.Flatten()]
    prev_dim = input_dim
    for i, next_dim in enumerate(next_dims):
        all_layers.append(
            nn.Linear(prev_dim, next_dim)
        )

        if i < len(next_dims) - 1:
            if dropout_rate is not None and dropout_rate > 0:
                all_layers.append(nn.Dropout(dropout_rate))
            if use_layer_norm:
                all_layers.append(nn.LayerNorm(next_dim))
            if activations is not None:
                all_layers.append(activations(prev_dim, next_dim))
        else:
            if last_layer_activation is not None:
                all_layers.append(last_layer_activation(prev_dim, next_dim))
        
        prev_dim = next_dim
    
    return nn.Sequential(*all_layers)

class RNNMLP(nn.Module):
    def __init__(
        self,
        input_dim : int,
        rnn_hidden_dim : int,
        rnn_layers : int,
        output_dim : int,
        mlp_hidden_dims : List[int],
        dropout_rate_rnn : Optional[float] = None,
        activations_mlp : Optional[Callable[[int, int], nn.Module]] = lambda prev_dim, next_dim: nn.ReLU(),
        last_layer_activation : Optional[Callable[[int, int], nn.Module]] = None,
        use_layer_norm_mlp : bool = False,
        dropout_rate_mlp : Optional[float] = None,
    ):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout_rate_rnn if dropout_rate_rnn is not None else 0.0,
            bidirectional=False
        )
        self.mlp = MLP(
            input_dim=rnn_hidden_dim,
            output_dim=output_dim,
            hidden_dims=mlp_hidden_dims,
            activations=activations_mlp,
            last_layer_activation=last_layer_activation,
            use_layer_norm=use_layer_norm_mlp,
            dropout_rate=dropout_rate_mlp
        )
        self.empty_state = torch.zeros((
            rnn_layers, rnn_hidden_dim
        ), dtype=torch.float32)
    
    def forward(
        self,
        obs_batch: torch.Tensor, 
        act_batch: Optional[torch.Tensor], 
        masks: Optional[torch.Tensor] = None, 
        state: Optional[torch.Tensor] = None, 
        is_update=False, 
        stage: AgentStage = AgentStage.ONLINE
    ) -> NNAgentNetworkOutput:
        rnn_input, rnn_in_state = self.map_input(obs_batch, act_batch, masks, state, is_update, stage)
        rnn_output, rnn_out_state = self.rnn.forward(
            rnn_input, 
            rnn_in_state
        )
        if isinstance(rnn_output, PackedSequence):
            rnn_output, rnn_output_lengths = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)

        mlp_input = rnn_output.reshape(rnn_output.size(0) * rnn_output.size(1), -1)
        mlp_output : torch.Tensor = self.mlp.forward(mlp_input)
        mlp_output = mlp_output.reshape(rnn_output.size(0), rnn_output.size(1), -1)
        return NNAgentNetworkOutput(
            output=mlp_output,
            masks=masks,
            state=rnn_out_state.transpose(0, 1), # (D * num_layers, N, H_out) => (N, D * num_layers, H_out)
            is_seq=True
        )

    @staticmethod
    def map_input(
        obs_batch: torch.Tensor, 
        act_batch: Optional[torch.Tensor], 
        masks: Optional[torch.Tensor] = None, 
        state: Optional[torch.Tensor] = None, 
        is_update=False, 
        stage: AgentStage = AgentStage.ONLINE
    ) -> Tuple[Union[torch.Tensor, PackedSequence], Optional[torch.Tensor]]:
        """
        Maps input to the format that is expected by the RNN model
        Taken from the rnn_mapper
        """
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

        return (input, state_input)
        

class ReshapeNNLayer(nn.Module):
    def __init__(self, shape : torch.Size):
        super().__init__()
        self.shape = shape
    
    def forward(self, x : torch.Tensor):
        real_shape = [x.size(0)] + list(self.shape)
        return x.view(real_shape)