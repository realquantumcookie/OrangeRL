import torch
import torch.nn as nn
from typing import List, Callable, Optional

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

class ReshapeNNLayer(nn.Module):
    def __init__(self, shape : torch.Size):
        super().__init__()
        self.shape = shape
    
    def forward(self, x : torch.Tensor):
        real_shape = [x.size(0)] + list(self.shape)
        return x.view(real_shape)