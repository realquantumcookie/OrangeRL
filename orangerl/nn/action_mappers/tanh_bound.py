from typing import Union, Tuple, Any
import gymnasium as gym
import torch
import numpy as np
from ...base.agent import AgentStage
from ..agent import NNAgentActionMapper, BatchedNNAgentDeterministicOutput, BatchedNNAgentStochasticOutput, BatchedNNAgentDictStateWrapper

class NNAgentTanhActionMapper(NNAgentActionMapper):
    def __init__(self, action_space: gym.spaces.Box) -> None:
        super().__init__(action_space)
        assert np.all(np.isfinite(action_space.low)) and np.all(np.isfinite(action_space.high)), "Action space must be bounded"

    """
    Forward pass of the action mapper
    param output: Output from the network, action shaped (batch_size, (*action_shape), 2) and state can be shaped whatever
    return: The batch of actions, shaped (batch_size, (*action_shape))
    """
    def forward(self, output : Union[torch.Tensor, Tuple[torch.Tensor, Any]], stage : AgentStage = AgentStage.ONLINE) -> BatchedNNAgentStochasticOutput:
        if isinstance(output, tuple):
            batch, states = output
        else:
            batch = output
            states = None
        
        std = torch.exp(batch[...,1])
        dist = torch.distributions.Normal(batch[...,0], std)
        
        transforms = [
            torch.distributions.transforms.TanhTransform(),
            torch.distributions.transforms.AffineTransform(loc = self.action_space.low[np.newaxis,:], scale = (self.action_space.high - self.action_space.low)[np.newaxis, :])
        ]
        
        transformed_dist = torch.distributions.TransformedDistribution(dist, transforms)
        return BatchedNNAgentStochasticOutput(
            action_dist = transformed_dist,
            states=self._wrap_state(states) if states is not None else None,
            stage=stage
        )