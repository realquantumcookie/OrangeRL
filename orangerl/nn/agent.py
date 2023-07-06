from ..base.agent import Agent, AgentOutput, AgentStage
from ..base.data import TransitionBatch, EnvironmentStep
from abc import abstractmethod, ABC
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Iterator, Optional, Union, Iterable, Tuple, Dict
from collections import OrderedDict
from dataclasses import dataclass
import gymnasium as gym

NNAgentOutput = AgentOutput[torch.Tensor, Any, torch.Tensor]

class BatchedNNAgentDictStateWrapper():
    batched_dict_state: Dict[str, Any]
    def __iter__(self) -> Iterator[Any]:
        constructed_iterator_dict = {}
        for key in self.batched_dict_state.keys():
            if isinstance(self.batched_dict_state, dict):
                def lambda_iter():
                    yield from __class__(self.batched_dict_state[key])
                constructed_iterator_dict[key] = lambda_iter()
            elif isinstance(self.batched_dict_state[key], Iterable):
                constructed_iterator_dict[key] = iter(self.batched_dict_state[key])
            else:
                raise ValueError("BatchedNNAgentDictStateWrapper only accepts dict of iterables")
        while True:
            constructed_iterator_dict = {}
            for key in self.batched_dict_state.keys():
                constructed_iterator_dict[key] = next(constructed_iterator_dict[key])
            yield constructed_iterator_dict

@dataclass
class BatchedNNAgentDeterministicOutput(Iterable[NNAgentOutput]):
    actions: torch.Tensor
    states: Optional[Iterable[Any]] = None
    log_probs: torch.Tensor
    
    def __iter__(self) -> Iterator[NNAgentOutput]:
        length = self.actions.shape[0]
        iter_state = iter(self.states) if self.states is not None else None
        for i in range(length):
            yield NNAgentOutput(
                action = self.actions[i],
                state = next(iter_state) if iter_state is not None else None,
                log_prob = self.log_probs[i]
            )

@dataclass
class BatchedNNAgentStochasticOutput(Iterable[NNAgentOutput]):
    action_dist: torch.distributions.Distribution
    states: Optional[Iterable[Any]] = None
    stage: AgentStage = AgentStage.ONLINE

    def __iter__(self) -> Iterator[NNAgentOutput]:
        length = self.action_dist.batch_shape[0]
        
        if self.stage == AgentStage.EVAL:
            try:
                actions = self.action_dist.mean
            except NotImplementedError:
                actions = self.action_dist.rsample()
        else:
            actions = self.action_dist.rsample()
        
        log_probs = self.action_dist.log_prob(actions)
        log_probs_sum = log_probs.view(length, -1).sum(dim=1)
        
        iter_state = iter(self.states) if self.states is not None else None

        for i in range(length):
            yield NNAgentOutput(
                action = actions[i],
                state = next(iter_state) if iter_state is not None else None,
                log_prob = log_probs_sum[i]
            )

class NNAgentActionMapper(ABC, nn.Module):
    def __init__(self, action_space : gym.spaces.Box) -> None:
        super().__init__()
        self.action_space = action_space
    
    @abstractmethod
    def forward(self, output : Union[torch.Tensor, Tuple[torch.Tensor, Any]], stage : AgentStage = AgentStage.ONLINE) -> Union[BatchedNNAgentStochasticOutput, BatchedNNAgentDeterministicOutput]:
        pass

    @staticmethod
    def _wrap_state(state : Any) -> Iterable[Any]:
        if isinstance(state, dict):
            return BatchedNNAgentDictStateWrapper(state)
        elif isinstance(state, Iterable):
            return state
        else:
            raise ValueError("NNAgentActionMapper only accepts dict of iterables")


class NNAgent(Agent[Union[np.ndarray,torch.Tensor], Union[np.ndarray, torch.Tensor], Any], nn.Module, ABC):
    def __init__(self, action_mapper : NNAgentActionMapper, device : Optional[Union[torch.device,str]] = None) -> None:
        super().__init__()
        self.action_mapper = action_mapper.to(device)
        self.device = device
        self.to(device)

    def map_action_batch(self, forward_output : Union[Any, Tuple[Any, Any]], stage : AgentStage = AgentStage.ONLINE) -> Union[BatchedNNAgentStochasticOutput, BatchedNNAgentDeterministicOutput]:
        return self.action_mapper(forward_output, stage=stage)

    """
    Forward pass of the actor
    param batch: The batch of observations, shaped (batch_size, *observation_shape)
    param state: The batch of states
    param stage: The stage of the agent (ONLINE, OFFLINE, EVAL, EXPLORE)
    param kwargs: Additional keyword arguments
    return: The batch of actions, shaped (batch_size, *), followed by an optional state variable
    """
    @abstractmethod
    def forward(
        self,
        batch: torch.Tensor,
        state: Optional[Any] = None,
        stage: AgentStage = AgentStage.ONLINE,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        pass

    def add_transitions(
        self, 
        transition : Union[TransitionBatch[Union[np.ndarray,torch.Tensor], Union[np.ndarray,torch.Tensor]], EnvironmentStep[Union[np.ndarray,torch.Tensor], Union[np.ndarray,torch.Tensor]]],
        stage : AgentStage = AgentStage.ONLINE
    ):
        pass

    @abstractmethod
    def update(self, batch_size : Optional[int] = None, *args, **kwargs) -> Dict[str, Any]:
        pass

    def get_action_batch(
        self, 
        observations : Union[np.ndarray,torch.Tensor, Iterable[Union[np.ndarray,torch.Tensor]]], 
        states : Optional[Any] = None, 
        stage : AgentStage = AgentStage.ONLINE
    ) -> Iterable[AgentOutput[torch.Tensor, Optional[torch.Tensor]]]:
        input_obs = self._batch_to_tensor(observations)
        input_state = states #self._batch_to_tensor(state) if state is not None else None
        output = self.forward(input_obs, state = input_state, stage = stage)
        return self.map_action_batch(output, stage)
    
    @staticmethod
    def _batch_to_tensor(
        batch: Union[Iterable[Union[np.ndarray, torch.Tensor]], np.ndarray, torch.Tensor],
    ):
        if isinstance(batch, (np.ndarray, torch.Tensor)):
            if isinstance(batch, np.ndarray):
                return torch.from_numpy(batch)
            else:
                return batch
        else:
            return torch.stack([torch.from_numpy(obs) if isinstance(obs, np.ndarray) else obs for obs in batch])
        
    @staticmethod
    def _batch_to_np(
        batch: Union[Iterable[Union[np.ndarray, torch.Tensor]], np.ndarray, torch.Tensor],
    ):
        if isinstance(batch, (np.ndarray, torch.Tensor)):
            if isinstance(batch, torch.Tensor):
                return batch.detach().cpu().numpy()
            else:
                return batch
        else:
            return np.stack([obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor) else obs for obs in batch])