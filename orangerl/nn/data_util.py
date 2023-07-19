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

from ..base import TransitionBatch, EnvironmentStep, EpisodeRollout, AgentOutput, AgentStage
from typing import Dict, Any, Tuple, Optional, Iterable, List, Union, Iterator
import torch
import numpy as np
from dataclasses import dataclass

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
class BatchedNNAgentDiscreteOutput(Iterable[NNAgentOutput]):
    action_probs: torch.Tensor
    states: Optional[Iterable[Any]] = None
    start: int = 0

    def __iter__(self) -> Iterator[NNAgentOutput]:
        length = self.action_probs.shape[0]
        iter_state = iter(self.states) if self.states is not None else None
        for i in range(length):
            sampled_idx = torch.multinomial(self.action_probs[i], 1, replacement = True)
            yield NNAgentOutput(
                action = sampled_idx + self.start,
                state = next(iter_state) if iter_state is not None else None,
                log_prob = torch.log(self.action_probs[i][sampled_idx])
            )


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

def transform_any_array_to_numpy(
    batch: Union[Iterable[Union[np.ndarray, torch.Tensor]], np.ndarray, torch.Tensor],
):
    if isinstance(batch, (np.ndarray, torch.Tensor)):
        if isinstance(batch, torch.Tensor):
            return batch.detach().cpu().numpy()
        else:
            return batch
    else:
        return np.stack([transform_any_array_to_numpy(b) for b in batch])

def transform_any_array_to_tensor(
    batch: Union[Iterable[Union[np.ndarray, torch.Tensor]], np.ndarray, torch.Tensor],
):
    # if isinstance(batch, (np.ndarray, torch.Tensor)):
    #     if isinstance(batch, np.ndarray):
    #         return torch.from_numpy(batch)
    #     else:
    #         return batch
    # else:
    #     return torch.stack([torch.from_numpy(obs) if isinstance(obs, np.ndarray) else obs for obs in batch], dim=0)
    return torch.asarray(batch)

def transform_transition_batch_to_torch_tensor(
    batch: TransitionBatch
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Transforms a TransitionBatch to NN friendly format
    returns (observations, actions, next_observations)
    """
    observations = []
    actions = []
    next_observations = []
    for transition in batch:
        assert isinstance(transition.observation, np.ndarray) or isinstance(transition.observation, torch.Tensor), "Observation must be a numpy array or torch tensor"
        assert isinstance(transition.action, np.ndarray) or isinstance(transition.action, torch.Tensor), "Action must be a numpy array or torch tensor"
        assert isinstance(transition.next_observation, np.ndarray) or isinstance(transition.next_observation, torch.Tensor), "Next observation must be a numpy array or torch tensor"

        observations.append(transition.observation)
        actions.append(transition.action)
        next_observations.append(transition.next_observation)

    observations = torch.stack([torch.from_numpy(obs) if isinstance(obs, np.ndarray) else obs for obs in observations], dim=0)
    actions = torch.stack([torch.from_numpy(act) if isinstance(act, np.ndarray) else act for act in actions], dim=0)
    next_observations = torch.stack([torch.from_numpy(obs) if isinstance(obs, np.ndarray) else obs for obs in next_observations], dim=0)
    return observations, actions, next_observations

def transform_episodes_to_torch_tensors(
    episodes: Iterable[EpisodeRollout]
) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Transforms a list of EpisodeRollout to NN friendly format
    returns Iterable(observations, actions, next_observations)
    """
    length_maps : Dict[int, List[EpisodeRollout]] = {}
    for episode in episodes:
        if episode.length not in length_maps:
            length_maps[episode.length] = []
        length_maps[episode.length].append(episode)
    for episode_length, episodes in length_maps.items():
        observations = []
        actions = []
        next_observations = []
        for episode in episodes:
            obs_ep, act_ep, next_obs_ep = transform_transition_batch_to_torch_tensor(episode)
            observations.append(obs_ep)
            actions.append(act_ep)
            next_observations.append(next_obs_ep)
        yield torch.stack(observations, dim=0), torch.stack(actions, dim=0), torch.stack(next_observations, dim=0)