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

from ..base import TransitionBatch, EnvironmentStep, EpisodeRollout
from typing import Dict, Any, Tuple, Optional, Iterable, List, Union
import torch
import numpy as np

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