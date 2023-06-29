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

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, Union, TypeVar, Generic
import numpy as np

_ObsST = TypeVar("_ObsST")
_ActST = TypeVar("_ActST")
@dataclass
class EnvironmentStep(Generic[_ObsST, _ActST]):
    observation: _ObsST
    action: _ActST
    next_observation: _ObsST
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]

_ObsBT = TypeVar("_ObsBT")
_ActBT = TypeVar("_ActBT")
@dataclass
class TransitionBatch(Generic[_ObsBT, _ActBT], Iterable[EnvironmentStep[_ObsBT, _ActBT]]):
    observations: Iterable[_ObsBT]
    actions: Iterable[_ActBT]
    next_observations: Iterable[_ObsBT]
    rewards: Iterable[float]
    terminations: Iterable[bool]
    truncations: Iterable[bool]
    infos: Iterable[Dict[str, Any]]
    length: int

    def __iter__(self) -> Iterator[EnvironmentStep[_ObsBT, _ActBT]]:
        iter_observations = iter(self.observations)
        iter_actions = iter(self.actions)
        iter_rewards = iter(self.rewards)
        iter_next_observations = iter(self.next_observations)
        iter_terminations = iter(self.terminations)
        iter_truncations = iter(self.truncations)
        iter_infos = iter(self.infos)
        for _ in range(self.length):
            yield EnvironmentStep(
                next(iter_observations),
                next(iter_actions),
                next(iter_next_observations),
                next(iter_rewards),
                next(iter_terminations),
                next(iter_truncations),
                next(iter_infos)
            )

    def concat(self, step : EnvironmentStep) -> "TransitionBatch[_ObsBT,_ActBT]":
        new_observations = list(self.observations)
        new_actions = list(self.actions)
        new_rewards = list(self.rewards)
        new_next_observations = list(self.next_observations)
        new_terminations = list(self.terminations)
        new_truncations = list(self.truncations)
        new_infos = list(self.infos)
        new_observations.append(step.observation)
        new_actions.append(step.action)
        new_rewards.append(step.reward)
        new_next_observations.append(step.next_observation)
        new_terminations.append(step.terminated)
        new_truncations.append(step.truncated)
        new_infos.append(step.info)
        return __class__(
            new_observations,
            new_actions,
            new_rewards,
            new_next_observations,
            new_terminations,
            new_truncations,
            new_infos,
            self.length + 1
        )

    def extend(self, batch: "TransitionBatch[_ObsBT, _ActBT]"):
        new_observations = list(self.observations) + list(batch.observations)
        new_actions = list(self.actions) + list(batch.actions)
        new_rewards = list(self.rewards) + list(batch.rewards)
        new_next_observations = list(self.next_observations) + list(batch.next_observations)
        new_terminations = list(self.terminations) + list(batch.terminations)
        new_truncations = list(self.truncations) + list(batch.truncations)
        new_infos = list(self.infos) + list(batch.infos)
        return TransitionBatch(
            new_observations,
            new_actions,
            new_rewards,
            new_next_observations,
            new_terminations,
            new_truncations,
            new_infos,
            self.length + batch.length
        )


    def sample(self, batch_size : int, randomness : np.random.Generator) -> "TransitionBatch[_ObsBT,_ActBT]":
        num_selected_per_index = np.zeros((self.length,), dtype=np.int32)
        num_selected = 0
        index_list = np.arange(self.length)
        while num_selected < batch_size:
            randomness.shuffle(index_list)
            if self.length <= batch_size - num_selected:
                num_selected_per_index[:] += 1
                num_selected += self.length
            else:
                num_selected_per_index[index_list[:batch_size - num_selected]] += 1
                num_selected += batch_size - num_selected
        new_observations = []
        new_actions = []
        new_rewards = []
        new_next_observations = []
        new_terminations = []
        new_truncations = []
        new_infos = []
        for idx, step in enumerate(self):
            for _ in range(int(num_selected_per_index[idx])):
                new_observations.append(step.observation)
                new_actions.append(step.action)
                new_rewards.append(step.reward)
                new_next_observations.append(step.next_observation)
                new_terminations.append(step.terminated)
                new_truncations.append(step.truncated)
                new_infos.append(step.info)
        
        shuffled_indices = np.arange(batch_size)
        randomness.shuffle(shuffled_indices)
        ret_observations = []
        ret_actions = []
        ret_rewards = []
        ret_next_observations = []
        ret_terminations = []
        ret_truncations = []
        ret_infos = []
        for idx in shuffled_indices:
            ret_observations.append(new_observations[idx])
            ret_actions.append(new_actions[idx])
            ret_rewards.append(new_rewards[idx])
            ret_next_observations.append(new_next_observations[idx])
            ret_terminations.append(new_terminations[idx])
            ret_truncations.append(new_truncations[idx])
            ret_infos.append(new_infos[idx])
        return __class__(
            ret_observations,
            ret_actions,
            ret_rewards,
            ret_next_observations,
            ret_terminations,
            ret_truncations,
            ret_infos,
            batch_size
        )
        

        
    
_ObsRT = TypeVar("_ObsRT")
_ActRT = TypeVar("_ActRT")
@dataclass
class EpisodeRollout(Generic[_ObsRT, _ActRT], TransitionBatch[_ObsRT, _ActRT]):
    observations: Iterable[_ObsRT]
    actions: Iterable[_ActRT]
    rewards: Iterable[float]
    end_termination: bool
    end_truncation: bool
    infos: Iterable[Dict[str, Any]]

    @property
    def terminations(self) -> Iterable[bool]:
        for _ in range(len(self.observations) - 1):
            yield False
        yield self.end_termination
    
    @property
    def truncations(self) -> Iterable[bool]:
        for _ in range(len(self.observations) - 1):
            yield False
        yield self.end_truncation
    