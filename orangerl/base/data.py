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
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, Union, TypeVar, Generic, List
import numpy as np

_ObsST = TypeVar("_ObsST")
_ActST = TypeVar("_ActST")
class EnvironmentStep(Generic[_ObsST, _ActST]):
    observation: _ObsST
    action: _ActST
    next_observation: _ObsST
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]

    def __init__(
        self,
        observation: _ObsST,
        action: _ActST,
        next_observation: _ObsST,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any]
    ) -> None:
        self.observation = observation
        self.action = action
        self.next_observation = next_observation
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info

_ObsBT = TypeVar("_ObsBT")
_ActBT = TypeVar("_ActBT")
class TransitionBatch(Generic[_ObsBT, _ActBT], Iterable[EnvironmentStep[_ObsBT, _ActBT]]):
    observations: Iterable[_ObsBT]
    actions: Iterable[_ActBT]
    next_observations: Iterable[_ObsBT]
    rewards: Iterable[float]
    terminations: Iterable[bool]
    truncations: Iterable[bool]
    infos: Iterable[Dict[str, Any]]
    length: int

    def __init__(
        self,
        observations: Iterable[_ObsBT],
        actions: Iterable[_ActBT],
        next_observations: Iterable[_ObsBT],
        rewards: Iterable[float],
        terminations: Iterable[bool],
        truncations: Iterable[bool],
        infos: Iterable[Dict[str, Any]],
        length: int
    ) -> None:
        self.observations = observations
        self.actions = actions
        self.next_observations = next_observations
        self.rewards = rewards
        self.terminations = terminations
        self.truncations = truncations
        self.infos = infos
        self.length = length

    def __len__(self) -> int:
        return self.length

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

    def sample_sequences(
        self,
        num_sequence: int = -1,
        transition_length_limit: int = -1,
        cut_transition_limit_sequences: bool = False,
        randomness: np.random.Generator = np.random.default_rng()
    ) -> List["EpisodeRollout"]:
        assert num_sequence > 0 or transition_length_limit > 0
        assert not (num_sequence > 0 and transition_length_limit > 0)
        sequences_start_idx = [0]
        for idx, truncated, terminated in zip(range(self.length), self.truncations, self.terminations):
            if truncated or terminated and idx + 1 < self.length:
                sequences_start_idx.append(idx + 1)
        
        assert num_sequence <= 0 or len(sequences_start_idx) <= num_sequence
        assert transition_length_limit <= 0 or self.length <= transition_length_limit
        
        if num_sequence > 0:
            randomness.shuffle(sequences_start_idx)
            cut_sequences_start_idx = sequences_start_idx[:num_sequence]
            cut_sequences_start_idx.sort()
            ret_sequences : List[EpisodeRollout] = []
            for idx, transition in enumerate(self):
                if idx in cut_sequences_start_idx:
                    current_sequence = EpisodeRollout(
                        [transition.observation],
                        [transition.action],
                        [transition.reward],
                        False,
                        False,
                        [transition.info]
                    )
                    ret_sequences.append(current_sequence)
                elif len(ret_sequences) == 0:
                    continue
                else:
                    current_sequence = ret_sequences[-1]
                    if current_sequence.end_termination or current_sequence.end_truncation:
                        break
                    current_sequence.observations.append(transition.observation)
                    current_sequence.actions.append(transition.action)
                    current_sequence.rewards.append(transition.reward)
                    current_sequence.infos.append(transition.info)
                    if transition.terminated:
                        current_sequence.end_termination = True
                    if transition.truncated:
                        current_sequence.end_truncation = True
            return ret_sequences
        else:
            # Sample from sequences with limit transition length
            # First make a list of sequences start idx with their length information
            sequences_start_idx_with_length = []
            for i in range(len(sequences_start_idx)):
                if i + 1 < len(sequences_start_idx):
                    sequences_start_idx_with_length.append((sequences_start_idx[i], sequences_start_idx[i + 1] - sequences_start_idx[i]))
                else:
                    sequences_start_idx_with_length.append((sequences_start_idx[i], self.length - sequences_start_idx[i]))
            # Shuffle the list
            randomness.shuffle(sequences_start_idx_with_length)
            # We only need to take sequences that satisfies the transition length limit
            cut_sequences_start_idx = []
            accumulated_length = 0
            for start_idx, length in sequences_start_idx_with_length:
                if accumulated_length <= transition_length_limit:
                    cut_sequences_start_idx.append(start_idx)
                    accumulated_length += length
                else:
                    break
            
            # Sort the list
            cut_sequences_start_idx.sort()

            transition_length = 0

            ret_sequences : List[EpisodeRollout] = []
            for idx, transition in enumerate(self):
                if transition_length >= transition_length_limit and cut_transition_limit_sequences:
                    break
                if idx in cut_sequences_start_idx:
                    transition_length += 1
                    current_sequence = EpisodeRollout(
                        [transition.observation],
                        [transition.action],
                        [transition.reward],
                        False,
                        False,
                        [transition.info]
                    )
                    ret_sequences.append(current_sequence)
                elif len(ret_sequences) == 0:
                    continue
                else:
                    current_sequence = ret_sequences[-1]
                    if current_sequence.end_termination or current_sequence.end_truncation:
                        break
                    transition_length += 1
                    current_sequence.observations.append(transition.observation)
                    current_sequence.actions.append(transition.action)
                    current_sequence.rewards.append(transition.reward)
                    current_sequence.infos.append(transition.info)
                    if transition.terminated:
                        current_sequence.end_termination = True
                    if transition.truncated:
                        current_sequence.end_truncation = True
            return ret_sequences

    def sample(self, batch_size : int, randomness : np.random.Generator = np.random.default_rng()) -> "TransitionBatch[_ObsBT,_ActBT]":
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
        return TransitionBatch(
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
class EpisodeRollout(Generic[_ObsRT, _ActRT], TransitionBatch[_ObsRT, _ActRT]):
    observations: Iterable[_ObsRT]
    actions: Iterable[_ActRT]
    rewards: Iterable[float]
    end_termination: bool
    end_truncation: bool
    infos: Iterable[Dict[str, Any]]

    def __init__(
        self,
        observations: Iterable[_ObsRT],
        actions: Iterable[_ActRT],
        rewards: Iterable[float],
        end_termination: bool,
        end_truncation: bool,
        infos: Iterable[Dict[str, Any]]
    ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.end_termination = end_termination
        self.end_truncation = end_truncation
        self.infos = infos

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
    