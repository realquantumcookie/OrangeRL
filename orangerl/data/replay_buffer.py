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

from ..base import TransitionBatch, EnvironmentStep
from typing import Any, TypeVar, Generic, Union, Dict, List

_ObsT = TypeVar("_ObsT")
_ActT = TypeVar("_ActT")
class ReplayBuffer(Generic[_ObsT, _ActT], TransitionBatch[_ObsT, _ActT]):
    def __init__(self, capacity : int = 100_000) -> None:
        TransitionBatch.__init__(self)
        self.observations: List[_ObsT] = []
        self.actions: List[_ActT] = []
        self.rewards: List[float] = []
        self.next_observations: List[_ObsT] = []
        self.terminations: List[bool] = []
        self.truncations: List[bool] = []
        self.infos: List[Dict[str, Any]] = []
        self._capacity = capacity
        self._next_idx = 0
    
    @property
    def length(self) -> int:
        return len(self.actions)

    @property
    def capacity(self) -> int:
        return self._capacity
    
    @capacity.setter
    def capacity(self, capacity : int) -> None:
        if capacity < self._capacity:
            self._next_idx = 0
        self._capacity = capacity

    def add_transition(self, transition : Union[TransitionBatch[_ObsT, _ActT], EnvironmentStep[_ObsT, _ActT]]) -> None:
        if isinstance(transition, TransitionBatch):
            remaining_capacity_to_add = transition.length
            if self.length < self.capacity:
                current_to_add = min(remaining_capacity_to_add, self.capacity - self.length)
                self.observations.extend(transition.observations[:current_to_add])
                self.actions.extend(transition.actions[:current_to_add])
                self.rewards.extend(transition.rewards[:current_to_add])
                self.next_observations.extend(transition.next_observations[:current_to_add])
                self.terminations.extend(transition.terminations[:current_to_add])
                self.truncations.extend(transition.truncations[:current_to_add])
                self.infos.extend(transition.infos[:current_to_add])
                remaining_capacity_to_add -= current_to_add
            while remaining_capacity_to_add > 0:
                current_to_add = min(remaining_capacity_to_add, self.capacity - self._next_idx)
                start = transition.length - remaining_capacity_to_add
                end = start + current_to_add
                self.observations[self._next_idx:self._next_idx + current_to_add] = transition.observations[start:end]
                self.actions[self._next_idx:self._next_idx + current_to_add] = transition.actions[start:end]
                self.rewards[self._next_idx:self._next_idx + current_to_add] = transition.rewards[start:end]
                self.next_observations[self._next_idx:self._next_idx + current_to_add] = transition.next_observations[start:end]
                self.terminations[self._next_idx:self._next_idx + current_to_add] = transition.terminations[start:end]
                self.truncations[self._next_idx:self._next_idx + current_to_add] = transition.truncations[start:end]
                self.infos[self._next_idx:self._next_idx + current_to_add] = transition.infos[start:end]
                self._next_idx = (self._next_idx + current_to_add) % self.capacity
                remaining_capacity_to_add -= current_to_add

        elif isinstance(transition, EnvironmentStep):
            if self.length >= self.capacity:
                self.observations[self._next_idx] = transition.observation
                self.actions[self._next_idx] = transition.action
                self.rewards[self._next_idx] = transition.reward
                self.next_observations[self._next_idx] = transition.next_observation
                self.terminations[self._next_idx] = transition.terminated
                self.truncations[self._next_idx] = transition.truncated
                self.infos[self._next_idx] = transition.info
                self._next_idx = (self._next_idx + 1) % self.capacity
            else:
                self.observations.append(transition.observation)
                self.actions.append(transition.action)
                self.rewards.append(transition.reward)
                self.next_observations.append(transition.next_observation)
                self.terminations.append(transition.terminated)
                self.truncations.append(transition.truncated)
                self.infos.append(transition.info)
        else:
            raise TypeError("Invalid transition type")
