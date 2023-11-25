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
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, Union, TypeVar, Generic, List, SupportsFloat, MutableSequence, Sequence
from abc import abstractmethod, abstractproperty, ABC
import numpy as np

_ObsST = TypeVar("_ObsST")
_ActST = TypeVar("_ActST")
class EnvironmentStep(Generic[_ObsST, _ActST]):
    observation: _ObsST
    action: _ActST
    next_observation: _ObsST
    reward: SupportsFloat
    terminated: bool
    truncated: bool
    info: Optional[Dict[str, Any]]

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

    def __repr__(self) -> str:
        return f"EnvironmentStep(observation={self.observation}, action={self.action}, next_observation={self.next_observation}, reward={self.reward}, terminated={self.terminated}, truncated={self.truncated}, info={self.info})"

    def __eq__(self, __value: object) -> bool:
        return __value is self

class TransitionBatch(Iterable[EnvironmentStep[_ObsST, _ActST]], Generic[_ObsST, _ActST], ABC):
    is_single_episode: bool
    is_time_sorted: bool

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __iter__(self) -> Iterable[EnvironmentStep[_ObsST, _ActST]]:
        pass

    def __repr__(self) -> str:
        limit_repr_len = 10

        transition_batch_name = __class__.__name__
        if self.is_single_episode:
            transition_batch_name = "[Episodic]" + transition_batch_name
        if self.is_time_sorted:
            transition_batch_name = "[Sorted]" + transition_batch_name
        ret = (
            self.__class__.__name__ + ":" + transition_batch_name if self.__class__ is not __class__ else transition_batch_name
        ) + "("

        repr_len = 1
        for environment_step in self:
            ret += str(environment_step) + ", "
            if repr_len >= limit_repr_len:
                ret += "..."
                break
            repr_len += 1
        ret += ")"
        return ret

class TransitionSequence(TransitionBatch[_ObsST, _ActST], Sequence[EnvironmentStep[_ObsST, _ActST]], Generic[_ObsST, _ActST], ABC):
    is_single_episode: bool
    is_time_sorted: bool

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __iter__(self) -> Iterable[EnvironmentStep[_ObsST, _ActST]]:
        pass
    
    @abstractmethod
    def __getitem__(self, index: Union[int, slice, np.ndarray]) -> Union[EnvironmentStep[_ObsST, _ActST], Sequence[EnvironmentStep[_ObsST, _ActST]]]:
        pass
    
    @abstractmethod
    def __add__(self, other : Union[EnvironmentStep[_ObsST, _ActST], Iterable[EnvironmentStep[_ObsST, _ActST]]]) -> "TransitionSequence[_ObsST,_ActST]":
        pass

    def __radd__(self, other : Union[EnvironmentStep[_ObsST, _ActST], Iterable[EnvironmentStep[_ObsST, _ActST]]]) -> "TransitionSequence[_ObsST,_ActST]":
        return self.__add__(other)

    @abstractmethod
    def sample_from_index(
        self,
        index: np.ndarray,
    ) -> Sequence[EnvironmentStep[_ObsST, _ActST]]:
        pass

class TransitionSampler(Generic[_ObsST, _ActST], Iterable[EnvironmentStep[_ObsST, _ActST]], ABC):
    @abstractmethod
    def seed(self, seed : Optional[int] = None) -> None:
        pass
    
    @abstractmethod
    def sample(
        self,
        transition_batch : TransitionSequence[_ObsST, _ActST],
        batch_size : int,
        repeat_sample = True,
        **kwargs
    ) -> np.ndarray:
        pass

class TransitionReplayBuffer(TransitionSequence[_ObsST, _ActST], MutableSequence[EnvironmentStep[_ObsST, _ActST]], Generic[_ObsST, _ActST], ABC):
    is_single_episode : bool = False
    is_time_sorted : bool = True
    capacity : Optional[int] = None
    sampler: TransitionSampler[_ObsST, _ActST]
    sample_batch_size : int
    sample_repeat : bool

    @abstractmethod
    def __setitem__(self, index: Union[int, slice, np.ndarray], value: Union[EnvironmentStep[_ObsST, _ActST], Iterable[EnvironmentStep[_ObsST, _ActST]]]) -> None:
        pass

    @abstractmethod
    def __delitem__(self, index: Union[int, slice, np.ndarray]) -> None:
        pass

    @abstractmethod
    def append(self, step : EnvironmentStep[_ObsST, _ActST]) -> None:
        """
        Add a single step to the buffer
        """
        pass

    @abstractmethod
    def extend(self, steps : Iterable[EnvironmentStep[_ObsST, _ActST]]) -> None:
        """
        Add multiple steps to the buffer
        """
        pass

    def __iadd__(self, values: Union[EnvironmentStep[_ObsST, _ActST], Iterable[EnvironmentStep[_ObsST, _ActST]]]) -> "TransitionReplayBuffer[_ObsST,_ActST]":
        if isinstance(values, EnvironmentStep):
            self.append(values)
        else:
            self.extend(values)
        return self

    def sample(
        self,
        **kwargs
    ) -> Sequence[EnvironmentStep[_ObsST, _ActST]]:
        idx = self.sampler.sample(self, self.sample_batch_size, self.sample_repeat, **kwargs)
        return self.sample_from_index(idx)