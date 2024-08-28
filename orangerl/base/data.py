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
from abc import abstractmethod, ABC
import numpy as np
from .common import Savable, Serializable
from os import PathLike

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

class TransitionIterable(Generic[_ObsST, _ActST], ABC):
    @abstractmethod
    def iter_transitions(self) -> Iterator[EnvironmentStep[_ObsST, _ActST]]:
        pass

class TransitionSequence(TransitionIterable[_ObsST, _ActST], Generic[_ObsST, _ActST], ABC):
    transition_len : int

    @staticmethod
    def from_iterable(
        iterable: Iterable[EnvironmentStep[_ObsST, _ActST]]
    ) -> "TransitionSequence[_ObsST, _ActST]":
        return TransitionSequenceListImpl(iterable)

    def iter_transitions(self) -> Iterator[EnvironmentStep[_ObsST, _ActST]]:
        for i in range(self.transition_len):
            yield self.transitions_at(i)
    
    @abstractmethod
    def transitions_at(self, index: Union[int, slice, Sequence[int]]) -> Union[EnvironmentStep[_ObsST, _ActST], "TransitionSequence[_ObsST, _ActST]"]:
        pass

class TransitionSequenceListImpl(TransitionSequence[_ObsST, _ActST], Generic[_ObsST, _ActST], MutableSequence[EnvironmentStep[_ObsST, _ActST]]):
    def __init__(
        self,
        iterable: Iterable[EnvironmentStep[_ObsST, _ActST]]
    ):
        self._transitions = list(iterable)
    
    @property
    def transition_len(self) -> int:
        return len(self._transitions)
    
    def __len__(self) -> int:
        return self.transition_len
    
    def __getitem__(self, index: Union[int, slice, Sequence[int]]) -> Union[EnvironmentStep[_ObsST, _ActST], "TransitionSequence[_ObsST, _ActST]"]:
        return self.transitions_at(index)
    
    def __setitem__(self, index: Union[int, slice, Sequence[int]], value: Union[EnvironmentStep[_ObsST, _ActST], Iterable[EnvironmentStep[_ObsST, _ActST]], TransitionIterable[_ObsST, _ActST]]) -> None:
        if isinstance(value, EnvironmentStep):
            assert isinstance(index, (int, np.integer)), "Index must be an integer"
            self._transitions[index] = value
        else: #elif isinstance(value, (TransitionIterable, Iterable)):
            if isinstance(index, slice):
                index = range(*index.indices(self.transition_len))
            if isinstance(value, TransitionIterable):
                value = value.iter_transitions()
            for i, v in zip(index, value):
                self._transitions[i] = v

    def __delitem__(self, index: Union[int, slice, Sequence[int]]) -> None:
        if isinstance(index, (int, np.integer, slice)):
            del self._transitions[index]
        else:
            for i in reversed(sorted(index)):
                del self._transitions[i]

    def __add__(self, values: Union[EnvironmentStep[_ObsST, _ActST], Iterable[EnvironmentStep[_ObsST, _ActST]], TransitionIterable[_ObsST, _ActST]]) -> "TransitionSequenceListImpl[_ObsST,_ActST]":
        if isinstance(values, EnvironmentStep):
            return TransitionSequenceListImpl(self._transitions + [values])
        elif isinstance(values, TransitionIterable):
            return TransitionSequenceListImpl(self._transitions + list(values.iter_transitions()))
        else: # elif isinstance(values, Iterable):
            return TransitionSequenceListImpl(self._transitions + list(values))

    def __radd__(self, values: Union[EnvironmentStep[_ObsST, _ActST], Iterable[EnvironmentStep[_ObsST, _ActST]], TransitionIterable[_ObsST, _ActST]]) -> "TransitionSequenceListImpl[_ObsST,_ActST]":
        if isinstance(values, EnvironmentStep):
            return TransitionSequenceListImpl([values] + self._transitions)
        elif isinstance(values, TransitionIterable):
            return TransitionSequenceListImpl(list(values.iter_transitions()) + self._transitions)
        else:
            return TransitionSequenceListImpl(list(values) + self._transitions)
    
    def __iadd__(self, values: Union[EnvironmentStep[_ObsST, _ActST], Iterable[EnvironmentStep[_ObsST, _ActST]], TransitionIterable[_ObsST, _ActST]]) -> "TransitionSequenceListImpl[_ObsST,_ActST]":
        if isinstance(values, EnvironmentStep):
            self.append(values)
        else:
            self.extend(values)
        return self

    def __iter__(self) -> Iterator[EnvironmentStep[_ObsST, _ActST]]:
        return self.iter_transitions()
    
    def iter_transitions(self) -> Iterator[EnvironmentStep[_ObsST, _ActST]]:
        return iter(self._transitions)
    
    def transitions_at(self, index: Union[int, slice, Sequence[int]]) -> Union[EnvironmentStep[_ObsST, _ActST], TransitionSequence[_ObsST, _ActST]]:
        if isinstance(index, (index, np.integer, slice)):
            return self._transitions[index]
        else:
            return TransitionSequenceListImpl([self._transitions[i] for i in index])
    
    def append(self, step : EnvironmentStep[_ObsST, _ActST]) -> None:
        self._transitions.append(step)
    
    def extend(self, steps : Union[TransitionIterable[_ObsST, _ActST], Iterable[EnvironmentStep[_ObsST, _ActST]]]) -> None:
        if isinstance(steps, TransitionIterable):
            self._transitions.extend(steps.iter_transitions())
        else:
            self._transitions.extend(steps)

class EpisodeIterable(TransitionIterable[_ObsST, _ActST], Generic[_ObsST, _ActST], ABC):
    @abstractmethod
    def iter_episodes(self) -> Iterator[TransitionSequence[_ObsST, _ActST]]:
        pass

class EpisodeSequence(EpisodeIterable[_ObsST, _ActST], TransitionSequence[_ObsST, _ActST], Generic[_ObsST, _ActST], ABC):
    episode_len : int

    @property
    def transition_len(self) -> int:
        return sum(e.transition_len for e in self.iter_episodes())

    @staticmethod
    def from_iterable(
        iterable: Iterable[TransitionSequence[_ObsST, _ActST]]
    ) -> "EpisodeSequence[_ObsST, _ActST]":
        return EpisodeSequenceListImpl(iterable)

    def iter_episodes(self) -> Iterator[TransitionSequence[_ObsST, _ActST]]:
        for i in range(self.episode_len):
            yield self.episodes_at(i)
    
    @abstractmethod
    def episodes_at(self, index: Union[int, slice, Sequence[int]]) -> Union[TransitionSequence[_ObsST, _ActST], Sequence[TransitionSequence[_ObsST, _ActST]]]:
        pass

    def transitions_at(self, index: int | slice | Sequence[int]) -> EnvironmentStep[_ObsST, _ActST] | TransitionSequence[_ObsST, _ActST]:
        if isinstance(index, (int, np.integer)):
            past_len = 0
            for i, episode in enumerate(self.iter_episodes()):
                if past_len + episode.transition_len > index:
                    return episode.transitions_at(index - past_len)
                past_len += episode.transition_len
            raise IndexError("Index out of range")
        else:
            all_transitions = TransitionSequenceListImpl([])
            for episode in self.iter_episodes():
                all_transitions += episode
            return all_transitions[index]

class EpisodeSequenceListImpl(EpisodeSequence[_ObsST, _ActST], Generic[_ObsST, _ActST], MutableSequence[TransitionSequence[_ObsST, _ActST]]):
    def __init__(
        self,
        iterable: Iterable[TransitionSequence[_ObsST, _ActST]]
    ):
        self._episodes = list(iterable)
    
    @property
    def episode_len(self) -> int:
        return len(self._episodes)
    
    def __len__(self) -> int:
        return self.episode_len
    
    def __getitem__(self, index: Union[int, slice, Sequence[int]]) -> Union[TransitionSequence[_ObsST, _ActST], Sequence[TransitionSequence[_ObsST, _ActST]]]:
        return self.episodes_at(index)
    
    def __setitem__(self, index: Union[int, slice, Sequence[int]], value: Union[TransitionSequence[_ObsST, _ActST], Iterable[TransitionSequence[_ObsST, _ActST]], EpisodeIterable[_ObsST, _ActST]]) -> None:
        if isinstance(value, TransitionSequence):
            assert isinstance(index, (int, np.integer)), "Index must be an integer"
            self._episodes[index] = value
        else: #elif isinstance(value, (TransitionIterable, Iterable)):
            if isinstance(index, slice):
                index = range(*index.indices(self.episode_len))
            if isinstance(value, EpisodeIterable):
                value = value.iter_episodes()
            for i, v in zip(index, value):
                self._episodes[i] = v

    def __delitem__(self, index: Union[int, slice, Sequence[int]]) -> None:
        if isinstance(index, (int, np.integer, slice)):
            del self._episodes[index]
        else:
            for i in reversed(sorted(index)):
                del self._episodes[i]

    def __add__(self, values: Union[TransitionSequence[_ObsST, _ActST], Iterable[TransitionSequence[_ObsST, _ActST]], EpisodeIterable[_ObsST, _ActST]]) -> "EpisodeSequenceListImpl[_ObsST,_ActST]":
        if isinstance(values, TransitionSequence):
            return EpisodeSequenceListImpl(self._episodes + [values])
        elif isinstance(values, EpisodeIterable):
            return EpisodeSequenceListImpl(self._episodes + list(values.iter_episodes()))
        else: # elif isinstance(values, Iterable):
            return EpisodeSequenceListImpl(self._episodes + list(values))

    def __radd__(self, values: Union[TransitionSequence[_ObsST, _ActST], Iterable[TransitionSequence[_ObsST, _ActST]], EpisodeIterable[_ObsST, _ActST]]) -> "EpisodeSequenceListImpl[_ObsST,_ActST]":
        if isinstance(values, TransitionSequence):
            return EpisodeSequenceListImpl([values] + self._episodes)
        elif isinstance(values, EpisodeIterable):
            return EpisodeSequenceListImpl(list(values.iter_episodes()) + self._episodes)
        else:
            return EpisodeSequenceListImpl(list(values) + self._episodes)
    
    def __iadd__(self, values: Union[TransitionSequence[_ObsST, _ActST], Iterable[TransitionSequence[_ObsST, _ActST]], EpisodeIterable[_ObsST, _ActST]]) -> "EpisodeSequenceListImpl[_ObsST,_ActST]":
        if isinstance(values, TransitionSequence):
            self.append(values)
        else:
            self.extend(values)
        return self

    def __iter__(self) -> Iterator[TransitionSequence[_ObsST, _ActST]]:
        return self.iter_episodes()
    
    def iter_episodes(self) -> Iterator[TransitionSequence[_ObsST, _ActST]]:
        return iter(self._episodes)
    
    def episodes_at(self, index: Union[int, slice, Sequence[int]]) -> Union[TransitionSequence[_ObsST, _ActST], "EpisodeSequenceListImpl[_ObsST, _ActST]"]:
        if isinstance(index, (index, np.integer, slice)):
            return self._episodes[index]
        else:
            return EpisodeSequenceListImpl([self._episodes[i] for i in index])
    
    def append(self, episode : TransitionSequence[_ObsST, _ActST]) -> None:
        self._episodes.append(episode)

    def extend(self, episodes : Union[EpisodeIterable[_ObsST, _ActST], Iterable[TransitionSequence[_ObsST, _ActST]]]) -> None:
        if isinstance(episodes, EpisodeIterable):
            self._episodes.extend(episodes.iter_episodes())
        else:
            self._episodes.extend(episodes)

_SampleOutputT = TypeVar("_SampleOutputT")
class ReplayBuffer(Savable, EpisodeSequence[_ObsST, _ActST], Generic[_ObsST, _ActST, _SampleOutputT], ABC):
    episode_len : int
    capacity : Optional[int] = None
    sample_batch_size : int
    sample_repeat : bool

    @abstractmethod
    def episodes_at(self, index: Union[int, slice, Sequence[int]]) -> Union[TransitionSequence[_ObsST, _ActST], Sequence[TransitionSequence[_ObsST, _ActST]]]:
        pass

    @abstractmethod
    def append(self, step : EnvironmentStep[_ObsST, _ActST], env_id : int = 0) -> None:
        """
        Add a single step to the buffer
        """
        pass

    @abstractmethod
    def extend(self, steps : TransitionIterable[_ObsST, _ActST], env_id : int = 0) -> None:
        """
        Add multiple steps to the buffer
        """
        pass

    @abstractmethod
    def batch_append(self, steps : TransitionIterable[_ObsST, _ActST], env_ids : np.ndarray) -> None:
        """
        Add multiple steps to the buffer in a batch
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear the buffer
        """
        pass

    @abstractmethod
    def save(self, path: Union[str, PathLike]) -> None:
        pass

    @abstractmethod
    def load(self, path: Union[str, PathLike]) -> None:
        pass

    @abstractmethod
    def sample(
        self,
        **kwargs
    ) -> Optional[_SampleOutputT]:
        pass
