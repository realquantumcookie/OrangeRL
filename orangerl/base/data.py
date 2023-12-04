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

class TransitionTransformation(Generic[_ObsST, _ActST]):
    @abstractmethod
    def transform_transition(self, transition : EnvironmentStep[_ObsST, _ActST]) -> EnvironmentStep[_ObsST, _ActST]:
        pass
    
    def transform_batch(self, batch : Sequence[EnvironmentStep[_ObsST, _ActST]], **kwargs) -> Sequence[EnvironmentStep[_ObsST, _ActST]]:
        return [self.transform_transition(transition) for transition in batch]

class TransitionBatch(Generic[_ObsST, _ActST], ABC):
    transition_len : int

    @abstractmethod
    def iter_transitions(self) -> Iterator[EnvironmentStep[_ObsST, _ActST]]:
        pass

    # def __repr__(self) -> str:
    #     limit_repr_len = 10

    #     transition_batch_name = __class__.__name__
    #     if self.is_single_episode:
    #         transition_batch_name = "[Episodic]" + transition_batch_name
    #     if self.is_time_sorted:
    #         transition_batch_name = "[Sorted]" + transition_batch_name
    #     ret = (
    #         self.__class__.__name__ + ":" + transition_batch_name if self.__class__ is not __class__ else transition_batch_name
    #     ) + "("

    #     repr_len = 1
    #     for environment_step in self:
    #         ret += str(environment_step) + ", "
    #         if repr_len >= limit_repr_len:
    #             ret += "..."
    #             break
    #         repr_len += 1
    #     ret += ")"
    #     return ret

class TransitionSequence(TransitionBatch[_ObsST, _ActST], Generic[_ObsST, _ActST], ABC):
    transition_len : int

    @staticmethod
    def from_iterable(
        iterable: Iterable[EnvironmentStep[_ObsST, _ActST]]
    ) -> "TransitionSequence[_ObsST, _ActST]":
        return TransitionSequenceListImpl(iterable)

    @abstractmethod
    def iter_transitions(self) -> Iterator[EnvironmentStep[_ObsST, _ActST]]:
        pass
    
    @abstractmethod
    def transitions_at(self, index: Union[int, slice, Sequence[int]]) -> Union[EnvironmentStep[_ObsST, _ActST], Sequence[EnvironmentStep[_ObsST, _ActST]]]:
        pass

class TransitionSequenceListImpl(TransitionSequence[_ObsST, _ActST], Generic[_ObsST, _ActST], MutableSequence[EnvironmentStep[_ObsST, _ActST]]):
    def __init__(
        self,
        iterable: Iterable[EnvironmentStep[_ObsST, _ActST]]
    ):
        if isinstance(iterable, Sequence):
            self._transitions = iterable
        else:
            self._transitions = list(iterable)
    
    @property
    def transition_len(self) -> int:
        return len(self._transitions)
    
    def __len__(self) -> int:
        return self.transition_len
    
    def __getitem__(self, index: Union[int, slice, Sequence[int]]) -> Union[EnvironmentStep[_ObsST, _ActST], Sequence[EnvironmentStep[_ObsST, _ActST]]]:
        return self.transitions_at(index)
    
    def __setitem__(self, index: Union[int, slice, Sequence[int]], value: Union[EnvironmentStep[_ObsST, _ActST], Iterable[EnvironmentStep[_ObsST, _ActST]], TransitionBatch[_ObsST, _ActST]]) -> None:
        if isinstance(value, EnvironmentStep):
            assert isinstance(index, (int, np.integer)), "Index must be an integer"
            self._transitions[index] = value
        else: #elif isinstance(value, (TransitionBatch, Iterable)):
            if isinstance(index, slice):
                index = range(*index.indices(self.transition_len))
            if isinstance(value, TransitionBatch):
                value = value.iter_transitions()
            for i, v in zip(index, value):
                self._transitions[i] = v

    def __delitem__(self, index: Union[int, slice, Sequence[int]]) -> None:
        if isinstance(index, (int, np.integer, slice)):
            del self._transitions[index]
        else:
            for i in reversed(sorted(index)):
                del self._transitions[i]

    def __add__(self, values: Union[EnvironmentStep[_ObsST, _ActST], Iterable[EnvironmentStep[_ObsST, _ActST]], TransitionBatch[_ObsST, _ActST]]) -> "TransitionSequenceListImpl[_ObsST,_ActST]":
        if isinstance(values, EnvironmentStep):
            return TransitionSequenceListImpl(self._transitions + [values], self.is_transition_single_episode, self.is_transition_time_sorted)
        elif isinstance(values, TransitionBatch):
            return TransitionSequenceListImpl(self._transitions + list(values.iter_transitions()), self.is_transition_single_episode, self.is_transition_time_sorted)
        else: # elif isinstance(values, Iterable):
            return TransitionSequenceListImpl(self._transitions + list(values), self.is_transition_single_episode, self.is_transition_time_sorted)

    def __radd__(self, values: Union[EnvironmentStep[_ObsST, _ActST], Iterable[EnvironmentStep[_ObsST, _ActST]], TransitionBatch[_ObsST, _ActST]]) -> "TransitionSequenceListImpl[_ObsST,_ActST]":
        if isinstance(values, EnvironmentStep):
            return TransitionSequenceListImpl([values] + self._transitions, self.is_transition_single_episode, self.is_transition_time_sorted)
        elif isinstance(values, TransitionBatch):
            return TransitionSequenceListImpl(list(values.iter_transitions()) + self._transitions, self.is_transition_single_episode, self.is_transition_time_sorted)
        else:
            return TransitionSequenceListImpl(list(values) + self._transitions, self.is_transition_single_episode, self.is_transition_time_sorted)
    
    def __iadd__(self, values: Union[EnvironmentStep[_ObsST, _ActST], Iterable[EnvironmentStep[_ObsST, _ActST]], TransitionBatch[_ObsST, _ActST]]) -> "TransitionSequenceListImpl[_ObsST,_ActST]":
        if isinstance(values, EnvironmentStep):
            self.append(values)
        else:
            self.extend(values)
        return self

    def __iter__(self) -> Iterator[EnvironmentStep[_ObsST, _ActST]]:
        return self.iter_transitions()
    
    def iter_transitions(self) -> Iterator[EnvironmentStep[_ObsST, _ActST]]:
        return iter(self._transitions)
    
    def transitions_at(self, index: Union[int, slice, Sequence[int]]) -> Union[EnvironmentStep[_ObsST, _ActST], Sequence[EnvironmentStep[_ObsST, _ActST]]]:
        if isinstance(index, (index, np.integer, slice)):
            return self._transitions[index]
        else:
            return [self._transitions[i] for i in index]
    
    def append(self, step : EnvironmentStep[_ObsST, _ActST]) -> None:
        self._transitions.append(step)
    
    def extend(self, steps : Union[TransitionBatch[_ObsST, _ActST], Iterable[EnvironmentStep[_ObsST, _ActST]]]) -> None:
        if isinstance(steps, TransitionBatch):
            self._transitions.extend(steps.iter_transitions())
        else:
            self._transitions.extend(steps)

class TransitionSampler(Generic[_ObsST, _ActST], Iterable[EnvironmentStep[_ObsST, _ActST]], ABC):
    @abstractmethod
    def seed(self, seed : Optional[int] = None) -> None:
        pass
    
    @abstractmethod
    def sample_idx(
        self,
        transition_batch : TransitionSequence[_ObsST, _ActST],
        batch_size : int,
        repeat_sample = True,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Sample indices from the transition batch
        Output:
            sample_idx: np.ndarray
                The sampled indices
            sample_mask: Optional[np.ndarray]
                The mask of the sampled indices
        """
        pass

_SampleOutputT = TypeVar("_SampleOutputT")
class TransitionReplayBuffer(TransitionSequence[_ObsST, _ActST], MutableSequence[EnvironmentStep[_ObsST, _ActST]], Generic[_ObsST, _ActST, _SampleOutputT], ABC):
    transition_len : int
    capacity : Optional[int] = None
    sampler: TransitionSampler[_ObsST, _ActST]
    sample_batch_size : int
    sample_repeat : bool
    sample_transforms : List[TransitionTransformation[_ObsST, _ActST]]

    @abstractmethod
    def iter_transitions(self) -> Iterator[EnvironmentStep[_ObsST, _ActST]]:
        pass
    
    @abstractmethod
    def idx_episode_begins_and_ends(self) -> Sequence[Tuple[int, int]]:
        """
        Returns a list of tuples (begin, end) where each tuple represents the beginning and end of an episode
        Note that some end indices may be >= len(self), which means that it should be read as end % len(self)
        """

        pass

    @abstractmethod
    def transitions_at(self, index: Union[int, slice, Sequence[int]]) -> Union[EnvironmentStep[_ObsST, _ActST], Sequence[EnvironmentStep[_ObsST, _ActST]]]:
        pass

    def __len__(self) -> int:
        return self.transition_len

    def __getitem__(self, index: Union[int, slice, Sequence[int]]) -> Union[EnvironmentStep[_ObsST, _ActST], Sequence[EnvironmentStep[_ObsST, _ActST]]]:
        return self.transitions_at(index)

    @abstractmethod
    def __setitem__(self, index: Union[int, slice, Sequence[int]], value: Union[EnvironmentStep[_ObsST, _ActST], Iterable[EnvironmentStep[_ObsST, _ActST]], TransitionBatch[_ObsST, _ActST]]) -> None:
        pass

    @abstractmethod
    def __delitem__(self, index: Union[int, slice, Sequence[int]]) -> None:
        pass

    @abstractmethod
    def append(self, step : EnvironmentStep[_ObsST, _ActST]) -> None:
        """
        Add a single step to the buffer
        """
        pass

    @abstractmethod
    def extend(self, steps : Union[Iterable[EnvironmentStep[_ObsST, _ActST]], TransitionBatch[_ObsST, _ActST]]) -> None:
        """
        Add multiple steps to the buffer
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear the buffer
        """
        pass

    def __iadd__(self, values: Union[EnvironmentStep[_ObsST, _ActST], Iterable[EnvironmentStep[_ObsST, _ActST]], TransitionBatch[_ObsST, _ActST]]) -> "TransitionReplayBuffer[_ObsST,_ActST]":
        if isinstance(values, EnvironmentStep):
            self.append(values)
        else:
            self.extend(values)
        return self

    @abstractmethod
    def sample(
        self,
        **kwargs
    ) -> _SampleOutputT:
        pass
