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

    def __repr__(self) -> str:
        return f"EnvironmentStep(observation={self.observation}, action={self.action}, next_observation={self.next_observation}, reward={self.reward}, terminated={self.terminated}, truncated={self.truncated}, info={self.info})"

    def __eq__(self, __value: object) -> bool:
        return __value is self

_StepST = TypeVar("_StepST", bound=EnvironmentStep)
class TransitionBatch(Iterable[_StepST], Generic[_StepST], ABC):
    is_single_episode: bool
    is_time_sorted: bool

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[_StepST]:
        pass

    def copy_mutable(self) -> "MutableTransitionSequence":
        """
        Copy this batch
        """
        return MutableTransitionSequence.from_steps(self)

    def __repr__(self) -> str:
        transition_batch_name = __class__.__name__
        if self.is_single_episode:
            transition_batch_name = "[Episodic]" + transition_batch_name
        if self.is_time_sorted:
            transition_batch_name = "[Sorted]" + transition_batch_name
        ret = (
            self.__class__.__name__ + ":" + transition_batch_name if self.__class__ is not __class__ else transition_batch_name
        ) + "("
        for environment_step in self:
            ret += str(environment_step) + ", "
        ret += ")"
        return ret

_BatchT = TypeVar("_BatchT", bound=TransitionBatch)
class TransitionSampler(Generic[_BatchT], Iterable[_StepST]):
    @abstractmethod
    def seed(self, seed : Optional[int] = None) -> None:
        pass
    
    @abstractmethod
    def sample(
        self,
        transition_batch : _BatchT,
        batch_size : int,
        **kwargs
    ) -> Sequence[EnvironmentStep]:
        pass

class EnvironmentStepInSeq(EnvironmentStep[_ObsST, _ActST], Generic[_ObsST, _ActST]):
    seq: "TransitionSequence[_ObsST, _ActST]"
    idx: int

    def __init__(self, seq: "TransitionSequence[_ObsST, _ActST]", idx: int) -> None:
        self.seq = seq
        self.idx = idx

    @property
    def observation(self) -> _ObsST:
        return self.seq.observations[self.idx]

    @property
    def action(self) -> _ActST:
        return self.seq.actions[self.idx]
    
    @property
    def next_observation(self) -> _ObsST:
        return self.seq.next_observations[self.idx]
    
    @property
    def reward(self) -> SupportsFloat:
        return self.seq.rewards[self.idx]
    
    @property
    def terminated(self) -> bool:
        return self.seq.terminations[self.idx]
    
    @property
    def truncated(self) -> bool:
        return self.seq.truncations[self.idx]
    
    @property
    def info(self) -> Dict[str, Any]:
        return self.seq.infos[self.idx]
    
    @property
    def prev(self) -> Optional["EnvironmentStepInSeq[_ObsST, _ActST]"]:
        if self.is_first_in_seq:
            return None
        else:
            return EnvironmentStepInSeq(self.seq, self.idx - 1)
    
    @property
    def next(self) -> Optional["EnvironmentStepInSeq[_ObsST, _ActST]"]:
        if self.is_last_in_seq:
            return None
        else:
            return EnvironmentStepInSeq(self.seq, self.idx + 1)
    
    @property
    def is_first_in_seq(self) -> bool:
        return self.idx == 0
    
    @property
    def is_last_in_seq(self) -> bool:
        return self.idx == len(self.seq) - 1
    
    @property
    def is_start_of_episode(self) -> bool:
        return self.is_first_in_seq or self.prev.terminated or self.prev.truncated
    
    @property
    def is_end_of_episode(self) -> bool:
        return self.is_last_in_seq or self.terminated or self.truncated

class TransitionSequence(TransitionBatch[EnvironmentStepInSeq[_ObsST, _ActST]], Sequence[EnvironmentStep[_ObsST, _ActST]], Generic[_ObsST, _ActST]):
    def __init__(
        self,
        observations: Sequence[_ObsST],
        actions: Sequence[_ActST],
        next_observations: Sequence[_ObsST],
        rewards: Sequence[SupportsFloat],
        terminations: Sequence[bool],
        truncations: Sequence[bool],
        infos: Sequence[Dict[str, Any]],
        is_single_episode: bool = False,
        is_time_sorted: bool = False,
    ) -> None:
        assert len(observations) == len(actions) == len(next_observations) == len(rewards) == len(terminations) == len(truncations) == len(infos)
        self.observations = observations
        self.actions = actions
        self.next_observations = next_observations
        self.rewards = rewards
        self.terminations = terminations
        self.truncations = truncations
        self.infos = infos
        self.is_single_episode = is_single_episode
        self.is_time_sorted = is_time_sorted
    
    @classmethod
    def from_steps(
        cls,
        steps: Iterable[EnvironmentStep[_ObsST, _ActST]], 
        is_single_episode: Optional[bool] = None,
        is_time_sorted: Optional[bool] = None
    ) -> "TransitionSequence[_ObsST,_ActST]":
        if (
            isinstance(steps, TransitionSequence) and
            isinstance(steps.observations, list) and
            isinstance(steps.actions, list) and
            isinstance(steps.next_observations, list) and
            isinstance(steps.rewards, list) and
            isinstance(steps.terminations, list) and
            isinstance(steps.truncations, list) and
            isinstance(steps.infos, list)
        ):
            return cls(
                steps.observations.copy(),
                steps.actions.copy(),
                steps.next_observations.copy(),
                steps.rewards.copy(),
                steps.terminations.copy(),
                steps.truncations.copy(),
                steps.infos.copy(),
                steps.is_single_episode,
                steps.is_time_sorted
            )


        is_single_episode_override = is_single_episode
        is_time_sorted_override = is_time_sorted
        if isinstance(steps, TransitionBatch):
            is_single_episode_override = steps.is_single_episode
            is_time_sorted_override = steps.is_time_sorted
        
        assert is_single_episode_override is not None and is_time_sorted_override is not None, \
            "is_single_episode and is_time_sorted must be specified when steps is not a TransitionBatch"

        observations = []
        actions = []
        next_observations = []
        rewards = []
        terminations = []
        truncations = []
        infos = []
        for step in steps:
            assert isinstance(step, EnvironmentStep)
            observations.append(step.observation)
            actions.append(step.action)
            next_observations.append(step.next_observation)
            rewards.append(step.reward)
            terminations.append(step.terminated)
            truncations.append(step.truncated)
            infos.append(step.info)
        return cls(
            observations,
            actions,
            next_observations,
            rewards,
            terminations,
            truncations,
            infos,
            is_single_episode_override,
            is_time_sorted_override
        )

    def __len__(self) -> int:
        return len(self.observations)
    
    def __contains__(self, value: object) -> bool:
        return isinstance(value, EnvironmentStep) and value in self.__iter__()
    
    def __iter__(self) -> Iterator[EnvironmentStepInSeq[_ObsST, _ActST]]:
        for idx in range(len(self)):
            yield EnvironmentStepInSeq(self, idx)
    
    def __getitem__(self, index: Union[int, slice]) -> Union[EnvironmentStepInSeq[_ObsST, _ActST], "TransitionSequence[_ObsST,_ActST]"]:
        if isinstance(index, slice):
            return TransitionSequence(
                self.observations[index],
                self.actions[index],
                self.next_observations[index],
                self.rewards[index],
                self.terminations[index],
                self.truncations[index],
                self.infos[index],
                self.is_single_episode,
                self.is_time_sorted
            )
        else:
            if not isinstance(index, int):
                raise TypeError("Index must be an integer or slice")
            return EnvironmentStepInSeq(self, index)
    
    def __add__(self, other : Union[EnvironmentStep[_ObsST, _ActST], Iterable[EnvironmentStep[_ObsST, _ActST]]]) -> "TransitionSequence[_ObsST,_ActST]":
        other_trans : Optional[TransitionSequence[_ObsST,_ActST]] = None
        if isinstance(other, EnvironmentStep):
            to_ret = __class__(
                self.observations + [other.observation],
                self.actions + [other.action],
                self.next_observations + [other.next_observation],
                self.rewards + [other.reward],
                self.terminations + [other.terminated],
                self.truncations + [other.truncated],
                self.infos + [other.info],
                self.is_single_episode,
                self.is_time_sorted
            )
            return to_ret
        if isinstance(other, TransitionSequence):
            other_trans = other
        elif isinstance(other, Iterable):
            other_trans = TransitionSequence.from_steps(other, False, False)
        
        if other_trans is None:
            raise TypeError(f"Cannot add {other} to {self}")
        else:
            return __class__(
                self.observations + other_trans.observations,
                self.actions + other_trans.actions,
                self.next_observations + other_trans.next_observations,
                self.rewards + other_trans.rewards,
                self.terminations + other_trans.terminations,
                self.truncations + other_trans.truncations,
                self.infos + other_trans.infos,
                False,
                self.is_time_sorted and other_trans.is_time_sorted
            )

    def __mul__(self, n : int) -> "TransitionSequence[_ObsST,_ActST]":
        return __class__(
            self.observations * n,
            self.actions * n,
            self.next_observations * n,
            self.rewards * n,
            self.terminations * n,
            self.truncations * n,
            self.infos * n
        )
    
    __rmul__ = __mul__


class MutableTransitionSequence(TransitionSequence[_ObsST, _ActST], MutableSequence[EnvironmentStep[_ObsST, _ActST]], Generic[_ObsST, _ActST]):
    def __init__(
        self,
        observations: MutableSequence[_ObsST],
        actions: MutableSequence[_ActST],
        next_observations: MutableSequence[_ObsST],
        rewards: MutableSequence[SupportsFloat],
        terminations: MutableSequence[bool],
        truncations: MutableSequence[bool],
        infos: MutableSequence[Dict[str, Any]],
        is_single_episode: bool = False,
        is_time_sorted: bool = False,
    ) -> None:
        assert len(observations) == len(actions) == len(next_observations) == len(rewards) == len(terminations) == len(truncations) == len(infos)
        self.observations = observations
        self.actions = actions
        self.next_observations = next_observations
        self.rewards = rewards
        self.terminations = terminations
        self.truncations = truncations
        self.infos = infos
        self.is_single_episode = is_single_episode
        self.is_time_sorted = is_time_sorted
    
    def __getitem__(self, index: Union[int, slice]) -> Union[EnvironmentStepInSeq[_ObsST, _ActST], "MutableTransitionSequence[_ObsST,_ActST]"]:
        if isinstance(index, slice):
            return MutableTransitionSequence(
                self.observations[index],
                self.actions[index],
                self.next_observations[index],
                self.rewards[index],
                self.terminations[index],
                self.truncations[index],
                self.infos[index]
            )
        else:
            if not isinstance(index, int):
                raise TypeError("Index must be an integer or slice")
            return EnvironmentStepInSeq(self, index)
    
    def __setitem__(self, index: Union[int, slice], value: Union[EnvironmentStep[_ObsST, _ActST], Iterable[EnvironmentStep[_ObsST,_ActST]]]) -> None:
        if isinstance(index, slice):
            v_mutable = None
            if isinstance(value, MutableTransitionSequence):
                v_mutable = value
            elif isinstance(value, Iterable):
                v_mutable = MutableTransitionSequence.from_steps(value)
            else:
                raise TypeError(f"Cannot set index {index} to {value}")
            
            self.observations[index] = v_mutable.observations
            self.actions[index] = v_mutable.actions
            self.next_observations[index] = v_mutable.next_observations
            self.rewards[index] = v_mutable.rewards
            self.terminations[index] = v_mutable.terminations
            self.truncations[index] = v_mutable.truncations
            self.infos[index] = v_mutable.infos
        else:
            if not isinstance(index, int):
                raise TypeError("Index must be an integer or slice")
            assert isinstance(value, EnvironmentStep)
            self.observations[index] = value.observation
            self.actions[index] = value.action
            self.next_observations[index] = value.next_observation
            self.rewards[index] = value.reward
            self.terminations[index] = value.terminated
            self.truncations[index] = value.truncated
            self.infos[index] = value.info
    
    def __delitem__(self, index: Union[int, slice]) -> None:
        del self.observations[index]
        del self.actions[index]
        del self.next_observations[index]
        del self.rewards[index]
        del self.terminations[index]
        del self.truncations[index]
        del self.infos[index]
    
    def __add__(self, other : Union[EnvironmentStep[_ObsST, _ActST], Iterable[EnvironmentStep[_ObsST, _ActST]]]) -> "MutableTransitionSequence[_ObsST,_ActST]":
        other_mutable : Optional[MutableTransitionSequence[_ObsST,_ActST]] = None
        if isinstance(other, EnvironmentStep):
            new : MutableTransitionSequence = __class__.from_steps(self)
            new.append(other)
            return new
        elif isinstance(other, MutableTransitionSequence):
            other_mutable = other
        elif isinstance(other, Iterable):
            other_mutable = MutableTransitionSequence.from_steps(other)
        
        if other_mutable is None:
            raise TypeError(f"Cannot add {other} to {self}")
        else:
            return MutableTransitionSequence(
                self.observations + other_mutable.observations,
                self.actions + other_mutable.actions,
                self.next_observations + other_mutable.next_observations,
                self.rewards + other_mutable.rewards,
                self.terminations + other_mutable.terminations,
                self.truncations + other_mutable.truncations,
                self.infos + other_mutable.infos
            )
    
    def __iadd__(self, other : Union[EnvironmentStep[_ObsST, _ActST], Iterable[EnvironmentStep[_ObsST, _ActST]]]) -> "MutableTransitionSequence[_ObsST,_ActST]":
        other_mutable = None
        if isinstance(other, EnvironmentStep):
            self.append(other)
            return self
        elif isinstance(other, MutableTransitionSequence):
            other_mutable = other
        elif isinstance(other, Iterable):
            other_mutable = MutableTransitionSequence.from_steps(other)

        if other_mutable is None:
            raise TypeError(f"Cannot add {other} to {self}")
        else:
            self.observations += other_mutable.observations
            self.actions += other_mutable.actions
            self.next_observations += other_mutable.next_observations
            self.rewards += other_mutable.rewards
            self.terminations += other_mutable.terminations
            self.truncations += other_mutable.truncations
            self.infos += other_mutable.infos
            return self

    def __mul__(self, n : int) -> "MutableTransitionSequence[_ObsST,_ActST]":
        return MutableTransitionSequence(
            self.observations * n,
            self.actions * n,
            self.next_observations * n,
            self.rewards * n,
            self.terminations * n,
            self.truncations * n,
            self.infos * n
        )
    
    __rmul__ = __mul__

    def __imul__(self, n : int) -> "MutableTransitionSequence[_ObsST,_ActST]":
        self.observations *= n
        self.actions *= n
        self.next_observations *= n
        self.rewards *= n
        self.terminations *= n
        self.truncations *= n
        self.infos *= n
        return self

    def append(self, value: EnvironmentStep[_ObsST, _ActST]) -> None:
        self.observations.append(value.observation)
        self.actions.append(value.action)
        self.next_observations.append(value.next_observation)
        self.rewards.append(value.reward)
        self.terminations.append(value.terminated)
        self.truncations.append(value.truncated)
        self.infos.append(value.info)
    
    def insert(self, index: int, value: EnvironmentStep[_ObsST, _ActST]) -> None:
        self.observations.insert(index, value.observation)
        self.actions.insert(index, value.action)
        self.next_observations.insert(index, value.next_observation)
        self.rewards.insert(index, value.reward)
        self.terminations.insert(index, value.terminated)
        self.truncations.insert(index, value.truncated)
        self.infos.insert(index, value.info)
    
    def pop(self, index: int = -1) -> EnvironmentStep[_ObsST, _ActST]:
        return EnvironmentStep(
            self.observations.pop(index),
            self.actions.pop(index),
            self.next_observations.pop(index),
            self.rewards.pop(index),
            self.terminations.pop(index),
            self.truncations.pop(index),
            self.infos.pop(index)
        )
    
    def remove(self, value: EnvironmentStep[_ObsST, _ActST]) -> None:
        idx = self.index(value)
        self.pop(idx)

    def index(self, value: EnvironmentStep[_ObsST, _ActST], start: int = 0, stop: int = ...) -> int:
        for idx in range(start, stop):
            if self[idx] == value:
                return idx
        raise ValueError(f"{value} is not in {self}")
    
    def reverse(self) -> None:
        self.observations.reverse()
        self.actions.reverse()
        self.next_observations.reverse()
        self.rewards.reverse()
        self.terminations.reverse()
        self.truncations.reverse()
        self.infos.reverse()
    
    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.next_observations.clear()
        self.rewards.clear()
        self.terminations.clear()
        self.truncations.clear()
        self.infos.clear()
    
    def extend(self, other : Iterable[EnvironmentStep[_ObsST,_ActST]]) -> None:
        self.__iadd__(other)
