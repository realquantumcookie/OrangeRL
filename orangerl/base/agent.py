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

from typing import Any, TypeVar, Generic, Optional, Union, Dict, Iterable, List, SupportsFloat
from .common import Savable
from .data import TransitionIterable, EnvironmentStep, ReplayBuffer
from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
import numpy as np
from dataclasses import dataclass
from os import PathLike

class AgentStage(Enum):
    EXPLORE = 1
    ONLINE = 2
    OFFPOLICY = 3
    OFFLINE = 4
    EVAL = 5

_ActT = TypeVar("_ActT")
_StateT = TypeVar("_StateT")
_LogProbT = TypeVar("_LogProbT", bound=SupportsFloat)
@dataclass
class ActionOutput(Generic[_ActT, _StateT, _LogProbT]):
    action: _ActT
    log_prob: _LogProbT
    state: Optional[_StateT]

class BatchedActionOutput(Generic[_ActT, _StateT, _LogProbT], ABC):
    num_actions: int

    @abstractmethod    
    def iter_actions(self) -> Iterable[ActionOutput[_ActT, _StateT, _LogProbT]]:
        pass

_ObsT = TypeVar("_ObsT")
class Actor(Generic[_ObsT, _ActT, _StateT, _LogProbT], Savable, ABC):
    is_sequence_model : bool
    current_stage : AgentStage
    empty_state : _StateT

    @abstractmethod
    def get_action(
        self, 
        observation : _ObsT, 
        state : Optional[_StateT] = None
    ) -> ActionOutput[_ActT, _StateT, _LogProbT]:
        pass

    @abstractmethod
    def seed(
        self, 
        seed : Optional[int] = None
    ) -> None:
        pass

    @abstractmethod
    def save(self, path : Union[str, PathLike]) -> None:
        pass

    @abstractmethod
    def load(self, path : Union[str, PathLike]) -> None:
        pass

class Agent(Generic[_ObsT, _ActT, _StateT, _LogProbT], Actor[_ObsT, _ActT, _StateT, _LogProbT], ABC):
    is_sequence_model : bool
    observe_transition_infos : bool
    replay_buffer : Optional[ReplayBuffer[_ObsT, _ActT, Any]]
    current_stage : AgentStage
    empty_state : _StateT

    @property
    def unwrapped(self) -> "Agent":
        return self
    
    @property
    def transform_parent(self) -> Optional["Agent"]:
        return None
    
    @abstractmethod
    def get_action(
        self, 
        observation : _ObsT, 
        state : Optional[_StateT] = None
    ) -> ActionOutput[_ActT, _StateT, _LogProbT]:
        pass

    @abstractmethod
    def seed(
        self, 
        seed : Optional[int] = None
    ) -> None:
        pass

    @abstractmethod
    def observe_transitions(
        self, 
        transition : Union[EnvironmentStep[_ObsT, _ActT], TransitionIterable[_ObsT, _ActT]],
        env_id : int = 0,
    ) -> None:
        pass

    @abstractmethod
    def save(self, path : Union[str, PathLike]) -> None:
        pass

    @abstractmethod
    def load(self, path : Union[str, PathLike]) -> None:
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Dict[str, Any]:
        pass

class AgentWrapper(Generic[_ObsT, _ActT, _StateT, _LogProbT], Agent[_ObsT, _ActT, _StateT, _LogProbT]):
    def __init__(self, agent : Agent[_ObsT, _ActT, _StateT, _LogProbT]) -> None:
        Agent.__init__(self)
        self._agent = agent

    def __getattr__(self, __name: str) -> Any:
        if __name.startswith("_"):
            raise AttributeError("Cannot access private attribute")
        return getattr(self._agent, __name)

    @property
    def is_sequence_model(self) -> bool:
        return self._agent.is_sequence_model
    
    @property
    def observe_transition_infos(self) -> bool:
        return self._agent.observe_transition_infos

    @property
    def replay_buffer(self) -> Optional[ReplayBuffer[_ObsT, _ActT, Any]]:
        return self._agent.replay_buffer
    
    @property
    def current_stage(self) -> AgentStage:
        return self._agent.current_stage
    
    @current_stage.setter
    def current_stage(self, stage : AgentStage) -> None:
        self._agent.current_stage = stage
    
    @property
    def empty_state(self) -> _StateT:
        return self._agent.empty_state

    @property
    def unwrapped(self) -> Agent:
        return self._agent.unwrapped

    @property
    def transform_parent(self) -> Optional[Agent]:
        return self._agent

    def get_action(
        self, 
        observation : _ObsT, 
        state : Optional[_StateT] = None
    ) -> ActionOutput[_ActT, _StateT, _LogProbT]:
        return Agent.get_action(self, observation, state)
    
    def seed(
        self, 
        seed : Optional[int] = None
    ) -> None:
        self._agent.seed(seed)

    def observe_transitions(
        self, 
        transition : Union[EnvironmentStep[_ObsT, _ActT], TransitionIterable[_ObsT, _ActT]],
        env_id : int = 0,
    ) -> None:
        self._agent.observe_transitions(transition, env_id)

    def save(self, path : Union[str, PathLike]) -> None:
        self._agent.save(path)

    def load(self, path : Union[str, PathLike]) -> None:
        self._agent.load(path)

    def update(self, *args, **kwargs) -> Dict[str, Any]:
        return self._agent.update(*args, **kwargs)
