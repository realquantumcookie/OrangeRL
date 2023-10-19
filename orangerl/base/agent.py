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
from .data import TransitionBatch, EnvironmentStep
from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
import numpy as np
from dataclasses import dataclass

class AgentStage(Enum):
    EXPLORE = 1
    ONLINE = 2
    OFFLINE = 3
    EVAL = 4

class AgentActionType(Enum):
    DISCRETE = 1
    CONTINUOUS = 2

_ActOutputT = TypeVar("_ActOutputT")
_StateOutputT = TypeVar("_StateOutputT")
_LogProbOutputT = TypeVar("_LogProbOutputT", bound=SupportsFloat)
@dataclass
class AgentOutput(Generic[_ActOutputT, _StateOutputT, _LogProbOutputT]):
    action: _ActOutputT
    log_prob: _LogProbOutputT
    state: _StateOutputT

_ObsT = TypeVar("_ObsT")
_ActT = TypeVar("_ActT")
_ActOT = TypeVar("_ActOT")
_StateT = TypeVar("_StateT")
_StateOT = TypeVar("_StateOT")
_LogProbOT = TypeVar("_LogProbOT", bound=SupportsFloat)
class Agent(Generic[_ObsT, _ActT, _ActOT, _StateT, _StateOT, _LogProbOT], ABC):
    is_sequence_model : bool
    action_type: AgentActionType
    np_random : Optional[np.random.Generator] = None

    def get_action(
        self, 
        observation : _ObsT, 
        state : Optional[_StateT], 
        stage : AgentStage = AgentStage.ONLINE
    ) -> AgentOutput[_ActOT, _StateOT, _LogProbOT]:
        return next(iter(self.get_action_batch([observation], None, stage))) if state is None else next(iter(self.get_action_batch([observation], [state], stage)))

    @property
    def unwrapped(self) -> "Agent[_ObsT, _ActT, _ActOT, _StateT, _StateOT, _LogProbOT]":
        return self

    @abstractmethod
    def get_action_batch(
        self, 
        observations : Iterable[_ObsT], 
        states : Optional[Iterable[Optional[_StateT]]], 
        stage : AgentStage = AgentStage.ONLINE
    ) -> Iterable[AgentOutput[_ActOT, _StateOT, _LogProbOT]]:
        pass

    @abstractmethod
    def add_transitions(
        self, 
        transition : Union[Iterable[EnvironmentStep[_ObsT, _ActT]], EnvironmentStep[_ObsT, _ActT]],
        stage : AgentStage = AgentStage.ONLINE
    ) -> None:
        pass

    @abstractmethod
    def update(self, stage: AgentStage = AgentStage.ONLINE, batch_size : Optional[int] = None, *args, **kwargs) -> Dict[str, Any]:
        pass

class AgentWrapper(Generic[_ObsT, _ActT, _ActOT, _StateT, _StateOT, _LogProbOT], Agent[_ObsT, _ActT, _ActOT, _StateT, _StateOT, _LogProbOT]):
    def __init__(self, agent : Agent[_ObsT, _ActT, _ActOT, _StateT, _StateOT, _LogProbOT]) -> None:
        Agent.__init__(self)
        self._agent = agent

    def __getattr__(self, __name: str) -> Any:
        if __name.startswith("_"):
            raise AttributeError("Cannot access private attribute")
        return getattr(self._agent, __name)

    @property
    def unwrapped(self) -> Agent[_ObsT, _ActT, _ActOT, _StateT, _StateOT, _LogProbOT]:
        return self._agent.unwrapped

    @property
    def action_type(self) -> AgentActionType:
        return self._agent.action_type
    
    @property
    def is_sequence_model(self) -> bool:
        return self._agent.is_sequence_model

    @property
    def np_random(self) -> Optional[np.random.Generator]:
        return self._agent.np_random

    def get_action(
        self, 
        observation : _ObsT, 
        state : Optional[_StateT], 
        stage : AgentStage = AgentStage.ONLINE
    ) -> AgentOutput[_ActOT, _StateOT, _LogProbOT]:
        return Agent.get_action(self, observation, state, stage)
    
    def get_action_batch(
        self, 
        observations : Iterable[_ObsT], 
        states : Optional[Iterable[Optional[_StateT]]], 
        stage : AgentStage = AgentStage.ONLINE
    ) -> Iterable[AgentOutput[_ActOT, _StateOT, _LogProbOT]]:
        return self._agent.get_action_batch(observations, states, stage)

    def add_transitions(
        self, 
        transition : Union[Iterable[EnvironmentStep[_ObsT, _ActT]], EnvironmentStep[_ObsT, _ActT]],
        stage : AgentStage = AgentStage.ONLINE
    ) -> None:
        self._agent.add_transitions(transition, stage)

    def update(self, stage: AgentStage = AgentStage.ONLINE, batch_size : Optional[int] = None, *args, **kwargs) -> Dict[str, Any]:
        return self._agent.update(stage, batch_size, *args, **kwargs)
