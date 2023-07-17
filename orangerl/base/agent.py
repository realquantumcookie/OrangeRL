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
from abc import ABC, abstractmethod
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
_ProbOutputT = TypeVar("_ProbOutputT", bound=SupportsFloat)
@dataclass
class AgentOutput(Generic[_ActOutputT, _StateOutputT, _ProbOutputT]):
    action: _ActOutputT
    log_prob: Union[float, _ProbOutputT]
    state: Optional[_StateOutputT] = None

_ObsT = TypeVar("_ObsT")
_ActT = TypeVar("_ActT")
_StateT = TypeVar("_StateT")
class Agent(Generic[_ObsT, _ActT, _StateT], ABC):
    randomness : np.random.Generator = np.random.default_rng()
    def get_action(
        self, 
        observation : _ObsT, 
        state : Optional[_StateT], 
        stage : AgentStage = AgentStage.ONLINE
    ) -> AgentOutput[_ActT, _StateT, SupportsFloat]:
        return next(iter(self.get_action_batch([observation], None, stage))) if state is None else next(iter(self.get_action_batch([observation], [state], stage)))

    @property
    def unwrapped(self) -> "Agent[_ObsT, _ActT]":
        return self

    @property
    def action_type(self) -> AgentActionType:
        pass

    @abstractmethod
    def get_action_batch(
        self, 
        observations : Iterable[_ObsT], 
        states : Optional[Iterable[Optional[_StateT]]], 
        stage : AgentStage = AgentStage.ONLINE
    ) -> Iterable[AgentOutput[_ActT, _StateT, SupportsFloat]]:
        pass

    def add_transitions(
        self, 
        transition : Union[TransitionBatch[_ObsT, _ActT], EnvironmentStep[_ObsT, _ActT]],
        stage : AgentStage = AgentStage.ONLINE
    ):
        pass

    @abstractmethod
    def update(self, batch_size : Optional[int] = None, *args, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def prefetch_update_data(self, batch_size : Optional[int] = None, *args, **kwargs) -> None:
        pass

_ObsWT = TypeVar("_ObsWT")
_ActWT = TypeVar("_ActWT")
_StateWT = TypeVar("_StateWT")
class AgentWrapper(Generic[_ObsWT, _ActWT, _StateWT], Agent[_ObsWT, _ActWT, _StateWT]):
    def __init__(self, agent : Union["AgentWrapper[_ObsWT, _ActWT, _StateWT]", Agent[_ObsWT, _ActWT, _StateWT]]) -> None:
        Agent.__init__(self)
        self.agent = agent

    def __getattr__(self, __name: str) -> Any:
        if __name.startswith("_"):
            raise AttributeError("Cannot access private attribute")
        return getattr(self.agent, __name)
    
    @property
    def is_sequence_model(self) -> bool:
        return False

    @property
    def action_type(self) -> AgentActionType:
        return self.agent.action_type

    def get_action(
        self, 
        observation : _ObsWT, 
        state : Optional[_StateWT], 
        stage : AgentStage = AgentStage.ONLINE
    ) -> AgentOutput[_ActWT, _StateWT, SupportsFloat]:
        return self.agent.get_action(observation, state, stage)
    
    def get_action_batch(
        self, 
        observations : Iterable[_ObsWT], 
        states : Optional[Iterable[Optional[_StateWT]]], 
        stage : AgentStage = AgentStage.ONLINE
    ) -> Iterable[AgentOutput[_ActWT, _StateWT, SupportsFloat]]:
        return self.agent.get_action_batch(observations, states, stage)

    def add_transitions(
        self, 
        transition : Union[TransitionBatch[_ObsWT, _ActWT], EnvironmentStep[_ObsWT, _ActWT]],
        stage : AgentStage = AgentStage.ONLINE
    ):
        self.agent.add_transitions(transition, stage)

    def update(self, batch_size : Optional[int] = None, *args, **kwargs) -> Dict[str, Any]:
        return self.agent.update(batch_size, *args, **kwargs)

    @property
    def unwrapped(self) -> Agent[_ObsWT, _ActWT]:
        return self.agent.unwrapped()
