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

from typing import Any, TypeVar, Generic, Optional, Union, Dict, Iterabl
from .datas import TransitionBatch, EnvironmentStep
from abc import ABC, abstractmethod
from enum import Enum

class AgentStage(Enum):
    EXPLORE = 1
    ONLINE = 2
    OFFLINE = 3

_ObsT = TypeVar("_ObsT")
_ActT = TypeVar("_ActT")
class Agent(Generic[_ObsT, _ActT], ABC):
    @abstractmethod
    def get_action(self, observation : _ObsT, stage : AgentStage = AgentStage.ONLINE) -> _ActT:
        pass

    @abstractmethod
    def get_action_batch(self, observation : Union[_ObsT, Iterable[_ObsT]], stage : AgentStage = AgentStage.ONLINE) -> Union[_ActT, Iterable[_ActT]]:
        pass

    @abstractmethod
    def add_transitions(self, transition : Union[TransitionBatch[_ObsT, _ActT], EnvironmentStep[_ObsT, _ActT]]):
        pass

    @abstractmethod
    def update(self, batch_size : Optional[int] = None, *args, **kwargs) -> Dict[str, Any]:
        pass

    @property
    def unwrapped(self) -> "Agent[_ObsT, _ActT]":
        return self

_ObsWT = TypeVar("_ObsWT")
_ActWT = TypeVar("_ActWT")
class AgentWrapper(Generic[_ObsWT, _ActWT], Agent[_ObsWT, _ActWT]):
    def __init__(self, agent : Union["AgentWrapper[_ObsWT, _ActWT]", Agent[_ObsWT, _ActWT]]) -> None:
        Agent.__init__(self)
        self.agent = agent

    def __getattr__(self, __name: str) -> Any:
        if __name.startswith("_"):
            raise AttributeError("Cannot access private attribute")
        return getattr(self.agent, __name)

    def unwrapped(self) -> Agent[_ObsT, _ActT]:
        return self.agent.unwrapped()