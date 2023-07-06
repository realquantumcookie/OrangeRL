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

from orangerl.base.agent import AgentStage
from orangerl.base.data import EnvironmentStep, TransitionBatch
from ..agent import NNAgent, NNAgentActionMapper
from ...data import ReplayBuffer
from typing import Any, Optional, Tuple, Union, Dict, Iterable
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym


class BehaviorCloningAgent(NNAgent):
    def __init__(
        self, 
        network : nn.Module,
        optimizer : torch.optim.Optimizer,
        action_mapper: NNAgentActionMapper,
        loss_fn : torch.nn.Module = torch.nn.MSELoss(),
        replay_buffer : ReplayBuffer[np.ndarray, np.ndarray] = ReplayBuffer(),
        device : Optional[Union[torch.device, str]] = None
    ) -> None:
        super().__init__(action_mapper=action_mapper, device = device)
        self.network = network.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn.to(device)
        self.replay_buffer = replay_buffer
    
    def forward(
        self, 
        batch: torch.Tensor, 
        state: Optional[Any] = None, 
        stage: AgentStage = AgentStage.ONLINE
    ) -> torch.Tensor | Tuple[torch.Tensor, Any]:
        if stage != AgentStage.EVAL:
            self.train()
        else:
            self.eval()
        
        return self.network(batch) if state is None else self.network(batch, state=state)

    def add_transitions(
        self, 
        transition : Union[TransitionBatch[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]], EnvironmentStep[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]],
        stage : AgentStage = AgentStage.ONLINE
    ):
        # BC Agent doesn't care about transitions
        pass

    def add_demonstrations(
        self,
        demonstrations : Iterable[TransitionBatch[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]],
        stage : AgentStage = AgentStage.ONLINE
    ):
        # TODO: Convert demonstrations to numpy arrays and insert into replay buffer
        pass

    def update(self, batch_size: int | None = None, *args, **kwargs) -> Dict[str, Any]:
        return super().update(batch_size, *args, **kwargs)
    