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
from orangerl.nn.data_util import Optional
from .agent import ImitationLearningAgent
from ..agent import NNAgentActionMapper
from ..data_util import transform_transition_batch_to_torch_tensor
from ...data import ReplayBuffer
from typing import Any, Optional, Tuple, Union, Dict
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym


class BehaviorCloningAgent(ImitationLearningAgent):
    def __init__(
        self, 
        network : nn.Module,
        optimizer : torch.optim.Optimizer,
        action_mapper: NNAgentActionMapper,
        loss_fn : torch.nn.Module = torch.nn.MSELoss(),
        replay_buffer : ReplayBuffer[np.ndarray, np.ndarray] = ReplayBuffer(),
        is_sequence_model : bool = False,
        device : Optional[Union[torch.device, str]] = None
    ) -> None:
        super().__init__(device = device)
        self._action_mapper = action_mapper
        self.network = network.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn.to(device)
        self.replay_buffer = replay_buffer
        self._is_sequence_model = is_sequence_model
    
    @property
    def action_mapper(self) -> NNAgentActionMapper:
        return self._action_mapper

    @action_mapper.setter
    def action_mapper(self, value : NNAgentActionMapper):
        self._action_mapper = value

    @property
    def is_sequence_model(self) -> bool:
        return self._is_sequence_model
    
    @is_sequence_model.setter
    def is_sequence_model(self, value : bool):
        self._is_sequence_model = value
    
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
        demonstrations : Union[TransitionBatch[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]], EnvironmentStep[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]],
        stage : AgentStage = AgentStage.ONLINE
    ):
        # TODO: Convert demonstrations to numpy arrays and insert into replay buffer
        self.replay_buffer.add_transition(demonstrations)

    def update(self, batch_size: int | None = None, *args, **kwargs) -> Dict[str, Any]:
        if batch_size is None:
            batch_size = self.replay_buffer.length
        
        is_sequence_data = False
        update_data = None
        if self._next_update_cache is None or self._next_update_cache["batch_size"] != batch_size:
            self.prefetch_update_data(batch_size=batch_size, *args, **kwargs)
        is_sequence_data = self._next_update_cache["is_sequence_data"]
        update_data = self._next_update_cache["update_data"]
        if is_sequence_data:
            all_acts = []
            for obs, exp_act in update_data["obs"]:
                act = self.forward(obs)
                all_acts.append(act)
            
            all_acts = torch.cat(all_acts, dim=0)
            exp_acts = torch.cat(update_data["exp_act"], dim=0)
            loss = self.loss_fn(all_acts, exp_acts)
        else:
            obs = update_data["obs"]
            exp_act = update_data["exp_act"]
            act = self.forward(obs)
            loss = self.loss_fn(act, exp_act)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def prefetch_update_data(self, batch_size: int | None = None, *args, **kwargs) -> None:
        assert batch_size is not None and batch_size > 0, "batch_size must be specified"
        if self.is_sequence_model:
            sequences = self.replay_buffer.sample_sequences(transition_length_limit=batch_size, randomness=self.randomness)
            all_obs = []
            all_exp_acts = []

            for seq in sequences:
                obs, exp_act, _ = transform_transition_batch_to_torch_tensor(seq)
                obs = obs.to(self.device, non_blocking=True)
                exp_act = exp_act.to(self.device, non_blocking=True)
                all_obs.append(obs)
                all_exp_acts.append(exp_act)
            
            self._next_update_cache = {
                "is_sequence_data": True,
                "update_data": {
                    "obs": all_obs,
                    "exp_act": all_exp_acts
                },
                "batch_size": batch_size,
            }

        else:
            batch = self.replay_buffer.sample(batch_size, randomness=self.randomness)
            obs, exp_act, _ = transform_transition_batch_to_torch_tensor(batch)
            obs = obs.to(self.device, non_blocking=True)
            exp_act = exp_act.to(self.device, non_blocking=True)
            self._next_update_cache = {
                "is_sequence_data": False,
                "update_data": {
                    "obs": obs,
                    "exp_act": exp_act
                },
                "batch_size": batch_size,
            }
