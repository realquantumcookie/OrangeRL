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

from ..base import Agent, AgentStage, TransitionBatch, EnvironmentStep, EpisodeRollout
from ..data import ReplayBuffer
from typing import Any, Optional, Union, Dict, Iterable
import numpy as np
import torch

_BCAgentObsAndActType = Union[np.ndarray, torch.Tensor]

class BehaviorCloningAgent(Agent[_BCAgentObsAndActType, _BCAgentObsAndActType]):
    def __init__(
        self, 
        network : torch.nn.Module,
        optimizer : torch.optim.Optimizer,
        loss_fn : torch.nn.Module = torch.nn.MSELoss(),
        replay_buffer : ReplayBuffer[np.ndarray, np.ndarray] = ReplayBuffer(),
        device : Optional[torch.device] = None
    ) -> None:
        super().__init__()
        self.network = network.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn.to(device)
        self.replay_buffer = replay_buffer
        self.device = device
    
    def get_action(self, observation : _BCAgentObsAndActType, stage : AgentStage = AgentStage.ONLINE) -> _BCAgentObsAndActType:
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).to(self.device)
        else:
            observation = observation.to(self.device)
        observation = torch.reshape(observation, (1, *observation.shape))
        with torch.no_grad():
            action : torch.Tensor = self.network.forward(observation)
        
        return action.squeeze(0)
    
    def get_action_batch(self, observation: _BCAgentObsAndActType | Iterable[_BCAgentObsAndActType], stage: AgentStage = AgentStage.ONLINE) -> _BCAgentObsAndActType | Iterable[_BCAgentObsAndActType]:
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).to(self.device)
        elif isinstance(observation, torch.Tensor):
            observation = observation.to(self.device)
        else:
            if len(observation) == 0:
                return []
            
            if isinstance(observation[0], np.ndarray):
                observation = torch.from_numpy(
                    np.stack(observation, axis=0)
                ).to(self.device)
            elif isinstance(observation[0], torch.Tensor):
                observation = torch.stack(observation, axis=0).to(self.device)
            else:
                raise ValueError("Invalid observation type")

        with torch.no_grad():
            action : torch.Tensor = self.network.forward(observation)
        
        return action
    
    def add_transitions(self, transition: TransitionBatch[_BCAgentObsAndActType, _BCAgentObsAndActType] | EnvironmentStep[_BCAgentObsAndActType, _BCAgentObsAndActType]):
        if isinstance(transition, EnvironmentStep):
            to_add = EnvironmentStep(
                observation=transition.observation if isinstance(transition.observation, np.ndarray) else transition.observation.cpu().numpy(),
                action=transition.action if isinstance(transition.action, np.ndarray) else transition.action.cpu().numpy(),
                next_observation=transition.next_observation if isinstance(transition.next_observation, np.ndarray) else transition.next_observation.cpu().numpy(),
                reward=transition.reward,
                terminated=transition.terminated,
                truncated=transition.truncated,
                info=transition.info
            )
            self.replay_buffer.add_transition(to_add)
        else:
            to_add = TransitionBatch(
                observations=[obs if isinstance(obs, np.ndarray) else obs.cpu().numpy() for obs in transition.observations],
                actions=[act if isinstance(act, np.ndarray) else act.cpu().numpy() for act in transition.actions],
                next_observations=[next_obs if isinstance(next_obs, np.ndarray) else next_obs.cpu().numpy() for next_obs in transition.next_observations],
                rewards=transition.rewards,
                terminations=transition.terminations,
                truncations=transition.truncations,
                infos=transition.infos
            )
            self.replay_buffer.add_transition(to_add)

    def update(self, batch_size : Optional[int] = None, *args, **kwargs) -> Dict[str, Any]:
        assert batch_size is not None and batch_size > 0, "Invalid batch size"
        batch = self.replay_buffer.sample(batch_size, np.random.Generator(np.random.PCG64()))
        observations = torch.from_numpy(np.stack(batch.observations, axis=0)).to(self.device)
        actions = torch.from_numpy(np.stack(batch.actions, axis=0)).to(self.device)
        target_actions = torch.from_numpy(np.stack(batch.actions, axis=0)).to(self.device)
        loss = self.loss_fn(self.network.forward(observations), target_actions)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        ret_info = {
            "Actor Loss": loss.item()
        }
        return ret_info