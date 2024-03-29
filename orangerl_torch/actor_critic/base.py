from orangerl import AgentStage, EnvironmentStep
from orangerl_torch import Tensor_Or_Numpy, Tensor_Or_TensorDict, NNAgent, NNAgentActor, NNAgentCritic, NNBatch, BatchedNNAgentOutput
from typing import Any, Iterator, Optional, Union, Iterable, Tuple, Dict, Generic, TypeVar, Callable
import gymnasium as gym
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence
import numpy as np
from tensordict import TensorDictBase, TensorDict
import copy
from abc import abstractmethod, ABC

class NNActorCriticAgent(NNAgent, ABC):
    actor : NNAgentActor
    critic : NNAgentCritic
    utd_ratio : int = 1
    actor_delay : int = 1

    @property
    def current_stage(self) -> AgentStage:
        return self._current_stage
    
    @current_stage.setter
    def current_stage(self, stage : AgentStage) -> None:
        if stage == AgentStage.EVAL:
            self.train(False)
        else:
            self.train(True)
        
        self._current_stage = stage
        self.actor.current_stage = stage

    def forward(
        self,
        obs_batch: Tensor_Or_TensorDict,
        masks: Optional[torch.Tensor] = None,
        state: Optional[Tensor_Or_TensorDict] = None,
        is_update = False,
        **kwargs: Any,
    ) -> BatchedNNAgentOutput:
        return self.actor.forward(
            obs_batch,
            masks,
            state,
            is_update,
            **kwargs
        )

    @abstractmethod
    def update_critic(self, minibatch : NNBatch, **kwargs) -> Dict[str, Any]:
        ...
    
    @abstractmethod
    def update_actor(self, minibatch : NNBatch, **kwargs) -> Dict[str, Any]:
        ...