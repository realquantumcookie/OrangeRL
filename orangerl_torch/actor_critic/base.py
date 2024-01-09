from orangerl import AgentStage, EnvironmentStep
from orangerl_torch import Tensor_Or_Numpy, Tensor_Or_TensorDict, NNAgent, NNAgentActor, NNAgentCritic, NNBatch
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

    @abstractmethod
    def update_critic(self, minibatch : NNBatch, **kwargs) -> Dict[str, Any]:
        ...
    
    @abstractmethod
    def update_actor(self, minibatch : NNBatch, **kwargs) -> Dict[str, Any]:
        ...