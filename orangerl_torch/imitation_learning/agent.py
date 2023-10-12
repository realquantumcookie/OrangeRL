from typing import Any, Dict, List, Union
from ...base import TransitionBatch, EnvironmentStep, AgentStage
from ..agent import NNAgent
import numpy as np
import torch

from abc import abstractmethod

class ImitationLearningAgent(NNAgent):
    @abstractmethod
    def add_demonstrations(
        self,
        demonstrations : Union[TransitionBatch[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]], EnvironmentStep[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]],
        stage : AgentStage = AgentStage.ONLINE
    ):
       pass