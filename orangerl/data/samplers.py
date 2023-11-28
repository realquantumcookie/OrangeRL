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

from ..base.data import TransitionBatch, EnvironmentStep, TransitionSampler
from typing import TypeVar, Generic, Optional
import numpy as np

_ObsT = TypeVar("_ObsT")
_ActT = TypeVar("_ActT")
class CommonTransitionSamplerBase(Generic[_ObsT, _ActT], TransitionSampler[_ObsT, _ActT]):
    def __init__(self, seed : Optional[int] = None):
        self.randomness = np.random.default_rng(seed)
    
    def seed(self, seed: Optional[int] = None) -> None:
        self.randomness = np.random.default_rng(seed)

class UniformTransitionSampler(CommonTransitionSamplerBase[_ObsT, _ActT]):
    def sample_idx(
        self,
        transition_batch : TransitionBatch[_ObsT, _ActT],
        batch_size : int,
        repeat_sample = True,
    ) -> np.ndarray:
        assert batch_size > 0, "Batch size must be positive"

        data_length = len(transition_batch)
        index_list = np.arange(data_length)

        if batch_size > data_length:
            if repeat_sample:
                index_list = np.tile(index_list, batch_size // data_length + 1)
            else:
                raise ValueError("Batch size is larger than data length")
        
        self.randomness.shuffle(index_list)
        return index_list[:batch_size]