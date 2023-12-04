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

from ..base.data import TransitionBatch, EnvironmentStep, TransitionSampler, TransitionReplayBuffer
from typing import TypeVar, Generic, Optional, Tuple, List, Any
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
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        assert batch_size > 0, "Batch size must be positive"

        data_length = len(transition_batch)
        index_list = np.arange(data_length, dtype=np.int64)

        if batch_size > data_length:
            if repeat_sample:
                index_list = np.tile(index_list, batch_size // data_length + 1)
            else:
                raise ValueError("Batch size is larger than data length")
        
        self.randomness.shuffle(index_list)
        return index_list[:batch_size], None

class UniformSequencedTransitionSampler(UniformTransitionSampler[_ObsT, _ActT]):
    """
    This class will sample a sequence of transitions indexes shaped as (batch_size, sequence_length)
    """

    def __init__(
        self, 
        max_sequence_length : Optional[int],
        seed: Optional[int] = None
    ):
        super().__init__(seed)
        assert max_sequence_length is None or max_sequence_length > 0, "Max sequence length must be positive"
        self.max_sequence_length = max_sequence_length
    
    def sample_idx(
        self,
        transition_batch : TransitionReplayBuffer[_ObsT, _ActT, Any],
        batch_size : int,
        repeat_sample = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        unif_sampled_idx, _ = super().sample_idx(transition_batch, batch_size, repeat_sample)
        episodes_idx = transition_batch.idx_episode_begins_and_ends()
        
        transition_length_batch = transition_batch.transition_len
        idx_lists : List[np.ndarray] = []
        for seq_end_idx in unif_sampled_idx:
            current_seq_list = []
            episode_start = -1

            for t_episode_start, t_episode_end in episodes_idx:
                if t_episode_start <= seq_end_idx <= t_episode_end or t_episode_start <= seq_end_idx <= t_episode_end - transition_length_batch:
                    episode_start = t_episode_start
                    break
            
            assert episode_start >= 0, "Cannot find episode for sampled index"
            start = episode_start if self.max_sequence_length is None else max(episode_start, seq_end_idx - self.max_sequence_length + 1)
            current_seq_list = np.arange(start, seq_end_idx + 1, dtype=np.int64)
            idx_lists.append(current_seq_list)
        
        max_seq_len = max([len(idx_list) for idx_list in idx_lists])
        masks = np.zeros((len(idx_lists), max_seq_len), dtype=np.bool_)
        idx_mat = np.zeros_like(masks, dtype=np.int64)
        for i, idx_list in enumerate(idx_lists):
            masks[i, :len(idx_list)] = True
            idx_mat[i, :len(idx_list)] = idx_list
        
        idx_mat %= transition_length_batch
        return idx_mat, masks
