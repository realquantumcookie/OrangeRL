from ..base.data import TransitionBatch, EnvironmentStep, TransitionSampler
from typing import Any, TypeVar, Generic, Optional, Union, Dict, Iterable, List, SupportsFloat, Sequence
import numpy as np

try:
    import torch
except ImportError:
    torch = None

class CommonSamplerBase(TransitionSampler):
    def __init__(self, seed : Optional[int] = None):
        self.randomness = np.random.default_rng(seed)
    
    def seed(self, seed: Optional[int] = None) -> None:
        self.randomness = np.random.default_rng(seed)


class UniformSampler(CommonSamplerBase):
    """
    To support uniform sampling in a custom Trasition Batch much faster, it is recommended to implement the __getitem__ method to support indexing through a list of indexes.
    """

    def sample(
        self,
        transition_batch : TransitionBatch,
        batch_size : int,
        repeat_sample = True,
    ) -> Sequence[EnvironmentStep]:
        assert batch_size > 0, "Batch size must be positive"

        data_length = len(transition_batch)
        index_list = np.arange(data_length)

        if batch_size < data_length:
            self.randomness.shuffle(index_list)
            try:
                return transition_batch[index_list[:batch_size]]
            except:
                pass
        
        
            try:
                to_ret = []
                for i in index_list[:batch_size]:
                    to_ret.append(transition_batch[i])
                return to_ret
            except:
                pass

        elif batch_size == data_length:
            return transition_batch

        num_selected_per_index = np.zeros(data_length, dtype=np.int32)
        num_selected = 0

        while num_selected < batch_size:
            remaining_to_select = batch_size - num_selected

            if data_length <= remaining_to_select:
                num_selected_per_index[:] += 1
                num_selected += data_length
                if not repeat_sample:
                    break
            else:
                self.randomness.shuffle(index_list)
                num_selected_per_index[index_list[:remaining_to_select]] += 1
                num_selected += remaining_to_select
        
        new_steps : List[EnvironmentStep] = []
        for idx, step in enumerate(transition_batch):
            if num_selected_per_index[idx] > 0:
                new_steps.append([step] * num_selected_per_index[idx])

        return new_steps