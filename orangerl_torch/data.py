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

from orangerl.base.data import TransitionBatch, EnvironmentStep, TransitionSequence
from tensordict import tensorclass, TensorDictBase, TensorDict
import torch
from typing import Iterator, Optional, Iterable, Union, MutableSequence, Dict, Any, Sequence
import numpy as np

Tensor_Or_Numpy = Union[torch.Tensor, np.ndarray]

def transform_any_array_to_numpy(
    batch: Union[Iterable[Union[np.ndarray, torch.Tensor]], np.ndarray, torch.Tensor],
):
    if isinstance(batch, torch.Tensor):
        return batch.detach().cpu().numpy()
    elif isinstance(batch, np.ndarray):
        return batch
    else:
        return np.stack([transform_any_array_to_numpy(b) for b in batch])

def transform_any_array_to_torch(
    batch,
):
    if isinstance(batch, np.ndarray):
        return torch.from_numpy(batch)
    elif isinstance(batch, torch.Tensor):
        return batch
    elif isinstance(batch, Iterable):
        return torch.stack([transform_any_array_to_torch(b) for b in batch])
    else:
        return torch.asarray(batch)

@tensorclass
class NNBatch(TransitionSequence[torch.Tensor, torch.Tensor]):
    """
    NNBatch is a dataclass that represents a batch of transitions.
    Batch size should be a tuple of integers, with (B,) or (B, sequence_length).
    If the batch_size is set to (B, sequence_length), then the batch is treated as a batch of sequences.
    """

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    masks: Optional[torch.Tensor]
    infos: Optional[TensorDict]
    is_transition_single_episode : bool
    is_transition_time_sorted : bool
    
    @property
    def transition_len(self) -> int:
        assert self.batch_dims >= 1 and self.batch_dims <= 2
        if self.masks is None:
            return int(torch.prod(self.batch_size).item())
        else:
            return torch.count_nonzero(self.masks).to(torch.int).item()

    def iter_transitions(self) -> Iterator[EnvironmentStep[torch.Tensor, torch.Tensor]]:
        assert self.batch_dims >= 1 and self.batch_dims <= 2
        flattened_self : NNBatch = self.flatten()
        for i in range(flattened_self.batch_size[0]):
            yield EnvironmentStep(
                flattened_self.observations[i],
                flattened_self.actions[i],
                flattened_self.next_observations[i],
                flattened_self.rewards[i].item(),
                flattened_self.terminations[i] > 0,
                flattened_self.truncations[i] > 0,
                flattened_self.infos[i] if flattened_self.infos is not None else None,
            )
    
    def transitions_at(self, index: Union[int, slice, Sequence[int]]) -> Union[EnvironmentStep[torch.Tensor, torch.Tensor], Sequence[EnvironmentStep[torch.Tensor, torch.Tensor]]]:
        assert self.batch_dims >= 1 and self.batch_dims <= 2
        flattened_self : NNBatch = self.flatten()
        if isinstance(index, int):
            return EnvironmentStep(
                flattened_self.observations[index],
                flattened_self.actions[index],
                flattened_self.next_observations[index],
                flattened_self.rewards[index].item(),
                flattened_self.terminations[index] > 0,
                flattened_self.truncations[index] > 0,
                flattened_self.infos[index] if flattened_self.infos is not None else None,
            )
        elif isinstance(index, slice):
            return [
                EnvironmentStep(
                    flattened_self.observations[i],
                    flattened_self.actions[i],
                    flattened_self.next_observations[i],
                    flattened_self.rewards[i].item(),
                    flattened_self.terminations[i] > 0,
                    flattened_self.truncations[i] > 0,
                    flattened_self.infos[i] if flattened_self.infos is not None else None,
                ) for i in range(*index.indices(flattened_self.batch_size[0]))
            ]
        else:  # Sequence[int]
            return [
                EnvironmentStep(
                    flattened_self.observations[i],
                    flattened_self.actions[i],
                    flattened_self.next_observations[i],
                    flattened_self.rewards[i].item(),
                    flattened_self.terminations[i] > 0,
                    flattened_self.truncations[i] > 0,
                    flattened_self.infos[i] if flattened_self.infos is not None else None,
                ) for i in index
            ]

