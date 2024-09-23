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

from orangerl import TransitionBatch, EnvironmentStep, TransitionSequence
from tensordict import tensorclass, TensorDictBase, TensorDict, is_tensor_collection
import torch
from typing import Iterator, Optional, Iterable, Union, MutableSequence, Dict, Any, Sequence
import numpy as np
from functools import reduce

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

def dict_to_tensor_dict(info_dict : Dict[str, Any]) -> TensorDictBase:
    if is_tensor_collection(info_dict):
        return info_dict
    else:
        copy_dict = {}
        for k, v in info_dict.items():
            if is_tensor_collection(v) or isinstance(v, torch.Tensor):
                copy_dict[k] = v
            if isinstance(v, dict):
                copy_dict[k] = dict_to_tensor_dict(v)
            elif isinstance(v, np.ndarray):
                copy_dict[k] = torch.from_numpy(v)
            elif isinstance(v, int):
                copy_dict[k] = torch.tensor(v, dtype=torch.int)
            elif isinstance(v, float):
                copy_dict[k] = torch.tensor(v, dtype=torch.float)
            elif isinstance(v, bool):
                copy_dict[k] = torch.tensor(v, dtype=torch.bool)
            else:
                continue
        return TensorDict(copy_dict, batch_size=())

@tensorclass
class TorchTransitionBatch(TransitionSequence[torch.Tensor, torch.Tensor]):
    """
    TorchTransitionBatch is a dataclass that represents a batch of transitions.
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
    
    @property
    def transition_len(self) -> int:
        if self.masks is None:
            return int(torch.prod(self.batch_size).item())
        else:
            return torch.count_nonzero(self.masks).to(torch.int).item()

    def iter_transitions(self) -> Iterator[EnvironmentStep[torch.Tensor, torch.Tensor]]:
        flattened_masked_self : TorchTransitionBatch = self.flatten() if self.masks is None else self[self.masks.bool()]
        for i in range(flattened_masked_self.batch_size[0]):
            yield EnvironmentStep(
                flattened_masked_self.observations[i],
                flattened_masked_self.actions[i],
                flattened_masked_self.next_observations[i],
                flattened_masked_self.rewards[i].item(),
                flattened_masked_self.terminations[i] > 0,
                flattened_masked_self.truncations[i] > 0,
                flattened_masked_self.infos[i] if flattened_masked_self.infos is not None else None,
            )
    
    def transitions_at(self, index: Union[int, slice, Sequence[int]]) -> Union[EnvironmentStep[torch.Tensor, torch.Tensor], Sequence[EnvironmentStep[torch.Tensor, torch.Tensor]]]:
        flattened_masked_self : TorchTransitionBatch = self.flatten() if self.masks is None else self[self.masks.bool()]
        if isinstance(index, int):
            return EnvironmentStep(
                flattened_masked_self.observations[index],
                flattened_masked_self.actions[index],
                flattened_masked_self.next_observations[index],
                flattened_masked_self.rewards[index].item(),
                flattened_masked_self.terminations[index] > 0,
                flattened_masked_self.truncations[index] > 0,
                flattened_masked_self.infos[index] if flattened_masked_self.infos is not None else None,
            )
        elif isinstance(index, slice):
            return [
                EnvironmentStep(
                    flattened_masked_self.observations[i],
                    flattened_masked_self.actions[i],
                    flattened_masked_self.next_observations[i],
                    flattened_masked_self.rewards[i].item(),
                    flattened_masked_self.terminations[i] > 0,
                    flattened_masked_self.truncations[i] > 0,
                    flattened_masked_self.infos[i] if flattened_masked_self.infos is not None else None,
                ) for i in range(*index.indices(flattened_masked_self.batch_size[0]))
            ]
        else:  # Sequence[int]
            return [
                EnvironmentStep(
                    flattened_masked_self.observations[i],
                    flattened_masked_self.actions[i],
                    flattened_masked_self.next_observations[i],
                    flattened_masked_self.rewards[i].item(),
                    flattened_masked_self.terminations[i] > 0,
                    flattened_masked_self.truncations[i] > 0,
                    flattened_masked_self.infos[i] if flattened_masked_self.infos is not None else None,
                ) for i in index
            ]
    
    @staticmethod
    def from_transitions(
        transitions : Iterable[EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy]],
        save_info : bool = False
    ) -> 'TorchTransitionBatch':
        if isinstance(transitions, TorchTransitionBatch):
            return transitions
        elif is_tensor_collection(transitions):
            return TorchTransitionBatch.from_tensordict(transitions)
        
        transitions = transitions if isinstance(transitions, Sequence) else list(transitions)
        transformed_infos = torch.stack([dict_to_tensor_dict(step.info) for step in transitions]) if save_info and reduce(
            lambda x, y: x and y, 
            [step.info is not None for step in transitions],
            True
        ) else None

        ret_data = TorchTransitionBatch(
            observations=torch.stack([transform_any_array_to_torch(step.observation) for step in transitions]),
            actions=torch.stack([transform_any_array_to_torch(step.action) for step in transitions]),
            rewards=torch.tensor([step.reward for step in transitions], dtype=torch.float32),
            next_observations=transform_any_array_to_torch([step.next_observation for step in transitions]),
            terminations=torch.tensor([step.terminated for step in transitions], dtype=torch.bool),
            truncations=torch.tensor([step.truncated for step in transitions], dtype=torch.bool),
            masks=None,
            infos=transformed_infos,
            batch_size=(len(transitions),)
        )
        return ret_data

