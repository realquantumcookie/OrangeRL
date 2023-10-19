from orangerl.base.data import TransitionBatch, EnvironmentStepInSeq, EnvironmentStep, MutableTransitionSequence, TransitionSequence
from tensordict import tensorclass
from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import PackedSequence
from typing import Optional, Iterable, Union, MutableSequence, Dict, Any, Sequence
import numpy as np
import copy

Tensor_Or_Numpy = Union[torch.Tensor, np.ndarray]

def transform_any_array_to_numpy(
    batch: Union[Iterable[Union[np.ndarray, torch.Tensor]], np.ndarray, torch.Tensor],
):
    if isinstance(batch, (np.ndarray, torch.Tensor)):
        if isinstance(batch, torch.Tensor):
            return batch.detach().cpu().numpy()
        else:
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
class NNBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor

@dataclass
class NNBatchSeq:
    observations: PackedSequence
    actions: PackedSequence
    rewards: PackedSequence
    next_observations: PackedSequence
    terminations: PackedSequence
    truncations: PackedSequence

class NNTransitionSequence(MutableTransitionSequence[torch.Tensor, torch.Tensor]):
    """
    This class represents a sequence of transitions represented as torch tensors.
    """
    
    def __init__(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        terminations: torch.Tensor,
        truncations: torch.Tensor,
        infos: MutableSequence[Dict[str,Any]],
        is_single_episode: bool = False,
        is_time_sorted: bool = False,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.observations = observations.to(device=device, dtype=dtype, non_blocking=True)
        self.actions = actions.to(device=device, dtype=dtype, non_blocking=True)
        self.next_observations = next_observations.to(device=device, dtype=dtype, non_blocking=True)
        self.rewards = rewards.to(device=device, dtype=dtype, non_blocking=True)
        self.terminations = terminations.to(device=device, dtype=torch.bool, non_blocking=True)
        self.truncations = truncations.to(device=device, dtype=torch.bool, non_blocking=True)
        self.infos = infos
        self.is_single_episode = is_single_episode
        self.is_time_sorted = is_time_sorted

    @staticmethod
    def from_steps(
        steps: Iterable[EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy]], 
        is_single_episode: Optional[bool] = None,
        is_time_sorted: Optional[bool] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "NNTransitionSequence":
        if (
            isinstance(steps, TransitionSequence) and
            isinstance(steps.observations, torch.Tensor) and
            isinstance(steps.actions, torch.Tensor) and
            isinstance(steps.next_observations, torch.Tensor) and
            isinstance(steps.rewards, torch.Tensor) and
            isinstance(steps.terminations, torch.Tensor) and
            isinstance(steps.truncations, torch.Tensor)
        ):
            return NNTransitionSequence(
                torch.clone(steps.observations),
                torch.clone(steps.actions),
                torch.clone(steps.next_observations),
                torch.clone(steps.rewards),
                torch.clone(steps.terminations),
                torch.clone(steps.truncations),
                copy.copy(steps.infos),
                batch_size=(steps.observations.size(0),),
                device=device,
                dtype=dtype,
            )

        is_single_episode_override = is_single_episode
        is_time_sorted_override = is_time_sorted
        if isinstance(steps, TransitionBatch):
            is_single_episode_override = steps.is_single_episode
            is_time_sorted_override = steps.is_time_sorted
        
        assert is_single_episode_override is not None and is_time_sorted_override is not None, \
            "is_single_episode and is_time_sorted must be specified when steps is not a TransitionBatch"

        observations = []
        actions = []
        next_observations = []
        rewards = []
        terminations = []
        truncations = []
        infos = []
        for step in steps:
            assert isinstance(step, EnvironmentStep)
            observations.append(transform_any_array_to_torch(step.observation))
            actions.append(transform_any_array_to_torch(step.action))
            next_observations.append(transform_any_array_to_torch(step.next_observation))
            rewards.append(float(step.reward))
            terminations.append(step.terminated)
            truncations.append(step.truncated)
            infos.append(step.info)
        
        observations = torch.stack(observations, dim=0)
        actions = torch.stack(actions, dim=0)
        rewards = torch.tensor(rewards, dtype=dtype, device=device)
        next_observations = torch.stack(next_observations, dim=0)
        terminations = torch.tensor(terminations, dtype=torch.bool, device=device)
        truncations = torch.tensor(truncations, dtype=torch.bool, device=device)

        return __class__(
            observations,
            actions,
            next_observations,
            rewards,
            terminations,
            truncations,
            infos,
            is_single_episode_override,
            is_time_sorted_override,
            device=device,
            dtype=dtype
        )

    def __len__(self) -> int:
        return self.observations.size(0)
    
    def __getitem__(
        self, 
        index: Union[int, slice, Sequence[int], Tensor_Or_Numpy]
    ):
        if isinstance(index, int):
            return EnvironmentStep(
                self.observations[index],
                self.actions[index],
                self.next_observations[index],
                self.rewards[index],
                self.terminations[index],
                self.truncations[index],
                self.infos[index],
            )
        elif isinstance(index, slice):
            indexes_infos = index
        elif isinstance(index, (Sequence, np.ndarray, torch.Tensor)):
            indexes_infos = torch.arange(len(self.infos))[index]
        else:
            raise TypeError("Index must be an integer, slice, sequence of integers, boolean/int tensor/ndarray")

        return __class__(
            self.observations[index],
            self.actions[index],
            self.next_observations[index],
            self.rewards[index],
            self.terminations[index],
            self.truncations[index],
            [self.infos[i] for i in indexes_infos],
            is_single_episode=self.is_single_episode,
            is_time_sorted=self.is_time_sorted
        )

    def __setitem__(
        self,
        index: Union[int, slice, Sequence[int], Tensor_Or_Numpy],
        value: Union[EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy], Iterable[EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy], "NNTransitionSequence"]]
    ):
        if isinstance(index, int):
            assert isinstance(value, EnvironmentStep)
            self.observations[index] = transform_any_array_to_torch(value.observation)
            self.actions[index] = transform_any_array_to_torch(value.action)
            self.next_observations[index] = transform_any_array_to_torch(value.next_observation)
            self.rewards[index] = float(value.reward)
            self.terminations[index] = value.terminated
            self.truncations[index] = value.truncated
            self.infos[index] = value.info
        elif isinstance(index, slice):
            indexes_infos = index
        elif isinstance(index, (Sequence, np.ndarray, torch.Tensor)):
            indexes_infos = torch.arange(len(self.infos))[index]
        else:
            raise TypeError("Index must be an integer, slice, sequence of integers, boolean/int tensor/ndarray")
        
        if isinstance(value, EnvironmentStep):
            self.observations[index] = transform_any_array_to_torch(value.observation).unsqueeze(0)
            self.actions[index] = transform_any_array_to_torch(value.action).unsqueeze(0)
            self.next_observations[index] = transform_any_array_to_torch(value.next_observation).unsqueeze(0)
            self.rewards[index] = float(value.reward)
            self.terminations[index] = value.terminated
            self.truncations[index] = value.truncated
            for info_i in indexes_infos:
                self.infos[info_i] = copy.deepcopy(value.info)
        else:
            if isinstance(value, NNTransitionSequence):
                assert len(value) == len(indexes_infos)
                self.observations[index] = value.observations
                self.actions[index] = value.actions
                self.next_observations[index] = value.next_observations
                self.rewards[index] = value.rewards
                self.terminations[index] = value.terminations
                self.truncations[index] = value.truncations
                if isinstance(indexes_infos, slice):
                    self.infos[indexes_infos] = value.infos
                else:
                    for i, v in zip(indexes_infos, value.infos):
                        self.infos[i] = v
            
            if not isinstance(value, Iterable):
                raise TypeError("Value must be an iterable of EnvironmentStep or EnvironmentStep")

            if isinstance(indexes_infos, slice):
                indexes_infos = list(range(len(self.infos)))[indexes_infos]
            
            for i, v in zip(indexes_infos, value):
                assert isinstance(v, EnvironmentStep)
                self.observations[i] = transform_any_array_to_torch(v.observation)
                self.actions[i] = transform_any_array_to_torch(v.action)
                self.next_observations[i] = transform_any_array_to_torch(v.next_observation)
                self.rewards[i] = float(v.reward)
                self.terminations[i] = v.terminated
                self.truncations[i] = v.truncated
                self.infos[i] = v.info

    def __delitem__(self, index: Union[int, slice, Tensor_Or_Numpy]) -> None:
        if isinstance(index, int):
            self.observations = torch.cat([self.observations[:index], self.observations[index+1:]], dim=0)
            self.actions = torch.cat([self.actions[:index], self.actions[index+1:]], dim=0)
            self.next_observations = torch.cat([self.next_observations[:index], self.next_observations[index+1:]], dim=0)
            self.rewards = torch.cat([self.rewards[:index], self.rewards[index+1:]], dim=0)
            self.terminations = torch.cat([self.terminations[:index], self.terminations[index+1:]], dim=0)
            self.truncations = torch.cat([self.truncations[:index], self.truncations[index+1:]], dim=0)
            del self.infos[index]
        elif isinstance(index, slice):
            indexes_infos = list(range(len(self.infos)))[index]
        elif isinstance(index, (Sequence, np.ndarray, torch.Tensor)):
            indexes_infos = list(torch.arange(len(self.infos))[index])
        else:
            raise TypeError("Index must be an integer, slice, sequence of integers, boolean/int tensor/ndarray")
        
        indexes_infos.sort()
        keep_indexes = sorted(set(range(len(self.infos))).difference(indexes_infos))
        self.observations = self.observations[keep_indexes]
        self.actions = self.actions[keep_indexes]
        self.next_observations = self.next_observations[keep_indexes]
        self.rewards = self.rewards[keep_indexes]
        self.terminations = self.terminations[keep_indexes]
        self.truncations = self.truncations[keep_indexes]
        self.infos = [self.infos[i] for i in keep_indexes]
        
    def __add__(self, other: Union[EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy], Iterable[EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy]]]) -> "NNTransitionSequence":
        if isinstance(other, EnvironmentStep):
            return __class__(
                torch.cat([self.observations, transform_any_array_to_torch(other.observation).unsqueeze(0)], dim=0),
                torch.cat([self.actions, transform_any_array_to_torch(other.action).unsqueeze(0)], dim=0),
                torch.cat([self.next_observations, transform_any_array_to_torch(other.next_observation).unsqueeze(0)], dim=0),
                torch.cat([self.rewards, torch.tensor([float(other.reward)], dtype=self.rewards.dtype, device=self.rewards.device)], dim=0),
                torch.cat([self.terminations, torch.tensor([other.terminated], dtype=torch.bool, device=self.terminations.device)], dim=0),
                torch.cat([self.truncations, torch.tensor([other.truncated], dtype=torch.bool, device=self.truncations.device)], dim=0),
                self.infos + [copy.deepcopy(other.info)],
                is_single_episode=self.is_single_episode,
                is_time_sorted=self.is_time_sorted
            )
        else:
            other_obs = 

    def to(
        self,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
        to_copy : bool = False,
    ) -> "NNTransitionSequence":
        new_observations = self.observations.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=to_copy)
        new_actions = self.actions.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=to_copy)
        new_next_observations = self.next_observations.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=to_copy)
        new_rewards = self.rewards.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=to_copy)
        new_terminations = self.terminations.to(device=device, dtype=torch.bool, non_blocking=non_blocking, copy=to_copy)
        new_truncations = self.truncations.to(device=device, dtype=torch.bool, non_blocking=non_blocking, copy=to_copy)

        new_obj = not(
            new_observations is self.observations and
            new_actions is self.actions and
            new_next_observations is self.next_observations and
            new_rewards is self.rewards and
            new_terminations is self.terminations and
            new_truncations is self.truncations
        )

        if new_obj:
            return __class__(
                new_observations,
                new_actions,
                new_next_observations,
                new_rewards,
                new_terminations,
                new_truncations,
                copy.copy(self.infos),
                is_single_episode=self.is_single_episode,
                is_time_sorted=self.is_time_sorted,
                device=device,
                dtype=dtype
            )
        else:
            return self


class NNReplayBuffer(MutableTransitionSequence[torch.Tensor, torch.Tensor]):
    def __init__(
        self,
        capacity: int,
        observation_shape: torch.Size,
        action_shape: torch.Size,
        is_time_sorted: bool = False,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.observations_buf : torch.Tensor = torch.empty((capacity, *observation_shape), device=device, dtype=dtype)
        self.actions_buf : torch.Tensor = torch.empty((capacity, *action_shape), device=device, dtype=dtype)
        self.rewards_buf : torch.Tensor = torch.empty((capacity,), device=device, dtype=dtype)
        self.next_observations_buf : torch.Tensor = torch.empty((capacity, *observation_shape), device=device, dtype=dtype)
        self.terminations_buf : torch.Tensor = torch.empty((capacity,), device=device, dtype=torch.bool)
        self.truncations_buf : torch.Tensor = torch.empty((capacity,), device=device, dtype=torch.bool)
        
        self.start_transition_step : int = 0 # The index of the earlier transition in the buffer
        
        self.next_data_idx : int = 0
        self.transition_count : int = 0
        self.is_single_episode : bool = False
        self.is_time_sorted : bool = is_time_sorted
    
    @property
    def capacity(self) -> int:
        return self.observations.size(0)

    def __len__(self) -> int:
        return self.transition_count
    
    def append(self, value : EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy]) -> None:
        assert isinstance(value.observation, (np.ndarray, torch.Tensor)), "Observation must be a numpy array or torch tensor"
        assert isinstance(value.action, (np.ndarray, torch.Tensor)), "Action must be a numpy array or torch tensor"
        assert isinstance(value.next_observation, (np.ndarray, torch.Tensor)), "Next observation must be a numpy array or torch tensor"

        self.observations[self.next_data_idx] = transform_any_array_to_torch(value.observation)
        self.actions[self.next_data_idx] = transform_any_array_to_torch(value.action)
        self.rewards[self.next_data_idx] = float(value.reward)
        self.next_observations[self.next_data_idx] = transform_any_array_to_torch(value.next_observation)
        self.terminations[self.next_data_idx] = value.terminated
        self.truncations[self.next_data_idx] = value.truncated
        
        self.next_data_idx += 1
        self.transition_count += 1
        if self.next_data_idx >= self.capacity:
            self.next_data_idx = 0
            self.is_single_episode = False
