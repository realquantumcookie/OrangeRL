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

from orangerl import TransitionBatch, EnvironmentStep, EnvironmentStepInSeq, AgentOutput
from typing import Dict, Any, Tuple, Iterable, List, Union, Optional, Callable
import torch
from torch.nn.utils.rnn import PackedSequence, pack_sequence
import numpy as np
from .data import NNBatch, NNBatchSeq, Tensor_Or_Numpy, transform_any_array_to_torch

def transform_sampled_batch_to_nonseq_model_input(
    batch: Iterable[EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy]],
    relabel_fn: Callable[[EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy]], EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy]] = lambda x: x,
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
) -> NNBatch:
    """
    Transforms a TransitionBatch to NN friendly format
    returns (observations, actions, next_observations)
    """
    observations = []
    actions = []
    rewards: List[float] = []
    next_observations = []
    terminations: List[bool] = []
    truncations: List[bool] = []
    
    for transition in batch:
        transition = relabel_fn(transition)
        assert isinstance(transition.observation, (np.ndarray, torch.Tensor)), "Observation must be a numpy array or torch tensor"
        assert isinstance(transition.action, (np.ndarray, torch.Tensor)), "Action must be a numpy array or torch tensor"
        assert isinstance(transition.next_observation, (np.ndarray, torch.Tensor)), "Next observation must be a numpy array or torch tensor"

        observations.append(
            transform_any_array_to_torch(transition.observation)
        )
        actions.append(
            transform_any_array_to_torch(transition.action)
        )
        rewards.append(float(transition.reward))
        next_observations.append(
            transform_any_array_to_torch(transition.next_observation)
        )
        terminations.append(transition.terminated)
        truncations.append(transition.truncated)

    observations = torch.stack(observations, dim=0).to(device=device, dtype=dtype, non_blocking=True)
    actions = torch.stack(actions, dim=0).to(device=device, dtype=dtype, non_blocking=True)
    rewards = torch.tensor(rewards, dtype=dtype, device=device)
    next_observations = torch.stack(next_observations, dim=0).to(device=device, dtype=dtype, non_blocking=True)
    terminations = torch.tensor(terminations, dtype=torch.bool, device=device)
    truncations = torch.tensor(truncations, dtype=torch.bool, device=device)
    return NNBatch(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminations=terminations,
        truncations=truncations,
        batch_size=(actions.size(0),),
    )

def transform_sampled_batch_to_seq_model_input(
    batch: Iterable[EnvironmentStepInSeq[Tensor_Or_Numpy, Tensor_Or_Numpy]],
    max_seq_len: int,
    relabel_fn: Callable[[EnvironmentStepInSeq[Tensor_Or_Numpy, Tensor_Or_Numpy], Optional[int]], EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy]] = lambda x: x,
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
) -> NNBatchSeq:
    sequences_obs = []
    sequences_act = []
    sequences_rew = []
    sequences_next_obs = []
    sequences_term = []
    sequences_trunc = []

    cache_map : Dict[int, Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor
    ]] = {}

    for step in batch:
        if step.idx in cache_map.keys():
            sequences_obs.append(cache_map[step.idx][0])
            sequences_act.append(cache_map[step.idx][1])
            sequences_rew.append(cache_map[step.idx][2])
            sequences_next_obs.append(cache_map[step.idx][3])
            sequences_term.append(cache_map[step.idx][4])
            sequences_trunc.append(cache_map[step.idx][5])
            continue

        current_seq_obs = []
        current_seq_act = []
        current_seq_rew = []
        current_seq_next_obs = []
        current_seq_term = []
        current_seq_trunc = []

        current_step_cur = step
        traced_steps = 0

        
        while current_step_cur is not None:
            if max_seq_len > 0 and traced_steps >= max_seq_len:
                break
            
            r_current_step_cur = relabel_fn(current_step_cur, traced_steps)
            if traced_steps > 0 and (
                r_current_step_cur.terminated or 
                r_current_step_cur.truncated
            ):
                break

            assert isinstance(r_current_step_cur.observation, (np.ndarray, torch.Tensor)), "Observation must be a numpy array or torch tensor"
            assert isinstance(r_current_step_cur.action, (np.ndarray, torch.Tensor)), "Action must be a numpy array or torch tensor"
            assert isinstance(r_current_step_cur.next_observation, (np.ndarray, torch.Tensor)), "Next observation must be a numpy array or torch tensor"

            current_seq_obs.append(transform_any_array_to_torch(r_current_step_cur.observation))
            current_seq_act.append(transform_any_array_to_torch(r_current_step_cur.action))
            current_seq_rew.append(float(r_current_step_cur.reward))
            current_seq_next_obs.append(transform_any_array_to_torch(r_current_step_cur.next_observation))
            current_seq_term.append(r_current_step_cur.terminated)
            current_seq_trunc.append(r_current_step_cur.truncated)

            current_step_cur = current_step_cur.prev
            traced_steps += 1

        current_seq_obs = torch.stack(
            list(reversed(current_seq_obs)),
            dim=0,
        ).to(device=device, dtype=dtype, non_blocking=True)
        current_seq_act = torch.stack(
            list(reversed(current_seq_act)),
            dim=0,
        ).to(device=device, dtype=dtype, non_blocking=True)
        current_seq_rew = torch.tensor(
            list(reversed(current_seq_rew)),
            dtype=dtype,
            device=device,
        )
        current_seq_next_obs = torch.stack(
            list(reversed(current_seq_next_obs)),
            dim=0,
        ).to(device=device, dtype=dtype, non_blocking=True)
        current_seq_term = torch.tensor(
            list(reversed(current_seq_term)),
            dtype=torch.bool,
            device=device,
        )
        current_seq_trunc = torch.tensor(
            list(reversed(current_seq_trunc)),
            dtype=torch.bool,
            device=device,
        )

        cache_map[step.idx] = (
            current_seq_obs,
            current_seq_act,
            current_seq_rew,
            current_seq_next_obs,
            current_seq_term,
            current_seq_trunc,
        )

        sequences_obs.append(current_seq_obs)
        sequences_act.append(current_seq_act)
        sequences_rew.append(current_seq_rew)
        sequences_next_obs.append(current_seq_next_obs)
        sequences_term.append(current_seq_term)
        sequences_trunc.append(current_seq_trunc)
    return NNBatchSeq(
        observations=pack_sequence(sequences_obs, enforce_sorted=False),
        actions=pack_sequence(sequences_act, enforce_sorted=False),
        rewards=pack_sequence(sequences_rew, enforce_sorted=False),
        next_observations=pack_sequence(sequences_next_obs, enforce_sorted=False),
        terminations=pack_sequence(sequences_term, enforce_sorted=False),
        truncations=pack_sequence(sequences_trunc, enforce_sorted=False)
    )
