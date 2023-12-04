from orangerl import TransitionBatch, EnvironmentStep, TransitionReplayBuffer, TransitionTransformation, TransitionSampler, UniformTransitionSampler
from .data import NNBatch, transform_any_array_to_torch, nnbatch_from_transitions
from tensordict import TensorDictBase, TensorDict, is_tensor_collection
from torchrl.data.replay_buffers.storages import Storage, ListStorage, LazyTensorStorage, LazyMemmapStorage
import torch
from typing import Optional, Callable, Any, Tuple, Union, Dict, Sequence, List, TypeVar, Generic, Iterable
import collections
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
from .data import Tensor_Or_Numpy
from asyncio import Future

def pin_memory(output: Union[TensorDictBase, torch.Tensor]) -> NNBatch:
    if output.device == torch.device("cpu"):
        return output.pin_memory()
    else:
        return output

def pin_memory_output(fun) -> Callable:
    """Calls pin_memory on outputs of decorated function if they have such method."""

    def decorated_fun(self, *args, **kwargs):
        output = fun(self, *args, **kwargs)
        if self.pin_output_memory:
            output = pin_memory(output)
        return output

    return decorated_fun

class NNReplayBuffer(TransitionReplayBuffer[torch.Tensor, torch.Tensor]):
    def __init__(
        self,
        *,
        storage: Storage = ListStorage(max_size=1_000),
        sample_batch_size: int = 256,
        sampler : TransitionSampler = UniformTransitionSampler(),
        pin_output_memory: bool = False,
        num_prefetch: Optional[int] = None,
        save_info: bool = False,
        sample_repeat : bool = True,
        sample_transforms: List[TransitionTransformation[torch.Tensor, torch.Tensor]] = [],
    ) -> None:
        storage.attach(self)
        self._storage = storage
        self._sampler = sampler
        self.sample_batch_size = sample_batch_size
        self.pin_output_memory = pin_output_memory
        assert num_prefetch is None or num_prefetch > 0, "prefetch number must be None or > 0"
        self._prefetch_num = num_prefetch
        self._prefetch_queue : collections.deque[Future] = None if num_prefetch is None else collections.deque(maxlen=num_prefetch)

        if self._prefetch_num is not None:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=self._prefetch_num)

        self.save_info = save_info
        self.sample_repeat = sample_repeat
        self.sample_transforms = sample_transforms

        self._idx_episode_ends : List[int] = []

        self._replay_lock = threading.RLock()
        self._futures_lock = threading.RLock()

        self._write_cursor : int = 0

    @property
    def transition_len(self) -> int:
        with self._replay_lock:
            return len(self._storage)
    
    @property
    def capacity(self) -> int:
        return self._storage.max_size

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"storage={self._storage}, "
            f"sampler={self._sampler}"
            ")"
        )

    @pin_memory_output
    def transitions_by_timedindex(self, index: Union[int, slice, Sequence[int]]) -> NNBatch:
        transition_length = self.transition_len
        capacity = self.capacity

        if isinstance(index, int):
            real_idx = np.array([index], dtype=np.int64)
        elif isinstance(index, slice):
            real_idx = np.arange(*index.indices(transition_length), dtype=np.int64)
        elif isinstance(index, Sequence):
            real_idx = np.array(index, dtype=np.int64)

        with self._replay_lock:
            if transition_length == capacity:
                assert real_idx.max() < capacity, "index out of range"
                data=self._storage[(real_idx + self._write_cursor + 1) % capacity]
            else:
                assert real_idx.max() < transition_length, "index out of range"
                data = self._storage[real_idx]
        
        if not isinstance(index, NNBatch):
            data = torch.stack(data, dim=0)
        
        return data

    def idx_episode_begins_and_ends(self) -> Sequence[int]:
        rst = []
        
        transition_length = self.transition_len
        if transition_length == 0:
            return rst
        
        with self._replay_lock:
            if transition_length == self.capacity:
                firstepisode_start = self._write_cursor
                lastepisode_end = self._write_cursor - 1
            else:
                firstepisode_start = 0
                lastepisode_end = transition_length - 1

            if len(self._idx_episode_ends) == 0:
                return [(firstepisode_start, lastepisode_end)]
            
            firstepisode_end_idx = -1
            for i, idx in enumerate(self._idx_episode_ends):
                if idx >= firstepisode_start:
                    firstepisode_end_idx = i
                    break
            if firstepisode_end_idx < 0:
                firstepisode_end_idx = 0
            for i_ep in range(len(self._idx_episode_ends) + 1):
                start_ep = (self._idx_episode_ends[(firstepisode_end_idx + i_ep - 1) % len(self._idx_episode_ends)] + 1) % transition_length if i_ep > 0 else firstepisode_start
                end_ep = self._idx_episode_ends[(firstepisode_end_idx + i_ep) % len(self._idx_episode_ends)] if i_ep < len(self._idx_episode_ends) else lastepisode_end
                if end_ep < start_ep:
                    end_ep += self.capacity
                rst.append((start_ep, end_ep))
        return rst

    @pin_memory_output
    def transitions_at(self, index: Union[int, slice, Sequence[int]]) -> NNBatch:
        with self._replay_lock:
            data = self._storage[index]
        
        if not isinstance(index, NNBatch):
            data = torch.stack(data, dim=0).flatten()
        
        return data

    __getitem__ = transitions_at

    def state_dict(self) -> Dict[str, Any]:
        return {
            "_storage": self._storage.state_dict(),
            "_write_cursor": self._write_cursor,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._storage.load_state_dict(state_dict["_storage"])
        self._write_cursor = state_dict["_write_cursor"]

    def append(self, step : EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy]) -> None:
        step_data = nnbatch_from_transitions([step])[0]
        self._add(step_data)

    def _add(self, data : TensorDictBase) -> None:
        assert data.batch_size == (), "data must be a single transition"
        assert data['masks'] is None or data['masks'].all(), "masks must be None or all True"
        
        to_insert = NNBatch(
            observations = data['observations'],
            actions = data['actions'],
            rewards = data['rewards'],
            next_observations = data['next_observations'],
            terminations = data['terminations'],
            truncations = data['truncations'],
            masks=None,
            infos = data['infos'] if self.save_info and 'infos' in data.keys() else None,
            batch_size=()
        )
        
        with self._replay_lock:
            index = self._write_cursor
            self._storage[index] = to_insert
            self._write_cursor = (self._write_cursor + 1) % self.capacity
            
            idx_in_episode_ends = index in self._idx_episode_ends
            if torch.logical_or(to_insert['terminations'], to_insert['truncations']).any():
                if not idx_in_episode_ends:
                    self._idx_episode_ends.append(index)
                    self._idx_episode_ends.sort()
            else:
                if idx_in_episode_ends:
                    self._idx_episode_ends.remove(index)
        
        return index

    def _extend(self, data: TensorDictBase) -> Sequence[int]:
        assert data.ndim == 1, "data must be a batch of transitions"
        data = data if data["masks"] is None else data[data["masks"]]
        to_insert = NNBatch(
            observations = data['observations'],
            actions = data['actions'],
            rewards = data['rewards'],
            next_observations = data['next_observations'],
            terminations = data['terminations'],
            truncations = data['truncations'],
            masks=None,
            infos = data['infos'] if self.save_info and 'infos' in data.keys() else None,
            batch_size=()
        )

        with self._replay_lock:
            index = np.arange(self._write_cursor, self._write_cursor + to_insert.size(0) + 1) % self.capacity
            self._storage[index[:-1]] = to_insert
            self._write_cursor = index[-1]

            # Update episode ends
            episode_ends = torch.logical_or(to_insert['terminations'], to_insert['truncations']).cpu().numpy()
            idx_in_episode_ends = np.isin(index[:-1], self._idx_episode_ends)
            need_sort = False
            for i, idx in enumerate(index[:-1]):
                if episode_ends[i]:
                    if not idx_in_episode_ends[i]:
                        self._idx_episode_ends.append(idx)
                        need_sort = True
                else:
                    if idx_in_episode_ends[i]:
                        self._idx_episode_ends.remove(idx)
            
            if need_sort:
                self._idx_episode_ends.sort()

        return index
 
    def extend(self, data: Union[Iterable[EnvironmentStep[Tensor_Or_Numpy, Tensor_Or_Numpy]], TransitionBatch[Tensor_Or_Numpy, Tensor_Or_Numpy]]) -> None:
        if is_tensor_collection(data):
            data_dict = data
        else:
            data_dict = nnbatch_from_transitions(data)
        return self._extend(data_dict)

    @pin_memory_output
    def _sample(self, **kwargs) -> NNBatch:
        with self._replay_lock:
            index, mask = self._sampler.sample_idx(self, self.sample_batch_size, self.sample_repeat, **kwargs)
            mask = torch.from_numpy(mask) if mask is not None else None
            index = torch.from_numpy(index)
            data : NNBatch = self._storage[index]
            data.masks = mask.to(device=data.device, dtype=torch.bool) if mask is not None else None
        
        for transform in self.sample_transforms:
            data = transform.transform_batch(data)
        return data

    def clear(self):
        """Empties the replay buffer and reset cursor to 0."""
        with self._replay_lock:
            self._idx_episode_ends = []
            self._write_cursor = 0
            self._storage._empty()
        with self._futures_lock:
            self._prefetch_queue.clear()

    def sample(
        self,
        **kwargs,
    ) -> NNBatch:
        """Samples a batch of data from the replay buffer.

        Uses Sampler to sample indices, and retrieves them from Storage.

        Args:
            batch_size (int, optional): size of data to be collected. If none
                is provided, this method will sample a batch-size as indicated
                by the sampler.
            return_info (bool): whether to return info. If True, the result
                is a tuple (data, info). If False, the result is the data.

        Returns:
            A batch of data selected in the replay buffer.
            A tuple containing this batch and info if return_info flag is set to True.
        """
        if self._prefetch_num is None:
            ret = self._sample(**kwargs)
        else:
            if len(self._prefetch_queue) == 0:
                ret = self._sample(**kwargs)
            else:
                with self._futures_lock:
                    ret = self._prefetch_queue.popleft().result()

            with self._futures_lock:
                while len(self._prefetch_queue) < self._prefetch_num:
                    fut = self._prefetch_executor.submit(self._sample, **kwargs)
                    self._prefetch_queue.append(fut)
        
        return ret

    def mark_update(self, index: Union[int, torch.Tensor]) -> None:
        """
        Used by torchrl Storage to mark that the data at the given index has been updated.
        """

        pass

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        _replay_lock = state.pop("_replay_lock", None)
        _futures_lock = state.pop("_futures_lock", None)
        if _replay_lock is not None:
            state["_replay_lock_placeholder"] = None
        if _futures_lock is not None:
            state["_futures_lock_placeholder"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]):
        if "_replay_lock_placeholder" in state:
            state.pop("_replay_lock_placeholder")
            _replay_lock = threading.RLock()
            state["_replay_lock"] = _replay_lock
        if "_futures_lock_placeholder" in state:
            state.pop("_futures_lock_placeholder")
            _futures_lock = threading.RLock()
            state["_futures_lock"] = _futures_lock
        self.__dict__.update(state)


