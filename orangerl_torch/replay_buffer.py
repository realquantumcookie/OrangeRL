from orangerl.base.data import TransitionBatch, EnvironmentStep, TransitionReplayBuffer, TransitionTransformation, TransitionSampler
from orangerl.data.samplers import UniformTransitionSampler
from .data import NNBatch, transform_any_array_to_torch
from tensordict import TensorDictBase
from torchrl.data.replay_buffers.storages import Storage, ListStorage, TensorStorage, LazyTensorStorage, LazyMemmapStorage
import torch
from typing import Optional, Callable, Any, Tuple, Union, Dict, Sequence, List, TypeVar, Generic
import collections
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np

def pin_memory(output: TensorDictBase) -> NNBatch:
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
    is_transition_single_episode : bool = False
    is_transition_time_sorted : bool = True

    def __init__(
        self,
        *,
        storage: Storage = ListStorage(max_size=1_000),
        sample_batch_size: int = 256,
        sampler : TransitionSampler = UniformTransitionSampler(),
        pin_output_memory: bool = False,
        num_prefetch: Optional[int] = None,
        sample_repeat : bool = True,
        sample_transforms: List[TransitionTransformation] = [],
    ) -> None:
        storage.attach(self)
        self._storage = storage
        self._sampler = sampler
        self.sample_batch_size = sample_batch_size
        self.pin_output_memory = pin_output_memory
        assert num_prefetch is None or num_prefetch > 0, "prefetch number must be None or > 0"
        self._prefetch_num = num_prefetch
        self.prefetch_queue = None if num_prefetch is None else collections.deque(maxlen=num_prefetch)

        if self._prefetch_num is not None:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=self._prefetch_num)

        self.sample_repeat = sample_repeat
        self.sample_transforms = sample_transforms

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

    @pin_memory_output
    def transitions_at(self, index: Union[int, slice, Sequence[int]]) -> NNBatch:
        with self._replay_lock:
            data = self._storage[index]
        
        if not isinstance(index, NNBatch):
            data = torch.stack(data, dim=0)
        
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

    def append(self, step : EnvironmentStep[torch.Tensor, torch.Tensor]) -> int:
        """Add a single element to the replay buffer.

        Args:
            data (Any): data to be added to the replay buffer

        Returns:
            index where the data lives in the replay buffer.
        """
        if self._transform is not None and (
            is_tensor_collection(data) or len(self._transform)
        ):
            data = self._transform.inv(data)
        return self._add(data)

    def _add(self, data):
        with self._replay_lock:
            index = self._writer.add(data)
            self._sampler.add(index)
        return index

    def _extend(self, data: Sequence) -> torch.Tensor:
        with self._replay_lock:
            index = self._writer.extend(data)
            self._sampler.extend(index)
        return index

    def extend(self, data: Sequence) -> torch.Tensor:
        """Extends the replay buffer with one or more elements contained in an iterable.

        If present, the inverse transforms will be called.`

        Args:
            data (iterable): collection of data to be added to the replay
                buffer.

        Returns:
            Indices of the data added to the replay buffer.
        """
        if self._transform is not None and (
            is_tensor_collection(data) or len(self._transform)
        ):
            data = self._transform.inv(data)
        return self._extend(data)

    def update_priority(
        self,
        index: Union[int, torch.Tensor],
        priority: Union[int, torch.Tensor],
    ) -> None:
        with self._replay_lock:
            self._sampler.update_priority(index, priority)

    @pin_memory_output
    def _sample(self, batch_size: int) -> Tuple[Any, dict]:
        with self._replay_lock:
            index, info = self._sampler.sample(self._storage, batch_size)
            info["index"] = index
            data = self._storage[index]
        if not isinstance(index, INT_CLASSES):
            data = self._collate_fn(data)
        if self._transform is not None and len(self._transform):
            is_td = True
            if not is_tensor_collection(data):
                data = TensorDict({"data": data}, [])
                is_td = False
            is_locked = data.is_locked
            if is_locked:
                data.unlock_()
            data = self._transform(data)
            if is_locked:
                data.lock_()
            if not is_td:
                data = data["data"]

        return data, info

    def empty(self):
        """Empties the replay buffer and reset cursor to 0."""
        self._writer._empty()
        self._sampler._empty()
        self._storage._empty()

    def sample(
        self, batch_size: Optional[int] = None, return_info: bool = False
    ) -> Any:
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
        if (
            batch_size is not None
            and self._batch_size is not None
            and batch_size != self._batch_size
        ):
            warnings.warn(
                f"Got conflicting batch_sizes in constructor ({self._batch_size}) "
                f"and `sample` ({batch_size}). Refer to the ReplayBuffer documentation "
                "for a proper usage of the batch-size arguments. "
                "The batch-size provided to the sample method "
                "will prevail."
            )
        elif batch_size is None and self._batch_size is not None:
            batch_size = self._batch_size
        elif batch_size is None:
            raise RuntimeError(
                "batch_size not specified. You can specify the batch_size when "
                "constructing the replay buffer, or pass it to the sample method. "
                "Refer to the ReplayBuffer documentation "
                "for a proper usage of the batch-size arguments."
            )
        if not self._prefetch:
            ret = self._sample(batch_size)
        else:
            if len(self._prefetch_queue) == 0:
                ret = self._sample(batch_size)
            else:
                with self._futures_lock:
                    ret = self._prefetch_queue.popleft().result()

            with self._futures_lock:
                while len(self._prefetch_queue) < self._prefetch_cap:
                    fut = self._prefetch_executor.submit(self._sample, batch_size)
                    self._prefetch_queue.append(fut)

        if return_info:
            return ret
        return ret[0]

    def mark_update(self, index: Union[int, torch.Tensor]) -> None:
        pass

    def __iter__(self):
        if self._sampler.ran_out:
            self._sampler.ran_out = False
        if self._batch_size is None:
            raise RuntimeError(
                "Cannot iterate over the replay buffer. "
                "Batch_size was not specified during construction of the replay buffer."
            )
        while not self._sampler.ran_out:
            data = self.sample()
            yield data

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


