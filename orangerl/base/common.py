from abc import ABC, abstractmethod
from typing import Union, TypeVar, Generic
from os import PathLike

_SerialT = TypeVar("_SerialT")
class Serializable(Generic[_SerialT],ABC):
    @abstractmethod
    def serialize(self) -> _SerialT:
        pass

    @abstractmethod
    def deserialize(self, serial : _SerialT) -> None:
        pass

class Savable(ABC):
    @abstractmethod
    def save(self, path : Union[str, PathLike]) -> None:
        pass

    @abstractmethod
    def load(self, path : Union[str, PathLike]) -> None:
        pass