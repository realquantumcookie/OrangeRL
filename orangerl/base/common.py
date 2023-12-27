from abc import ABC, abstractmethod, abstractproperty
from typing import Union
from os import PathLike


class Savable(ABC):
    @abstractmethod
    def save(self, path : Union[str, PathLike]) -> None:
        pass

    @abstractmethod
    def load(self, path : Union[str, PathLike]) -> None:
        pass