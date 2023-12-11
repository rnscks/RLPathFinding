import util
from abc import ABC, abstractmethod

class Node(ABC):
    def __init__(self) -> None:
        super().__init__()  


    @abstractmethod
    def __str__(self, string: str) -> str:
        return string
    
    @abstractmethod
    def __eq__(self, bool_type: bool) -> bool:
        return bool_type

    def __hash__(self) -> int:
        return super().__hash__()


