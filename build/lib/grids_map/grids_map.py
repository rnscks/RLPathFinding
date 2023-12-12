from node.pathfinding_node import PNode
from abc import ABC, abstractmethod

class GridsMap(ABC):
    

    @abstractmethod
    def set_start_node(self, node: PNode):
        pass
    
    @abstractmethod
    def set_start_node(self):
        pass
    
    @abstractmethod
    def get_start_node(self, start_node: PNode) -> PNode:
        pass
    
    @abstractmethod
    def get_end_node(self, end_node: PNode) -> PNode:
        pass
    
    @abstractmethod
    def __iter__(self):
        pass    
    
    @abstractmethod
    def __getitem__(self, *args,  **kwargs) -> PNode:
        pass
    
    @abstractmethod
    def __len__(self) -> PNode:
        pass
    

