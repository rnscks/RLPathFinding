from typing import Optional

from .node import Node  


class PNode(Node):
    def __init__(self, i: int, j: int, k: int) -> None:
        super().__init__()   
        self.i: int = i
        self.j: int = j
        self.k: int = k
        
        self.f: float = 0
        self.g: float = 0   
        self.h: float = 0
        
        self.is_obstacle: bool = False
        self.parent: Optional["PNode"] = None
        
        self.is_start_node: bool = False    
        self.is_end_node: bool = False
        
        
    def __str__(self) -> str:
        return super().__str__(f"({self.i}, {self.j}, {self.k})")
    
    def __eq__(self, other: 'PNode') -> bool:
        bool_type: bool = self.i == other.i and self.j == other.j and self.k == other.k
        return super().__eq__(bool_type)
        
    
if __name__ == "__main__":
    pnode = PNode(1,2,3)    
    
    print(pnode)
    pass    