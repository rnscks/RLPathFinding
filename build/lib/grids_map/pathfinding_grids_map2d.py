from .grids_map import GridsMap  
from node.pathfinding_node import PNode

class GridsMap2D(GridsMap):
    def __init__(self, max_len: int) -> None:
        super().__init__()
        self.max_len: int = max_len 
        
        self.node_map: list[list[PNode]] = \
        [[PNode(i, j, 0) for i in range(max_len)] for j in range(max_len)]
        
        self.start_node: PNode = self.node_map[0][0]
        self.start_node.is_start_node = True    
        self.end_node: PNode = self.node_map[max_len - 1][max_len - 1]
        self.end_node.is_end_node = True
        self.current_node: PNode = self.start_node  
        self.current_node.is_current_node = True    
        
        
    def set_current_node(self, new_current_node: PNode) -> None:
        self.current_node.is_current_node = False
        new_current_node.is_current_node = True
        self.current_node = new_current_node 
        return  
        
    def get_current_node(self) -> PNode:
        return self.current_node
        
    def get_start_node(self) -> PNode:
        return self.start_node
    
    def get_end_node(self) -> PNode: 
        return self.end_node
    
    def set_start_node(self, start_node: PNode) -> None:
        self.start_node = start_node
        return
    
    def __iter__(self) -> PNode:
        for i in range(self.max_len):
            for j in range(self.max_len):
                yield self.node_map[i][j]
                    
    def __getitem__(self, index: tuple[int]) -> PNode:
        if (index[0] < 0 or index[0] >= self.max_len or index[1] < 0 or index[1] >= self.max_len):
            return self.current_node
        return self.node_map[index[0]][index[1]]
    
    def __len__(self) -> PNode:
        return self.max_len    
    

    