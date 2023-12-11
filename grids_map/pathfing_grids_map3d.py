from .grids_map import GridsMap  
from node.pathfinding_node import PNode

class GridsMap3D(GridsMap):
    def __init__(self, max_len: int) -> None:
        super().__init__()
        self.max_len: int = max_len 
        
        self.node_map: list[list[list[PNode]]] = \
        [[[PNode(i, j, k) for i in range(max_len)] for j in range(max_len)] for k in range(max_len)]
        self.start_node: PNode = self.node_map[0][0][0]   
        self.start_node.is_start_node = True    
        self.end_node: PNode = self.node_map[max_len - 1][max_len - 1][max_len - 1] 
        self.end_node.is_end_node = True
        
    
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
                for k in range(self.max_len):
                    yield self.node_map[i][j][k]
                    
    def __getitem__(self, index: tuple[int]) -> PNode:
        return self.node_map[index[0]][index[1]][index[2]] 
    


if __name__ == "__main__":
    grids_map = GridsMap3D(10)    
    for node in grids_map:
        print(node) 
        
    print(grids_map[0, 0, 0])
    pass    