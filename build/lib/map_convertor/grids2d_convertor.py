import torch
import numpy as np


from grids_map.pathfinding_grids_map2d import GridsMap2D


class MapConvertorGrids2D:
    def __init__(self, grids2d: GridsMap2D) -> None:
        self.grids2d: GridsMap2D = grids2d  
    
    
    def convert_to_numpy(self) -> np.ndarray:
        max_size = self.grids2d.max_len
        
        grids2d_np = np.zeros((max_size, max_size), dtype=np.int8)
        for node in self.grids2d:
            if (node.is_obstacle):
                grids2d_np[node.i, node.j] = 1  
                
            if (node.is_current_node):
                grids2d_np[node.i, node.j] = 2  
                
            if (node.is_end_node):
                grids2d_np[node.i, node.j] = 3
                
        return grids2d_np
    
    def convert_to_tensor(self) -> torch.Tensor:
        return torch.FloatTensor(self.convert_to_numpy())   
    
    
        