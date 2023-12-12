import torch
import numpy as np
from typing import Optional


from grids_map.pathfinding_grids_map3d import GridsMap3D


class MapConvertorGrids3D:
    def __init__(self, grids3d: GridsMap3D) -> None:
        self.grids3d: GridsMap3D = grids3d  
    
    
    def convert_to_numpy(self) -> np.ndarray:
        max_size = self.grids3d.max_len
        
        grids3d_np = np.zeros((max_size, max_size, max_size), dtype=np.int8)
        for node in self.grids3d:
            if (node.is_obstacle):
                grids3d_np[node.i, node.j, node.k] = 1  
                
            if (node.is_start_node):
                grids3d_np[node.i, node.j, node.k] = 2  
                
            if (node.is_end_node):
                grids3d_np[node.i, node.j, node.k] = 3
                
        return grids3d_np
    
    def convert_to_tensor(self) -> torch.Tensor:
        return torch.FloatTensor(self.convert_to_numpy())   
    
    
        