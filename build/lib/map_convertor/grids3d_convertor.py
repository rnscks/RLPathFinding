import torch
import numpy as np
from typing import Optional
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.TopoDS import TopoDS_Shape


from grids_map.pathfing_grids_map3d import GridsMap3D


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
    
    def convert_to_brep_box(self, start_pnt: gp_Pnt, end_pnt: gp_Pnt) -> list[list[list[TopoDS_Shape]]]:
        max_len = self.grids3d.max_len
        gap: float = (start_pnt.X() - end_pnt.X()) / max_len
        box_shape_list: list[list[list[Optional[TopoDS_Shape]]]] = \
            [[[None for _ in range(max_len)] for _ in range(max_len)] for _ in range(max_len)]
        
        for i in range(max_len):
            for j in range(max_len):
                for k in range(max_len):
                    if (self.grids3d[i, j, k].is_obstacle):
                        minPoint = gp_Pnt((start_pnt.X(
                        ) + i*gap), (start_pnt.Y() + j*gap), (start_pnt.Z() + k*gap))
                        maxPoint = gp_Pnt((start_pnt.X(
                        ) + (i + 1)*gap), (start_pnt.Y() + (j + 1)*gap), (start_pnt.Z() + (k + 1)*gap))
                        box_shape_list[i][j][k] = BRepPrimAPI_MakeBox(minPoint, maxPoint).Shape()   
                        
        return box_shape_list
    
        