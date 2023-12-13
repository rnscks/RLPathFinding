from torch import FloatTensor
import numpy as np  

from map_generator.grids2d_generator import BoxGeneratorforGrids2D
from map_convertor.grids2d_convertor import MapConvertorGrids2D

class DQNGrids:
    def __init__(self) -> None:
        self.grids = BoxGeneratorforGrids2D(12).generate_grids()
        self.action_table = [(-1, 0, 0), (0, 1, 0), (1, 0, 0), (0, -1, 0), (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0)]
        self.closeed_set = []
    
    def __react(self, action: FloatTensor) -> bool:
        step = self.action_table[int(action.item())]
        currnet_node = self.grids.get_current_node()
        next_step = currnet_node.get_moving_index(step)
        next_node = self.grids[next_step]
        
        if (next_node.is_obstacle):
            return False
        
        # if (next_node in self.closeed_set):
        #     return False
        
        self.closeed_set.append(next_node) 
        if (next_node is self.grids.end_node):
            self.grids.set_current_node(self.grids[next_step])
            print("reach the end")
            return True
        self.grids.set_current_node(self.grids[next_step])
        return False
    
    def get_reset(self):
        self.grids.set_current_node(self.grids.start_node)
        #self.closeed_set.clear()
        #self.closeed_set.append(self.grids.start_node)
        convertor = MapConvertorGrids2D(self.grids)
        return convertor.convert_to_tensor()
    
    def step(self, action: FloatTensor):
        convertor = MapConvertorGrids2D(self.grids)
        end_node = self.grids.get_end_node()
        current_node = self.grids.get_current_node()
        x_dist = np.abs(current_node.get_index()[0] - end_node.get_index()[0])
        y_dist = np.abs(current_node.get_index()[1] - end_node.get_index()[1])
        if (self.__react(action)):
            return convertor.convert_to_tensor(), 1200, True
        
        return convertor.convert_to_tensor(), -(x_dist * y_dist), False