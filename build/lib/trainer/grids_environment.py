from torch import FloatTensor

from map_generator.grids2d_generator import BoxGeneratorforGrids2D
from map_convertor.grids2d_convertor import MapConvertorGrids2D

class DQNGrids:
    def __init__(self) -> None:
        self.grids = BoxGeneratorforGrids2D(12).generate_grids()
        self.action_table = [(-1, 0, 0), (0, 1, 0), (1, 0, 0), (0, -1, 0), (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0)]
    
    def __react(self, action: FloatTensor) -> bool:
        step = self.action_table[int(action.item())]
        currnet_node = self.grids.get_current_node()
        next_step = currnet_node.get_moving_index(step)
        if (next_step is self.grids.end_node):
            return True
        self.grids.set_current_node(self.grids[next_step])
        
        return False
    
    def get_reset(self):
        convertor = MapConvertorGrids2D(self.grids)
        return convertor.convert_to_tensor()
    
    def step(self, action: FloatTensor):
        convertor = MapConvertorGrids2D(self.grids)
        if (self.__react(action)):
            return convertor.convert_to_tensor(), 100
        
        return convertor.convert_to_tensor(), 0
