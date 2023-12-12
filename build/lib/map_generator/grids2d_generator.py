from random import randint

from map_generator.grids_generator import MapGenerator
from grids_map.pathfinding_grids_map2d import GridsMap2D  

class BoxGeneratorforGrids2D(MapGenerator):
    def __init__(self, map_size: int) -> None:   
        super().__init__()
        self.map_size: int = map_size

    
    def generate_grids(self) -> GridsMap2D:
        grids2d = GridsMap2D(self.map_size)
        start_index = self.map_size
        end_index = 0
        
        box_index_list = []
        for _ in range(self.map_size * 2):
            i: int = randint(end_index + 1, start_index - 2)
            j: int = randint(end_index + 1, start_index - 2)
            box_index_list.append((i, j, 0))
        box_index_list.sort()
        
        for i in range(end_index, len(box_index_list) - 1, 2):
            s1, t1 = min(box_index_list[i][0], box_index_list[i + 1][0]), max(
                box_index_list[i][0], box_index_list[i + 1][0]) + 1
            for x in range(s1, t1):
                s2, t2 = min(box_index_list[i][1], box_index_list[i + 1][1]), max(
                    box_index_list[i][1], box_index_list[i + 1][1]) + 1
                for y in range(s2, t2):
                    grids2d[x, y].is_obstacle = True

        return grids2d


if __name__ == "__main__":
    box_generator = BoxGeneratorforGrids2D(10)
    grids3d = box_generator.generate_grids()
    for node in grids3d:
        if (node.is_obstacle):
            print(node)
    pass