from abc import ABC, abstractmethod

class MapGenerator(ABC):
    
    
    @abstractmethod
    def generate_grids(self):
        pass
