import unittest
from map_generator.grids2d_generator import BoxGeneratorforGrids2D

class TestBoxGeneratorforGrids2D(unittest.TestCase):
    def test_generate_grids(self):
        box_generator = BoxGeneratorforGrids2D(10)
        grids2d = box_generator.generate_grids()

        # Assert that the generated grids2d is of the correct size
        self.assertEqual(len(grids2d), 10)

        # Assert that the generated grids2d contains obstacles
        has_obstacles = False
        for node in grids2d:
            if node.is_obstacle:
                has_obstacles = True

        self.assertTrue(has_obstacles)

if __name__ == "__main__":
    unittest.main()