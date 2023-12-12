import unittest
from grids_map.pathfinding_grids_map2d import GridsMap2D, PNode

class TestGridsMap2D(unittest.TestCase):
    def setUp(self):
        self.max_len = 5
        self.grid_map = GridsMap2D(self.max_len)

    def test_set_current_node(self):
        new_current_node = PNode(2, 3, 0)
        self.grid_map.set_current_node(new_current_node)
        self.assertTrue(new_current_node.is_current_node)
        self.assertEqual(self.grid_map.current_node, new_current_node)

    def test_get_current_node(self):
        current_node = self.grid_map.get_current_node()
        self.assertEqual(current_node, self.grid_map.start_node)

    def test_get_start_node(self):
        start_node = self.grid_map.get_start_node()
        self.assertEqual(start_node, self.grid_map.start_node)

    def test_get_end_node(self):
        end_node = self.grid_map.get_end_node()
        self.assertEqual(end_node, self.grid_map.end_node)

    def test_set_start_node(self):
        new_start_node = PNode(1, 1, 0)
        self.grid_map.set_start_node(new_start_node)
        self.assertEqual(self.grid_map.start_node, new_start_node)

    def test_iter(self):
        nodes = list(self.grid_map)
        self.assertEqual(len(nodes), self.max_len * self.max_len)
        self.assertEqual(nodes[0], self.grid_map[0, 0])
        self.assertEqual(nodes[-1], self.grid_map[self.max_len - 1, self.max_len - 1])

    def test_getitem(self):
        node = self.grid_map[2, 3]
        self.assertEqual(node, self.grid_map.node_map[2][3])

if __name__ == '__main__':
    unittest.main()