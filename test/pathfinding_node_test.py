import unittest
from node.pathfinding_node import PNode

class TestPNode(unittest.TestCase):
    def test_get_moving_index(self) -> tuple[int]:
        pnode = PNode(0, 0, 0)
        pnode.get_moving_index((0, 2, 1))
        self.assertEqual((0, 2, 1), (0, 2, 1))
    
    def test_get_index(self):
        pnode = PNode(1, 2, 3)
        index = pnode.get_index()
        self.assertEqual(index, (1, 2, 3))

    def test_str(self):
        pnode = PNode(1, 2, 3)
        string = str(pnode)
        self.assertEqual(string, "(1, 2, 3)")

    def test_eq(self):
        pnode1 = PNode(1, 2, 3)
        pnode2 = PNode(1, 2, 3)
        self.assertEqual(pnode1, pnode2)

if __name__ == "__main__":
    unittest.main()