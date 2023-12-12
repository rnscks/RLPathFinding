import unittest
from torch import FloatTensor
from trainer.grids_environment import DQNGrids

class TestDQNGrids(unittest.TestCase):
    def setUp(self):
        self.dqn_grids = DQNGrids()
    
    def test_react(self):
        action = FloatTensor([0])
        result = self.dqn_grids._DQNGrids__react(action)
        self.assertEqual(result, False)
    
    def test_get_reset(self):
        result = self.dqn_grids.get_reset()
        self.assertIsNotNone(result)
    
    def test_step(self):
        action = FloatTensor([1])
        next_state, reward = self.dqn_grids.step(action)
        self.assertIsNotNone(next_state)
        self.assertEqual(reward, 0)
    
if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()