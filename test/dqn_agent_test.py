import unittest
import torch
from torch import FloatTensor
from trainer.dqn_agent import DQNAgent

class TestDQNAgent(unittest.TestCase):
    def setUp(self):
        self.agent = DQNAgent()

    def test_get_training_action(self):
        network_input = FloatTensor(1, 1, 12, 12).fill_(1)
        time_step = 10
        action = self.agent.get_training_action(network_input, time_step)
        self.assertIsInstance(action, torch.Tensor)
        self.assertEqual(action.shape, torch.Size([1]))
        self.assertTrue(action.item() >= 0 and action.item() < 8)

    def test_get_optimalAction(self):
        network_input = FloatTensor(3, 1, 12, 12).fill_(1)
        action = self.agent.get_optimalAction(network_input)
        self.assertIsInstance(action, torch.Tensor)
        self.assertEqual(action.shape, torch.Size([1]))
        self.assertTrue(action.item() >= 0 and action.item() < 8)

    def test_get_eplisontread(self):
        time_step = 20
        epsilon = self.agent.get_eplisontread(time_step)
        self.assertIsInstance(epsilon, float)
        self.assertTrue(epsilon >= 0 and epsilon <= 1)

if __name__ == '__main__':
    unittest.main()
