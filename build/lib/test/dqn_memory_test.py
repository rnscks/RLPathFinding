import unittest
from torch import FloatTensor

from trainer.dqn_memory import DQNMemory

class DQNMemoryTest(unittest.TestCase):
    def test_dqn_memory(self):
        state = FloatTensor([[1, 2, 3] * 4] * 12)
        action = FloatTensor([0, 1, 2, 3, 4, 5, 6, 7])
        next_state = state
        reward = FloatTensor([0.5])

        memory = DQNMemory(state, action, next_state, reward)

        self.assertEqual(memory.state.tolist(), [[1, 2, 3] * 4] * 12)
        self.assertEqual(memory.action.tolist(), [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(memory.next_state.tolist(), [[1, 2, 3] * 4] * 12)
        self.assertEqual(memory.reward.tolist(), [0.5])

if __name__ == '__main__':
    unittest.main()