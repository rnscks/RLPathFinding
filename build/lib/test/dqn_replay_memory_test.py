import unittest
from torch import FloatTensor

from trainer.dqn_replay_memory import DQNReplayMemory   

class DQNReplayMemoryTest(unittest.TestCase):
    def setUp(self):
        self.capacity = 10000
        self.memory = DQNReplayMemory(self.capacity)


    def test_push(self):
        state = FloatTensor([[1, 2, 3]] * 12)
        action = FloatTensor([0] * 8)
        next_state = state
        reward = FloatTensor([1])


        self.memory.push(state, action, next_state, reward)

        self.assertEqual(len(self.memory), 1)

    def test_sample(self):
        batch_size = 2

        # Add some sample memories to the replay memory
        state1 = FloatTensor([[1, 2, 3]] * 12)
        action1 = FloatTensor([0] * 8)
        next_state1 = state1
        reward1 = FloatTensor([0])
        self.memory.push(state1, action1, next_state1, reward1)

        state2 = FloatTensor([[7, 8, 9]] * 12)
        action2 = FloatTensor([1] * 8)
        next_state2 = state2
        reward2 = FloatTensor([1])
        self.memory.push(state2, action2, next_state2, reward2)

        # Sample memories from the replay memory
        sampled_memories = self.memory.sample(batch_size)

        self.assertEqual(len(sampled_memories), batch_size)

    def test_len(self):
        self.assertEqual(len(self.memory), 0)

        state = FloatTensor([[1, 2, 3]] * 12)
        action = FloatTensor([0] * 8)
        next_state = state
        reward = FloatTensor([0])
        self.memory.push(state, action, next_state, reward)

        self.assertEqual(len(self.memory), 1)
        

if __name__ == '__main__':
    unittest.main()
