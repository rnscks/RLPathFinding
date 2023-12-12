import unittest
from trainer.dqn_memory_batch import DQNMemoryBatch
from trainer.dqn_memory import DQNMemory
import torch

class TestDQNMemoryBatch(unittest.TestCase):
    def setUp(self):
        # Create some sample data for testing
        self.states = [torch.tensor([[1.0] * 12] * 12), torch.tensor([[4.0] * 12] * 12)]
        self.actions = [torch.tensor([0.0] * 8), torch.tensor([2.0] * 8)]
        self.next_states = [torch.tensor([[7.0] * 12] * 12), torch.tensor([[10.0] * 12] * 12)]
        self.rewards = [torch.tensor([0]), torch.tensor([1])]
        self.batch_size = 2


    def test_reconstruct_memories(self):
        # Test the ReconstructMemories static method
        memory_list = [
            DQNMemory(torch.tensor([[1.0] * 12] * 12), torch.tensor([0.0] * 8), torch.tensor([[4.0] * 12] * 12), torch.tensor([0])),
            DQNMemory(torch.tensor([[7.0] * 12] * 12), torch.tensor([1.0] * 8), torch.tensor([[10.0] * 12] * 12), torch.tensor([1]))
        ]
        memory_batch = DQNMemoryBatch.reconstruct_memories(memory_list,  self.batch_size)

        self.assertEqual(memory_batch.sate_batch.shape, (2, 1, 12, 12))
        self.assertEqual(memory_batch.action_batch.shape, (2, 8))
        self.assertEqual(memory_batch.next_state_batch.shape, (2, 1, 12, 12))
        self.assertEqual(memory_batch.reward_batch.shape, (2, 1))
        self.assertEqual(memory_batch.non_final_mask.shape, (2,))
        self.assertEqual(memory_batch.non_final_next_state_batch.shape, (2 , 1, 12, 12))

if __name__ == '__main__':
    unittest.main()