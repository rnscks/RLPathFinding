from torch import FloatTensor
from collections import  deque
import random

from trainer.dqn_memory import DQNMemory

class DQNReplayMemory:
    def __init__(self, capacity = 10000):
        self.memory = deque([], maxlen = capacity)
        return

    def push(self, state: FloatTensor, action: FloatTensor, next_state: FloatTensor, reward: FloatTensor):
        self.memory.append(DQNMemory(state.unsqueeze(0), action, next_state, reward))
        return

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
