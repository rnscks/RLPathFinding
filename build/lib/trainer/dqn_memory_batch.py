from trainer.dqn_memory import DQNMemory
import torch
from torch import FloatTensor, BoolTensor   

class DQNMemoryBatch:
    def __init__(self, states: list[FloatTensor], actions: list[FloatTensor], next_state_batch: list[FloatTensor], rewards: list[FloatTensor], batch_size: int) -> None:
        self.sate_batch: FloatTensor = torch.stack(states, dim=0)
        self.action_batch: FloatTensor = torch.cat(actions).view(-1, 8)
        self.next_state_batch: FloatTensor = torch.stack(next_state_batch, dim=0)
        self.reward_batch: FloatTensor = torch.cat(rewards).view(-1, 1).float()   
        self.non_final_mask: BoolTensor = torch.tensor(tuple(map(lambda s: s is not None, self.next_state_batch)), dtype = torch.bool)
        self.non_final_next_state_batch: FloatTensor = torch.stack([next_state for next_state in self.next_state_batch if next_state is not None], dim = 0).float()
        return

    
    @staticmethod
    def reconstruct_memories(memoryList: list["DQNMemory"], batch_size: int) -> "DQNMemoryBatch":
        states = []
        actions = []
        next_states = []
        rewards = []
        for memory in memoryList:
            states.append(memory.state.unsqueeze(0))
            actions.append(memory.action)
            next_states.append(memory.next_state.unsqueeze(0))
            rewards.append(memory.reward)

        return DQNMemoryBatch(states, actions, next_states, rewards,  batch_size)