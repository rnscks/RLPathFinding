from torch import FloatTensor

class DQNMemory:
    def __init__(self, state: FloatTensor, action: FloatTensor, next_state: FloatTensor, reward: FloatTensor) -> None:
        self.state = state  
        self.action = action
        self.next_state = next_state
        self.reward = reward
        pass
    
