import torch
import torch.nn as nn
import torch.optim as optim


from trainer.dqn_agent import DQNAgent  
from trainer.dqn_memory_batch import DQNMemoryBatch 
from trainer.dqn_replay_memory import DQNReplayMemory   



class DQNModelOptimizer:
    def __init__(self, dqn_agent: DQNAgent, replay_memory: DQNReplayMemory):
        self.agent = dqn_agent
        self.replay_memory = replay_memory
        self.hyperparameter = dqn_agent.dqn_hyperparameter
        self.optimizer = optim.AdamW(dqn_agent.dqn_policy_network.parameters(), lr=self.hyperparameter.learning_rate, amsgrad=True)
        pass
    
    
    def train_each_batch(self) -> None:
        batch_size = self.hyperparameter.batch_size
        if (batch_size > len(self.replay_memory)):
            return

        sampling_memories = self.replay_memory.sample(batch_size)
        memories_batch: DQNMemoryBatch = DQNMemoryBatch.reconstruct_memories(sampling_memories,  batch_size)
        state_action_qvalue = self.agent.dqn_policy_network(memories_batch.sate_batch.view(-1, 1, 12, 12)).gather(1, memories_batch.action_batch.long())
        expected_qvalue = self.__predict_qvalue(batch_size, memories_batch)    
        return self.__optimize_model(state_action_qvalue, expected_qvalue)
        
    def __predict_qvalue(self, batch_size: int, memories_batch: DQNMemoryBatch) -> torch.Tensor:
        predicted_qvalue = torch.zeros(batch_size).float()
        with torch.no_grad():
            all_prediction_value= self.agent.dqn_target_network(memories_batch.non_final_next_state_batch.view(-1, 1, 12, 12)).max(1)[0]
            predicted_qvalue[memories_batch.non_final_mask] = all_prediction_value[memories_batch.non_final_mask]
            predicted_qvalue= predicted_qvalue.unsqueeze(1)  
        return predicted_qvalue * self.agent.dqn_hyperparameter.discount_factor + memories_batch.reward_batch
        
    def __optimize_model(self, state_action_qvalue, expected_qvalue):   
        lossFunction = nn.SmoothL1Loss()
        loss = lossFunction(state_action_qvalue, expected_qvalue)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.dqn_policy_network.parameters(), 100)
        self.optimizer.step()
        return loss.item() / self.hyperparameter.batch_size