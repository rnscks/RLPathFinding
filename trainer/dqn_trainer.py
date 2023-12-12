import torch

from trainer.grids_environment import DQNGrids  
from trainer.dqn_optimizer import DQNModelOptimizer
from trainer.dqn_agent import DQNAgent  
from trainer.dqn_replay_memory import DQNReplayMemory   

class DQNGrids2D:   
    def __init__(self, epochs: int, batch_size: int) -> None:
        self.episodes = epochs
        self.batch_size = batch_size
        self.agent = DQNAgent()
        self.replay_memory = DQNReplayMemory(capacity=1000)
        self.modelOptimizer = DQNModelOptimizer(self.agent, self.replay_memory)  
        self.grids2d_env = DQNGrids()
        return


    def train(self):
        train_loss_average_list = []
        for episode in range(self.episodes):
            current_state = self.grids2d_env.reset()    
            network_input = current_state
            time_steps = 36
            for time_step in range(time_steps):
                current_action = self.agent.get_training_action(network_input, time_steps + 1)
                
                observation, reward = self.grids2d_env.step(current_action.item()) 
                reward = torch.tensor([reward]).float()
                
                if (time_step == time_steps - 1):
                    next_state = None
                else:   
                    next_state = observation
                    
                self.replay_memory.Push(network_input, current_action, next_state, reward)
                network_input = next_state

                train_loss_averages = self.modelOptimizer.train_each_batch()
                target_net_statedict = self.agent.dqn_target_network.state_dict()
                policy_net_statedict = self.agent.dqn_policy_network.state_dict()

                tau = self.agent.dqn_hyperparameter.netWork_updating_rate
                
                for key in policy_net_statedict:
                    target_net_statedict[key] = policy_net_statedict[key]* tau+ target_net_statedict[key]*(1-tau)

                self.agent.dqn_policy_network.load_state_dict(target_net_statedict)
                
                if (train_loss_averages is None):
                    continue
                train_loss_average_list.append(train_loss_averages)
            
        return train_loss_average_list
