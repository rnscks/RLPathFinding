import torch

from trainer.grids_environment import DQNGrids  
from trainer.dqn_optimizer import DQNModelOptimizer
from trainer.dqn_agent import DQNAgent  
from trainer.dqn_replay_memory import DQNReplayMemory   

class DQNTrainer:   
    def __init__(self, epochs: int, batch_size: int) -> None:
        self.episodes = epochs
        self.batch_size = batch_size
        self.agent = DQNAgent()
        self.agent.dqn_hyperparameter.batch_size = batch_size
        self.replay_memory = DQNReplayMemory(capacity=15000)
        self.model_optimizer = DQNModelOptimizer(self.agent, self.replay_memory)  
        self.grids2d_env = DQNGrids()
        return


    def train(self):
        train_loss_average_list = []
        reward_average_list = []
        epsilon_average_list = []   
        for episode in range(self.episodes):
            current_state = self.grids2d_env.get_reset()    
            network_input = current_state
            time_steps = 600
            sumofreward = 0
            for time_step in range(time_steps):
                current_action = self.agent.get_training_action(network_input, 3 *episode + 1 + 100)
                epsilon = self.agent.get_eplisontread(3 *episode + 1 + 100)
                epsilon_average_list.append(epsilon)    
                observation, reward, reach_the_end = self.grids2d_env.step(current_action) 
                reward = torch.tensor([reward]).float()
                sumofreward += reward.item()
                
                if (time_step == time_steps - 1):
                    next_state = torch.zeros((12, 12), dtype=torch.float)
                else:   
                    next_state = observation
                    
                self.replay_memory.push(network_input, current_action, next_state, reward)
                network_input = next_state

                train_loss_averages = self.model_optimizer.train_each_batch()
                target_net_statedict = self.agent.dqn_target_network.state_dict()
                policy_net_statedict = self.agent.dqn_policy_network.state_dict()

                tau = self.agent.dqn_hyperparameter.netWork_updating_rate
                
                for key in policy_net_statedict:
                    target_net_statedict[key] = policy_net_statedict[key]* tau+ target_net_statedict[key]*(1-tau)

                self.agent.dqn_policy_network.load_state_dict(target_net_statedict)
                
                if (train_loss_averages is None):
                    continue
                if (reach_the_end is True):
                    break
                train_loss_average_list.append(train_loss_averages)
            sumofreward /= time_step  
            reward_average_list.append(sumofreward) 
            print("current node: ", self.grids2d_env.grids.get_current_node())
            print("exit time: ", time_step)
            print("epsilon: ",epsilon)
            print("episode: ",episode)
            
        return train_loss_average_list, reward_average_list, epsilon_average_list


