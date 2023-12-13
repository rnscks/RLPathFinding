import torch
from torch import FloatTensor
import numpy as np
import random

from trainer.dqn import DQN
from trainer.dqn_hyperparameter import DQNHyperparameters   

##


class DQNAgent:
    def __init__(self):        
        self.dqn_policy_network = DQN()
        self.dqn_target_network = DQN()

        self.dqn_hyperparameter = DQNHyperparameters()
        return

    def get_training_action(self, network_input: FloatTensor, time_step: int):
        network_input = network_input.unsqueeze(0).unsqueeze(0) 
        sample_number = random.random()
        eps_threshold =  self.get_eplisontread(time_step)
        if (sample_number > eps_threshold):
            with torch.no_grad():
                optimal_action = torch.argmax(self.dqn_policy_network(network_input.view(-1, 1, 12, 12)))
                return torch.tensor([optimal_action], dtype=torch.float32)

        return torch.tensor([random.randrange(8)], dtype=torch.float32)
    

    def get_optimal_action(self, network_input):
        with torch.no_grad():
            optimal_action = torch.argmax(self.dqn_target_network(network_input.view(-1, 1, 12, 12)))
            return torch.tensor([optimal_action], dtype=torch.float32)

    def get_eplisontread(self, time_step: int):
        eplison_end = self.dqn_hyperparameter.epsilon_end
        eplison_start = self.dqn_hyperparameter.epsilon_start
        eplison_decay = self.dqn_hyperparameter.eplison_decay
        epsthreshold =  eplison_end + (eplison_start - eplison_end) * np.exp(-1. * time_step / eplison_decay)

        return epsthreshold
