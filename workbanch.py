import torch
import numpy as np

from map_convertor.grids2d_convertor import MapConvertorGrids2D
from trainer.dqn_trainer import DQNTrainer  
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    epochs = 600
    batch_size = 128
    trainer = DQNTrainer(epochs, batch_size)
    
    train_loss_average_list, rewards_average_list, epsilone_average_list = trainer.train()
    
    numpy_array = MapConvertorGrids2D(trainer.grids2d_env.grids).convert_to_numpy()
    np.save('tensor.npy', numpy_array)

        
    model = trainer.agent.dqn_target_network.state_dict()
    torch.save(model, 'model.pth')
    plt.figure(1)
    plt.plot(train_loss_average_list)
    plt.figure(2)
    plt.plot(rewards_average_list)  
    plt.figure(3)
    plt.plot(epsilone_average_list) 
    plt.show()
    
    