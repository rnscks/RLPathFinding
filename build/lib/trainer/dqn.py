import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.fc = nn.Linear(12*12*32, 8)
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 12*12*32)
        x = self.fc(x)
        
        return x