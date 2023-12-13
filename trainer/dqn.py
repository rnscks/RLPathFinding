import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # Assuming input size of 12x12x1 (single channel)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0)
        
        self.fc1 = nn.Linear(3200, 256)  # First fully connected layer
        self.fc2 = nn.Linear(256, 8)  # Output layer for 8 actions

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _initialize_weights(self):
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

# Create a DQN object with the architecture specified above
dqn = DQN()
