import torch.nn as nn

class CNNQNetwork(nn.Module):
    def __init__(self, num_actions=4, in_channels=16):
        super().__init__()
        # Input shape expected later: (Batch, 16, 4, 4)
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # After two 3x3 convolutions with padding=1, input remains 4x4
        # Feature map size: 128 * 4 * 4 = 2048
        self.fc_net = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        # OpenSpiel2048EnvShaped returns flat observation of 16*4*4 = 256 size
        x = x.reshape(-1, 16, 4, 4).float()
        x = self.conv_net(x)
        x = x.reshape(x.size(0), -1)  # Flatten
        return self.fc_net(x)
