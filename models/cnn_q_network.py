import torch
import torch.nn as nn

class CNNQNetwork(nn.Module):
    def __init__(self, num_actions=4, in_channels=16):
        super().__init__()
        
        # Multi-Branch "Tuple" architecture for 2048 game mechanics
        # Evaluates 2-tile, 3-tile, 4-tile sliding chunks plus 2x2 blocks
        self.conv_1x2 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=(1, 2)), nn.ReLU())
        self.conv_2x1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=(2, 1)), nn.ReLU())
        
        self.conv_1x3 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=(1, 3)), nn.ReLU())
        self.conv_3x1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=(3, 1)), nn.ReLU())
        
        self.conv_1x4 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=(1, 4)), nn.ReLU())
        self.conv_4x1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=(4, 1)), nn.ReLU())
        
        self.conv_2x2 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=(2, 2)), nn.ReLU())
        
        # Combined feature size:
        # 1x2/2x1: 4x3x64 * 2 = 1536
        # 1x3/3x1: 4x2x64 * 2 = 1024
        # 1x4/4x1: 4x1x64 * 2 = 512
        # 2x2: 3x3x64 = 576
        # Sum = 3648
        
        self.fc_net = nn.Sequential(
            nn.Linear(3648, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        # Input state is flattened across 16 channels: size 16 * 4 * 4
        x = x.reshape(-1, 16, 4, 4).float()
        
        out1 = self.conv_1x2(x).flatten(start_dim=1)
        out2 = self.conv_2x1(x).flatten(start_dim=1)
        out3 = self.conv_1x3(x).flatten(start_dim=1)
        out4 = self.conv_3x1(x).flatten(start_dim=1)
        out5 = self.conv_1x4(x).flatten(start_dim=1)
        out6 = self.conv_4x1(x).flatten(start_dim=1)
        out7 = self.conv_2x2(x).flatten(start_dim=1)
        
        combined = torch.cat([out1, out2, out3, out4, out5, out6, out7], dim=1)
        
        return self.fc_net(combined)

