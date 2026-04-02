import torch.nn as nn

class DuelingCNNQNetwork(nn.Module):
    def __init__(self, num_actions, in_channels=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(128 * 4 * 4, 512)
        self.relu = nn.ReLU()
        self.v = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1))
        self.a = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, num_actions))
        
    def forward(self, x):
        x = x.reshape(-1, 16, 4, 4).float()
        feat = self.relu(self.fc(self.conv(x)))
        v = self.v(feat)
        a = self.a(feat)
        return v + (a - a.mean(dim=1, keepdim=True))
