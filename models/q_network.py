import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Simple MLP baseline."""
    def __init__(self, obs_dim, num_actions, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    def forward(self, x):
        return self.net(x)

class RefinedCNNEncoder(nn.Module):
    """
    CNN Encoder with split kernels (1x4, 4x1, 4x4) inspired by ml2048.
    Inputs: (N, 256) where 256 is flattened 16-channel 4x4 bitboard.
    Outputs: Latent vector of size out_features.
    """
    def __init__(self, out_features=512, multiplier=4):
        super().__init__()
        # Input channels = 16 (one-hot tile values)
        in_c = 16
        
        # Horizontal Branch: 1x4 kernel
        self.hori_conv = nn.Sequential(
            nn.Conv2d(in_c, in_c * multiplier, kernel_size=(1, 4), groups=in_c),
            nn.ReLU(),
            nn.Conv2d(in_c * multiplier, out_features // 4, kernel_size=1),
            nn.ReLU()
        )
        
        # Vertical Branch: 4x1 kernel
        self.vert_conv = nn.Sequential(
            nn.Conv2d(in_c, in_c * multiplier, kernel_size=(4, 1), groups=in_c),
            nn.ReLU(),
            nn.Conv2d(in_c * multiplier, out_features // 4, kernel_size=1),
            nn.ReLU()
        )
        
        # Global Branch: 4x4 kernel
        self.glob_conv = nn.Sequential(
            nn.Conv2d(in_c, in_c * multiplier, kernel_size=(4, 4), groups=in_c),
            nn.ReLU(),
            nn.Conv2d(in_c * multiplier, out_features // 4, kernel_size=1),
            nn.ReLU()
        )
        
        # Flatten and combine
        # hori: (N, F/4, 4, 1) -> 4 * F/4 = F
        # vert: (N, F/4, 1, 4) -> 4 * F/4 = F
        # glob: (N, F/4, 1, 1) -> F/4
        # Total units = F + F + F/4 = 2.25 F
        combined_units = (out_features // 4) * (4 + 4 + 1)
        
        self.fc = nn.Sequential(
            nn.Linear(combined_units, out_features),
            nn.ReLU()
        )

    def forward(self, x):
        # x is (N, 256) -> (N, 16, 4, 4)
        x = x.view(-1, 16, 4, 4)
        
        h = self.hori_conv(x).view(x.size(0), -1)
        v = self.vert_conv(x).view(x.size(0), -1)
        g = self.glob_conv(x).view(x.size(0), -1)
        
        combined = torch.cat([h, v, g], dim=1)
        return self.fc(combined)

class RefinedCNNQNetwork(nn.Module):
    """
    Dueling Q-Network using the RefinedCNNEncoder.
    """
    def __init__(self, num_actions, encoder_features=512, dueling=True):
        super().__init__()
        self.encoder = RefinedCNNEncoder(out_features=encoder_features)
        self.dueling = dueling
        
        if dueling:
            # Value stream
            self.value_head = nn.Sequential(
                nn.Linear(encoder_features, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
            # Advantage stream
            self.adv_head = nn.Sequential(
                nn.Linear(encoder_features, 256),
                nn.ReLU(),
                nn.Linear(256, num_actions)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(encoder_features, 256),
                nn.ReLU(),
                nn.Linear(256, num_actions)
            )

    def forward(self, x):
        latent = self.encoder(x)
        if self.dueling:
            val = self.value_head(latent)
            adv = self.adv_head(latent)
            return val + (adv - adv.mean(dim=1, keepdim=True))
        else:
            return self.head(latent)
