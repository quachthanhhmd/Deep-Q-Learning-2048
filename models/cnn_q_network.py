import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Pre-activation residual block for stable gradient flow through deeper CNN.
    Uses BatchNorm to stabilize training with shaped rewards of varying scale.
    """
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.net(x), inplace=True)  # Skip connection


class CNNQNetwork(nn.Module):
    """
    Dueling CNN Q-Network with Residual Blocks for 2048.

    Architecture:
        Input (B, 16, 4, 4) [one-hot encoded board]
        → Stem Conv 256 + BN + ReLU
        → ResidualBlock x2  (preserves spatial info)
        → Flatten → 4096-dim feature vector
        → Dueling heads: Value V(s)  +  Advantage A(s,a)
        → Q(s,a) = V(s) + [A(s,a) - mean_a A(s,a)]

    Why this works better for 2048:
    - 256 filters: enough capacity to distinguish log2 tile patterns
    - Residual connections: stable gradients, learns compound merge patterns
    - BatchNorm: tames reward signal variance across different game phases
    - Dueling heads: decouples "how good is this board overall" from "which move is best"
    """

    def __init__(self, num_actions=4, in_channels=16):
        super().__init__()

        NUM_FILTERS = 256

        # Stem: project one-hot channels to feature space
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, NUM_FILTERS, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.ReLU(inplace=True),
        )

        # Two residual blocks for deep spatial reasoning
        self.res1 = ResidualBlock(NUM_FILTERS)
        self.res2 = ResidualBlock(NUM_FILTERS)

        feat_dim = NUM_FILTERS * 4 * 4  # 256 * 4 * 4 = 4096

        # Dueling Value head: V(s)
        self.value_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

        # Dueling Advantage head: A(s, a)
        self.adv_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions),
        )

        # Weight initialization for stable early training
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Reshape flat obs vector to (B, 16, 4, 4) one-hot board
        x = x.reshape(-1, 16, 4, 4).float()

        # Feature extraction
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = x.flatten(1)  # (B, 4096)

        # Dueling decomposition
        v = self.value_head(x)          # (B, 1)
        a = self.adv_head(x)            # (B, num_actions)

        # Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]
        return v + (a - a.mean(dim=1, keepdim=True))
