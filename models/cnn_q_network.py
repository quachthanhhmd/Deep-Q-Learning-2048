import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNQNetwork(nn.Module):
    """
    Multi-Branch Dueling CNN Q-Network for 2048.

    Combines the proven multi-branch directional convolution architecture
    (1x2 horizontal + 2x1 vertical kernels) from the reference notebook
    with modern DQN improvements for stable off-policy training.

    Architecture:
        Input (B, 16, 4, 4) one-hot encoded board
        ┌─ Conv(1×2) → GN → ReLU ─ h1 (B,128,4,3) ─┬─ Conv(1×2) → GN → ReLU ─ hh (B,128,4,2)
        │                                            └─ Conv(2×1) → GN → ReLU ─ hv (B,128,3,3)
        │
        └─ Conv(2×1) → GN → ReLU ─ v1 (B,128,3,4) ─┬─ Conv(1×2) → GN → ReLU ─ vh (B,128,3,3)
                                                     └─ Conv(2×1) → GN → ReLU ─ vv (B,128,2,4)

        Concat [h1, v1, hh, hv, vh, vv] → 7424-dim
        ├─ Value head:     FC(256) → ReLU → FC(1)
        └─ Advantage head: FC(256) → ReLU → FC(4)
        → Q(s,a) = V(s) + A(s,a) - mean[A]

    Why Multi-Branch works for 2048:
    - Kernel 1×2 detects horizontal merge pairs (left/right moves)
    - Kernel 2×1 detects vertical merge pairs (up/down moves)
    - Layer 2 cross-branches detect compound merge combos
    - Concat of all 6 maps gives both local + semi-global board view
    - 3×3 kernels waste capacity on diagonal relationships that don't exist in 2048

    Our improvements over the reference notebook:
    - Dueling heads: decouple board evaluation from action selection
    - GroupNorm: stable normalization for off-policy replay buffer training
    - Zero-init output: prevents Q-value explosion in early training
    - Double DQN compatible (handled in train.py)
    """

    def __init__(self, num_actions=4, in_channels=16):
        super().__init__()

        depth1 = 128  # Layer 1 filter count
        depth2 = 128  # Layer 2 filter count

        # === Layer 1: Directional convolutions on raw one-hot board ===
        # Horizontal branch: captures left-right adjacency (merge potential)
        self.conv1_h = nn.Conv2d(in_channels, depth1, kernel_size=(1, 2))
        self.gn1_h = nn.GroupNorm(1, depth1)

        # Vertical branch: captures up-down adjacency (merge potential)
        self.conv1_v = nn.Conv2d(in_channels, depth1, kernel_size=(2, 1))
        self.gn1_v = nn.GroupNorm(1, depth1)

        # === Layer 2: Cross-directional convolutions ===
        # From horizontal (4,3): explore both directions at deeper level
        self.conv2_hh = nn.Conv2d(depth1, depth2, kernel_size=(1, 2))  # → (4, 2)
        self.gn2_hh = nn.GroupNorm(1, depth2)

        self.conv2_hv = nn.Conv2d(depth1, depth2, kernel_size=(2, 1))  # → (3, 3)
        self.gn2_hv = nn.GroupNorm(1, depth2)

        # From vertical (3,4): explore both directions at deeper level
        self.conv2_vh = nn.Conv2d(depth1, depth2, kernel_size=(1, 2))  # → (3, 3)
        self.gn2_vh = nn.GroupNorm(1, depth2)

        self.conv2_vv = nn.Conv2d(depth1, depth2, kernel_size=(2, 1))  # → (2, 4)
        self.gn2_vv = nn.GroupNorm(1, depth2)

        # Feature dimensions after flatten + concat:
        # h1:  4×3×128 = 1536    v1:  3×4×128 = 1536
        # hh:  4×2×128 = 1024    hv:  3×3×128 = 1152
        # vh:  3×3×128 = 1152    vv:  2×4×128 = 1024
        # Total = 7424
        feat_dim = 7424

        # Dueling Value head: V(s) — "how good is this board state overall?"
        self.value_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        # Dueling Advantage head: A(s,a) — "which direction is best here?"
        self.adv_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_actions),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # CRITICAL: Zero-init final output layers so initial Q ≈ 0
        # Prevents catastrophic loss explosion at the start of training
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)
        nn.init.zeros_(self.adv_head[-1].weight)
        nn.init.zeros_(self.adv_head[-1].bias)

    def forward(self, x):
        # Reshape flat observation to (B, 16, 4, 4) one-hot board
        x = x.reshape(-1, 16, 4, 4).float()

        # === Layer 1: Directional feature extraction ===
        h1 = F.relu(self.gn1_h(self.conv1_h(x)), inplace=True)   # (B, 128, 4, 3)
        v1 = F.relu(self.gn1_v(self.conv1_v(x)), inplace=True)   # (B, 128, 3, 4)

        # === Layer 2: Cross-directional deepening ===
        hh = F.relu(self.gn2_hh(self.conv2_hh(h1)), inplace=True)  # (B, 128, 4, 2)
        hv = F.relu(self.gn2_hv(self.conv2_hv(h1)), inplace=True)  # (B, 128, 3, 3)
        vh = F.relu(self.gn2_vh(self.conv2_vh(v1)), inplace=True)  # (B, 128, 3, 3)
        vv = F.relu(self.gn2_vv(self.conv2_vv(v1)), inplace=True)  # (B, 128, 2, 4)

        # === Concat all 6 feature maps (layer1 + layer2) ===
        feat = torch.cat([
            h1.flatten(1),   # 1536
            v1.flatten(1),   # 1536
            hh.flatten(1),   # 1024
            hv.flatten(1),   # 1152
            vh.flatten(1),   # 1152
            vv.flatten(1),   # 1024
        ], dim=1)            # Total: 7424

        # === Dueling decomposition ===
        v = self.value_head(feat)   # (B, 1)
        a = self.adv_head(feat)     # (B, num_actions)

        # Q(s,a) = V(s) + [A(s,a) - mean_a A(s,a)]
        return v + (a - a.mean(dim=1, keepdim=True))
