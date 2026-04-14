import torch
import torch.nn as nn
import torch.nn.functional as F
from .q_network import RefinedCNNEncoder

class PPOActorCriticNetwork(nn.Module):
    """
    PPO Actor-Critic using the RefinedCNNEncoder.
    """
    def __init__(self, num_actions, encoder_features=512):
        super().__init__()
        self.encoder = RefinedCNNEncoder(out_features=encoder_features)
        
        # Actor Head: Outputs logits for each action
        self.actor_head = nn.Sequential(
            nn.Linear(encoder_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        
        # Critic Head: Outputs a single state value
        self.critic_head = nn.Sequential(
            nn.Linear(encoder_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x, valid_actions_mask=None):
        latent = self.encoder(x)
        
        logits = self.actor_head(latent)
        value = self.critic_head(latent)
        
        if valid_actions_mask is not None:
            # Mask illegal actions in the logits (set to -inf or very small number)
            # valid_actions_mask is (N, 4)
            logits = logits.masked_fill(~valid_actions_mask.to(torch.bool), -1e9)
            
        return logits, value

    def get_value(self, x):
        latent = self.encoder(x)
        return self.critic_head(latent)

    def get_action_and_value(self, x, action=None, valid_actions_mask=None):
        logits, value = self.forward(x, valid_actions_mask)
        probs = torch.distributions.Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), value
