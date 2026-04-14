import random
import numpy as np
import torch

def make_legal_mask(num_actions, legal_actions_list):
    mask = np.zeros(num_actions, dtype=np.float32)
    mask[legal_actions_list] = 1.0
    return mask

@torch.no_grad()
def masked_greedy_action(q_net, obs, legal_actions_list, num_actions, epsilon=0.0, device="cpu"):
    if random.random() < epsilon:
        return random.choice(legal_actions_list)

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    
    out = q_net(obs_t)
    if isinstance(out, tuple):
        q = out[0].squeeze(0) # PPO returns (logits, value)
    else:
        q = out.squeeze(0) # DQN returns q_values

    legal_mask = torch.zeros(num_actions, dtype=torch.bool, device=device)
    legal_mask[legal_actions_list] = True

    q_masked = q.masked_fill(~legal_mask, -1e9)
    action = int(torch.argmax(q_masked).item())
    return action

def select_action_with_tracking(q_net, obs, legal_actions_list, num_actions, epsilon, device):
    action = masked_greedy_action(q_net, obs, legal_actions_list, num_actions, epsilon, device)
    return action, True
