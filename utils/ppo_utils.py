import torch
import numpy as np

class PPORolloutBuffer:
    """
    Rollout buffer for PPO. Stores experience from multiple steps/games.
    """
    def __init__(self, obs_dim, num_actions, num_steps, num_games, device="cpu"):
        self.obs = torch.zeros((num_steps, num_games, obs_dim), device=device)
        self.actions = torch.zeros((num_steps, num_games), device=device)
        self.logprobs = torch.zeros((num_steps, num_games), device=device)
        self.rewards = torch.zeros((num_steps, num_games), device=device)
        self.dones = torch.zeros((num_steps, num_games), device=device)
        self.values = torch.zeros((num_steps, num_games), device=device)
        self.valid_masks = torch.zeros((num_steps, num_games, num_actions), device=device)
        
        self.step = 0
        self.num_steps = num_steps
        self.num_games = num_games
        self.device = device

    def add(self, rollout_step, obs, action, logprob, reward, done, value, valid_mask):
        self.obs[rollout_step] = torch.tensor(obs, device=self.device)
        self.actions[rollout_step] = action
        self.logprobs[rollout_step] = logprob
        self.rewards[rollout_step] = torch.tensor(reward, device=self.device)
        self.dones[rollout_step] = torch.tensor(done, device=self.device)
        self.values[rollout_step] = value.flatten()
        self.valid_masks[rollout_step] = torch.tensor(valid_mask, device=self.device)

    def get_batches(self, next_value, next_done, gamma, gae_lambda):
        # Advantages computation (GAE)
        advantages = torch.zeros_like(self.rewards)
        lastgaelam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - next_done.float()
                nextvalues = next_value.flatten()
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
            
            # temporal difference error
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        
        returns = advantages + self.values
        
        # Flatten for batch training
        b_obs = self.obs.reshape((-1,) + self.obs.shape[2:])
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)
        b_masks = self.valid_masks.reshape((-1,) + self.valid_masks.shape[2:])
        
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_masks
