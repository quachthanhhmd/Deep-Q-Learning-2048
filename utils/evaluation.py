import numpy as np
import torch
from .action_selection import masked_greedy_action

def evaluate_model(q_net, env_class, num_seeds=10, max_steps=5_000, device="cpu"):
    returns, max_tiles = [], []
    q_net.eval()
    with torch.no_grad():
        for seed in range(num_seeds):
            env = env_class(seed=2000 + seed)
            obs = env.reset(seed=2000 + seed)
            done, ep_return, steps, max_tile = False, 0.0, 0, 0
            while not done and steps < max_steps:
                legal = env.legal_actions()
                if not legal: break
                action = masked_greedy_action(q_net, obs, legal, env.num_actions, epsilon=0.0, device=device)
                obs, reward, done, info = env.step(action)
                ep_return += reward
                steps += 1
                board = info.get('board')
                if board is not None:
                    max_tile = max(max_tile, int(np.max(board)))
            returns.append(ep_return)
            max_tiles.append(max_tile)
    q_net.train()
    return float(np.mean(returns)), float(np.std(returns)), float(np.mean(max_tiles))
