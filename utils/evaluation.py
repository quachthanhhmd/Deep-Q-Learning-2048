import numpy as np
import torch
from .action_selection import masked_greedy_action

def evaluate_model(model, env_class, num_seeds=10, max_steps=5_000, device="cpu"):
    """Quick evaluation for periodic logging."""
    returns, max_tiles = [], []
    model.eval()
    with torch.no_grad():
        for seed in range(num_seeds):
            env = env_class(seed=2000 + seed)
            obs = env.reset()
            done, ep_return, steps, max_tile = False, 0.0, 0, 0
            while not done and steps < max_steps:
                legal = env.legal_actions()
                if not legal: break
                action = masked_greedy_action(model, obs, legal, env.num_actions, epsilon=0.0, device=device)
                obs, reward, done, info = env.step(action)
                ep_return += reward
                steps += 1
                board = info.get('board')
                if board is not None:
                    max_tile = max(max_tile, int(np.max(board)))
            returns.append(ep_return)
            max_tiles.append(max_tile)
    model.train()
    return float(np.mean(returns)), float(np.std(returns)), float(np.mean(max_tiles))

def run_comprehensive_eval(model, env_class, num_episodes=1000, max_steps=5_000, device="cpu"):
    """Detailed evaluation over a large number of games to collect metrics for reporting."""
    all_returns = []
    all_max_tiles = []
    all_steps = []
    
    model.eval()
    with torch.no_grad():
        for i in range(num_episodes):
            env = env_class(seed=10000 + i)
            obs = env.reset()
            done, ep_return, steps, max_tile = False, 0.0, 0, 0
            
            while not done and steps < max_steps:
                legal = env.legal_actions()
                if not legal: break
                
                action = masked_greedy_action(model, obs, legal, env.num_actions, epsilon=0.0, device=device)
                obs, reward, done, info = env.step(action)
                
                ep_return += reward
                steps += 1
                board = info.get('board')
                if board is not None:
                    max_tile = max(max_tile, int(np.max(board)))
            
            all_returns.append(ep_return)
            all_max_tiles.append(max_tile)
            all_steps.append(steps)
            
    model.train()
    return {
        "returns": all_returns,
        "max_tiles": all_max_tiles,
        "steps": all_steps
    }
