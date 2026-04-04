import os
import json
import argparse
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from envs import OpenSpiel2048Env, OpenSpiel2048EnvShaped, OpenSpiel2048EnvCNNShaped
from models import QNetwork, DuelingQNetwork, DuelingCNNQNetwork, CNNQNetwork
from utils import (ReplayBuffer, make_legal_mask, select_action_with_tracking, evaluate_model)

def parse_args():
    parser = argparse.ArgumentParser(description="Train 2048 RL Agents")
    parser.add_argument("--experiment", type=str, default="dqn", choices=["dqn", "ddqn", "dddqn", "d3qn", "cnn_ddqn"],
                        help="Choose which experiment to run")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to hyperparameters JSON config file")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def moving_average(x, w=20):
    if len(x) < w: return np.asarray(x)
    return np.convolve(x, np.ones(w)/w, mode="valid")

def main():
    args = parse_args()
    config = load_config(args.config)
    EXPERIMENT_TYPE = args.experiment.lower()

    # Extract Config
    SEED = config["seed"]
    NUM_EPISODES = config["num_episodes"]
    BUFFER_SIZE = config["buffer_size"]
    BATCH_SIZE = config["batch_size"]
    GAMMA = config["gamma"]
    LR = config["lr"]
    TARGET_SYNC_EVERY = config["target_sync_every"]
    LEARN_START = config["learn_start"]
    LEARN_EVERY = config["learn_every"]
    EPS_START = config["eps_start"]
    EPS_END = config["eps_end"]
    EPS_DECAY_STEPS = config["eps_decay_steps"]
    MAX_STEPS_PER_EPISODE = config["max_steps_per_episode"]
    GRAD_CLIP = config["grad_clip"]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"--- Running Experiment: {EXPERIMENT_TYPE} ---")
    print(f"Using Device: {DEVICE}")

    # Set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 1. Initialize Environment and Architecture
    if EXPERIMENT_TYPE in ["d3qn", "cnn_ddqn"]:
        # Group 2: Enhanced Model with categorical / spatial features
        if EXPERIMENT_TYPE == "d3qn":
            train_env = OpenSpiel2048EnvShaped(seed=SEED)
            obs_dim = 16 * 4 * 4  # 256
            num_actions = train_env.num_actions
            q_net = DuelingCNNQNetwork(num_actions, in_channels=16).to(DEVICE)
        else: # cnn_ddqn
            train_env = OpenSpiel2048EnvCNNShaped(seed=SEED)
            obs_dim = 16 * 4 * 4  # 256
            num_actions = train_env.num_actions
            q_net = CNNQNetwork(num_actions, in_channels=16).to(DEVICE)
    else:
        # Group 1: Ablation (dqn, ddqn, dddqn)
        train_env = OpenSpiel2048Env(seed=SEED)
        obs_dim = train_env.obs_dim  # Usually ~100+ for Group 1 (flat vector)
        num_actions = train_env.num_actions
        
        if EXPERIMENT_TYPE == "dddqn":
            q_net = DuelingQNetwork(obs_dim, num_actions).to(DEVICE)
        else: # dqn and ddqn
            q_net = QNetwork(obs_dim, num_actions).to(DEVICE)

    print(f"Environment initialized: type={type(train_env).__name__}")
    print(f"Observation dimension: {obs_dim}")
    print(f"Number of actions: {num_actions}")

    target_net = copy.deepcopy(q_net).to(DEVICE)
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    replay = ReplayBuffer(BUFFER_SIZE)
    global_step = 0

    print(f"Environment initialized: obs_dim={obs_dim}, num_actions={num_actions}")

    def epsilon_by_step(step):
        frac = min(1.0, step / EPS_DECAY_STEPS)
        return EPS_START + frac * (EPS_END - EPS_START)

    def get_update_loss(batch):
        obs = torch.tensor(np.asarray(batch.obs), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(batch.action, dtype=torch.int64, device=DEVICE).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE)
        next_obs = torch.tensor(np.asarray(batch.next_obs), dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE)
        next_legal_mask = torch.tensor(np.asarray(batch.next_legal_mask), dtype=torch.bool, device=DEVICE)

        q_values = q_net(obs)
        q_sa = q_values.gather(1, actions).squeeze(1)

        with torch.no_grad():
            if EXPERIMENT_TYPE in ["ddqn", "dddqn", "d3qn", "cnn_ddqn"]:
                # Double Q learning loop
                next_q_online = q_net(next_obs)
                next_q_online = next_q_online.masked_fill(~next_legal_mask.to(torch.bool), -1e9)
                best_next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
                next_q_target = target_net(next_obs)
                next_max_q = next_q_target.gather(1, best_next_actions).squeeze(1)
            else:
                # Standard DQN update
                next_q = target_net(next_obs)
                next_q = next_q.masked_fill(~next_legal_mask.to(torch.bool), -1e9)
                next_max_q = torch.max(next_q, dim=1).values
                
            next_max_q = torch.where(dones > 0.5, torch.zeros_like(next_max_q), next_max_q)
            target = rewards + GAMMA * next_max_q

        loss = F.mse_loss(q_sa, target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q_net.parameters(), GRAD_CLIP)
        optimizer.step()
        return float(loss.item())

    # 2. Training Loop
    episode_returns, episode_lengths, loss_history, max_tile_history, illegal_avoided_ratio = [], [], [], [], []

    for episode in tqdm(range(1, NUM_EPISODES + 1), desc=f"Training {EXPERIMENT_TYPE}"):
        obs = train_env.reset(seed=SEED + episode)
        done, ep_return, ep_len, legal_count, max_tile_ep = False, 0.0, 0, 0, 0
        
        while not done and ep_len < MAX_STEPS_PER_EPISODE:
            eps = epsilon_by_step(global_step)
            legal = train_env.legal_actions()
            legal_mask = make_legal_mask(num_actions, legal)
            
            action, was_legal = select_action_with_tracking(
                q_net, obs, legal, num_actions, eps, DEVICE
            )
            if was_legal:
                legal_count += 1
                
            next_obs, reward, done, info = train_env.step(action)
            board = info.get("board")
            if board is not None:
                max_tile_ep = max(max_tile_ep, int(np.max(board)))
                
            next_legal = info["legal_actions"] if not done else []
            next_legal_mask = make_legal_mask(num_actions, next_legal)
            
            replay.add(obs, action, reward, next_obs, done, legal_mask, next_legal_mask)
            
            obs = next_obs
            ep_return += reward
            ep_len += 1
            global_step += 1
            
            if len(replay) >= LEARN_START and global_step % LEARN_EVERY == 0:
                batch = replay.sample(BATCH_SIZE)
                loss = get_update_loss(batch)
                loss_history.append(loss)
                
            if global_step % TARGET_SYNC_EVERY == 0:
                target_net.load_state_dict(q_net.state_dict())
                
        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        max_tile_history.append(max_tile_ep)
        illegal_avoided_ratio.append(legal_count / ep_len if ep_len > 0 else 1.0)

        # 2a. Periodic Logging
        log_interval = 100
        if NUM_EPISODES <= 20: 
            log_interval = 2 # Show more logs in debug mode
            
        if episode % log_interval == 0:
            avg_return = np.mean(episode_returns[-log_interval:])
            avg_max_tile = np.mean(max_tile_history[-log_interval:])
            avg_loss = np.mean(loss_history[-log_interval:]) if len(loss_history) >= log_interval else (np.mean(loss_history) if loss_history else 0)
            avg_legal = np.mean(illegal_avoided_ratio[-log_interval:])
            print(f"Episode {episode:5d}/{NUM_EPISODES} | "
                  f"Return: {avg_return:7.1f} | "
                  f"Max Tile: {avg_max_tile:5.1f} | "
                  f"Loss: {avg_loss:8.4f} | "
                  f"Legal%: {avg_legal * 100:5.1f}% | "
                  f"Steps: {global_step:7d} | "
                  f"Eps: {eps:.3f}")

    print("Training complete.")

    # 3. Final Evaluation
    mean_r, std_r, mean_tile = evaluate_model(q_net, type(train_env), num_seeds=10, device=DEVICE)
    print(f"[{EXPERIMENT_TYPE}] 10-seed eval: Mean Return = {mean_r:.1f} ± {std_r:.1f}, Mean Max Tile = {mean_tile:.1f}")

    # 4. Plots
    plt.figure(figsize=(20, 4))
    
    plt.subplot(1, 4, 1)
    plt.plot(episode_returns, alpha=0.35, label="episode return")
    ma = moving_average(episode_returns, 20)
    plt.plot(range(len(ma)), ma, label="moving avg (20)")
    plt.title(f"{EXPERIMENT_TYPE} Training Returns")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()

    plt.subplot(1, 4, 2)
    log2_tiles = [np.log2(t + 1) for t in max_tile_history]
    plt.plot(log2_tiles)
    plt.title(f"{EXPERIMENT_TYPE} Max Tile (log2)")
    plt.xlabel("Episode")
    
    plt.subplot(1, 4, 3)
    plt.plot(loss_history, alpha=0.8)
    plt.title(f"{EXPERIMENT_TYPE} Loss")
    plt.xlabel("Update Step")
    
    plt.subplot(1, 4, 4)
    plt.plot(illegal_avoided_ratio)
    plt.title(f"{EXPERIMENT_TYPE} Illegal Avoidance")
    plt.xlabel("Episode")

    plt.tight_layout()
    plt.savefig(f"{EXPERIMENT_TYPE}_results.png")
    print(f"Saved {EXPERIMENT_TYPE}_results.png")

if __name__ == "__main__":
    main()
