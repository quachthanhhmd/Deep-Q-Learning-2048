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

from utils import (ReplayBuffer, make_legal_mask, select_action_with_tracking, evaluate_model, run_comprehensive_eval, generate_evaluation_report)
from utils.ppo_utils import PPORolloutBuffer

def parse_args():
    parser = argparse.ArgumentParser(description="2048 Ablation Study")
    parser.add_argument("--experiment", type=str, default="dqn_base", 
                        choices=["dqn_base", "dqn_refined", "ppo_refined"],
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
    if EXPERIMENT_TYPE == "dqn_base":
        print("--- Mode: DQN Base (Original DQN + MLP + Raw) ---")
        train_env = OpenSpiel2048Env(seed=SEED)
        obs_dim = train_env.obs_dim
        num_actions = train_env.num_actions
        model = QNetwork(obs_dim, num_actions).to(DEVICE)
        
    elif EXPERIMENT_TYPE == "dqn_refined":
        print("--- Mode: DQN Refined (D3QN + SplitCNN + Corner Reward) ---")
        train_env = Refined2048Env(seed=SEED, reward_type="corner")
        obs_dim = train_env.obs_dim
        num_actions = train_env.num_actions
        model = RefinedCNNQNetwork(num_actions, dueling=True).to(DEVICE)
        
    elif EXPERIMENT_TYPE == "ppo_refined":
        print("--- Mode: PPO Refined (PPO + SplitCNN + Corner Reward) ---")
        train_env = Refined2048Env(seed=SEED, reward_type="corner")
        obs_dim = train_env.obs_dim
        num_actions = train_env.num_actions
        model = PPOActorCriticNetwork(num_actions).to(DEVICE)

    print(f"Environment initialized: type={type(train_env).__name__}")
    print(f"Observation dimension: {obs_dim}, Actions: {num_actions}")

    # Common metrics containers
    episode_returns, episode_lengths, loss_history, max_tile_history, illegal_avoided_ratio = [], [], [], [], []

    def save_results(exp_name, model):
        plt.figure(figsize=(20, 4))
        plt.subplot(1, 4, 1)
        plt.plot(episode_returns, alpha=0.35, label="episode return")
        ma = moving_average(episode_returns, 20)
        plt.plot(range(len(ma)), ma, label="moving avg (20)")
        plt.title(f"{exp_name} Returns")
        plt.xlabel("Episode")
        plt.legend()
        plt.subplot(1, 4, 2)
        plt.plot([np.log2(t + 1) for t in max_tile_history])
        plt.title(f"{exp_name} Max Tile (log2)")
        plt.subplot(1, 4, 3)
        plt.plot(loss_history, alpha=0.8)
        plt.title(f"{exp_name} Loss")
        plt.subplot(1, 4, 4)
        plt.plot(illegal_avoided_ratio)
        plt.title(f"{exp_name} Legal Avoidance")
        plt.tight_layout()
        plt.savefig(f"{exp_name}_results.png")
        print(f"Saved {exp_name}_results.png")

    def run_dqn_training():
        nonlocal global_step
        target_net = copy.deepcopy(model).to(DEVICE)
        target_net.eval()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        replay = ReplayBuffer(BUFFER_SIZE)
        
        def get_dqn_loss(batch):
            obs_b = torch.tensor(np.asarray(batch.obs), dtype=torch.float32, device=DEVICE)
            actions_b = torch.tensor(batch.action, dtype=torch.int64, device=DEVICE).unsqueeze(1)
            rewards_b = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE)
            next_obs_b = torch.tensor(np.asarray(batch.next_obs), dtype=torch.float32, device=DEVICE)
            dones_b = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE)
            n_legal_mask = torch.tensor(np.asarray(batch.next_legal_mask), dtype=torch.bool, device=DEVICE)

            q_vals = model(obs_b)
            q_sa = q_vals.gather(1, actions_b).squeeze(1)

            with torch.no_grad():
                # Double DQN logic
                n_q_online = model(next_obs_b)
                n_q_online = n_q_online.masked_fill(~n_legal_mask, -1e9)
                best_n_actions = torch.argmax(n_q_online, dim=1, keepdim=True)
                n_q_target = target_net(next_obs_b)
                n_max_q = n_q_target.gather(1, best_n_actions).squeeze(1)
                n_max_q = torch.where(dones_b > 0.5, torch.zeros_like(n_max_q), n_max_q)
                target_q = rewards_b + GAMMA * n_max_q

            loss = F.smooth_l1_loss(q_sa, target_q)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            return loss.item()

        for episode in tqdm(range(1, NUM_EPISODES + 1), desc=f"Training {EXPERIMENT_TYPE}"):
            obs = train_env.reset()
            done, ep_return, ep_len, legal_count, max_tile_ep = False, 0.0, 0, 0, 0
            while not done and ep_len < MAX_STEPS_PER_EPISODE:
                eps = EPS_START + min(1.0, global_step / EPS_DECAY_STEPS) * (EPS_END - EPS_START)
                legal = train_env.legal_actions()
                l_mask = make_legal_mask(num_actions, legal)
                action, was_legal = select_action_with_tracking(model, obs, legal, num_actions, eps, DEVICE)
                if was_legal: legal_count += 1
                next_obs, reward, done, info = train_env.step(action)
                if info.get("board") is not None: max_tile_ep = max(max_tile_ep, int(np.max(info["board"])))
                n_legal = info["legal_actions"] if not done else []
                n_l_mask = make_legal_mask(num_actions, n_legal)
                replay.add(obs, action, reward, next_obs, done, l_mask, n_l_mask)
                obs, ep_return, ep_len, global_step = next_obs, ep_return + reward, ep_len + 1, global_step + 1
                if len(replay) >= LEARN_START and global_step % LEARN_EVERY == 0:
                    loss_history.append(get_dqn_loss(replay.sample(BATCH_SIZE)))
                if global_step % TARGET_SYNC_EVERY == 0:
                    target_net.load_state_dict(model.state_dict())
            
            episode_returns.append(ep_return)
            max_tile_history.append(max_tile_ep)
            illegal_avoided_ratio.append(legal_count / ep_len if ep_len > 0 else 1.0)
            if episode % 100 == 0:
                print(f"Ep {episode} | Ret: {np.mean(episode_returns[-100:]):.1f} | Loss: {np.mean(loss_history[-100:]):.4f} | Eps: {eps:.3f}")

    def run_ppo_training():
        nonlocal global_step
        optimizer = optim.Adam(model.parameters(), lr=config.get("ppo_lr", 3e-4), eps=1e-5)
        PPO_EPOCHS = config.get("ppo_epochs", 4)
        PPO_CLIP = config.get("ppo_clip", 0.2)
        PPO_ENTROPY_COEF = config.get("ppo_entropy_coef", 0.01)
        PPO_CRITIC_COEF = config.get("ppo_vf_coef", 0.5)
        PPO_STEPS = config.get("ppo_steps", 128)
        GAE_LAMBDA = config.get("gae_lambda", 0.95)
        
        rollout = PPORolloutBuffer(obs_dim, num_actions, PPO_STEPS, 1, DEVICE)
        
        episode_count = 0
        pbar = tqdm(total=NUM_EPISODES, desc=f"Training {EXPERIMENT_TYPE}")
        
        obs = train_env.reset()
        ep_return, ep_len, max_tile_ep, legal_count = 0, 0, 0, 0
        
        while episode_count < NUM_EPISODES:
            # Collect fixed number of steps
            for s in range(PPO_STEPS):
                global_step += 1
                legal = train_env.legal_actions()
                l_mask = make_legal_mask(num_actions, legal)
                with torch.no_grad():
                    action, logprob, entropy, value = model.get_action_and_value(
                        torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0),
                        valid_actions_mask=torch.tensor(l_mask, dtype=torch.bool, device=DEVICE).unsqueeze(0)
                    )
                
                next_obs, reward, done, info = train_env.step(action.item())
                if info.get("board") is not None: max_tile_ep = max(max_tile_ep, int(np.max(info["board"])))
                rollout.add(s, obs, action, logprob, reward, done, value, l_mask)
                
                obs, ep_return, ep_len = next_obs, ep_return + reward, ep_len + 1
                
                if done or ep_len >= MAX_STEPS_PER_EPISODE: 
                    episode_count += 1
                    pbar.update(1)
                    episode_returns.append(ep_return)
                    max_tile_history.append(max_tile_ep)
                    illegal_avoided_ratio.append(1.0) # Masking handles this

                    if episode_count % 100 == 0:
                        print(f"Ep {episode_count} | Ret: {np.mean(episode_returns[-100:]):.1f} | Loss: {np.mean(loss_history[-100:]) if loss_history else 0:.4f}")
                    
                    obs = train_env.reset()
                    ep_return, ep_len, max_tile_ep, legal_count = 0, 0, 0, 0
            
            # PPO Update
            with torch.no_grad():
                _, next_val = model(torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0))
            b_obs, b_lp, b_act, b_adv, b_ret, b_val, b_msk = rollout.get_batches(
                next_val, torch.tensor([done], device=DEVICE), GAMMA, GAE_LAMBDA
            )
            
            for _ in range(PPO_EPOCHS):
                new_act, new_lp, entropy, new_val = model.get_action_and_value(b_obs, b_act, b_msk)
                logratio = new_lp - b_lp
                ratio = logratio.exp()
                mb_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = 0.5 * F.smooth_l1_loss(new_val.flatten(), b_ret)
                entropy_loss = entropy.mean()
                loss = pg_loss - PPO_ENTROPY_COEF * entropy_loss + v_loss * PPO_CRITIC_COEF
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                loss_history.append(loss.item())

        pbar.close()
    global_step = 0
    if EXPERIMENT_TYPE in ["dqn_base", "dqn_refined"]:
        run_dqn_training()
    else:
        run_ppo_training()

    save_results(EXPERIMENT_TYPE, model)
    
    # --- Final Comprehensive Evaluation (1000 games) ---
    eval_games = 1000
    if NUM_EPISODES <= 20: eval_games = 5 # Rapid debug mode
    
    print(f"\n--- Starting Final Comprehensive Evaluation ({eval_games} games) ---")
    eval_results = run_comprehensive_eval(model, type(train_env), num_episodes=eval_games, device=DEVICE)
    
    # Generate Advanced Report
    generate_evaluation_report(eval_results, EXPERIMENT_TYPE)
    
    # Final Summary Print
    mean_r = np.mean(eval_results["returns"])
    std_r = np.std(eval_results["returns"])
    mean_tile = np.mean(eval_results["max_tiles"])
    print(f"\n[FINAL REPORT: {EXPERIMENT_TYPE}]")
    print(f"Mean Return: {mean_r:.1f} ± {std_r:.1f}")
    print(f"Mean Max Tile: {mean_tile:.1f}")
    print(f"Best Score: {np.max(eval_results['returns']):.0f}")
    print(f"Max Tile Reached: {np.max(eval_results['max_tiles'])}")

if __name__ == "__main__":
    main()
