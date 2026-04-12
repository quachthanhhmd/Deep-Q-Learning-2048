import numpy as np
import math
from .openspiel_env import (
    OpenSpiel2048Env,
    parse_board_numbers,
    auto_resolve_chance_nodes,
    legal_actions,
    state_return,
    state_reward
)

def extract_one_hot_obs(state, player_id=0):
    board = parse_board_numbers(state)
    if board is None:
        board = np.zeros((4, 4), dtype=np.int64)
    
    one_hot = np.zeros((16, 4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            val = board[i, j]
            if val == 0:
                one_hot[0, i, j] = 1.0
            else:
                power = int(math.log2(val))
                power = min(power, 15)
                one_hot[power, i, j] = 1.0
                
    return one_hot.reshape(-1)

class OpenSpiel2048EnvCNN(OpenSpiel2048Env):
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.game.new_initial_state()
        auto_resolve_chance_nodes(self.state, self.rng)
        return extract_one_hot_obs(self.state, self.player_id)

    def step(self, action):
        if self.state is None:
            raise RuntimeError('Call reset() before step().')
        if self.state.is_terminal():
            raise RuntimeError('Episode already ended. Call reset().')

        legal = legal_actions(self.state, self.player_id)
        if action not in legal:
            raise ValueError(f'Illegal action {action}. Legal actions: {legal}')

        prev_return = state_return(self.state, self.player_id)

        self.state.apply_action(int(action))
        auto_resolve_chance_nodes(self.state, self.rng)

        if not self.state.is_terminal():
            next_obs = extract_one_hot_obs(self.state, self.player_id)
        else:
            next_obs = np.zeros(16 * 4 * 4, dtype=np.float32)
            
        new_return = state_return(self.state, self.player_id)
        reward = new_return - prev_return
        done = self.state.is_terminal()
        info = {
            'legal_actions': legal_actions(self.state, self.player_id) if not done else [],
            'state_return': new_return,
            'state_reward_raw': state_reward(self.state, self.player_id),
            'board': parse_board_numbers(self.state),
            'state_text': str(self.state),
        }
        return next_obs, float(reward), done, info

class OpenSpiel2048EnvCNNShaped(OpenSpiel2048EnvCNN):
    def step(self, action):
        # 1. Capture previous board state
        prev_board = parse_board_numbers(self.state)
        prev_max = np.max(prev_board) if prev_board is not None else 0
        prev_empty = np.sum(prev_board == 0) if prev_board is not None else 0

        # 2. Execute original step
        next_obs, raw_reward, done, info = super().step(action)
        board = info.get('board')

        shaped_reward = 0.0
        if board is not None:
            next_max = np.max(board)
            next_empty = np.sum(board == 0)

            # --- Dense merge reward (log-scaled raw score delta) ---
            # raw_reward is the actual score delta from the game engine
            # (sum of merged tile values). Log-scale prevents high-value
            # merges from dominating and drowning out smaller merge signals.
            if raw_reward > 0:
                shaped_reward += math.log1p(raw_reward) * 0.25

            # --- New max tile bonus ---
            # Extra incentive when the agent reaches a new personal best tile.
            # Scaled by log2 so 1024 is rewarded more than 512, etc.
            if next_max > prev_max:
                shaped_reward += math.log2(next_max) * 0.3

            # --- Empty tile bonus ---
            # More empty cells = more future merge opportunities.
            # In OpenSpiel, a new tile is added after the move (+1 correction).
            empty_delta = (next_empty - prev_empty + 1)
            shaped_reward += empty_delta * 0.1

        info['raw_reward_unshaped'] = raw_reward
        return next_obs, float(shaped_reward), done, info


