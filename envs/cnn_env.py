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
        next_obs, raw_reward, done, info = super().step(action)
        board = info.get('board')
        
        shaped_reward = 0.0
        if board is not None:
            # 1. Base log reward
            if raw_reward > 0:
                shaped_reward += math.log2(raw_reward + 1)
            
            # 2. Empty spots reward
            empty_spots = np.sum(board == 0)
            shaped_reward += empty_spots * 0.1  
            
            # 3. Corner reward
            max_val = np.max(board)
            if board[3, 3] == max_val:
                shaped_reward += 5.0
                
            # 4. Monotonicity
            weights = np.array([
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.4, 0.6, 0.8],
                [0.3, 0.6, 1.0, 1.5],
                [0.4, 0.8, 1.5, 3.0]
            ])
            # Avoid RuntimeWarning by replacing 0 with 1 before log2 (log2(1) = 0 anyway)
            log_board = np.where(board > 0, np.log2(np.maximum(board, 1)), 0)
            shaped_reward += np.sum(log_board * weights) * 0.1

        info['raw_reward_unshaped'] = raw_reward
        return next_obs, float(shaped_reward), done, info
