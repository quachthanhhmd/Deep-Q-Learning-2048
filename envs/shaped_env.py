import numpy as np
from .openspiel_env import OpenSpiel2048Env, parse_board_numbers, auto_resolve_chance_nodes, extract_obs, legal_actions, state_return, state_reward

def board_to_binary_channels(board, max_power=15):
    channels = np.zeros((max_power + 1, 4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            tile = int(board[i, j])
            if tile > 0:
                p = int(np.log2(tile))
                if p <= max_power:
                    channels[p, i, j] = 1.0
            else:
                channels[0, i, j] = 1.0
    return channels

def extract_binary_obs(state, player_id=0):
    board = parse_board_numbers(state)
    if board is None: board = np.zeros((4, 4), dtype=np.int64)
    return board_to_binary_channels(board).reshape(-1)

class OpenSpiel2048EnvShaped(OpenSpiel2048Env):
    def reset(self, seed=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.state = self.game.new_initial_state()
        auto_resolve_chance_nodes(self.state, self.rng)
        return extract_binary_obs(self.state, self.player_id)

    def step(self, action):
        legal = legal_actions(self.state, self.player_id)
        if action not in legal: raise ValueError(f"Illegal action {action}. Legal actions: {legal}")
        prev_r = state_return(self.state, self.player_id)
        self.state.apply_action(int(action))
        auto_resolve_chance_nodes(self.state, self.rng)
        done = self.state.is_terminal()
        next_o = extract_binary_obs(self.state, self.player_id) if not done else np.zeros(16 * 4 * 4, dtype=np.float32)
        r_raw = state_reward(self.state, self.player_id)
        board = parse_board_numbers(self.state) if not done else np.zeros((4, 4), dtype=np.int64)
        empty = int(np.sum(board == 0)) if board is not None else 0
        shaped_reward = (state_return(self.state, self.player_id) - prev_r) * 0.1 + np.log2(empty + 1)
        info = {
            "legal_actions": legal_actions(self.state, self.player_id) if not done else [],
            "state_reward_raw": r_raw,
            "board": board,
            "state_text": str(self.state),
        }
        return next_o, float(shaped_reward), done, info
