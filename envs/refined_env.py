import numpy as np
from .openspiel_env import OpenSpiel2048Env, parse_board_numbers, auto_resolve_chance_nodes, extract_obs, legal_actions, state_return

def board_to_one_hot(board, max_power=15):
    """
    Converts 4x4 board to (16, 4, 4) one-hot encoded tensor.
    Channels: 0 (empty), 1 (2), 2 (4), ..., 15 (32768)
    """
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

class Refined2048Env(OpenSpiel2048Env):
    def __init__(self, seed=42, reward_type="raw", corner_index=(0, 0)):
        super().__init__(seed=seed)
        self.reward_type = reward_type # "raw" or "corner"
        self.corner_index = corner_index
        # One-hot obs is 16 * 4 * 4 = 256
        self.obs_dim = 16 * 4 * 4 

    def get_obs(self):
        board = parse_board_numbers(self.state)
        if board is None: board = np.zeros((4, 4), dtype=np.int64)
        return board_to_one_hot(board).reshape(-1)

    def reset(self, seed=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.state = self.game.new_initial_state()
        auto_resolve_chance_nodes(self.state, self.rng)
        return self.get_obs()

    def step(self, action):
        legal = self.legal_actions()
        if action not in legal:
            # Fallback if illegal action is passed (though learner should prevent this)
            return self.get_obs(), -1.0, self.state.is_terminal(), {"legal_actions": legal}

        # Save previous state for potential-based reward
        prev_board = parse_board_numbers(self.state)
        prev_return = state_return(self.state, self.player_id)

        # Apply action
        self.state.apply_action(int(action))
        auto_resolve_chance_nodes(self.state, self.rng)
        
        done = self.state.is_terminal()
        next_obs = self.get_obs() if not done else np.zeros(self.obs_dim, dtype=np.float32)
        
        # Raw reward (score difference)
        new_return = state_return(self.state, self.player_id)
        reward = new_return - prev_return
        
        # Potential-based Corner Bonus
        if self.reward_type == "corner":
            bonus = 0
            factor = 64.0 # Scaling factor for the bonus (increased per research)
            
            curr_board = parse_board_numbers(self.state) if not done else None
            
            # Corner position (default 0,0)
            ci, cj = self.corner_index
            
            def get_potential(board):
                if board is None: return 0
                tile = board[ci, cj]
                return factor * tile if tile > 0 else 0

            # Delta potential
            bonus = get_potential(curr_board) - get_potential(prev_board)
            reward += bonus

        info = {
            "legal_actions": self.legal_actions() if not done else [],
            "board": parse_board_numbers(self.state),
            "raw_reward": new_return - prev_return
        }
        
        return next_obs, float(reward), done, info
