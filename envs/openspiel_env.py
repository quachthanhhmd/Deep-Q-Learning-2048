import re
import numpy as np
import pyspiel

def extract_obs(state, player_id=0):
    for fn_name, args in [("observation_tensor", (player_id,)),("observation_tensor", tuple()),("information_state_tensor", (player_id,)),("information_state_tensor", tuple()),]:
        fn = getattr(state, fn_name, None)
        if fn is None: continue
        try:
            obs = fn(*args)
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            return obs
        except TypeError: pass
    raise RuntimeError("Could not extract an observation tensor from state.")

def legal_actions(state, player_id=0):
    try: return list(state.legal_actions(player_id))
    except TypeError: return list(state.legal_actions())

def sample_chance_action(state, rng):
    outcomes = state.chance_outcomes()
    actions, probs = zip(*outcomes)
    idx = rng.choice(len(actions), p=np.asarray(probs, dtype=np.float64))
    return actions[idx]

def auto_resolve_chance_nodes(state, rng):
    while state.is_chance_node() and not state.is_terminal():
        a = sample_chance_action(state, rng)
        state.apply_action(a)
    return state

def state_return(state, player_id=0):
    vals = state.returns()
    return float(vals[player_id]) if len(vals) > player_id else 0.0

def state_reward(state, player_id=0):
    vals = state.rewards()
    return float(vals[player_id]) if len(vals) > player_id else 0.0

def parse_board_numbers(state):
    txt = str(state)
    nums = [int(x) for x in re.findall(r"\d+", txt)]
    if len(nums) >= 16:
        nums = nums[-16:]
        return np.array(nums, dtype=np.int64).reshape(4, 4)
    return None

class OpenSpiel2048Env:
    def __init__(self, seed=42):
        self.game = pyspiel.load_game("2048")
        self.player_id = 0
        self.num_actions = self.game.num_distinct_actions()
        self.obs_dim = self.game.observation_tensor_size()
        self.rng = np.random.default_rng(seed)
        self.state = None

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.game.new_initial_state()
        auto_resolve_chance_nodes(self.state, self.rng)
        return extract_obs(self.state, self.player_id)

    def step(self, action):
        if self.state is None: raise RuntimeError("Call reset() before step().")
        if self.state.is_terminal(): raise RuntimeError("Episode already ended. Call reset().")
        legal = legal_actions(self.state, self.player_id)
        if action not in legal: raise ValueError(f"Illegal action {action}. Legal actions: {legal}")
        prev_return = state_return(self.state, self.player_id)
        self.state.apply_action(int(action))
        auto_resolve_chance_nodes(self.state, self.rng)
        next_obs = extract_obs(self.state, self.player_id) if not self.state.is_terminal() else np.zeros(self.obs_dim, dtype=np.float32)
        new_return = state_return(self.state, self.player_id)
        reward = new_return - prev_return
        done = self.state.is_terminal()
        info = {
            "legal_actions": legal_actions(self.state, self.player_id) if not done else [],
            "state_return": new_return,
            "state_reward_raw": state_reward(self.state, self.player_id),
            "board": parse_board_numbers(self.state),
            "state_text": str(self.state),
        }
        return next_obs, float(reward), done, info

    def legal_actions(self):
        if self.state is None or self.state.is_terminal(): return []
        return legal_actions(self.state, self.player_id)

    def render(self):
        if self.state is None: print("<env not reset>")
        else: print(self.state)
