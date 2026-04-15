# Deep Reinforcement Learning for 2048

This repository contains multiple Reinforcement Learning architectures and environments tailored to solve the game of 2048. We provide implementations to evaluate both standard Q-learning baselines and advanced policy-gradient models with sophisticated reward shaping.

## Prerequisites

Ensure you have your environment set up with PyTorch and standard scientific libraries:
```bash
pip install torch numpy pyspiel tqdm matplotlib
```

## Available Configurations
You can modify the training hyperparameters by supplying a JSON configuration file.
- **`config.json`**: Use this for standard, long-running training.
- **`debug_config.json`**: Contains heavily truncated episodes and buffer sizes for quickly testing code functionality.

---

## Running the Experiments

To run an experiment, use the `train.py` entry point specifying the `--experiment` flag.

### 1. DQN Base (`dqn_base`)
The baseline experiment. Uses a standard flattened observation and an MLP-based `DuelingQNetwork`. No spatial convolutions or advanced reward potentials are used—a standard Reinforcement Learning setting to establish baseline metrics.

**Command:**
```bash
python train.py --experiment dqn_base --config config.json
```

### 2. DQN Refined (`dqn_refined`)
An advanced Deep Q-Network. This architecture replaces the MLP baseline with a Spatial `RefinedCNNQNetwork`. It leverages One-Hot Encoding and Directional Convolutions to properly interpret 2D tile combinations. It utilizes Corner-Potential reward shaping.

**Command:**
```bash
python train.py --experiment dqn_refined --config config.json
```

### 3. PPO Refined (`ppo_refined`)
A Proximal Policy Optimization (PPO) implementation. This shares the `RefinedCNNEncoder` with the advanced DQN but substitutes Q-Learning for an Actor-Critic discrete policy gradient approach. Known for higher stability during training. Let this architecture explore the environment with configured Entropy regularization.

**Command:**
```bash
python train.py --experiment ppo_refined --config config.json
```

---

## Debugging
If you are iterating on the architecture or reward functions, you should test your changes rapidly using the debug config before committing to a full run.

```bash
# Example quick debug run
python train.py --experiment ppo_refined --config debug_config.json
```

## Result Checkpoints
When training completes or hits evaluation milestones, the system outputs `.pth` files. Use these model weights to perform deterministic inference / post-training evaluations!

## Kaggle Running Env
I have shared the kaggle script in the submission, please open this file to run the code in kaggle if you don't want to run locally. 