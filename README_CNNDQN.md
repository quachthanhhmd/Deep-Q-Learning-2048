# CNN DQN - 2048 Optimization

This document summarizes the recent architectural and configuration improvements applied to the Convolutional Neural Network Deep Q-Network (CNN DQN) agent for the game 2048. 

The modifications were executed to tackle premature convergence (getting stuck around the 128/256 tile mark) and gradient noise, unlocking the model's potential to discover long-term strategies.

## Overall Architecture Flow

```mermaid
flowchart TD
    A[2048 Game Board 4x4] -->|Extract State| B[One-Hot Representation\nShape: 16x4x4]
    B -->|7 Parallel Branches| C(Multi-Branch Convolutions\n1x2, 2x1, 1x3, 3x1, 1x4, 4x1, 2x2)
    C -->|Concatenated Feature Maps| D(Flatten to 3648 features)
    D --> E(Dense / FC Layers)
    E --> F[Q-Values\nUp, Down, Left, Right]
    F -->|Epsilon-Greedy Action| A
    A -->|State, Reward, Done| G[(Replay Buffer)]
    G -.->|Sample Batch 128| H[Q-Network Update & Target Sync]
```

## Summary of Changes

### 1. Multi-Branch Convolution (Tuple Network) (`models/cnn_q_network.py`)
- **Problem**: The original model used standard `3x3` and `2x2` convolutions, which inherently struggle to capture the row/col sliding mechanics of 2048 since tiles can merge from across the entire board. This limited the logical comprehension of the network, causing a convergence barrier around the 256.0 Max Tile mark.
- **Solution**: 
  - Overhauled the core structure to employ **7 parallel convolution branches** specifically mirroring 2048 tuple geometries.
  - Implemented asymmetrical kernels: `1x2`, `2x1`, `1x3`, `3x1`, `1x4` (full row), `4x1` (full col), and `2x2`.
  - Feature maps from all 7 distinct branches are concatenated into a massive `3648` dimensional representation before passing into the Dense layers, granting the model robust spatial awareness.

### 2. Reduced Noise in Reward Shaping (`envs/cnn_env.py`)
- **Problem**: A dense shaping reward of `empty_spots * 1.0` heavily distracted the agent, prioritizing keeping tiles empty over creating larger values (which is the actual path to higher scores). Additionally, `np.log2(board)` sometimes triggered `RuntimeWarning: divide by zero` due to evaluating `0` values prematurely.
- **Solution**: 
  - Diminished the scalar multiplier for empty spots down to `0.1`, forcing the agent to rely more on the foundational game score reward and the monotonic layout penalty.
  - Implemented a `np.maximum(board, 1)` guard inside the log calculation to sanitize values and suppress runtime warnings without affecting logic.

### 3. Extended Phase of Exploration (`config.json`)
- **Problem**: Over the course of testing, `eps_decay_steps` was set to `200,000`. In a 15,000 episode training run, this meant the agent stopped exploring and strictly adhered to a highly exploitative policy in less than ~15% of the timeframe, causing immediate local optima trapping.
- **Solution**: 
  - Modified `eps_decay_steps` up to `2,000,000`. The extensive epsilon decay trajectory allows the model to consistently encounter and experience deeper, late-game board states before exploiting the learned policy.
  
## Usage Notice
* If analyzing loss curves: Note that DQN loss is naturally extremely noisy due to target-network synchronization and Replay Buffer randomized sampling. Focus evaluation metrics primarily on the *Mean Return*, *Mean Max Tile*, and *Illegal Avoidance* graphs.
* When doing rapid testing (e.g. debugging via `num_episodes=200`), remember to temporarily scale down `eps_decay_steps` accordingly (e.g., to `20000`) so that you can effectively observe the shift from exploration to exploitation!
