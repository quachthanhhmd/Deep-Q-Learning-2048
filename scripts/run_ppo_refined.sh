#!/bin/bash
# run_ppo_refined.sh
echo "Running PPO Refined Experiment (PPO + SplitCNN + CornerBonus)..."
python train.py --experiment ppo_refined --config config.json
