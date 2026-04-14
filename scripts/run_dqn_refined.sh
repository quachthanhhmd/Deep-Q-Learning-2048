#!/bin/bash
# run_dqn_refined.sh
echo "Running DQN Refined Experiment (D3QN + SplitCNN + CornerBonus)..."
python train.py --experiment dqn_refined --config config.json
