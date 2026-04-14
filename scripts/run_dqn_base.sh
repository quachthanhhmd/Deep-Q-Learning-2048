#!/bin/bash
# run_dqn_base.sh
echo "Running DQN Base Experiment (DQN + MLP)..."
python train.py --experiment dqn_base --config config.json
