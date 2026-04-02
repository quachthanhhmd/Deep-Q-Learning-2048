#!/bin/bash
echo "Starting DDQN Experiment..."
python ../train.py --experiment DDQN --config ../config.json
