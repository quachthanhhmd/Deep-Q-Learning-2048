@echo off
rem run_dqn_base.bat
echo Running DQN Base Experiment (DQN + MLP)...
python train.py --experiment dqn_base --config config.json
