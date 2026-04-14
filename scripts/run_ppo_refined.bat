@echo off
rem run_ppo_refined.bat
echo Running PPO Refined Experiment (PPO + SplitCNN + CornerBonus)...
python train.py --experiment ppo_refined --config config.json
