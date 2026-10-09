# ACT Simulation Baseline

## Date
2026-10-05

## Purpose
Verify the complete imitation-learning simulation pipeline before moving to real robot data collection.

## Environment
- WSL2
- Ubuntu 26.04.1 LTS
- Python 3.12
- LeRobot 0.6.1
- PyTorch 2.11 CPU
- MuJoCo / Gymnasium Robotics
- Policy: ACT

## Pipeline Verified
1. Scripted expert demonstration collection
2. Raw demonstration storage
3. Conversion to LeRobotDataset
4. ACT training
5. Checkpoint saving/loading
6. Policy evaluation in simulation
7. Interactive `play.py` interface

## Dataset
- Task: FetchPickAndPlace-v4
- Demonstrations: 20
- Total frames: 1000
- Collection source: scripted expert
- LeRobot dataset:
  `outputs/datasets/fetch_pick_place`

## ACT Configuration
- Parameters: ~51.6M
- Batch size: 8
- Chunk size: 100
- Observation steps: 1
- Vision backbone: ResNet18
- Training device: CPU

## Training Runs

### 20,000-Step Evaluation
- Episodes: 20
- Successes: X/20
- Success rate: Y%
- Evaluation video stored locally at `outputs/eval_act_20000.mp4`
- Main observed failures: TBD

### Smoke Test
- Steps: 50
- Result: training completed successfully
- Evaluation: 0/3 success
- Purpose: pipeline validation only

### Main Baseline
- Steps: 20,000
- Training completed successfully

## Current Status
The full simulation imitation-learning pipeline is operational.

## Next Steps
- Evaluate the 20,000-step ACT checkpoint
- Measure success rate over standardized trials
- Test generalization by changing object/goal starting conditions
- Inspect LeRobot observation/action data in detail
- Add keyboard teleoperation for human demonstration collection
- Transition the same workflow to physical leader/follower teleoperation
