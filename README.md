# Variance-Controlled Advantage Scaling PPO (VECTR-PPO)

### Overview

This project implements **Proximal Policy Optimization (PPO)** with multiple
methods for controlling variance in reinforcement learning training. The main
goal is to study how reward scaling, reward transformations, and advantage
variance control affect policy gradient optimization stability and learning
performance.

The implementation supports three versions of the algorithm for model training:

1. Baseline PPO
2. Reward Transformation PPO
3. VECTR PPO (with Variance-Controlled Advantage Scaling)

### Running the Algorithms

1. Baseline

```
python train.py --reward-transform identity

```

2. Reward Transformation

|Transform|Description|
|---|---|
|identity|No change (baseline model)|
|scale|Multiply rewards by a scalar constant|
|zscore|Normalize reward distribution|
|minmax|Scale rewards within [0,1]|
|tanh|Nonlinear reward compression using hyperbolic tangent|


```
python train.py --reward-transform zscore
python train.py --reward-transform minmax
python train.py --reward-transform tanh
python train.py --reward-transform scale --reward-scale 2.0
```

3. VECTR PPO (Variance-Controlled Advantage Scaling)
```
python train.py --use-vectr
```

## All Command Line Arguments
This implementation is set up with the intention to be run in large batches via bash scripts

| Argument            | Default     | Description                    |
| ------------------- | ----------- | ------------------------------ |
| --env-id            | CartPole-v1 | Gymnasium environment          |
| --seed              | 0           | Random seed                    |
| --total-updates     | 1000        | PPO updates                    |
| --rollout-steps     | 2048        | Steps per rollout              |
| --minibatch-size    | 64          | PPO minibatch size             |
| --update-epochs     | 10          | PPO epochs per update          |
| --lr                | 3e-4        | Learning rate                  |
| --gamma             | 0.99        | Discount factor                |
| --gae-lambda        | 0.95        | GAE lambda                     |
| --clip-coef         | 0.2         | PPO clip epsilon               |
| --reward-transform  | identity    | Reward transform               |
| --reward-scale      | 1.0         | Reward scale factor            |
| --reward-target-std | 1.0         | Target std for z-score         |
| --reward-tanh-gain  | 1.0         | Tanh transform gain            |
| --use-vectr         | False       | Enable VECTR advantage scaling |
| --vectr-target-std  | 1.0         | Target advantage std           |
| --device            | cpu         | cpu or cuda                    |
| --outdir            | runs/vectr  | Output directory               |



