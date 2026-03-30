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
python train.py --reward-transformation identity

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
