# Metrics Documentation (`metrics.csv`)

This document explains all variables logged during training and how they relate
to the project’s goal: understanding how variance affects policy gradient
optimization.

---

Each metric corresponds to one part of this pipeline.

---

# Bookkeeping Metrics

## `update`
- **What:** PPO update index (0 → N)
- **Meaning:** Training iteration
- **Use:** Primary x-axis for plots

---

## `global_step`
- **What:** Total number of environment steps taken
- **Meaning:** True measure of training progress
- **Relation:** `global_step = update × rollout_steps`
- **Use:** Alternative x-axis (common in RL papers)

---

# Performance Metrics

## `episode_return_mean`
- **What:** Mean return over recent episodes
- **Meaning:** Overall agent performance
- **Interpretation:**
  - Increasing → learning
  - Flat → stagnation
- **Use:** Main learning curve

---

## `episode_length_mean`
- **What:** Average episode length
- **Meaning:** How long the agent survives or interacts
- **Interpretation:**
  - Task-dependent (e.g., longer is better in CartPole)
- **Use:** Sanity check for learning behavior

---

# Reward Statistics

## `raw_reward_mean`
- **What:** Mean of rewards from the environment
- **Meaning:** Baseline reward signal

---

## `raw_reward_var`
- **What:** Variance of raw rewards
- **Meaning:** Natural reward variability of the environment
- **Use:** Baseline comparison for transformations

---

## `transformed_reward_mean`
- **What:** Mean after reward transformation
- **Meaning:** Shifted/scaled reward signal
- **Note:** Mean has limited effect in PPO due to baseline subtraction

---

## `transformed_reward_var`
- **What:** Variance after transformation
- **Meaning:** **Controlled variable in reward-based experiments**

- **Use:**
  - Confirms transformation is working
  - Key variable in hypothesis testing

---

# Advantage Statistics 

## `advantage_mean`
- **What:** Mean of computed advantages
- **Meaning:** Bias of advantage estimates
- **Note:** Usually near zero after normalization

---

## `advantage_var`
- **What:** Variance of advantage values
- **Meaning:** **Strength of the policy gradient signal**


# Optimization / Gradient Metrics

## `grad_norm`
- **What:** Norm of gradients after backpropagation
- **Meaning:** Magnitude of parameter updates

### Interpretation:
- Very low → little learning (flat landscape)
- Moderate → healthy updates
- Very high → unstable learning

---

## `policy_loss`
- **What:** PPO objective loss
- **Meaning:** Policy update magnitude
- **Note:** Not directly interpretable alone

---

## `value_loss`
- **What:** Value function (critic) MSE loss
- **Meaning:** Accuracy of value estimates

### Interpretation:
- High → poor value estimates
- Low → accurate value predictions

---

## `entropy`
- **What:** Policy entropy
- **Meaning:** Randomness of policy actions

### Interpretation:
- High entropy → more exploration
- Low entropy → more deterministic policy

---

# Stability Metrics

## `approx_kl`
- **What:** Approximate KL divergence between old and new policy
- **Meaning:** Size of policy updates

### Interpretation:
- Low → stable updates
- High → aggressive updates (can destabilize)

---

## `clipfrac`
- **What:** Fraction of samples where PPO clipping was applied
- **Meaning:** Frequency of constrained updates

### Interpretation:
- High → updates too large (clipping active)
- Low → small updates

---
