# Indoor RL Nav: 2D LiDAR Navigation with JAX

![Indoor Environment](./indoor_gif.gif)

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![JAX](https://img.shields.io/badge/JAX-Accelerated-blueviolet)
![Status](https://img.shields.io/badge/Status-Active-success)

## Overview

**Indoor RL Nav** is a high-performance 2D training environment and simulation framework designed for mobile robots equipped with 2D LiDAR sensors. 

The primary goal of this repository is to train Reinforcement Learning agents to navigate complex indoor scenarios. Unlike standard static mazes, this environment challenges the agent to handle **dynamic environments** containing both static geometry (walls) and moving obstacles (simulating pedestrians).

The framework leverages **JAX** to accelerate environment stepping and training, supporting state-of-the-art algorithms including **PPO** and **DreamerV3**.

## Key Features

* **2D LiDAR Perception:** Agents operate using sparse 2D ray-cast data, simulating real-world laser scanners found on platforms like Turtlebots.
* **Dynamic Obstacle Avoidance:** The environment includes logic for moving entities, forcing the agent to learn predictive navigation and collision avoidance.
* **JAX Acceleration:** Custom environment wrappers allow for massive parallelization on GPU/TPU, significantly reducing training times compared to standard CPU-bound Gym environments.
* **Algorithm Support:** Includes implementations for:
    * Proximal Policy Optimization (PPO)
    * DreamerV3 (Model-Based RL)

## Repository Structure

```text
.
├── dreamerv3_lidar.py    # DreamerV3 implementation adapted for LiDAR
├── train_ppo.py          # Main training loop for PPO
├── run_ppo.py            # Inference script to test trained PPO agents
├── rl_trainer.py         # Base trainer utilities
├── visualMain.py         # Real-time visualization tool
├── main.py               # General entry point
├── envs/                 # Environment definitions
│   ├── gym_nav_env.py    # Core gymnasium environment
│   ├── jax_env.py        # JAX-optimized wrapper
│   ├── simple_env.py     # Simplified debugging environment
│   └── bouncing.py       # Logic for dynamic obstacles
├── *.msgpack             # Pre-trained model weights
└── requirements.txt      # Project dependencies
