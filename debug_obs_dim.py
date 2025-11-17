#!/usr/bin/env python3

from envs.gym_nav_env import GymNavEnv
import numpy as np

def test_observation_dimensions():
    # Test with the same parameters as in training
    NUM_RAYS = 60
    NUM_PEOPLE = 20
    
    env = GymNavEnv(
        render_mode=None,
        num_rays=NUM_RAYS,
        num_people=NUM_PEOPLE,
    )
    
    print(f"Expected observation space: {env.observation_space}")
    print(f"Expected shape: {env.observation_space.shape}")
    print(f"Expected obs_dim: {env.observation_space.shape[0]}")
    
    # Test reset
    obs, info = env.reset()
    print(f"Actual observation shape after reset: {obs.shape}")
    print(f"Actual observation length: {len(obs)}")
    
    # Test a few steps
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1} - Observation shape: {obs.shape}, length: {len(obs)}")
        
        if terminated or truncated:
            obs, info = env.reset()
            print(f"Reset after step {i+1} - Observation shape: {obs.shape}, length: {len(obs)}")
    
    env.close()

if __name__ == "__main__":
    test_observation_dimensions()
