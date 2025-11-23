# run_ppo.py

import time

from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC

from envs.gym_nav_env import GymNavEnv
from train_ppo import NUM_RAYS   

def main():
    env = GymNavEnv(render_mode="human", num_rays=NUM_RAYS, num_people=50)

    #model = PPO.load("ppo_gym_nav_env2", env=env)

    obs, info = env.reset()
    done = False
    terminated = False
    truncated = False

    while True:
        #action, _ = model.predict(obs, deterministic=True)
        action = [0.5, 0.0]  # Constant action for testing
        print(f"Action taken: v={action[0]:.2f}, w={action[1]:.2f}")
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Episode ended:", info.get("termination_reason"))
            time.sleep(1.0)
            obs, info = env.reset()
            terminated = truncated = False

    env.close()


if __name__ == "__main__":
    main()
