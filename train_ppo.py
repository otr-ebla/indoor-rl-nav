import gymnasium as gym
from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC
from envs.gym_nav_env import GymNavEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import tensorboard

NUM_RAYS = 60
NUM_PEOPLE = 20
N_ENVS = 100

def make_env(rank: int):
    def _init():
        env = GymNavEnv(
            render_mode=None,
            num_rays=NUM_RAYS,
            num_people=NUM_PEOPLE,
        )
        env = Monitor(env)
        return env
    return _init

def main():
    use_subproc = True

    if use_subproc:
        env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    else:
        env = DummyVecEnv([make_env(i) for i in range(N_ENVS)])


    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        verbose=1,
        tensorboard_log="./tensorboard_logs/ppo_nav",  # <-- add this
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
    )

    total_timesteps = 15000000
    print(f"Training PPO with {N_ENVS} parallel envs, total_timesteps={total_timesteps}.")

    model.learn(total_timesteps=total_timesteps,
                tb_log_name="15M_025v_002rewdist",
                )

    model.save("ppo_gym_nav_env2")

    print("Training completed and model saved.")

if __name__ == "__main__":
    main()