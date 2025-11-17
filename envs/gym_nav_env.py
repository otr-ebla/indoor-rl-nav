import gymnasium as gym
from gymnasium import spaces
import numpy as np

from envs.simple_env import Simple2DEnv

class GymNavEnv(gym.Env):
    """
    Gymnasium-compatible wrapper around Simple2DEnv.
    Observation:
      - x, y (normalized to [-1, 1])
      - cos(theta), sin(theta)
      - goal_dx, goal_dy (relative goal, normalized to [-1, 1])
      - lidar distances normalized to [0, 1]
    Action:
      - [v, w] in [0, 1] x [-1, 1]
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        render_mode: str | None = None,
        num_rays: int = 32,
        num_people: int = 20,
    ):  
        super().__init__()
    
        # Create internal simulator
        self.env = Simple2DEnv(
            num_rays=num_rays,
            max_steps=1000,
            num_people=num_people,
        )
        self.render_mode = render_mode

        self.num_rays = num_rays
        self.num_people = num_people
        obs_dim = 2 + self.num_rays

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            dtype = np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        x = self.env.x
        y = self.env.y
        theta = self.env.theta

        goal_x = self.env.goal_x
        goal_y = self.env.goal_y

        goal_dx = goal_x - x
        goal_dy = goal_y - y

        rho = np.sqrt(goal_dx**2 + goal_dy**2)
        theta_goal = np.arctan2(goal_dy, goal_dx)
        angle_diff = theta_goal - theta
        wrapped_angle = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        lidar = self.env._compute_lidar()
        lidar = np.array(lidar, dtype=np.float32)
        lidar_norm = np.clip(lidar / self.env.max_lidar_distance, 0.0, 1.0)

        obs = np.concatenate([
            [rho, wrapped_angle / np.pi],
            lidar_norm,
        ]).astype(np.float32)

        return obs
    
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        _ = self.env.reset()
        obs = self._get_obs()
        info = {}
        return obs, info
    
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        v = float(np.clip(action[0], 0.0, 1.0))
        w = float(np.clip(action[1], -1.0, 1.0))

        _, reward, done, info = self.env.step((v, w))
        obs = self._get_obs()

        terminated = False
        truncated = done
        reason = info.get("terminated_reason", None)

        if done:
            if reason == "max_steps_reached":
                truncated = True
            else:
                terminated = True

        if self.render_mode == "human":
            self.env.render()

        return obs, reward, terminated, truncated, info 
    
    def render(self):
        if self.render_mode == "human":
            self.env.render()   

    def close(self):
        self.env.close()



 