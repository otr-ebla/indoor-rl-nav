import gymnasium as gym
from gymnasium import spaces
import numpy as np

from envs.simple_env import Simple2DEnv

MAX_LIN_VEL = 1.0  # m/s
MAX_ANG_VEL = 3.0  # rad/s


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
            max_steps=400,
            num_people=num_people,
        )
        self.render_mode = render_mode

        self.num_rays = num_rays
        self.num_people = num_people

        self.stack_dim = 5
        # Pre-allocate a fixed-size flattened lidar buffer (num_rays * stack_dim,)
        obs_dim = 2 + self.num_rays*self.stack_dim

        self.lidar_stack = None

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([0.0, -MAX_ANG_VEL]), 
            high=np.array([MAX_LIN_VEL, MAX_ANG_VEL]), 
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

        if self.stack_dim > 1:
            if self.lidar_stack is None:
                self.lidar_stack = [lidar_norm.copy() for _ in range(self.stack_dim)]
            else:
                self.lidar_stack.pop(0)
                self.lidar_stack.append(lidar_norm.copy())
            lidar_stack = np.concatenate(self.lidar_stack, axis=0)
        else:
            lidar_stack = lidar_norm.astype(np.float32)

        obs = np.concatenate([
            [rho, wrapped_angle / np.pi],
            lidar_stack,
        ]).astype(np.float32)

        return obs
    
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        _ = self.env.reset()

        # Reset the lidar buffer to ensure consistent dimensions
        if self.stack_dim > 1:
            lidar = self.env._compute_lidar()
            lidar = np.array(lidar, dtype=np.float32)
            lidar_norm = np.clip(lidar / self.env.max_lidar_distance, 0.0, 1.0)

            self.lidar_stack = [lidar_norm.copy() for _ in range(self.stack_dim)]   

        obs = self._get_obs()
        info = {}
        return obs, info
    
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        v = float(np.clip(action[0], 0.0, MAX_LIN_VEL))
        w = float(np.clip(action[1], -MAX_ANG_VEL, MAX_ANG_VEL))

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



 