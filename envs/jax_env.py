from dataclasses import dataclass
import jax
import jax.numpy as jnp

@dataclass # python decorator, automatically creates a simple data container class, with "just" attributes, no methods
class EnvConfig:
    dt: float
    room_width: float
    room_height: float
    max_lin_vel: float
    max_ang_vel: float

@dataclass
class EnvState:
    step: jnp.int32
    x: jnp.float32
    y: jnp.float32
    theta: jnp.float32

def reset(key, cfg):
    """
    Resets the environment to an initial state.
    For now:
    - robot is places at the center of the room
    - heading is 0 radians
    - step counter is 0
    - observation is [x, y, theta]
    """
    x0 = cfg.room_width / 2.0
    y0 = cfg.room_height / 2.0

    state = EnvState(
        step = jnp.array(0, dtype=jnp.int32),   
        x = jnp.array(x0),
        y = jnp.array(y0),
        theta = jnp.array(0.0, dtype=jnp.float32),
    )
    obs = jnp.array([state.x, state.y, state.theta], dtype=jnp.float32) 
    return state, obs, key

def step(state: EnvState, action: jnp.ndarray, cfg: EnvConfig):
    """
    One simulation step.

    action: jnp.array([v, w])
        v: linear velocity
        w: angular velocity

    - simple unicycle model
    """
    v_cmd = action[0]
    w_cmd = action[1]

    v = jnp.clip(v_cmd, -cfg.max_lin_vel, cfg.max_lin_vel)
    w = jnp.clip(w_cmd, -cfg.max_ang_vel, cfg.max_ang_vel)

    dx = v * jnp.cos(state.theta) * cfg.dt
    dy = v * jnp.sin(state.theta) * cfg.dt
    dtheta = w * cfg.dt

    x_new = state.x + dx
    y_new = state.y + dy
    theta_new = state.theta + dtheta

    # Build new EnvState
    new_state = EnvState(
        step = state.step + jnp.int32(1),
        x
    )


def render(state, cfg):
    pass