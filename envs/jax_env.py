from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np  
import matplotlib.pyplot as plt
import time

from typing import NamedTuple

class StaticConfig(NamedTuple):
    dt: float
    room_width: float
    room_height: float
    max_lin_vel: float
    max_ang_vel: float
    robot_radius: float
    num_rays: int 
    max_lidar_distance: float
    num_people: int
    people_radius: float

class Obstacles(NamedTuple):
    centers: jnp.ndarray   # (N, 2)
    radii:   jnp.ndarray   # (N,)


class EnvState(NamedTuple):
    step: jnp.int32
    x: jnp.float32
    y: jnp.float32
    theta: jnp.float32
    people_positions: jnp.ndarray  # (num_people, 2)
    people_vel: jnp.ndarray

def reset(cfg: StaticConfig, obstacles: Obstacles):
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

    base_people_pos = jnp.array([
        [3.0, 5.0],   # left of robot
        [7.0, 5.0],   # right
        [5.0, 7.5],   # above
    ], dtype=jnp.float32)


    if cfg.num_people <= base_people_pos.shape[0]:
        people_pos = base_people_pos[:cfg.num_people]
    else:
        reps = cfg.num_people // base_people_pos.shape[0] + 1
        people_pos = jnp.tile(base_people_pos, (reps, 1))[:cfg.num_people]

    base_people_vel = jnp.array([
        [0.5, 0.0],
        [-0.5, 0.0],
        [0.0, 0.5],
    ], dtype=jnp.float32)   

    if cfg.num_people <= base_people_vel.shape[0]: # if enough base velocities are available
        people_vel = base_people_vel[:cfg.num_people] # assign them directly
    else: # if not enough, tile the base velocities
        reps = cfg.num_people // base_people_vel.shape[0] + 1 # how many times to repeat
        people_vel = jnp.tile(base_people_vel, (reps, 1))[:cfg.num_people]

    state = EnvState(
        step = jnp.array(0, dtype=jnp.int32),   
        x = jnp.array(x0),
        y = jnp.array(y0),
        theta = jnp.array(0.0, dtype=jnp.float32),
        people_positions = people_pos,
        people_vel = people_vel,
    )

    #lidar = lidar_scan(state, cfg)  
    lidar = lidar_scan_jit(state, cfg, obstacles)

    obs = jnp.concatenate([
        jnp.array([state.x, state.y, state.theta], dtype=jnp.float32), lidar], axis=0)
    return state, obs

def _min_gap_to_circles(
        px: jnp.ndarray,
        py: jnp.ndarray,
        centers: jnp.ndarray,
        radii: jnp.ndarray,
        robot_radius: float,
    ) -> jnp.ndarray:
    """
    Minimum distance gap between the robot (circle of radius robot_radius)
    and a set of circles defined by centers and radii.

    gap = distance(center_robot, center_obst) - (robot_radius + obst_radius)

    Returns +inf if there are no circles.
    """
    if radii.size == 0:
        return jnp.array(jnp.inf, dtype=jnp.float32)
    
    cx = centers[:, 0]
    cy = centers[:, 1]
    r  = radii

    dx = cx - px
    dy = cy - py

    dist = jnp.sqrt(dx * dx + dy * dy)
    combined_r = robot_radius + r
    gap = dist - combined_r
    return jnp.min(gap)




def step(state: EnvState, action: jnp.ndarray, cfg: StaticConfig, obstacles: Obstacles):
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

    # ---- Keep the robot inside the room ----
    x_new = jnp.clip(x_new, 0.0, cfg.room_width)
    y_new = jnp.clip(y_new, 0.0, cfg.room_height)

    # ---- Update people positions with simple wall bouncing ----

    people_pos_prop = state.people_positions + state.people_vel * cfg.dt

    px = people_pos_prop[:, 0]
    py = people_pos_prop[:, 1]
    vx = state.people_vel[:, 0]
    vy = state.people_vel[:, 1]

    # Bounce on vertical walls
    hit_left = px < 0.0
    hit_right = px > cfg.room_width
    vx_new = jnp.where(hit_left | hit_right, -vx, vx)
    px_clamped = jnp.clip(px, 0.0, cfg.room_width)

    # Bounce on horizontal walls
    hit_bottom = py < 0.0
    hit_top = py > cfg.room_height
    vy_new = jnp.where(hit_bottom | hit_top, -vy, vy)
    py_clamped = jnp.clip(py, 0.0, cfg.room_height)

    people_pos_new = jnp.stack([px_clamped, py_clamped], axis=1) 
    people_vel_new = jnp.stack([vx_new, vy_new], axis=1)

    # Build new EnvState
    new_state = EnvState(
        step = state.step + jnp.int32(1),
        x = x_new,
        y = y_new,
        theta = theta_new,
        people_positions = people_pos_new,
        people_vel = people_vel_new,
    )

    #lidar = lidar_scan(new_state, cfg)
    lidar = lidar_scan_jit(new_state, cfg, obstacles)

    obs = jnp.concatenate([
        jnp.array([new_state.x, new_state.y, new_state.theta], dtype=jnp.float32),
        lidar,
        ], 
    axis=0)

    # ----- Collision checks with obstacles and people ----
    gap_obst = _min_gap_to_circles(
        new_state.x,
        new_state.y,
        obstacles.centers,
        obstacles.radii,
        cfg.robot_radius,
    )

    if cfg.num_people > 0:
        people_radii = jnp.full((cfg.num_people,), cfg.people_radius, dtype=jnp.float32)
        gap_people = _min_gap_to_circles(
            new_state.x,
            new_state.y,
            new_state.people_positions,
            people_radii,
            cfg.robot_radius,
        )
    else:
        gap_people = jnp.array(jnp.inf, dtype=jnp.float32)

    min_gap = jnp.minimum(gap_obst, gap_people)
    collision = min_gap < 0.0

    # ---- reward and done usinf LiDAR + collision ----
    min_dist = jnp.min(lidar)
    base_reward = (min_dist/cfg.max_lidar_distance).item() # convert to python float

    collision_penalty = jnp.where(collision, -1.0, 0.0).item() # convert to python float

    reward = float(base_reward + collision_penalty)

    safety_margin = 0.2
    done = bool((min_dist < safety_margin) | collision) # convert to python bool
    if done:
        print("Collision detected! min_dist =", float(min_dist))

    return new_state, obs, reward, done 

def _ray_distace_to_walls(x0: jnp.ndarray,
                          y0: jnp.ndarray,
                          angle: jnp.ndarray,
                          cfg: StaticConfig) -> jnp.ndarray:
    """
    Distance from (x0, y0) along ray at "angle" to the walls of the room.
    
    Ray direction is (dx, dy) = (cos(angle), sin(angle))
    We compute insersections with each of the 4 walls, and take the minimum positive distance.
    If none are valid, return max_lidar_distance.
    """
    dx = jnp.cos(angle)
    dy = jnp.sin(angle)

    eps = 1e-6

    t_x0 = (0.0 - x0) / (dx + eps)
    y_x0 = y0 + t_x0 * dy
    valid_x0 = (t_x0 > 0.0) & (y_x0 >= 0.0) & (y_x0 <= cfg.room_height)
    dist_x0 = jnp.where(valid_x0, t_x0, jnp.inf)

    t_xW = (cfg.room_width - x0) / (dx + eps)
    y_xW = y0 + t_xW * dy
    valid_xW = (t_xW > 0.0) & (y_xW >= 0.0) & (y_xW <= cfg.room_height)
    dist_xW = jnp.where(valid_xW, t_xW, jnp.inf)

    t_y0 = (0.0 - y0) / (dy + eps)
    x_y0 = x0 + t_y0 * dx
    valid_y0 = (t_y0 > 0.0) & (x_y0 >= 0.0) & (x_y0 <= cfg.room_width)
    dist_y0 = jnp.where(valid_y0, t_y0, jnp.inf)

    t_yH = (cfg.room_height - y0) / (dy + eps)
    x_yH = x0 + t_yH * dx
    valid_yH = (t_yH > 0.0) & (x_yH >= 0.0) & (x_yH <= cfg.room_width)
    dist_yH = jnp.where(valid_yH, t_yH, jnp.inf)

    dist = jnp.minimum(jnp.minimum(dist_x0, dist_xW), jnp.minimum(dist_y0, dist_yH))

    return jnp.minimum(dist, cfg.max_lidar_distance)

def _ray_distance_to_circles(
    x0: jnp.ndarray,
    y0: jnp.ndarray,
    angle: jnp.ndarray,
    obstacles: Obstacles,
    max_range: float,
) -> jnp.ndarray:
    """
    Distance from (x0, y0) along ray 'angle' to the nearest circular obstacle.
    Returns max_range if there is no valid intersection.
    """
    # If no obstacles at all, just return max_range
    if obstacles.radii.size == 0:
        return jnp.array(max_range, dtype=jnp.float32)

    dx = jnp.cos(angle)
    dy = jnp.sin(angle)

    # Ray origin
    px = x0
    py = y0

    # Circle centers and radii (vectorized over all obstacles)
    centers = obstacles.centers   # shape (N, 2)
    radii   = obstacles.radii     # shape (N,)

    cx = centers[:, 0]
    cy = centers[:, 1]
    r  = radii

    # Vector from circle center to ray origin: m = p0 - c
    mx = px - cx
    my = py - cy

    # Quadratic coefficients: t^2 + b t + c2 = 0  (since |d|=1)
    b  = 2.0 * (dx * mx + dy * my)
    c2 = mx * mx + my * my - r * r

    disc = b * b - 4.0 * c2  # discriminant

    valid_discriminant = disc >= 0.0
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))

    # Two possible intersection parameters
    t1 = (-b - sqrt_disc) / 2.0
    t2 = (-b + sqrt_disc) / 2.0

    t1_valid = valid_discriminant & (t1 > 0.0)
    t2_valid = valid_discriminant & (t2 > 0.0)

    t1_val = jnp.where(t1_valid, t1, max_range)
    t2_val = jnp.where(t2_valid, t2, max_range)

    t_candidates = jnp.minimum(t1_val, t2_val)

    # Minimum over all obstacles
    min_t = jnp.min(t_candidates)

    return jnp.minimum(min_t, max_range).astype(jnp.float32)

def _ray_distance_to_walls_and_circles(
    x0: jnp.ndarray,
    y0: jnp.ndarray,
    angle: jnp.ndarray,
    cfg: StaticConfig,
    obstacles: Obstacles,
    people_pos: jnp.ndarray,
) -> jnp.ndarray:
    d_walss = _ray_distace_to_walls(x0, y0, angle, cfg)
    d_circles = _ray_distance_to_circles(x0, y0, angle, obstacles, cfg.max_lidar_distance)

    if cfg.num_people > 0:
        people_obstacles = Obstacles(
            centers=people_pos,
            radii=jnp.full((cfg.num_people,), cfg.people_radius, dtype=jnp.float32),
        )
        d_people = _ray_distance_to_circles(x0, y0, angle, people_obstacles, cfg.max_lidar_distance)
    else: 
        d_people = jnp.array(cfg.max_lidar_distance, dtype=jnp.float32)

    return jnp.minimum(jnp.minimum(d_walss, d_circles), d_people)


def lidar_scan(
        state: EnvState, 
        cfg: StaticConfig,
        obstacles: Obstacles
) -> jnp.ndarray:
    """
    Real Lidar to walls
    - num rays beams around
    - ray 0 aligned with robot heading
    """

    base_angles = jnp.linspace(0.0, 2.0 * jnp.pi, cfg.num_rays, endpoint=False)
    ray_angles = (state.theta + base_angles) 

    dist_fn = jax.vmap(
        _ray_distance_to_walls_and_circles, 
        in_axes=(None, None, 0, None, None, None),
    ) # 
    distances = dist_fn(
        state.x, state.y, 
        ray_angles, 
        cfg,
        obstacles,
        state.people_positions,
    )

    return distances.astype(jnp.float32)    

lidar_scan_jit = jax.jit(lidar_scan, static_argnums=(1,))













def render(state: EnvState, cfg: StaticConfig, ax=None, obstacles: Obstacles=None):
    """
    Minimal 2D rendering:
    - draw room as a rectangle
    - draw robot as a point (or small circle)
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
        created_fig = True

    ax.clear()

    ax.set_xlim(0, cfg.room_width)
    ax.set_ylim(0, cfg.room_height)
    ax.set_aspect('equal', 'box')

    ax.set_title(f"Step: {int(state.step)}, pos=({state.x:.2f}, {state.y:.2f}), theta={state.theta:.2f} rad")

    room_rect = plt.Rectangle(
        (0,0), 
        cfg.room_width, 
        cfg.room_height, 
        fill=None, 
        linewidth=2,
        )
    
    ax.add_patch(room_rect)

    robot = plt.Circle(
        (float(state.x), float(state.y)),
        radius=cfg.robot_radius,
        color='blue',
    )
    ax.add_patch(robot)

    arrow_len = 0.4
    ax.arrow(
        float(state.x),
        float(state.y),
        arrow_len * float(jnp.cos(state.theta)),
        arrow_len * float(jnp.sin(state.theta)),
        head_width=0.15,
        head_length=0.3,
        length_includes_head=True,
        color="blue",
    )

    # --- draw circular obstacles ---
    for (cx, cy), r in zip(np.array(obstacles.centers), np.array(obstacles.radii)):
        circle = plt.Circle(
            (float(cx), float(cy)),
            radius=float(r),
            color='gray',
        )
        ax.add_patch(circle)

    # --- draw people as circles ---
    if state.people_positions is not None:
        for px, py in np.array(state.people_positions):
            person = plt.Circle(
                (float(px), float(py)),
                radius=cfg.people_radius,
                color='green',
            )
            ax.add_patch(person)
    
    # ---- NEW: draw lidar rays ----
    #lidar = lidar_scan(state, cfg)
    lidar = lidar_scan_jit(state, static_cfg, obstacles)

    base_angles = jnp.linspace(0.0, 2.0 * jnp.pi, static_cfg.num_rays, endpoint=False)
    ray_angles = (state.theta + base_angles)

    x0 = float(state.x)
    y0 = float(state.y)

    for d, ang in zip(np.array(lidar), np.array(ray_angles)):
        x1 = x0 + float(d) * float(jnp.cos(ang))
        y1 = y0 + float(d) * float(jnp.sin(ang))
        ax.plot([x0, x1], [y0, y1], color='red', linewidth=0.5, alpha=0.5)

    plt.pause(0.001)

    if created_fig:
        return ax
    




























if __name__ == "__main__":
    # 1) Create config
    obst_centers = jnp.array([
        [3.0, 4.0],
        [7.0, 6.0],
    ], dtype=jnp.float32)

    obst_radii = jnp.array([0.7, 1.0], dtype=jnp.float32)

    static_cfg = StaticConfig(
        dt=0.01,
        room_width=10.0,
        room_height=10.0,
        max_lin_vel=1.0,
        max_ang_vel=jnp.pi,
        robot_radius=0.2,
        num_rays=36,
        max_lidar_distance=10.0,
        num_people=3,
        people_radius=0.2,
    )

    obstacles = Obstacles(
        centers=obst_centers,
        radii=obst_radii,
    )

    # 2) Reset env
    state, obs = reset(static_cfg, obstacles)
    print("Initial state:", state)
    print("Initial obs:", obs)

    action = jnp.array([0.5, 0.3], dtype=jnp.float32)

    # 4) Create figure and get axes
    fig, ax = plt.subplots(figsize=(5, 5))

    # 5) Main loop
    for t in range(10000):
        t0 = time.time()

        # Step the environment
        state, obs, reward, done = step(state, action, static_cfg, obstacles)

        print(f"Step={int(state.step)}, reward={reward:.3f}, done={done}")

        # Render current state
        render(state, static_cfg, ax=ax, obstacles=obstacles)

        # Enforce near-real-time: each step takes at least cfg.dt seconds
        elapsed = time.time() - t0
        if elapsed < static_cfg.dt:
            time.sleep(static_cfg.dt - elapsed)

        if done:
            print("Episode terminated at step", int(state.step))
            break

    print("Simulation finished.")
    plt.show()
