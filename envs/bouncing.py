import os

# === FORZA JAX A CUDA PRIMA DI IMPORTARE JAX ===
os.environ["JAX_PLATFORMS"] = "cuda"
os.environ["JAX_PLATFORM_NAME"] = "cuda"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/home/LABAUT/alberto_vaglio/cuda12-local"

import jax
import jax.numpy as jnp
from jax import lax

@jax.jit    
def bounce_vs_circles(people_pos, people_vel,
                      circle_centers, circle_radii,
                      people_radius, restitution=0.0):
    """
    people_pos:   (P, 2)
    people_vel:   (P, 2)
    circle_centers: (C, 2)
    circle_radii:   (C,)
    """
    if circle_centers is None or circle_centers.shape[0] == 0:
        return people_pos, people_vel

    # (P, C, 2)
    diff = people_pos[:, None, :] - circle_centers[None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1)              # (P, C)

    min_dist = people_radius + circle_radii[None, :]   # (P, C)
    penetration = jnp.maximum(0.0, min_dist - dist)    # (P, C)

    # Normali (P, C, 2)
    normal = diff / (dist[..., None] + 1e-8)

    # --- correzione di posizione ---
    # spostamento per ogni coppia (P, C)
    pos_corr_each = normal * penetration[..., None]    # (P, C, 2)

    # sommo tutte le correzioni da tutti i cerchi
    pos_corr_total = jnp.sum(pos_corr_each, axis=1)    # (P, 2)
    people_pos = people_pos + pos_corr_total

    # --- riflessione velocità ---
    # calcolo una normale media per persona: somma delle normali dove c'è penetrazione
    # collision_mask = penetration > 0.0                 # (P, C)
    # normal_sum = jnp.sum(normal * collision_mask[..., None], axis=1)  # (P, 2)

    # normal_norm = jnp.linalg.norm(normal_sum, axis=-1, keepdims=True) # (P, 1)
    # has_collision = normal_norm[..., 0] > 0.0

    # # normalizzata
    # n = normal_sum / (normal_norm + 1e-8)              # (P, 2)

    # # rifletto solo per chi ha collisione
    # proj = jnp.sum(people_vel * n, axis=-1, keepdims=True)  # (P, 1)
    # vel_reflected = people_vel - (1.0 + restitution) * proj * n

    # people_vel = jnp.where(has_collision[:, None], vel_reflected, people_vel)

    # persona in collisione con almeno un cerchio
    collided = jnp.any(penetration > 0.0, axis=1)  # (P,)

    # se hai toccato qualcosa, semplicemente inverti la velocità
    people_vel = jnp.where(collided[:, None], -people_vel, people_vel)

    return people_pos, people_vel

@jax.jit
def bounce_vs_rectangles(people_pos, people_vel,
                         rect_centers, rect_sizes,
                         people_radius, restitution=0.0):
    """
    rect_centers: (R, 2)
    rect_sizes:  (R, 2) full sizes (width, height)
    people_pos:  (P, 2)
    """
    if rect_centers is None or rect_centers.shape[0] == 0:
        return people_pos, people_vel

    half = rect_sizes / 2.0                              # (R, 2)

    # (P, R, 2)
    diff = people_pos[:, None, :] - rect_centers[None, :, :]

    # punto più vicino dentro il rettangolo
    clamped = jnp.clip(diff, -half[None, :, :], half[None, :, :])
    closest = rect_centers[None, :, :] + clamped        # (P, R, 2)

    # vettore dal closest al centro della persona
    d = people_pos[:, None, :] - closest                # (P, R, 2)
    dist = jnp.linalg.norm(d, axis=-1)                  # (P, R)

    penetration = jnp.maximum(0.0, people_radius - dist) # (P, R)

    # normali
    normal = d / (dist[..., None] + 1e-8)               # (P, R, 2)

    # --- correzione posizione ---
    pos_corr_each = normal * penetration[..., None]     # (P, R, 2)
    pos_corr_total = jnp.sum(pos_corr_each, axis=1)     # (P, 2)
    people_pos = people_pos + pos_corr_total

    # --- riflessione velocità ---
    collided = jnp.any(penetration > 0.0, axis=1)  # (P,)
    people_vel = jnp.where(collided[:, None], -people_vel, people_vel)

    return people_pos, people_vel


@jax.jit
def bounce_people_on_obstacles(people_pos, people_vel,
                               circles, rectangles,
                               cfg, restitution=0.0, max_iterations=4):
    """
    Applica più iterazioni di risoluzione delle collisioni
    per essere sicuro di eliminare la penetrazione.
    """

    circle_centers = None
    circle_radii = None
    if (circles is not None) and (circles.centers.size > 0):
        circle_centers = circles.centers  # (C, 2)
        circle_radii   = circles.radii    # (C,)


    rect_centers = None
    rect_sizes   = None
    if (rectangles is not None) and (rectangles.x_min.size > 0):
        xmin = rectangles.x_min
        ymin = rectangles.y_min
        xmax = rectangles.x_max
        ymax = rectangles.y_max

        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        rect_centers = jnp.stack([cx, cy], axis=-1)  #

        w = xmax - xmin
        h = ymax - ymin
        rect_sizes = jnp.stack([w, h], axis=-1)      #

    def body_fun(_, carry):
        p, v = carry

        # 1) rimbalzo contro cerchi
        if circle_centers is not None:
            p, v = bounce_vs_circles(p, v,
                                     circle_centers, circle_radii,
                                     cfg.people_radius, restitution)

        # 2) rimbalzo contro rettangoli
        if rect_centers is not None:
            p, v = bounce_vs_rectangles(p, v,
                                        rect_centers, rect_sizes,
                                        cfg.people_radius, restitution)

        # 3) rimbalzo tra persone
        p, v = bounce_vs_people(p, v,
                                cfg.people_radius, restitution)
        
        p, v = bounce_vs_walls(p, v,
                               cfg.people_radius,
                               cfg.room_width,
                               cfg.room_height,
                               restitution)

        return (p, v)

    # iterazioni (pochissime, perché Step 1 già riduce la penetrazione)
    p_final, v_final = jax.lax.fori_loop(
        0, max_iterations, body_fun, (people_pos, people_vel)
    )

    return p_final, v_final


@jax.jit
def bounce_vs_people(people_pos, people_vel, people_radius, restitution=0.0):
    """ Gestisce le collisioni e il rimbalzo tra le persone. """
    P = people_pos.shape[0]
    
    # Non eseguire se ci sono meno di 2 persone
    # (Usiamo jnp.where per il controllo condizionale in un contesto JIT)
    if P <= 1:
        return people_pos, people_vel

    # (P, P, 2) - Vettore differenza tra tutte le coppie (diff[i, j] = pos[i] - pos[j])
    diff = people_pos[:, None, :] - people_pos[None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1)  # (P, P)

    # Maschera per escludere P vs P (un oggetto non collide con se stesso)
    i = jnp.arange(P)
    mask = i[:, None] != i[None, :]

    # Distanza minima di non compenetrazione (2*r)
    min_dist = 2.0 * people_radius
    
    # Penetrazione: max(0, min_dist - dist). Mascherata per escludere P vs P.
    penetration_unmasked = jnp.maximum(0.0, min_dist - dist)
    penetration = jnp.where(mask, penetration_unmasked, 0.0) # (P, P)

    # Normali (P, P, 2)
    # n = diff / dist (vettore normalizzato dalla persona 'j' alla persona 'i')
    normal = diff / (dist[..., None] + 1e-8)

    # --- A) Correzione di Posizione ---
    # Dividiamo per 2 perché la penetrazione viene calcolata da entrambe le parti
    pos_corr_each = normal * penetration[..., None] / 2.0
    
    # Somma delle correzioni subite da 'i' da parte di tutti gli 'j'
    pos_corr_total = jnp.sum(pos_corr_each, axis=1)    # (P, 2)
    people_pos = people_pos + pos_corr_total

    # --- B) Riflessione Velocità (Inversione semplificata) ---
    # Una persona 'i' ha subito collisione se c'è penetrazione con almeno un 'j'
    collided = jnp.any(penetration > 0.0, axis=1)  # (P,)
    
    # Se hai toccato qualcosa, inverti la velocità
    people_vel = jnp.where(collided[:, None], -people_vel, people_vel)

    return people_pos, people_vel

# Aggiungi questa funzione nel tuo file bouncing.py

@jax.jit
def bounce_vs_walls(people_pos, people_vel, people_radius, room_width, room_height, restitution=0.0):
    """ Gestisce le collisioni e il rimbalzo delle persone contro le pareti. """
    
    # 1. Correzione Posizione (per rimuovere la penetrazione)
    
    # Calcola la penetrazione massima in ciascuna direzione
    pen_left = jnp.maximum(0.0, people_radius - people_pos[:, 0])
    pen_right = jnp.maximum(0.0, people_radius - (room_width - people_pos[:, 0]))
    pen_bottom = jnp.maximum(0.0, people_radius - people_pos[:, 1])
    pen_top = jnp.maximum(0.0, people_radius - (room_height - people_pos[:, 1]))
    
    # Aggiorna la posizione
    people_pos = people_pos.at[:, 0].add(pen_left - pen_right)
    people_pos = people_pos.at[:, 1].add(pen_bottom - pen_top)
    
    # 2. Riflessione Velocità
    
    # Maschera per le collisioni: True se c'è stata penetrazione (che ora è corretta)
    collided_x = (pen_left > 0.0) | (pen_right > 0.0)
    collided_y = (pen_bottom > 0.0) | (pen_top > 0.0)
    
    # Rifletti la componente X se c'è collisione orizzontale
    people_vel = people_vel.at[:, 0].set(
        jnp.where(collided_x, -people_vel[:, 0], people_vel[:, 0])
    )
    # Rifletti la componente Y se c'è collisione verticale
    people_vel = people_vel.at[:, 1].set(
        jnp.where(collided_y, -people_vel[:, 1], people_vel[:, 1])
    )
    
    return people_pos, people_vel

























# def sample_free_position(
#     rng_key: jnp.ndarray,
#     cfg: StaticConfig,
#     obstacles: Obstacles,
#     rect_obst: RectObstacles,
#     entity_radius: float,
#     margin: float = 0.2,
# ) -> jnp.ndarray:
#     """
#     Sample a random position (x, y) inside the room such that the entity
#     (a circle of radius `entity_radius`) does NOT intersect any circular obstacle.

#     Returns:
#       new_key, x, y
#     """
#     key = rng_key
#     room_margin = entity_radius + margin

#     while True:
#         key, subkey = jrandom.split(key)
#         x = jrandom.uniform(
#             subkey,
#             shape=(),
#             minval=room_margin,
#             maxval=cfg.room_width - room_margin,
#         )
#         key, subkey = jrandom.split(key)
#         y = jrandom.uniform(
#             subkey,
#             shape=(),
#             minval=room_margin,
#             maxval=cfg.room_height - room_margin,
#         )   

#         # ---- 1) Check circular obstacles ----
#         if obstacles.radii.size == 0:
#             safe_circles = True
#         else:
#             cx = obstacles.centers[:, 0]
#             cy = obstacles.centers[:, 1]
#             r = obstacles.radii

#             dx = cx - x
#             dy = cy - y
#             dist = jnp.sqrt(dx * dx + dy * dy)

#             combined_r = r + entity_radius 
#             gap = dist - combined_r

#             safe_circles = bool(jnp.all(gap >= 0.0))

#         # ---- 2) Check rectangular obstacles ----
#         if rect_obst is None or rect_obst.x_min.size == 0:
#             safe_rects = True
#         else:
#             xmin = rect_obst.x_min
#             ymin = rect_obst.y_min
#             xmax = rect_obst.x_max
#             ymax = rect_obst.y_max
#             closest_x = jnp.clip(x, xmin, xmax)
#             closest_y = jnp.clip(y, ymin, ymax)

#             dxr = closest_x - x
#             dyr = closest_y - y
#             dist_rect = jnp.sqrt(dxr * dxr + dyr * dyr)

#             safe_rects = bool(jnp.all(dist_rect >= entity_radius))

#         safe = safe_circles and safe_rects
#         if safe:
#             return key, x, y





# def sample_random_obstacles(
#         rng_key: jnp.ndarray,
#         cfg: StaticConfig,
# ):
#     """
#     Random circular obstacles:
#       - K ~ Uniform{min_circ_obstacles, ..., max_circ_obstacles}
#       - radius in [obst_min_radius, obst_max_radius]
#       - centers inside the room
#       - optional non-overlap via obst_clearance
#     """
#     key = rng_key
#     centers = []
#     radii = []

#     key, k_key = jrandom.split(key)
#     num_obst = int(jrandom.randint(
#         k_key,
#         shape=(),
#         minval=cfg.min_circ_obstacles,
#         maxval=cfg.max_circ_obstacles + 1,)
#     )

#     max_tries = 1000

#     for i in range(num_obst):
#         placed = False
#         tries = 0
#         while not placed and tries < max_tries:
#             tries += 1

#             key, subkey_r = jrandom.split(key)
#             r = jrandom.uniform(
#                 subkey_r,
#                 shape=(),
#                 minval=cfg.obst_min_radius,
#                 maxval=cfg.obst_max_radius,
#             )

#             margin = float(r) + 0.1
#             key, subkey_c = jrandom.split(key)
#             cx = jrandom.uniform(
#                 subkey_c,
#                 shape=(),
#                 minval=margin,
#                 maxval=cfg.room_width - margin,
#             )
#             key, subkey_c2 = jrandom.split(key)
#             cy = jrandom.uniform(
#                 subkey_c2,
#                 shape=(),
#                 minval=margin,
#                 maxval=cfg.room_height - margin,
#             )

#             if not centers:
#                 centers.append(jnp.array([cx, cy], dtype=jnp.float32))
#                 radii.append(r.astype(jnp.float32))
#                 placed = True
#             else:
#                 existing_centers = jnp.stack(centers, axis=0)
#                 existing_radii = jnp.array(radii, dtype=jnp.float32)

#                 dx = existing_centers[:, 0] - cx
#                 dy = existing_centers[:, 1] - cy
#                 dist = jnp.sqrt(dx * dx + dy * dy)

#                 min_allowed = existing_radii + r + cfg.obst_clearance
#                 ok = bool(jnp.all(dist >= min_allowed))

#                 if ok:
#                     centers.append(jnp.array([cx, cy], dtype=jnp.float32))
#                     radii.append(r.astype(jnp.float32))
#                     placed = True

#         if not placed:
#             break

#     if centers:
#         centers_arr = jnp.stack(centers, axis=0)
#         radii_arr = jnp.array(radii, dtype=jnp.float32)
#     else:
#         centers_arr = jnp.zeros((0,2), dtype=jnp.float32)
#         radii_arr = jnp.zeros((0,), dtype=jnp.float32)

#     obstacles = Obstacles(
#         centers=centers_arr,
#         radii=radii_arr,
#     )
#     return key, obstacles


# def sample_random_rectangles(
#         rng_key: jnp.ndarray,
#         cfg: StaticConfig,
# ) -> tuple[jnp.ndarray, RectObstacles]:
#     """
#     Random axis-aligned rectangles:
#       - count ~ Uniform{min_rect_obstacles, ..., max_rect_obstacles}
#       - width  in [rect_min_width,  rect_max_width]
#       - height in [rect_min_height, rect_max_height]
#       - placed fully inside the room (no special non-overlap for now)
#     """

#     key = rng_key
#     key, k_key = jrandom.split(key)
#     num_rects = int(jrandom.randint(
#         k_key,
#         shape=(),
#         minval=cfg.min_rect_obstacles,
#         maxval=cfg.max_rect_obstacles + 1,
#     ))

#     if num_rects == 0:
#         rects = RectObstacles(
#             x_min=jnp.zeros((0,), dtype=jnp.float32),
#             y_min=jnp.zeros((0,), dtype=jnp.float32),
#             x_max=jnp.zeros((0,), dtype=jnp.float32),
#             y_max=jnp.zeros((0,), dtype=jnp.float32),
#         )
#         return key, rects
    
#     x_min_list = []
#     y_min_list = []
#     x_max_list = []
#     y_max_list = []

#     for i in range(num_rects):
#         # sample width and height
#         key, w_key = jrandom.split(key)
#         width = jrandom.uniform(
#             w_key,
#             shape=(),
#             minval=cfg.rect_min_width,
#             maxval=cfg.rect_max_width,
#         )
#         key, h_key = jrandom.split(key)
#         height = jrandom.uniform(
#             h_key,
#             shape=(),
#             minval=cfg.rect_min_height,
#             maxval=cfg.rect_max_height,
#         )

#         # pick center so rectangle stays in room
#         margin_x = float(width) / 2.0 + 0.1
#         margin_y = float(height) / 2.0 + 0.1

#         key, cx_key = jrandom.split(key)
#         cx = jrandom.uniform(
#             cx_key,
#             shape=(),
#             minval=margin_x,
#             maxval=cfg.room_width - margin_x,
#         )
#         key, cy_key = jrandom.split(key)
#         cy = jrandom.uniform(
#             cy_key,
#             shape=(),
#             minval=margin_y,
#             maxval=cfg.room_height - margin_y,
#         )

#         x_min_list.append(cx - width / 2.0)
#         y_min_list.append(cy - height / 2.0)
#         x_max_list.append(cx + width / 2.0)
#         y_max_list.append(cy + height / 2.0)

#     rects = RectObstacles(
#         x_min=jnp.array(x_min_list, dtype=jnp.float32),
#         y_min=jnp.array(y_min_list, dtype=jnp.float32),
#         x_max=jnp.array(x_max_list, dtype=jnp.float32),
#         y_max=jnp.array(y_max_list, dtype=jnp.float32),
#     )
#     return key, rects

# def reset(rng_key: jnp.ndarray,
#           cfg: StaticConfig,
#           ):
#     """
#     Resets the environment with random robot + people positions.
#     Returns:
#         new_key, state, obs
#     """
#     # Split key: one for robot, one for people
#     rng_key, obstacles = sample_random_obstacles_jittable(rng_key, cfg)

#     rng_key, rect_obst = sample_random_rectangles_jittable(rng_key, cfg)

#     rng_key, x0, y0 = sample_free_position_jittable(
#         rng_key,
#         cfg,
#         obstacles,
#         rect_obst,
#         entity_radius=cfg.robot_radius,
#         margin=0.2,
#     )

#     goal_key = rng_key
#     while True:
#         goal_key, gx, gy = sample_free_position(
#             goal_key,
#             cfg,
#             obstacles,
#             rect_obst,
#             entity_radius=cfg.goal_radius,
#             margin=cfg.goal_min_robot_dist,
#         )
#         goal_dx = gx - x0
#         goal_dy = gy - y0
#         goal_dist = jnp.sqrt(goal_dx**2 + goal_dy**2)
#         if goal_dist >= cfg.goal_min_robot_dist:
#             break

#     rng_key = goal_key
#     goal_pos = jnp.array([gx, gy], dtype=jnp.float32)   

#     # ---- 2) Random people positions ----
#     if cfg.num_people > 0:
#         people_positions = []
#         key = rng_key
#         for i in range(cfg.num_people):
#             key, px, py = sample_free_position(
#                 key,
#                 cfg,
#                 obstacles,
#                 rect_obst,
#                 entity_radius=cfg.people_radius,
#                 margin=0.1,
#             )
#             people_positions.append(jnp.array([px, py], dtype=jnp.float32))
#         people_pos = jnp.stack(people_positions, axis=0)  
#         rng_key = key
#     else:
#         people_pos = jnp.zeros((0,2), dtype=jnp.float32)  # no people



#     # ---- Assign people velocities ----

#     if cfg.num_people > 0:
#         spped = 0.5
#         rng_key, vel_key = jrandom.split(rng_key)
#         angles = jrandom.uniform(
#             vel_key,
#             shape=(cfg.num_people,),
#             minval=0.0,
#             maxval=2.0 * jnp.pi,
#         )
#         vx = spped * jnp.cos(angles)
#         vy = spped * jnp.sin(angles)
#         people_vel = jnp.stack([vx, vy], axis=1)
#     else:
#         people_vel = jnp.zeros((0,2), dtype=jnp.float32)  # no people



#     state = EnvState(
#         step = jnp.array(0, dtype=jnp.int32),   
#         x = jnp.array(x0, dtype=jnp.float32),
#         y = jnp.array(y0, dtype=jnp.float32),
#         theta = jnp.array(0.0, dtype=jnp.float32),
#         people_positions = people_pos,
#         people_vel = people_vel,
#         goal_pos = goal_pos,
#     )

#     #lidar = lidar_scan(state, cfg)  
#     lidar = lidar_scan(state, cfg, obstacles, rect_obst)

#     obs = jnp.concatenate([
#         jnp.array([state.x, state.y, state.theta], dtype=jnp.float32), lidar], axis=0)
#     return rng_key, state, obs, obstacles, rect_obst