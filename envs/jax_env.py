from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.lax as lax
import numpy as np  
import matplotlib.pyplot as plt
import time
from typing import NamedTuple, Tuple
from functools import partial
from envs.bouncing import bounce_people_on_obstacles

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

    # CIRCULAR OBSTACLES
    min_circ_obstacles: int
    max_circ_obstacles: int
    obst_min_radius: float
    obst_max_radius: float
    obst_clearance: float

    # RECTANGULAR OBSTACLES
    min_rect_obstacles: int
    max_rect_obstacles: int
    rect_min_width: float
    rect_max_width: float
    rect_min_height: float
    rect_max_height: float

    goal_radius: float
    goal_min_robot_dist: float
    goal_reward: float

class Obstacles(NamedTuple):
    centers: jnp.ndarray   # (N, 2)
    radii:   jnp.ndarray   # (N,)

class RectObstacles(NamedTuple):
    x_min: jnp.ndarray  # (N,)
    y_min: jnp.ndarray  # (N,)
    x_max: jnp.ndarray  # (N,)
    y_max: jnp.ndarray  # (N,)



class EnvState(NamedTuple):
    step: jnp.int32
    x: jnp.float32
    y: jnp.float32
    theta: jnp.float32
    people_positions: jnp.ndarray  # (num_people, 2)
    people_vel: jnp.ndarray
    goal_pos: jnp.ndarray  # (2,)


@jax.jit(static_argnums=1)
def sample_free_position_jittable(
    rng_key, 
    cfg, 
    obstacles, 
    rect_obst, 
    entity_radius, 
    margin=0.2, 
    max_tries=200) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    room_margin = entity_radius + margin

    def cond_fun(carry):
        i, _, _, _, safe = carry
        return (i < max_tries) & (safe == jnp.bool_(False))
    
    # 2. Funzione del Corpo (Il tentativo di campionamento)
    def body_fun(carry):
        i, key, _, _, _ = carry
        
        # Split della chiave per campionare x e y
        key, subkey_x, subkey_y = jrandom.split(key, 3)
        
        # Campionamento di x
        x = jrandom.uniform(
            subkey_x,
            shape=(),
            minval=room_margin,
            maxval=cfg.room_width - room_margin,
        )
        
        # Campionamento di y
        y = jrandom.uniform(
            subkey_y,
            shape=(),
            minval=room_margin,
            maxval=cfg.room_height - room_margin,
        )   

        # ---- A) Check ostacoli circolari ----
        if obstacles.radii.size == 0:
            safe_circles = jnp.bool_(True)
        else:
            cx = obstacles.centers[:, 0]
            cy = obstacles.centers[:, 1]
            r = obstacles.radii

            dx = cx - x
            dy = cy - y
            # Calcolo della distanza al quadrato per evitare jnp.sqrt (leggermente più veloce)
            dist_sq = dx * dx + dy * dy 

            combined_r = r + entity_radius 
            
            # Controllo se la distanza è maggiore o uguale alla somma dei raggi al quadrato
            safe_circles = jnp.all(dist_sq >= combined_r * combined_r)

        # ---- B Check ostacoli rettangolari ----
        if rect_obst.x_min.size == 0: # Controlla solo se l'array è vuoto (rect_obst non è None)
            safe_rects = jnp.bool_(True)
        else:
            xmin = rect_obst.x_min
            ymin = rect_obst.y_min
            xmax = rect_obst.x_max
            ymax = rect_obst.y_max
            
            # Trova il punto più vicino sul rettangolo a (x, y)
            closest_x = jnp.clip(x, xmin, xmax)
            closest_y = jnp.clip(y, ymin, ymax)

            dxr = closest_x - x
            dyr = closest_y - y
            dist_rect_sq = dxr * dxr + dyr * dyr

            # E' safe se la distanza dal punto più vicino è >= al raggio dell'entità
            safe_rects = jnp.all(dist_rect_sq >= entity_radius * entity_radius)

        # La posizione è safe se è safe da entrambi i tipi di ostacoli
        safe = safe_circles & safe_rects
        
        # Aggiorna il carry: i+1, la nuova chiave, la nuova x, la nuova y, e il nuovo flag safe
        return (i + 1, key, x, y, safe)

    # Inizializzazione del carry per forzare il primo ciclo
    # (i, key, x, y, safe)
    initial_x = jnp.zeros(())
    initial_y = jnp.zeros(())
    initial_safe = jnp.bool_(False)
    
    final_i, final_key, final_x, final_y, final_safe = jax.lax.while_loop(
        cond_fun, 
        body_fun, 
        (0, rng_key, initial_x, initial_y, initial_safe)
    )

    # --- Gestione del Fallimento del Loop ---
    # Se il loop fallisce dopo max_tries, ritorna una posizione (ad es. 0, 0)
    # e solleva un avvertimento, ma in JAX è meglio ritornare un valore valido.
    
    # Non è strettamente necessario in un ambiente RL ben progettato, 
    # ma assicura che il JIT non crashi se safe rimane False.
    # Useremo la posizione sampleata nell'ultimo tentativo.
    
    return final_key, final_x, final_y

@jax.jit(static_argnums=1)
def sample_random_obstacles_jittable(
    rng_key: jnp.ndarray,
    cfg: StaticConfig,
) -> tuple[jnp.ndarray, Obstacles]:
    
    key, k_key = jrandom.split(rng_key)
    num_obst_max = cfg.max_circ_obstacles
    num_obst = jrandom.randint(
        k_key,
        shape=(),
        minval=cfg.min_circ_obstacles,
        maxval=num_obst_max + 1,
    )

    initial_centers = jnp.zeros((num_obst_max, 2), dtype=jnp.float32)
    initial_radii = jnp.zeros((num_obst_max,), dtype=jnp.float32)
    initial_count = jnp.int32(0) 

    def scan_body(carry, i):
        key, centers_arr, radii_arr, current_count = carry

        max_tries = 200

        def cond_fun(carry_inner):
            tries, _, _, _, _, placed_flag = carry_inner
            return (tries < max_tries) & (placed_flag == jnp.bool_(False))

        def body_fun(carry_inner):
            tries, key_inner, _, _, _, _ = carry_inner
            
            key_inner, subkey_r, subkey_c, subkey_c2 = jrandom.split(key_inner, 4)
            
            r = jrandom.uniform(
                subkey_r, shape=(), 
                minval=cfg.obst_min_radius, 
                maxval=cfg.obst_max_radius
            )
            
            margin = r + 0.1
            cx = jrandom.uniform(subkey_c, shape=(), minval=margin, maxval=cfg.room_width - margin)
            cy = jrandom.uniform(subkey_c2, shape=(), minval=margin, maxval=cfg.room_height - margin)
            
            existing_centers = centers_arr
            existing_radii = radii_arr
            
            ok = jnp.bool_(True)
            
            def check_overlap_fn(existing_data):
                ec, er = existing_data
                dx = ec[:, 0] - cx
                dy = ec[:, 1] - cy
                dist_sq = dx * dx + dy * dy

                min_allowed = (er + r + cfg.obst_clearance)**2
                safety_mask_all = dist_sq >= min_allowed
                
                idx_array = jnp.arange(ec.shape[0])
                valid_mask = idx_array < current_count
                
                return jnp.all(safety_mask_all | ~valid_mask)

            ok_if_exists = jax.lax.cond(
                current_count > 0,
                check_overlap_fn,
                lambda x: jnp.bool_(True),
                (existing_centers, existing_radii)
            )
            ok = ok_if_exists

            return (tries + 1, key_inner, cx, cy, r, ok)

        initial_r = jnp.zeros(()) 
        initial_cx = jnp.zeros(())
        initial_cy = jnp.zeros(())
        
        final_tries, key_out, final_cx, final_cy, final_r, final_placed = jax.lax.while_loop(
            cond_fun, 
            body_fun, 
            (0, key, initial_cx, initial_cy, initial_r, jnp.bool_(False))
        )
        
        def place_obstacle(carry_place):
            centers, radii, count = carry_place
            centers = centers.at[count].set(jnp.array([final_cx, final_cy]))
            radii = radii.at[count].set(final_r)
            count += 1
            return centers, radii, count

        def skip_obstacle(carry_place):
            return carry_place

        centers_arr, radii_arr, current_count = jax.lax.cond(
            (final_placed) & (current_count < num_obst),
            place_obstacle,
            skip_obstacle,
            (centers_arr, radii_arr, current_count)
        )
        
        return (key_out, centers_arr, radii_arr, current_count), None

    final_carry, _ = jax.lax.scan(
        scan_body, 
        (key, initial_centers, initial_radii, initial_count), 
        jnp.arange(num_obst_max)
    )
    
    final_key, final_centers, final_radii, final_count = final_carry


    def return_empty_obstacles():
        # Deve restituire la stessa dimensione di final_centers/final_radii (N_max)
        return Obstacles(
            centers=jnp.zeros((num_obst_max, 2), dtype=jnp.float32),
            radii=jnp.zeros((num_obst_max,), dtype=jnp.float32),
        )

    def return_filled_obstacles():
        # Questo ramo è già a dimensione N_max (final_centers, final_radii)
        return Obstacles(
            centers=final_centers, 
            radii=final_radii,
        )

    obstacles = jax.lax.cond(
        final_count > 0,
        return_filled_obstacles,
        return_empty_obstacles,
    )
    
    return final_key, obstacles

@jax.jit(static_argnums=1)
def sample_random_rectangles_jittable(
    rng_key: jnp.ndarray,
    cfg: StaticConfig,
) -> tuple[jnp.ndarray, RectObstacles]:
    """
    Random axis-aligned rectangles, implementato in modo vettorializzato.
    """
    
    N_max = cfg.max_rect_obstacles
    
    # 1. Determina il numero effettivo di rettangoli (num_rects)
    key, k_key = jrandom.split(rng_key)
    num_rects = jrandom.randint(
        k_key,
        shape=(),
        minval=cfg.min_rect_obstacles,
        maxval=N_max + 1,
    )
    
    # 2. Genera TUTTI i parametri necessari per N_max candidati in modo vettorializzato
    key, w_key, h_key, cx_key, cy_key = jrandom.split(key, 5)

    # Campiona width e height per N_max rettangoli
    widths = jrandom.uniform(
        w_key,
        shape=(N_max,),
        minval=cfg.rect_min_width,
        maxval=cfg.rect_max_width,
    )
    
    heights = jrandom.uniform(
        h_key,
        shape=(N_max,),
        minval=cfg.rect_min_height,
        maxval=cfg.rect_max_height,
    )

    # Calcola i margini e campiona i centri in modo vettorializzato
    margin_x = widths / 2.0 + 0.1
    margin_y = heights / 2.0 + 0.1

    # JAX può usare array come minval/maxval per jrandom.uniform
    cx = jrandom.uniform(
        cx_key,
        shape=(N_max,),
        minval=margin_x, 
        maxval=cfg.room_width - margin_x, 
    )
    
    cy = jrandom.uniform(
        cy_key,
        shape=(N_max,),
        minval=margin_y,
        maxval=cfg.room_height - margin_y,
    )

    # 3. Calcola i valori finali (x_min, ecc.) in modo vettorializzato
    half_w = widths / 2.0
    half_h = heights / 2.0
    
    x_min_all = cx - half_w
    y_min_all = cy - half_h
    x_max_all = cx + half_w
    y_max_all = cy + half_h

    # 4. Ritaglia i risultati alla dimensione effettiva (num_rects)
    
    def return_empty_rects():
        # Questo ramo deve restituire N_max array pieni di zeri.
        zeros_n = jnp.zeros((N_max,), dtype=jnp.float32)
        return (zeros_n, zeros_n, zeros_n, zeros_n)

    def return_filled_rects():
        # Questo ramo restituisce i 4 array calcolati, che hanno già dimensione N_max.
        return (x_min_all, y_min_all, x_max_all, y_max_all)\
        
    all_arrays = (x_min_all, y_min_all, x_max_all, y_max_all)
    
    # Usa lax.cond per eseguire lo slicing solo se num_rects > 0
    x_min_final, y_min_final, x_max_final, y_max_final = jax.lax.cond(
        num_rects > 0,
        return_filled_rects,
        return_empty_rects
    )
    
    rects = RectObstacles(
        x_min=x_min_final,
        y_min=y_min_final,
        x_max=x_max_final,
        y_max=y_max_final,
    )
    
    return key, rects

# Nota: Devi assicurarti che le tue funzioni jittable abbiano un parametro
#       max_tries per la gestione di while_loop in JAX.

@jax.jit(static_argnums=1)
def reset(rng_key: jnp.ndarray,
                   cfg: StaticConfig,
                   ) -> tuple[jnp.ndarray, EnvState, jnp.ndarray, Obstacles, RectObstacles]:
    """
    Resets the environment in a JAX-idiomatic way (JIT-able).
    Returns:
        new_key, state, obs, obstacles, rect_obst
    """
    
    # 1. Campionamento degli ostacoli fissi (già JAX-idiomatico)
    rng_key, obstacles = sample_random_obstacles_jittable(rng_key, cfg)
    rng_key, rect_obst = sample_random_rectangles_jittable(rng_key, cfg)

    # 2. Campionamento della posizione del Robot (già JAX-idiomatico)
    rng_key, x0, y0 = sample_free_position_jittable(
        rng_key,
        cfg,
        obstacles,
        rect_obst,
        entity_radius=cfg.robot_radius,
        margin=0.2,
    )
    
    robot_pos = jnp.array([x0, y0], dtype=jnp.float32)

    # 3. Campionamento della Posizione Goal (Sostituzione di while True)
    
    def cond_fun_goal(carry):
        i, _, _, _, dist_sq = carry
        # Continua finché non è stato posizionato (dist_sq è corto) O i tentativi sono troppi
        return (i < 200) & (dist_sq < cfg.goal_min_robot_dist**2)

    # Nel file jax_env.py, all'interno di def reset(...) -> body_fun_goal:

    def body_fun_goal(carry):
        i, key, _, _, _ = carry
        
        
        key, sample_key = jrandom.split(key)
        
        # Campiona una nuova posizione (riceve 3 valori: chiave_aggiornata, gx, gy)
        new_key, gx, gy = sample_free_position_jittable(
            sample_key, # Usa la chiave per il campionamento
            cfg,
            obstacles,
            rect_obst,
            entity_radius=cfg.goal_radius,
            margin=cfg.goal_min_robot_dist,
        )
    
        # Calcola la distanza dal robot
        goal_dx = gx - x0
        goal_dy = gy - y0
        new_dist_sq = goal_dx**2 + goal_dy**2

        # Restituiamo 5 elementi: (i + 1, chiave_per_loop_successivo, gx, gy, new_dist_sq)
        return (i + 1, new_key, gx, gy, new_dist_sq)

    # Esecuzione del ciclo goal
    # (i, key, gx, gy, dist_sq)
    initial_dist_sq = jnp.zeros(()) # Forziamo il primo ciclo (distanza 0)
    final_i, rng_key, gx, gy, final_dist_sq = jax.lax.while_loop(
        cond_fun_goal, 
        body_fun_goal, 
        (0, rng_key, x0, y0, initial_dist_sq) # Inizializziamo a pos robot per forzare while
    )
    goal_pos = jnp.array([gx, gy], dtype=jnp.float32)


    # 4. Campionamento delle Posizioni delle Persone (Sostituzione di for loop)

    if cfg.num_people > 0:
        
        # Usiamo lax.scan per campionare le posizioni delle persone
        def sample_person_scan_body(carry, i):
            key = carry
            key, new_key = jrandom.split(key)
            
            # Campiona la posizione
            new_key, px, py = sample_free_position_jittable(
                new_key, cfg, obstacles, rect_obst, cfg.people_radius, margin=0.1
            )
            
            return new_key, jnp.array([px, py])

        # Esegui la scansione per cfg.num_people volte
        rng_key, people_pos = jax.lax.scan(
            sample_person_scan_body, 
            rng_key, # carry iniziale
            jnp.arange(cfg.num_people)
        )
    else:
        people_pos = jnp.zeros((0,2), dtype=jnp.float32)


    # 5. Assegnazione Velocità delle Persone (già vettorializzato)
    if cfg.num_people > 0:
        speed = 0.5
        rng_key, vel_key = jrandom.split(rng_key)
        angles = jrandom.uniform(
            vel_key,
            shape=(cfg.num_people,),
            minval=0.0,
            maxval=2.0 * jnp.pi,
        )
        vx = speed * jnp.cos(angles)
        vy = speed * jnp.sin(angles)
        people_vel = jnp.stack([vx, vy], axis=1)
    else:
        people_vel = jnp.zeros((0,2), dtype=jnp.float32)

    # 6. Creazione dello Stato e Osservazione
    state = EnvState(
        step = jnp.array(0, dtype=jnp.int32),   
        x = x0,
        y = y0,
        theta = jnp.array(0.0, dtype=jnp.float32),
        people_positions = people_pos,
        people_vel = people_vel,
        goal_pos = goal_pos,
    )

    dxg = goal_pos[0] - state.x
    dyg = goal_pos[1] - state.y
    dist_to_goal = jnp.sqrt(dxg * dxg + dyg * dyg)

    goal_global_angle = jnp.arctan2(dyg, dxg)
    angle_to_goal = goal_global_angle - state.theta
    angle_to_goal = jnp.arctan2(jnp.sin(angle_to_goal), jnp.cos(angle_to_goal))

    lidar = lidar_scan(state, cfg, obstacles, rect_obst)

    obs = jnp.concatenate([
            jnp.array([dist_to_goal, angle_to_goal], dtype=jnp.float32), 
            lidar
            ], 
        axis=0)
        
    return rng_key, state, obs, obstacles, rect_obst



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
    """

    # Definizione della logica per il caso con ostacoli (True branch)
    def calc_gap(centers, radii):
        cx = centers[:, 0]
        cy = centers[:, 1]
        r  = radii

        dx = cx - px
        dy = cy - py

        dist = jnp.sqrt(dx * dx + dy * dy)
        combined_r = robot_radius + r
        gap = dist - combined_r
        return jnp.min(gap)
    
    # Definizione della logica per il caso senza ostacoli (False branch)
    def no_gap(centers, radii):
        return jnp.array(jnp.inf, dtype=jnp.float32)

    # Uso di lax.cond: la condizione deve essere statica (basata su .size)
    result = jax.lax.cond(
        radii.size == 0, # Condizione statica: l'array è vuoto?
        no_gap,
        calc_gap,
        centers, radii # Vengono passati come argomenti ai branch
    )

    return result

def _min_gap_to_rectangles(
    px: jnp.ndarray,
    py: jnp.ndarray,
    rects: RectObstacles,
    robot_radius: float,
) -> jnp.ndarray:
    """
    Minimum distance gap between robot circle and axis-aligned rectangles.
    """
    
    # Definizione della logica per il caso con ostacoli (True branch)
    def calc_gap(rects):
        xmin = rects.x_min
        ymin = rects.y_min
        xmax = rects.x_max
        ymax = rects.y_max

        closest_x = jnp.clip(px, xmin, xmax)
        closest_y = jnp.clip(py, ymin, ymax)

        dx = closest_x - px
        dy = closest_y - py
        dist = jnp.sqrt(dx * dx + dy * dy)
        gap = dist - robot_radius
        return jnp.min(gap)

    # Definizione della logica per il caso senza ostacoli (False branch)
    def no_gap(rects):
        return jnp.array(jnp.inf, dtype=jnp.float32)

    # Uso di lax.cond
    result = jax.lax.cond(
        rects.x_min.size == 0, # Condizione statica: l'array è vuoto?
        no_gap,
        calc_gap,
        rects # Passato come unico argomento al branch
    )

    return result

@jax.jit
def _bounce_all_obstacles_single_iteration(
    people_pos: jnp.ndarray,
    people_vel: jnp.ndarray,
    circ_obst: Obstacles,
    rect_obst: RectObstacles,
    people_radius: float,
    room_width: float,
    room_height: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Single iteration that handles ALL obstacle types including walls.
    """
    def per_person(p, v):
        # Check walls first (most common collision)
        wall_penetration = jnp.array(-jnp.inf, dtype=jnp.float32)
        wall_normal = jnp.array([0.0, 0.0], dtype=jnp.float32)
        
        # Left wall
        left_pen = people_radius - p[0]
        # Right wall  
        right_pen = people_radius - (room_width - p[0])
        # Bottom wall
        bottom_pen = people_radius - p[1]
        # Top wall
        top_pen = people_radius - (room_height - p[1])
        
        wall_pens = jnp.array([left_pen, right_pen, bottom_pen, top_pen])
        wall_normals = jnp.array([
            [1.0, 0.0],   # left
            [-1.0, 0.0],  # right  
            [0.0, 1.0],   # bottom
            [0.0, -1.0]   # top
        ])
        
        wall_j_hit = jnp.argmax(wall_pens)
        wall_penetration = wall_pens[wall_j_hit]
        wall_normal = wall_normals[wall_j_hit]
        
        # Check circular obstacles
        circ_penetration = jnp.array(-jnp.inf, dtype=jnp.float32)
        circ_normal = jnp.array([0.0, 0.0], dtype=jnp.float32)
        
        if circ_obst.radii.size > 0:
            diff_circ = circ_obst.centers - p
            dist_circ = jnp.linalg.norm(diff_circ, axis=1)
            combined_r_circ = circ_obst.radii + people_radius
            penetration_circ = combined_r_circ - dist_circ
            
            circ_j_hit = jnp.argmax(penetration_circ)
            circ_penetration = penetration_circ[circ_j_hit]
            n_circ = diff_circ[circ_j_hit]
            circ_normal = n_circ / (jnp.linalg.norm(n_circ) + 1e-6)
        
        # Check rectangular obstacles
        rect_penetration = jnp.array(-jnp.inf, dtype=jnp.float32)
        rect_normal = jnp.array([0.0, 0.0], dtype=jnp.float32)
        
        if rect_obst.x_min.size > 0:
            xmin = rect_obst.x_min
            ymin = rect_obst.y_min
            xmax = rect_obst.x_max
            ymax = rect_obst.y_max
            
            closest_x = jnp.clip(p[0], xmin, xmax)
            closest_y = jnp.clip(p[1], ymin, ymax)
            
            diff_rect = jnp.stack([closest_x - p[0], closest_y - p[1]], axis=1)
            dist_rect = jnp.linalg.norm(diff_rect, axis=1)
            penetration_rect = people_radius - dist_rect
            
            rect_j_hit = jnp.argmax(penetration_rect)
            rect_penetration = penetration_rect[rect_j_hit]
            n_rect = diff_rect[rect_j_hit]
            rect_normal = n_rect / (jnp.linalg.norm(n_rect) + 1e-6)
        
        # Find the most penetrating obstacle overall
        max_wall_pen = jnp.maximum(wall_penetration, 0.0)
        max_circ_pen = jnp.maximum(circ_penetration, 0.0)
        max_rect_pen = jnp.maximum(rect_penetration, 0.0)
        
        # Array di tutte le penetrazioni massime (Indici: 0=Wall, 1=Circle, 2=Rect)
        all_pens = jnp.stack([max_wall_pen, max_circ_pen, max_rect_pen]) # shape (3,)

        # 2. Trovare la penetrazione massima e l'indice
        max_pen = jnp.max(all_pens)
        j_hit_global = jnp.argmax(all_pens)

        # 3. Raccogliere le normali candidate in un array (Wall, Circle, Rect)
        all_normals = jnp.stack([wall_normal, circ_normal, rect_normal], axis=0) # shape (3, 2)

        # 4. Selezionare la normale corretta usando l'indice (j_hit_global)
        bounce_normal = all_normals[j_hit_global]
        
        # --- FINE NUOVA SELEZIONE ---
        
        def do_bounce(p, v, normal, penetration):
            overlap = penetration + 1e-3
            p_new = p + normal * overlap
            vn = jnp.dot(v, normal)
            v_new = v - 2.0 * vn * normal # Rimbalzo
            return p_new, v_new
        
        def no_bounce(p, v, normal, penetration):
            return p, v
        
        p_new, v_new = jax.lax.cond(
            max_pen > 0.0,
            do_bounce,
            no_bounce,
            p, v, bounce_normal, max_pen
        )
        
        return p_new, v_new
    
    return jax.vmap(per_person, in_axes=(0, 0))(people_pos, people_vel)



@jax.jit
def _bounce_people_on_obstacles(
    people_pos: jnp.ndarray,
    people_vel: jnp.ndarray,
    circ_obst: Obstacles,
    rect_obst: RectObstacles,
    people_radius: float,
    room_width: float,
    room_height: float,
    max_iterations: int = 10,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Iterative collision resolution with ALL obstacle types including walls.
    """
    if people_pos.shape[0] == 0:
        return people_pos, people_vel
    
    def body_fun(i, carry):
        p, v = carry
        p_new, v_new = _bounce_all_obstacles_single_iteration(
            p, v, circ_obst, rect_obst, people_radius, room_width, room_height
        )
        return (p_new, v_new)
    
    # Use lax.fori_loop for proper JAX compilation
    people_pos_final, people_vel_final = jax.lax.fori_loop(
        0, max_iterations, body_fun, (people_pos, people_vel)
    )
    
    return people_pos_final, people_vel_final

def correct_people_vs_obstacles_once(pos, circles, rectangles, r):
    """
    Apply one-shot correction from circles and rectangles.
    No bouncing here, only remove penetration.
    """
    # --- Circles correction ---
    if (circles is not None) and (circles.centers.size > 0):
        center = circles.centers
        radii = circles.radii

        diff = pos[:, None, :] - center[None, :, :] # (P, C, 2)    
        dist = jnp.linalg.norm(diff, axis=-1)

        min_dist = r + radii[None, :]  # (1, C)
        overlap = jnp.maximum(0.0, min_dist - dist)

        dirs = diff / (dist[..., None] + 1e-8)  # (P, C, 2)

        pos = pos + jnp.sum(dirs * overlap[..., None], axis=1)

    # ---- Rectangles correction ----
    if (rectangles is not None) and (rectangles.x_min.size > 0):
        xmin = rectangles.x_min
        ymin = rectangles.y_min
        xmax = rectangles.x_max
        ymax = rectangles.y_max

        closest_x = jnp.clip(pos[:, 0:1], xmin[None, :], xmax[None, :])  # (P, R)
        closest_y = jnp.clip(pos[:, 1:2], ymin[None, :], ymax[None, :])  # (P, R)

        dx = closest_x - pos[:, 0:1]  # (P, R)
        dy = closest_y - pos[:, 1:2]  # (P, R
        dist = jnp.sqrt(dx * dx + dy * dy + 1e-8)  # (P, R)

        penetration = jnp.maximum(0.0, r - dist)  # (P, R)  
        nx = dx / (dist + 1e-8)  # (P, R)
        ny = dy / (dist + 1e-8)  # (P,

        corr_x = nx * penetration  # (P, R)
        corr_y = ny * penetration  # (P, R)

        total_corr_x = jnp.sum(corr_x, axis=1)  # (P,)
        total_corr_y = jnp.sum(corr_y, axis=1)  # (P,)
        pos = pos + jnp.stack([total_corr_x, total_corr_y], axis=1)

    return pos

def conservative_people_step(people_pos, people_vel, cfg, circles, rectangles):
    """
    Reduce interpenetration risk by moving people in N small steps
    with early collision correction.
    """
    N = 5

    dt_sub = cfg.dt / N

    def body(i, state):
        pos = state
        pos_new = pos + people_vel * dt_sub
        pos_corrected = correct_people_vs_obstacles_once(
            pos_new,
            circles, rectangles,
            cfg.people_radius,
        )
        return pos_corrected
    
    final_pos = lax.fori_loop(0, N, body, people_pos)
    return final_pos

def step(state: EnvState, 
        action: jnp.ndarray, 
        cfg: StaticConfig, 
        obstacles: Obstacles,
        rect_obst: RectObstacles,):
    """
    One simulation step.
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

    # ---- Update people positions ----
    #people_pos_prop = state.people_positions + state.people_vel * cfg.dt
    people_pos_prop = state.people_positions + state.people_vel * cfg.dt


    # ---- Robust bouncing vs obstacles and walls ----
    people_pos_new, people_vel_new = _bounce_people_on_obstacles(
        people_pos_prop,       # Posizioni proposte
        state.people_vel,      # Velocità attuali
        obstacles,             # Ostacoli circolari
        rect_obst,             # Ostacoli rettangolari
        cfg.people_radius,     # <--- Passa il valore float, non tutto 'cfg'
        cfg.room_width,        # <--- Passa width esplicito
        cfg.room_height,       # <--- Passa height esplicito
        max_iterations=3       # (Opzionale, default è 10 ma 3 basta per performance)
    )
    

    # Build new EnvState
    new_state = EnvState(
        step = state.step + jnp.int32(1),
        x = x_new,
        y = y_new,
        theta = theta_new,
        people_positions = people_pos_new,
        people_vel = people_vel_new,
        goal_pos = state.goal_pos,
    )

    #lidar = lidar_scan(new_state, cfg)
    lidar = lidar_scan(new_state, cfg, obstacles, rect_obst)

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

    # Rectangles
    gap_rects = _min_gap_to_rectangles(
        new_state.x,
        new_state.y,
        rect_obst,
        cfg.robot_radius,
    )

    min_gap = jnp.minimum(jnp.minimum(gap_obst, gap_people), gap_rects)
    collision = min_gap < 0.0

    # ---- reward and done usinf LiDAR + collision ----
    min_dist = jnp.min(lidar)
    base_reward = (min_dist/cfg.max_lidar_distance) # convert to python float

    collision_penalty = jnp.where(collision, -cfg.goal_reward, 0.0) # convert to python float

    # --- Goal Check ---
    gx, gy = new_state.goal_pos[0], new_state.goal_pos[1]
    dxg = gx - new_state.x
    dyg = gy - new_state.y
    dist_to_goal = jnp.sqrt(dxg**2 + dyg**2)

    # Angolo del goal rispetto al robot (nel frame globale)
    goal_global_angle = jnp.arctan2(dyg, dxg)

    # Angolo del goal rispetto alla direzione del robot (nel frame del robot)
    angle_to_goal = goal_global_angle - new_state.theta
    # Normalizza l'angolo tra [-pi, pi]
    angle_to_goal = jnp.arctan2(jnp.sin(angle_to_goal), jnp.cos(angle_to_goal))

    # --- Costruzione dell'Osservazione Finale ---
    # Rimuovi x, y, theta (coordinate globali) e usa solo i dati locali:
    obs = jnp.concatenate([
        jnp.array([dist_to_goal, angle_to_goal], dtype=jnp.float32), # Polari
        lidar,
        ], 
    axis=0)

    reached_goal = dist_to_goal <= cfg.goal_radius  

    goal_bonus = jnp.where(reached_goal, cfg.goal_reward, 0.0)# convert to python float

    reward = base_reward + collision_penalty + goal_bonus

    safety_margin = 0.2
    done = (min_dist < safety_margin) | collision | reached_goal # convert to python bool
 

    return new_state, obs, reward, done 

@partial(jax.jit, static_argnums=(3,))
def auto_reset_step(
    state: EnvState,
    obs: jnp.ndarray,
    action: jnp.ndarray,
    cfg: StaticConfig,
    obstacles: Obstacles,
    rect_obst: RectObstacles,
    rng_key: jnp.ndarray
) -> Tuple[EnvState, jnp.ndarray, jnp.float32, jnp.bool_, Obstacles, RectObstacles]:
    """
    Esegue uno step e, se done=True, resetta l'ambiente (inclusi nuovi ostacoli).
    Tutto in JAX puro, senza ricompilazione.
    """
    
    # 1. Esegui lo step normale
    # Nota: dobbiamo importare 'step' o averlo definito sopra
    next_state, next_obs, reward, done = step(state, action, cfg, obstacles, rect_obst)

    # 2. Prepara il potenziale reset (calcolato SEMPRE per via di JAX)
    rng_key, reset_key = jax.random.split(rng_key)
    
    # Genera un ambiente completamente nuovo (nuovi ostacoli, nuova mappa)
    _, reset_state, reset_obs, reset_obst, reset_rect = reset(reset_key, cfg)

    # 3. Seleziona tra continuazione o reset usando done come switch
    # Usa tree_map per gestire le strutture dati annidate (EnvState, Obstacles)
    
    def where_struct(s1, s2):
        # Se done è True, prendi s2 (reset), altrimenti s1 (next)
        return jax.tree_util.tree_map(
            lambda x, y: jnp.where(done, y, x), 
            s1, s2
        )

    final_state = where_struct(next_state, reset_state)
    final_obs = jnp.where(done, reset_obs, next_obs) # Obs è un array semplice
    
    # Anche gli ostacoli cambiano se l'episodio finisce!
    final_obstacles = where_struct(obstacles, reset_obst)
    final_rect_obstacles = where_struct(rect_obst, reset_rect)

    # Nota: il reward e done ritornati sono quelli dello step appena concluso.
    # Il reset influenza lo stato per il passo *successivo*.
    
    return final_state, final_obs, reward, done, final_obstacles, final_rect_obstacles


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

def _ray_distance_to_walls_obstacles_people(
    x0: jnp.ndarray,
    y0: jnp.ndarray,
    angle: jnp.ndarray,
    cfg: StaticConfig,
    circ_obst: Obstacles,
    rect_obst: RectObstacles,
    people_pos: jnp.ndarray,   # (num_people, 2)
) -> jnp.ndarray:
    # 1) Distance to room walls
    d_walls = _ray_distace_to_walls(x0, y0, angle, cfg)

    # 2) Distance to circular obstacles
    d_circles = _ray_distance_to_circles(
        x0,
        y0,
        angle,
        circ_obst,
        cfg.max_lidar_distance,
    )

    # 3) Distance to rectangular obstacles
    d_rects = _ray_distance_to_rectangles(
        x0,
        y0,
        angle,
        rect_obst,
        cfg.max_lidar_distance,
    )

    # 4) Distance to people (as circles with radius cfg.people_radius)
    if cfg.num_people > 0:
        people_obstacles = Obstacles(
            centers=people_pos,
            radii=jnp.full(
                (cfg.num_people,),
                cfg.people_radius,
                dtype=jnp.float32,
            ),
        )
        d_people = _ray_distance_to_circles(
            x0,
            y0,
            angle,
            people_obstacles,
            cfg.max_lidar_distance,
        )
    else:
        d_people = jnp.array(cfg.max_lidar_distance, dtype=jnp.float32)

    return jnp.minimum(jnp.minimum(d_walls, jnp.minimum(d_circles, d_rects)), d_people)


def _ray_distance_to_rectangles(
    x0: jnp.ndarray,
    y0: jnp.ndarray,
    angle: jnp.ndarray,
    rects: RectObstacles,
    max_range: float,
) -> jnp.ndarray:
    """
    Distance from (x0, y0) along ray 'angle' to the nearest axis-aligned rectangular obstacle.
    Rectangles given by xmin, ymin, xmax, ymax arrays.
    Returns max_range if no valid intersection.
    """
    if rects.x_min.size == 0:
        return jnp.array(max_range, dtype=jnp.float32)
    
    dx = jnp.cos(angle)
    dy = jnp.sin(angle)

    eps = 1e-6

    xmin = rects.x_min
    ymin = rects.y_min
    xmax = rects.x_max
    ymax = rects.y_max
    
    # ----- X slabs -----
    tx1 = (xmin - x0) / (dx + eps)
    tx2 = (xmax - x0) / (dx + eps)
    tmin_x = jnp.minimum(tx1, tx2)
    tmax_x = jnp.maximum(tx1, tx2)

    # ----- Y slabs -----
    ty1 = (ymin - y0) / (dy + eps)
    ty2 = (ymax - y0) / (dy + eps)
    tmin_y = jnp.minimum(ty1, ty2)
    tmax_y = jnp.maximum(ty1, ty2)

    t_enter = jnp.maximum(tmin_x, tmin_y)
    t_exit = jnp.minimum(tmax_x, tmax_y)

    valid = (t_exit > 0.0) & (t_enter <= t_exit)
    t_hit = jnp.where(t_enter > 0.0, t_enter, t_exit)

    t_hit = jnp.where(valid, t_hit, max_range)
    min_t = jnp.min(t_hit)
    return jnp.minimum(min_t, max_range).astype(jnp.float32)

def lidar_scan(
        state: EnvState, 
        cfg: StaticConfig,
        obstacles: Obstacles,
        rect_obst: RectObstacles,
) -> jnp.ndarray:
    """
    Real Lidar to walls
    - num rays beams around
    - ray 0 aligned with robot heading
    """

    base_angles = jnp.linspace(0.0, 2.0 * jnp.pi, cfg.num_rays, endpoint=False)
    ray_angles = (state.theta + base_angles) 

    dist_fn = jax.vmap(
        _ray_distance_to_walls_obstacles_people, 
        in_axes=(None, None, 0, None, None, None, None),
    ) # 
    distances = dist_fn(
        state.x, state.y, 
        ray_angles, 
        cfg,
        obstacles,
        rect_obst,
        state.people_positions,
    )

    return distances.astype(jnp.float32)    














def render(state: EnvState, 
    cfg: StaticConfig, 
    ax=None, 
    obstacles: Obstacles=None,
    rect_obst: RectObstacles=None):
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

        # --- draw rectangular obstacles ---
    if rect_obst is not None and rect_obst.x_min.size > 0:
        for xmin, ymin, xmax, ymax in zip(
            np.array(rect_obst.x_min),
            np.array(rect_obst.y_min),
            np.array(rect_obst.x_max),
            np.array(rect_obst.y_max),
        ):
            rect = plt.Rectangle(
                (float(xmin), float(ymin)),
                float(xmax - xmin),
                float(ymax - ymin),
                color='brown',
                alpha=0.5,
            )
            ax.add_patch(rect)


    # --- draw goal as a red circle ---
    gx, gy = float(state.goal_pos[0]), float(state.goal_pos[1])
    ax.scatter(
        gx, gy,
        marker='*',
        s=100,
        color='gold',
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
    )

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
    lidar = lidar_scan(state, cfg, obstacles, rect_obst)

    base_angles = jnp.linspace(0.0, 2.0 * jnp.pi, cfg.num_rays, endpoint=False)
    ray_angles = (state.theta + base_angles)

    x0 = float(state.x)
    y0 = float(state.y)

    base_grey = (0.6, 0.6, 0.6) 
    base_red  = (1.0, 0.0, 0.0)

    for d, ang in zip(np.array(lidar), np.array(ray_angles)):
        x1 = x0 + float(d) * float(jnp.cos(ang))
        y1 = y0 + float(d) * float(jnp.sin(ang))

        color_ray_dist = 5.0
        norm_d = min(d, color_ray_dist) / color_ray_dist
        proximity = (1.0 - norm_d) 

        r = base_grey[0] + proximity * (base_red[0] - base_grey[0])
        g = base_grey[1] + proximity * (base_red[1] - base_grey[1])
        b = base_grey[2] + proximity * (base_red[2] - base_grey[2])

        alpha = 0.9 * proximity
        rgba = (r, g, b, alpha)

        ax.plot([x0, x1], [y0, y1], color=rgba, linewidth=0.5, alpha=0.5)

        ax.plot(
                x1, y1,
                marker='o',
                markersize=2,
                color=rgba,
            )

    plt.pause(0.001)

    if created_fig:
        return ax
    




def fast_rollout(num_steps: int = 10000):
    static_cfg = StaticConfig(
        dt=0.1,
        room_width=15.0,
        room_height=15.0,
        max_lin_vel=1.0,
        max_ang_vel=jnp.pi,
        robot_radius=0.2,
        num_rays=108,
        max_lidar_distance=20.0,
        num_people=15,
        people_radius=0.2,

        min_circ_obstacles=2,
        max_circ_obstacles=7,
        obst_min_radius=0.3,
        obst_max_radius=1.5,
        obst_clearance=0.2,

        min_rect_obstacles=1,
        max_rect_obstacles=5,
        rect_min_width=1.0,
        rect_max_width=3.0,
        rect_min_height=1.0,
        rect_max_height=3.0,

        goal_radius=0.3,
        goal_min_robot_dist=3.0,
        goal_reward=10.0,
    )

    rng_key = jrandom.PRNGKey(0)
    rng_key, state, obs, obstacles, rect_obst = reset(rng_key, static_cfg)

    step_fn = make_step_fn(static_cfg, obstacles, rect_obst)
    action = jnp.array([0.5, 0.3], dtype=jnp.float32)

    state, obs, reward, done = step_fn(state, action)  # warm-up JIT

    t0 = time.time()
    steps_done = 0
    for t in range(num_steps):
        state, obs, reward, done = step_fn(state, action)
        steps_done += 1

        if done:
            rng_key, state, obs, obstacles, rect_obst = reset(rng_key, static_cfg)
            step_fn = make_step_fn(static_cfg, obstacles, rect_obst)
            break
    t1 = time.time()

    fps = steps_done / (t1 - t0)
    print(f"Fast rollout: {steps_done} steps, ~{fps:.1f} FPS (no render, no sleep)")























if __name__ == "__main__":
    static_cfg = StaticConfig(
        dt=0.1, # Rallentiamo il dt per un'animazione più fluida
        room_width=15.0,
        room_height=15.0,
        max_lin_vel=1.0,
        max_ang_vel=jnp.pi,
        robot_radius=0.2,
        num_rays=108,
        max_lidar_distance=20.0,
        num_people=30, # Riduciamo per la visualizzazione
        people_radius=0.2,

        min_circ_obstacles=2,
        max_circ_obstacles=5,
        obst_min_radius=0.3,
        obst_max_radius=1.5,
        obst_clearance=0.2,

        min_rect_obstacles=2,
        max_rect_obstacles=6,
        rect_min_width=1.0,
        rect_max_width=3.0,
        rect_min_height=1.0,
        rect_max_height=3.0,

        goal_radius=0.3,
        goal_min_robot_dist=3.0,
        goal_reward=10.0,
    )

    rng_key = jrandom.PRNGKey(int(time.time()))
    action = jnp.array([0.5, 0.3], dtype=jnp.float32)

    num_episodes = 20
    max_steps_per_episode = 300 

    # 1) Creazione della figura e assi
    fig, ax = plt.subplots(figsize=(5, 5))

    # 2) Loop principale per il rendering
    for ep in range(num_episodes):
        print(f"=== Episodio {ep} ===")

        # A) Reset dell'ambiente (chiamando la tua funzione ottimizzata)
        rng_key, state, obs, obstacles, rect_obst = reset(
            rng_key, 
            static_cfg,
        )
        
        # B) Crea la funzione step JIT-compilata
        step_fn = make_step_fn(static_cfg, obstacles, rect_obst)

        for t in range(max_steps_per_episode):
            t0 = time.time()

            # Step dell'ambiente (JIT-compiled)
            state, obs, reward, done = step_fn(state, action)

            # C) Rendering dello stato corrente
            render(
                state, 
                static_cfg, 
                ax=ax, 
                obstacles=obstacles,
                rect_obst=rect_obst,
            )
                
            # D) Pausa per la visualizzazione
            elapsed = time.time() - t0
            if elapsed < static_cfg.dt:
                time.sleep(static_cfg.dt - elapsed)

            if done:
                print(f"Episodio {ep} terminato a step {int(state.step)}")
                break

    print("Simulazione di rendering terminata.")
    # Mantieni aperta la finestra Matplotlib alla fine
    plt.show() 

    
