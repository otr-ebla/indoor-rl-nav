import jax
import jax.numpy as jnp
from flax import linen as nn
import distrax 
import jax.lax as lax
from flax.training import train_state
import optax

# --- 1. ENCODER CON LAYERNORM ---
class SensorEncoder(nn.Module):
    @nn.compact
    def __call__(self, lidar_data, goal_polar):
        combined_input = jnp.concatenate([lidar_data, goal_polar], axis=-1)
        
        x = nn.Dense(features=256)(combined_input)
        x = nn.LayerNorm()(x) # <--- STABILIZZATORE
        x = nn.elu(x)
        
        x = nn.Dense(features=256)(x)
        x = nn.LayerNorm()(x) # <--- STABILIZZATORE
        x = nn.elu(x)
        
        return x

# --- 2. RSSM CON LAYERNORM ---
class RSSM(nn.Module):
    deterministic_size: int = 512
    stochastic_size: int = 32 
    discrete_classes: int = 32

    def setup(self):
        self.fc_input = nn.Dense(512)
        self.ln_input = nn.LayerNorm() # <---
        self.rnn = nn.GRUCell(features=self.deterministic_size)
        
        self.fc_prior = nn.Dense(self.stochastic_size * self.discrete_classes)
        
        self.fc_post_embed = nn.Dense(512)
        self.ln_post_embed = nn.LayerNorm() # <---
        self.fc_post = nn.Dense(self.stochastic_size * self.discrete_classes)

    def __call__(self, prev_state, prev_action, embed_obs):
        prev_h, prev_z = prev_state
        
        # 1. Input Memoria
        x = jnp.concatenate([prev_z.reshape((prev_z.shape[0], -1)), prev_action], axis=-1)
        x = self.fc_input(x)
        x = self.ln_input(x) # Norm
        x = nn.elu(x)
        
        # 2. GRU
        h_next, _ = self.rnn(x, prev_h)
        
        # 3. Prior
        prior_out = self.fc_prior(h_next)
        prior_logits = prior_out.reshape(-1, self.stochastic_size, self.discrete_classes)
        
        # 4. Posterior
        x_post = jnp.concatenate([h_next, embed_obs], axis=-1)
        x_post = self.fc_post_embed(x_post)
        x_post = self.ln_post_embed(x_post) # Norm
        x_post = nn.elu(x_post)
        
        post_out = self.fc_post(x_post)
        post_logits = post_out.reshape(-1, self.stochastic_size, self.discrete_classes)
            
        return h_next, post_logits, prior_logits

    def img_step(self, prev_state, prev_action):
        prev_h, prev_z = prev_state
        x = jnp.concatenate([prev_z.reshape((prev_z.shape[0], -1)), prev_action], axis=-1)
        x = self.fc_input(x)
        x = self.ln_input(x) # Norm
        x = nn.elu(x)
        
        h_next, _ = self.rnn(x, prev_h)
        
        prior_out = self.fc_prior(h_next)
        prior_logits = prior_out.reshape(-1, self.stochastic_size, self.discrete_classes)
        prior_probs = nn.softmax(prior_logits)

        return (h_next, prior_probs), prior_logits

# --- 3. HEADS CON LAYERNORM ---
class WorldModelHeads(nn.Module):   
    input_shape: int

    @nn.compact
    def __call__(self, state_features):
        # Decoder Osservazioni
        x_obs = nn.Dense(512)(state_features)
        x_obs = nn.LayerNorm()(x_obs)
        x_obs = nn.elu(x_obs)
        x_obs = nn.Dense(512)(x_obs)
        x_obs = nn.LayerNorm()(x_obs)
        x_obs = nn.elu(x_obs)
        obs_pred = nn.Dense(self.input_shape)(x_obs) 

        # Decoder Reward
        x_rew = nn.Dense(256)(state_features)
        x_rew = nn.LayerNorm()(x_rew)
        x_rew = nn.elu(x_rew)
        reward_pred = nn.Dense(1)(x_rew) 

        # Decoder Continuità
        x_cont = nn.Dense(256)(state_features)
        x_cont = nn.LayerNorm()(x_cont)
        x_cont = nn.elu(x_cont)
        cont_pred = nn.Dense(1)(x_cont) 

        return obs_pred, reward_pred, cont_pred

# --- 4. ACTOR & CRITIC CON LAYERNORM ---
class Actor(nn.Module):
    action_dim: int = 2

    @nn.compact
    def __call__(self, state_features):
        x = nn.Dense(256)(state_features)
        x = nn.LayerNorm()(x) # Importante per evitare azioni giganti!
        x = nn.elu(x)
        x = nn.Dense(256)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)

        out = nn.Dense(self.action_dim * 2)(x)
        mean, std_raw = jnp.split(out, 2, axis=-1)
        std = nn.softplus(std_raw) + 0.1 
        dist = distrax.MultivariateNormalDiag(mean, std)
        return dist

class Critic(nn.Module):
    @nn.compact
    def __call__(self, state_features):
        x = nn.Dense(256)(state_features)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = nn.Dense(256)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        value = nn.Dense(1)(x)
        return value

# --- 5. DREAMER (Invariato, ma include i moduli aggiornati) ---
class Dreamer(nn.Module):
    def setup(self):
        self.encoder = SensorEncoder()
        self.rssm = RSSM()
        self.heads = WorldModelHeads(input_shape=110) 
        self.actor = Actor()
        self.critic = Critic()

    def __call__(self, lidar, goal, actions):
        batch_size = lidar.shape[0]
        embed = self.encoder(lidar, goal)
        actions_T = jnp.swapaxes(actions, 0, 1)
        embed_T = jnp.swapaxes(embed, 0, 1)

        def scan_step(prev_state, inputs):
            action_t, embed_t = inputs
            h_next, post_logits, prior_logits = self.rssm(prev_state, action_t, embed_t)
            post_probs = nn.softmax(post_logits)
            return (h_next, post_probs), (h_next, post_logits, prior_logits)
        
        init_h = jnp.zeros((batch_size, self.rssm.deterministic_size))
        init_z = jnp.zeros((batch_size, self.rssm.stochastic_size, self.rssm.discrete_classes))
        _ = self.rssm((init_h, init_z), actions_T[0], embed_T[0])
        
        _, (h_seq_T, post_logits_seq_T, prior_logits_seq_T) = lax.scan(
            scan_step, (init_h, init_z), (actions_T, embed_T)
        )
        
        h_seq = jnp.swapaxes(h_seq_T, 0, 1)
        post_logits_seq = jnp.swapaxes(post_logits_seq_T, 0, 1)
        prior_logits_seq = jnp.swapaxes(prior_logits_seq_T, 0, 1)
        
        z_seq = nn.softmax(post_logits_seq)
        features = jnp.concatenate([h_seq, z_seq.reshape(batch_size, -1, self.rssm.stochastic_size*32)], axis=-1)
        pred_obs, pred_rew, pred_cont = self.heads(features)

        _ = self.actor(features)
        _ = self.critic(features)
        
        return pred_obs, pred_rew, pred_cont, post_logits_seq, prior_logits_seq, features
    
    def imagination(self, start_h, start_z, horizon=15):
        def dream_step(prev_state, _):
            h, z = prev_state
            features = jnp.concatenate([h, z.reshape(h.shape[0], -1)], axis=-1)
            dist = self.actor(features)
            action = dist.sample(seed=jax.random.PRNGKey(0)) 
            next_state, _ = self.rssm.img_step(prev_state, action)
            
            h_next, z_next = next_state
            feat_next = jnp.concatenate([h_next, z_next.reshape(h_next.shape[0], -1)], axis=-1)
            
            reward = self.heads(feat_next)[1] 
            value = self.critic(feat_next)
            
            return next_state, (reward, value, action, features)

        _, (dream_rews, dream_vals, dream_acts, dream_feats) = lax.scan(
            dream_step, (start_h, start_z), None, length=horizon
        )
        return dream_rews, dream_vals, dream_acts, dream_feats

    def policy_step(self, prev_state, lidar, goal, prev_action):
        embed = self.encoder(lidar, goal)
        h_next, post_logits, _ = self.rssm(prev_state, prev_action, embed)
        post_probs = nn.softmax(post_logits)
        features = jnp.concatenate([h_next, post_probs.reshape(h_next.shape[0], -1)], axis=-1)
        dist = self.actor(features)
        action = dist.mode()
        next_state = (h_next, post_probs)
        return action, next_state

# --- LOSS HELPERS (Con KL Safe + Free Bits) ---
def compute_kl_loss(post_logits, prior_logits):
    post_probs = nn.softmax(post_logits)
    prior_probs = nn.softmax(prior_logits)
    
    # Clip per evitare log(0) = nan
    post_probs = jnp.clip(post_probs, 1e-7, 1.0)
    prior_probs = jnp.clip(prior_probs, 1e-7, 1.0)

    diff_log = jnp.log(post_probs) - jnp.log(prior_probs)
    kl = jnp.sum(post_probs * diff_log, axis=-1) 
    
    # Free Bits: Minimo 1.0 di KL per non schiacciare troppo la dinamica
    kl = jnp.maximum(kl, 1.0) 

    return jnp.mean(kl)

def calculate_lambda_returns(rewards, values, discount=0.99, lambda_=0.95):
    next_values = jnp.concatenate([values[1:], values[-1:]], axis=0)
    inputs = (rewards, next_values)
    
    def step_fn(last_lambda_ret, inp):
        r, v_next = inp
        v_target = r + discount * ((1 - lambda_) * v_next + lambda_ * last_lambda_ret)
        return v_target, v_target

    last_val = values[-1]
    _, lambda_returns = lax.scan(step_fn, last_val, inputs, reverse=True)
    return lambda_returns

# --- TRAIN FUNCTIONS (Con Reward Scaling) ---
@jax.jit
def train_world_model(state, batch):
    def loss_fn(params):
        pred_obs, pred_rew, pred_cont, post_logits, prior_logits, features = state.apply_fn(
            params, batch['lidar'], batch['goal'], batch['actions']
        )

        target_obs = jnp.concatenate([batch['lidar'], batch['goal']], axis=-1)
        target_rew = batch['rewards']

        loss_obs = jnp.mean((pred_obs - target_obs) ** 2)
        
        # Squeeze e SCALING del reward (diviso 10) per stabilità
        loss_rew = jnp.mean((pred_rew.squeeze(-1) - target_rew * 0.1) ** 2)

        loss_kl = compute_kl_loss(post_logits, prior_logits)
        
        # Pesi tipici Dreamer
        total_loss = loss_obs + loss_rew + 0.5 * loss_kl

        logs = {
            'loss_total': total_loss,
            'loss_obs': loss_obs,
            'loss_rew': loss_rew,
            'loss_kl': loss_kl
        }
        return total_loss, (logs, features)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logs, features)), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, logs, features

@jax.jit
def train_actor_critic(state, features):
    h_dim = 512
    batch_time = features.shape[0] * features.shape[1]
    flat_features = features.reshape(batch_time, -1)
    
    start_h = flat_features[:, :h_dim]
    start_z_flat = flat_features[:, h_dim:]
    start_z = start_z_flat.reshape(batch_time, 32, 32)
    
    start_h = jax.lax.stop_gradient(start_h)
    start_z = jax.lax.stop_gradient(start_z)

    def loss_fn(params):
        dream_rews, dream_vals, dream_acts, dream_feats = state.apply_fn(
            params, start_h, start_z, method=Dreamer.imagination
        )
        
        # Scaling del reward sognato per coerenza con WM
        targets = calculate_lambda_returns(dream_rews * 0.1, dream_vals)
        targets = jax.lax.stop_gradient(targets)
        
        loss_critic = jnp.mean((dream_vals - targets) ** 2)
        loss_actor = -jnp.mean(targets)
        
        # Piccola penalità entropica per incoraggiare l'esplorazione
        # (Opzionale, ma aiuta a non collassare su un'azione)
        # loss_actor += -1e-4 * entropy... (omessa per semplicità)
        
        total_loss = loss_actor + loss_critic
        return total_loss, {'loss_actor': loss_actor, 'loss_critic': loss_critic}

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logs), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, logs