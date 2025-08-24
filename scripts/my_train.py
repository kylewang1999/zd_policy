import sys, os, os.path as osp
import optax
import jax
import jax.numpy as jnp
from jax import random
from tqdm import tqdm
import flax.nnx as nnx
import matplotlib.pyplot as plt

from flax.struct import dataclass as flax_dataclass
from flax.struct import field, PyTreeNode

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from double_integrator import *


@flax_dataclass
class CfgData:
    box_width: float = 2.0  # box width for x0s
    batch_size: int = 512
    policies: list[str] = field(pytree_node=False, default_factory=lambda: ["policy_v", "noop"])
    policy_ratios: list[float] = field(pytree_node=False, default_factory=lambda: [0.5, 0.5])

@flax_dataclass
class CfgTrain:
    num_epochs: int = 100
    num_batches: int = 4
    num_logs: int = 10
    lr: float = 1e-2
    

def make_plots(rom: DoubleIntegratorROM, 
               ret: IntegratorOutput, 
               aux_ret: IntegratorAuxOutput, 
               loss_ret: LossOutput, 
               box_width: float = 2.0):
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.set_xlim(-box_width, box_width)
    ax1.set_ylim(-box_width, box_width)
    ax2.set_xlim(-0.1, 0.1)
    ax2.set_ylim(-0.1, 0.1)
    ax1.set_title('(x1,x2) trajectories')
    ax2.set_title('(x1,x2) trajectories (zoomed)')

    for ax in (ax1, ax2):
        ax.set_xlabel('x1 (z)')
        ax.set_ylabel('x2 (y)')
        ax.grid(True, alpha=0.3)

    # Zero dynamics line
    x1s = jnp.linspace(-box_width, box_width, 100)
    x2s = -rom.cfg_rom.kpsi * x1s
    for ax in (ax1, ax2, ax3):
        ax.plot(x1s, x2s, 'k--', linewidth=2, label='Zero dynamics line')

    # rollout trajectories
    for ax in (ax1, ax2):
        for i in range(ret.xs.shape[0]):
            ax.plot(ret.xs[i, :, 0], ret.xs[i, :, 1])
            ax.scatter(ret.xs[i, 0, 0], ret.xs[i, 0, 1], color='red')
            
    # streamplot
    N = 100
    X1, X2 = jnp.meshgrid(
        jnp.linspace(-box_width, box_width, N),
        jnp.linspace(-box_width, box_width, N),
        indexing='xy'
    )
    U = X2
    V = jax.vmap(rom.policy_v, in_axes=(0, 0))(
        rearrange(X2, 'n1 n2 -> (n1 n2)'), rearrange(X1, 'n1 n2 -> (n1 n2)')
    )
    V = rearrange(V.squeeze(), '(n1 n2) -> n1 n2', n1=N, n2=N)
    X1n, X2n, Un, Vn = map(np.asarray, (X1, X2, U, V))
    ax3.streamplot(X1n, X2n, Un, Vn, density=1.2, linewidth=0.8, arrowsize=1.2, minlength=0.2)
            
    plt.show()

    # M_\psi error
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title(r'||y(t) - \psi(z(t))||')
    ax1.set_xlabel('t')
    ax1.grid(True, alpha=0.3)
    ax2.set_title(r'V_z(z(t))')
    ax2.set_xlabel('t')
    ax2.grid(True, alpha=0.3)

    for i in range(aux_ret.xs.shape[0]):
        ax1.plot(aux_ret.ts[:-1], aux_ret.es[i])
        ax2.plot(aux_ret.ts[:-1], aux_ret.lyaps[i])
        
    
    
    attrs = ["y_proj", "z_proj", "stable_m", "invari_m", "nondegenerate_enc"]
    fig, axes = plt.subplots(1, len(attrs), figsize=(4*len(attrs), 3))
    for ax, attr in zip(axes, attrs):
        ax.set_title(attr)
        ax.set_xlabel('t')
        ax.grid(True, alpha=0.3)
        
        for i in range(aux_ret.xs.shape[0]):
            ax.plot(aux_ret.ts[:-1], getattr(loss_ret, attr)[i])

    plt.show()


def get_expert_batch(rom: DoubleIntegratorROM, 
                     integrator: Integrator,
                     cfg_data: CfgData = CfgData(),
                     rng: jax.random.PRNGKey = jax.random.PRNGKey(0)):
    
    '''
    Get a batch of expert data for training.
    NOTE: We might not need this if decoding is not needed. For now assume we don't need this.
    '''
    
    key_x, key_u = jax.random.split(rng, 2)
    x0s = jax.random.uniform(key_x, (cfg_data.batch_size, rom.cfg_rom.dim_x), 
                             minval=-cfg_data.box_width, maxval=cfg_data.box_width)
    
    n_policy_v = int(cfg_data.batch_size * cfg_data.policy_ratios[0])
    n_policy_noop = cfg_data.batch_size - n_policy_v
    x0s_policy_v = x0s[:n_policy_v]
    x0s_policy_noop = x0s[n_policy_v:]

    ret_policy_v = integrator.apply(x0s_policy_v, rom)
    ret_policy_noop = integrator.apply(x0s_policy_noop, rom, lambda y, z: jnp.zeros((rom.cfg_rom.dim_u,)))
    
    return ret_policy_v, ret_policy_noop


def train(rom: NNDoubleIntegratorROM, 
          integrator: Integrator, 
          cfg_train: CfgTrain, 
          cfg_data: CfgData, 
          rng: jax.random.PRNGKey):

    def get_batch(rng: jax.random.PRNGKey):
        key_x = rng
        x0s = jax.random.uniform(key_x, 
                                 (cfg_data.batch_size, rom.cfg_rom.dim_x), 
                                 minval=-cfg_data.box_width, 
                                 maxval=cfg_data.box_width)
        return x0s
    
    @nnx.jit
    def step(model: NNDoubleIntegratorROM, batch):
        
        x0s = batch
        
        def loss_fn(_m):
            int_out  = integrator.apply(x0s, rom=_m)           
            loss_out = integrator.compute_loss(int_out, rom=_m)
            return jnp.mean(loss_out.total), loss_out          
        
        return nnx.value_and_grad(loss_fn, has_aux=True)(model)
    
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.1 * cfg_train.lr, peak_value=cfg_train.lr,
        warmup_steps=int(0.1 * cfg_train.num_epochs), decay_steps=cfg_train.num_epochs,
    )
    tx  = optax.adam(lr_schedule)
    opt = nnx.Optimizer(rom, tx, wrt=nnx.Param)
    rngs = jax.random.split(rng, cfg_train.num_batches)
    
    try:
        
        epoch_losses = []
        
        for i in (pbar := tqdm(range(cfg_train.num_epochs))):
            
            batch_losses = []
            for b in range(cfg_train.num_batches):
            
                batch = get_batch(rngs[b])
                (loss, loss_out), grads = step(rom, batch)
                
                opt.update(grads=grads)
                batch_losses.append(loss)
                
            epoch_loss = jnp.mean(jnp.array(batch_losses))
            epoch_losses.append(epoch_loss)
            pbar.set_postfix({"loss": f"{epoch_loss:.2e}", "lr": f"{lr_schedule(i):.2e}"})
            
            if i % (cfg_train.num_epochs // cfg_train.num_logs) == 0:
                pass
            
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
        return
    
    plt.plot(np.arange(len(epoch_losses)), epoch_losses)