import sys, os, os.path as osp
import optax
import jax
import jax.numpy as jnp
from jax import random
from tqdm import tqdm
import flax.nnx as nnx
import matplotlib.pyplot as plt
import numpy as np
from functools import wraps
from pprint import pprint

from flax.struct import dataclass as flax_dataclass
from flax.struct import field, PyTreeNode
import jax.tree_util as jtu
from flax.traverse_util import flatten_dict, unflatten_dict

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from double_integrator import *


def catch_keyboard_interrupt(message: str = "Training interrupted by user"):
    """
    Decorator to handle KeyboardInterrupt gracefully in training functions.
    
    Args:
        message: Custom message to print when interrupted
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                print(message)
                return None
        return wrapper
    return decorator


@flax_dataclass
class CfgData:
    box_width: float = 2.0  # box width for x0s
    batch_size: int = 512
    policies: list[str] = field(pytree_node=False, default_factory=lambda: ["policy_v", "noop"])
    policy_ratios: list[float] = field(pytree_node=False, default_factory=lambda: [0.5, 0.5])

@flax_dataclass
class CfgTrain:
    num_epochs: int = 10
    num_batches: int = 5
    num_logs: int = 10
    lr: float = 5e-2
    enable_lr_schedule: bool = False


def corrupt_params(rom: NNDoubleIntegratorROM, 
                   rng: jax.random.PRNGKey,
                   corruption_factor: float = 1.0):
    params = rom.get_nn_params()
    leaves, treedef = jtu.tree_flatten(params)
    keys = jax.random.split(rng, len(leaves))
    keys_tree = jtu.tree_unflatten(treedef, keys)

    corrupted_params = jtu.tree_map(
        lambda x, k: x + jax.random.normal(k, x.shape) * corruption_factor,
        params, keys_tree
    )
    rom.set_nn_params(corrupted_params)


def make_traj_plots(rom: DoubleIntegratorROM | NNDoubleIntegratorROM, 
                    int_out: IntegratorOutput, 
                    aux_out: IntegratorAuxOutput, 
                    box_width: float = 2.0,
                    max_traj_num: int = 100):
    
    max_traj_num = min(max_traj_num, int_out.xs.shape[0])
    
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(25, 5))
    ax1.set_xlim(-box_width, box_width)
    ax1.set_ylim(-box_width, box_width)
    ax2.set_xlim(-box_width/5, box_width/5)
    ax2.set_ylim(-box_width/5, box_width/5)
    ax1.set_title('(z1,z2) trajectories')
    ax2.set_title('(z1,z2) trajectories (zoomed)')
    ax3.set_title('(x1,x2) trajectories')
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.grid(True, alpha=0.3)

    # zero dynamics line
    if not isinstance(rom, NNDoubleIntegratorROM):
        x1s = jnp.linspace(-box_width, box_width, 100).reshape(-1,1)
        psi_slope = jax.vmap(jax.grad(lambda x: rom.policy_psi(x).squeeze()))(x1s).squeeze()
        x2s = psi_slope * x1s
        for ax in (ax1, ax2):
            ax.plot(x1s, x2s, 'k--', linewidth=2, label='Zero dynamics line')
            
    else:
        zs = jnp.linspace(-box_width, box_width, 100).reshape(-1,1)
        ys = jax.vmap(rom.policy_psi, in_axes=(0,))(zs)
        E = rom.get_nn_params()["nn_encoder"]["kernel"]
        E_inv = jnp.linalg.inv(E.T)
        zetas = jnp.stack([ys, zs], axis=-1)
        # xs_backproj = E_inv.T @ zetas
        # x1s, x2s = xs_backproj[:, 0], xs_backproj[:, 1]
        for ax in (ax1, ax2, ax3):
            ax.plot(ys, zs, 'k--', linewidth=2, label='Zero dynamics line')
            # ax.plot(x1s, x2s, 'k--', linewidth=2, label='Zero dynamics line')

    # rollout trajectories in x space
    # for ax in (ax1, ax2):
    #     for i in range(int_out.xs.shape[0]):
    #         ax.plot(int_out.xs[i, :, 0], int_out.xs[i, :, 1])
    #         ax.scatter(int_out.xs[i, 0, 0], int_out.xs[i, 0, 1], color='red')
    
    # rollout trajectories in E(x) space
    for ax in (ax1, ax2):
        for i in range(max_traj_num):
            ax.plot(aux_out.ys[i, :, 0], aux_out.zs[i, :, 0])
            ax.scatter(aux_out.ys[i, 0, 0], aux_out.zs[i, 0, 0], color='red')
            
    # streamplot
    N = 100
    X1, X2 = jnp.meshgrid(
        jnp.linspace(-box_width, box_width, N),
        jnp.linspace(-box_width, box_width, N),
        indexing='xy'
    )
    
    flat_x = jnp.stack([rearrange(X1, 'n1 n2 -> (n1 n2)'),
                        rearrange(X2, 'n1 n2 -> (n1 n2)')], axis=-1) 
    ys, zs = jax.vmap(rom.encode, in_axes=(0,))(flat_x)
    v_flat = jax.vmap(rom.policy_v, in_axes=(0, 0))(ys, zs)
    u_flat = jax.vmap(rom.map_v_to_u, in_axes=(0,))(v_flat)
    flax_dx = jax.vmap(rom.dyn_x, in_axes=(0, 0))(flat_x, u_flat)
    
    DX1 = rearrange(flax_dx[:, 0], '(n1 n2) -> n1 n2', n1=N, n2=N)
    DX2 = rearrange(flax_dx[:, 1], '(n1 n2) -> n1 n2', n1=N, n2=N)
    
    X1n, X2n, Un, Vn = map(np.asarray, (X1, X2, DX1, DX2))
    ax3.streamplot(X1n, X2n, Un, Vn, density=1.2, linewidth=0.8, arrowsize=1.2, minlength=0.2)

    # M_\psi error
    ax4.set_title(r'||y(t) - \psi(z(t))||')
    ax4.set_xlabel('t')
    ax4.grid(True, alpha=0.3)
    ax5.set_title(r'V_z(z(t))')
    ax5.set_xlabel('t')
    ax5.grid(True, alpha=0.3)

    for i in range(max_traj_num):
        ax4.plot(aux_out.ts[:-1], aux_out.es[i])
        ax5.plot(aux_out.ts[:-1], aux_out.lyaps[i])

    plt.show()
    return fig
    

def make_loss_plots(aux_out: IntegratorAuxOutput, 
                    loss_out: LossOutput, 
                    title: str = "Losses"):
    
    fig, axes = plt.subplots(1, len(loss_out.attrs), figsize=(4*len(loss_out.attrs), 5))
    fig.suptitle(title)
    for ax, attr in zip(axes, loss_out.attrs):
        ax.set_title(attr)
        ax.set_xlabel('t')
        ax.grid(True, alpha=0.3)
        
        for i in range(aux_out.xs.shape[0]):
            ax.plot(aux_out.ts[:-1], getattr(loss_out, attr)[i])
            
    plt.show()
    return fig


def change_rom_param_type(rom: NNDoubleIntegratorROM, 
                          to_param_type: nnx.Param | nnx.Variable,
                          component_list: list[str] = ["nn_encoder", "nn_decoder", 
                                                       "nn_fy", "nn_gy", "nn_fz", "nn_psi"]):
    """
    Change the type of specified parameters to nnx.Variable instead of nnx.Param.
    This prevents them from being updated during training.
    - If to_param_type is nnx.Param, the parameters will be unfrozen.
    - If to_param_type is nnx.Variable, the parameters will be frozen.
    
    """
    for module_name in component_list:
        if hasattr(rom, module_name):
            module = getattr(rom, module_name)
            # Convert all Param variables to regular Variables (frozen)
            for attr_name in ['kernel', 'bias']:
                if hasattr(module, attr_name):
                    param = getattr(module, attr_name)
                    if isinstance(param, nnx.Param):
                        # Replace with a frozen Variable
                        setattr(module, attr_name, to_param_type(param.value))
                        
                        
def freeze_parameters(rom: NNDoubleIntegratorROM, 
                      component_list: list[str] = ["nn_encoder", "nn_decoder", 
                                                   "nn_fy", "nn_gy", "nn_fz", "nn_psi"]):
    change_rom_param_type(rom, nnx.Variable, component_list)
    

def unfreeze_parameters(rom: NNDoubleIntegratorROM, 
                        component_list: list[str] = ["nn_encoder", "nn_decoder", 
                                                     "nn_fy", "nn_gy", "nn_fz", "nn_psi"]):
    change_rom_param_type(rom, nnx.Param, component_list)


def get_batch(rom_expert: DoubleIntegratorROM, 
              integrator: Integrator,
              cfg_data: CfgData = CfgData(),
              rng: jax.random.PRNGKey = jax.random.PRNGKey(0),
              init_states_only: bool = False):
    
    '''
    Get a batch of expert data for training.
    NOTE: We might not need this if decoding is not needed. For now assume we don't need this.
    '''
    
    key_x, key_u = jax.random.split(rng, 2)
    dim_x, dim_u = rom_expert.cfg_rom.dim_x, rom_expert.cfg_rom.dim_u
    batch_size = cfg_data.batch_size
    box_width = cfg_data.box_width
    
    x0s = jax.random.uniform(key_x, (batch_size, dim_x), 
                             minval=-box_width, maxval=box_width)
    if init_states_only:
        return IntegratorOutput(xs=x0s[:, None, :], us=jnp.zeros((batch_size, 1, dim_u)))
    
    
    n_policy_v = int(cfg_data.batch_size * cfg_data.policy_ratios[0])
    x0s_policy_v = x0s[:n_policy_v]
    x0s_policy_noop = x0s[n_policy_v:]

    ret_policy_v = integrator.apply(x0s_policy_v, rom_expert)
    ret_policy_noop = integrator.apply(x0s_policy_noop, rom_expert, 
                                       policy_fun=lambda y, z: jnp.zeros((dim_u,)))
    
    
    return IntegratorOutput(
        xs = jnp.concatenate([ret_policy_v.xs, ret_policy_noop.xs], axis=0),
        us = jnp.concatenate([ret_policy_v.us, ret_policy_noop.us], axis=0)
    )


@catch_keyboard_interrupt("Training interrupted by user")
def train(rom_nn: NNDoubleIntegratorROM, 
          rom_expert: DoubleIntegratorROM,
          integrator: Integrator, 
          cfg_train: CfgTrain, 
          cfg_data: CfgData, 
          cfg_loss: CfgLoss,
          rng: jax.random.PRNGKey):
    
    total_steps = cfg_train.num_epochs * cfg_train.num_batches
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.1 * cfg_train.lr, 
        peak_value=cfg_train.lr,
        warmup_steps=int(0.1 * total_steps), 
        decay_steps=total_steps,
    )
    if cfg_train.enable_lr_schedule:
        tx = optax.adam(lr_schedule)
    else:
        tx = optax.adam(cfg_train.lr)
    opt = nnx.Optimizer(rom_nn, tx, wrt=nnx.Param)
    rngs = jax.random.split(rng, cfg_train.num_batches)
    
    
    @nnx.jit
    def step(model: NNDoubleIntegratorROM, batch: IntegratorOutput):
    
        def loss_fn(_m: NNDoubleIntegratorROM):
            int_out  = integrator.apply(batch.xs[:,0,:], rom=_m)     
            loss_out = integrator.compute_loss(int_out, batch, rom=_m, cfg_loss=cfg_loss)
            aux_out = integrator.post_apply(int_out, rom=_m)
            return jnp.mean(loss_out.total), (int_out, aux_out, loss_out)
        
        return nnx.value_and_grad(loss_fn, has_aux=True)(model)

    global_step = 0
    epoch_losses = []
    for i in (pbar := tqdm(range(cfg_train.num_epochs))):
        
        batch_losses = []
        for b in range(cfg_train.num_batches):
        
            batch = get_batch(rom_expert, integrator, cfg_data, rngs[b], 
                              init_states_only= not cfg_loss.supervised)
            (loss, (int_out, aux_out, loss_out)), grads = step(rom_nn, batch)
            
            opt.update(grads=grads)
            batch_losses.append(loss)
            global_step += 1
            
        epoch_loss = jnp.mean(jnp.array(batch_losses))
        epoch_losses.append(epoch_loss)
        lr = lr_schedule(global_step) if cfg_train.enable_lr_schedule else cfg_train.lr
        pbar.set_postfix({"loss": f"{epoch_loss:.2e}", "lr": f"{lr:.2e}"})
        
        if i % (cfg_train.num_epochs // cfg_train.num_logs) == 0:
            pass
        
    print("\nParams after training:")
    pprint(rom_nn.get_nn_params(), width=120)
    
    make_traj_plots(rom_nn, int_out, aux_out)
    make_loss_plots(aux_out, loss_out)
    
    plt.plot(epoch_losses)
    plt.title("Losses")
    plt.show()
    
    
    return loss_out # last batch loss


if __name__ == "__main__":

    cfg_rollout = CfgRollout()
    cfg_rom = CfgDIROM()
    cfg_train = CfgTrain()
    cfg_data = CfgData()

    ts = jnp.arange(cfg_rollout.t0, cfg_rollout.t1 + cfg_rollout.dt, cfg_rollout.dt)
    rng = jax.random.PRNGKey(42)
    rom_nn = NNDoubleIntegratorROM(cfg_rom=cfg_rom)
    rom_expert = DoubleIntegratorROM(cfg_rom=cfg_rom)

    rom_nn.set_nn_params({
        "nn_encoder": {"kernel": jnp.array([[0., 1.], [1., 0.]])},
        "nn_decoder": {"kernel": jnp.array([[0., 1.], [1., 0.]])},
        "nn_gy":      {"kernel": jnp.array([[0.]]), "bias": jnp.array([1.])},
        # "nn_fy":      {"kernel": jnp.array([[0.]])},                      
        # "nn_fz":      {"kernel": jnp.array([[1.], [0.]])},                # ż = y
        # "nn_psi":      {"kernel": jnp.array([[-0.4]])}, # on purpose give it a incorrect value.
    })

    print("Before training:")
    pprint(rom_nn.get_nn_params(), width=120)

    integrator = Integrator(
        solver=partial(dfx.diffeqsolve, solver=dfx.Tsit5()),
        ts=ts
    )


    # freeze_parameters(rom_nn, ["nn_encoder", "nn_decoder"])
    if (train_ae_only:=False):
        cfg_loss_ae = CfgLoss(autoencoder=1.0, nondegenerate_enc=0.0, supervised=True)
        train(rom_nn, rom_expert, integrator, cfg_train, cfg_data, cfg_loss_ae, rng)

    if (train_dyn_only:=True):
        # cfg_loss_dyn = CfgLoss(autoencoder=0.0, y_proj=1.0, z_proj=1.0, nondegenerate_enc=1.0, supervised=True)
        cfg_loss_dyn = CfgLoss(autoencoder=1.0, y_proj=0.0, z_proj=0.0, nondegenerate_enc=0.0, supervised=True)
        freeze_parameters(rom_nn, ["nn_encoder", "nn_decoder"])
        train(rom_nn, rom_expert, integrator, cfg_train, cfg_data, cfg_loss_dyn, rng)

    if (train_zd_policy_only:=False):
        cfg_loss_zd = CfgLoss(stable_m=1.0, invari_m=1.0, supervised=True)
        freeze_parameters(rom_nn, ["nn_encoder", "nn_decoder", "nn_fz", "nn_gy", "nn_fy"])
        train(rom_nn, rom_expert, integrator, cfg_train, cfg_data, cfg_loss_zd, rng)
        
    print("After training:")
    pprint(rom_nn.get_nn_params(), width=120)