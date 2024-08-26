import jax.random
from train import wandb_model_load
from hydra.utils import instantiate
import wandb
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp

plt.rc('font', family='serif')

'''Visualizes the Policy, and a sample trajectory'''


def main():
    # exp_name = 'noelcs/zero_dynamics_policies/fdudsriz'
    # exp_name = 'wdc3iii/zero_dynamics_policies/fu8f9dse'
    # exp_name = 'wdc3iii/zero_dynamics_policies/bfctdz0z'
    exp_name = 'wdc3iii/zero_dynamics_policies/ry2eihqa'

    model_tag = os.path.basename(exp_name)

    model_name = f'{exp_name}_model:v0'
    api = wandb.Api()
    cfg, state = wandb_model_load(api, model_name)

    new_rng, split = jax.random.split(state.rng)
    state = state.replace(rng=new_rng)

    # Instantiate
    pretrain_name = cfg.pretrained_net
    model_name = f'{pretrain_name}_model:best'
    api = wandb.Api()
    cfg_pre, state_pre = wandb_model_load(api, model_name)

    integrator_pre = instantiate(cfg_pre.loss.integrator)
    psi_policy_network = integrator_pre.mlp
    policy = instantiate(cfg.psi_policy)(mlp=psi_policy_network)
    integrator = instantiate(cfg.loss.integrator)(psi_policy=policy)
    vpi, loss_fn = instantiate(cfg.loss)(integrator=integrator)

    params = vpi.init(split, 0, jnp.array([0, 0, 0, 0, 0, 0, 0.3445, 0, 0, 1.7]))  # Hack for training the hopper)

    params['psi_policy']['params'] = state.params['psi_policy']['params']

    # Sample the policy

    # setup rng
    rng = jax.random.PRNGKey(cfg.seed)
    rng, split = jax.random.split(rng)

    zs = jax.random.uniform(split, shape=(10000, 4), minval=-1, maxval=1)
    zs = zs * jnp.array([[2, 2, 0.8, 0.8]])
    eta_d = jax.vmap(lambda z_: policy.apply(params['psi_policy'], z_, method='psi'))(zs)
    plt.figure()
    ax = plt.subplot(121)
    # im = plot_2D_irregular_heatmap(zs[:, 0], zs[:, 2], eta_d[:, 0],ax=ax)
    # plot_scatter_value(zs[:, 0], zs[:, 2], eta_d[:, 0])
    im = ax.scatter(zs[:, 0], zs[:, 2], c=eta_d[:, 0], s=10)
    plt.colorbar(im, ax=ax)
    plt.xlabel('x')
    plt.ylabel('dotx')
    plt.title('theta1_d')
    ax = plt.subplot(122)
    # im = plot_2D_irregular_heatmap(zs[:, 1], zs[:, 3], eta_d[:, 1],ax=ax)
    # plot_scatter_value(zs[:, 1], zs[:, 3], eta_d[:, 1])
    im = ax.scatter(zs[:, 1], zs[:, 3], c=eta_d[:, 1], s=10)
    plt.colorbar(im, ax=ax)
    plt.xlabel('y')
    plt.ylabel('doty')
    plt.title('theta2_d')
    plt.show()

    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={30}"

    # @partial(jax.jit, backend="cpu")
    # def train_step(batch):
    #     batch = batch.reshape((30, 1, -1))
    #     return jax.pmap(lambda z: loss_fn(params, z), in_axes=(0,))(batch)

    zs_losses = zs[:30, :]
    # losses = train_step(zs_losses)
    losses = jax.vmap(loss_fn, in_axes=(None, 0))(params, zs_losses.reshape((30, 1, -1)))
    plt.figure()
    ax = plt.subplot(121)
    # im = plot_2D_irregular_heatmap(zs[:, 0], zs[:, 2], eta_d[:, 0],ax=ax)
    # plot_scatter_value(zs[:, 0], zs[:, 2], eta_d[:, 0])
    im = ax.scatter(zs_losses[:, 0], zs_losses[:, 2], c=losses, s=10)
    plt.colorbar(im, ax=ax)
    plt.xlabel('x')
    plt.ylabel('dotx')
    plt.title('loss')
    ax = plt.subplot(122)
    # im = plot_2D_irregular_heatmap(zs[:, 1], zs[:, 3], eta_d[:, 1],ax=ax)
    # plot_scatter_value(zs[:, 1], zs[:, 3], eta_d[:, 1])
    im = ax.scatter(zs_losses[:, 1], zs_losses[:, 3], c=losses, s=10)
    plt.colorbar(im, ax=ax)
    plt.xlabel('y')
    plt.ylabel('doty')
    plt.title('loss')
    plt.show()


    # tf_model = tfZeroInvariantMLP(
    #     policy.mlp.mlp,
    #     params['params']['mlp']['mlp'],
    #     4,
    #     cfg_pre.loss.integrator.mlp.mlp.activation._target_,
    #     clip=cfg.u_max
    # )
    #
    # # Validate tf_model vs flax model
    # zs = jax.random.uniform(split, shape=(10000, 4), minval=-1, maxval=1)
    # eta_d_flax = jax.vmap(lambda z_: policy.apply(params, z_, method='psi'))(zs)
    #
    # eta_d_tf = tf_model(np.array(zs))
    # err = eta_d_flax - eta_d_tf
    # print("jax2onnx max error: ", np.max(np.abs(np.array(err))))
    #
    # onnx_model = tf2onnx.convert.from_keras(tf_model)
    # onnx.save(onnx_model[0], os.getcwd()+f'/../models/trained_model_{model_tag}.onnx')

    print("success!")


if __name__ == "__main__":
    import os
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={128}"
    from jax import config

    config.update("jax_enable_x64", True)
    from jax.experimental.compilation_cache import compilation_cache as cc

    cc.initialize_cache("/tmp/jax_cache")

    main()
