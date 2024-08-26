import jax.random
from simulator import ClosedLoopIntegrator
from train import wandb_model_load
from hydra.utils import instantiate
import wandb
import jax
import matplotlib.pyplot as plt
from plots import plot_2D_irregular_heatmap
import jax.numpy as jnp
from functools import partial
import diffrax as de


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

'''Visualizes the Policy, and a sample trajectory'''


def main():
    exp_name = 'wdc3iii/zero_dynamics_policies/8m7nppkv'
    # exp_name = 'wdc3iii/zero_dynamics_policies/1grgzbnl'
    model_name = f'{exp_name}_model:best'
    api = wandb.Api()
    cfg, state = wandb_model_load(api, model_name)

    new_rng, split = jax.random.split(state.rng)
    state = state.replace(rng=new_rng)

    # Instantiate the Policy
    integrator = instantiate(cfg.loss.integrator)

    psi_policy = integrator.psi_policy
    psi_policy.K = jnp.array([[5, 2*jnp.sqrt(5)]])
    _, psi_params = psi_policy.init_with_output(split, 0, jnp.zeros((4,)))
    psi_params = state.params["psi_policy"]
    new_rng, split = jax.random.split(new_rng)
    lqr_policy = integrator.control_policy
    lqr_params = lqr_policy.init(split, 0, jnp.zeros((2,)), jnp.zeros((2,)))
    lqr_params = state.params["control_policy"]

    # Sample the policy
    new_rng, split = jax.random.split(new_rng)
    # zs = jax.random.uniform(split, shape=(500, 2), minval=-5, maxval=5)
    zs = jax.random.uniform(split, shape=(500, 2), minval=-1, maxval=1)
    eta_d = jax.vmap(lambda z: psi_policy.apply(psi_params, z, method="eta_d"), in_axes=(0,))(zs)

    # Plot
    plt.figure()
    ax = plt.subplot(121)
    plot_2D_irregular_heatmap(zs[:, 0], zs[:, 1], eta_d[:, 0], ax=ax)
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.title('eta1_d')
    ax = plt.subplot(122)
    plot_2D_irregular_heatmap(zs[:, 0], zs[:, 1], eta_d[:, 1], ax=ax)
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.title('eta2_d')
    plt.show()

    # Examine closed loop behavior
    T = 20
    ts = jnp.linspace(0, T, T * 100)
    eval_integrator = ClosedLoopIntegrator(
        policy=psi_policy,
        robot=integrator.dyn,
        ts=ts,
        de_solver=partial(
            de.diffeqsolve,
            solver=de.Tsit5(),
            stepsize_controller=de.PIDController(rtol=1e-6, atol=1e-12, dtmin=1e-5)
        )
    )

    eval_params = eval_integrator.init(split, ts[0], jnp.zeros((4,)))
    eval_params["policy"] = psi_params
    x0 = jnp.array([0, 0.6, 0, 0])
    int_out = eval_integrator.apply(eval_params, x0)

    plt.figure()
    plt.subplot(211)
    plt.plot(ts, int_out.xs)
    plt.xlabel('t')
    plt.legend(['x1', 'x2', 'dx1', 'dx2'])
    plt.ylabel('state')
    plt.subplot(212)
    plt.plot(ts[:-1], int_out.us)
    plt.xlabel('t')
    plt.ylabel('u')
    plt.show()
    print("success?")


if __name__ == "__main__":
    import os
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_ENABLE_X64"] = "true"
    from jax import config

    config.update("jax_enable_x64", True)
    from jax.experimental.compilation_cache import compilation_cache as cc

    cc.initialize_cache("/tmp/jax_cache")

    main()
