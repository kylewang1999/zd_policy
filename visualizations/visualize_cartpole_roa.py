import pickle
import jax.random

import robots
from simulator import ClosedLoopIntegrator
from train import wandb_model_load
from hydra.utils import instantiate
import wandb
import jax
import numpy as np
import scipy
import matplotlib.pyplot as plt
from utils import flat_meshgrid
from plots import plot_2D_irregular_heatmap, plot_scatter_value
import jax.numpy as jnp
from functools import partial
import diffrax as de
from control import RelaxedZDynPolicyPD

# ACTUATED_CTRL = "PD"
# ACTUATED_CTRL = "LQR"
ACTUATED_CTRL = "LQR_ZDP"
# ACTUATED_CTRL = "LQR_ZDP_FBL"
# ACTUATED_CTRL = "FBL"
# ROBOT = 'nominal'
# ROBOT = 'lazy'
ROBOT = 'unstable_base'
# ROBOT = 'unstable_pole'
N=129
# N=32
gamma = 1.0 # cart cubic damping
# gamma=0.1 # lazy input scaling
# gamma = None
threshold = 1e-3
# K_FB = [1000, 632.456]
# K_FB = jnp.array([20, 5.65685])
K_FB = jnp.array([20, 8.9])
# K_FB = [200, 63.2456]

# K = jnp.array([10.9545, 30])
# Nominal Gains
# K = jnp.array([30,10.9545])
# K = jnp.array([6,4.89898])
# K = jnp.array([8,5.65685])
K = jnp.array([2,2.82843])
# Lazy Gains
# K = jnp.array([400,40])
# Unstable Base Gains
# K = jnp.array([40,3.4641])

plt.rc('font', family='serif')

'''Visualizes the Policy, and a sample trajectory'''


def main():
    # exp_name = 'ivan/zero_dynamics_policies/h24i2u8g'
    # exp_name = 'ivan/zero_dynamics_policies/p86d6qq0'
    exp_name = 'ivan/zero_dynamics_policies/lxk5gjot'
    # exp_name = 'ivan/zero_dynamics_policies/2zomeeix'
    # exp_name = 'ivan/zero_dynamics_policies/l50d69bj'
    # exp_name = 'ivan/zero_dynamics_policies/jufox6qc'
    # exp_name = 'ivan/zero_dynamics_policies/l50d69bj'

    model_name = f'{exp_name}_model:best'

    # if os.path.exists(f'../data/{os.path.basename(exp_name)}.pkl'):
    #     with open(f'../data/{os.path.basename(exp_name)}.pkl', 'rb') as f:
    #         data = pickle.load(f)
    #         cfg = data['cfg']
    #         state = data['state']
    # else:
    api = wandb.Api()
    cfg, state = wandb_model_load(api, model_name)
    with open(f'../data/{os.path.basename(exp_name)}.pkl', 'wb') as f:
        pickle.dump({'cfg': cfg, 'state': state}, f)

    new_rng, split = jax.random.split(state.rng)
    state = state.replace(rng=new_rng)

    # Instantiate the Policy
    if ACTUATED_CTRL=="PD":
        psi_policy_fbl = instantiate(cfg.loss.integrator.psi_policy)
        psi_policy = RelaxedZDynPolicyPD(
            mlp=psi_policy_fbl.mlp,
            dyn=psi_policy_fbl.dyn,
            angle_rep=psi_policy_fbl.angle_rep,
            output_bounds=psi_policy_fbl.output_bounds,
            K=K
        )
    elif ACTUATED_CTRL=="FBL":
        psi_policy_cfg = cfg.loss.integrator.psi_policy
        psi_policy_cfg.K.object = K_FB
        psi_policy = instantiate(cfg.loss.integrator.psi_policy)
    elif ACTUATED_CTRL=="LQR_ZDP":
        from control import InvariantLinearController
        psi_policy_fbl = instantiate(cfg.loss.integrator.psi_policy)
        psi_policy = RelaxedZDynPolicyPD(
            mlp=InvariantLinearController(K=-np.array(
                [[-2.2948, -1.5337], [11.6199, 1.7723]])),
            dyn=psi_policy_fbl.dyn,
            angle_rep=psi_policy_fbl.angle_rep,
            output_bounds=psi_policy_fbl.output_bounds,
            K=K
        )
    elif ACTUATED_CTRL=="LQR_ZDP_FBL":
        from control import InvariantLinearController, RelaxedZDynPolicy

        psi_policy_fbl = instantiate(cfg.loss.integrator.psi_policy)
        psi_policy = RelaxedZDynPolicy(
            mlp=InvariantLinearController(K=-np.array(
                [[-2.2948, -1.5337], [11.6199, 1.7723]])),
            dyn=psi_policy_fbl.dyn,
            angle_rep=psi_policy_fbl.angle_rep,
            output_bounds=psi_policy_fbl.output_bounds,
            K=K_FB
        )
    elif ACTUATED_CTRL=="LQR":
        from control import LinearController
        from robots import Cartpole, UnstableCartpole
        dyn = UnstableCartpole()
        t0 = jnp.array(0.0)
        x0 = jnp.zeros((dyn.state_dim,))
        u0 = jnp.zeros((dyn.action_dim,))
        dyn_params = dyn.init(jax.random.PRNGKey(0), t0, x0, u0)
        A, B = jax.jacobian(dyn.apply, argnums=(2, 3))(dyn_params, t0, x0, u0)
        Q = jnp.eye(dyn.state_dim)
        R = 1e-2 * jnp.eye(dyn.action_dim)
        P = jnp.array(scipy.linalg.solve_continuous_are(A, B, Q, R))
        LQR_K = jnp.linalg.solve(R, B.T @ P)
        print(f"K: {LQR_K}")
        psi_policy = LinearController(K=LQR_K)
    else:
        raise ValueError(f"Invalid controller type {ACTUATED_CTRL}")

    state_n = 4
    params = psi_policy.init(split, 0, jnp.zeros((state_n,)))
    params['params'] = state.params['psi_policy']['params']

    # Sample the policy

    # setup rng
    rng = jax.random.PRNGKey(cfg.seed)
    rng, split = jax.random.split(rng)
    dataset = instantiate(cfg.dataset)(rng=split)
    zs = flat_meshgrid(dataset.lower_bound[0],
                       dataset.upper_bound[0],
                       dataset.lower_bound[1],
                       dataset.upper_bound[1],
                       N=N)
    if ACTUATED_CTRL not in ["LQR", "LQR_ZDP","LQR_ZDP_FBL"]:
        eta_d = jax.vmap(lambda z_: psi_policy.apply(params, z_, method="eta_d"), in_axes=(0,))(zs)

        # Plot
        plt.figure()
        ax = plt.subplot(121)
        im = ax.scatter(zs[:, 0], zs[:, 1], c=eta_d[:, 0], s=10)
        plt.colorbar(im, ax=ax)
        plt.xlabel('Nq = th')
        plt.ylabel('NDdq ~ dth')
        plt.title('eta1 d')
        ax = plt.subplot(122)
        im = ax.scatter(zs[:, 0], zs[:, 1], c=eta_d[:, 1], s=10)
        plt.colorbar(im, ax=ax)
        # plt.colorbar(im, ax=ax)
        plt.xlabel('Nq = th')
        plt.ylabel('NDdq ~ dth')
        plt.title('eta2 d')
        plt.show()

    # Close that loop
    # T = 5
    T = 25
    ts = jnp.linspace(0, T, int(T * 100))
    from robots import LazyCartpole
    if ROBOT == 'nominal':
        from robots import Cartpole
        robot = Cartpole()
    elif ROBOT == 'lazy':
        assert gamma is not None
        robot = LazyCartpole(gamma=gamma)
    elif ROBOT == 'unstable_base':
        assert gamma is not None
        robot = robots.UnstableBaseCartpole(gamma=gamma)
    elif ROBOT == 'unstable_pole':
        assert gamma is not None
        robot = robots.UnstableCartpole(gamma=gamma)
    else:
        raise ValueError(f"Invalid robot {ROBOT}")
    eval_integrator = ClosedLoopIntegrator(
        policy=psi_policy,
        robot=robot,
        ts=ts,
        de_solver=partial(
            de.diffeqsolve,
            solver=de.Tsit5(),
            stepsize_controller=de.PIDController(rtol=1e-6, atol=1e-12, dtmin=1e-5),
            discrete_terminating_event=de.DiscreteTerminatingEvent(lambda state, **kwargs: jnp.linalg.norm(psi_policy.apply(params, 0, state.y)) > 1e4)
        )
    )
    eval_params = eval_integrator.init(split, ts[0], jnp.zeros((4,)))
    eval_params["policy"] = params

    if ACTUATED_CTRL not in ["LQR", "LQR_ZDP", "LQR_ZDP_FBL"]:
        # One initial condition
        # x0 = jnp.array([0, 2, 8, 0])
        z0 = jnp.array([3, 0])
        n0 = psi_policy.apply(params, z0, method='eta_d')
        x0 = psi_policy.dyn.phi_inv(0, n0, z0)
        int_out = eval_integrator.apply(eval_params, x0)
        ns, zs = jax.vmap(lambda x_: psi_policy.dyn.phi(0, x_))(int_out.xs)
        nds = jax.vmap(lambda z_: psi_policy.apply(params, z_, method='eta_d'))(zs)
        err = ns - nds
        plt.figure()
        plt.subplot(311)
        plt.plot(ts, int_out.xs)
        plt.xlabel('t')
        plt.legend(['x1', 'x2', 'dx1', 'dx2'])
        plt.ylabel('state')
        plt.subplot(312)
        plt.plot(ts[:-1], int_out.us)
        plt.xlabel('t')
        plt.ylabel('u')
        plt.subplot(313)
        plt.plot(ts, err)
        plt.xlabel('t')
        plt.ylabel('err')
        plt.show()

    @jax.vmap
    def bint(xi):
        x0_ = jnp.array([0, xi[0], 0, xi[1]])
        int_out_ = eval_integrator.apply(eval_params, x0_)
        return int_out_

    # xis = flat_meshgrid(-3, 3, -15, 15, N=101)
    xis = flat_meshgrid(-3.15, 3.15, -10, 10, N=N)
    bints = bint(xis)
    final_norms = jnp.linalg.norm(bints.xs[:, -1, :], axis=-1)
    # converged = xis.at[final_norms > 0.01, :].set(jnp.nan)
    converged = xis[final_norms < threshold]
    if ACTUATED_CTRL in ["LQR", "LQR_ZDP", "LQR_ZDP_FBL"]:
        exp_name = f"{ACTUATED_CTRL}_{ROBOT}"
    scipy.io.savemat(f'../data/cartpole_roa_{os.path.basename(exp_name)}.mat',
                     {f'roa_{os.path.basename(exp_name)}': np.array(converged),
                      f'xs_{os.path.basename(exp_name)}': np.array(bints.xs),
                      f'us_{os.path.basename(exp_name)}': np.array(bints.us),
                      f'ts_{os.path.basename(exp_name)}': np.array(ts),

                     })
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    plot_2D_irregular_heatmap(xis[:, 0], xis[:, 1], final_norms, log_z=True,
                              ax=axs[0])
    axs[0].set_xlim([-1.5, 1.5])
    axs[0].set_ylim([-7, 7])
    axs[1].scatter(converged[:, 0], converged[:, 1], 10, 'r', alpha=0.4)
    axs[1].set_xlim([-1.5, 1.5])
    axs[1].set_ylim([-7, 7])
    track_label = ("PD", K) if ACTUATED_CTRL else ("FBL", K_FB)
    plt.title(f"RoA {ROBOT}, gamma: {gamma}, {track_label[0]}: {track_label[1]}")
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
