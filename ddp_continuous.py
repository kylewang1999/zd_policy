import flax.struct
import jax
import jax.numpy as jnp
from typing import Callable
from jax import custom_jvp

@flax.struct.dataclass
class DDPVars:
    xk: jnp.array        # N x n
    uk: jnp.array        # (N-1) x m
    xk_bar: jnp.ndarray  # N x n
    uk_bar: jnp.ndarray  # N x m
    Vxx: jnp.ndarray     # n x n
    vx: jnp.ndarray      # n x 1
    Qxx: jnp.ndarray     # N x n x n
    Quu: jnp.ndarray     # N x m x m
    Qux: jnp.ndarray     # N x m x n
    qx: jnp.ndarray      # N x n
    qu: jnp.ndarray      # N x m


@flax.struct.dataclass
class DDPQuadCost:
    Qk: jnp.array     # N x n x n
    R: jnp.array      # m x m
    x_des: jnp.array  # N x n

    def __call__(self, i, x, u):
        diff = x - self.x_des
        return 0.5*(diff.T @ self.Qk[i] @ diff + u.T @ self.R @ u)

    def horizon_cost(self, x, u):
        diff = x - self.x_des[None]
        return 0.5 * (jnp.sum(diff @ self.Qk * diff) + jnp.sum(u @ self.R * u))


@flax.struct.dataclass
class LQRDynamics:
    Ak: jnp.array  # (N-1) x n x n
    Bk: jnp.array  # (N-1) x n x m
    Ck: jnp.array  # (N-1) x n


def linearize_dynamics_function(dyn, dt):
    """
    Linearize dynamics about the current state and discretize
    """
    def linearize(x, u):
        A, B = jax.jacfwd(dyn, argnums=(0, 1))(x, u)
        d_n = A.shape[0]
        d_m = B.shape[1]
        xdot = dyn(x, u)
        C = xdot - A @ x - B @ u
        dss = jax.scipy.linalg.expm(jnp.vstack([
            jnp.hstack([A, B, C[:, None]]),
            jnp.zeros((d_m + 1, d_m + d_n + 1))
        ]) * dt)
        return LQRDynamics(Ak=dss[:d_n, :d_n], Bk=dss[:d_n, d_n:d_n+d_m], Ck=dss[:d_n, -1])
    linearize_func = jax.vmap(linearize)
    linearize_func = jax.jit(linearize_func)
    return linearize_func


def ddp(traj: DDPVars, cost: DDPQuadCost, lin_dyn: LQRDynamics, dynamics: Callable):
    N = traj.xk.shape[0]  # num steps
    dc = jax.grad(cost, argnums=(1, 2))
    d2c = jax.hessian(cost, argnums=(1, 2))

    # Set terminal conditions
    Vxx = d2c(N, traj.xk_bar[N], jnp.zeros_like(traj.uk_bar[-1]))[0][0]
    vx = dc(N, traj.xk_bar[N], jnp.zeros_like(traj.uk_bar[-1]))[0]

    traj = traj.replace(Vxx=Vxx, vx=vx)

    def backward_step(j, traj_):
        # NOTE: at each iteration, changes to Qxx, ect. are nearly symmetric (up to say 1e-6), but errors compound,
        # resulting in nearly 1e4 asymmetry at the end. Enforcing symmetry of these at each iteration is a minimal
        # adjustment, enforces symmetry over whole range.
        i = N - 2 - j
        dct = dc(i, traj.xk_bar[i], traj_.uk_bar[i])
        d2ct = d2c(i, traj.xk_bar[i], traj_.uk_bar[i])

        qx = dct[0] + lin_dyn.Ak[i].T @ traj_.vx
        qu = dct[1] + lin_dyn.Bk[i].T @ traj_.vx

        Qxx = d2ct[0][0] + lin_dyn.Ak[i].T @ traj_.Vxx @ lin_dyn.Ak[i]
        Qxx = (Qxx.T + Qxx) / 2  # Enforce symmetry
        Quu = d2ct[1][1] + lin_dyn.Bk[i].T @ traj_.Vxx @ lin_dyn.Bk[i]
        Quu = (Quu.T + Quu) / 2  # Enforce symmetry
        Qux = d2ct[1][0] + lin_dyn.Bk[i].T @ traj_.Vxx @ lin_dyn.Ak[i]

        Quu_inv = jnp.linalg.inv(Quu)
        Vxx = Qxx - Qux.T @ Quu_inv @ Qux
        Vxx = (Vxx.T + Vxx) / 2  # Enforce symmetry
        vx = qx - Qux.T @ (Quu_inv @ qu)

        return traj_.replace(Vxx=Vxx,
                             vx=vx,
                             Qxx=traj_.Qxx.at[i].set(Qxx),
                             Quu=traj_.Quu.at[i].set(Quu),
                             Qux=traj_.Qux.at[i].set(Qux),
                             qx=traj_.qx.at[i].set(qx),
                             qu=traj_.qu.at[i].set(qu))

    traj = jax.lax.fori_loop(0, N-1, backward_step, traj)
    traj_bar_cost = cost.horizon_cost(traj.xk_bar, traj.uk_bar)
    decay_factor = 0.5
    alpha_0 = 1.0

    def forward_step(i, traj_, alpha_):
        du = jnp.linalg.inv(traj_.Quu[i]) @ traj_.qu[i]

        u = traj_.uk_bar[i] + alpha_ * du - jnp.linalg.inv(traj_.Quu[i]) @ traj_.Qux[i] @ (traj_.xk[i] - traj_.xk_bar[i])
        u = jnp.clip(u, -1e4, 1e4)
        xfp = dynamics(traj_.xk[i], u)
        xfp = jnp.clip(xfp, -1e2, 1e2)
        return traj_.replace(xk=traj_.xk.at[i+1].set(xfp), uk=traj_.uk.at[i].set(u))

    def cond_fun(fc):
        traj_, alpha_ = fc
        new_cost = cost.horizon_cost(traj_.xk, traj_.uk)
        nan_check = jnp.isnan(new_cost)
        true_cost_larger = new_cost >= traj_bar_cost
        alpha_reasonable = alpha_ > 1e-6
        return (true_cost_larger & alpha_reasonable) | (alpha_ == alpha_0 / decay_factor) | nan_check

    def line_search_step(fc):
        traj_, alpha_ = fc
        alpha_ = alpha_ * decay_factor
        new_traj = jax.lax.fori_loop(0, N-1, lambda i, _traj_: forward_step(i, _traj_, alpha_), traj_)
        return new_traj, alpha_

    traj, alpha = jax.lax.while_loop(cond_fun, line_search_step, (traj, alpha_0 / decay_factor))
    # jax.debug.print("{x},", x=alpha)
    return traj


def ddp_from_nz0_func(dyn, N, dt, ddp_iter, warm_start_policy, whole_traj=False):
    lin_func = linearize_dynamics_function(lambda x, u: dyn.dynamics(x, u), dt)
    cost_params = get_cost_params(N, dyn.d_m, dyn.x_star())

    def warm_start(i, ws):
        u = warm_start_policy(0, *dyn.phi(0, ws.xk_bar[i]))[None]
        u = jnp.clip(u, -1e4, 1e4)
        xk = ws.xk_bar[i] + dt * dyn.dynamics(ws.xk_bar[i], u)
        xk = jnp.clip(xk, -1e2, 1e2)
        return ws.replace(
            xk_bar=ws.xk_bar.at[i+1].set(xk),
            uk_bar=ws.uk_bar.at[i].set(u)
        )

    def ddp_iter_body(_, traj):
        lin_dyn = lin_func(traj.xk_bar[:-1], traj.uk_bar)
        traj = ddp(traj=traj,
                   cost=cost_params,
                   lin_dyn=lin_dyn,
                   dynamics=lambda x, u: x + dt * dyn.dynamics(x, u))
        return traj.replace(xk_bar=traj.xk_bar.at[:].set(traj.xk),
                            uk_bar=traj.uk_bar.at[:].set(traj.uk))

    def ddp_from_nz0_traj(nz0):
        x0 = dyn.phi_inv(0, nz0[:2], nz0[2:])
        traj = DDPVars(
            xk=jnp.repeat(x0[None], N, axis=0),
            xk_bar=jnp.repeat(x0[None], N, axis=0),
            uk=jnp.zeros((N - 1, dyn.d_m)),
            uk_bar=jnp.zeros((N - 1, dyn.d_m)),
            Vxx=jnp.zeros((dyn.d_n, dyn.d_n)),
            vx=jnp.zeros((dyn.d_n,)),
            Qxx=jnp.zeros((N - 1, dyn.d_n, dyn.d_n)),
            Quu=jnp.zeros((N - 1, dyn.d_m, dyn.d_m)),
            Qux=jnp.zeros((N - 1, dyn.d_m, dyn.d_n)),
            qx=jnp.zeros((N - 1, dyn.d_n)),
            qu=jnp.zeros((N - 1, dyn.d_m)),
        )
        traj = jax.lax.fori_loop(0, N - 1, warm_start, traj)
        traj = jax.lax.fori_loop(0, ddp_iter, ddp_iter_body, traj)
        return traj

    @custom_jvp
    def ddp_from_nz0(nz0):
        # traj = ddp_from_nz0_traj(nz0)
        # # return jnp.where(jnp.linalg.norm(traj.xk_bar[-1]) < 0.1, traj.uk_bar[0], warm_start_policy(0, traj.xk_bar[0]))
        # return jnp.where(False, traj.uk_bar[0], warm_start_policy(0, traj.xk_bar[0]))
        # x0 = dyn.phi_inv(0, *dyn.nz_split(nz0))
        return warm_start_policy(0, *dyn.nz_split(nz0))

    @ddp_from_nz0.defjvp
    def u_star_jvp(primals, tangents):
        nz0, = primals
        # x0 = dyn.phi_inv(0, *dyn.nz_split(nz0))
        dxdnz = jax.jacfwd(lambda nz: dyn.phi_inv(0, *dyn.nz_split(nz)))(nz0)
        v, = tangents
        # Solve OCP
        traj = ddp_from_nz0_traj(nz0)
        # Differentiate solution
        u_ddp = traj.uk[0]
        Quu = traj.Quu[0]
        Qux = traj.Qux[0]
        du_dx_ddp = -jnp.linalg.inv(Quu) @ Qux
        u_warm, du_dxv_warm = jax.jvp(lambda nz_: warm_start_policy(0, *dyn.nz_split(nz_)), (nz0,), (v,))
        u = jnp.where(jnp.linalg.norm(traj.xk_bar[-1]) < 0.1, u_ddp, u_warm)
        du_dxv = jnp.where(jnp.linalg.norm(traj.xk_bar[-1]) < 0.1, du_dx_ddp @ dxdnz @ v, du_dxv_warm)
        # jax.debug.breakpoint()
        return u, du_dxv
        # jax.debug.breakpoint()
        # return u_warm, du_dxv_warm

    if whole_traj:
        return ddp_from_nz0_traj
    return ddp_from_nz0


def get_cost_params(N, m, x_star):
    Q = jnp.eye(4)
    R = jnp.eye(m) * 0.01
    P = jnp.array([
        [1.7920, 3.8804, 1.1055, 1.2055],
        [3.8804, 24.7722, 5.7479, 7.0286],
        [1.1055, 5.7479, 1.5855, 1.7647],
        [1.2055, 7.0286, 1.7647, 2.1528]
    ])

    Qk = jnp.repeat(Q[None], N, axis=0)
    Qk = Qk.at[-1].set(P)
    cost_ = DDPQuadCost(Qk=Qk, R=R, x_des=x_star)

    return cost_
