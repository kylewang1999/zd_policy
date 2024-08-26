import flax.struct
import jax
import jax.numpy as jnp
from jaxopt import BoxCDQP
from utils import InputBounds
from typing import Callable
from robots import HopperH2H
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


def linearize_dynamics_function(dyn):
    """
    Linearize dynamics about the current state and discretize
    """
    def linearize(x, u):
        A, B = jax.jacrev(dyn, argnums=(0, 1))(x, u)
        xdot = dyn(x, u)
        C = xdot - A @ x - B @ u
        return LQRDynamics(Ak=A, Bk=B, Ck=C)
    linearize_func = jax.vmap(linearize)
    linearize_func = jax.jit(linearize_func)
    return linearize_func


def qb_bounds_state(constraints):
    def qp_bounds_(xg_post, u_bar):
        del xg_post
        delta_l = constraints.lower - u_bar
        delta_u = constraints.upper - u_bar
        bound = (delta_l, delta_u)
        return bound
    return qp_bounds_


def qp_bounds_torque(constraints, Kp, Kd):
    def qp_bounds_(xg_post, u_bar):
        delta_l = 1/Kp * (constraints.lower + Kp*xg_post[:2] + Kd * xg_post[2:4]) - u_bar
        delta_u = 1/Kp * (constraints.upper + Kp*xg_post[:2] + Kd * xg_post[2:4]) - u_bar
        bound = (delta_l, delta_u)
        return bound
    return qp_bounds_


def ddp(traj: DDPVars, cost: DDPQuadCost, lin_dyn: LQRDynamics,
        dynamics_ground: Callable, dynamics_flight: Callable,
        qp_bounds: Callable, batch_line_search=None):
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
        xgp = dynamics_ground(traj_.xk[i])
        ineq_bounds = qp_bounds(xgp, traj.uk_bar[i])
        kt = BoxCDQP().run(traj.uk_bar[i],
                           params_obj=(traj_.Quu[i], traj_.qu[i]),
                           params_ineq=ineq_bounds
                           ).params
        u = traj.uk_bar[i] + alpha_ * kt - jnp.linalg.inv(traj_.Quu[i]) @ traj_.Qux[i] @ (traj_.xk[i] - traj_.xk_bar[i])
        u = jnp.clip(u, a_min=ineq_bounds[0] + traj.uk_bar[i], a_max=ineq_bounds[1] + traj.uk_bar[i])
        xfp = dynamics_flight(xgp, u)
        return traj_.replace(xk=traj_.xk.at[i+1].set(xfp), uk=traj_.uk.at[i].set(u))


    if batch_line_search is None:
        def cond_fun(fc):
            traj_, alpha_ = fc
            new_cost = cost.horizon_cost(traj_.xk, traj_.uk)
            nan_check = jnp.isnan(new_cost)
            true_cost_larger = new_cost >= traj_bar_cost
            alpha_reasonable = alpha_ > 1e-8
            return (true_cost_larger & alpha_reasonable) | (alpha_ == alpha_0 / decay_factor) | nan_check
        def line_search_step(fc):
            traj_, alpha_ = fc
            alpha_ = alpha_ * decay_factor
            new_traj = jax.lax.fori_loop(0, N-1, lambda i, _traj_: forward_step(i, _traj_, alpha_), traj_)
            return new_traj, alpha_

        traj, alpha = jax.lax.while_loop(cond_fun, line_search_step, (traj, alpha_0 / decay_factor))
    else:
        alphas = jnp.array([alpha_0 / (decay_factor ** i) for i in range(batch_line_search)])
        def line_search(traj, alpha):
            return jax.lax.fori_loop(0, N-1, lambda i, _traj_: forward_step(i, _traj_, alpha), traj)
        line_search_trajs = jax.vmap(line_search, in_axes=(None, 0))(traj, alphas)
        new_costs = jax.vmap(cost.horizon_cost)(line_search_trajs.xk, line_search_trajs.uk)
        idx = jnp.argmin(new_costs)
        traj = jax.tree_map(lambda x: x[idx], line_search_trajs)

    return traj


def ddp_from_nz0_func(dyn, u_max, N, ddp_iter, raibert_heuristic, Kp=100, Kd=20, torque_bounds=False, whole_traj=False):
    lin_func = linearize_dynamics_function(dyn.f)
    cost_params = get_cost_params(N, dyn.d_m, dyn.x_star())
    bounds = InputBounds(lower=-u_max * jnp.ones(dyn.d_m), upper=u_max * jnp.ones(dyn.d_m))
    if torque_bounds:
        qp_bounds = qp_bounds_torque(bounds, Kp, Kd)
    else:
        qp_bounds = qb_bounds_state(bounds)

    def warm_start(i, ws):
        u = raibert_heuristic(ws.xk_bar[i])
        return ws.replace(
            xk_bar=ws.xk_bar.at[i+1].set(dyn.f(ws.xk_bar[i], u)),
            uk_bar=ws.uk_bar.at[i].set(u)
        )

    def ddp_iter_body(_, traj):
        lin_dyn = lin_func(traj.xk_bar[:-1], traj.uk_bar)
        traj = ddp(traj=traj,
                   cost=cost_params,
                   lin_dyn=lin_dyn,
                   dynamics_ground=dyn.ground,
                   dynamics_flight=dyn.flight,
                   qp_bounds=qp_bounds)
        return traj.replace(xk_bar=traj.xk_bar.at[:].set(traj.xk),
                            uk_bar=traj.uk_bar.at[:].set(traj.uk))

    def ddp_from_nz0_traj(nz0):
        x0 = HopperH2H.x_from_nz(nz0[:2], nz0[2:])
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
        traj = ddp_from_nz0_traj(nz0)
        return traj.uk[0]

    A_in = jnp.vstack([jnp.eye(2), -jnp.eye(2)])

    @ddp_from_nz0.defjvp
    def u_star_jvp(primals, tangents):
        nz0, = primals
        v, = tangents
        x0 = HopperH2H.x_from_nz(nz0[:2], nz0[2:])
        xg_p = dyn.ground(x0)
        # Solve OCP
        traj = ddp_from_nz0_traj(nz0)
        # Differentiate solution
        u = traj.uk[0]
        Quu = traj.Quu[0]
        Qux = traj.Qux[0][:, jnp.array([0, 1, 4, 5, 7, 8])]
        delta_bounds = qp_bounds(xg_p, u)
        b_in = jnp.hstack([delta_bounds[0] + u, -(delta_bounds[1] + u)])
        lam = -jnp.linalg.pinv(A_in.T) @ traj.qu[0]
        lam = jnp.where(jnp.abs(A_in @ u - b_in) < 1e-3, lam, 0)
        Dug = jnp.block([
            [Quu, A_in.T],
            [jnp.diag(lam) @ A_in, jnp.diag(A_in @ u - b_in)]
        ])
        db_dx = jnp.vstack(jax.jacfwd(qp_bounds, argnums=0)(xg_p, u))[:, jnp.array([0, 1, 4, 5, 7, 8])]
        Dog = jnp.vstack([Qux, db_dx])
        du_dx = -jnp.hstack([jnp.eye(2), jnp.zeros((2, 4))]) @ (jnp.linalg.inv(Dug) @ Dog)

        return u, du_dx @ v

    if whole_traj:
        return ddp_from_nz0_traj
    return ddp_from_nz0


def get_cost_params(N, m, x_star):
    Q = jnp.block([
        [jnp.zeros((4, 10))],
        [jnp.zeros((2, 4)), jnp.eye(2), jnp.zeros((2, 4))],
        [jnp.zeros((1, 10))],
        [jnp.zeros((2, 7)), jnp.eye(2), jnp.zeros((2, 1))],
        [jnp.zeros((1, 10))]
    ])
    # P = jnp.array([
    #     [41.6283, 0, 0, 0, 6.4528, 0, 0, 12.5623, 0, 0],
    #     [0, 41.6283, 0, 0, 0, 6.4528, 0, 0, 12.5623, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [6.4528, 0, 0, 0, 3.9869, 0, 0, 2.0197, 0, 0],
    #     [0, 6.4528, 0, 0, 0, 3.9869, 0, 0, 2.0197, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [12.5623, 0, 0, 0, 2.0197, 0, 0, 4.7936, 0, 0],
    #     [0, 12.5623, 0, 0, 0, 2.0197, 0, 0, 4.7936, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # ])
    P = jnp.array([
        [164.5857,0,0,0,14.6487,0,0,.2715,0,0],
        [0,164.5857,0,0,0,14.6487,0,0,49.2715,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [14.6487,0,0,0,5.2841,0,0,4.4940,0,0],
        [0,14.6487,0,0,0,5.2841,0,0,4.4940,0],
        [0,0,0,0,0,0,0,0,0,0],
        [49.2715,0,0,0,4.4940,0,0,15.7542,0,0],
        [0,49.2715,0,0,0,4.4940,0,0,15.7542,0],
        [0,0,0,0,0,0,0,0,0,0]
    ])

    Qk = jnp.repeat(Q[None], N, axis=0)
    Qk = Qk.at[-1].set(P)
    # R = 0.01 * jnp.eye(m)
    R = 50 * jnp.eye(m)
    cost_ = DDPQuadCost(Qk=Qk, R=R, x_des=x_star)

    return cost_
