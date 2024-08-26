import functools
import flax.struct
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from flax import linen as nn
from flax import struct
from jaxopt import BoxCDQP
from robots import AbstractZeroDynamics, HopperH2H
from jaxopt import LevenbergMarquardt
from utils import InputBounds, AngleRepresentation, HopperInputBounds
from typing import Callable
from jaxlie import SO3
from jax.nn.initializers import variance_scaling


class SinMLP(nn.Module):
    n_hidden: int
    n_outputs: int
    n_layers: int

    @nn.compact
    def __call__(self, x):
        for i in range(self.n_layers):

            if i == 0:
                x = nn.Dense(self.n_hidden,
                             kernel_init=variance_scaling(scale=1,
                                                          mode="fan_in",
                                                          distribution="uniform")
                             )(x)
                x = jnp.sin(30 * x)
            else:
                x = nn.Dense(self.n_hidden,
                             kernel_init=variance_scaling(6 / (30 ** 2),
                                                          mode="fan_in",
                                                          distribution="uniform")
                             )(x)
                x = jnp.sin(x)
        x = nn.Dense(self.n_outputs)(x)
        return x


class MLP(nn.Module):
    n_hidden: int
    n_outputs: int
    n_layers: int
    activation: Callable = nn.softplus

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_layers):
            x = nn.Dense(self.n_hidden)(x)
            x = self.activation(x)
        x = nn.Dense(self.n_outputs)(x)
        return x


class TwinMLP(nn.Module):
    n_hidden1: int
    n_outputs1: int
    n_layers1: int
    n_hidden2: int
    n_outputs2: int
    n_layers2: int
    activation: Callable = nn.softplus

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_layers1):
            x1 = nn.Dense(self.n_hidden1)(x[0:4:2])
            x1 = self.activation(x1)
        x1 = nn.Dense(self.n_outputs1)(x1)
        for _ in range(self.n_layers2):
            x2 = nn.Dense(self.n_hidden2)(x[1:4:2])
            x2 = self.activation(x2)
        x2 = nn.Dense(self.n_outputs2)(x2)
        return jnp.concatenate([x1, x2])


class ZeroInvariantFunction(nn.Module):
    mlp: MLP

    @nn.compact
    def __call__(self, x):
        x = self.mlp(x) - self.mlp(jnp.zeros_like(x))
        return x


class TVMLP(nn.Module):
    n_hidden: int
    n_outputs: int
    n_layers: int

    activation: Callable = nn.softplus

    @nn.compact
    def __call__(self, t, x):
        # x = jnp.concatenate([x, t[None]], axis=0)
        for i in range(self.n_layers):
            if i == 1:
                x = jnp.concatenate([x, self.time_embedding(t)], axis=0)
            x = nn.Dense(self.n_hidden)(x)
            x = self.activation(x)
        x = nn.Dense(self.n_outputs)(x)
        return x

    def time_embedding(self, t):
        # continuous version of temporal embedding
        t_emb = self.variable('buffers', 't_embedding', self._init_time_embedding).value
        cos_embedding = jnp.cos(t * t_emb)
        sin_embedding = jnp.sin(t * t_emb)
        emb = jnp.concatenate([cos_embedding, sin_embedding], axis=0)
        return emb.flatten()

    def _init_time_embedding(self):
        # time positional embedding adapted for continuous setting
        #  https://github.com/yang-song/score_sde/blob/0acb9e0ea3b8cccd935068cd9c657318fbc6ce4c/models/layers.py#L450
        half_dim = self.n_hidden // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(0, half_dim) * -emb)
        return emb

class HopperMLPPolicy(nn.Module):

    mlp: MLP
    output_bounds: jdc.Static[HopperInputBounds] = struct.field(pytree_node=True)
    angle_rep: jdc.Static[AngleRepresentation] = struct.field(pytree_node=True,default=None)  # angle indexes
    R: jdc.Static[np.array] = struct.field(pytree_node=True,default=None)  # angle indexes
    clip: jdc.Static[bool] = struct.field(pytree_node=True,default=True)  # angle indexes
    rpy_torques: jdc.Static[bool] = struct.field(pytree_node=True,default=False)  # angle indexes
    @nn.compact
    def __call__(self, t,  x):

        if self.angle_rep is not None:
            x = self.angle_rep(x)
        u = self.mlp(x)
        if self.R is not None:
            if self.rpy_torques:
                angle_quat = SO3.from_quaternion_xyzw(jnp.array([0.3646,-0.2795,0.1160, 0.8806],
                                                                dtype=x.dtype))
                u = angle_quat.inverse().as_matrix() @ u
            u = self.output_bounds.qp(self.R, x, u)
        elif self.clip:
            u = self.output_bounds.clip(x, u)
        else:
            u = self.output_bounds(x, u)
        return u


class MLPPolicy(nn.Module):
    mlp: MLP
    angle_rep: jdc.Static[AngleRepresentation] = struct.field(pytree_node=True, default=None)  # angle indexes
    output_bounds: jdc.Static[InputBounds] = struct.field(pytree_node=True, default=None)  # angle indexes

    @nn.compact
    def __call__(self, t, x):
        if self.angle_rep is not None:
            x = self.angle_rep(x)
        x = self.mlp(x)
        if self.output_bounds is not None:
            x = self.output_bounds(x)
        return x


class ZDynPolicy(nn.Module):
    mlp: MLP
    dyn: AbstractZeroDynamics
    K: jdc.Static[np.array]
    angle_rep: jdc.Static[AngleRepresentation] = struct.field(pytree_node=True, default=None)  # angle indexes
    output_bounds: jdc.Static[InputBounds] = struct.field(pytree_node=True, default=None)  # angle indexes

    @nn.compact
    def __call__(self, t, x):
        fx = self.dyn.f(t, x)
        gx = self.dyn.g(t, x)
        # using n for eta
        dphin_dx, dphiz_dx = jax.jacfwd(lambda w: self.dyn.phi(t, w))(x)
        dn2_dx = dphin_dx[dphin_dx.shape[0] // 2:, :]

        def Lwpsi(w):
            return self.psi_dot(t, *self.dyn.phi(t, w))

        jnp.concatenate((dphin_dx, dphiz_dx), axis=0)
        a1 = dn2_dx @ fx
        b1 = dn2_dx @ gx
        # dPhi is implicit in dpsi call, because the composition is defined therein
        psi_dot, a2 = jax.jvp(Lwpsi, primals=(x,), tangents=(fx,))
        b2 = jax.jacfwd(Lwpsi)(x) @ (gx)
        eta, z = self.dyn.phi(t, x)

        error = eta - jnp.concatenate([
            self.psi(z),
            psi_dot
        ])
        u = jnp.linalg.lstsq((b1 - b2), -a1 + a2 - self.K @ error,
                             rcond=None)[0]
        if self.output_bounds is not None:
            u = self.output_bounds.clip(u)
        return u

    def psi(self, z):
        if self.angle_rep is not None:
            z = self.angle_rep(z)
        z = self.mlp(z)
        return z

    def psi_dot(self, t, eta, z):
        return jax.jvp(fun=self.psi,
                       primals=(z,),
                       tangents=(self.dyn.omega(t, self.dyn.nz_join(eta, z)),))[1]

    def residual(self, eta_2, dpsidz, psi, z):
        eta = jnp.concatenate((psi, eta_2))
        return dpsidz @ self.dyn.omega(0.0, self.dyn.nz_join(eta, z)) - eta_2

    def eta_d(self, z):
        psi = self.psi(z)
        dpsidz = jax.jacobian(self.psi)(z)
        solver = LevenbergMarquardt(self.residual)
        sol = solver.run(jnp.zeros_like(psi), dpsidz, psi, z)

        return jnp.concatenate((psi, sol.params))


class RelaxedZDynPolicy(ZDynPolicy):

    def psi(self, z):
        psi, _ = self.psi_eta2(z)
        return psi

    def psi_eta2(self, z):
        return jnp.split(self.eta_d(z), 2, axis=-1)

    def relaxed_residual(self, z):
        psi, eta2d = self.psi_eta2(z)
        dpsidz = jax.jacfwd(self.psi)(z)
        return self.residual(eta2d, dpsidz, psi, z)

    def eta_d(self, z):
        if self.angle_rep is not None:
            z = self.angle_rep(z)
        z = self.mlp(z)
        return z


class RelaxedZDynPolicyPD(RelaxedZDynPolicy):

    @nn.compact
    def __call__(self, t, x):
        eta, z = self.dyn.phi(t, x)
        psi_z, dp_dz_omega, = jax.jvp(self.eta_d, (z,), (self.dyn.omega(t, self.dyn.nz_join(eta, z)),))
        e = eta - psi_z

        fx = self.dyn.f(t, x)
        gx = self.dyn.g(t, x)
        # using n for eta
        dphin_dx, dphiz_dx = jax.jacfwd(lambda w: self.dyn.phi(t, w))(x)
        dn2_dx = dphin_dx[dphin_dx.shape[0] // 2:, :]
        dp_dz2 = dp_dz_omega[dp_dz_omega.shape[0] // 2]

        a1 = dn2_dx @ fx
        b1 = dn2_dx @ gx

        u_ff1 = jnp.linalg.lstsq(b1, -a1 + dp_dz2, rcond=None)[0]

        fx = self.dyn.f(t, x)
        gx = self.dyn.g(t, x)
        # using n for eta
        dphin_dx, dphiz_dx = jax.jacfwd(lambda w: self.dyn.phi(t, w))(x)
        dn2_dx = dphin_dx[dphin_dx.shape[0] // 2:, :]

        def Lwpsi(w):
            return self.psi_dot(t, *self.dyn.phi(t, w))

        jnp.concatenate((dphin_dx, dphiz_dx), axis=0)
        a1 = dn2_dx @ fx
        b1 = dn2_dx @ gx
        # dPhi is implicit in dpsi call, because the composition is defined therein
        psi_dot, a2 = jax.jvp(Lwpsi, primals=(x,), tangents=(fx,))
        b2 = jax.jacfwd(Lwpsi)(x) @ (gx)
        # eta, z = self.dyn.phi(t, x)

        error = eta - jnp.concatenate([
            self.psi(z),
            psi_dot
        ])
        u_ff = jnp.linalg.lstsq((b1 - b2), -a1 + a2, rcond=None)[0]
        # u = jnp.array([0.0]) + self.K @ error
        u = u_ff1 + jnp.array([0.0]) + self.K @ error
        # u = jnp.array([self.K @ e])
        # u = u_ff
        if self.output_bounds is not None:
            u = self.output_bounds.clip(u)
        return u


class HopperZDynPolicy(nn.Module):
    mlp: MLP
    dyn: HopperH2H
    output_bounds: jdc.Static[InputBounds] = struct.field(pytree_node=True, default=None)  # output bounds indexes

    @nn.compact
    def __call__(self, t, x):
        del t
        # technically predict impact time here
        z = self.dyn.predict_impact_z(x)
        # Then evaluate stuff
        u = self.psi(z)
        if self.output_bounds is not None:
            u = self.output_bounds.clip(u)
        return u

    def psi(self, z):
        u = self.mlp(z)
        if self.output_bounds is not None:
            u = self.output_bounds.clip(u)
        return u


class HopperRaibertZDynPolicy(nn.Module):
    K: jdc.Static[np.ndarray]
    dyn: HopperH2H

    @nn.compact
    def __call__(self, t, n0, z0):
        del t
        # technically predict impact time here
        z = self.dyn.predict_impact_z(self.dyn.ground(self.dyn.x_from_nz(n0, z0)))
        # Then evaluate stuff
        clipped = jnp.clip(z, jnp.array([-0.5, -0.5, -jnp.inf, -jnp.inf]), jnp.array([0.5, 0.5, jnp.inf, jnp.inf]))
        u = -self.K @ clipped
        return jnp.clip(u, jnp.array([-0.1, -0.1]), jnp.array([0.1, 0.1]))


class HackedHopperControlPolicy(nn.Module):
    K: jdc.Static[np.ndarray]

    def __call__(self, t, n0, z0):
        clipped = jnp.clip(z0, jnp.array([-0.5, -0.5, -jnp.inf, -jnp.inf]), jnp.array([0.5, 0.5, jnp.inf, jnp.inf]))
        u = -self.K @ clipped
        return jnp.clip(u, jnp.array([-0.1, -0.1]), jnp.array([0.1, 0.1]))


class HackedHopperZDynPolicy(nn.Module):
    K: jdc.Static[np.ndarray]
    mlp: MLP
    dyn: nn.Module

    @nn.compact
    def __call__(self, t, x):
        del t
        # technically predict impact time here
        z = self.dyn.predict_impact_z(x)
        # Then evaluate stuff
        return self.mlp(z)

    def psi(self, z):
        clipped = jnp.clip(z, jnp.array([-0.5, -0.5, -jnp.inf, -jnp.inf]), jnp.array([0.5, 0.5, jnp.inf, jnp.inf]))
        u = -self.K @ clipped
        return jnp.clip(u, jnp.array([-0.1, -0.1]), jnp.array([0.1, 0.1]))


class ZDynLinearDesiredOutput(nn.Module):
    dyn: AbstractZeroDynamics
    Kz: jdc.Static[np.array]

    def __call__(self, z):
        return -self.Kz @ z


class LinearController(nn.Module):
    K: jdc.Static[np.array]

    def __call__(self, t, x):
        return -self.K @ x
class InvariantLinearController(LinearController):

    def __call__(self, x):
        return super().__call__(0, x)

class NZLinearController(nn.Module):
    Kn: jdc.Static[np.array]
    Kz: jdc.Static[np.array]

    def __call__(self, t, n, z):
        return -self.Kn @ n - self.Kz @ z


class InvariantValuePolicy(nn.Module):
    @flax.struct.dataclass
    class TrainTerms:
        u: jnp.array
        val: jnp.array
        res: jnp.array
        d2v_dx2: jnp.array
        dv_dx: jnp.array
        fx: jnp.array
        gx: jnp.array

    val_fn: MLP
    dyn: nn.Module
    R: jdc.Static[np.array] = struct.field(pytree_node=False)
    angle_rep: jdc.Static[AngleRepresentation] = struct.field(pytree_node=True, default=None)  # angle indexes
    output_bounds: jdc.Static[InputBounds] = struct.field(pytree_node=True, default=None)  # angle indexes

    @nn.compact
    def __call__(self, t, x):
        gx = self.dyn.g(t, x)
        val, dvdx = jax.value_and_grad(self.v)(x)
        init = jnp.zeros((self.dyn.action_dim,))
        dvdx_gx = dvdx @ gx
        sol = BoxCDQP().run(init,
                            params_obj=(2 * self.R, (dvdx_gx)),
                            params_ineq=(self.output_bounds.lower,
                                         self.output_bounds.upper))
        return sol.params

    def train_terms(self, t, x):
        fx = self.dyn.f(t, x)
        gx = self.dyn.g(t, x)
        val, dvdx = jax.value_and_grad(self.v)(x)
        init = jnp.zeros((self.dyn.action_dim,), dtype=x.dtype)
        dvdx_gx = dvdx @ gx
        sol = BoxCDQP().run(init,
                            params_obj=(2 * self.R, (dvdx_gx)),
                            params_ineq=(self.output_bounds.lower,
                                         self.output_bounds.upper))
        u = sol.params
        qp_cost = u @ self.R @ u + dvdx_gx @ u
        res = -dvdx @ fx - qp_cost
        d2v_dx2 = jax.hessian(self.v)(x)
        return InvariantValuePolicy.TrainTerms(u=u,
                                               val=val,
                                               res=res,
                                               d2v_dx2=d2v_dx2,
                                               dv_dx=dvdx,
                                               fx=fx,
                                               gx=gx)

    def v(self, x):
        # sum because we assume valfn return a (1,) but we only want a scalar
        # return jnp.sum(jnp.exp(self.val_fn(self.angle_rep(x))))
        return jnp.sum(jax.nn.relu(self.val_fn(self.angle_rep(x))))
        # return jnp.sum((self.val_fn(self.angle_rep(x)))**2)


class PercentileInvariantValuePolicy(nn.Module):
    @flax.struct.dataclass
    class TrainTerms:
        u: jnp.array
        val: jnp.array
        res: jnp.array
        d2v_dx2: jnp.array
        dv_dx: jnp.array
        fx: jnp.array
        gx: jnp.array
        percentiles: jnp.array
        dpercentiles_dx: jnp.array

    val_fn: MLP
    dyn: nn.Module
    R: jdc.Static[np.array] = struct.field(pytree_node=False)
    angle_rep: jdc.Static[AngleRepresentation] = struct.field(pytree_node=True, default=None)  # angle indexes
    output_bounds: jdc.Static[InputBounds] = struct.field(pytree_node=True, default=None)  # angle indexes

    @nn.compact
    def __call__(self, t, x):
        gx = self.dyn.g(t, x)
        val, dvdx = jax.value_and_grad(self.v)(x)
        init = jnp.zeros((self.dyn.action_dim,))
        dvdx_gx = dvdx @ gx
        sol = BoxCDQP().run(init,
                            params_obj=(2 * self.R, (dvdx_gx)),
                            params_ineq=(self.output_bounds.lower,
                                         self.output_bounds.upper))
        return sol.params

    def train_terms(self, t, x):
        fx = self.dyn.f(t, x)
        gx = self.dyn.g(t, x)
        val, dvdx = jax.value_and_grad(self.v)(x)
        init = jnp.zeros((self.dyn.action_dim,), dtype=x.dtype)
        dvdx_gx = dvdx @ gx
        sol = BoxCDQP().run(init,
                            params_obj=(2 * self.R, (dvdx_gx)),
                            params_ineq=(self.output_bounds.lower,
                                         self.output_bounds.upper))
        u = sol.params
        qp_cost = u @ self.R @ u + dvdx_gx @ u
        res = -dvdx @ fx - qp_cost
        d2v_dx2 = jax.hessian(self.v)(x)
        dpercentiles_dx = jax.jacobian(self.percentiles)(x)
        return PercentileInvariantValuePolicy.TrainTerms(u=u,
                                                         percentiles=self.percentiles(x),
                                                         dpercentiles_dx=dpercentiles_dx,
                                                         val=val,
                                                         res=res,
                                                         d2v_dx2=d2v_dx2,
                                                         dv_dx=dvdx,
                                                         fx=fx,
                                                         gx=gx)

    def percentiles(self, x):
        values = self.val_fn(self.angle_rep(x))
        return jnp.cumsum(jax.nn.relu(values))
        # return jnp.cumsum(jax.nn.relu(values))
        # h0 = jax.nn.relu(values[0])
        # h = h0 + jnp.cumsum(jax.nn.softplus(values[1:]))
        # return jnp.concatenate([h0[None], h])

    def v(self, x):
        # sum because we assume valfn return a (1,) but we only want a scalar
        # return jnp.sum(jnp.exp(self.val_fn(self.angle_rep(x))))
        return jnp.sum(jax.nn.relu(self.val_fn(self.angle_rep(x))[0]))
        # return jnp.sum(jax.nn.relu(self.val_fn(self.angle_rep(x))[0]))
        # return jnp.sum((self.val_fn(self.angle_rep(x)))**2)


class HopperInvariantValuePolicy(nn.Module):
    @flax.struct.dataclass
    class TrainTerms:
        u: jnp.array
        val: jnp.array
        d2v_dx2: jnp.array
        dv_dx: jnp.array
        gx: jnp.array

    val_fn: MLP
    dyn: nn.Module
    R: jdc.Static[np.array] = struct.field(pytree_node=False)
    angle_rep: jdc.Static[AngleRepresentation] = struct.field(pytree_node=True,default=None)  # angle indexes
    output_bounds: jdc.Static[HopperInputBounds] = struct.field(pytree_node=True, default=None)     # angle indexes

    @nn.compact
    def __call__(self, t, x):
        gx = self.dyn.g(t, x)
        val, dvdx = jax.value_and_grad(self.v)(x)
        dvdx_gx = dvdx @ gx
        return self.output_bounds.qp(self.R, x, dvdx_gx)


    def train_terms(self, t, x):
        gx = self.dyn.g(t, x)
        val, dvdx = jax.value_and_grad(self.v)(x)
        dvdx_gx = dvdx @ gx
        u = self.output_bounds.qp(self.R, x, dvdx_gx)
        # d2v_dx2 = jax.hessian(self.v, argnums=1)(t, x)
        d2v_dx2 = jnp.zeros((x.shape[0], x.shape[0]), dtype=x.dtype)
        return HopperInvariantValuePolicy.TrainTerms(u=u,
                                                     val=val,
                                                     d2v_dx2=d2v_dx2,
                                                     dv_dx=dvdx,
                                                     gx=gx)

    def v(self, x):
        return jnp.sum(jax.nn.relu(self.val_fn(self.angle_rep(x))))


class PiecewiseConstantController(nn.Module):
    ts_init: jnp.array
    us_init: jnp.array

    def setup(self):
        self.ts = self.variable('buffers', 'ts', lambda: self.ts_init)
        self.us = self.variable('buffers', 'us', lambda: self.us_init)

    def __call__(self, t, x):
        idx = jnp.searchsorted(self.ts.value, t, side='right') - 1
        idx = jnp.clip(idx, 0, self.us.value.shape[0] - 1)
        return self.us.value[idx]


class StochasticMLP(nn.Module):
    mlp: MLP

    @nn.compact
    def __call__(self, x, rng):
        # Compute the standard deviations.
        # log_sigma is a learnable parameter, exp(log_sigma) ensures STD are positive.
        log_sigma = self.param(
            'log_sigma',
            lambda rng_: nn.initializers.zeros(rng_, (self.mlp.n_outputs,))
        )
        u_sigma = jnp.exp(log_sigma)

        u_means = self.mlp(x)

        # If deterministic, use mean. Otherwise, sample from N(mean, std)
        u = (jax.random.normal(rng, u_means.shape) + u_means) * u_sigma

        # If requested, return log probabilities as well (required for training)
        log_probs = -0.5 * ((u - u_means) / u_sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi) - jnp.log(u_sigma)

        # Tanh the result to [-1, 1] to handle input bounds
        return u, jnp.sum(log_probs)

    def u_means(self, x):
        return self.mlp(x)


class RaibertPolicy:
    K: jdc.Static[jnp.ndarray]
    state_bound: InputBounds
    action_bound: InputBounds
    p_max: float

    def __init__(self, K, state_bound, action_bound):
        self.K = K
        self.state_bound = state_bound
        self.action_bound = action_bound

    def __call__(self, x):
        err = self.state_bound.clip(x)
        return self.action_bound.clip(-self.K @ err)

    @staticmethod
    def get_state_lower_bound(p_max):
        return jnp.array(
            [-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, -p_max, -p_max, -jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf])

    @staticmethod
    def get_state_upper_bound(p_max):
        return jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf, p_max, p_max, jnp.inf, jnp.inf, jnp.inf, jnp.inf])


def raibert_policy(K, state_bound, action_bound):
    def _raibert_policy_(x):
        err = state_bound.clip(x)
        return action_bound.clip(-K @ err)

    return _raibert_policy_


class DDPPolicy(nn.Module):
    A_in: jnp.ndarray = struct.field(pytree_node=False)
    b_in: jnp.ndarray = struct.field(pytree_node=False)
    ddp_func: functools.partialmethod

    def __call__(self, t, n0, z0):
        del t
        nz0 = jnp.hstack([n0, z0])
        return self.ddp_func(nz0)

    @staticmethod
    def get_action_lower_bound(u_max):
        return jnp.array([-u_max, -u_max])

    @staticmethod
    def get_action_upper_bound(u_max):
        return jnp.array([u_max, u_max])

    @staticmethod
    def get_A_in():
        return jnp.vstack([jnp.eye(2), -jnp.eye(2)])

    @staticmethod
    def get_b_in(u_max):
        return u_max * jnp.ones((4,))


class DDPPolicyContinuous(nn.Module):
    ddp_func: functools.partialmethod

    def __call__(self, t, n0, z0):
        del t
        nz0 = jnp.hstack([n0, z0])
        return self.ddp_func(nz0)
