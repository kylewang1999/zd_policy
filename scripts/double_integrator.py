import numpy as np
import jax, flax
from flax.struct import dataclass as flax_dataclass
from flax.struct import field
import flax.nnx as nnx
import jax.numpy as jnp
import diffrax as dfx
import jax_dataclasses as jdc
from typing import Callable
from functools import partial
from jax import vmap

@flax_dataclass
class CfgRollout:
    t0: float = 0.0
    t1: float = 5.0
    dt: float = 0.01
    
@flax_dataclass
class CfgROM:
    kpsi: float = 1.0
    ke: float = 1.0
    kv: float = 1.0
    lamv: float = 4.0
    
@flax_dataclass
class CfgPD:
    kp: float = 1.0
    kd: float = 1.0

    
def encode(x: jnp.ndarray) -> tuple[float, float]:
    y = x[..., 1]
    z = x[..., 0]
    return y, z


def decode(y: float | jnp.ndarray, z: float | jnp.ndarray) -> jnp.ndarray:
    return jnp.stack([y, z], axis=-1)


def dyn_x(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    x1dot = x[..., 1]
    x2dot = u
    return jnp.stack([x1dot, x2dot], axis=-1)


def dyn_yz(y:jnp.array, z:jnp.array, u:jnp.array, komega:float) -> tuple[jnp.array, jnp.array]:
    ydot = u
    zdot = y
    return ydot, zdot


def lyap(z: jnp.array, kv: float) -> float:
    return 0.5 * kv * (z ** 2)


def policy_v(y: jnp.ndarray, z: jnp.ndarray, cfg: CfgROM) -> jnp.array:
    v = -cfg.kpsi * y - cfg.ke * (y + cfg.kpsi*z)
    return v


@flax_dataclass
class ArgsDiffrax:
    cfg_rom: CfgROM = field(pytree_node=False, default=CfgROM())
    encode: Callable = field(pytree_node=False, default=encode)
    decode: Callable = field(pytree_node=False, default=decode)
    dyn_x: Callable = field(pytree_node=False, default=dyn_x)
    dyn_yz: Callable = field(pytree_node=False, default=dyn_yz)
    lyap: Callable = field(pytree_node=False, default=lyap)
    policy_v: Callable = field(pytree_node=False, default=policy_v)

@flax_dataclass
class IntegratorOutput:
    xs: jnp.ndarray
    us: jnp.ndarray
    
    
@flax_dataclass
class RetRollout:
    xs: jnp.ndarray
    ys: jnp.ndarray
    zs: jnp.ndarray
    us: jnp.ndarray
    vs: jnp.ndarray
    es: jnp.ndarray
    ts: jnp.ndarray


class Integrator(flax.struct.PyTreeNode):
    solver: Callable
    ts: jnp.array = field(pytree_node=False)
    args_dfx: ArgsDiffrax = field(pytree_node=False, default=ArgsDiffrax())

    @property
    def dt0(self) -> float: return self.ts[1] - self.ts[0]

    @property
    def n_steps(self) -> int: return len(self.ts) - 1


    def init(self, rng, t, x0, mutable=flax.core.DenyList('intermediates')):
        # place holder for policy and robot NN parameters
        return {
            "policy": {},
            "robot": {},
            "iter": 0
        }


    def apply(self, x0):
        
        int_out = IntegratorOutput(
            xs=jnp.zeros((self.n_steps+1, 2)),
            us=jnp.zeros((self.n_steps, 1))
        )
        int_out = int_out.replace(xs=int_out.xs.at[0].set(x0))
        
        def body(i, carry):
            int_out = carry
            x_curr = int_out.xs[i]
            y, z = self.args_dfx.encode(x_curr)
            u = self.args_dfx.policy_v(y, z, self.args_dfx.cfg_rom)
            u = jnp.atleast_1d(u)

            def _term(t, x, args):
                # args is (params, u), use self.args_dfx for dynamics function
                # return self.args_dfx.dyn_x(x, args[1].squeeze())
                return self.args_dfx.dyn_x(x, u.squeeze())
            
            sol = self.solver(
                dfx.ODETerm(_term),
                dt0=self.dt0,
                t0=self.ts[i],
                t1=self.ts[i + 1],
                y0=x_curr,
                # args=(params, u)
                args=None,
            )
            
            int_out = int_out.replace(
                xs=int_out.xs.at[i + 1].set(sol.ys[-1]),
                us=int_out.us.at[i].set(u)
            )

            return int_out
        
        int_out_final = jax.lax.fori_loop(
            lower=0, 
            upper=self.n_steps, 
            body_fun=body, 
            init_val=int_out
        )
        int_out_final = int_out
        return int_out_final
    

    def apply_batch(self, x0s):
        
        int_out = IntegratorOutput(
            xs=jnp.zeros((x0s.shape[0], self.n_steps+1, 2)),  # (B, T+1, 2)
            us=jnp.zeros((x0s.shape[0], self.n_steps, 1))     # (B, T, 1)
        )
        int_out = int_out.replace(xs=int_out.xs.at[:,0].set(x0s))
        
        def body(i, carry):
            int_out = carry
            x_curr = int_out.xs[:,i]
            y, z = vmap(self.args_dfx.encode, in_axes=0)(x_curr)
            u = vmap(self.args_dfx.policy_v, in_axes=(0, 0, None))(y, z, self.args_dfx.cfg_rom)

            def _term(t, x, args):
                # args is (params, u), use self.args_dfx for dynamics function
                # return self.args_dfx.dyn_x(x, args[1].squeeze())
                return vmap(self.args_dfx.dyn_x, in_axes=(0, 0))(x, u)
            
            sol = self.solver(
                dfx.ODETerm(_term),
                dt0=self.dt0,
                t0=self.ts[i],
                t1=self.ts[i + 1],
                y0=x_curr,
                # args=(params, u)
                args=None,
            )
            
            int_out = int_out.replace(
                xs=int_out.xs.at[:,i + 1].set(sol.ys[-1]),
                us=int_out.us.at[:,i].set(u[:,None])
            )

            return int_out
        
        int_out_final = jax.lax.fori_loop(
            lower=0, 
            upper=self.n_steps, 
            body_fun=body, 
            init_val=int_out
        )
        int_out_final = int_out
        return int_out_final
    

 
def rollout(x0: jnp.ndarray, cfg_rollout: CfgRollout, cfg_rom: CfgROM) -> RetRollout:

    ts = jnp.arange(cfg_rollout.t0, cfg_rollout.t1 + cfg_rollout.dt, cfg_rollout.dt)

    def _term(t, x, args: ArgsDiffrax):
        y, z = encode(x)
        v = args.policy_v(y, z, args.cfg_rom)
        u = v
        return dyn_x(x, u)

    term   = dfx.ODETerm(_term)
    solver = dfx.Tsit5()
    saveat = dfx.SaveAt(ts=ts)

    sol = dfx.diffeqsolve(
        term, solver,
        t0=cfg_rollout.t0, t1=cfg_rollout.t1,
        dt0=cfg_rollout.dt,
        y0=jnp.asarray(x0, dtype=jnp.float32),
        saveat=saveat,
        args=ArgsDiffrax(),
    )

    xs = sol.ys                              # (T, 2)
    ys, zs = xs[:, 1], xs[:, 0]
    vs = policy_v(ys, zs, cfg_rom)           # (T,)
    us = vs                                  # u ≡ v
    es = ys - cfg_rom.kpsi * zs

    return RetRollout(xs, ys, zs, us, vs, es, ts)


if __name__ == "__main__":
    cfg_rom = CfgROM()
    cfg_rollout = CfgRollout()
    x0 = jnp.array([1.0, 1.0])
    ret = rollout(x0, cfg_rollout, cfg_rom)

    rng = jax.random.PRNGKey(42)
    ts = jnp.arange(cfg_rollout.t0, cfg_rollout.t1 + cfg_rollout.dt, cfg_rollout.dt)
    x0s_batch = jax.random.normal(rng, (20, 2))

    integrator = Integrator(
        solver=partial(dfx.diffeqsolve, solver=dfx.Tsit5()),
        ts=ts,
        args_dfx=ArgsDiffrax(cfg_rom=cfg_rom),
    )

    # single
    out1 = integrator.apply(x0s_batch[0])
    print("Single integrator shape:", out1.xs.shape)

    # batched (vmap over x0), self is static
    out2 = integrator.apply_batch(x0s_batch)
    print("Batched integrator shape:", out2.xs.shape)   # (B, T, 2)