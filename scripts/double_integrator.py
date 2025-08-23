import numpy as np
import jax, flax
from flax.struct import dataclass as flax_dataclass
from flax.struct import field, PyTreeNode
import flax.nnx as nnx
import jax.numpy as jnp
import diffrax as dfx
import jax_dataclasses as jdc
from typing import Callable
from functools import partial
from jax import vmap
from einops import rearrange

@flax_dataclass
class CfgRollout:
    t0: float = 0.0
    t1: float = 5.0
    dt: float = 0.01
    
@flax_dataclass
class CfgPD:
    kp: float = 1.0
    kd: float = 1.0


@flax_dataclass
class CfgDIROM:
    kpsi: float = 1.0
    ke: float = 1.0
    kv: float = 1.0
    lamv: float = 4.0

class DoubleIntegratorROM(PyTreeNode):
    cfg_rom: CfgDIROM = field(pytree_node=False, default=CfgDIROM())
    

    def encode(self, x: jnp.ndarray) -> tuple[float, float]:
        y = x[..., 1]
        z = x[..., 0]
        return y, z

    def decode(self, y: float | jnp.ndarray, z: float | jnp.ndarray) -> jnp.ndarray:
        return jnp.hstack([z, y])

    def dyn_x(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        x1dot = x[..., 1]
        x2dot = u
        return jnp.hstack([x1dot, x2dot])

    def dyn_y(self, y:jnp.array, u:jnp.array) -> jnp.array:
        ydot = u
        return ydot

    def dyn_z(self, y:jnp.array, z:jnp.array) -> jnp.array:
        zdot = y
        return zdot
    
    def policy_v(self, y: jnp.ndarray, z: jnp.ndarray) -> jnp.array:
        v = -self.cfg_rom.kpsi * y - self.cfg_rom.ke * (y + self.cfg_rom.kpsi*z)
        return v

    def policy_psi(self, z: jnp.ndarray) -> jnp.array:
        return -self.cfg_rom.kpsi * z
    
    def lyap(self, z: jnp.array) -> float:
        return 0.5 * self.cfg_rom.kv * (z ** 2)



class NNDoubleIntegratorROM(DoubleIntegratorROM):
    cfg_rom: CfgDIROM = field(pytree_node=False, default=CfgDIROM())
    rngs: nnx.Rngs = field(pytree_node=False, default=nnx.Rngs(0))
    nn_encoder: nnx.Module = field(pytree_node=False, default=nnx.Linear(2,2, use_bias=False, rngs=nnx.Rngs(0)))
    nn_decoder: nnx.Module = field(pytree_node=False, default=nnx.Linear(2,2, use_bias=False, rngs=nnx.Rngs(0)))
    nn_fy: nnx.Module = field(pytree_node=False, default=nnx.Linear(1,1, use_bias=False, rngs=nnx.Rngs(0)))
    nn_gy: nnx.Module = field(pytree_node=False, default=nnx.Linear(1,1, use_bias=True, rngs=nnx.Rngs(0)))
    nn_fz: nnx.Module = field(pytree_node=False, default=nnx.Linear(2,1, use_bias=False, rngs=nnx.Rngs(0)))
    nn_psi: nnx.Module = field(pytree_node=False, default=nnx.Linear(1,1, use_bias=False, rngs=nnx.Rngs(0)))
    
    
    def hardcode_nn_params(self, ):
        """Explicitly set parameters for all neural network modules.
        
        Args:
            params: Dictionary containing parameter arrays for each module.
                   Keys should match the module names (e.g., 'nn_decoder', 'nn_gy', etc.)
        """

        self.nn_encoder.kernel.value = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        self.nn_decoder.kernel.value = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        
        self.nn_fy.kernel.value = jnp.array([[0.0]])
        self.nn_gy.kernel.value = jnp.array([[0.0]])
        self.nn_gy.bias.value = jnp.array([1.0])
        
        self.nn_fz.kernel.value = jnp.array([[1.0], [0.0]])
        self.nn_psi.kernel.value = jnp.array([[self.cfg_rom.kpsi]])
    
    
    def encode(self, x: jnp.ndarray) -> tuple[float, float]:
        yz = self.nn_encoder(x)
        y, z = yz[..., 0], yz[..., 1]
        return y, z
    
    def decode(self, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        return self.nn_decoder(jnp.stack([y, z], axis=-1))
    
    def dyn_x(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return super().dyn_x(x, u)
    
    def dyn_y(self, y: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return self.nn_fy(y) + self.nn_gy(y) * u
    
    def dyn_z(self, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        return self.nn_fz(jnp.hstack([y, z]))
    
    def policy_psi(self, z: jnp.ndarray) -> jnp.ndarray:
        return -self.nn_psi(z)
    
    def dist_to_manifold(self, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        return y - self.policy_psi(z)
    
    def policy_v(self, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:   
        y = jnp.atleast_1d(y)
        z = jnp.atleast_1d(z)
        
        gy = self.nn_gy(y)
        g_inv = 1.0 / gy
        zdot = self.dyn_z(y, z)
        
        _, dpsidz_times_omega = jax.jvp(self.policy_psi, (z,), (zdot,))
        fy = self.nn_fy(y)
        return g_inv * (dpsidz_times_omega
                        - self.cfg_rom.ke * self.dist_to_manifold(y, z)
                        - fy)
    
    def lyap(self, z: jnp.array) -> float:
        return super().lyap(z)



@flax_dataclass
class IntegratorOutput:
    xs: jnp.ndarray
    us: jnp.ndarray
    
    
@flax_dataclass
class IntegratorDebugOutput:
    xs: jnp.ndarray
    ys: jnp.ndarray
    zs: jnp.ndarray
    us: jnp.ndarray
    vs: jnp.ndarray
    es: jnp.ndarray
    ts: jnp.ndarray
    lyaps: jnp.ndarray


class Integrator(PyTreeNode):
    solver: Callable
    ts: jnp.array = field(pytree_node=False)
    rom: DoubleIntegratorROM = field(pytree_node=False, default=DoubleIntegratorROM())

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


    def apply(self, x0s):
        ''' Integrate the dynamics from self.ts[0] to self.ts[-1] with initial condition x0s.
        Input:
            x0s: (B, 2)
        Output:
            int_out: IntegratorOutput
        '''
        
        int_out = IntegratorOutput(
            xs=jnp.zeros((x0s.shape[0], self.n_steps+1, 2)),  # (B, T+1, 2)
            us=jnp.zeros((x0s.shape[0], self.n_steps, 1))     # (B, T, 1)
        )
        int_out = int_out.replace(xs=int_out.xs.at[:,0].set(x0s))
        
        def body(i, carry):
            int_out = carry
            x_curr = int_out.xs[:,i]
            y, z = vmap(self.rom.encode, in_axes=0)(x_curr)
            u = vmap(self.rom.policy_v, in_axes=(0, 0))(y, z)
            
            if u.ndim == 1:
                u = u[:,None]

            def _term(t, x, args):
                # args is (params, u), use self.args_dfx for dynamics function
                # return self.args_dfx.dyn_x(x, args[1].squeeze())
                return vmap(self.rom.dyn_x, in_axes=(0, 0))(x, u)
            
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
                us=int_out.us.at[:,i].set(u)
            )

            return int_out
        
        int_out_final = jax.lax.fori_loop(
            lower=0, 
            upper=self.n_steps, 
            body_fun=body, 
            init_val=int_out
        )
        return int_out_final
    
    
    def post_apply(self, int_out: IntegratorOutput) -> IntegratorDebugOutput:
        ''' Augment IntegratorOutput with debug information. '''
        xs, us = int_out.xs, int_out.us # (B, T+1, 2), (B, T, 1)
        B = xs.shape[0]
        ys, zs = vmap(self.rom.encode, in_axes=0)(rearrange(xs, 'b t d -> (b t) d'))
        ys = ys[..., None]
        zs = zs[..., None]
        vs = vmap(self.rom.policy_v, in_axes=(0, 0))(ys, zs)
        psis = vmap(self.rom.policy_psi, in_axes=0)(zs)
        es = jnp.abs(ys - psis)
        lyaps = vmap(self.rom.lyap, in_axes=0)(zs)
        
        xs = xs
        ys = rearrange(ys, '(b t) d -> b t d', b=B)
        zs = rearrange(zs, '(b t) d -> b t d', b=B)
        us = us
        vs = rearrange(vs, '(b t) d -> b t d', b=B)
        es = rearrange(es, '(b t) d -> b t d', b=B)
        lyaps = rearrange(lyaps, '(b t) d -> b t d', b=B)
        
        return IntegratorDebugOutput(xs, ys, zs, us, vs, es, self.ts, lyaps)
    


if __name__ == "__main__":
    cfg_rom = CfgDIROM()
    cfg_rollout = CfgRollout()

    rng = jax.random.PRNGKey(42)
    ts = jnp.arange(cfg_rollout.t0, cfg_rollout.t1 + cfg_rollout.dt, cfg_rollout.dt)
    x0s_batch = jax.random.normal(rng, (20, 2))

    integrator = Integrator(
        solver=partial(dfx.diffeqsolve, solver=dfx.Tsit5()),
        ts=ts,
        rom=DoubleIntegratorROM(cfg_rom=cfg_rom),
    )

    out = integrator.apply(x0s_batch)
    print("Batched integrator shape:", out.xs.shape)   # (B, T, 2)