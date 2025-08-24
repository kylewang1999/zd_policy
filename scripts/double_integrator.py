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
class CfgDIROM:
    dim_x: int = 2
    dim_y: int = 1
    dim_z: int = 1
    dim_u: int = 1
    kpsi: float = 1.0
    ke: float = 1.0
    kv: float = 1.0
    lamv: float = 4.0
    rngs: nnx.Rngs = field(pytree_node=False, default=nnx.Rngs(0))


class DoubleIntegratorROM(PyTreeNode):
    cfg_rom: CfgDIROM = field(pytree_node=False, default=CfgDIROM())
    
    '''
    Note: Input to any of the following functions should not be batched.
          Batched-application should be done explictly using vmap.
    '''
    

    def encode(self, x: jnp.ndarray) -> tuple[float, float]:
        '''
        Input: (2,). Should not be batched.
        Output: Tuple[(1,), (1,)]
        '''
        y = x[..., 1:2]
        z = x[..., 0:1]
        return y, z

    def decode(self, y: float | jnp.ndarray, z: float | jnp.ndarray) -> jnp.ndarray:
        return jnp.hstack([z, y])

    def dyn_x(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        x1dot = x[..., 1:2]
        x2dot = jnp.atleast_1d(u)
        return jnp.hstack([x1dot, x2dot])

    def dyn_y(self, y:jnp.array, u:jnp.array) -> jnp.array:
        ydot = jnp.atleast_1d(u)
        return ydot

    def dyn_z(self, y:jnp.array, z:jnp.array) -> jnp.array:
        zdot = jnp.atleast_1d(y)
        return zdot
    
    def policy_v(self, y: jnp.ndarray, z: jnp.ndarray) -> jnp.array:
        v = -self.cfg_rom.kpsi * y - self.cfg_rom.ke * (y + self.cfg_rom.kpsi*z)
        return v

    def policy_psi(self, z: jnp.ndarray) -> jnp.array:
        return -self.cfg_rom.kpsi * z
    
    def map_v_to_u(self, v: jnp.ndarray) -> jnp.array:
        return v
    
    def lyap(self, z: jnp.array) -> float:
        return 0.5 * self.cfg_rom.kv * (z ** 2)
    
    
    # helpers for loss functions
    def _encode_vec(self, x: jnp.ndarray) -> jnp.ndarray:
        """Vector-valued encoder: returns (y,z) concatenated -> shape (2,)."""
        y, z = self.encode(x)
        return jnp.hstack([y, z])
    
    def _jac_encoder(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.jacfwd(self._encode_vec)(x)

    def _jac_y(self, x: jnp.ndarray) -> jnp.ndarray:
        J = jax.jacfwd(self._encode_vec)(x)  # (2,2)
        return J[0:1, :]                     # (1,2)

    def _jac_z(self, x: jnp.ndarray) -> jnp.ndarray:
        J = jax.jacfwd(self._encode_vec)(x)
        return J[1:2, :]                   

    def _split_f_g(self, y: jnp.ndarray):
        '''
        Generic decomposition dyn_y(y,u) = f_y(y) + G_y(y) u via jacobian w.r.t. u.
        Returns f_y(y):(1,), G_y(y):(1,1).
        '''
        u0 = jnp.zeros_like(y)
        f_y = self.dyn_y(y, u0)
        G_y = jax.jacfwd(lambda _u: self.dyn_y(y, _u))(u0)
        return f_y, G_y

    def _sqnorm1(self, a: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(a * a, keepdims=True)


    # loss functions
    def loss_y_proj(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        '''
        L_y = || Dy(x) f_xu - [ f_y(y) + G_y(y) u ] ||^2  (shape (1,))
        '''
        y, z = self.encode(x)
        Dy   = self._jac_y(x)
        f_xu = self.dyn_x(x, u)
        term1 = Dy @ f_xu
        
        f_y, G_y = self._split_f_g(y)
        term2 = f_y + (G_y @ jnp.atleast_1d(u))
        
        return self._sqnorm1(term1 - term2)


    def loss_z_proj(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        '''
        L_ω = || Dz(x) f_xu - ω(y,z) ||^2  (shape (1,))
        '''
        y, z = self.encode(x)
        Dz   = self._jac_z(x)
        f_xu = self.dyn_x(x, u)
        term1 = Dz @ f_xu
        term2 = self.dyn_z(y, z)
        return self._sqnorm1(term1 - term2)


    def loss_stable_m(
        self,
        z: jnp.ndarray,
    ) -> jnp.ndarray:
        '''
        L_{ω,stab}(z) = max{0, <∇V(z), ω(y,z)> + λ V(z)} with ∇V(z)=z, V=0.5||z||^2.
        Params:
            z: (1,)
            lam: float
        Returns:
            (1,)
        '''
        lam = self.cfg_rom.lamv
        y = self.policy_psi(z)
        omega = self.dyn_z(y, z)
        zTomega = jnp.sum(z * omega, keepdims=True)
        V = self.lyap(z)
        stab = zTomega + lam * V
        return jnp.maximum(0.0, stab)


    def loss_invari_m(self, z: jnp.ndarray) -> jnp.ndarray:
        '''
        L_inv = || f(ψ(z),z) + G(ψ(z)) v(ψ(z),z) - (∂ψ/∂z)(z) ω(ψ(z),z) ||^2  (shape (1,))
        Uses per-sample v = policy_v(ψ(z), z), f,G from dyn_y decomposition.
        '''
        y = self.policy_psi(z)
        v = self.policy_v(y, z)

        Jpsi = jax.jacfwd(self.policy_psi)(z)
        omega = self.dyn_z(y, z)

        term1  = self.dyn_y(y, v)
        term2 = (Jpsi @ omega)
        return self._sqnorm1(term1 - term2)


    def loss_nondegenerate_enc(
        self,
        x: jnp.ndarray,
        alpha_det: float = 1.0,   # weight for (det-1)^2
        beta_orth: float = 1.0,   # weight for ||JᵀJ - I||_F^2
        gamma_pos: float = 0.0    # weight to prefer det>0 (orientation)
    ) -> jnp.ndarray:
        '''
        Encourage J_E(x) ∈ SL(n) approximately: det→1, orthonormal columns, det>0.
        Returns shape (1,).
        '''
        J = self._jac_encoder(x)
        n = J.shape[0]
        detJ = jnp.linalg.det(J)
        I = jnp.eye(n, dtype=J.dtype)
        
        term_det = (jnp.abs(detJ) - 1.0) ** 2            # (det-1)^2 close to 0
        term_orth = jnp.sum((J.T @ J - I) ** 2) # orthonormality: Frobenius norm of JᵀJ - I
        term_pos = jax.nn.softplus(-detJ)       # orientation hinge: penalize negative det softly
        return (alpha_det * term_det + beta_orth * term_orth + gamma_pos * term_pos)[None]


class NNDoubleIntegratorROM(DoubleIntegratorROM):
    cfg_rom: CfgDIROM = field(pytree_node=False, default=CfgDIROM())
    nn_encoder: nnx.Module = field(pytree_node=False, default=nnx.Linear(2, 2, use_bias=False, rngs=nnx.Rngs(0)))
    nn_decoder: nnx.Module = field(pytree_node=False, default=nnx.Linear(2, 2, use_bias=False, rngs=nnx.Rngs(0)))
    nn_fy: nnx.Module = field(pytree_node=False, default=nnx.Linear(1, 1, use_bias=False, rngs=nnx.Rngs(0)))
    nn_gy: nnx.Module = field(pytree_node=False, default=nnx.Linear(1, 1, use_bias=True,  rngs=nnx.Rngs(0)))
    nn_fz: nnx.Module = field(pytree_node=False, default=nnx.Linear(2, 1, use_bias=False, rngs=nnx.Rngs(0)))
    nn_psi: nnx.Module = field(pytree_node=False, default=nnx.Linear(1, 1, use_bias=False, rngs=nnx.Rngs(0)))

    @property
    def default_nn_params(self) -> dict:
        kpsi = float(self.cfg_rom.kpsi)
        return {
            "nn_encoder": {"kernel": jnp.array([[0., 1.], [1., 0.]])},        
            "nn_decoder": {"kernel": jnp.array([[0., 1.], [1., 0.]])},        
            "nn_fy":      {"kernel": jnp.array([[0.]])},                      
            "nn_gy":      {"kernel": jnp.array([[0.]]), "bias": jnp.array([1.])},
            "nn_fz":      {"kernel": jnp.array([[1.], [0.]])},                # ż = y
            "nn_psi":     {"kernel": jnp.array([[kpsi]])},                    # ψ(z) = kψ z (we negate outside)
        }
        
    def get_nn_params(self) -> dict:
        out = {}
        for name in ("nn_encoder", "nn_decoder", "nn_fy", "nn_gy", "nn_fz", "nn_psi"):
            mod = getattr(self, name)
            d = {}
            if hasattr(mod, "kernel"):
                d["kernel"] = mod.kernel.value
            if hasattr(mod, "bias"):
                d["bias"] = mod.bias.value
            out[name] = d
        return out
    
    def set_nn_params(self, params: dict, *, strict: bool = True, cast: bool = True) -> None:
        '''
        Params:
            params: dict of module names to dicts of attributes to values
            strict: if True, raise an error if a module or attribute is not found
            cast: if True, cast the values to the correct dtype
        '''
        for name, pdict in params.items():
            mod = getattr(self, name, None)
            if mod is None:
                if strict:
                    raise AttributeError(f"Module '{name}' not found on self.")
                else:
                    continue
            for attr, arr in pdict.items():
                if not hasattr(mod, attr):
                    if strict:
                        raise AttributeError(f"Module '{name}' has no attribute '{attr}'.")
                    else:
                        continue
                ref = getattr(mod, attr).value
                val = jnp.asarray(arr, dtype=ref.dtype) if cast else arr
                # sanity check shapes:
                if ref.shape != val.shape:
                    raise ValueError(f"Shape mismatch for {name}.{attr}: expected {ref.shape}, got {val.shape}.")
                getattr(mod, attr).value = val
    
    
    def hardcode_nn_params(self):
        self.set_nn_params(self.default_nn_params)


    def encode(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        yz = self.nn_encoder(x)
        y, z = yz[0:1], yz[1:2]
        return y, z

    def decode(self, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        return self.nn_decoder(jnp.hstack([y, z]))

    def dyn_x(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return super().dyn_x(x, u)

    def dyn_y(self, y: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return self.nn_fy(y) + self.nn_gy(y) * u

    def dyn_z(self, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        return self.nn_fz(jnp.hstack([y, z]))

    def policy_psi(self, z: jnp.ndarray) -> jnp.ndarray:
        return -self.nn_psi(z)

    def policy_v(self, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        y = jnp.atleast_1d(y)
        z = jnp.atleast_1d(z)
        gy   = self.nn_gy(y)
        ginv = 1.0 / gy
        zdot = self.dyn_z(y, z)
        _, dpsi_omega = jax.jvp(self.policy_psi, (z,), (zdot,))
        fy = self.nn_fy(y)
        return ginv * (dpsi_omega - self.cfg_rom.ke * (y - self.policy_psi(z)) - fy)

    def lyap(self, z: jnp.ndarray) -> jnp.ndarray:
        return super().lyap(z)



@flax_dataclass
class IntegratorOutput:
    xs: jnp.ndarray
    us: jnp.ndarray
    
    
@flax_dataclass
class IntegratorAuxOutput:
    xs: jnp.ndarray
    ys: jnp.ndarray
    zs: jnp.ndarray
    us: jnp.ndarray
    vs: jnp.ndarray
    es: jnp.ndarray
    ts: jnp.ndarray
    lyaps: jnp.ndarray


@flax_dataclass
class LossOutput:
    y_proj: jnp.ndarray
    z_proj: jnp.ndarray
    stable_m: jnp.ndarray
    invari_m: jnp.ndarray
    nondegenerate_enc: jnp.ndarray
    total: jnp.ndarray


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


    def apply(self, x0s, policy_fun: Callable=None):
        ''' Integrate the dynamics from self.ts[0] to self.ts[-1] with initial condition x0s.
        Input:
            x0s: (B, 2)
        Output:
            int_out: IntegratorOutput containing xs (B,T+1,2) and us (B,T,1)
        '''
        if policy_fun is None:
            policy_fun = self.rom.policy_v
        
        int_out = IntegratorOutput(
            xs=jnp.zeros((x0s.shape[0], self.n_steps+1, 2)),  # (B, T+1, 2)
            us=jnp.zeros((x0s.shape[0], self.n_steps, 1))     # (B, T, 1)
        )
        int_out = int_out.replace(xs=int_out.xs.at[:,0].set(x0s))
        
        def step(i, carry):
            int_out = carry
            x_curr = int_out.xs[:, i]                          # (B,2)
            y, z   = vmap(self.rom.encode, in_axes=0)(x_curr)  # (B,1),(B,1)
            v      = vmap(policy_fun, in_axes=(0, 0))(y, z)    # (B,1)
            u      = vmap(self.rom.map_v_to_u, in_axes=0)(v)

            def _term(t, x, args):
                return vmap(self.rom.dyn_x, in_axes=(0, 0))(x, v)     # (B,2)

            sol = self.solver(
                dfx.ODETerm(_term),
                dt0=self.dt0,
                t0=self.ts[i],
                t1=self.ts[i + 1],
                y0=x_curr,
                args=None,
            )
            return int_out.replace(
                xs=int_out.xs.at[:, i + 1].set(sol.ys[-1]),
                us=int_out.us.at[:, i].set(u),
            )
        
        return jax.lax.fori_loop(
            lower=0, 
            upper=self.n_steps, 
            body_fun=step, 
            init_val=int_out
        )
    
    
    def post_apply(self, int_out: IntegratorOutput) -> tuple[IntegratorAuxOutput, LossOutput]:
        ''' Augment IntegratorOutput with debug information. 
        Note: last time step in `xs` is truncated. So both `xs` and `us` have shape (B,T,*).
        '''
        xs, us = int_out.xs, int_out.us
        B, T = us.shape[0:2]
        xs = xs[:, :-1, :]
        flat_xs = rearrange(xs, 'b t d -> (b t) d')
        flat_us = rearrange(us, 'b t d -> (b t) d')

        ys, zs = vmap(self.rom.encode, in_axes=0)(flat_xs)       # (B*T1,1), (B*T1,1)
        vs     = vmap(self.rom.policy_v, in_axes=(0, 0))(ys, zs) # (B*T1,1)
        psis   = vmap(self.rom.policy_psi, in_axes=0)(zs)        # (B*T1,1)
        es     = jnp.abs(ys - psis)                              # (B*T1,1)
        lyaps  = vmap(self.rom.lyap, in_axes=0)(zs)              # (B*T1,1)
        aux_out = IntegratorAuxOutput(
            xs, 
            rearrange(ys,   '(b t) d -> b t d', b=B), 
            rearrange(zs,   '(b t) d -> b t d', b=B), 
            us, 
            rearrange(vs,   '(b t) d -> b t d', b=B), 
            rearrange(es,   '(b t) d -> b t d', b=B), 
            self.ts, 
            rearrange(lyaps,'(b t) d -> b t d', b=B)
        )
        
        return aux_out
    
    
    def compute_loss(self, int_out: IntegratorOutput) -> LossOutput:
        xs, us = int_out.xs, int_out.us
        B, T = us.shape[0:2]
        xs = xs[:, :-1, :]
        flat_xs = rearrange(xs, 'b t d -> (b t) d')
        flat_us = rearrange(us, 'b t d -> (b t) d')
        
        ys, zs = vmap(self.rom.encode, in_axes=0)(flat_xs)       # (B*T1,1), (B*T1,1)
        
        l_y_proj = vmap(self.rom.loss_y_proj, in_axes=(0, 0))(flat_xs, flat_us)
        l_z_proj = vmap(self.rom.loss_z_proj, in_axes=(0, 0))(flat_xs, flat_us)
        l_stable_m = vmap(self.rom.loss_stable_m, in_axes=0)(zs)
        l_invari_m = vmap(self.rom.loss_invari_m, in_axes=0)(zs)
        l_nondegenerate_enc = vmap(self.rom.loss_nondegenerate_enc, in_axes=0)(flat_xs)
        total = l_y_proj + l_z_proj + l_stable_m + l_invari_m + l_nondegenerate_enc
        loss_out = LossOutput(
            y_proj=rearrange(l_y_proj, '(b t) d -> b t d', b=B),
            z_proj=rearrange(l_z_proj, '(b t) d -> b t d', b=B),
            stable_m=rearrange(l_stable_m, '(b t) d -> b t d', b=B),
            invari_m=rearrange(l_invari_m, '(b t) d -> b t d', b=B),
            nondegenerate_enc=rearrange(l_nondegenerate_enc, '(b t) d -> b t d', b=B),
            total=rearrange(total, '(b t) d -> b t d', b=B),
        )
        
        return loss_out



@flax_dataclass
class CfgTrain:
    pass


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