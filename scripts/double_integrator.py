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
    
    @property
    def ts(self) -> jnp.ndarray:
        return jnp.arange(self.t0, self.t1 + self.dt, self.dt)

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
   
    loss_scale_dict: dict[str, float] = field(pytree_node=False, default_factory=lambda: {
        "y_proj": 1.0,
        "z_proj": 1.0,
        "stable_m": 1.0,
        "invari_m": 1.0,
        "nondegenerate_enc": 0.0,
    })


class DoubleIntegratorROM():
    
    def __init__(self, cfg_rom: CfgDIROM):
        self.cfg_rom = cfg_rom
    
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


class NNDoubleIntegratorROM(DoubleIntegratorROM, nnx.Module):

    def __init__(self, cfg_rom: CfgDIROM, rngs: nnx.Rngs=nnx.Rngs(0)):
        super().__init__(cfg_rom)
        self.nn_encoder = nnx.Linear(2, 2, use_bias=False, rngs=rngs)
        self.nn_decoder = nnx.Linear(2, 2, use_bias=False, rngs=rngs)
        self.nn_fy      = nnx.Linear(1, 1, use_bias=False, rngs=rngs)
        self.nn_gy      = nnx.Linear(1, 1, use_bias=True,  rngs=rngs)
        self.nn_fz      = nnx.Linear(2, 1, use_bias=False, rngs=rngs)
        self.nn_psi     = nnx.Linear(1, 1, use_bias=False, rngs=rngs)


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
    
    def set_nn_params(self, params: dict):
        for name, pd in params.items():
            mod = getattr(self, name)
            for attr, arr in pd.items():
                
                val = getattr(mod, attr).value
                if val is None: continue
                
                arr = jnp.asarray(arr, dtype=val.dtype)
                if arr.shape != val.shape:
                    raise ValueError(f"Shape mismatch for {name}.{attr}: {arr.shape} vs {val.shape}")
                getattr(mod, attr).value = arr
    
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
        ginv = 1.0 / (gy + 1e-8)
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
    
    @property
    def attrs(self):
        return ("y_proj", "z_proj", "stable_m", "invari_m", "nondegenerate_enc", "total")



class Integrator(PyTreeNode):
    solver: Callable
    ts: jnp.array = field(pytree_node=False)

    @property
    def dt0(self) -> float: return self.ts[1] - self.ts[0]

    @property
    def n_steps(self) -> int: return len(self.ts) - 1


    def apply(self, x0s, rom: DoubleIntegratorROM, policy_fun: Callable=None):
        ''' Integrate the dynamics from self.ts[0] to self.ts[-1] with initial condition x0s.
        Input:
            x0s: (B, 2)
        Output:
            int_out: IntegratorOutput containing xs (B,T+1,2) and us (B,T,1)
        '''
        B = x0s.shape[0]
        if policy_fun is None:
            policy_fun = rom.policy_v
        
        int_out = IntegratorOutput(
            xs=jnp.zeros((B, self.n_steps+1, 2)),  # (B, T+1, 2)
            us=jnp.zeros((B, self.n_steps, 1))     # (B, T, 1)
        )
        
        int_out = int_out.replace(xs=int_out.xs.at[:,0].set(x0s))
        
        def step(i, carry):
            int_out = carry
            x_curr = int_out.xs[:, i]                          # (B,2)
            y, z   = vmap(rom.encode, in_axes=0)(x_curr)  # (B,1),(B,1)
            v      = vmap(policy_fun, in_axes=(0, 0))(y, z)    # (B,1)
            u      = vmap(rom.map_v_to_u, in_axes=0)(v)

            def _term(t, x, args):
                return vmap(rom.dyn_x, in_axes=(0, 0))(x, u)     # (B,2)

            sol = self.solver(
                dfx.ODETerm(_term),
                dt0=self.dt0, t0=self.ts[i], t1=self.ts[i+1], y0=x_curr, args=None
            )
            return int_out.replace(
                xs=int_out.xs.at[:, i + 1].set(sol.ys[-1]),
                us=int_out.us.at[:, i].set(u),
            )
        
        return jax.lax.fori_loop(0, self.n_steps, step, int_out)
    
    def post_apply(self, int_out: IntegratorOutput, rom: DoubleIntegratorROM) -> IntegratorAuxOutput:
        ''' Augment IntegratorOutput with debug information. 
        Note: last time step in `xs` is truncated. So both `xs` and `us` have shape (B,T,*).
        '''        
        xs, us = int_out.xs, int_out.us
        B, T = us.shape[:2]
        xs_t = xs[:, :-1, :]
        flat_xs = rearrange(xs_t, 'b t d -> (b t) d')

        ys, zs = vmap(rom.encode)(flat_xs)
        vs     = vmap(rom.policy_v)(ys, zs)
        psis   = vmap(rom.policy_psi)(zs)
        es     = jnp.abs(ys - psis)
        lyaps  = vmap(rom.lyap)(zs)

        return IntegratorAuxOutput(
            xs_t,
            rearrange(ys,    '(b t) d -> b t d', b=B),
            rearrange(zs,    '(b t) d -> b t d', b=B),
            us,
            rearrange(vs,    '(b t) d -> b t d', b=B),
            rearrange(es,    '(b t) d -> b t d', b=B),
            self.ts,
            rearrange(lyaps, '(b t) d -> b t d', b=B),
        )
    
    def compute_loss(self, int_out: IntegratorOutput, rom: DoubleIntegratorROM) -> LossOutput:
        xs, us = int_out.xs, int_out.us
        B, T = us.shape[:2]
        xs_t = xs[:, :-1, :]
        flat_xs = rearrange(xs_t, 'b t d -> (b t) d')
        flat_us = rearrange(us,   'b t d -> (b t) d')

        ys, zs = vmap(rom.encode)(flat_xs)
        l_y_proj  = vmap(rom.loss_y_proj, in_axes=(0,0))(flat_xs, flat_us) * rom.cfg_rom.loss_scale_dict["y_proj"]
        l_z_proj  = vmap(rom.loss_z_proj, in_axes=(0,0))(flat_xs, flat_us) * rom.cfg_rom.loss_scale_dict["z_proj"]
        l_stab    = vmap(rom.loss_stable_m)(zs) * rom.cfg_rom.loss_scale_dict["stable_m"]
        l_invari  = vmap(rom.loss_invari_m)(zs) * rom.cfg_rom.loss_scale_dict["invari_m"]
        l_nondec  = vmap(rom.loss_nondegenerate_enc)(flat_xs) * rom.cfg_rom.loss_scale_dict["nondegenerate_enc"]
        total     = l_y_proj + l_z_proj + l_stab + l_invari + l_nondec

        return LossOutput(
            y_proj=rearrange(l_y_proj, '(b t) d -> b t d', b=B),
            z_proj=rearrange(l_z_proj, '(b t) d -> b t d', b=B),
            stable_m=rearrange(l_stab, '(b t) d -> b t d', b=B),
            invari_m=rearrange(l_invari, '(b t) d -> b t d', b=B),
            nondegenerate_enc=rearrange(l_nondec, '(b t) d -> b t d', b=B),
            total=rearrange(total, '(b t) d -> b t d', b=B),
        )