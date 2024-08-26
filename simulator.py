import jax
import flax
import jax.lax
import numpy as np
import diffrax as de
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable
import jax_dataclasses as jdc

from robots import HopperH2H


@flax.struct.dataclass
class IntegratorOutput:
    xs: jnp.ndarray
    us: jnp.ndarray


class ClosedLoopIntegrator(flax.struct.PyTreeNode):
    policy: nn.Module
    robot: nn.Module
    ts: jdc.Static[np.ndarray]
    de_solver: Callable
    @property
    def dt0(self):
        return self.ts[1] - self.ts[0]

    @property
    def n_steps(self):
        return len(self.ts) - 1

    def init(self, rng, t, x0, mutable=flax.core.DenyList('intermediates')):
        _, policy_rng, robot_rng = jax.random.split(rng, 3)
        out = self.policy.init_with_output(policy_rng, t, x0, mutable=mutable)
        if isinstance(out, tuple) and len(out) == 2:
            u, policy_params = out
        elif not isinstance(out, tuple):
            # if there are no parameters init_with_output only returns the output
            policy_params = {}
            u = out
        else:
            raise ValueError("policy init_with_output should return either 1 or 2 values")
        if "intermediates" in policy_params and "plan" in policy_params["intermediates"]:
            policy_params['intermediates']['plan'] = jax.tree_map(
                lambda x: jnp.repeat(jnp.zeros_like(x)[None], axis=0, repeats=self.n_steps),
                policy_params['intermediates']['plan'][0])
        return {
            "policy": policy_params,
            "robot": self.robot.init(robot_rng, t, x0, u),
            "iter": 0
        }
    def apply(self, params, x0, mutable=False):
        int_out = IntegratorOutput(
            xs=jnp.zeros((self.n_steps+1, self.robot.state_dim)),
            us=jnp.zeros((self.n_steps, self.robot.action_dim))
        )
        int_out = int_out.replace(xs=int_out.xs.at[0].set(x0))
        # for initialization only
        def body(i, carry_params):
            carry, params = carry_params
            policy_params = params["policy"]
            robot_params = params["robot"]
            in_params = dict()
            for k, v in policy_params.items():
                if k != "intermediates":
                    in_params[k] = v
                else:
                    in_params["intermediates"] = {"plan": tuple()}
            if mutable:
                u, policy_out_state = self.policy.apply(in_params, self.ts[i], carry.xs[i],mutable=mutable)
                new_plan = policy_out_state["intermediates"]["plan"][0]
                policy_params["intermediates"]["plan"] = \
                    policy_params["intermediates"]["plan"].replace(
                    xk=policy_params["intermediates"]["plan"].xk.at[i].set(new_plan.xk),
                    uk=policy_params["intermediates"]["plan"].uk.at[i].set(new_plan.uk)
                )
            else:
                u = self.policy.apply(policy_params, self.ts[i], carry.xs[i],
                                      mutable=mutable)
            def vf(t, x, args):
                return self.robot.apply(args[0], t, x, args[1])

            sol = self.de_solver(
                de.ODETerm(vf),
                dt0=self.dt0,
                t0=self.ts[i],
                t1=self.ts[i + 1],
                y0=carry.xs[i],
                args=(robot_params, u)
            )
            return (carry.replace(xs=carry.xs.at[i + 1].set(sol.ys[-1]),
                                  us=carry.us.at[i].set(u)),
                    params)

        # loop_out = (int_out, params)
        # for i in range(self.n_steps):
        #     loop_out = body(i, loop_out)
        loop_out = jax.lax.fori_loop(
            lower=0,
            upper=self.n_steps+1,
            body_fun=body,
            init_val=(int_out, params))
        if mutable:
            return loop_out
        else:
            return loop_out[0]


@flax.struct.dataclass
class TangentSpaceMinIntegratorOutput:
    dphidz_zstardot: jnp.ndarray
    eta_star_dot: jnp.ndarray
    z: jnp.ndarray
    nd: jnp.ndarray


class TangentSpaceMinClosedLoopIntegrator(flax.struct.PyTreeNode):
    psi_policy: nn.Module
    control_policy: nn.Module
    dyn: nn.Module

    def init(self, rng, t, x0, mutable=flax.core.DenyList('intermediates')):
        _, psi_policy_rng, control_policy_rng, robot_rng = jax.random.split(rng, 4)
        robot_params = self.dyn.init(robot_rng, t, x0, jnp.zeros((self.dyn.action_dim,)))
        n0, z0 = self.dyn.phi(t, x0)
        etad, psi_policy_params = self.psi_policy.init_with_output(psi_policy_rng, t, x0,
                                                        mutable=mutable)
        u, control_policy_params = self.control_policy.init_with_output(control_policy_rng, t, n0, z0,
                                                        mutable=mutable)
        if "intermediates" in psi_policy_rng and "plan" in psi_policy_rng["intermediates"]:
            psi_policy_rng['intermediates']['plan'] = jax.tree_map(
                lambda x: jnp.repeat(jnp.zeros_like(x)[None], axis=0, repeats=self.n_steps),
                psi_policy_rng['intermediates']['plan'][0])
        if "intermediates" in control_policy_rng and "plan" in control_policy_rng["intermediates"]:
            control_policy_rng['intermediates']['plan'] = jax.tree_map(
                lambda x: jnp.repeat(jnp.zeros_like(x)[None], axis=0, repeats=self.n_steps),
                control_policy_rng['intermediates']['plan'][0])
        return {
            "control_policy": control_policy_params,
            "psi_policy": psi_policy_params,
            "robot": robot_params
        }

    def apply(self, params, z):
        eta = self.psi_policy.apply(params["psi_policy"], z, method="eta_d")
        u_star = self.control_policy.apply(params["control_policy"], 0, eta, z)
        nz_dot = self.dyn.apply(params["robot"], 0, self.dyn.nz_join(eta, z), u_star, method="nz_dynamics")
        eta_star_dot, z_star_dot = self.dyn.nz_split(nz_dot)
        _, dphidz_zstardot = jax.jvp(
            lambda z_: self.psi_policy.apply(params["psi_policy"], z_, method="eta_d"),
            (z,), (z_star_dot,)
        )
        return TangentSpaceMinIntegratorOutput(dphidz_zstardot=dphidz_zstardot, eta_star_dot=eta_star_dot,z=z,nd=eta)


@flax.struct.dataclass
class DiscreteInvarianceIntegratorOutput(flax.struct.PyTreeNode):
    z: jnp.ndarray
    u: jnp.ndarray
    eta: jnp.ndarray
    z_p: jnp.ndarray
    eta_p: jnp.ndarray
    psi_zp: jnp.ndarray


class DiscreteInvarianceIntegrator(flax.struct.PyTreeNode):
    psi_policy: nn.Module
    control_policy: nn.Module
    dyn: nn.Module

    def init(self, rng, t, x0):
        n0, z0 = HopperH2H.nz_from_x(x0)
        _, psi_policy_rng, control_policy_rng, robot_rng = jax.random.split(rng, 4)
        eta_d, psi_policy_params = self.psi_policy.init_with_output(psi_policy_rng, t, x0)
        x0 = HopperH2H.x_from_nz(eta_d, z0)
        robot_params = self.dyn.init(robot_rng, t, x0, jnp.zeros((self.dyn.d_m,)))

        # control_policy_params = self.control_policy.init(control_policy_rng, t, eta_d, z0)
        # return {
        #     "control_policy": control_policy_params,
        #     "psi_policy": psi_policy_params,
        #     "robot": robot_params
        # }

        return {
            "control_policy": {},
            "psi_policy": psi_policy_params,
            "robot": robot_params
        }

    def apply(self, params, z):
        eta = self.psi_policy.apply(params["psi_policy"], z, method="psi")
        u_star = self.control_policy.apply(params["control_policy"], 0, eta, z)
        x = HopperH2H.x_from_nz(eta, z)
        xp = self.dyn.apply(params["robot"], 0, x, u_star)
        eta_p, zp = HopperH2H.nz_from_x(xp)
        psi_zp = self.psi_policy.apply(params["psi_policy"], zp, method="psi")
        return DiscreteInvarianceIntegratorOutput(z=z, u=u_star, eta=eta, z_p=zp, eta_p=eta_p, psi_zp=psi_zp)


@flax.struct.dataclass
class SupervisedLossOut:
    eta_d: jnp.ndarray
    eta_true: jnp.ndarray


class SupervisedLossIntegrator(flax.struct.PyTreeNode):
    mlp: nn.Module

    def init(self, rng, t, z0):
        _, mlp_rng = jax.random.split(rng)
        params = self.mlp.init(mlp_rng, z0)
        return {"mlp": params}

    def apply(self, params, zu):
        z = zu[0]
        eta_true = zu[1]
        eta_d = self.mlp.apply(params["mlp"], z)
        return SupervisedLossOut(eta_true=eta_true, eta_d=eta_d)
