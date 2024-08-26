from jax import numpy as jnp
import jaxlie as lie
import jax


def pendulum_error(pred_x, true_x):
    true_theta = lie.SO2.from_radians(true_x[0])
    pred_theta = lie.SO2.from_radians(pred_x[0])
    true_theta_dot = true_x[1]
    pred_theta_dot = pred_x[1]

    theta_error = jnp.sum((true_theta.inverse() @ pred_theta).log() ** 2)
    theta_dot_error = (true_theta_dot - pred_theta_dot) ** 2
    return theta_error + theta_dot_error


def cartpole_error(pred_x, true_x):
    true_theta = lie.SO2.from_radians(true_x[1])
    pred_theta = lie.SO2.from_radians(pred_x[1])

    theta_error = jnp.sum((true_theta.inverse() @ pred_theta).log() ** 2)
    linear_error = (true_x[0] - pred_x[0])**2 + (true_x[2] - pred_x[2])**2 + (true_x[3] - pred_x[3])**2
    return theta_error + linear_error


def invariance_loss(integrator):

    batched_integrator = jax.vmap(integrator.apply, in_axes=(None, 0))

    def loss(params, z0s):
        out = batched_integrator(params, z0s)
        loss_ = jnp.mean(jax.vmap(lambda doteta, dphi_dz_zdot: jnp.mean(jnp.square(doteta - dphi_dz_zdot)))(out.eta_star_dot, out.dphidz_zstardot))
        return loss_, out

    return integrator, loss


def discrete_invariance_loss(integrator):
    batched_integrator = jax.vmap(integrator.apply, in_axes=(None, 0))

    def loss(params, z0s):
        out = batched_integrator(params, z0s)
        loss_ = jnp.mean(jnp.square((out.eta_p - out.psi_zp)))
        return loss_, out

    return integrator, loss


def supervised_loss(integrator):

    batched_integrator = jax.vmap(integrator.apply, in_axes=(None, 0))

    def loss(params, batch):
        out = batched_integrator(params, batch)
        # loss_ = jnp.mean((out.eta_d - out.eta_true) ** 2)
        loss_ = jnp.mean(jnp.abs(out.eta_d - out.eta_true))
        return loss_, out


    return integrator, loss
