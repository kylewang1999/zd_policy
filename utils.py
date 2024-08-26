import wandb
import numpy as np
import jax
import jax_dataclasses as jdc
from flax import linen as nn
from jax import numpy as jnp
from matplotlib.colors import LogNorm
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
from plots import plot_2D_irregular_heatmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from jaxopt import BoxCDQP


@jdc.pytree_dataclass
class InputBounds:
    lower: jdc.Static[np.array]
    upper: jdc.Static[np.array]

    def __call__(self, x):
        x = nn.tanh(x)
        x = ((self.upper - self.lower) / 2) * x
        x = x + ((self.upper + self.lower) / 2)
        return x

    def clip(self, x):
        return jnp.clip(x, self.lower, self.upper)


@jdc.pytree_dataclass
class HopperInputBounds:
    lower: jdc.Static[np.array]
    upper: jdc.Static[np.array]
    rev_time: bool = False

    def __call__(self, x, u):
        u = nn.tanh(u)
        ub = self.ub(x)
        lb = self.lb(x)
        u = ((ub - lb) / 2) * u
        u = u + ((ub + lb) / 2)
        return u

    def qp(self, R, x, u):
        sol = BoxCDQP().run(u,
                            params_obj=(2*R, (u)),
                            params_ineq=(self.lb(x), self.ub(x)))
        return sol.params


    def clip(self, x, u):
        return jnp.clip(u, self.lb(x), self.ub(x))

    def lb(self, x):
        w_critical = 511
        w_max = 600
        max_torque = 2.0
        slope = - max_torque / (w_max - w_critical)
        ws = x[-3:]
        if self.rev_time:
            ws = -ws
        return jnp.where(ws > -w_critical,
                         self.lower,
                         jnp.where(
                             ws > -w_max,
                             jnp.maximum(slope * (ws + w_max),
                                         self.lower),
                             0.0))

    def ub(self, x):
        w_critical = 511
        w_max = 600
        max_torque = 2.0
        slope = - max_torque / (w_max - w_critical)
        ws = x[-3:]
        if self.rev_time:
            ws = -ws
        return jnp.where(ws < w_critical,
                         self.upper,
                         jnp.where(
                             ws < w_max,
                             jnp.minimum(slope * (ws - w_max),
                                         self.upper),
                             0.0))

@jdc.pytree_dataclass
class AngleRepresentation:
    angle_mask: jdc.Static[np.array]

    def __call__(self, x):
        angles = x[..., self.angle_mask]
        not_angles = x[..., ~self.angle_mask]
        x = jnp.concatenate([jnp.sin(angles), 1-jnp.cos(angles), not_angles], axis=-1)
        return x
@jdc.pytree_dataclass
class NormalizedAngleRepresentation:
    angle_mask: jdc.Static[np.array]
    nangle_bounds: jdc.Static[np.array] # maximum absolute value of the input x

    def __call__(self, x):
        angles = x[..., self.angle_mask]
        not_angles = x[..., ~self.angle_mask] / self.nangle_bounds
        x = jnp.concatenate([jnp.sin(angles), 1-jnp.cos(angles), not_angles], axis=-1)
        return x


def unnormalize_dict(normalized_dict, sep="/"):
    result = {}
    for key, value in normalized_dict.items():
        keys = key.split(sep)
        d = result
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return result


def scalar_summary(name, value, **kwargs):
    shape = value.shape
    if len(shape) == 0:
        wandb.log({name: value}, **kwargs)
    elif len(shape) == 2:
        # assume batch of trajectories
        # integrate trajectory
        # mean over trajectories
        wandb.log({name: value.sum(axis=-1).mean()}, **kwargs)
    else:
        # assume batch of scalars, mean over them
        wandb.log({name: value.mean()}, **kwargs)

def compton_value_summary(name, value, **kwargs):
    zs = value.int_out.z
    nd = value.int_out.nd

    if kwargs['step'] % 100 != 0:
        return
    fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2, figsize=(10, 5))
    cax0 = make_axes_locatable(ax0).append_axes("right", size="5%", pad=0.05)
    im_diff = plot_2D_irregular_heatmap(zs[:, 0], zs[:, 1],
                                        nd[:, 0], ax=ax0)

    fig.colorbar(im_diff, cax=cax0)
    cax1 = make_axes_locatable(ax1).append_axes("right", size="5%", pad=0.05)
    im_diff = plot_2D_irregular_heatmap(zs[:, 0], zs[:, 1],
                                        nd[:, 1], ax=ax1)

    fig.colorbar(im_diff, cax=cax1)
    plt.tight_layout()
    wandb.log({"Pendulum Value": wandb.Image(fig)})
    plt.close(fig)

def cartpole_tangent_invariance_summary(name, value, **kwargs):

    if kwargs["step"] % 10 != 1:
        return

    # dphidz_zstardot = value.dphidz_zstardot
    # eta_star_dot = value.eta_star_dot
    # z = value.z
    # nd = value.nd
    #
    # loss = jnp.abs(eta_star_dot - dphidz_zstardot)
    # fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # plot_2D_irregular_heatmap(z[:, 0].flatten(),
    #                           z[:, 1].flatten(),
    #                           nd[:, 0].flatten(), ax=ax0)
    # ax0.set_xlabel('Nq = th')
    # ax0.set_ylabel('NDdotq ~ thdot')
    # plot_2D_irregular_heatmap(z[:, 0].flatten(),
    #                           z[:, 1].flatten(),
    #                           nd[:, 1].flatten(), ax=ax1)
    # ax1.set_xlabel('Nq = th')
    # ax1.set_ylabel('NDdotq ~ thdot')
    # plt.tight_layout()
    # wandb.log({"Cartpole Policy": wandb.Image(fig)}, **kwargs)
    # plt.close(fig)
    #
    # fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # plot_2D_irregular_heatmap(z[:, 0].flatten(),
    #                           z[:, 1].flatten(),
    #                           loss[:, 0].flatten(), ax=ax0)
    # ax0.set_xlabel('Nq = th')
    # ax0.set_ylabel('NDdotq ~ thdot')
    # plot_2D_irregular_heatmap(z[:, 0].flatten(),
    #                           z[:, 1].flatten(),
    #                           loss[:, 1].flatten(), ax=ax1)
    # ax1.set_xlabel('Nq = th')
    # ax1.set_ylabel('NDdotq ~ thdot')
    # plt.tight_layout()
    # wandb.log({"Cartpole Loss": wandb.Image(fig)}, **kwargs)
    # plt.close(fig)

def hopper_zd_value_summary(name, value, plot_every_n, **kwargs):
    # z = value.z.reshape((value.z.shape[0]*value.z.shape[1],-1))
    # eta = value.eta.reshape((value.eta.shape[0]*value.eta.shape[1],-1))
    # u = value.u.reshape((value.u.shape[0]*value.u.shape[1],-1))
    zp = value.z_p
    eta_p = value.eta_p
    psi_zp = value.psi_zp
    z = value.z
    eta = value.eta

    # np.savetxt(f"data/z{kwargs['step']}.csv", z, delimiter=',')
    # np.savetxt(f"data/eta{kwargs['step']}.csv", eta, delimiter=',')
    # np.savetxt(f"data/u{kwargs['step']}.csv", u, delimiter=',')
    # np.savetxt(f"data/zp{kwargs['step']}.csv", zp, delimiter=',')
    # np.savetxt(f"data/etap{kwargs['step']}.csv", eta_p, delimiter=',')
    # np.savetxt(f"data/psizp{kwargs['step']}.csv", psi_zp, delimiter=',')

    # First step is 1, !=0 won't trigger the first time
    if kwargs['step'] % plot_every_n != 1:
        return

    # fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2, figsize=(10, 5))
    # cax0 = make_axes_locatable(ax0).append_axes("right", size="5%", pad=0.05)
    # im_diff = plot_2D_irregular_heatmap(z[:, 0], z[:, 2], eta[:, 0], ax=ax0)
    # plt.xlabel('y')
    # plt.ylabel('doty')
    # fig.colorbar(im_diff, cax=cax0)
    # cax1 = make_axes_locatable(ax1).append_axes("right", size="5%", pad=0.05)
    # im_diff = plot_2D_irregular_heatmap(z[:, 1], z[:, 3], eta[:, 1], ax=ax1)
    # plt.xlabel('y')
    # plt.ylabel('doty')
    # fig.colorbar(im_diff, cax=cax1)
    # plt.tight_layout()
    # wandb.log({"Hopper Policy": wandb.Image(fig)}, **kwargs)
    # plt.close(fig)
    #
    # err = jnp.abs(eta_p - psi_zp)
    # fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # cax0 = make_axes_locatable(ax0).append_axes("right", size="5%", pad=0.05)
    # im_diff = ax0.scatter(zp[:, 0], zp[:, 2], c=err[:, 0])
    # plt.xlabel('y')
    # plt.ylabel('doty')
    # fig.colorbar(im_diff, cax=cax0)
    # cax1 = make_axes_locatable(ax1).append_axes("right", size="5%", pad=0.05)
    # im_diff = ax1.scatter(zp[:, 1], zp[:, 3], c=err[:, 1])
    # plt.xlabel('y')
    # plt.ylabel('doty')
    # fig.colorbar(im_diff, cax=cax1)
    # plt.tight_layout()
    # wandb.log({"Hopper Loss": wandb.Image(fig)}, **kwargs)
    # plt.close(fig)

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    plot_2D_irregular_heatmap(z[:, 0].flatten(),
                              z[:, 2].flatten(),
                              eta[:, 0].flatten(), ax=ax0)
    # plot_scatter_value(z[:, 0], z[:, 2], eta[:, 0], ax=ax0, resolution=128,
    #                    norm=None, default_value=np.nan)
    ax0.set_xlabel('y')
    ax0.set_ylabel('doty')
    plot_2D_irregular_heatmap(z[:, 1].flatten(),
                              z[:, 3].flatten(),
                              eta[:, 1].flatten(), ax=ax1)
    # plot_scatter_value(z[:, 1], z[:, 3], eta[:, 1], ax=ax1, resolution=128,
    #                    norm=None, default_value=np.nan)
    ax1.set_xlabel('y')
    ax1.set_ylabel('doty')
    plt.tight_layout()
    wandb.log({"Hopper Policy": wandb.Image(fig)}, **kwargs)
    plt.close(fig)


def pendulum_value_summary(name, value, **kwargs):
    xs = value.int_out.xs
    vs = value.bts
    # normalize angles
    # theta = jnp.arctan2(jnp.sin(xs[:, 0]), jnp.cos(xs[:, 0]))
    # xs = xs.at[:, :, 0].set(theta)
    # select only reasonable velocities
    # vel_filter =( -5 <= xs[: ,: ,1]) & (xs[: ,: ,1] <= 5 )
    # xs = xs[vel_filter]
    # vs = value.train_terms.val[vel_filter]

    # xs = xs.reshape(-1, 2)
    # vs = vs.flatten()
    if kwargs['step']  % 500 != 0:
        return
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1,ncols=3, figsize=(15, 5))
    cax0 = make_axes_locatable(ax0).append_axes("right", size="5%", pad=0.05)
    diff = value.int_out.vs - value.train_terms.val
    im_diff = plot_2D_irregular_heatmap(xs[:, 0], xs[:, 1], diff, ax=ax0)
    fig.colorbar(im_diff, cax=cax0)
    cax1 = make_axes_locatable(ax1).append_axes("right", size="5%", pad=0.05)
    im_v_pred = plot_2D_irregular_heatmap(xs[:, 0], xs[:, 1], value.train_terms.val, ax=ax1)
    fig.colorbar(im_v_pred, cax=cax1)
    ax2.scatter(xs[:, 0], xs[:, 1], s=1, alpha=0.1, c=value.int_out.vs)
    plt.tight_layout()
    wandb.log({"Pendulum Value": wandb.Image(fig)})
    plt.close(fig)
    # Adversarial Summary
    # fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1,ncols=4, figsize=(20, 5))
    # cax0 = make_axes_locatable(ax0).append_axes("right", size="5%", pad=0.05)
    # diff = value.int_out.vs - value.train_terms.val
    # im_diff = plot_2D_irregular_heatmap(xs[:, 0], xs[:, 1],
    #                                     diff, ax=ax0)
    #
    # fig.colorbar(im_diff, cax=cax0)
    # cax1 = make_axes_locatable(ax1).append_axes("right", size="5%", pad=0.05)
    # im_v_pred = plot_2D_irregular_heatmap(xs[:, 0], xs[:, 1], value.train_terms.val, ax=ax1)
    # fig.colorbar(im_v_pred, cax=cax1)
    # ax2.scatter(xs[:, 0], xs[:, 1], s=1, alpha=0.1, c=value.int_out.vs)
    # cax3 = make_axes_locatable(ax3).append_axes("right", size="5%", pad=0.05)
    # im_mu_pred = plot_2D_irregular_heatmap(xs[:, 0], xs[:, 1], value.mus, ax=ax3)
    # fig.colorbar(im_mu_pred, cax=cax3)
    # plt.tight_layout()
    # wandb.log({"Pendulum Value": wandb.Image(fig)})
    # plt.close(fig)

def pendulum_dvdx_summary(name, value, **kwargs):
    # normalize angles
    # theta = jnp.arctan2(jnp.sin(xs[:, 0]), jnp.cos(xs[:, 0]))
    # xs = xs.at[:, :, 0].set(theta)
    # select only reasonable velocities
    # vel_filter =( -5 <= xs[: ,: ,1]) & (xs[: ,: ,1] <= 5 )
    # xs = xs[vel_filter]
    # vs = value.train_terms.val[vel_filter]

    # xs = xs.reshape(-1, 2)
    # vs = vs.flatten()
    if kwargs['step'].item() % 100 != 0:
        return
    xs = value.int_out.xs
    vs = value.bts
    ps = value.int_out.ps

    p_thresh = 1e3
    ps_thresh = jnp.abs(ps)
    ps_thresh = ps_thresh.at[abs(ps_thresh[:, 0]) < p_thresh, 0].set(jnp.nan)
    ps_thresh = ps_thresh.at[abs(ps_thresh[:, 1]) < p_thresh, 1].set(jnp.nan)

    dv_dx = value.train_terms.dv_dx
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1,ncols=4, figsize=(20, 5))
    cax0 = make_axes_locatable(ax0).append_axes("right", size="5%", pad=0.05)
    im_dv_dx1 = ax0.scatter(xs[:, 0], xs[:, 1],norm=LogNorm(), c=dv_dx[:, 0],s=7,alpha=0.5)

    fig.colorbar(im_dv_dx1, norm=LogNorm(), cax=cax0)
    cax1 = make_axes_locatable(ax1).append_axes("right", size="5%", pad=0.05)
    im_ps1 = ax1.scatter(xs[:, 0], xs[:, 1], norm=LogNorm(), c=ps_thresh[:, 0],s=7,alpha=0.5)
    fig.colorbar(im_ps1,  cax=cax1)
    cax2 = make_axes_locatable(ax2).append_axes("right", size="5%", pad=0.05)
    im_dv_dx2 = ax2.scatter(xs[:, 0], xs[:, 1], norm=LogNorm(), c=dv_dx[:, 1],s=7,alpha=0.5)
    fig.colorbar(im_dv_dx2, norm=LogNorm(), cax=cax2)
    cax3 = make_axes_locatable(ax3).append_axes("right", size="5%", pad=0.05)
    im_ps2 = ax3.scatter(xs[:, 0], xs[:, 1],norm=LogNorm(), c=ps_thresh[:,1],s=7,alpha=0.5)
    fig.colorbar(im_ps2, norm=LogNorm(), cax=cax3)
    plt.tight_layout()
    wandb.log({"Pendulum Value": wandb.Image(fig)})
    plt.close(fig)

def make_log_bins(x, num_bins):
    return jnp.logspace(jnp.log10(x.min()), jnp.log10(x.max()), num_bins)


def hopper_policy_summary(name, value, ts, log_every_n_steps=100, **kwargs):
    if kwargs['step'] % log_every_n_steps != 0:
        return
    out = value.int_out
    n_batch = out.xs.shape[0]
    n_ts = ts.shape[0]
    bts = ts[None].repeat(out.xs.shape[0], axis=0).flatten()
    u_bts = ts[None, :-1].repeat(out.us.shape[0], axis=0).flatten()
    n_y_bins = 64
    state_titles = ["Roll", "Pitch", "Yaw",
              "Dot Roll", "Dot Pitch",
              "Dot Yaw", "fw1", "fw2", "fw3"]
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 20), )
    axs = axs.flatten()
    s_axs = axs[:-3]

    for i, ax in enumerate(s_axs):
        ax.hist2d(bts, out.xs[:, :, i].flatten(), bins=(n_ts, n_y_bins),
                  norm=LogNorm())
        ax.set_title(state_titles[i])
    u_axs = axs[-3:]
    for i, ax in enumerate(u_axs):
        ax.hist2d(u_bts, out.us[:, :, i].flatten(), bins=(n_ts-1, n_y_bins),
                  norm=LogNorm())
        ax.set_title(f"Action {i}")
    plt.tight_layout()
    wandb.log({"Hopper Policy Statistics": wandb.Image(fig)})
    plt.close(fig)



def no_summary(name, value, **kwargs):
    pass


def flat_meshgrid(l1, u1, l2, u2, N=100):
    z1s = jnp.linspace(l1, u1, N)
    z2s = jnp.linspace(l2, u2, N)
    z1s, z2s = jnp.meshgrid(z1s, z2s, indexing='xy')
    z0s = jnp.vstack((jnp.ravel(z1s, order='C'), jnp.ravel(z2s, order='C'))).T
    return z0s

@jax.custom_jvp
def diff_relu(x):
    return jax.nn.relu(x)


@diff_relu.defjvp
def diff_relu_jvp(primals, tangents):
    x = primals[0]
    x_dot = tangents[0]
    y = diff_relu(x)
    return y, x_dot

@jax.custom_jvp
def orth_box_qp(Q_diag, q, lower, upper):
    u = jnp.clip(-q / (Q_diag*2), lower, upper)
    return u

@orth_box_qp.defjvp
def orth_box_qp_jvp(primals, tangents):
    Q_diag, q, lower, upper = primals
    Q_diag_dot, q_dot, lower_dot, upper_dot = tangents
    u = orth_box_qp(Q_diag, q, lower, upper)
    lambda_vals = 2*Q_diag * u + q
    lambda_upper = (lambda_vals < 0) * lambda_vals
    lambda_lower = (lambda_vals > 0) * lambda_vals
    z_n = jnp.zeros_like(u)
    ds_lower_active = jnp.concatenate([])
    u == lower
    u == upper
    ds = jnp.select(condlist=[
            u == lower,
            u == upper
        # ,
        #     u != lower and u != upper,
        ],
        choicelist=[
            jnp.concatenate([lower_dot,
                             q_dot + 2 * (lower_dot * Q_diag + Q_diag_dot * u),
                             z_n],
                            axis=0),
            jnp.concatenate([upper_dot,
                             z_n,
                             -q_dot - 2 * (upper_dot * Q_diag + Q_diag_dot * u)],
                            axis=0)],
        default=jnp.concatenate([
            -(q_dot + 2 * (Q_diag_dot * u))/ (2 * Q_diag),
            z_n,
            z_n
        ], axis=0)
    )

    # Q = jnp.diag(Q_diag)
    #
    # z_nxn = jnp.zeros_like(Q)
    # I_nxn = jnp.eye(Q_diag.shape[0])
    # KKT = jnp.block([
    #     [2*Q, -I_nxn, I_nxn],
    #     [-jnp.diag(lambda_lower), jnp.diag(lower - u), z_nxn],
    #     [jnp.diag(lambda_upper), z_nxn, jnp.diag(u - upper)]
    # ])
    #
    # rhs = -jnp.block([
    #     [2*u * Q_diag_dot + q_dot],
    #     [lambda_lower * lower_dot],
    #     [-lambda_upper * upper_dot]
    # ])
    # # although this should use lst_sq, it does not support forward mode AD yet.
    # ds = jnp.linalg.pinv(KKT) @ rhs
    # # ds = jnp.linalg.lstsq(KKT, rhs)[0]
    # # return u, ds[:u.shape[0], 0]


    Q_inv = jnp.diag(1 / (2*Q_diag))
    du = jnp.block([jnp.eye(Q_diag.shape[0]), -Q_inv, Q_inv]) @ ds
    # def check_is_nan_or_inf(_du):
    #     if jnp.any(jnp.isnan(_du)) or jnp.any(jnp.isinf(_du)):
    #         raise ValueError("du is nan")
    #
    # jax.debug.callback(check_is_nan_or_inf, _du=du)

    return u, du[:, 0]


def tfMLP(mlp, params: dict, input_size: int, activation):
    if activation == 'flax.linen.relu':
        activation = keras.activations.relu
    elif activation == 'flax.linen.tanh':
        activation = keras.activations.tanh
    elif activation == 'flax.linen.softplus':
        activation = keras.activations.softplus
    else:
        raise ValueError(f'Tensorflow activation corresponding to {activation} not supported')
    layers = []
    for ii in range(mlp.n_layers):
        if ii == 0:
            layers.append(keras.layers.Dense(mlp.n_hidden, batch_size=1, input_shape=(input_size,), activation=activation))
        else:
            layers.append(keras.layers.Dense(mlp.n_hidden, batch_size=1, input_shape=(mlp.n_hidden,), activation=activation))
    layers.append(keras.layers.Dense(mlp.n_outputs, batch_size=1, input_shape=(mlp.n_hidden,)))
    tf_model = keras.models.Sequential(layers)

    for ii in range(len(tf_model.layers)):
        tf_model.layers[ii].set_weights([params[f'Dense_{ii}']['kernel'], params[f'Dense_{ii}']['bias']])

    return tf_model


def tfZeroInvariantMLP(mlp, params: dict, input_size: int, activation, clip=None):

    class TFZeroInvMLP(keras.Model):

        def __init__(self, mlp, offset):
            super().__init__()
            self.mlp = mlp
            self.offset = offset
            self.clip = clip

        def call(self, inputs, training=False):
            out = self.mlp(inputs) - self.offset
            if self.clip is not None:
                out = tf.clip_by_value(out, -self.clip, self.clip)
            return out

    tfmlp = tfMLP(mlp, params, input_size, activation)
    offset = tfmlp(tf.zeros((1, input_size)))
    tf_zeroinvmlp = TFZeroInvMLP(tfmlp, offset)

    return tf_zeroinvmlp


def hardshrink(x, lambd=0.5):
    return jnp.where(jnp.abs(x) < lambd, 0, x - jnp.sign(x) * lambd)