import functools

import jax.numpy as jnp
import jax
import jax_dataclasses as jdc
# import mujoco
import numpy as np
from flax import linen as nn


class AbstractZeroDynamics:
    state_dim: int
    action_dim: int
    n_dim: int
    z_dim: int

    def f(self, t, x):
        raise NotImplementedError

    def g(self, t, x):
        raise NotImplementedError

    def phi(self, t, x):
        raise NotImplementedError

    def phi_inv(self, t, eta, z):
        raise NotImplementedError

    def phi_z(self, t, x):
        _, z = self.phi(t, x)
        return z

    def phi_eta(self, t, x):
        eta, _ = self.phi(t, x)
        return eta

    def nz_n(self, nz):
        return nz[:self.n_dim]

    def nz_z(self, nz):
        return nz[self.n_dim:]

    @staticmethod
    def nz_join(n, z):
        return jnp.hstack((n, z))

    def nz_split(self, nz):
        return self.nz_n(nz), self.nz_z(nz)

    def omega(self, t, nz):
        eta, z = self.nz_split(nz)
        x = self.phi_inv(t, eta, z)
        _, zdot = jax.jvp(lambda w: self.phi_z(t, w),
                          (x,), (self.f(t, x),))
        return zdot

    def f_hat(self, t, nz):
        eta, z = self.nz_split(nz)
        x = self.phi_inv(t, eta, z)
        _, f = jax.jvp(lambda w: self.phi_eta(t, w), (x,), (self.f(t, x),))
        return f

    def g_hatu(self, t, nz, u):
        eta, z = self.nz_split(nz)
        x = self.phi_inv(t, eta, z)
        _, gu = jax.jvp(lambda w: self.phi_eta(t, w), (x,), (self.g(t, x) @ u,))
        return gu

    def g_hat(self, t, nz):
        eta, z = self.nz_split(nz)
        x = self.phi_inv(t, eta, z)
        ghat = jax.jacfwd(self.phi_eta, argnums=1)(t, x) @ self.g(t, x)
        return ghat

    def nz_dynamics(self, t, nz, u):
        return jnp.hstack((self.f_hat(t, nz) + self.g_hatu(t, nz, u), self.omega(t, nz)))


class HybridAbstractZeroDynamics:

    def f1(self, t, x):
        raise NotImplementedError

    def f2(self, t, x):
        raise NotImplementedError

    def g1(self, t, x):
        raise NotImplementedError

    def g2(self, t, x):
        raise NotImplementedError

    def R_12(self, t, x):
        raise NotImplementedError

    def R_21(self, t, x):
        raise NotImplementedError

    def S_12(self, t, x):
        raise NotImplementedError

    def S_21(self, t, x):
        raise NotImplementedError

    def phi(self, x):
        raise NotImplementedError


class Pendulum(nn.Module):
    state_dim: int = 2
    action_dim: int = 1
    mass: float = 1.0
    l: float = 1.0
    gravity: float = 9.8

    @nn.compact
    def __call__(self, t, x, u):
        theta, theta_dot = x[..., 0], x[..., 1]
        damp = (-theta_dot*0.1)/(self.mass*self.l**2)
        theta_ddot = damp + (self.gravity/self.l)*jnp.sin(theta) + u[..., 0]/(self.mass * self.l**2)
        xdot = jnp.stack([theta_dot, theta_ddot], axis=-1)
        return xdot

    def f(self, t, x):
        theta, theta_dot = x[..., 0], x[..., 1]
        f_1 = theta_dot
        f_2 = (self.gravity/self.l)*jnp.sin(theta)
        return jnp.stack([f_1, f_2], axis=-1)

    def g(self, t, x):
        return jnp.array([[0.0], [1.0/(self.mass * self.l**2)]])


class UnderactuatedLinearSystem(nn.Module, AbstractZeroDynamics):
    state_dim: int = 4
    action_dim: int = 1
    n_dim: int = 2
    z_dim: int = 2

    @nn.compact
    def __call__(self, t, x, u):
        return self.f(t, x) + self.g(t, x) @ u

    def f(self, t, x):
        return jnp.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [1, -2, -1, 2]]) @ x

    def g(self, t, x):
        return jnp.array([[0], [1], [0], [0]])

    def phi(self, t, x):
        return x[:2], x[2:]

    def phi_inv(self, t, eta, z):
        return jnp.hstack((eta, z))


class BackSteppableUnderactuatedLinearSystem(nn.Module, AbstractZeroDynamics):
    state_dim: int = 4
    action_dim: int = 1
    n_dim: int = 2
    z_dim: int = 2

    @nn.compact
    def __call__(self, t, x, u):
        return self.f(t, x) + self.g(t, x) @ u

    def f(self, t, x):
        return jnp.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [1, 0, -1, 2]]) @ x

    def g(self, t, x):
        return jnp.array([[0], [1], [0], [0]])

    def phi(self, t, x):
        return x[:2], x[2:]

    def phi_inv(self, t, eta, z):
        return jnp.hstack((eta, z))
    

class ComptonSystem(nn.Module, AbstractZeroDynamics):
    state_dim: int = 4
    action_dim: int = 1
    n_dim: int = 2
    z_dim: int = 2
    m: float = 1.0
    k: float = 1.0
    b: float = -1.0  # Negative damping
    # [x1 x2 x1dot x2dot]

    @nn.compact
    def __call__(self, t, x, u):
        return self.f(t, x) + self.g(t, x) @ u
    
    def f(self, t, x):
        return jnp.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-self.k / self.m, self.k/self.m, -self.b/self.m, self.b/self.m],
            [self.k / self.m, -self.k/self.m, self.b/self.m, -self.b/self.m]
        ]) @ x
    
    def g(self, t, x):
        return jnp.array([[0], [0], [1], [0]])

    def phi(self, t, x):
        nz = jnp.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ]) @ x
        return nz[:2], nz[2:]
    
    def phi_inv(self, t, eta, z):
        return jnp.linalg.inv(jnp.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])) @ jnp.hstack([eta, z])



class DoublePendulum(nn.Module):
    state_dim: int = 4
    action_dim: int = 2
    mass_1: float = 1.0
    mass_2: float = 1.0
    l_1: float = 1.0
    l_2: float = 1.0
    g: float = 9.81

    @nn.compact
    def __call__(self, t, x, u):
        # unpacking params and states
        m1, m2, l1, l2, g = self.mass_1, self.mass_2, self.l_1, self.l_2, self.g
        th1 = x[..., 0] + np.pi
        th2 = x[..., 1]
        dth1 = x[..., 2]
        dth2 = x[..., 3]
        u1 = u[..., 0]
        u2 = u[..., 1]

        # optimized forward dynamics
        t2 = jnp.cos(th2)
        t3 = jnp.sin(th1)
        t4 = jnp.sin(th2)
        t5 = th1 + th2
        t6 = 2 * dth1
        t7 = l1 ** 2
        t8 = t2 ** 2
        t9 = l1 * t2
        t10 = jnp.sin(t5)
        t11 = 1.0 / t7
        t12 = dth2 + t6
        t13 = m1 * t7
        t14 = m2 * t7
        t15 = m2 * t8
        t16 = l2 + t9
        t17 = l2 * m2 * t10
        t21 = dth2 * l1 * l2 * m2 * t4 * t12
        t18 = -t15
        t19 = g * t17
        t20 = m1 + m2 + t18
        t22 = 1.0 / t20
        ddth1 = (t21 + u1 - g * (t17 + l1 * t3 * (m1 + m2))) / (t13 + t14 - t8 * t14) + (t16 * (t19 - u2 - (dth1 * dth2 * l1 * l2 * m2 * t4) / 2.0 + (dth1 * l1 * l2 * m2 * t4 * t12) / 2.0)) / (l2 * t13 + l2 * t14 - l2 * t8 * t14)
        ddth2 = -(t11 * t16 * t22 * (t21 + u1 - g * (t17 + l1 * m1 * t3 + l1 * m2 * t3))) / l2 - (1.0 / l2 ** 2 * t11 * t22 * (t19 - u2 + dth1 ** 2 * l1 * l2 * m2 * t4) * (t13 + t14 + l2 ** 2 * m2 + l2 * m2 * t9 * 2.0)) / m2

        return jnp.stack((dth1, dth2, ddth1, ddth2), dim=-1)

    def f(self, t, x):
        theta_1, theta_2 = x[..., 0], x[..., 1]
        theta_1_dot, theta_2_dot = x[..., 2], x[..., 3]
        f_1 = theta_1_dot
        f_2 = theta_2_dot
        f_3 = (self.g/self.l_1)*jnp.sin(theta_1) + (theta_2_dot**2 * jnp.sin(theta_1 - theta_2) + self.l_2 * theta_2_dot**2 * jnp.sin(theta_1 - theta_2) - self.g * jnp.sin(theta_2))/(self.l_1 * (2 - self.mass_2 * jnp.cos(2 * theta_1 - 2 * theta_2)))
        f_4 = (2 * self.l_1 * theta_1_dot**2 * jnp.sin(theta_1 - theta_2) + self.g * jnp.sin(theta_1) + self.l_2 * theta_2_dot**2 * jnp.sin(theta_1 - theta_2) * jnp.cos(theta_1 - theta_2) + self.g * jnp.sin(theta_2) * jnp.cos(theta_1 - theta_2))/(self.l_2 * (2 - self.mass_2 * jnp.cos(2 * theta_1 - 2 * theta_2)))
        return jnp.stack([f_1, f_2, f_3, f_4], axis=-1)

    def g(self, t, x):
        return jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0/(self.mass_1 * self.l_1**2), 0.0], [0.0, 1.0/(self.mass_2 * self.l_2**2)]])

class Acrobot(DoublePendulum):
    action_dim: int = 1
    n_dim: int = 2
    z_dim: int = 2

    @nn.compact
    def __call__(self, t, x, u):
        u_full = jnp.stack([np.zeros_like(u[..., 0]), u[..., 0]], axis=-1)
        return super().__call__(t, x, u_full)

    def g(self, t, x):
        return jnp.array([[0.0], [0.0], [0.0], [1.0/(self.mass_2 * self.l_2**2)]])


class Cartpole(nn.Module, AbstractZeroDynamics):
    state_dim: int = 4
    d_n: int = 4
    d_m: int = 1
    action_dim: int = 1
    n_dim: int = 2
    z_dim: int = 2
    l: float = 1.0
    gr: float = 9.81
    mc: float = 1.0
    mp: float = 1.0

    @nn.compact
    def __call__(self, t, x, u):
        fx = self.f(t, x)
        gx = self.g(t, x)
        return fx + gx @ u

    def dynamics(self, x, u):
        fx = self.f(0, x)
        gx = self.g(0, x)
        return fx + gx @ u

    def f(self, t, x):
        x, th, dx, dth = x[0], x[1], x[2], x[3]
        return jnp.stack([
            dx,
            dth,
            (self.mp*jnp.sin(th)*(self.l*dth**2 - self.gr*jnp.cos(th)))/(-self.mp*jnp.cos(th)**2 + self.mc + self.mp),
            (jnp.sin(th)*(- self.l*self.mp*jnp.cos(th)*dth**2 + self.gr*self.mc + self.gr*self.mp))/(self.l*(- self.mp*jnp.cos(th)**2 + self.mc + self.mp))
        ])

    def g(self, t, x):
        x, th, dx, dth = x[0], x[1], x[2], x[3]
        return jnp.array([
            [0],
            [0],
            [1/(- self.mp*jnp.cos(th)**2 + self.mc + self.mp)],
            [-jnp.cos(th)/(self.l*(- self.mp*jnp.cos(th)**2 + self.mc + self.mp))]
        ])

    def phi(self, t, state):
        x, th, dx, dth = state[0], state[1], state[2], state[3]
        eta = jnp.stack([x, dx])
        z = jnp.stack([
            th,
            dth * self.mp * self.l ** 2 + dx * self.mp * jnp.cos(th) * self.l])
        return eta, z

    def phi_inv(self, t, eta, z):
        eta1_, eta2_ = eta[0], eta[1]
        z1_, z2_ = z[0], z[1]
        return jnp.stack([
           eta1_,
           z1_,
           eta2_,
           (z2_ - self.l*self.mp*eta2_*jnp.cos(z1_))/(self.l**2*self.mp)
           ])

    @staticmethod
    def x_star():
        return jnp.zeros((Cartpole.d_n,))

from utils import hardshrink

class UnstableCartpole(Cartpole):
    gamma: float = 1.0

    def f(self, t, x):
        x, th, dx, dth = x[0], x[1], x[2], x[3]
        damping = self.gamma*hardshrink(dx, 0.01)
        return jnp.stack([
            dx,
            dth,
            -(self.gamma*jnp.cos(th)*damping - self.mp*jnp.sin(th)*dth**2*self.l**2 + self.gr*self.mp*jnp.cos(th)*jnp.sin(th)*self.l)/(self.l*(- self.mp*jnp.cos(th)**2 + self.mc + self.mp)),
            (self.gamma*damping*(self.mp + self.mc) - jnp.cos(th)*jnp.sin(th)*dth**2*self.l**2*self.mp**2 + self.gr*jnp.sin(th)*self.l*self.mp**2 + self.gr*self.mc*jnp.sin(th)*self.l*self.mp)/(self.l**2*self.mp*(- self.mp*jnp.cos(th)**2 + self.mc + self.mp))
        ])

class UnstableBaseCartpole(Cartpole):
    gamma: float = 1.0

    def f(self, t, x):
        x, th, dx, dth = x[0], x[1], x[2], x[3]
        damping = self.gamma*hardshrink(dx, 0.01)
        return jnp.stack([
            dx,
            dth,
            (damping + self.mp*jnp.sin(th)*(self.l*dth**2 - self.gr*jnp.cos(th)))/(-self.mp*jnp.cos(th)**2 + self.mc + self.mp),
            (jnp.sin(th)*(- self.l*self.mp*jnp.cos(th)*dth**2 + self.gr*self.mc + self.gr*self.mp))/(self.l*(- self.mp*jnp.cos(th)**2 + self.mc + self.mp))
        ])

class LazyCartpole(Cartpole):
    gamma: float = 0.001

    def g(self, t, x):
        orig_g = super().g(t, x)
        return orig_g * self.gamma

class UnstableAcrobot(Acrobot):
    gamma: float = 1.0



class Hopper(nn.Module):
    state_dim: int = 17
    action_dim: int = 3
    n_dim: int = 6
    z_dim: int = 11
    r0: float = 0.3445
    gr: float = 9.81
    k: float = 11732
    m: float = 5.91
    c: float = 0
    I: float = 0.0975
    Iyaw: float = 0.0279
    la_qw: float = 0.8806
    la_qx: float = 0.3646
    qy: float = -0.2795
    qz: float = 0.1160

    # TODO: Add yaw, flywheel dynamics
    # State Def: [th, ph, ya, x, y, z, r, thd, phd, yad, xd, yd, zd, rd, w1d, w2d, w3d]

    @nn.compact
    def __call__(self, t, x, u, dom):
        return self.f(t, x, dom) + self.g(t, x, dom) @ u

    def f(self, t, x, dom):
        return dom * self.f_1(t, x) + (1 - dom) * self.f_0(t, x)

    def g(self, t, x, dom):
        return dom * self.g_1(t, x) + (1 - dom) * self.g_0(t, x)

    def f_0(self, t, x):
        th, ph, ya, x, y, z, r, thd, phd, yad, xd, yd, zd, rd, w1d, w2d, w3d = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16]
        return jnp.stack([
            thd, phd, yad, xd, yd, zd, rd,
            0,   0,   0,   0,  0, -self.gr, 0, 0, 0, 0
        ])

    def f_1(self, t, x):
        th, ph, ya, x, y, z, r, thd, phd, yad, xd, yd, zd, rd, w1d, w2d, w3d = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16]
        return jnp.stack([
            thd, phd, yad, xd, yd, zd, rd,
            1 / (self.m * r ** 2) * (-2 * self.m * r * rd * thd - self.m * r ** 2 * jnp.sin(th) * jnp.cos(th) * phd ** 2 + self.m * self.gr * r * jnp.sin(th) * jnp.cos(ph)),
            1 / (self.m * r ** 2 * jnp.cos(th) ** 2) * (-2 * self.m * r * rd * phd * jnp.cos(th) ** 2 + 2 * self.m * r ** 2 * phd * jnp.sin(th) * jnp.cos(th) + self.m * self.gr * r * jnp.cos(th) * jnp.sin(ph)),
            0, 0, 0, 0,
            1 / self.m * (self.m * r * thd ** 2 + self.m * r * jnp.cos(th) ** 2 * phd ** 2 - self.m * self.gr * jnp.cos(th) * jnp.cos(ph) + self.k * (self.r0 - r) - self.c * rd),
            0, 0, 0
        ])

    def g_0(self, t, x):
        return jnp.vstack([
            jnp.zeros((self.action_dim,)),jnp.zeros((self.action_dim,)),jnp.zeros((self.action_dim,)),jnp.zeros((self.action_dim,)),jnp.zeros((self.action_dim,)),jnp.zeros((self.action_dim,)),jnp.zeros((self.action_dim,)),
            jnp.array([1 / self.I,0,0]),jnp.array([0,1 / self.I,0]),jnp.array([0,0,1 / self.Iyaw]),
            jnp.zeros((self.action_dim,)),jnp.zeros((self.action_dim,)),jnp.zeros((self.action_dim,)),jnp.zeros((self.action_dim,)),
            -jnp.array([1 / self.I,0,0]),-jnp.array([0,1 / self.I,0]),-jnp.array([0,0,1 / self.Iyaw])
        ])

    def g_1(self, t, x):
        return jnp.zeros((self.state_dim, self.action_dim))

    def R_01(self, t, x):
        # TODO: Correct flight-> Ground reset map
        th, ph, ya, x, y, z, r, thd, phd, yad, xd, yd, zd, rd, w1d, w2d, w3d = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16]
        rtp_d = jnp.squeeze(jnp.linalg.inv(jnp.array([
            [jnp.sin(th), self.r0 * jnp.cos(th), 0],
            [jnp.cos(th) * jnp.sin(ph), -r * jnp.sin(th) * jnp.sin(ph), self.r0 * jnp.cos(ph) * jnp.cos(th)],
            [jnp.cos(th) * jnp.cos(ph), - self.r0 * jnp.sin(th) * jnp.cos(ph), - self.r0 * jnp.cos(th) * jnp.sin(ph)]
        ])) @ jnp.array([[xd], [yd], [zd]]))
        return jnp.array([
            th, ph, ya, x - self.r0 * jnp.sin(th), y - self.r0 * jnp.cos(th) * jnp.sin(ph), z - self.r0 * jnp.cos(th) * jnp.cos(ph), self.r0,
            rtp_d[1], rtp_d[2], 0, 0, 0, 0, rtp_d[0], w1d, w2d, w3d
        ])

    def R_10(self, t, x):
        # TODO: Correct ground-> flight reset map
        th, ph, ya, x, y, z, r, thd, phd, yad, xd, yd, zd, rd, w1d, w2d, w3d = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16]
        xyz_d = jnp.squeeze(jnp.array([
            [jnp.sin(th),  self.r0 * jnp.cos(th), 0],
            [jnp.cos(th) * jnp.sin(ph), -r * jnp.sin(th) * jnp.sin(ph), self.r0 * jnp.cos(ph) * jnp.cos(th)],
            [jnp.cos(th) * jnp.cos(ph), -self.r0 * jnp.sin(th) * jnp.cos(ph), -self.r0 * jnp.cos(th) * jnp.sin(ph)]
        ]) @ jnp.array([[rd], [thd], [phd]]))
        return jnp.array([
            th, ph, ya, x + self.r0 * jnp.sin(th), y + self.r0 * jnp.cos(th) * jnp.sin(ph), z + self.r0 * jnp.cos(th) * jnp.cos(ph), self.r0,
            thd, phd, 0, xyz_d[0], xyz_d[1], xyz_d[2], 0, w1d, w2d, w3d
        ])

    def backward_R_01(self, t, x):
        return self.R_10(t, x)

    def backward_R_10(self, t, x):
        return self.R_01(t, x)

    def ode_hybrid_event(self, state, **kwargs):
        return self.hybrid_event(state.y)

    def pmp_hybrid_event(self, state, **kwargs):
        return self.backward_hybrid_event(state.y.x)

    def hybrid_event(self, x):
        return self.S_01(x) | self.S_10(x)

    def backward_hybrid_event(self, x):
        return self.backward_S_01(x) | self.backward_S_10(x)

    def S_01(self, x):
        z = x[5]
        z_dot = x[12]
        th = x[0]
        th_dot = x[7]
        phi = x[1]
        phi_dot = x[8]
        # Foot height
        z_foot = z - self.r0 * jnp.cos(th) * jnp.cos(phi)
        z_dot_foot = z_dot + self.r0 * (th_dot * jnp.sin(th) * jnp.cos(phi) + phi_dot * jnp.cos(th) * jnp.sin(phi))
        return (z_foot <= 0) & (z_dot_foot < 0) & (z_dot != 0)

    def S_10(self, x):
        # When viewed forwards in time, this is the guard from ground to flight.
        # However, this function will be called when integrating backward in time, so x is a state on the flight and
        # needs to be transferred into the ground
        r = x[6]
        rdot = x[13]
        return (r >= self.r0) & (rdot > 0)

    def backward_S_01(self, x):
        # When viewed forwards in time, this is the guard from flight to ground.
        # However, this function will be called when integrating backward in time, so x is a state on the ground and
        # needs to be transferred into flight
        r = x[6]
        rdot = x[13]
        return (r >= self.r0) & (rdot < 0)


    def backward_S_10(self, x):
        # When viewed forwards in time, this is the guard from ground to flight.
        # However, this function will be called when integrating backward in time, so x is a state on the flight and
        # needs to be transferred into ground
        z = x[5]
        z_dot = x[12]
        th = x[0]
        th_dot = x[7]
        phi = x[1]
        phi_dot = x[8]
        # Foot height
        z_foot = z - self.r0 * jnp.cos(th) * jnp.cos(phi)
        z_dot_foot = z_dot + self.r0 * (th_dot * jnp.sin(th) * jnp.cos(phi) + phi_dot * jnp.cos(th) * jnp.sin(phi))
        return (z_foot <= 0) & (z_dot_foot > 0) & (z_dot != 0)


class Acrobot(nn.Module):
    state_dim: int = 4
    action_dim: int = 1
    l: float = 1.0
    gr: float = 9.81
    m1: float = 1.0
    m2: float = 1.0
    lc: float = l / 2.0
    I1: float = 1.0
    I2: float = 1.0

    @nn.compact
    def __call__(self, t, x, u):
        fx = self.f(t, x)
        gx = self.g(t, x)
        return fx + gx @ u

    def f(self, t, x):
        t1, t2, td1, td2 = x[0], x[1], x[2], x[3]
        return jnp.stack([
            td1,
            td2,
            (self.I2*(self.l*self.lc*self.m2*jnp.sin(t2)*td2**2 + 2*self.l*self.lc*self.m2*td1*jnp.sin(t2)*td2 + self.gr*self.m2*(self.lc*jnp.sin(t1 + t2) + self.l*jnp.sin(t1)) + self.gr*self.lc*self.m1*jnp.sin(t1)))/(- self.l**2*self.lc**2*self.m2**2*jnp.cos(t2)**2 + self.I2*self.l**2*self.m2 + self.I1*self.I2) - ((- self.l*self.lc*self.m2*jnp.sin(t2)*td1**2 + self.gr*self.lc*self.m2*jnp.sin(t1 + t2))*(self.I2 + self.l*self.lc*self.m2*jnp.cos(t2)))/(- self.l**2*self.lc**2*self.m2**2*jnp.cos(t2)**2 + self.I2*self.l**2*self.m2 + self.I1*self.I2),
            ((- self.l*self.lc*self.m2*jnp.sin(t2)*td1**2 + self.gr*self.lc*self.m2*jnp.sin(t1 + t2))*(self.m2*self.l**2 + 2*self.lc*self.m2*jnp.cos(t2)*self.l + self.I1 + self.I2))/(- self.l**2*self.lc**2*self.m2**2*jnp.cos(t2)**2 + self.I2*self.l**2*self.m2 + self.I1*self.I2) - ((self.I2 + self.l*self.lc*self.m2*jnp.cos(t2))*(self.l*self.lc*self.m2*jnp.sin(t2)*td2**2 + 2*self.l*self.lc*self.m2*td1*jnp.sin(t2)*td2 + self.gr*self.m2*(self.lc*jnp.sin(t1 + t2) + self.l*jnp.sin(t1)) + self.gr*self.lc*self.m1*jnp.sin(t1)))/(- self.l**2*self.lc**2*self.m2**2*jnp.cos(t2)**2 + self.I2*self.l**2*self.m2 + self.I1*self.I2)
        ])

    def g(self, t, x):
        t1, t2, td1, td2 = x[0], x[1], x[2], x[3]
        return jnp.vstack([
            0,
            0,
            -(self.I2 + self.l*self.lc*self.m2*jnp.cos(t2))/(- self.l**2*self.lc**2*self.m2**2*jnp.cos(t2)**2 + self.I2*self.l**2*self.m2 + self.I1*self.I2),
            (self.m2*self.l**2 + 2*self.lc*self.m2*jnp.cos(t2)*self.l + self.I1 + self.I2)/(- self.l**2*self.lc**2*self.m2**2*jnp.cos(t2)**2 + self.I2*self.l**2*self.m2 + self.I1*self.I2)
        ])

    def phi(self, t, x):
        t1, t2, td1, td2 = x[0], x[1], x[2], x[3]
        return jnp.stack([
            t2,
            td2,
            t1,
            td1*(self.m2*self.l**2 + 2*self.lc*self.m2*jnp.cos(t2)*self.l + self.I1 + self.I2) + td2*(self.I2 + self.l*self.lc*self.m2*jnp.cos(t2))])
    def phi_inv(self, t, eta, z):
       eta1_, eta2_ = eta[0], eta[1]
       z1_, z2_ = z[0], z[1]
       return jnp.stack([
           z1_,
           eta1_,
           (z2_ - eta2_*(self.I2 + self.l*self.lc*self.m2*jnp.cos(eta1_)))/(self.m2*self.l**2 + 2*self.lc*self.m2*jnp.cos(eta1_)*self.l + self.I1 + self.I2),
           eta2_,
           ])

import diffrax as de
class DiscretizedDynamics:
    def __init__(self, dynamics, dt, int):
        self.dynamics = dynamics
        self.dt = dt
        self.int = int

    def __call__(self, x, u):
        sol = self.int(de.ODETerm(lambda t, _x, args: self.dynamics(_x, u)),
                  dt0=self.dt,
                  t0=0.0,
                  t1=self.dt,
                  y0=x)

        return sol.ys[-1]


class HopperH2H(nn.Module):
    integrator: functools.partialmethod
    d_n: int = 10
    d_m: int = 2
    t_hop: float = 0.35
    t_contact: float = 0.1
    k: float = 11732
    m: float = 5.9100
    g: float = 9.81
    I_xx: float = 0.0975
    I_yy: float = 0.0975
    I_zz: float = 0.0279
    r0: float = 0.3445
    v0: float = g * t_hop / 2
    E: float = 0.5*m * v0**2

    def __call__(self, t, x, u):
        del t
        return self.f(x, u)

    def x_star(self):
        return jnp.array([0, 0, 0, 0, 0, 0, self.r0, 0, 0, self.dot_h0(0, 0, self.r0)])

    def f(self, x, u):
        return self.flight(self.ground(x), u)

    def f_w_post(self, x, u):
        x_post = self.ground(x)
        return self.flight(x_post, u), x_post

    def flight(self, x, u):
        h0 = x[6]                   # Initial height
        hf = self.h0(u[0], u[1])    # Final height
        t_apex = jnp.clip(x[-1] / self.g, 1e-4, jnp.inf)     # Time to apex
        z_apex = h0 + t_apex * x[-1] - 0.5 * self.g * t_apex ** 2   # height of apex
        t_down = jnp.sqrt(2*(jnp.clip(z_apex - hf, 1e-4, jnp.inf)) / self.g)
        # time from apex to impact
        t = t_apex + t_down                                         # flight time
        return jnp.hstack([u, 0, 0, x[4:6] + x[7:9] * t, hf, x[7:9], -self.g * t_down])

    def predict_impact_z(self, x):
        h0 = x[6]  # Initial height
        hf = self.h0(0, 0)  # Final height approximate via vertical
        t_apex = jnp.clip(x[-1] / self.g, 1e-4, jnp.inf)  # Time to apex
        z_apex = h0 + t_apex * x[-1] - 0.5 * self.g * t_apex ** 2  # height of apex
        t_down = jnp.sqrt(2 * (jnp.clip(z_apex - hf, 1e-4, jnp.inf)) / self.g)  # time from apex to impact
        t = t_apex + t_down
        return jnp.hstack([x[4:6] + x[7:9] * t, x[7:9]])

    def ground(self, x):
        # Record foot location
        foot = jnp.array([x[4] - self.r0*jnp.sin(x[0]), x[5] - self.r0*jnp.cos(x[0])*jnp.sin(x[1])])
        # Transfer xyz velocity to angular, radial velocity (impact map preserving linear momentum of mass)
        # To Will: Try this instead of computing the inverse directly if
        # you know it is invertible
        # rtp_d = jnp.linalg.inv(self.d_rtp2d_xyz(x)) @ x[7:]
        rtp_d = jnp.linalg.solve(self.d_rtp2d_xyz(x), x[7:])
        # Post impact map state
        xg_plus = jnp.array([x[0], x[1], self.r0, rtp_d[1], rtp_d[2], rtp_d[0]])
        # Solve ODE
        sol = self.integrator(de.ODETerm(lambda t, xg, args: self.dot_ground(xg)), y0=xg_plus, t0=0, t1=self.t_contact)
        xg_m = sol.ys.reshape((-1,))
        # Transfer velocity to linear, preserving rotation (impact map preserving linear momentum of mass)
        xyz_d = self.d_rtp2d_xyz(xg_m) @ jnp.array([xg_m[5], xg_m[3], xg_m[4]])
        return jnp.hstack([xg_m[:2], xg_m[3:5], self.r0*jnp.sin(xg_m[0]) + foot[0], self.r0*jnp.cos(xg_m[0])*jnp.sin(xg_m[1]) + foot[1], self.r0*jnp.cos(xg_m[0])*jnp.cos(xg_m[1]), xyz_d])

    def dot_ground(self, x_g):
        th, ph, r, th_d, ph_d, r_d = x_g

        r_dd = 1 / self.m * (self.m * r * th_d **2 + self.m * r * jnp.cos(th)**2 * ph_d**2 - self.m * self.g * jnp.cos(th) * jnp.cos(ph) + self.k * (self.r0 - r))
        th_dd = 1 / (self.m * r**2) * (-2 * self.m * r * r_d * th_d - self.m * r**2 * jnp.sin(th) * jnp.cos(th) * ph_d**2 + self.m * self.g * r * jnp.sin(th) * jnp.cos(ph))
        ph_dd = 1 / (self.m * r**2 * jnp.cos(th)**2) * (-2 * self.m * r * r_d * ph_d * jnp.cos(th)**2 + 2 * self.m * r**2 * ph_d * jnp.sin(th) * jnp.cos(th) + self.m * self.g * r * jnp.cos(th) * jnp.sin(ph))
        return jnp.array([th_d, ph_d, r_d, th_dd, ph_dd, r_dd])

    @staticmethod
    def h0(theta, phi):
        return HopperH2H.r0 * jnp.cos(theta) * jnp.cos(phi)

    @staticmethod
    def dot_h0(dot_x, dot_y, h):
        return -jnp.sqrt(
            (HopperH2H.E - 0.5*HopperH2H.m*HopperH2H.g*(HopperH2H.r0 - h) - 0.5*HopperH2H.m*(dot_x**2 + dot_y**2)) /
            (0.5*HopperH2H.m)
        )

    @staticmethod
    def x_from_nz(eta, z):
        return jnp.hstack(
            [eta, 0, 0, z[:2], HopperH2H.h0(eta[0], eta[1]), z[2:], HopperH2H.dot_h0(z[2], z[3], HopperH2H.h0(eta[0], eta[1]))])

    @staticmethod
    def nz_from_x(x):
        return x[:2], x[jnp.array([4, 5, 7, 8])]

    @staticmethod
    def d_rtp2d_xyz(x):
        th, ph = x[:2]
        return jnp.array([
            [jnp.sin(th), HopperH2H.r0*jnp.cos(th), 0],
            [jnp.cos(th)*jnp.sin(ph), -HopperH2H.r0*jnp.sin(th)*jnp.sin(ph), HopperH2H.r0*jnp.cos(th)*jnp.cos(ph)],
            [jnp.cos(th)*jnp.cos(ph), -HopperH2H.r0*jnp.sin(th)*jnp.cos(ph), -HopperH2H.r0*jnp.cos(th)*jnp.sin(ph)]
        ])

    def warm_start(self, x0, N):
        thd_bar = jnp.zeros((N, 2))
        weight = jnp.linspace(0, 1, N+1)
        x_bar = jnp.outer(1 - weight, x0) + jnp.outer(weight, self.x_star)
        return x_bar, thd_bar

    @staticmethod
    def event_gf(x):
        # This is the guard event from ground to flight.
        r = x[2]
        rdot = x[-1]
        return (r >= HopperH2H.r0) & (rdot > 0)

    @staticmethod
    def ode_gf_event(state, **kwargs):
        return HopperH2H.event_gf(state.y)
