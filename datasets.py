import jax
import jax.random
import jax.numpy as jnp
from torch.utils.data import Dataset
from jax.random import uniform
from robots import HopperH2H
from control import raibert_policy


class UniformRandomDataset(Dataset):
    def __init__(self, rng, lower_bound, upper_bound, length):
        """
        Initializes the dataset with bounds and dimensions of samples.
        :param lower_bound: The lower bound of the uniform distribution.
        :param upper_bound: The upper bound of the uniform distribution.
        :param length: The total number of samples in the dataset.
        """
        self.dist = uniform(key=rng, shape=lower_bound.shape,
                            minval=lower_bound,
                            maxval=upper_bound)
        self.rng = rng
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.length = length

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return self.length

    def __getitem__(self, idx):
        """
        Generates and returns a random sample.
        :param idx: Index of the sample (not used in this dataset).
        :return: A random sample tensor.
        """
        self.rng, split = jax.random.split(self.rng)
        sample = uniform(key=split,
                         shape=(len(idx),)+self.lower_bound.shape,
                         minval=self.lower_bound[None],
                         maxval=self.upper_bound[None])
        return sample


class SupervisedRaibertSampler(Dataset):
    def __init__(self, rng, hopper, length, state_bound, K, pos_bound, action_bound):
        self.rng = rng
        self.hopper = hopper
        self.length = length
        self.state_bound = state_bound
        self.raibert_policy = raibert_policy(K, pos_bound, action_bound)

    def __len__(self):
        return self.length

    def _construct_supervised_point(self, nz0):
        eta_d = self.raibert_policy(nz0)
        # Sim forwards one hop to get the matching zero dynamics state
        x0 = HopperH2H.x_from_nz(nz0[:2], nz0[2:])
        xf = self.hopper.f(x0, eta_d)
        # Return z, eta pair
        return jnp.hstack([xf[4:6], xf[7:9]]), eta_d

    def __getitem__(self, idx):
        self.rng, split = jax.random.split(self.rng)
        # Randomly sample initial conditions
        # x0 = uniform(key=split,
        #              shape=(len(idx),) + self.state_bound.lower.shape,
        #              minval=self.state_bound.lower[None],
        #              maxval=self.state_bound.upper[None])
        n0 = uniform(key=split,
                     shape=(len(idx),) +(2,),
                     minval=self.state_bound.lower[None, :2],
                     maxval=self.state_bound.upper[None, :2])
        self.rng, split = jax.random.split(self.rng)
        z_pos0 = uniform(key=split,
                     shape=(len(idx),) +(2,),
                     minval=self.state_bound.lower[None, 2:4],
                     maxval=self.state_bound.upper[None, 2:4])
        self.rng, split = jax.random.split(self.rng)
        z_vel0 = uniform(key=split,
                     shape=(len(idx),) +(2,),
                     minval=jnp.array([-1, -1])[None, :],
                     maxval=jnp.array([1, 1])[None, :])

        z_vel0_ = jnp.sign(z_vel0) * jnp.power(jnp.abs(z_vel0),1./3.)

        z_vel0_ *= self.state_bound.lower[4:]

        x0 = jnp.hstack([n0, z_pos0, z_vel0_])

        # Construct z, eta pairs
        return jax.vmap(self._construct_supervised_point, in_axes=0)(x0)
