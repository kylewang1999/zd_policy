import logging
from functools import partial
from pathlib import Path
from typing import Any

import flax
import hydra
import jax
import optax
import orbax.checkpoint
import pandas as pd
import wandb
from flax.training import orbax_utils
from hydra.utils import instantiate
from omegaconf import OmegaConf
from orbax.checkpoint import PyTreeCheckpointer
from tqdm import tqdm
from utils import unnormalize_dict
from jax import numpy as jnp
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from simulator import DiscreteInvarianceIntegratorOutput
import time


@flax.struct.dataclass
class State:
    params: Any
    opt_state: Any
    rng: Any
    step: int

    def get_lr(self):
        return self.opt_state[1].hyperparams["learning_rate"].item()


def wandb_model_load(api, artifact_name):
    artifact = api.artifact(artifact_name)
    run = artifact.logged_by()
    config = run.config
    config = unnormalize_dict(config)
    config = OmegaConf.create(config)

    dir = artifact.download(root=Path("/tmp/wandb_downloads"))
    state = State(**PyTreeCheckpointer().restore(dir))

    return config, state


class CheckPointManager:
    def __init__(self, metric_name="loss"):
        self.metric_name = metric_name
        self.best_loss = float("inf")
        self.ckpt_path = Path(f"/tmp/orbax_checkpoints/{wandb.run.id}")
        self.orbax_saver = orbax.checkpoint.PyTreeCheckpointer()

    def save(self, state, metric, epoch, step):
        self._orbax_save(state)
        artifact = wandb.Artifact(
            type="model",
            name=f"{wandb.run.id}_model",
            metadata={self.metric_name: metric, "epoch": epoch, "step": step},
        )

        artifact.add_dir(str(self.ckpt_path))

        aliases = ["latest"]

        if self.best_loss > metric:
            self.best_loss = metric
            aliases.append("best")

        wandb.run.log_artifact(artifact, aliases=aliases)

    def _orbax_save(self, state):
        save_args = orbax_utils.save_args_from_target(state)
        self.orbax_saver.save(str(self.ckpt_path), state, save_args=save_args, force=True)


@hydra.main(
    config_path=str(Path(__file__).parent / "configs"),
    config_name="default",
    version_base="1.2",
)
def main(cfg):
    logger = logging.getLogger(__name__)

    # setup rng
    rng = jax.random.PRNGKey(cfg.seed)
    rng, split = jax.random.split(rng)
    dataset = instantiate(cfg.dataset)(rng=split)
    def collate_fn(batch):
        return batch[0]
    loader = DataLoader(
        dataset=dataset,
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=BatchSampler(
            sampler=RandomSampler(dataset), batch_size=cfg.batch_size, drop_last=False
        ),
    )
    # rng, split = jax.random.split(rng)
    pretrain_name = cfg.pretrained_net
    model_name = f'{pretrain_name}_model:best'
    api = wandb.Api()
    cfg_pre, state_pre = wandb_model_load(api, model_name)

    tic1 = time.time()
    print("Instantiate pretrained network")
    integrator_pre = instantiate(cfg_pre.loss.integrator)
    tic2 = time.time()
    print("\t", tic2 - tic1)
    psi_policy_network = integrator_pre.mlp
    print("Instantiate psi policy")
    psi_policy = instantiate(cfg.psi_policy)(mlp=psi_policy_network)
    tic3 = time.time()
    print("\t", tic3 - tic2)
    print("Instantiate integrator")
    integrator = instantiate(cfg.loss.integrator)(psi_policy=psi_policy)
    tic4 = time.time()
    print("\t", tic4 - tic3)
    print("Instantiate loss")
    vpi, loss_fn = instantiate(cfg.loss)(integrator=integrator)
    tic5 = time.time()
    print("\t", tic5 - tic4)
    print("Initialize Loss")
    # make dummy inputs to create state
    dummy_state = jnp.array([0, 0, 0, 0, 0, 0, 0.3445, 0, 0, -1.7])  # Hack for training the hopper
    dummy_time = jnp.array(0.0)

    rng, split = jax.random.split(rng)
    params = vpi.init(split, dummy_time, dummy_state)

    params['psi_policy']['params']['mlp'] = state_pre.params['mlp']['params']
    print("\t", time.time() - tic5)
    print("Initialization Time: ", time.time() - tic1)

    # detect dtype and cast params
    from jax import random

    x = random.uniform(random.PRNGKey(0), (1000,), dtype=jnp.float64)
    dtype = x.dtype
    params = jax.tree_map(lambda x: jnp.array(x, dtype=dtype), params)

    lr_scheduler = instantiate(cfg.lr_scheduler)
    grad_clipper = instantiate(cfg.grad_clipper)

    # parameter injection does not play well with the hydra instantiation
    # so we do it manually
    optimizer_config = OmegaConf.to_container(cfg.optimizer, resolve=True)
    optimizer_name = optimizer_config.pop("_target_")
    optimizer_fn = hydra.utils.get_method(optimizer_name)
    optimizer_fn = optax.inject_hyperparams(optimizer_fn)

    optimizer = optimizer_fn(learning_rate=lr_scheduler, **optimizer_config)
    optimizer = optax.chain(grad_clipper, optimizer)
    opt_state = optimizer.init(params)
    state = State(step=0, opt_state=opt_state, params=params, rng=rng)

    # Specify number of threads to run
    num_threads = cfg.num_threads
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_threads}"
    # assert cfg.batch_size % num_threads == 0

    z_data = jax.random.uniform(jax.random.PRNGKey(0), shape=(10000, 4), minval=-1, maxval=1)
    z_data = z_data * jnp.array([[2, 2, 0.8, 0.8]])

    @partial(jax.jit, backend="cpu")
    def train_step(state, batch):

        # PMAP
        loss_map = lambda z: jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(state.params, z)
        # np.savetxt("batch.csv", batch, delimiter=',')
        batch_size = jnp.size(batch, 0)
        batch = batch.reshape((num_threads, batch_size // num_threads, -1))
        (loss, aux), grad = jax.pmap(loss_map,in_axes=(0,))(batch)
        grad = jax.tree_map(lambda x_: x_.mean(axis=0), grad)
        loss = jnp.mean(loss)

        # VMAP
        # (loss, aux), grad = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(state.params, batch)

        updates, new_opt_state = optimizer.update(grad, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        eta_s = psi_policy.apply(new_params['psi_policy'], z_data, method='psi')
        aux = DiscreteInvarianceIntegratorOutput(
            z=z_data, u=aux.u.reshape((batch_size, -1)), eta=eta_s, z_p=aux.z_p.reshape((batch_size, -1)), eta_p=aux.eta_p.reshape((batch_size, -1)), psi_zp=aux.psi_zp.reshape((batch_size, -1))
        )

        new_state = state.replace(params=new_params, opt_state=new_opt_state, step=state.step + 1)
        return new_state, loss, aux

    # train_step = checkify.checkify(train_step, errors=checkify.user_checks)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = pd.json_normalize(cfg_dict, sep="/").to_dict(orient="records")[0]

    wandb.init(
        project="train_zdp",
        entity="ZeroDynamicsPolicies",
        name=f"{cfg.experiment_name}",
        config=cfg_dict
    )
    ckpt_manager = CheckPointManager(metric_name="loss")
    for i in range(cfg.num_epochs):
        loss = 0.0
        epoch_loss = 0.0
        pbar = tqdm(loader)
        for batch in pbar:
            if isinstance(batch, list):
                batch = tuple(batch)
            state, loss, aux = train_step(state, batch)
            aux_logger = instantiate(cfg.aux_logger)
            for k, v in aux_logger.items():
                if k != "all":
                    v(k, getattr(aux, k), step=state.step)
                else:
                    v("all", aux, step=state.step)
            wandb.log(
                {
                    "loss_step": loss.item(),
                    "lr_step": state.get_lr()
                },
                step=state.step,
            )
            pbar.set_postfix({"loss": loss, "lr": state.get_lr()})
            epoch_loss += loss
            if state.step % 50 == 0:
                ckpt_manager.save(state, loss.item(), epoch=i, step=state.step)
        wandb.log(
            {"loss_epoch": epoch_loss.item() / len(loader), "lr_epoch": state.get_lr()},
            step=state.step,
        )


if __name__ == "__main__":
    import os

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={min(os.cpu_count(), 128)}"
    from jax import config

    config.update("jax_enable_x64", True)
    # config.update("jax_debug_nans", True)
    from jax.experimental.compilation_cache import compilation_cache as cc

    cc.initialize_cache("/tmp/jax_cache")

    main()
