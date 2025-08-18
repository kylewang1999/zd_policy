import sys
import os.path as osp
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

from jax import numpy as jnp
from torch.utils.data import DataLoader, BatchSampler, RandomSampler

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import os

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from train import State, CheckPointManager, wandb_model_load
from utils import unnormalize_dict

CONFIG_DIR = str(Path(__file__).parent.parent / "configs")


def load_config_with_overrides(config_name: str):
    """
    Load a config that specifies a base_config and overrides.
    
    Args:
        config_name: Name of the config file (without .yaml extension)
    
    Returns:
        OmegaConf configuration with base config and overrides applied
    """
    if not GlobalHydra().is_initialized():
        initialize_config_dir(config_dir=CONFIG_DIR)
    
    override_cfg = compose(config_name=config_name)
    
    base_config_name = override_cfg.base_config
    if base_config_name.endswith('.yaml'):
        base_config_name = base_config_name[:-5]  # Remove .yaml extension
    
    overrides = override_cfg.get('overrides', {})
    base_cfg = compose(config_name=base_config_name)
    base_cfg.update(overrides)
    
    return base_cfg


def main(config_name: str = "my_zdp"):
    
    cfg = load_config_with_overrides(config_name)
    logging.getLogger('absl').setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)

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

    # plt.figure()
    # ax = plt.subplot(121)
    # im = plot_2D_irregular_heatmap(dataset.x_values[:, 0], dataset.x_values[:, 2], dataset.y_values[:, 0],ax=ax)
    # plt.colorbar(im, ax=ax)
    # plt.xlabel('x')
    # plt.ylabel('dotx')
    # plt.title('theta1_d')
    # ax = plt.subplot(122)
    # im = plot_2D_irregular_heatmap(dataset.x_values[:, 1], dataset.x_values[:, 3], dataset.y_values[:, 1], ax=ax)
    # plt.colorbar(im, ax=ax)
    # plt.xlabel('x')
    # plt.ylabel('dotx')
    # plt.title('theta1_d')
    # plt.show()

    if "warmstart" in cfg and cfg.warmstart is not None:
        api = wandb.Api()
        model_name = f'{cfg.warmstart}_model:best'
        ws_cfg, ws_state = wandb_model_load(api, model_name)
        cfg.loss.integrator.policy = ws_cfg.loss.value_policy
        vpi, loss_fn = instantiate(cfg.loss)
        # ws_cfg.value_policy
        dummy_state = jnp.array([0, 0, 0, 0, 0, 0, 0.3445, 0, 0])  # Hack for training the hopper
        dummy_time = jnp.array(0.0)

        rng, split = jax.random.split(rng)
        vpi_params = vpi.init(split, dummy_time, dummy_state)
        vpi_params["policy"] = ws_state.params
    else:
        vpi, loss_fn = instantiate(cfg.loss)
        # make dummy inputs to create state

        # int_params  = integrator.init(split, dummy_time, dummy_state)

        dummy_state = jnp.zeros((4,))
        # dummy_state = jnp.zeros((dataset.x_values.shape[1],))  # Hack for training the hopper
        # dummy_state = jnp.array([0, 0, 0, 0, 0, 0, 0.3445, 0, 0])  # Hack for training the hopper
        dummy_time = jnp.array(0.0)

        rng, split = jax.random.split(rng)
        vpi_params = vpi.init(split, dummy_time, dummy_state)
        # params = (int_params, vpi_params)
    params = vpi_params
    # detect dtype and cast params
    from jax import random

    x = random.uniform(random.PRNGKey(0), (1000,), dtype=jnp.float64)
    dtype = x.dtype
    params = jax.tree.map(lambda x: jnp.array(x, dtype=dtype), params)

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


    @partial(jax.jit, backend="gpu")
    def train_step(state, batch):
        (loss, aux), grad = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(state.params, batch)
        grad_norm_squared = jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(y ** 2), grad, 0.0)
        updates, new_opt_state = optimizer.update(grad, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        # Why was this hack required? Lets remove it for now
        # if "iter" in state.params.keys():
        #     new_params["iter"] = state.params["iter"] + 1
        new_state = state.replace(params=new_params, opt_state=new_opt_state, step=state.step + 1)
        return new_state, loss, (grad_norm_squared, aux)

    # train_step = checkify.checkify(train_step, errors=checkify.user_checks)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = pd.json_normalize(cfg_dict, sep="/").to_dict(orient="records")[0]

    wandb.init(
        project="zero_dynamics",
        name=f"{cfg.experiment_name}",
        config=cfg_dict
    )
    ckpt_manager = CheckPointManager(metric_name="loss")
    aux_logger = instantiate(cfg.aux_logger)
    for i in range(cfg.num_epochs):
        loss = 0.0
        epoch_loss = 0.0
        pbar = tqdm(loader)
        for batch in pbar:
            if isinstance(batch, list):
                batch = tuple(batch)
            state, loss, aux_grad = train_step(state, batch)
            grad_norm, aux = aux_grad

            for k, v in aux_logger.items():
                if k != "all":
                    v(k, getattr(aux, k), step=state.step)
                else:
                    v("all", aux, step=state.step)
            wandb.log(
                {
                    "loss_step": loss.item(),
                    "lr_step": state.get_lr(),
                    "grad_norm": grad_norm
                },
                step=state.step,
            )
            pbar.set_postfix({"loss": loss, "lr": state.get_lr()})
            epoch_loss += loss
            if state.step % 100 == 0:
                ckpt_manager.save(state, loss.item(), epoch=i, step=state.step)
        wandb.log(
            {"loss_epoch": epoch_loss.item() / len(loader), "lr_epoch": state.get_lr()},
            step=state.step,
        )

if __name__ == "__main__":
    main()


# # Get absolute path to configs directory
# config_dir = os.path.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "configs")

# # Initialize Hydra with the config directory
# with initialize_config_dir(config_dir=config_dir):
#     # Compose the configuration
#     cfg = compose(config_name="zdp_cartpole_lqr")
    
#     # Call the function directly with the config
#     # Note: We need to call the actual function, not the decorated version
#     from train import main
#     # Get the original function (before decoration)
#     original_main = main.__wrapped__ if hasattr(main, '__wrapped__') else main
#     original_main(cfg)


