import torch.nn as nn

# import cv2
from torchvision.utils import save_image

import logging
from pathlib import Path
import sys
from typing import List, Union
import os
import wandb
from time import time
from tqdm import tqdm

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import torch
from glob import glob
from copy import deepcopy
from collections import OrderedDict

from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer

import itertools

from policy_models.utils.utils import (
    get_git_commit_hash,
    get_last_checkpoint,
    initialize_pretrained_weights,
    print_system_env_info,
)

# from torch.nn.parallel import DistributedDataParallel as DDP

#################################################################################
#                             Training Helper Functions                         #
#################################################################################


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log.txt"),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################


# @hydra.main(config_path="./policy_conf", config_name="VPP_calvin_train_jepa")
def train(cfg: DictConfig) -> None:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)  # new
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])  # new
    # accelerator = Accelerator()
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    device = accelerator.device
    # new added
    torch.set_float32_matmul_precision("medium")

    if accelerator.is_main_process:
        os.makedirs(
            cfg.log_dir, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        from datetime import datetime

        uuid = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # experiment_dir = f"{cfg.log_dir}/{uuid}"  # Create an experiment folder
        # experiment_dir = f"./checkpointsfused_calvin_ac_model"
        experiment_dir = f"./checkpointssvd_jepa_con_calvin_ac_model/2025-9-24"
        checkpoint_dir = (
            f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        )
        eval_dir = f"{experiment_dir}/eval"
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")

    datamodule = hydra.utils.instantiate(cfg.datamodule)  # cfg.datamodule=calvin
    print("DEBUG: datamodule instantiated")
    datamodule.setup()
    print("DEBUG: datamodule setup done")
    if accelerator.is_main_process:
        logger.info(
            f"Global batch size {cfg.batch_size:,} num_processes ({accelerator.num_processes})"
        )
    chk = get_last_checkpoint(Path.cwd())
    train_loader = datamodule.train_dataloader()["lang"]
    test_loader = datamodule.val_dataloader()["lang"]
    # chk = get_last_checkpoint(Path('/home/temp_store/code/calvin_d/logs/runs/2023-09-10/17-52-50/saved_models/epoch=09_eval_lh/avg_seq_len=2.62.ckpt'))
    # Load Model
    model = hydra.utils.instantiate(cfg.model)
    if "pretrain_chk" in cfg:
        initialize_pretrained_weights(model, cfg)

    if cfg.use_ckpt_path:
        state_dict = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
        # print('state_dict_key:', state_dict['model'].keys())
        print("load_from_ckpt:", cfg.ckpt_path)
        # c = []
        # hydra.initialize(config_path="../../conf")
        # hydra.main(config_name="config_abc.yaml")(lambda x: c.append(x))()
        model = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(state_dict["model"], strict=False)

    model = model.to(device)
    model.process_device()

    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = model.configure_optimizers()["optimizer"]
    Ir_scheduler = model.configure_optimizers()["lr_scheduler"]["scheduler"]

    model.on_train_start()
    if accelerator.is_main_process:
        logger.info(f"model parameter init")
    ema = deepcopy(model).to(
        device
    )  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    ema.eval()
    model.train()
    model, opt, loader = accelerator.prepare(model, opt, train_loader)
    print("DEBUG: model, opt, loader prepared")
    # model = DDP(model, find_unused_parameters=True)

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    eval_batch = None
    best_eval_loss = 1e8
    if accelerator.is_main_process:
        logger.info(f"Training for {cfg.max_epochs} epochs...")

    max_train_steps = cfg.max_epochs * len(loader)
    progress_bar = tqdm(
        range(train_steps, max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(cfg.max_epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        running_loss = 0

        for idx, data_batch in enumerate(loader):
            if accelerator.is_main_process:
                logger.info(f"DEBUG speed: Beginning process epoch {epoch} step {idx}")
            with accelerator.autocast():
                if accelerator.is_main_process:
                    logger.info(f"DEBUG speed: Beginning model(data_batch)")
                loss = model(data_batch)
                if accelerator.is_main_process:
                    logger.info(f"DEBUG speed: Finished model(data_batch)")
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            Ir_scheduler.step()
            update_ema(ema, model)
            running_loss += loss
            log_steps += 1
            train_steps += 1
            progress_bar.update(1)
            if train_steps % cfg.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item()

                if accelerator.is_main_process:
                    logger.info(
                        f"(step={train_steps:07d}) Train total Loss : {avg_loss:.6f}, Train Steps/Sec: {steps_per_sec:.2f}"
                    )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()
            if train_steps % cfg.save_steps == 0 and accelerator.is_main_process:
                checkpoint = {
                    "model": (
                        model.module.state_dict()
                        if accelerator.num_processes > 1
                        else model.state_dict()
                    ),
                    "args": cfg,
                }
                os.makedirs(checkpoint_dir, exist_ok=True)
                last_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                print("DEBUG save model to:", last_path)
                torch.save(checkpoint, last_path)


def setup_logger(cfg: DictConfig, model: LightningModule):
    """
    Set up the logger (tensorboard or wandb) from hydra config.

    Args:
        cfg: Hydra config
        model: LightningModule

    Returns:
        logger
    """
    pathlib_cwd = Path.cwd()
    if "group" in cfg.logger:
        cfg.logger.group = pathlib_cwd.parent.name
        cfg.logger.name = pathlib_cwd.parent.name + "/" + pathlib_cwd.name
        cfg.logger.id = cfg.logger.name.replace("/", "_")
        train_logger = hydra.utils.instantiate(cfg.logger)
        # train_logger.watch(model)
    else:
        train_logger = hydra.utils.instantiate(cfg.logger)
    return train_logger


if __name__ == "__main__":
    # os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    # Set CUDA device IDs

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23"
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    os.environ["TOKENIZERS_PARALLELISM"] = "True"
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_model_path", type=str, default="")
    parser.add_argument("--text_encoder_path", type=str, default="")
    parser.add_argument("--root_data_dir", type=str, default="")

    args = parser.parse_args()
    from hydra import compose, initialize

    with initialize(config_path="./policy_conf", job_name="VPP_Calvinabc_train"):
        cfg = compose(config_name="VPP_Calvinabc_train")
    cfg.model.pretrained_model_path = args.video_model_path
    cfg.model.text_encoder_path = args.text_encoder_path
    cfg.root_data_dir = args.root_data_dir
    cfg.datamodule.root_data_dir = args.root_data_dir
    train(cfg)
