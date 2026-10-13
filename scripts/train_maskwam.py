"""Training script for real/sim MaskWAM models with T-shape multi-camera input.

The selected dataset config provides the platform-specific consumer dataset,
while this entry point passes separate video_height/video_width to the model.

Usage:
    scripts/train_maskwam.sh xtrainer
    scripts/train_maskwam.sh libero
    scripts/train_maskwam.sh xtrainer training.batch_size=2
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import copy
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Compatibility for older wandb releases on NumPy >= 2.0.
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128

from torch.utils.data import random_split
import torch
import wandb
from tqdm import tqdm
import random
import socket
from pathlib import Path
import transformers
from transformers import logging
import diffusers
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from accelerate.utils import set_seed
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration
from utils.logger import save_wandb, initialize_wandb
import matplotlib.pyplot as plt
from utils.vis import plot_all_joints

logging.set_verbosity_error()

def set_global_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cast_training_params(model, dtype=torch.float32):
    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.parameters():
            if param.requires_grad:
                param.data = param.to(dtype)

def cycle(iterable):
    """The equivalent of itertools.cycle, but safe for Pytorch dataloaders."""
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

def safe_mean(x):
    return np.mean(x) if len(x) > 0 else 0.

@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="train_maskwam",
)
def train(cfg: DictConfig) -> None:
    model_type = str(getattr(cfg.model, "type", "mask_wan_mot_sim"))
    if model_type == "mask_wan_mot_real":
        from mask_wan_mot_real.modeling_mask_dit_action import MaskDitVideoPolicy
        from mask_wan_mot_real.configuration_mask_wan_mot import MaskWanMotConfig
    elif model_type == "mask_wan_mot_sim":
        from mask_wan_mot_sim.modeling_mask_dit_action import MaskDitVideoPolicy
        from mask_wan_mot_sim.configuration_mask_wan_mot import MaskWanMotConfig
    else:
        raise ValueError(f"Unsupported MaskWAM model type: {model_type}")
    print(f"Using {model_type} action expert joint training with T-shape input")
    # for better performance, but reduce reproducibility
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Enable TF32 for faster training on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator_project_config = ProjectConfiguration(total_limit=cfg.checkpoints_total_limit)
    accelerator = Accelerator(
        mixed_precision=cfg.training.mixed_precision,
        gradient_accumulation_steps=cfg.training.grad_accumulation_steps,
        deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=cfg.deepspeed),
        project_config=accelerator_project_config,
    )

    expected_world_size = os.environ.get("MASKWAM_EXPECTED_WORLD_SIZE")
    if expected_world_size and accelerator.num_processes != int(expected_world_size):
        raise RuntimeError(
            f"Distributed world size is {accelerator.num_processes}, "
            f"but the launcher requested {expected_world_size} processes"
        )
    print(
        f"[distributed] host={socket.gethostname()} "
        f"machine_rank={os.environ.get('MASKWAM_MACHINE_RANK', 'unknown')} "
        f"rank={accelerator.process_index} local_rank={accelerator.local_process_index} "
        f"world_size={accelerator.num_processes} "
        f"is_main_process={accelerator.is_main_process}",
        flush=True,
    )

    if cfg.resume_ckpt:
        resume_ckpt = cfg.resume_ckpt
        dataset_cfg = cfg.dataset
        training_keys = set(cfg.training.keys())
        cfg_prev = OmegaConf.load(Path(cfg.resume_ckpt).parent / "train_maskwam.yaml")
        OmegaConf.set_struct(cfg_prev, False)
        cfg_prev.pop("dataset", None)
        cfg_prev.training = OmegaConf.create(
            {key: value for key, value in cfg_prev.training.items() if key in training_keys}
        )
        cfg = OmegaConf.merge(cfg_prev, {"resume_ckpt": resume_ckpt, "dataset": dataset_cfg})
        cfg.ckpt_save_dir = os.path.dirname(cfg.resume_ckpt)

    if not cfg.debug:
        if accelerator.is_main_process:
            cfg = initialize_wandb(cfg)
            if not cfg.resume_ckpt:
                wandb.config.update(OmegaConf.to_container(cfg))
                os.makedirs(cfg.ckpt_save_dir, exist_ok=True)
                save_wandb(wandb, cfg.ckpt_save_dir)
                OmegaConf.save(cfg, Path(cfg.ckpt_save_dir) / "train_maskwam.yaml")

        accelerator.wait_for_everyone()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif accelerator.mixed_precision == "no":
        weight_dtype = torch.float32

    ############# Config Model ############
    accelerator.print("Initialize Mask WAN MOT Model (T-shape)")

    # Pass mask-specific config to MaskWanMotConfig.
    # `input_mask` controls whether the current-frame mask is used as a condition,
    # while `predict_mask` controls future mask prediction/loss.
    use_mask = bool(getattr(cfg.training, "use_mask", True))
    input_mask = bool(getattr(cfg.training, "input_mask", False)) if use_mask else False
    predict_mask = bool(getattr(cfg.training, "predict_mask", False)) if use_mask else False

    config = MaskWanMotConfig(
        n_action_steps=cfg.dataset.action_chunk_size + cfg.dataset.img_history_size - 1,
        chunk_size=cfg.dataset.action_chunk_size + cfg.dataset.img_history_size - 1,
        optimizer_lr=cfg.training.optimizer_lr,
        optimizer_betas=tuple(cfg.training.optimizer_betas),
        optimizer_eps=cfg.training.optimizer_eps,
        optimizer_weight_decay=cfg.training.optimizer_weight_decay,
        scheduler_warmup_steps=cfg.training.scheduler_warmup_steps,
        scheduler_decay_steps=cfg.training.scheduler_decay_steps,
        scheduler_decay_lr=cfg.training.scheduler_decay_lr,
        resize_imgs_with_padding=(cfg.dataset.video_height, cfg.dataset.video_width),
        batch_size=cfg.training.batch_size,
        # Video settings: T-shape uses separate height/width
        num_video_frames=cfg.dataset.num_video_frames,
        video_height=cfg.dataset.video_height,
        video_width=cfg.dataset.video_width,
        # WAN pretrained paths
        wan_checkpoint_path=cfg.model.wan.checkpoint_path,
        wan_config_path=cfg.model.wan.config_path,
        wan_vae_path=cfg.model.wan.vae_path,
        wan_precision=cfg.model.wan.precision,
        # Mask settings
        use_mask=use_mask,
        input_mask=input_mask,
        predict_mask=predict_mask,
        mask_random=float(getattr(cfg.training, "mask_random", 1.0)),
        mask_loss_weight=float(getattr(cfg.training, "mask_loss_weight", 1.0)),
        action_loss_weight=float(getattr(cfg.training, "action_loss_weight", 1.0)),
        video_loss_weight=float(getattr(cfg.training, "video_loss_weight", 1.0)),
    )

    if cfg.resume_ckpt:
        accelerator.print(f"Resuming from checkpoint {cfg.resume_ckpt}")
        policy = MaskDitVideoPolicy.from_pretrained(
            os.path.join(cfg.resume_ckpt, "model"),
            config=config,
            local_files_only=True,
        )
    else:
        policy = MaskDitVideoPolicy(config)
        accelerator.print("Initialized fresh MaskDitVideoPolicy (T-shape)")

    # Set joint loss weights
    if hasattr(policy, "model"):
        policy.model.action_loss_weight = float(getattr(cfg.training, "action_loss_weight", 1.0))
        policy.model.video_loss_weight = float(getattr(cfg.training, "video_loss_weight", 1.0))
        policy.model.mask_loss_weight = float(getattr(cfg.training, "mask_loss_weight", 0.4))
        accelerator.print(
            f"Joint loss weights: action={policy.model.action_loss_weight}, "
            f"video={policy.model.video_loss_weight}, "
            f"mask={policy.model.mask_loss_weight}"
        )
        accelerator.print(
            f"use_mask={policy.model.use_mask}, input_mask={policy.model.input_mask}, "
            f"predict_mask={policy.model.predict_mask}"
        )

    policy.to(weight_dtype)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    ############# Config Optimizer ############
    accelerator.print("Initialize Optimizer")
    total_train_steps = (
        cfg.training.max_training_steps
        * accelerator.num_processes
        * cfg.training.grad_accumulation_steps
    )
    optimizer = config.get_optimizer_preset().build(
        filter(lambda p: p.requires_grad, policy.parameters())
    )
    config.scheduler_warmup_steps *= accelerator.num_processes
    config.scheduler_decay_steps *= accelerator.num_processes
    lr_scheduler = config.get_scheduler_preset().build(optimizer, total_train_steps)

    ############# Config Dataset and Dataloader ############
    accelerator.print("Initialize T-shape Dataset and Dataloader")

    if cfg.dataset.type == "hdf5":
        DatasetClass = hydra.utils.get_class(cfg.dataset.consumer_dataset._target_)
        CollatorClass = hydra.utils.get_class(cfg.dataset.data_collator._target_)
        train_dataset = DatasetClass(config=cfg)
        data_collator = CollatorClass(config=cfg)
        train_dataloader = hydra.utils.instantiate(
            cfg.dataloader, dataset=train_dataset, collate_fn=data_collator
        )
    else:
        raise NotImplementedError("Only support `hdf5` dataset format now.")

    ############# Prepare `accelerator` ############
    accelerator.print("Prepare action dataloader")
    policy, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, train_dataloader, lr_scheduler
    )

    if cfg.resume_ckpt:
        accelerator.print(f"Resuming optimizer and scheduler from checkpoint {cfg.resume_ckpt}")
        accelerator.load_state(
            os.path.join(cfg.resume_ckpt, "state", "training_state.pth")
        )
        accelerator.wait_for_everyone()

    total_batch_size = (
        cfg.training.batch_size
        * accelerator.num_processes
        * cfg.training.grad_accumulation_steps
    )

    accelerator.print(f"***** Running {model_type} T-shape training *****")
    accelerator.print(f"  Total parameters:     {num_total_params}")
    accelerator.print(f"  Trainable parameters: {num_learnable_params}")
    accelerator.print(f"  Num examples VLA = {len(train_dataset)}")
    accelerator.print(f"  Instantaneous batch size per device = {cfg.training.batch_size}")
    accelerator.print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    accelerator.print(
        f"  Gradient Accumulation steps = {accelerator.gradient_accumulation_steps}"
    )
    accelerator.print(f"  Total optimization steps = {cfg.training.max_training_steps}")
    accelerator.print(f"  use_mask = {use_mask}, predict_mask = {predict_mask}")
    accelerator.print(f"  T-shape video: {cfg.dataset.video_height}x{cfg.dataset.video_width}")

    train_loss = []
    train_metrics = ["loss"]
    val_metrics = ["loss", "dist", "gt", "pred"]
    train_log_dict = {"action_training": {m: [] for m in train_metrics}}
    val_log_dict   = {"action_training": {m: [] for m in val_metrics}}

    train_loader_iter = cycle(train_dataloader)
    current_step = int(cfg.resume_ckpt.split("/")[-1]) + 1 if cfg.resume_ckpt else 0
    progress_bar = tqdm(
        range(cfg.training.max_training_steps),
        ncols=100,
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Training")
    progress_bar.update(current_step)

    for n_iter in range(current_step, cfg.training.max_training_steps):
        policy.train()
        for _ in range(accelerator.gradient_accumulation_steps):
            batch = next(train_loader_iter)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(weight_dtype)

            with accelerator.accumulate(policy):
                output = policy.forward(batch)
                loss = output["loss"]
                train_log_dict["action_training"]["loss"].append(output["flow_loss"])
                if "video_loss" in output:
                    train_log_dict["action_training"].setdefault("video_loss", []).append(output["video_loss"])
                if "mask_loss" in output:
                    train_log_dict["action_training"].setdefault("mask_loss", []).append(output["mask_loss"])

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy.parameters(), cfg.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            train_loss.append(loss.item())
        if (
            (n_iter + 1) % (cfg.training.train_frequency) == 0
            and accelerator.is_main_process
            and not cfg.debug
        ):
            total_norm = 0.0
            for p in policy.parameters():
                if p.requires_grad:
                    param_norm = p.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            log_dict = {
                "train/loss": safe_mean(train_loss),
                "train/flow_loss": safe_mean(train_log_dict["action_training"]["loss"]),
                "train/video_loss": safe_mean(train_log_dict["action_training"].get("video_loss", [])),
                "train/total_weight_norm": total_norm,
                "train/iter": n_iter,
                "train/lr": optimizer.param_groups[0]["lr"],
            }
            # Log mask loss if available
            if "mask_loss" in train_log_dict["action_training"]:
                log_dict["train/mask_loss"] = safe_mean(train_log_dict["action_training"]["mask_loss"])

            wandb.log(log_dict, step=n_iter)
            train_loss = []
            train_log_dict = {"action_training": {m: [] for m in train_metrics}}

        progress_bar.update(1)
        progress_bar.set_postfix({"loss": safe_mean(train_loss)})

        # validation
        if (n_iter + 1) % (cfg.training.eval_frequency) == 0:
            accelerator.wait_for_everyone()
            accelerator.print(f"\n Evaluate at n_iter {n_iter}.")
            eval_model = accelerator.unwrap_model(policy)
            eval_model.eval()

            for i in tqdm(
                range(cfg.training.max_evaluation_steps),
                desc="Validation",
                ncols=10,
                disable=not accelerator.is_local_main_process,
            ):
                with torch.no_grad():
                    batch = next(train_loader_iter)
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(weight_dtype)

                    output = eval_model(batch)
                    loss = output["loss"]

                    output_res = eval_model.forward_evaluate(batch, tau_v=1.0)
                    pred, gt = output_res["pred"], output_res["gt"]
                    val_log_dict["action_training"]["pred"] = pred
                    val_log_dict["action_training"]["gt"]   = gt

                    all_predictions, all_targets, all_loss = accelerator.gather_for_metrics(
                        (pred, gt, loss)
                    )
                    dist = (all_predictions[:, :, :14] - all_targets[:, :, :14]).abs().mean()
                    all_loss = all_loss.mean()

                    val_log_dict["action_training"]["loss"].append(all_loss.item())
                    val_log_dict["action_training"]["dist"].append(dist.item())

            eval_model.train()

            if accelerator.is_main_process and not cfg.debug:
                fig = plot_all_joints(
                    val_log_dict["action_training"]["gt"][0].float().detach().cpu().numpy(),
                    val_log_dict["action_training"]["pred"][0].float().detach().cpu().numpy(),
                    wandb=True,
                )
                wandb.log(
                    {
                        "val/action_loss": safe_mean(val_log_dict["action_training"]["loss"]),
                        "val/action_dist": safe_mean(val_log_dict["action_training"]["dist"]),
                        "val/iter": n_iter,
                        "all_angles_subplot": wandb.Image(fig),
                    },
                    step=n_iter,
                )

                val_log_dict = {"action_training": {m: [] for m in val_metrics}}

            plt.close("all")
            torch.cuda.empty_cache()
            accelerator.wait_for_everyone()

        # save checkpoint
        if (
            (n_iter + 1) % (cfg.training.ckpt_frequency ) == 0
            and not cfg.debug
        ):
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                ckpt_model = accelerator.unwrap_model(policy)
                model_to_save = copy.deepcopy(ckpt_model)
                model_to_save.save_pretrained(
                    os.path.join(cfg.ckpt_save_dir, f"{n_iter}", "model"),
                    safe_serialization=False,
                )
                del model_to_save
                gc.collect()

            if cfg.save_training_state:
                accelerator.save_state(
                    os.path.join(cfg.ckpt_save_dir, f"{n_iter}", "state", "training_state.pth")
                )
                accelerator.print(f"Saved state checkpoint at step {n_iter}.")

            torch.cuda.empty_cache()

    accelerator.end_training()

if __name__ == "__main__":
    train()
