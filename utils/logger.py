import os
from pathlib import Path
import json

import wandb
from omegaconf import open_dict

def save_wandb(wandb, save_dir):
    wandb_info = {
        "entity": wandb.run.entity,
        "project": wandb.run.project,
        "name": wandb.run.name,
        "id": wandb.run.id
    }
    with open(os.path.join(save_dir, "wandb_run_info.json"), "w") as f:
        json.dump(wandb_info, f, indent=2)


def initialize_wandb(cfg):
    if cfg.resume_ckpt:
        save_dir = Path(cfg.resume_ckpt)
        with open(save_dir.parent / "wandb_run_info.json", "r") as f:
            wandb_info = json.load(f)
        cfg.wandb_entity = wandb_info["entity"]
        cfg.wandb_project = wandb_info["project"]
        cfg.exp_name = wandb_info["name"]
        with open_dict(cfg):
            cfg['wandb_id'] = wandb_info["id"]

    wandb.init(project=cfg.wandb_project,
               name=f"{cfg.exp_name}",
               id= None if cfg.resume_ckpt else None,
               mode="online",
               resume=None if cfg.resume_ckpt else None)
            #    id=cfg.wandb_id if cfg.resume_ckpt else None,
            #    mode="online",
            #    resume="must" if cfg.resume_ckpt else None)

    return cfg
