#!/usr/bin/env python3
"""T-shape consumer dataset and collator for processed real-robot episodes.

Handles non-square T-shape images (e.g., 480x384 instead of 224x224).
Uses (video_height, video_width) from config instead of a single image_size.
"""

from __future__ import annotations

import random
import traceback
from typing import Any, Dict, List, Optional, Sequence

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from torch.utils.data import ConcatDataset, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.hdf5_dataset_xtrainer_T import HDF5VLADataset

def process_video_images(
    current_image: np.ndarray,
    future_images: np.ndarray,
    height: int = 480,
    width: int = 384,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process T-shape video images with non-square target size."""
    combined = np.concatenate([current_image[None, ...], future_images], axis=0)
    combined_tensor = torch.from_numpy(combined).permute(0, 3, 1, 2).float() / 255.0
    combined_tensor = resize_with_pad(combined_tensor, width, height, pad_value=0.0)
    return combined_tensor[0], combined_tensor[1:]


def process_video_masks(
    current_mask: np.ndarray,
    future_masks: np.ndarray,
    height: int = 480,
    width: int = 384,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process T-shape masks with non-square target size."""
    combined = np.concatenate([current_mask[None, ...], future_masks], axis=0)
    combined_tensor = torch.from_numpy(combined.astype(np.float32)).unsqueeze(1)
    combined_tensor = resize_with_pad(combined_tensor, width, height, pad_value=0.0, mode="nearest")
    combined_tensor = combined_tensor.squeeze(1)
    return combined_tensor[0], combined_tensor[1:]


def resize_with_pad(
    img: torch.Tensor,
    width: int,
    height: int,
    pad_value: float = 0.0,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize an image batch/tensor while preserving aspect ratio and center pad."""
    if img.ndim != 4:
        raise ValueError(f"Expected (b, c, h, w), got {img.shape}")

    cur_height, cur_width = img.shape[2:]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = max(1, int(round(cur_height / ratio)))
    resized_width = max(1, int(round(cur_width / ratio)))

    interpolate_kwargs = {
        "size": (resized_height, resized_width),
        "mode": mode,
    }
    if mode != "nearest":
        interpolate_kwargs["align_corners"] = False

    resized = F.interpolate(img, **interpolate_kwargs)

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))
    pad_w_left = pad_width // 2
    pad_h_top = pad_height // 2
    return F.pad(
        resized,
        (pad_w_left, pad_width - pad_w_left, pad_h_top, pad_height - pad_h_top),
        value=pad_value,
    )


def _process_language_embeddings_batch(language_embeddings: List[torch.Tensor], text_len: int = 512) -> torch.Tensor:
    """Pad a batch of variable-length T5 embeddings to [B, text_len, dim]."""
    padded_embeddings: List[torch.Tensor] = []
    for emb in language_embeddings:
        if emb.ndim == 3 and emb.shape[0] == 1:
            emb = emb.squeeze(0)
        if emb.shape[0] <= text_len:
            padded = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])], dim=0)
        else:
            padded = emb[:text_len]
        padded_embeddings.append(padded)
    return torch.stack(padded_embeddings, dim=0)


class ConsumerDataset(Dataset):
    """T-shape consumer dataset. Uses (video_height, video_width) for non-square images."""

    def __init__(
        self,
        config,
        base_dataset: Optional[HDF5VLADataset] = None,
    ):
        super().__init__()
        self.config = config

        self.img_history_size = config.dataset.img_history_size
        self.use_hdf5 = config.dataset.use_hdf5
        # Whether to load/process mask data (read from cfg.training.use_mask)
        self.use_mask = bool(getattr(config.training, "use_mask", True)) if hasattr(config, "training") else True
        # T-shape uses separate height/width
        self.video_height = config.dataset.video_height
        self.video_width = config.dataset.video_width

        self.hdf5_dataset = None
        self.base_dataset_list = []
        self.DATASET_NAMES = []

        if self.use_hdf5:
            if base_dataset is not None:
                self.base_dataset_list = [base_dataset]
                self.hdf5_dataset = base_dataset
                self.DATASET_NAMES.extend(list(base_dataset.get_dataset_name()))
            else:
                print(f"Loading dataset {config.dataset.dataset_list}")
                for baseset_dict in config.dataset.dataset_list:
                    print(f"Loading dataset {baseset_dict}")
                    baseset = instantiate(baseset_dict, cfg=config, _recursive_=False)
                    self.base_dataset_list.append(baseset)
                    self.DATASET_NAMES.extend(list(baseset_dict.sample_weights.keys()))

                if len(self.base_dataset_list) == 1:
                    self.hdf5_dataset = self.base_dataset_list[0]
                else:
                    self.hdf5_dataset = ConcatDataset(self.base_dataset_list)

            self.dataset_name2id = {name: i for i, name in enumerate(self.DATASET_NAMES)}
            self.dataset_id2name = {i: name for i, name in enumerate(self.DATASET_NAMES)}
            print(self.dataset_name2id)
            print(f"Training Model on `{list(self.DATASET_NAMES)}` from `{config.dataset.hdf5_dir}`")
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return len(self.hdf5_dataset)

    def get_dataset_name2id(self) -> Dict[str, int]:
        return self.dataset_name2id

    def get_dataset_id2name(self) -> Dict[int, str]:
        return self.dataset_id2name

    def __getitem__(self, index: int) -> Dict[str, Any]:
        while True:
            data_dict: Optional[Dict[str, Any]] = None
            try:
                sample = self.hdf5_dataset[index]
                meta = sample["meta"]

                data_dict = {
                    "dataset_name": meta["dataset_name"],
                    "data_idx": self.dataset_name2id[meta["dataset_name"]],
                    "states": sample["state"],
                    "actions": sample["actions"],
                    "state_elem_mask": sample["state_indicator"],
                    "instructions": meta["instruction"],
                }

                # T-shape images: (IMG_HISTORY_SIZE, H_T, W, 3)
                front_images = self._preprocess_image_history(sample["cam_high"])
                data_dict["images"] = front_images

                # Only process mask history when use_mask=True
                if self.use_mask and "cam_high_det_mask" in sample:
                    segment_masks = self._preprocess_mask_history(sample["cam_high_det_mask"])
                    data_dict["segment_masks"] = segment_masks

                # T-shape video data (single-view interface)
                if "current_image" in sample and "future_images" in sample:
                    cur_img, fut_imgs = process_video_images(
                        sample["current_image"], sample["future_images"],
                        height=self.video_height, width=self.video_width,
                    )
                    data_dict["current_image"] = cur_img
                    data_dict["future_images"] = fut_imgs

                if "current_seg_mask" in sample and "future_seg_masks" in sample and self.use_mask:
                    cur_mask, fut_masks = process_video_masks(
                        sample["current_seg_mask"], sample["future_seg_masks"],
                        height=self.video_height, width=self.video_width,
                    )
                    data_dict["current_seg_mask"] = cur_mask
                    data_dict["future_seg_masks"] = fut_masks

                if "t5_embed" in sample and sample["t5_embed"] is not None:
                    data_dict["t5_embed"] = sample["t5_embed"]

                for key in ("states", "actions", "state_elem_mask"):
                    if isinstance(data_dict[key], np.ndarray):
                        data_dict[key] = torch.from_numpy(data_dict[key]).float()

                return data_dict
            except BaseException as exc:
                if data_dict is not None:
                    print(f"Error while processing sample from {data_dict.get('dataset_name')}: {exc}")
                else:
                    print(f"Error while processing sample: {exc}")
                traceback.print_exc()
                index = (index + 1) % max(1, len(self))

    def _preprocess_image_history(
        self,
        images: np.ndarray,
    ) -> List[torch.Tensor]:
        """Preprocess T-shape image history using (video_height, video_width)."""
        processed: List[torch.Tensor] = []
        for image in images:
            tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float()
            tensor = resize_with_pad(
                tensor.unsqueeze(0),
                self.video_width,
                self.video_height,
                pad_value=0.0,
            )[0]
            processed.append(tensor)

        return processed

    def _preprocess_mask_history(
        self,
        masks: np.ndarray,
    ) -> List[torch.Tensor]:
        processed: List[torch.Tensor] = []
        for mask in masks:
            tensor = torch.from_numpy(mask.astype(np.float32))[None, ...]
            tensor = resize_with_pad(
                tensor.unsqueeze(0),
                self.video_width,
                self.video_height,
                pad_value=0.0,
                mode="nearest",
            )[0]
            processed.append(tensor)
        return processed


class DataCollator:
    """Collate T-shape real-robot samples into a training batch."""

    def __init__(self, config) -> None:
        self.config = config
        self.IMG_HISORY_SIZE = config.dataset.img_history_size
        self.use_mask = bool(getattr(config.training, "use_mask", True)) if hasattr(config, "training") else True

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        states = torch.stack([instance["states"] for instance in instances], dim=0)
        actions = torch.stack([instance["actions"] for instance in instances], dim=0)
        state_elem_mask = torch.stack([instance["state_elem_mask"] for instance in instances], dim=0)

        images = torch.stack([torch.stack(instance["images"], dim=0) for instance in instances], dim=0)

        # Only stack mask data when use_mask=True and data is present
        if self.use_mask and "segment_masks" in instances[0]:
            segment_masks = torch.stack(
                [torch.stack(instance["segment_masks"], dim=0) for instance in instances],
                dim=0,
            )
        else:
            segment_masks = None

        tasks = [instance["instructions"] for instance in instances]
        dataset_names = [instance["dataset_name"] for instance in instances]
        data_idx = torch.as_tensor([instance["data_idx"] for instance in instances], dtype=torch.long)

        front_images = images.float() / 255.0
        front_masks = segment_masks.float() if segment_masks is not None else None

        batch = {
            "dataset_name": dataset_names,
            "data_idx": data_idx,
            "observation.images.front": front_images,
            "observation.images.top_head": front_images,
            "observation.state": states[:, 0, :],
            "observation.state_indicator": state_elem_mask,
            "action": actions,
            "task": tasks,
        }
        if front_masks is not None:
            batch["observation.images.front.segment_mask"] = front_masks
            batch["observation.images.top_head.segment_mask"] = front_masks
        if all("t5_embed" in instance and instance["t5_embed"] is not None for instance in instances):
            language_embeddings = [torch.as_tensor(instance["t5_embed"]) for instance in instances]
            batch["t5_embed"] = _process_language_embeddings_batch(language_embeddings)
        if "current_image" in instances[0]:
            batch["current_image"] = torch.stack([instance["current_image"] for instance in instances], dim=0)
        if "future_images" in instances[0]:
            batch["future_images"] = torch.stack([instance["future_images"] for instance in instances], dim=0)
        if "current_seg_mask" in instances[0]:
            batch["current_seg_mask"] = torch.stack([instance["current_seg_mask"] for instance in instances], dim=0)
        if "future_seg_masks" in instances[0]:
            batch["future_seg_masks"] = torch.stack([instance["future_seg_masks"] for instance in instances], dim=0)

        return batch


def visualize_T_sample(
    batch: Dict[str, Any],
    save_path: str = "real_T_vis.png",
) -> None:
    """Visualize T-shape current image/mask and future images/masks."""
    cur_img = batch["current_image"][0]
    fut_imgs = batch["future_images"][0]
    num_future = fut_imgs.shape[0]
    ncols = 1 + num_future

    has_mask = "current_seg_mask" in batch and "future_seg_masks" in batch
    nrows = 1 + int(has_mask)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)

    def _to_numpy_img(t: torch.Tensor) -> np.ndarray:
        img = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        return (img * 255).astype(np.uint8)

    row_idx = 0
    axes[row_idx, 0].imshow(_to_numpy_img(cur_img))
    axes[row_idx, 0].set_title("T-shape Current", fontsize=9)
    axes[row_idx, 0].axis("off")
    for i in range(num_future):
        axes[row_idx, i + 1].imshow(_to_numpy_img(fut_imgs[i]))
        axes[row_idx, i + 1].set_title(f"T-shape Fut {i}", fontsize=9)
        axes[row_idx, i + 1].axis("off")
    row_idx += 1

    if has_mask:
        cur_mask = batch["current_seg_mask"][0].cpu().numpy()
        fut_masks = batch["future_seg_masks"][0].cpu().numpy()

        axes[row_idx, 0].imshow(cur_mask, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 0].set_title("T-shape Cur Mask", fontsize=9)
        axes[row_idx, 0].axis("off")
        for i in range(num_future):
            axes[row_idx, i + 1].imshow(fut_masks[i], cmap="gray", vmin=0, vmax=1)
            axes[row_idx, i + 1].set_title(f"T-shape Fut Mask {i}", fontsize=9)
            axes[row_idx, i + 1].axis("off")
        row_idx += 1

    task_text = batch.get("task", [""])[0]
    if task_text:
        fig.suptitle(task_text, fontsize=11, y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {save_path}")


@hydra.main(
    version_base=None,
    config_path="./configs",
    config_name="train_maskwam",
)
def test(cfg):
    cfg.dataloader.batch_size = 1
    cfg.dataloader.num_workers = 1
    cfg.dataloader.shuffle = True
    cfg.dataloader.pin_memory = False
    cfg.dataloader.persistent_workers = False

    dataset = ConsumerDataset(cfg)
    print(len(dataset))
    print(dataset.__getitem__(0).keys())
    print(dataset.__getitem__(0)["states"].shape)
    print(dataset.__getitem__(0)["actions"].shape)

    data_collator = DataCollator(cfg)
    train_dataloader = hydra.utils.instantiate(cfg.dataloader, dataset=dataset, collate_fn=data_collator)
    num_vis = 8
    for i, batch in enumerate(train_dataloader):
        if i >= num_vis:
            break
        print(f"--- Sample {i} ---")
        print(batch.keys())
        print(batch["observation.images.front"].shape)
        print(batch["observation.state"].shape)
        print(batch["action"].shape)
        if "future_images" in batch:
            print(f"future_images: {batch['future_images'].shape}")
        if "current_seg_mask" in batch:
            print(f"current_seg_mask: {batch['current_seg_mask'].shape}")
        if "future_seg_masks" in batch:
            print(f"future_seg_masks: {batch['future_seg_masks'].shape}")

        visualize_T_sample(batch, save_path=f"real_T_vis_{i}.png")


if __name__ == "__main__":
    test()
