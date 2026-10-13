"""Offline single-chunk XTrainer inference."""

from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path

import numpy as np

from checkpoint_loader import CheckpointInfo, SIM_MODEL_TYPE, load_policy
from .preprocessing import (
    concat_t_mask,
    concat_t_shape,
    consistent_front_crop,
    pad_last_dim,
    pink_bowls_crop,
)


def denormalize_action_chunk(
    normalized: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    action_steps: int,
) -> np.ndarray:
    values = np.asarray(normalized, dtype=np.float32)
    mean = np.asarray(action_mean, dtype=np.float32).reshape(-1)
    std = np.asarray(action_std, dtype=np.float32).reshape(-1)
    if values.ndim != 2:
        raise ValueError(f"action chunk 必须是二维数组，实际 {values.shape}")
    if len(mean) != len(std) or values.shape[1] < len(mean):
        raise ValueError(
            f"action/norm 维度不匹配: action={values.shape}, mean={mean.shape}, std={std.shape}"
        )
    return values[:action_steps, : len(mean)] * std + mean


class OfflineXTrainerPolicy:
    def __init__(
        self,
        checkpoint_path: str | Path,
        norm_path: str | Path,
        *,
        wan_checkpoint_path: str | Path | None = None,
        wan_config_path: str | Path | None = None,
        wan_vae_path: str | Path | None = None,
        t5_embed_path: str | Path | None = None,
        device: str = "cuda",
        state_dim: int = 16,
    ) -> None:
        import torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("请求使用 CUDA，但当前 PyTorch 环境没有可用 GPU")
        self.device = device
        self.dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.state_dim = state_dim
        self.policy, self.info = load_policy(
            checkpoint_path,
            device=device,
            dtype="bfloat16" if device.startswith("cuda") else "float32",
            wan_checkpoint_path=wan_checkpoint_path,
            wan_config_path=wan_config_path,
            wan_vae_path=wan_vae_path,
        )
        self._load_norm(norm_path)
        self._t5_cache: dict[str, object] = {}
        self._default_t5 = None
        self._t5_encoder = getattr(getattr(self.policy, "model", None), "t5_encoder", None)
        if t5_embed_path:
            self._default_t5 = torch.load(
                Path(t5_embed_path).expanduser(),
                map_location="cpu",
            )
        elif self._t5_encoder is None and wan_checkpoint_path:
            self._init_t5_encoder(Path(wan_checkpoint_path))

    def _load_norm(self, norm_path: str | Path) -> None:
        path = Path(norm_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"归一化统计不存在: {path}")
        with path.open("rb") as handle:
            stats = pickle.load(handle)
        state_mean_key = "eep_mean" if "eep_mean" in stats else "qpos_mean"
        state_std_key = "eep_std" if "eep_std" in stats else "qpos_std"
        required = (state_mean_key, state_std_key, "action_mean", "action_std")
        missing = [key for key in required if key not in stats]
        if missing:
            raise KeyError(f"归一化文件缺少字段: {missing}")
        self.state_mean = np.asarray(stats[state_mean_key], dtype=np.float32)
        self.state_std = np.asarray(stats[state_std_key], dtype=np.float32)
        self.action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(stats["action_std"], dtype=np.float32)
        if np.any(self.state_std == 0) or np.any(self.action_std == 0):
            raise ValueError("归一化统计包含 0 标准差")

    def _init_t5_encoder(self, wan_path: Path) -> None:
        from wan22_core.bak.wan.modules.t5 import T5EncoderModel

        self._t5_encoder = T5EncoderModel(
            text_len=512,
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=str(wan_path / "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=str(wan_path / "google" / "umt5-xxl"),
        )

    def _t5_embed(self, prompt: str):
        import torch

        if self._default_t5 is not None:
            value = self._default_t5[0] if isinstance(self._default_t5, list) else self._default_t5
            return value.to(device=self.device, dtype=self.dtype)
        if prompt in self._t5_cache:
            return self._t5_cache[prompt].to(device=self.device, dtype=self.dtype)
        if self._t5_encoder is None:
            raise RuntimeError(
                "没有 T5 embedding。请提供 t5_embed_path，或提供包含 T5 权重的 WAN 路径。"
            )
        with torch.no_grad():
            output = self._t5_encoder([prompt], self.device)
        value = output[0] if isinstance(output, list) else output.squeeze(0)
        self._t5_cache[prompt] = value.detach().cpu()
        return value.to(device=self.device, dtype=self.dtype)

    @staticmethod
    def _resize_tensor(tensor, height: int, width: int, mode: str):
        import torch.nn.functional as functional

        current_height, current_width = tensor.shape[-2:]
        ratio = max(current_width / width, current_height / height)
        resized_height = max(1, int(round(current_height / ratio)))
        resized_width = max(1, int(round(current_width / ratio)))
        kwargs = {"size": (resized_height, resized_width), "mode": mode}
        if mode != "nearest":
            kwargs["align_corners"] = False
        resized = functional.interpolate(tensor, **kwargs)
        pad_height = height - resized_height
        pad_width = width - resized_width
        return functional.pad(
            resized,
            (
                pad_width // 2,
                pad_width - pad_width // 2,
                pad_height // 2,
                pad_height - pad_height // 2,
            ),
        )

    def _build_batch(
        self,
        front: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        state: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        *,
        use_pink_bowls_crop: bool,
    ):
        import torch

        if use_pink_bowls_crop:
            front, mask = pink_bowls_crop(front, mask)
        else:
            front, mask = consistent_front_crop(front, mask)
        t_image = concat_t_shape(front, left, right)
        t_mask = concat_t_mask(mask)

        image_tensor = torch.from_numpy(t_image).permute(2, 0, 1).float().div(255).unsqueeze(0)
        image_tensor = self._resize_tensor(
            image_tensor,
            self.info.video_height,
            self.info.video_width,
            "bilinear",
        ).to(device=self.device, dtype=self.dtype)

        mask_tensor = torch.from_numpy(t_mask).float()[None, None]
        mask_tensor = self._resize_tensor(
            mask_tensor,
            self.info.video_height,
            self.info.video_width,
            "nearest",
        )
        if self.info.model_type == SIM_MODEL_TYPE:
            rgb_mask = torch.zeros((1, 3, *mask_tensor.shape[-2:]), dtype=mask_tensor.dtype)
            rgb_mask[:, 0] = mask_tensor[:, 0]
            mask_tensor = rgb_mask
        else:
            mask_tensor = mask_tensor[:, 0]

        raw_state = np.asarray(state, dtype=np.float32).reshape(-1)
        if len(raw_state) != len(self.state_mean):
            raise ValueError(
                f"state 维度 {len(raw_state)} 与 norm 维度 {len(self.state_mean)} 不一致"
            )
        normalized = (raw_state - self.state_mean) / self.state_std
        normalized = pad_last_dim(normalized[None], self.state_dim)
        return {
            "observation.state": torch.from_numpy(normalized).to(
                device=self.device, dtype=self.dtype
            ),
            "current_image": image_tensor,
            "current_seg_mask": mask_tensor.to(device=self.device, dtype=self.dtype),
            "t5_embed": [self._t5_embed(prompt)],
        }, t_image

    def infer_first_chunk(
        self,
        *,
        front: np.ndarray,
        left_wrist: np.ndarray,
        right_wrist: np.ndarray,
        state: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        use_pink_bowls_crop: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, CheckpointInfo]:
        import torch

        batch, t_image = self._build_batch(
            front,
            left_wrist,
            right_wrist,
            state,
            mask,
            prompt,
            use_pink_bowls_crop=use_pink_bowls_crop,
        )
        self.policy.reset()
        with torch.inference_mode():
            first = self.policy.select_action(batch, save_video_latent=False)
            queue = getattr(self.policy, "_action_queue", deque())
            actions = [first]
            while queue:
                actions.append(queue.popleft())
        normalized = torch.cat(actions, dim=0).float().cpu().numpy()
        chunk = denormalize_action_chunk(
            normalized,
            self.action_mean,
            self.action_std,
            self.info.action_steps,
        )
        return chunk, t_image, self.info

    def infer_video_and_first_chunk(
        self,
        *,
        front: np.ndarray,
        left_wrist: np.ndarray,
        right_wrist: np.ndarray,
        state: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        use_pink_bowls_crop: bool = False,
        tau_v: float = 0.0,
        video_num_inference_steps: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, CheckpointInfo]:
        """Fully denoise and decode the predicted future video plus its action chunk."""
        import torch

        if not 0.0 <= tau_v <= 1.0:
            raise ValueError(f"tau_v 必须在 [0, 1]，实际为 {tau_v}")
        batch, t_image = self._build_batch(
            front,
            left_wrist,
            right_wrist,
            state,
            mask,
            prompt,
            use_pink_bowls_crop=use_pink_bowls_crop,
        )
        self.policy.reset()
        with torch.inference_mode():
            prepared_state = self.policy.prepare_state(batch)
            video_latents, mask_latents, language_embeddings = (
                self.policy.prepare_video_mask_latent_and_t5_embeddings(batch)
            )
            results = self.policy.model.sample_actions(
                prepared_state,
                video_latents,
                mask_latents,
                language_embeddings,
                tau_v=tau_v,
                save_video_latent=True,
                use_video_kv_cache=True,
                video_num_inference_steps=video_num_inference_steps,
            )

        normalized = (
            results["pred"][0, :, : len(self.action_mean)].float().cpu().numpy()
        )
        chunk = denormalize_action_chunk(
            normalized,
            self.action_mean,
            self.action_std,
            self.info.action_steps,
        )
        predicted = results.get("video_latent", {}).get("predicted_frames")
        if predicted is None:
            raise RuntimeError("模型没有返回 predicted_frames")
        frames = (
            predicted[0]
            .permute(1, 2, 3, 0)
            .mul(255)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .numpy()
        )
        return chunk, frames, t_image, self.info
