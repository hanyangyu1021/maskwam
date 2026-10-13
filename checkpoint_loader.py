"""Checkpoint inspection and loading for the flat MaskWAM layout."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


REAL_MODEL_TYPE = "mask_wan_mot_real"
SIM_MODEL_TYPE = "mask_wan_mot_sim"
SUPPORTED_MODEL_TYPES = frozenset({REAL_MODEL_TYPE, SIM_MODEL_TYPE})


def _supported_config_payload(
    raw_config: dict[str, Any],
    config_cls: type,
) -> dict[str, Any]:
    allowed_fields = {field.name for field in fields(config_cls)}
    return {
        key: value
        for key, value in raw_config.items()
        if key == "type" or key in allowed_fields
    }


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    model_type: str
    input_mask: bool
    predict_mask: bool
    action_steps: int
    chunk_size: int
    video_height: int
    video_width: int
    max_state_dim: int
    max_action_dim: int
    use_mask: bool
    raw_config: dict[str, Any]


def _resolve_checkpoint_dir(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path).expanduser().resolve()
    if path.is_file() and path.name == "config.json":
        path = path.parent
    config_path = path / "config.json"
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint 目录不存在: {path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint 缺少 config.json: {config_path}")
    return path


def inspect_checkpoint(checkpoint_path: str | Path) -> CheckpointInfo:
    path = _resolve_checkpoint_dir(checkpoint_path)
    raw = json.loads((path / "config.json").read_text(encoding="utf-8"))
    model_type = raw.get("type")
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"不支持的 checkpoint type={model_type!r}，"
            f"当前支持: {sorted(SUPPORTED_MODEL_TYPES)}"
        )

    chunk_size = int(raw.get("chunk_size", 0) or 0)
    action_steps = int(raw.get("n_action_steps", chunk_size) or 0)
    if action_steps > chunk_size:
        raise ValueError(
            f"n_action_steps ({action_steps}) 不能大于 chunk_size ({chunk_size})"
        )

    use_mask = bool(raw.get("use_mask", True))
    return CheckpointInfo(
        path=path,
        model_type=str(model_type),
        input_mask=bool(raw.get("input_mask", use_mask)),
        predict_mask=bool(raw.get("predict_mask", False)),
        action_steps=action_steps,
        chunk_size=chunk_size,
        video_height=int(raw.get("video_height", 0) or 0),
        video_width=int(raw.get("video_width", 0) or 0),
        max_state_dim=int(raw.get("max_state_dim", 32) or 32),
        max_action_dim=int(raw.get("max_action_dim", 32) or 32),
        use_mask=use_mask,
        raw_config=raw,
    )


def _policy_classes(model_type: str):
    if model_type == SIM_MODEL_TYPE:
        from mask_wan_mot_sim.configuration_mask_wan_mot import MaskWanMotConfig
        from mask_wan_mot_sim.modeling_mask_dit_action import MaskDitVideoPolicy
    else:
        from mask_wan_mot_real.configuration_mask_wan_mot import MaskWanMotConfig
        from mask_wan_mot_real.modeling_mask_dit_action import MaskDitVideoPolicy
    return MaskWanMotConfig, MaskDitVideoPolicy


def load_policy(
    checkpoint_path: str | Path,
    *,
    device: str = "cuda",
    dtype: str = "bfloat16",
    wan_checkpoint_path: str | Path | None = None,
    wan_config_path: str | Path | None = None,
    wan_vae_path: str | Path | None = None,
    local_files_only: bool = True,
):
    import torch
    from lerobot.configs.policies import PreTrainedConfig

    info = inspect_checkpoint(checkpoint_path)
    ConfigCls, PolicyCls = _policy_classes(info.model_type)
    serialized_config = _supported_config_payload(info.raw_config, ConfigCls)
    with tempfile.TemporaryDirectory(prefix="maskwam_config_") as temp_dir:
        (Path(temp_dir) / "config.json").write_text(
            json.dumps(serialized_config),
            encoding="utf-8",
        )
        config = PreTrainedConfig.from_pretrained(
            temp_dir, local_files_only=True
        )
    if not isinstance(config, ConfigCls):
        raise TypeError(
            f"checkpoint type={info.model_type!r} 解析为 {type(config).__name__}，"
            f"预期 {ConfigCls.__name__}"
        )

    if wan_checkpoint_path is not None:
        wan_root = Path(wan_checkpoint_path).expanduser().resolve()
        config.wan_checkpoint_path = str(wan_root)
        config.wan_config_path = str(
            Path(wan_config_path).expanduser().resolve()
            if wan_config_path is not None
            else wan_root
        )
        config.wan_vae_path = str(
            Path(wan_vae_path).expanduser().resolve()
            if wan_vae_path is not None
            else wan_root / "Wan2.2_VAE.pth"
        )
    else:
        if wan_config_path is not None:
            config.wan_config_path = str(Path(wan_config_path).expanduser().resolve())
        if wan_vae_path is not None:
            config.wan_vae_path = str(Path(wan_vae_path).expanduser().resolve())

    policy = PolicyCls.from_pretrained(
        str(info.path),
        config=config,
        local_files_only=local_files_only,
    )
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(str(dtype).lower(), torch.bfloat16)
    policy = policy.to(device=device, dtype=torch_dtype) if device else policy.to(dtype=torch_dtype)
    policy.eval()
    return policy, info
