from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("platform", ["libero", "xtrainer"])
def test_platform_config_is_parameterized(platform):
    text = (ROOT / "configs" / "dataset" / f"{platform}.yaml").read_text()
    assert f"platform: {platform}" in text
    assert "${data_root}" in text


def test_training_config_uses_external_roots():
    text = (ROOT / "configs" / "train_maskwam.yaml").read_text()
    for variable in (
        "MASKWAM_ROOT",
        "MASKWAM_DATA_ROOT",
        "MASKWAM_PRETRAINED_ROOT",
        "MASKWAM_CHECKPOINT_ROOT",
    ):
        assert variable in text


def test_mask_input_is_enabled_without_copy_initialization():
    training_text = (ROOT / "configs" / "train_maskwam.yaml").read_text()
    real_dataset_text = (ROOT / "configs" / "dataset" / "xtrainer.yaml").read_text()
    libero_dataset_text = (ROOT / "configs" / "dataset" / "libero.yaml").read_text()
    assert "input_mask: ${dataset.input_mask}" in training_text
    assert "input_mask: true" in real_dataset_text
    assert "input_mask: false" in libero_dataset_text

    model_text = (
        ROOT / "mask_wan_mot_sim" / "modeling_mask_dit_action.py"
    ).read_text()
    assert "new_weight[..., old_out_dim:, :].copy_" not in model_text
    assert "current_seg_mask = torch.zeros_like(current_seg_mask)" not in model_text
    assert "if drop_input_mask:" in model_text


def test_attention_mask_auxiliary_loss_is_not_configured():
    training_text = (ROOT / "configs" / "train_maskwam.yaml").read_text()
    model_text = (
        ROOT / "mask_wan_mot_sim" / "modeling_mask_dit_action.py"
    ).read_text()
    assert "attn_mask_" not in training_text
    assert "capture_action_video_attn" not in model_text


def test_released_libero_checkpoint_is_the_eval_default():
    checkpoint_text = (ROOT / "configs" / "checkpoints.yaml").read_text()
    eval_text = (ROOT / "platforms" / "libero" / "eval_libero_T.py").read_text()
    assert "${root}/MaskWAM-libero-ckpt" in checkpoint_text
    assert 'CHECKPOINT_ROOT / "MaskWAM-libero-ckpt"' in eval_text
    assert 'DATA_ROOT / "libero_vam" / "qpos_mean_std_online.pkl"' in eval_text


def test_released_libero_dataset_is_the_training_default():
    dataset_text = (ROOT / "configs" / "dataset" / "libero.yaml").read_text()
    assert "libero_10_robot: 1.0" in dataset_text


def test_flash_attention_is_installed_after_pytorch():
    requirements = (ROOT / "requirements.txt").read_text().splitlines()
    install_text = (ROOT / "docs" / "INSTALL.md").read_text()
    assert "flash-attn" not in requirements
    assert "pip install --no-build-isolation flash-attn" in install_text


def test_training_logger_does_not_import_legacy_lerobot_policy_path():
    logger_text = (ROOT / "utils" / "logger.py").read_text()
    assert "lerobot.common.policies" not in logger_text
