import json
from dataclasses import dataclass

import pytest

from checkpoint_loader import _supported_config_payload, inspect_checkpoint


def write_config(tmp_path, **overrides):
    config = {
        "type": "mask_wan_mot_real",
        "input_mask": True,
        "predict_mask": True,
        "chunk_size": 48,
        "n_action_steps": 48,
        "video_height": 384,
        "video_width": 320,
        "max_state_dim": 32,
        "max_action_dim": 32,
    }
    config.update(overrides)
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def test_inspect_real_checkpoint(tmp_path):
    info = inspect_checkpoint(write_config(tmp_path))
    assert info.model_type == "mask_wan_mot_real"
    assert info.input_mask is True
    assert info.action_steps == 48
    assert (info.video_height, info.video_width) == (384, 320)


def test_inspect_sim_checkpoint(tmp_path):
    info = inspect_checkpoint(
        write_config(
            tmp_path,
            type="mask_wan_mot_sim",
            chunk_size=16,
            n_action_steps=16,
        )
    )
    assert info.model_type == "mask_wan_mot_sim"
    assert info.action_steps == 16


@pytest.mark.parametrize("model_type", ["mask_wan_mot", "mask_wan_mot2"])
def test_rejects_legacy_model_types(tmp_path, model_type):
    with pytest.raises(ValueError, match="不支持"):
        inspect_checkpoint(write_config(tmp_path, type=model_type))


def test_rejects_unknown_model(tmp_path):
    with pytest.raises(ValueError, match="不支持"):
        inspect_checkpoint(write_config(tmp_path, type="fastwam"))


def test_rejects_action_steps_larger_than_chunk(tmp_path):
    with pytest.raises(ValueError, match="不能大于"):
        inspect_checkpoint(
            write_config(tmp_path, chunk_size=16, n_action_steps=48)
        )


def test_unsupported_checkpoint_metadata_is_filtered():
    @dataclass
    class Config:
        chunk_size: int = 48

    payload = _supported_config_payload(
        {"type": "test", "chunk_size": 16, "removed_option": True},
        Config,
    )
    assert payload == {"type": "test", "chunk_size": 16}
