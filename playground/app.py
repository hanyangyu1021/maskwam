"""Gradio playground for one-click, one-chunk XTrainer inference."""

from __future__ import annotations

import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np

from platforms.real.infer.episode import load_episode_frame
from platforms.real.infer.policy import OfflineXTrainerPolicy
from platforms.real.infer.preprocessing import overlay_mask
from platforms.real.infer.sam3_point import SAM3PointSegmenter
from paths import (
    CHECKPOINT_ROOT,
    DATA_ROOT,
    PRETRAINED_ROOT,
    PROJECT_ROOT,
    WAN_ROOT,
)


DEFAULT_CHECKPOINT = CHECKPOINT_ROOT / "MaskWAM-real-ckpt"
_external_norm = DATA_ROOT / "converted_xtrainer" / "norm_all" / "qpos_mean_std_online.pkl"
_bundled_norm = (
    PROJECT_ROOT
    / "platforms"
    / "real"
    / "realdata_utils"
    / "norm_all"
    / "qpos_mean_std_online.pkl"
)
DEFAULT_NORM = _external_norm if _external_norm.exists() else _bundled_norm
DEFAULT_SAM3 = PRETRAINED_ROOT / "sam3" / "sam3.pt"
DEFAULT_WAN = WAN_ROOT
DEFAULT_VAE = WAN_ROOT / "Wan2.2_VAE.pth"
DEFAULT_EPISODE = (
    PROJECT_ROOT
    / "realdata"
    / "mask_prompt"
    / "bowls"
    / "episode_00002__grasp-the-bottle"
)
DEFAULT_ASSET_ROOT = PROJECT_ROOT / "playground" / "assets" / "default_bowls"


def _read_default_asset(camera: str) -> np.ndarray:
    from PIL import Image

    path = DEFAULT_ASSET_ROOT / f"{camera}_frame_00000.png"
    return np.asarray(Image.open(path).convert("RGB"))


def _read_default_metadata() -> tuple[str, str, str]:
    payload = json.loads((DEFAULT_ASSET_ROOT / "frame_00000.json").read_text())
    state = json.dumps(payload["state"])
    frame_index = int(payload["frame_index"])
    status = f"已加载内置截图 frame={frame_index}, state_dim={len(payload['state'])}"
    return state, str(payload["prompt"]), status


def load_frame(episode_dir: str, frame_index: int):
    frame = load_episode_frame(episode_dir, int(frame_index))
    return (
        frame.cam_high,
        frame.cam_left_wrist,
        frame.cam_right_wrist,
        json.dumps(frame.state.tolist()),
        frame.prompt,
        f"已加载 frame={frame.frame_index}, state_dim={len(frame.state)}",
    )


def select_target(event: gr.SelectData):
    x, y = (int(event.index[0]), int(event.index[1]))
    return [x, y], f"目标点: ({x}, {y})"


@lru_cache(maxsize=2)
def _segmenter(checkpoint: str, device: str) -> SAM3PointSegmenter:
    return SAM3PointSegmenter(checkpoint, device=device)


@lru_cache(maxsize=2)
def _policy(
    checkpoint: str,
    norm: str,
    wan: str,
    vae: str,
    t5_embed: str,
    device: str,
) -> OfflineXTrainerPolicy:
    return OfflineXTrainerPolicy(
        checkpoint,
        norm,
        wan_checkpoint_path=wan or None,
        wan_config_path=wan or None,
        wan_vae_path=vae or None,
        t5_embed_path=t5_embed or None,
        device=device,
    )


def _action_figure(actions: np.ndarray):
    import matplotlib.pyplot as plt

    dimensions = actions.shape[1]
    columns = 4
    rows = int(np.ceil(dimensions / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(14, 2.5 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index < dimensions:
            axis.plot(actions[:, index], linewidth=1.5)
            axis.set_title(f"action[{index}]")
            axis.grid(alpha=0.25)
        else:
            axis.axis("off")
    figure.suptitle(f"First action chunk: {actions.shape[0]} x {dimensions}")
    figure.tight_layout()
    return figure


def _write_video(path: Path, frames: np.ndarray, fps: float = 4.0) -> None:
    import av

    values = np.asarray(frames, dtype=np.uint8)
    if values.ndim != 4 or values.shape[-1] != 3 or len(values) == 0:
        raise ValueError(f"Unexpected predicted video shape: {values.shape}")
    height, width = values.shape[1:3]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=int(fps))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for frame in values:
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def run_inference(
    original_front,
    left_wrist,
    right_wrist,
    state_json,
    prompt,
    selected_point,
    checkpoint,
    norm,
    sam3_checkpoint,
    wan_path,
    vae_path,
    t5_embed_path,
    device,
    pink_bowls,
):
    required_images = {
        "cam_high": original_front,
        "cam_left_wrist": left_wrist,
        "cam_right_wrist": right_wrist,
    }
    missing = [name for name, value in required_images.items() if value is None]
    if missing:
        raise ValueError(f"缺少图像: {missing}")
    if not selected_point or len(selected_point) != 2:
        raise ValueError("请在 cam_high 图像上单击一个目标碗")
    try:
        state = np.asarray(json.loads(state_json), dtype=np.float32)
    except Exception as error:
        raise ValueError(f"state 必须是 JSON 数组: {error}") from error

    point = (int(selected_point[0]), int(selected_point[1]))
    segmenter = _segmenter(str(Path(sam3_checkpoint).expanduser()), device)
    mask = segmenter.segment(np.asarray(original_front), point)
    policy = _policy(
        str(Path(checkpoint).expanduser()),
        str(Path(norm).expanduser()),
        str(Path(wan_path).expanduser()) if wan_path else "",
        str(Path(vae_path).expanduser()) if vae_path else "",
        str(Path(t5_embed_path).expanduser()) if t5_embed_path else "",
        device,
    )
    actions, predicted_frames, _, info = policy.infer_video_and_first_chunk(
        front=np.asarray(original_front),
        left_wrist=np.asarray(left_wrist),
        right_wrist=np.asarray(right_wrist),
        state=state,
        mask=mask,
        prompt=prompt,
        use_pink_bowls_crop=bool(pink_bowls),
        tau_v=0.0,
        video_num_inference_steps=10,
    )

    output_dir = Path(os.environ.get("MASKWAM_OUTPUT_ROOT", "outputs/playground"))
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        prefix="predicted_video_",
        suffix=".mp4",
        dir=output_dir,
        delete=False,
    ) as handle:
        video_file = handle.name
    _write_video(Path(video_file), predicted_frames)
    with tempfile.NamedTemporaryFile(
        prefix="action_chunk_",
        suffix=".npy",
        dir=output_dir,
        delete=False,
    ) as handle:
        np.save(handle, actions)
        output_file = handle.name

    status = (
        f"完成：model={info.model_type}, input_mask={info.input_mask}, "
        f"chunk={actions.shape}, checkpoint={info.path}"
    )
    return (
        video_file,
        overlay_mask(original_front, mask, point),
        _action_figure(actions),
        actions,
        output_file,
        status,
    )


def build_app():
    default_front = _read_default_asset("cam_high")
    default_left = _read_default_asset("cam_left_wrist")
    default_right = _read_default_asset("cam_right_wrist")
    default_state, default_prompt, default_status = _read_default_metadata()
    with gr.Blocks(title="MaskWAM XTrainer Playground") as demo:
        gr.Markdown(
            "# MaskWAM XTrainer Playground\n"
            "加载离线 episode 或上传三路图像，在 `cam_high` 单击目标物体，"
            "通过 10 步去噪生成未来视频；action chunk 仅作为调试信息。"
        )
        selected_point = gr.State([])

        with gr.Row():
            episode_dir = gr.Textbox(value=str(DEFAULT_EPISODE), label="Episode 目录")
            frame_index = gr.Number(value=0, precision=0, label="帧号")
            load_button = gr.Button("加载 episode")
        with gr.Row():
            front = gr.Image(
                value=default_front,
                type="numpy",
                label="cam_high（在这里点击目标）",
            )
            left = gr.Image(value=default_left, type="numpy", label="cam_left_wrist")
            right = gr.Image(value=default_right, type="numpy", label="cam_right_wrist")
        state = gr.Textbox(value=default_state, label="16-D state（JSON 数组）")
        prompt = gr.Textbox(
            value=default_prompt,
            label="语言指令",
        )
        click_status = gr.Textbox(label="点击状态", interactive=False)

        with gr.Accordion("模型与外部资产", open=False):
            checkpoint = gr.Textbox(value=str(DEFAULT_CHECKPOINT), label="MaskWAM checkpoint")
            norm = gr.Textbox(value=str(DEFAULT_NORM), label="归一化 pkl")
            sam3 = gr.Textbox(value=str(DEFAULT_SAM3), label="SAM3 checkpoint")
            wan = gr.Textbox(value=str(DEFAULT_WAN), label="WAN/T5 目录")
            vae = gr.Textbox(value=str(DEFAULT_VAE), label="WAN VAE")
            t5_embed = gr.Textbox(value="", label="可选预计算 T5 embedding")
            device = gr.Dropdown(["cuda", "cpu"], value="cuda", label="设备")
            pink_bowls = gr.Checkbox(value=False, label="启用 pink_bowls 特殊 crop")

        run_button = gr.Button("分割目标并生成预测视频", variant="primary")
        with gr.Row():
            predicted_video = gr.Video(
                label="未来视频预测（10-step 去噪）",
                autoplay=True,
                loop=True,
                height=360,
                scale=2,
            )
            overlay = gr.Image(
                type="numpy",
                label="目标 mask",
                height=360,
                scale=1,
            )
        with gr.Accordion("Action 调试信息", open=False):
            action_plot = gr.Plot(label="Action 曲线")
            action_values = gr.Dataframe(label="Action 数值", interactive=False)
            action_file = gr.File(label="下载 actions.npy")
        status = gr.Textbox(value=default_status, label="状态", interactive=False)

        load_button.click(
            load_frame,
            [episode_dir, frame_index],
            [front, left, right, state, prompt, status],
        )
        front.select(select_target, None, [selected_point, click_status])
        run_button.click(
            run_inference,
            [
                front,
                left,
                right,
                state,
                prompt,
                selected_point,
                checkpoint,
                norm,
                sam3,
                wan,
                vae,
                t5_embed,
                device,
                pink_bowls,
            ],
            [predicted_video, overlay, action_plot, action_values, action_file, status],
        )
    return demo


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_app().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
