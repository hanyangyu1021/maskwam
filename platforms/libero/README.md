# LIBERO 数据准备与配置

本目录提供从 LIBERO 原始 demonstrations 到 MaskWAM 训练数据的完整流程：

1. `download_step1.py`：下载原始数据。
2. `regenerate_mask_step2.py`：回放 demonstrations，生成操作物体 mask。
3. `convert_to_vam_step3.py`：转换成 MaskWAM episode 格式。
4. `norm_libero.py`：生成训练所需的 state/action normalization。

以下示例使用 `libero_10`，命令均从仓库根目录执行。下载流程也适用于
`libero_spatial`、`libero_object` 和 `libero_goal`；只需相应替换
`LIBERO_SUITE`。

## 1. 安装与环境变量

先按 LIBERO 上游说明安装 LIBERO，再安装本仓库依赖：

```bash
pip install -r requirements.txt
```

本项目固定使用 `mujoco==3.3.2`。其它 MuJoCo 版本可能导致 LIBERO 的纹理、
颜色或 segmentation 渲染异常。

```bash
export MASKWAM_ROOT=/path/to/maskwam
export MASKWAM_DATA_ROOT=/path/to/data
export MASKWAM_WAN_PATH=/path/to/Wan2.2-TI2V-5B
export MASKWAM_CHECKPOINT_ROOT=/path/to/checkpoints
export LIBERO_ROOT=/path/to/LIBERO
export LIBERO_CONFIG_PATH=$MASKWAM_DATA_ROOT/libero_config
export PYTHONPATH=$MASKWAM_ROOT:$LIBERO_ROOT:$PYTHONPATH

export LIBERO_SUITE=libero_10
export LIBERO_RAW_DIR=$MASKWAM_DATA_ROOT/libero/$LIBERO_SUITE
export LIBERO_MASK_DIR=$MASKWAM_DATA_ROOT/libero_mask/$LIBERO_SUITE
export LIBERO_VAM_ROOT=$MASKWAM_DATA_ROOT/libero_vam
export LIBERO_DATASET_NAME=${LIBERO_SUITE}_robot

cd "$MASKWAM_ROOT"
```

无显示器节点使用 EGL。`MUJOCO_EGL_DEVICE_ID` 是
`CUDA_VISIBLE_DEVICES` 中的局部设备编号：

```bash
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
```

## 2. 下载 LIBERO 数据

```bash
python platforms/libero/download_step1.py \
  --download-dir "$MASKWAM_DATA_ROOT/libero" \
  --datasets "$LIBERO_SUITE" \
  --use-huggingface
```

下载后应得到：

```text
$LIBERO_RAW_DIR/*.hdf5
```

## 3. 生成 object mask

完整处理一个 suite：

```bash
python platforms/libero/regenerate_mask_step2.py \
  --libero_task_suite "$LIBERO_SUITE" \
  --libero_raw_data_dir "$LIBERO_RAW_DIR" \
  --libero_target_dir "$LIBERO_MASK_DIR"
```

脚本会在 MuJoCo 中逐条回放 demonstration，并保存：

- 两路 RGB：`agentview_rgb`、`eye_in_hand_rgb`
- 两路二值 object mask：`agentview_obj_mask`、`eye_in_hand_obj_mask`
- robot state、action 和回放结果
- `${LIBERO_SUITE}_metainfo.json`

只有回放成功的轨迹会写入输出 HDF5，因此生成数量可能少于原始 demo 数量。
如需先快速检查，可使用独立输出目录并添加：

```bash
--task_id 0 --max_demos 5
```

## 4. 转换为 MaskWAM episode

```bash
python platforms/libero/convert_to_vam_step3.py \
  --source_dir "$LIBERO_MASK_DIR" \
  --output_dir "$LIBERO_VAM_ROOT" \
  --dataset_name "$LIBERO_DATASET_NAME" \
  --meta_info "$LIBERO_MASK_DIR/${LIBERO_SUITE}_metainfo.json"
```

输出结构：

```text
$LIBERO_VAM_ROOT/
└── $LIBERO_DATASET_NAME/
    └── task_<id>__<instruction>/
        └── episode_<id>/
            ├── episode.hdf5
            ├── cam_high.mp4
            ├── cam_left_wrist.mp4
            ├── cam_high_mask.mp4
            ├── cam_left_wrist_mask.mp4
            └── instructions.json
```

mask 视频使用 RGB 表示：红色为操作物体，黑色为背景。当前 LIBERO 转换
不生成蓝色 robot mask。

## 5. 生成 normalization

对转换后的全部 episode 计算统计量：

```bash
python platforms/libero/norm_libero.py \
  --processed_dir "$LIBERO_VAM_ROOT/$LIBERO_DATASET_NAME" \
  --output_dir "$LIBERO_VAM_ROOT"
```

输出文件为：

```text
$LIBERO_VAM_ROOT/qpos_mean_std_online.pkl
```

训练和后续 eval 必须使用同一份 normalization。评测已有 checkpoint 时，
应使用该 checkpoint 训练时的 normalization，不要重新计算。

## 6. 配置数据集

如需直接使用发布版训练数据，可跳过第 2–5 节并下载完整数据集：

```bash
export MODELSCOPE_ENDPOINT=https://www.modelscope.ai
ms download hanyangyu1021/MaskWAM-libero \
  --repo-type dataset \
  --local-dir "$MASKWAM_DATA_ROOT"
```

修改 `configs/dataset/libero.yaml` 中这三项，使其与转换结果一致：

```yaml
hdf5_dir: ${data_root}/libero_vam

hdf5_dataset:
  _target_: data.hdf5_dataset_libero_T.HDF5VLADataset
  sample_weights:
    libero_10_robot: 1.0

mean_std_path: ${data_root}/libero_vam/qpos_mean_std_online.pkl
```

`sample_weights` 的 key 必须与 `--dataset_name` 完全一致。多个 suite 可以放在
同一个 `LIBERO_VAM_ROOT` 下，并通过权重混合：

```yaml
sample_weights:
  libero_spatial_robot: 1.0
  libero_object_robot: 1.0
  libero_goal_robot: 1.0
  libero_10_robot: 1.0
```

同时确认顶层环境变量：

```bash
export MASKWAM_DATA_ROOT=/path/to/data
```

## 7. 验证 dataloader

```bash
python dataset_T_sim.py \
  dataset=libero \
  +num_vis=1 \
  +vis_dir="$LIBERO_VAM_ROOT/visualizations"
```

该命令会经过正式 dataloader 读取一个 batch，并保存 T-shape RGB/mask
可视化。正常 tensor 形状为：

```text
current_image:       [B, 3, 384, 320]
future_images:       [B, 8, 3, 384, 320]
current_seg_mask:    [B, 3, 384, 320]
future_seg_masks:    [B, 8, 3, 384, 320]
action:              [B, 48, 16]
```

## 8. 训练

完成数据配置后：

```bash
scripts/train_maskwam.sh libero
```

batch size、训练步数、mask 输入和 loss 权重等训练参数位于
`configs/train_maskwam.yaml`，也可通过 Hydra 命令行覆盖：

```bash
scripts/train_maskwam.sh libero \
  training.batch_size=2 \
  training.max_training_steps=200000
```

## 9. LIBERO eval 与录像

下载发布的 checkpoint 和训练时使用的 normalization：

```bash
pip install -U modelscope-hub

ms download hanyangyu1021/MaskWAM-libero-ckpt \
  --local-dir "$MASKWAM_CHECKPOINT_ROOT/MaskWAM-libero-ckpt"

ms download hanyangyu1021/MaskWAM-libero \
  libero_vam/qpos_mean_std_online.pkl \
  --repo-type dataset \
  --local-dir "$MASKWAM_DATA_ROOT"
```

使用默认下载路径评测：

```bash
scripts/eval_libero.sh \
  --task_suite "$LIBERO_SUITE" \
  --num_trials 50 \
  --record_video \
  --video_dir eval_videos \
  --results_dir eval_results
```

eval 使用与数据生成相同的 EGL 配置，并必须使用 checkpoint 对应的
normalization 文件。发布 checkpoint 的 `input_mask=false`，eval 会根据
checkpoint config 自动使用零输入 mask，无需创建 segmentation 环境。
