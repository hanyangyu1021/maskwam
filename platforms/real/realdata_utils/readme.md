# xtrainer 数据处理工具

把 **xtrainer（真机采集）** 原始 HDF5 转换为统一的「episode 文件夹」格式，供下游 `MaskWAM-Seg`（mask 标注、segmentation 构建等）继续处理。该工具位于仓库的：

```text
platforms/real/realdata_utils/
```

本目录只包含 XTrainer 转换脚本；转换产物应写入仓库外的数据目录。

---

## 统一的 episode 文件夹格式

```
converted_xtrainer/
  <task>/
    episode_00000__<task-slug>/
      episode.hdf5
      cam_high.mp4
      cam_left_wrist.mp4
      cam_right_wrist.mp4
      instructions.json     # {"seen": [...]}
      meta.json
```

`episode.hdf5` 关键字段：

| 字段 | 说明 |
|---|---|
| `observations/state_eep` | 末端位姿（eepos） |
| `observations/state_qpos` | 关节角（qpos） |
| `actions_eep` | 动作序列（原样保存） |

xtrainer 数据没有 `segmentation_primary.npz`；该文件由下游 MaskWAM-Seg 标注后再生成。

---

## 流水线

1. **转格式** — `convert_xtrainer_to_episode_folders.py` 把 xtrainer HDF5 转为 episode 文件夹
2. **标注 mask** — 进入 `maskseg/`，用其 server/client 标注 SAM mask
3. **合成最终数据** — 运行 `maskseg/tools/convert_sam_mask_to_segmentation.py`

---

## 转换脚本

### `convert_xtrainer_to_episode_folders.py`

唯一正式转换入口。同时写入 `state_eep` + `state_qpos`，默认导出三路相机。

```bash
python convert_xtrainer_to_episode_folders.py \
    --input-dir /path/to/xtrainer/hdf5/bottles \
    --output-dir converted_xtrainer/bottles \
    --camera-views cam_high cam_left_wrist cam_right_wrist
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--input-dir` | **必填**，含 `*.hdf5` 的目录 |
| `--output-dir` | **必填**，episode 文件夹输出目录 |
| `--camera-views` | 默认 `cam_high cam_left_wrist cam_right_wrist` |
| `--fps` | 默认 `30.0`（与 xtrainer 真机采集一致） |
| `--skip-existing` | 已有 episode 文件夹整体跳过 |
| `--max-episodes N` | 只转前 N 条，方便调试 |
| `--ffmpeg-path` | 指定 ffmpeg；找不到时回退 OpenCV |

### `convert_all_context_xtrainer.sh`

按 task 子目录批量调用上面的转换器：

```bash
bash convert_all_context_xtrainer.sh \
    /path/to/xtrainer2/context/hdf5 \
    converted_xtrainer \
    --skip-existing
```

可用环境变量覆盖 Python / 转换脚本路径：`PYTHON_BIN`、`CONVERTER`。

---

## 目录速查

| 文件 / 目录 | 作用 |
|---|---|
| `convert_xtrainer_to_episode_folders.py` | xtrainer HDF5 → episode 文件夹 |
| `convert_all_context_xtrainer.sh` | 批量按 task 转换 |
| `converted_xtrainer/` | 转换产物输出根目录 |
| `maskseg/` | MaskWAM-Seg：下游 mask 标注 + segmentation 合成工具 |
