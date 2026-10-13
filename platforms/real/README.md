# Real/XTrainer workflow

This directory contains real-data conversion and offline inference utilities.
MaskWAM does not include a robot-control or action-execution interface.

## Data preparation

1. Convert raw XTrainer HDF5 recordings with
   [`realdata_utils/`](realdata_utils/).
2. Annotate target objects with [`../../maskseg/`](../../maskseg/).
3. Generate `segmentation_primary.npz` with
   `maskseg/tools/convert_sam_mask_to_segmentation.py`.

The training dataset must contain `episode.hdf5`, camera videos,
`instructions.json`, and `segmentation_primary.npz` for each episode.

## Training

From the repository root:

```bash
scripts/train_maskwam.sh xtrainer \
  dataset.hdf5_dir=/path/to/converted_xtrainer
```

The default normalization file is
`platforms/real/realdata_utils/norm_all/qpos_mean_std_online.pkl`.

## Offline inference

The supported public inference entry point is the
[`playground/`](../../playground/) application:

```bash
scripts/playground.sh --host 127.0.0.1 --port 7860
```

It reads recorded observations, predicts a future video and action chunk, and
does not send actions to a robot. Validate action conventions, limits, safety
interlocks, and emergency-stop behavior independently before integrating any
prediction into a physical robot controller.
