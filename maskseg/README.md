[中文](README_zh.md)

# MaskWAM-Seg

MaskWAM-Seg is the video annotation and segmentation module of MaskWAM. It targets robotic manipulation data and supports:

- Qt-based point annotation
- SAM3 video object segmentation
- Multiple objects, positive/negative points, and bidirectional tracking
- Multi-round mask review and correction
- Video-level and clip-level language annotation
- Flask-based multi-user task distribution

The repository includes `xtrainer_test/bowls` as a minimal example dataset. The segmentation model is **SAM3**.

## Workflow

The segmentation workflow is:

1. Load an original video in the client.
2. Add positive and negative point prompts on keyframes.
3. Save the point annotation.
4. Run `parse_sam.py` to generate masks and an overlay video.
5. Review the overlay video in the next annotation round.
6. Repeat until the sample is finished or marked as problematic/hard.

Language and segmentation annotation use the same client but have separate task pools and outputs.

## Select the Dataset to Annotate

The videos to segment are specified in `asserts/demo_data/dataset.json`. `user_config` does not select the dataset.

`dataset.json` maps video paths to optional language pre-annotations. Use an empty string when no pre-annotation exists:

```json
{
  "/path/to/your_dataset/video_001.mp4": "",
  "/path/to/your_dataset/video_002.mp4": ""
}
```

Paths can be absolute or relative to the `MaskWAM-Seg` directory. The bundled `dataset.json` points to `xtrainer_test/bowls`; new users should replace it with their own video list. Put one username per line in `asserts/demo_data/user_list.txt`, then generate both task pools:

```bash
python tools/generate_annotation_pool.py --input asserts/demo_data/dataset.json --output asserts/demo_data --user-list asserts/demo_data/user_list.txt
```

This command overwrites the current task-pool state. Back up annotations that still matter before running it.

Data flow:

```text
dataset.json
    │ generate_annotation_pool.py
    ▼
no_annotation_sam.json      videos waiting to be claimed
    │ requested by a client
    ▼
has_annotation_sam.json     videos currently checked out
    │ Save and Load
    ├──► human_anno_sam/.../sam/*.npz
    └──► user_config/sam/<username>.txt
```

`config/config.yaml` tells the server where these pools, the user list, and `user_config` are located.

`user_config` is runtime history generated automatically by the server:

- `<username>.txt`: videos saved in round 0
- `<username>_1.txt`, `_2.txt`, `_3.txt`: review-round history
- `<username>_finish.txt`: finished samples
- `<username>_hard.txt`: hard samples
- `<username>_question.txt`: problematic samples
- `user_config/lang/`: language-annotation history

An empty history file is normal. For example, requesting round-3 statistics can create an empty `<username>_3.txt`. These files normally should not be edited manually.

## Installation and Startup

Install dependencies from the MaskWAM repository root:

### 1. Install

Python 3.8+ is required. SAM inference also requires a CUDA-compatible PyTorch environment.

```bash
pip install -r requirements.txt
git clone https://github.com/facebookresearch/sam3.git third_party/sam3
pip install -e third_party/sam3

# Accept the facebook/sam3 license on Hugging Face before downloading.
hf auth login
hf download facebook/sam3 --local-dir pretrained_models/sam3
cd maskseg
```

The expected checkpoint path is
`pretrained_models/sam3/sam3.pt` from the repository root. The SAM3 repository is gated, so request
access and accept its license on Hugging Face before running the download.

### 2. Configure

Edit `config/config.yaml`:

```yaml
project_name: "MaskWAM-Seg"

sam:
  checkpoint: "../pretrained_models/sam3/sam3.pt"
  device: "cuda:0"

server:
  root_dir: "."
  video_dir: ""
```

When commands are run from `MaskWAM-Seg`, `root_dir: "."` points to the module automatically and does not need to change after the directory is moved. The bundled video paths are relative to that directory, so `video_dir` can remain empty.

### 3. Start the Server

```bash
python server/server.py --port 5000
```

Health endpoint:

```text
http://127.0.0.1:5000/health
```

### 4. Start the Client

Open another terminal in the `maskseg/` directory:

```bash
python client/client.py
```

Example login:

- Username: `root`
- Server: `127.0.0.1`
- Port: `5000`
- Mode: segmentation annotation
- Review round: `0`

The client loads the two videos in `xtrainer_test/bowls`.

## Segmentation Annotation

Mouse controls:

- Left click: positive point inside the target.
- Right click: negative point in an area to exclude.

Keyboard shortcuts:

- `A` / `D`: previous / next frame
- `F`: play / pause
- `K`: next object
- `W`: mark contact frame
- `S`: mark object-separation frame
- `L`: confirm object-to-segment binding
- `Backspace`: remove the current keyframe marker
- `J`: save and load the next video

```text
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │
├───┼───┼───┼───┼───┼───╆━━━╅━━━╅━━━┪
│ A │ S │ D │ F │ G │ H ┃ J ┃ K ┃ L ┃
└───┴───┴───┴───┴───┴───┺━━━┻━━━┻━━━┛
                           Save  Obj  Bind

Mouse: left click = positive point    right click = negative point
```

`J`, `K`, and `L` are adjacent: save, switch object, and confirm binding. After clicking point prompts with the mouse, these frequent actions can be completed with one hand and minimal movement.

Minimal procedure:

1. Find a clear frame.
2. Add positive points and, if needed, negative points.
3. Press `K` when annotating another object.
4. Press `L` to confirm object binding.
5. Press `J` or click **Save and Load**.

Important:

- Press `L` before saving, even for a single object.
- Use `Bidirectional Mode` when points are added on multiple frames.
- Closing the client returns the current task to the pool; unsaved points are discarded.

## Generate SAM Masks

After round-0 annotation:

```bash
python tools/parse_sam.py --username root --time 0
```

To override the configured checkpoint, pass
`--sam3-ckpt /path/to/sam3.pt`.

Outputs:

- Point annotations: `asserts/demo_data/human_anno_sam/0/sam/`
- Masks: `asserts/demo_data/human_anno_sam/0/sam_mask/`
- Overlay videos: `asserts/demo_data/human_anno_sam/0/sam_video/`

Mask files contain a boolean `masks` array with shape:

```text
(number of objects, number of frames, 1, height, width)
```

## Review Rounds

After round 0:

1. Select review round `1` in the client.
2. Inspect the generated overlay video.
3. Add corrective positive or negative points.
4. Press `L` and save.
5. Run:

```bash
python tools/parse_sam.py --username root --time 1
```

Use `--time 2` and `--time 3` for later rounds. Reviewers can mark samples as finished, problematic, or hard.

## Language Annotation

Select language annotation when launching the client.

- `Enter`: edit the video-level description
- `W`: mark clip start
- `S`: mark clip end and enter its description
- `L`: edit the current clip
- `Backspace`: delete the current clip
- `A` / `D`: previous / next frame
- `F`: play / pause

A video-level description is required before saving.

## Project Structure

```text
MaskWAM/
├── maskseg/
    ├── client/                    # Qt annotation client
    ├── server/                    # Flask annotation server
    ├── tools/                     # Task-pool, SAM, and conversion tools
    ├── config/config.yaml         # Model and path configuration
    ├── asserts/demo_data/         # Task pools and annotation outputs
    ├── user_config/               # Per-user annotation history
    └── xtrainer_test/bowls/       # Minimal example dataset
└── third_party/sam3/              # External SAM3 checkout (not tracked)
```

## Troubleshooting

- Start `client/client.py` from the `maskseg/` directory so it can find
  `config/config.yaml`.
- Usernames must exist in `asserts/demo_data/user_list.txt`.
- If `parse_sam.py` finds no work, confirm that the client saved an annotation and inspect `user_config/sam/<username>.txt`.
- Download the SAM3 checkpoint into `../pretrained_models/sam3`, or supply a different path with `--sam3-ckpt`.

## Acknowledgements

This project builds upon the open-source code of [RoboInter-Tools](https://github.com/InternRobotics/RoboInter/tree/main/RoboInterTools). We thank the RoboInter team for making this excellent annotation toolkit publicly available.

## License

The license follows the upstream project.
