<div align="center">

# MaskWAM: Unifying Mask Prompting and Prediction for World-Action Models

<a href="#"><img src="https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg"></a> &nbsp;
<a href="https://hanyangyu1021.github.io/maskwam.github.io/"><img src="https://img.shields.io/badge/Project-MaskWAM-blue"></a> &nbsp;
<a href="#"><img src="https://img.shields.io/badge/Code-coming%20soon-lightgrey"></a> &nbsp;
<a href="#"><img src="https://img.shields.io/badge/License-MIT-green.svg"></a>

[Hanyang Yu](https://hanyangyu1021.github.io/)<sup>1,*,†</sup>,
[Haitao Lin](https://hetolin.github.io/)<sup>2,3,*</sup>,
[Jingbo Zhang](https://eckertzhang.github.io/)<sup>2,3</sup>,
[Wenyao Zhang](https://zhangwenyao1.github.io/)<sup>2,3</sup>,
[Chenghao Gu](https://chenghaogu.github.io/)<sup>4,†</sup>,
[Heng Li](https://hengli.me/)<sup>1</sup>,
[Ping Tan](https://ece.hkust.edu.hk/pingtan)<sup>1,†</sup>

<sup>1</sup>The Hong Kong University of Science and Technology  
<sup>2</sup>Tencent Robotics X, <sup>3</sup>Futian Laboratory, <sup>4</sup>Tsinghua University

<sup>*</sup>Equal contribution &nbsp;&nbsp; <sup>†</sup>Corresponding author

<img src="assets/teaser.png" width="95%">

</div>

This repository is the official implementation of **MaskWAM: Unifying Mask Prompting and Prediction for World-Action Models**.

**MaskWAM** is an object-centric **World-Action Model (WAM)** that unifies **mask prompting** and **mask prediction** for robotic manipulation. It uses masks both as explicit spatial prompts and as future prediction targets, enabling stronger target grounding, better distractor robustness, and improved generalization in both language-clear and language-ambiguous manipulation tasks.

---

## News

- **[2026.xx.xx] Code is coming soon.** We are finalizing training, inference, and evaluation code.
- **[2026.xx.xx] Project page released.** Please visit the [project website](https://hanyangyu1021.github.io/maskwam.github.io/) for videos and visualizations.

---

## Highlights

- **Mask prompting for spatial grounding.**  
  A first-frame target mask provides an explicit spatial anchor, reducing ambiguity when multiple similar objects appear in the scene.

- **Future mask prediction for object-centric learning.**  
  MaskWAM predicts future masks together with future RGB frames, encouraging the model to focus on task-relevant regions instead of distractors or backgrounds.

- **Unified RGB-mask-action modeling.**  
  RGB latents, mask latents, and action tokens are jointly modeled in a unified WAM framework.

- **Strong performance in simulation and real-world tasks.**  
  MaskWAM is evaluated on LIBERO, RoboTwin 2.0, and real-robot manipulation tasks.

---

## Overview

World-Action Models are promising for robotic control because they jointly model future visual dynamics and robot actions. However, existing WAMs still face important spatial bottlenecks:

1. **Text-only conditioning is ambiguous** in cluttered scenes, especially when multiple visually similar objects are present.
2. **RGB-only future prediction is weakly grounded**, so the model may focus on task-irrelevant backgrounds or distractors.
3. **Visual prompting and future prediction are often separated**, limiting the ability of spatial prompts to directly shape action generation.

MaskWAM addresses these issues by integrating object masks into WAMs in two complementary ways:

- **Mask as input:** an optional first-frame mask prompt specifies the target object.
- **Mask as output:** future mask prediction provides object-centric supervision.
- **Mask as part of world-action modeling:** RGB, masks, and actions are jointly learned in one unified architecture.

---

## Method

<div align="center">
<img src="assets/overall.png" width="95%">
</div>

MaskWAM jointly predicts future RGB frames, future masks, and action chunks under a unified world-action modeling framework.

### Architecture

Given current observations, language instructions, robot states, and optional first-frame mask prompts, MaskWAM encodes RGB and mask observations into latent representations. The noisy RGB and mask latents are channel-concatenated and denoised by a unified DiT-style backbone. The model jointly optimizes future RGB prediction, future mask prediction, and action generation.

### Key components

- **Unified RGB-mask latent modeling:** RGB videos and mask videos are encoded into a shared latent space and concatenated before entering the video diffusion backbone.
- **Future mask prediction:** the model predicts object-centric future masks together with RGB futures, providing stronger supervision than RGB reconstruction alone.
- **Action chunk generation:** action chunks are generated through a transformer-based action branch that jointly attends to visual and mask representations.
- **Optional first-frame visual prompting:** at inference time, MaskWAM can use a first-frame target mask to resolve spatial ambiguity.
- **Efficient inference:** during deployment, full future videos do not need to be decoded; actions can be generated through partial denoising and cached representations.

---

## Experiments

We evaluate MaskWAM on simulation benchmarks and real-world robotic manipulation tasks.

---

### LIBERO Benchmark

<div align="center">

| Method | Type | Spatial | Object | Goal | Long | Avg |
|---|---:|---:|---:|---:|---:|---:|
| WorldVLA | VLA | 87.6 | 96.2 | 83.4 | 60.0 | 81.8 |
| GR00T-N1 | VLA | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| π₀ | VLA | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| π₀.₅ | VLA | 98.6 | 98.2 | 98.0 | 92.4 | 96.8 |
| Motus | WAM | 96.8 | 99.8 | 96.6 | **97.6** | 97.7 |
| FastWAM | WAM | 98.2 | **100.0** | 97.0 | 95.2 | 97.6 |
| Ours (RGB-only) | WAM | 96.8 | 99.6 | 97.0 | 95.8 | 97.3 |
| Ours (Mask-only) | WAM | 97.2 | 99.8 | 97.4 | 96.0 | 97.6 |
| **Ours** | WAM | **98.8** | **100.0** | **98.2** | 96.4 | **98.4** |

</div>

MaskWAM achieves **98.4%** average success rate on LIBERO. The improvement over the RGB-only variant shows that future mask prediction benefits standard text-conditioned WAMs even when no visual prompt is used during deployment.

---

### RoboTwin 2.0 Benchmark

<div align="center">

| Method | Hammer | Bell | Card | Burger | Stand | Shoe | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| π₀ | 68 | 72 | 81 | 79 | 63 | 74 | 72.8 |
| FastWAM | 83 | 87 | 92 | 94 | 80 | 90 | 87.7 |
| Ours (RGB-only) | 82 | 87 | 91 | 93 | 79 | 92 | 87.3 |
| Ours (Mask-only) | 85 | 90 | 93 | 93 | 81 | 91 | 88.8 |
| **Ours** | **88** | **93** | **95** | **97** | **85** | **95** | **92.2** |

</div>

MaskWAM achieves **92.2%** average success rate across six randomized RoboTwin 2.0 tasks, outperforming both π₀ and FastWAM.

---

### Real-Robot Language-Clear Tasks

We evaluate four language-clear real-world manipulation tasks:

1. Stacking three nested bowls.
2. Hanging a mug onto a designated peg.
3. Opening a drawer, picking and placing a pen, and closing the drawer.
4. Folding a towel.

<div align="center">

| Method | Type | Task 1 | Task 2 | Task 3 | Task 4 | Avg |
|---|---|---:|---:|---:|---:|---:|
| π₀ | VLA | 57 | 54 | 54 | 58 | 55.8 |
| π₀.₅ | VLA | 83 | 55 | 74 | 77 | 72.3 |
| FastWAM | WAM | 88 | 76 | 77 | 75 | 79.0 |
| Ours (RGB-only) | WAM | 86 | 77 | 76 | 78 | 79.3 |
| **Ours** | WAM | **91** | **82** | **81** | **83** | **84.3** |

</div>

---

### Real-Robot Language-Ambiguous Tasks

Language-ambiguous tasks require precise target grounding among visually similar objects:

1. Grasping a target bowl among multiple bowls.
2. Picking up a specified cup.
3. Picking and placing a specified bottle.
4. Grasping a specified cosmetic item from densely arranged similar objects.

<div align="center">
<img src="assets/shadow_bar_chart.png" width="90%">
</div>

MaskWAM uses first-frame mask prompts to resolve target ambiguity and remains robust across in-distribution scenes, distractors, novel instances, and lighting changes.

---

## Visualization

<div align="center">
<img src="assets/mask_more.png" width="95%">
</div>

MaskWAM jointly predicts future RGB frames and future object masks. For visualization only, we decode full future sequences offline. During real-world deployment, MaskWAM can generate actions with partial denoising and does not need to decode full future videos.

---

## Real-Robot Videos

More real-robot videos are available on the [project page](https://hanyangyu1021.github.io/maskwam.github.io/).

The project page includes demonstrations for:

- Language-clear tasks: stacking bowls, hanging a mug, drawer manipulation, and towel folding.
- Language-ambiguous tasks: target bowl, target cup, target bottle, and target cosmetics.
- Generalization settings: in-distribution, distractors, novel instances, and lighting changes.

---

## Getting Started

The code is currently being prepared for release.

We plan to release:

- Training code
- Inference code
- Model checkpoints
- Data preparation tools
- LIBERO evaluation scripts
- RoboTwin 2.0 evaluation scripts
- Real-robot deployment examples

---

## TODO

- [ ] Release training code
- [ ] Release inference code
- [ ] Release model checkpoints
- [ ] Release data preparation tools
- [ ] Release LIBERO evaluation scripts
- [ ] Release RoboTwin 2.0 evaluation scripts
- [ ] Add installation instructions
- [ ] Add real-robot deployment instructions

---

## Citation

If you find MaskWAM useful in your research, please consider citing:

```bibtex
@article{yu2026maskwam,
  title   = {MaskWAM: Unifying Mask Prompting and Prediction for World-Action Models},
  author  = {Hanyang Yu and Haitao Lin and Jingbo Zhang and Wenyao Zhang and Chenghao Gu and Heng Li and Ping Tan},
  journal = {},
  year    = {2026}
}
