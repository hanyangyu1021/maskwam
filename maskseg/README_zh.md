[English](README.md)

# MaskWAM-Seg

MaskWAM-Seg 是 MaskWAM 项目中的视频标注与分割模块，面向机器人操作数据，支持：

- 基于 Qt 的人工点击标注
- SAM3 视频目标分割
- 多物体、正负样本点和双向跟踪
- 多轮 Mask 质检与修正
- 视频级和片段级语言标注
- Flask 多用户任务分发

仓库只附带 `xtrainer_test/bowls` 作为最小示例数据。分割模型使用 **SAM3**。

## 工作流程

分割标注采用以下循环：

1. 客户端加载原始视频。
2. 标注员在关键帧上添加正、负样本点。
3. 客户端保存点击配置。
4. `parse_sam.py` 默认使用 SAM3 生成 Mask 和叠加视频。
5. 标注员在下一轮检查叠加视频并修正错误。
6. 重复处理，直到标记为完成、问题样本或困难样本。

语言标注与分割标注使用同一个客户端，但任务池和输出文件彼此独立。

## 指定要标注的数据集

要分割哪些视频，由 `asserts/demo_data/dataset.json` 指定；`user_config` 不负责配置数据集。

### 1. 编辑数据清单

`dataset.json` 使用“视频路径到语言预标注路径”的映射。仅做分割、没有语言预标注时，值填空字符串：

```json
{
  "/path/to/your_dataset/video_001.mp4": "",
  "/path/to/your_dataset/video_002.mp4": ""
}
```

视频路径可以是绝对路径，也可以是相对于 `MaskWAM-Seg` 文件夹的路径。使用相对路径时，服务器和客户端都应从 `MaskWAM-Seg` 文件夹启动。仓库自带的 `dataset.json` 已指向 `xtrainer_test/bowls`，新用户应将其替换为自己的视频清单。

### 2. 配置标注用户

编辑 `asserts/demo_data/user_list.txt`，每行一个用户名：

```text
root
```

### 3. 生成任务池

下面的命令会读取 `dataset.json`，并同时生成语言标注和分割标注任务池：

```bash
python tools/generate_annotation_pool.py --input asserts/demo_data/dataset.json --output asserts/demo_data --user-list asserts/demo_data/user_list.txt
```

重新生成任务池会覆盖当前任务状态，请先备份仍需保留的标注。

### 4. 数据如何被加载

```text
dataset.json
    │ generate_annotation_pool.py
    ▼
no_annotation_sam.json     待领取的视频
    │ 客户端向服务器请求视频
    ▼
has_annotation_sam.json    已被客户端领取、正在标注的视频
    │ Save and Load
    ├──► human_anno_sam/.../sam/*.npz
    └──► user_config/sam/<用户名>.txt
```

配置文件 `config/config.yaml` 决定服务器读取哪几个任务池文件：

```yaml
server:
  root_dir: "."
  no_annotation_sam: "asserts/demo_data/no_annotation_sam.json"
  has_annotation_sam: "asserts/demo_data/has_annotation_sam.json"
  user_list_file: "asserts/demo_data/user_list.txt"
  user_history_dir: "user_config"
```

### `user_config` 是什么

`user_config` 是服务器自动生成的运行历史，不是输入数据集：

- `user_config/sam/<用户名>.txt`：第 0 轮已保存的视频
- `<用户名>_1.txt`、`<用户名>_2.txt`、`<用户名>_3.txt`：各质检轮已保存的视频
- `<用户名>_finish.txt`：已经完成的样本
- `<用户名>_hard.txt`：困难样本
- `<用户名>_question.txt`：问题样本
- `user_config/lang/`：语言标注历史

这些文件可能为空。例如客户端查询第 3 轮统计时，服务器会自动创建空的 `<用户名>_3.txt`，这不代表已经完成过第 3 轮标注。通常不需要手动编辑 `user_config`。

## 安装与启动

先从 MaskWAM 仓库根目录安装依赖：

### 1. 安装依赖

要求 Python 3.8+，运行 SAM 推理需要支持 CUDA 的 PyTorch 环境。

```bash
pip install -r requirements.txt
git clone https://github.com/facebookresearch/sam3.git third_party/sam3
pip install -e third_party/sam3

# 下载前需先在 Hugging Face 接受 facebook/sam3 的许可协议
hf auth login
hf download facebook/sam3 --local-dir pretrained_models/sam3
cd maskseg
```

从仓库根目录看，默认 checkpoint 路径为 `pretrained_models/sam3/sam3.pt`。SAM3 仓库需要
授权访问，请先在 Hugging Face 申请权限并接受许可协议，再执行下载命令。

### 2. 检查配置

编辑 `config/config.yaml`，重点确认：

```yaml
project_name: "MaskWAM-Seg"

sam:
  checkpoint: "../pretrained_models/sam3/sam3.pt"
  device: "cuda:0"

server:
  root_dir: "."
  video_dir: ""
```

从 `MaskWAM-Seg` 文件夹运行命令时，`root_dir: "."` 会自动指向当前模块，无需在移动目录后修改绝对路径。示例视频路径相对于该文件夹，因此 `video_dir` 保持空字符串即可。

### 3. 启动服务器

```bash
python server/server.py --port 5000
```

健康检查地址：

```text
http://127.0.0.1:5000/health
```

### 4. 启动客户端

另开一个终端，并仍从项目根目录启动：

```bash
python client/client.py
```

示例配置：

- 用户名：`root`
- 服务器地址：`127.0.0.1`
- 端口：`5000`
- 标注模式：`分割标注`
- 质检次数：`0`

客户端会从 `xtrainer_test/bowls` 加载两段示例视频。

## 分割标注操作

### 鼠标操作

- 左键：添加正样本点，表示目标物体内部。
- 右键：添加负样本点，表示需要排除的区域。

### 常用快捷键

- `A` / `D`：上一帧 / 下一帧
- `F`：播放或暂停
- `K`：切换到下一个物体
- `W`：标记接触帧
- `S`：标记物体分段帧
- `L`：确认物体 ID 与视频段的绑定
- `Backspace`：删除当前关键帧标记
- `J`：保存并加载下一条视频

界面按钮还支持撤销最后一个点、删除当前物体标注和清空全部标注。

```text
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │
├───┼───┼───┼───┼───┼───╆━━━╅━━━╅━━━┪
│ A │ S │ D │ F │ G │ H ┃ J ┃ K ┃ L ┃
└───┴───┴───┴───┴───┴───┺━━━┻━━━┻━━━┛
                            保存  物体  绑定

鼠标：左键 = 正样本点    右键 = 负样本点
```

`J`、`K`、`L` 在键盘上连续排列，分别对应保存、切换物体和确认绑定。完成鼠标点选后，右手无需大范围移动即可单手完成这三个高频操作。

### 最小标注步骤

1. 用 `A`、`D` 或进度条找到清晰关键帧。
2. 左键点击目标物体，必要时用右键排除背景。
3. 多物体任务可按 `K` 切换物体。
4. 按 `L` 确认物体绑定。
5. 按 `J`，或点击 **Save and Load** 保存。

注意：

- 保存前必须按一次 `L`，即使视频里只有一个物体。
- 在多个帧上添加了点击点时，必须使用 `Bidirectional Mode`。
- 关闭窗口会将当前任务退回任务池；未保存的点击不会保留。

## 生成 SAM Mask

完成第 0 轮点击标注后运行：

```bash
python tools/parse_sam.py --username root --time 0
```

如需覆盖配置中的 checkpoint，可传入
`--sam3-ckpt /path/to/sam3.pt`。

输出位置：

- 点击配置：`asserts/demo_data/human_anno_sam/0/sam/`
- Mask：`asserts/demo_data/human_anno_sam/0/sam_mask/`
- 叠加视频：`asserts/demo_data/human_anno_sam/0/sam_video/`

Mask 文件包含：

```python
{
    "masks": np.ndarray
}
```

数组形状为 `(物体数, 帧数, 1, 高, 宽)`，类型为 `bool`。

## 多轮质检

第 0 轮处理完成后，生成的叠加视频会进入下一轮任务池：

1. 客户端选择质检次数 `1`。
2. 查看 Mask 叠加结果。
3. 在错误帧补充正、负样本点。
4. 按 `L` 后保存。
5. 运行：

```bash
python tools/parse_sam.py --username root --time 1
```

后续轮次使用 `--time 2`、`--time 3`。质检时可以将样本标记为 Finish、问题样本或困难样本。

## 语言标注

客户端启动时选择 `语言标注`。

- `Enter`：编辑视频级描述
- `W`：标记片段开始帧
- `S`：标记片段结束帧并填写片段描述
- `L`：编辑当前片段描述
- `Backspace`：删除当前片段
- `A` / `D`：上一帧 / 下一帧
- `F`：播放或暂停

保存前必须填写视频级描述。

## 项目结构

```text
MaskWAM/
├── maskseg/
    ├── client/                    # Qt 标注客户端
    ├── server/                    # Flask 标注服务器
    ├── tools/                     # 任务池、SAM 推理和格式转换工具
    ├── config/config.yaml         # 模型与路径配置
    ├── asserts/demo_data/         # 任务池和标注输出
    ├── user_config/               # 用户标注历史
    └── xtrainer_test/bowls/       # 最小示例数据
└── third_party/sam3/              # 外部 SAM3 源码（不纳入 Git）
```

## 常见问题

### 客户端找不到配置

请从项目根目录运行：

```bash
python client/client.py
```

不要先进入 `client/` 再运行。

### 用户名不可用

用户名必须存在于 `asserts/demo_data/user_list.txt`。

### `parse_sam.py` 没有处理视频

先确认客户端已经保存标注，并检查：

```text
user_config/sam/<用户名>.txt
```

### SAM3 是否是默认模型

是。`tools/parse_sam.py` 使用 SAM3；请将 checkpoint 下载到
`../pretrained_models/sam3`，或通过 `--sam3-ckpt` 指定其他路径。

## 致谢

本项目基于 [RoboInter-Tools](https://github.com/InternRobotics/RoboInter/tree/main/RoboInterTools) 的开源代码开发。感谢 RoboInter 团队公开这套优秀的标注工具。

## License

许可证与上游项目保持一致。
