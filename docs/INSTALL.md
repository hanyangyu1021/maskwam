# Installation

> Note: code is being prepared for release. The steps below describe the
> intended setup and may change before the official release.

## 1. Clone the repository

```bash
git clone https://github.com/hanyangyu1021/maskwam.git
cd maskwam
```

## 2. Create the environment

```bash
conda create -n maskwam python=3.10 -y
conda activate maskwam
```

## 3. Install PyTorch

Install the build matching your CUDA version, e.g. for CUDA 11.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 4. Install MaskWAM

```bash
pip install -e .
# or: pip install -r requirements.txt
```

## 5. (Optional) Benchmark dependencies

- LIBERO: follow the official LIBERO installation guide.
- RoboTwin 2.0: follow the official RoboTwin 2.0 installation guide.

## Verify

```bash
python -c "import maskwam; print(maskwam.__version__)"
```
