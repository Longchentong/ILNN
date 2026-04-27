# Intrinsic Lorentz Neural Network

Official PyTorch code for **Intrinsic Lorentz Neural Network (ILNN)**.

ILNN is a fully intrinsic Lorentz hyperbolic neural network built around:

- **PLFC**: a point-to-hyperplane Lorentz fully connected layer.
- **GyroLBN**: Lorentz batch normalization with gyro-centering and gyro-scaling.
- **Log-radius patch concatenation** for Lorentz convolution.
- **Gyro-additive bias**, Lorentz activation, and Lorentz dropout.

This cleaned repository contains the image-classification and genomics
experiments used by the paper.

## Repository Layout

```text
lib/                         Shared Lorentz, GyroBN, Geoopt, and model modules
experiments/vision/          CIFAR-10/100 ResNet-18 experiments
experiments/genomics/        TEB and GUE sequence-classification experiments
scripts/                     Multi-seed reproduction scripts
tools/hyperbolicity/         Dataset hyperbolicity utilities
data/                        Local datasets, not committed
outputs/                     Checkpoints and logs, not committed
```

## Installation

The original experiments used Python 3.8, PyTorch 1.13, CUDA 11.7, and A100
GPUs.

```bash
conda env create -f environment.yml
conda activate ilnn
```

Alternatively:

```bash
pip install -r requirements.txt
```

Install a PyTorch build matching your CUDA version if the pinned conda
environment is not suitable for your machine.

## Data

CIFAR-10/100 are downloaded automatically by `torchvision`.

Genomics datasets are expected under `data/`; see [data/README.md](data/README.md)
for the exact layout. The repository does not commit GUE/TEB files or generated
checkpoints.

## Quick Smoke Test

```bash
python tests/smoke_test.py
```

This only checks that the core Lorentz modules import and run a small forward
pass. It does not train a model.

## Reproduce Vision Experiments

Single run:

```bash
python experiments/vision/train.py \
  -c experiments/vision/config/ILNN-CIFAR10.txt \
  --device cuda:0 \
  --seed 1
```

```bash
python experiments/vision/train.py \
  -c experiments/vision/config/ILNN-CIFAR100.txt \
  --device cuda:0 \
  --seed 1
```

Five-seed script:

```bash
DEVICE=cuda:0 SEEDS="1 2 3 4 5" bash scripts/run_vision_cifar.sh
```

## Reproduce Genomics Experiments

GUE promoter and Covid tasks:

```bash
DEVICE=cuda:0 bash scripts/run_genomics_gue.sh
```

TEB pseudogene tasks:

```bash
DEVICE=cuda:0 bash scripts/run_genomics_teb.sh
```

Run one GUE split manually:

```bash
python experiments/genomics/train.py \
  -c experiments/genomics/configs/ILNN_GUE_promoter.txt \
  --device cuda:0 \
  --dataset_name prom_core_tata \
  --seed 1
```

Run one TEB split manually:

```bash
python experiments/genomics/train.py \
  -c experiments/genomics/configs/ILNN_TEB_pseudogenes.txt \
  --device cuda:0 \
  --dataset_name processed_pseudogenes \
  --seed 1
```

## Main Components

- PLFC: [lib/lorentz/layers/LFC.py](lib/lorentz/layers/LFC.py)
- GyroLBN: [lib/GyroBN/GyroBNH.py](lib/GyroBN/GyroBNH.py) and
  [lib/lorentz/layers/LBnorm.py](lib/lorentz/layers/LBnorm.py)
- Log-radius concatenation and Lorentz dropout:
  [lib/lorentz/layers/LConv.py](lib/lorentz/layers/LConv.py)

## Notes

The code vendors a local Geoopt snapshot because the experiments rely on custom
Lorentz operations and optimizers. `RiemannianLineSearch` is optional in this
release because newer SciPy versions removed a function required by Geoopt's old
line-search implementation; ILNN uses `RiemannianSGD` and `RiemannianAdam`.

## Citation

```bibtex
@article{shi2026intrinsic,
  title={Intrinsic Lorentz Neural Network},
  author={Shi, Xianglong and Chen, Ziheng and Jiang, Yunhan and Sebe, Nicu},
  journal={arXiv preprint arXiv:2602.23981},
  year={2026}
}
```

