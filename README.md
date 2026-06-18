# ASEF-Net: Asynchronous Spatial Exposure Fields for Low-Light Image Enhancement

PyTorch implementation of **ASEF-Net**.

> **ASEF-Net: Learning Spatially Asynchronous Exposure Trajectories for Low-Light Image Enhancement**
>
> This repository contains the official implementation. The paper is currently under review.

---

## Overview

ASEF-Net models per-region exposure evolution as a spatially-varying dynamical system. Each spatial location owns independent exposure velocity `v(x,y)`, noise confidence `n(x,y)`, and color stability `c(x,y)`, jointly driving a Neural ODE for continuous latent-space exposure evolution.

### Key Components

- **Exposure State Field (ESF)**: Encodes continuous exposure states in latent space
- **Physics-Informed Exposure Modulation Field (PIEMF)**: Enforces bounded spatial modulation over local exposure responses
- **Spatial Exposure Prior (SEP)**: Three condition fields (`v`, `n`, `c`) with physical constraints to guide region-adaptive exposure dynamics
- **Confidence-Aware Trajectory Reconstruction**: Multiple latent exposure states are decoded and fused via learned credibility weights

---

## Project Structure

```
ASEF-Net/
├── models/
│   ├── asef_net_paper_version.py   # Main ASEF-Net implementation (paper version)
│   ├── sef_net_v4.py               # Original SEF-Net baseline
│   └── __init__.py
├── data/
│   ├── dataset.py                  # Dataset loaders
│   └── __init__.py
├── utils/
│   ├── loss.py                     # Loss functions
│   ├── metrics.py                  # PSNR, SSIM, LPIPS
│   ├── visualize.py                # Visualization tools
│   └── __init__.py
├── configs/
│   ├── lol_v1.yaml                 # Config for LOL-v1
│   ├── lol_v2.yaml                 # Config for LOL-v2 Real
│   ├── lol_v2_lite.yaml            # Config for LOL-v2 Real (Lite)
│   ├── sef_net_v4.yaml             # Config for SEF-Net baseline
│   └── ablation_configs.yaml       # Ablation study configs
├── train.py                        # Training script
├── test.py                         # Testing / inference script
├── requirements.txt                # Dependencies
└── README.md
```

---

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download datasets and organize as follows:

**LOL-v1:**
```
LOL/
├── our485/
│   ├── low/
│   └── high/
└── eval15/
    ├── low/
    └── high/
```

**LOL-v2 (Real):**
```
LOL-v2/
└── Real/
    ├── train/
    │   ├── low/
    │   └── high/
    └── test/
        ├── low/
        └── high/
```

Update dataset paths in `configs/*.yaml`:
```yaml
dataset:
  root: "/path/to/your/dataset"
```

### 3. Training

```bash
# Train ASEF-Net on LOL-v1
python train.py --config configs/lol_v1.yaml

# Train on LOL-v2
python train.py --config configs/lol_v2.yaml

# Resume from checkpoint
python train.py --config configs/lol_v1.yaml --resume checkpoints/best.pth
```

### 4. Testing

```bash
# Test on dataset with GT (metrics: PSNR/SSIM/LPIPS)
python test.py \
    --config configs/lol_v1.yaml \
    --weights checkpoints/best.pth \
    --mode dataset

# Test on a folder of images (no GT)
python test.py \
    --config configs/lol_v1.yaml \
    --weights checkpoints/best.pth \
    --mode folder \
    --input_dir /path/to/your/low_light_images \
    --output_dir results/
```

---

## Implementation Details

### Model Architecture

The implementation strictly follows the five-module architecture described in the paper:

1. **Exposure Initialization Encoder** (Eq. 2): Encodes low-light input into initial feature representation
2. **Exposure State Field Constructor** (Eq. 3-5): Constructs continuous exposure states with position encoding
3. **Condition Field Generation** (Eq. 7-9): Generates `v(x,y)`, `n(x,y)`, `c(x,y)` fields with physical constraints
4. **Physics-Informed Spatial Exposure Dynamics** (Eq. 10-13): Neural ODE solver with PIEMF and velocity modulation
5. **Exposure Trajectory Reconstruction** (Eq. 14-18): Confidence-aware multi-stage fusion

### Loss Functions

The loss function follows Eq. 19-23 in the paper:

- `L_char`: Charbonnier reconstruction loss (Eq. 19)
- `L_perc`: Perceptual loss via VGG-19 (Eq. 20)
- `L_mono`: Monotonicity loss for trajectory consistency (Eq. 21)
- `L_sep`: SEP prior loss enforcing negative correlation (Eq. 22)
- `L_temp`: Trajectory smoothness loss (Eq. 23)

Total loss: `L = λ_char·L_char + λ_perc·L_perc + λ_mono·L_mono + λ_sep·L_sep + λ_temp·L_temp` (Eq. 24)

---

## Ablation Study

To reproduce ablation experiments, modify `models/asef_net_paper_version.py`:

| Ablation | How to modify |
|----------|--------------|
| w/o `v(x,y)` | Set `v = torch.ones_like(v)` in `SpatialExposureFlowSolver.forward()` |
| w/o `n(x,y)` | Set `n = torch.zeros_like(n)` in `SpatialExposureFlowSolver.forward()` |
| w/o `c(x,y)` | Set `c = torch.zeros_like(c)` in `SpatialExposureFlowSolver.forward()` |
| w/o PIEMF | Replace `self.piemf(z)` with `torch.ones_like(z)` |
| w/o multi-stage fusion | Return `I_list[-1]` instead of weighted fusion in `ExposureTrajectoryReconstruction.forward()` |

---

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- torchvision
- numpy
- pillow
- pyyaml
- opencv-python
- scikit-image
- lpips

See `requirements.txt` for complete list.

---

## Datasets

- [LOL-v1](https://daooshee.github.io/BMVC2018website/)
- [LOL-v2](https://github.com/flyywh/CVPR-2020-Semi-Low-Light)
- [ExDark](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) (for downstream detection evaluation)

---

## Acknowledgements

This work builds upon the following excellent open-source projects:

- [NAFNet](https://github.com/megvii-research/NAFNet) for efficient block design
- [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE) for curve-based enhancement baseline
- [Neural ODE](https://github.com/rtqichen/torchdiffeq) for continuous dynamics

---

## License

This project is released under the MIT License.

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{asefnet2025,
  title={ASEF-Net: Learning Spatially Asynchronous Exposure Trajectories for Low-Light Image Enhancement},
  journal={arXiv preprint},
  year={2025}
}
```