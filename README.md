# ASEF-Net

Official PyTorch implementation of **ASEF-Net: Learning Spatially Asynchronous Exposure Trajectories for Low-Light Image Enhancement** (Neurocomputing submission).

ASEF-Net models low-light enhancement as an exposure-indexed latent-space evolution process: a U-Net-style encoder provides the initial exposure state, a Neural ODE evolves it along a continuous exposure coordinate under three predicted spatial condition fields (exposure velocity, noise confidence, color stability) and a physics-inspired bounded modulation field (PIEMF), and a confidence-aware decoder fuses the decoded trajectory into the enhanced image.

## Installation

Python 3.10+ and PyTorch 2.2+ are recommended. Install a PyTorch build matching your CUDA setup, then:

```bash
pip install -r requirements.txt
```

For the downstream detection experiments (Section 4.8), additionally install [ultralytics](https://github.com/ultralytics/ultralytics): `pip install ultralytics`.

## Datasets

Paired images must share the same filename stem in their `low` and `high` folders; the loader fails if any pair is missing.

### LOL-v1 (485 train / 15 test pairs)

```text
datasets/LOL-v1/
├── our485/
│   ├── low/
│   └── high/
└── eval15/
    ├── low/
    └── high/
```

### CUED (Construction UAV Enhancement Dataset, 1,200 train / 300 test pairs)

```text
datasets/CUED/
├── train/{low,high}/
└── test/
    ├── low/
    ├── high/
    ├── valid_masks/        # 8-bit masks; nonzero = geometrically valid overlap
    └── metadata.csv        # image_id,site_id
```

Night-time images are geometrically registered to their daytime references with SIFT matching and RANSAC homography estimation before full-reference evaluation; all methods are scored on the same valid-overlap masks.

```bash
python -m scripts.register_cued \
  --low-dir /path/to/cued/night --high-dir /path/to/cued/day \
  --out-dir datasets/CUED/test
```

Check `registration_report.csv` and review rejected homographies before evaluating. CUED is planned for release upon acceptance.

### ExDark

Used without paired references for the no-reference statistics and the zero-shot detection protocol; the original distribution (class folders with `.txt` annotations) is read directly by the scripts below.

## Training

Default configurations reproduce the manuscript settings: 512×512 inputs, 512-channel bottleneck, T=10 exposure stages, 10 RK4 steps per stage, Adam (lr 1e-4, ×0.5 every 100 epochs), batch size 4, 300 epochs, loss weights (char 1.0, perc 0.1, mono 0.05, sep 0.1, temp 0.05), δ=2.0, τ=0.3, ε=0.1, η=1e-6. Training never reads the test split.

```bash
python train.py --config configs/lol_v1.yaml          # LOL-v1
python train.py --config configs/cued.yaml            # CUED
python train.py --config configs/lol_v1.yaml --resume runs/asef_lol_v1/last.pt
```

Repeated-seed runs use the same command with a different `training.seed` (the manuscript reports seeds 2026/2027/2028).

## Evaluation

Evaluates a fixed checkpoint on the test split and exports per-image metrics (PSNR, 11×11 Gaussian-window SSIM, VGG-LPIPS on [-1,1] inputs) plus a JSON summary. No test-time ground-truth-mean correction, self-ensembling, or per-image tuning is applied.

```bash
python evaluate.py --config configs/lol_v1.yaml \
  --checkpoint runs/asef_lol_v1/last.pt --out results/lol_v1_per_image.csv

python evaluate.py --config configs/cued.yaml \
  --checkpoint runs/asef_cued/last.pt --out results/cued_per_image.csv
```

Zero-shot cross-dataset transfer (LOL-v1 checkpoint on the CUED test split):

```bash
python evaluate.py --config configs/cued.yaml \
  --checkpoint runs/asef_lol_v1/last.pt --out results/cued_zeroshot.csv
```

Single-image / folder enhancement:

```bash
python infer.py --config configs/lol_v1.yaml \
  --checkpoint runs/asef_lol_v1/last.pt \
  --input path/to/low_light_images --output outputs/enhanced
```

## Reproducing the paper analyses

| Manuscript table | Command |
|---|---|
| LOL-v1 / CUED comparisons | `train.py` then `evaluate.py` (above); compared methods use their official released code and checkpoints under the stated protocol |
| Illumination-heterogeneity stratification | `python -m scripts.stratify --csv results/cued_per_image.csv --low-dir datasets/CUED/test/low --out results/stratified.json` |
| Bootstrap CIs (image-level / site-clustered) | `python -m scripts.bootstrap --ours results/cued_per_image.csv --baseline results/clode_cued.csv --cluster --assignment-csv results/stratified.assignment.csv` |
| Repeated-seed mean ± std | `python -m scripts.seed_aggregate --csvs results/seed2026.csv results/seed2027.csv results/seed2028.csv` |
| ExDark no-reference statistics | `python -m scripts.exdark_stats --input outputs/exdark/asef_net` |
| Efficiency (params, latency, FPS, VRAM, FLOPs) | `python -m scripts.benchmark --config configs/lol_v1.yaml --checkpoint runs/asef_lol_v1/last.pt` |
| Stage-count sweep (T = 1/3/5/10) | `python -m scripts.benchmark --config configs/lol_v1.yaml --stages 1` (likewise 3, 5) |
| ExDark zero-shot detection (YOLOv11n) | `python -m scripts.detection_exdark --exdark-root /path/to/ExDark --images outputs/exdark/asef_net` |
| CUED detection (YOLOv11n, 100 epochs) | `python -m scripts.detection_cued --labels-root /path/to/cued_det_labels --train-images outputs/cued_train/asef_net --test-images outputs/cued_test/asef_net` |

Ablation variants (`model.variant` in the YAML): `full`, `v-spatial-mean`, `n-spatial-mean`, `c-spatial-mean`, `no-local-PIEMF`, `mean-fusion`, `last-stage-only`. Spatial-field ablations are separately trained runs with the corresponding variant; `model.stages` and `model.rk4_steps` control the trajectory configuration.

## Method summary

- **Exposure initialization:** 3-level downsampling to a 512-channel bottleneck at H/8 × W/8.
- **Spatial Exposure Prior:** nonnegative velocity via Softplus; noise confidence and color stability via Sigmoid.
- **PIEMF:** `R = softplus(R_g + ε·tanh(ΔR))` with ε = 0.1, bounding the spatial response deviation by ε while remaining positive.
- **Dynamics:** `dZ/de = (1+v) ⊙ f_θ([Z, n, c, p(e)]) ⊙ (R+η)^{-1}`, η = 1e-6; f_θ uses convolutions, GroupNorm and SiLU.
- **Solver:** fixed-step RK4 with four dynamics evaluations per step; the exposure position encoding is queried at the continuous RK4 stage coordinates.
- **Reconstruction:** one decoder shared by all stages; three-convolution attention produces unnormalized logits followed by a single Softmax over the stage dimension.
- **Objective:** elementwise Charbonnier on 8-bit RGB residuals, VGG-19 perceptual loss (conv1_2/conv2_2/conv3_3/conv4_3), trajectory monotonicity, SEP luminance-correlation prior, and trajectory smoothness.

## Implementation notes

- The shared decoder reconstructs each exposure-stage image from the latent state alone (`I_t = D(Z_t)` in Eq. 15); all stages share the same decoder weights, and the encoder produces only the bottleneck feature used for exposure-state construction.
- The monotonicity and smoothness regularizers reduce L1 per stage spatially and sum over stages, matching the mean-reduced Charbonnier scale.
- The SEP correlation regularizer is computed on the original velocity prediction, before any spatial-mean broadcasting in the ablation variants.
- `scripts/benchmark.py` reports the nominal RK4 stage evaluations (4TN) and the count of dynamics calls through the general entry point; on this implementation both equal 400 for the default configuration. Latency and FPS follow the two independent measurement protocols described in the manuscript; measured values depend on the GPU, driver and PyTorch build.
- Training uses horizontal/vertical flip augmentation and gradient-norm clipping at 1.0.

## Verification

```bash
pytest -q                                                    # unit + equation tests
python -m scripts.audit_model --config configs/lol_v1.yaml    # parameter / structure report
python -m scripts.verify_paper_alignment --config configs/lol_v1.yaml --strict
python -m scripts.make_smoke_data                             # synthetic pipeline check
python train.py --config configs/smoke.yaml --no-vgg
python evaluate.py --config configs/smoke.yaml --checkpoint runs/smoke/last.pt --out results/smoke.csv --no-lpips
```

The smoke pipeline runs on synthetic data and validates execution and file I/O only; research results must use the real datasets and configurations above.

## License

Released under the [MIT License](LICENSE).

## Citation

```bibtex
@article{dong2026asefnet,
  title={ASEF-Net: Learning Spatially Asynchronous Exposure Trajectories for Low-Light Image Enhancement},
  author={Dong, Xiaoqin and Gao, Rui and Wen, Jiangtao and Li, Tianxiang},
  journal={Neurocomputing},
  year={2026}
}
```
