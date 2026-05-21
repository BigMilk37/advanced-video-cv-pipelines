# Neural Video Restoration: High-Frequency Motion Deblurring & Computational Profiling

A deep learning framework designed to restore sharp video sequences from blurred frame streams. This repository implements a multi-frame neural deblurring engine built on a Non-Linear Activation Free Network (NAFNet). Beyond basic restoration, this module focuses on computational efficiency profiling, fault-tolerant training, and temporal attention visualization.

---

## Performance Benchmarks

| Metric | Target / Configuration |
|--------|------------------------|
| Peak Signal-to-Noise Ratio (PSNR) | 30.12 dB |
| Training frame triplets (GoPro) | 2,103 across 22 video segments |
| Testing frame triplets (GoPro) | 1,111 across 22 video segments |
| Profiling | GFLOPs, inference latency, Rate-Distortion curves |

---

## Architecture Design & Mechanics

Video motion deblurring conventionally suffers from high computational overhead. This engine addresses these bottlenecks by utilizing an activation-free convolutional design that processes multiple temporal reference points concurrently.

### 1. Spatial-Temporal Context Ingestion

The neural engine evaluates context sequences via an explicit frame-triplet model ($t-1$, $t$, $t+1$).

- **Tensor Aggregation:** Ingested images are concatenated directly along the channel dimension ($B \times 3C \times H \times W$) to pass global multi-frame boundary indicators into the primary encoder layers.
- **Dynamic Padding Blocks:** Input tensors are symmetrically padded via feature reflections to match internal stride dividers (powers of $2^N$), ensuring artifact-free downsampling through the U-Net structure.

### 2. Non-Linear Activation Free Module (NAFNet)

Removes traditional non-linear activation functions (ReLU, GeLU) entirely to preserve raw floating-point pixel distributions:

- **Simple Gate Mechanics:** Splits internal feature matrices into symmetrical channel pairs ($X_1, X_2$) and computes element-wise multiplication matrices ($X_1 \odot X_2$) inside normalized LayerNorm bounds.
- **Simple Channel Attention (SCA):** Compresses global spatial descriptors down to single-channel vector averages using $1 \times 1$ convolutions to efficiently adapt channel distributions.

---

## Advanced Pipeline Features

### Fault-Tolerant Training

- **Atomic Checkpointing:** Safe state-saving (`save_checkpoint_atomic`) prevents weight corruption during unexpected runtime interruptions.
- **Numeric Validation:** Real-time NaN/Inf gradient checking during the backward pass to catch gradient explosions instantly.
- **Dynamic Monitoring:** Live Train/Validation PSNR plotting hooked directly into the evaluation loop.

### Computational Profiling

Includes an explicit profiling environment to evaluate deployment feasibility:

- Extracts total model parameters and GFLOPs natively.
- Benchmarks real-time inference latency across variable tensor sizes.
- Generates Rate-Distortion (RD) curves plotting visual quality metrics (PSNR/SSIM) against computational expense.

### Temporal Attention Visualization

Features a custom 3D correlation visualization engine mapping cross-frame patch attention. By plotting high-correlation structural patches between the blur triplet and the target anchor, the framework provides an interpretable proxy for how the network tracks motion trajectories.

---

## Dataset Layout

The pipeline consumes multi-channel target sequences from structured dataset directories:

```
GoPro/
├── train/
│   └── [Video_ID]/
│       ├── sharp/        # Target high-frequency ground truth images
│       └── blur_gamma/   # Distorted temporal frame representations
└── test/
```

---

## Quickstart

### Prerequisites

```bash
conda install -c conda-forge pytorch torchvision albumentations opencv tqdm matplotlib thop
```

> `thop` or `ptflops` is required for GFLOPs profiling.

### Model Evaluation

```python
import torch
from torch.utils.data import DataLoader
from deblur_core import UNet, NAFNetBlock, GoProDataset, evaluate, check_numerics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = UNet(block_type=NAFNetBlock).to(device)
checkpoint = torch.load("./weights/deblur_checkpoint_atomic.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

# Stream test dataset
test_dataset = GoProDataset(data_path="GoPro/test", transform=None)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

# Run evaluation
mean_psnr, mean_ssim = evaluate(model, test_loader, device)
print(f"PSNR = {mean_psnr:.2f} dB, SSIM = {mean_ssim:.4f}")
```
