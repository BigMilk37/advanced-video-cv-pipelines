# Neural Video Restoration: High-Frequency Motion Deblurring

A deep learning framework designed to restore sharp video sequences from blurred frame streams caused by sudden camera manipulation, high-frequency motion artifacts, or slow exposure fluctuations. This repository implements a multi-frame neural deblurring engine built on a Non-Linear Activation Free Network (NAFNet) architecture, optimized for the GoPro Large multi-scale video dataset.

## Performance Benchmark

| Metric | Target |
|--------|--------|
| Peak Signal-to-Noise Ratio (PSNR) | ≥ 27.00 dB |
| Training frame triplets (GoPro) | 2,103 across 22 video segments |
| Testing frame triplets (GoPro) | 1,111 across 22 video segments |

---

## Architecture Design & Mechanics

Video motion deblurring conventionally suffers from high computational overhead and the generation of hallucinatory spatial artifacts. This engine addresses these bottlenecks by utilizing an activation-free convolutional design that processes multiple temporal reference points concurrently.

### 1. Spatial-Temporal Context Ingestion

The neural engine evaluates context sequences via an explicit frame-triplet model ($t-1$, $t$, $t+1$) to reconstruct a singular sharp frame map matching anchor step $t$.

- **Tensor Aggregation:** Ingested images are concatenated directly along the channel dimension ($B \times 3C \times H \times W$) to pass global multi-frame boundary indicators into the primary encoder layers.
- **Dynamic Padding Blocks:** To comply with deep U-Net downsampling pipelines without crop fragmentation, input tensors are symmetrically padded via feature reflections to match internal stride dividers (powers of $2^N$).

### 2. Non-Linear Activation Free Module (NAFNet)

This pipeline removes traditional non-linear activation functions (such as ReLU, GeLU, or LeakyReLU) entirely, avoiding quantization risks and preserving raw floating-point pixel distributions:

- **Simple Gate Mechanics:** Replaces standard activation functions by splitting internal feature matrices into symmetrical channel pairs ($X_1, X_2$) and computing element-wise multiplication matrices ($X_1 \odot X_2$) inside normalized LayerNorm bounds.
- **Simple Channel Attention (SCA):** A streamlined weighting layer that compresses global spatial descriptors down to single-channel vector averages, utilizing single $1 \times 1$ convolutions to efficiently adapt channel distributions.

```
[Frame Triplet (t-1, t, t+1)] ──► Channel Concat (3xC) ──► Input Padding Block
                                                                │
  ┌─────────────────────────────────────────────────────────────┘
  ▼
[Encoder Blocks (Down)] ──► [Simple Gate / SCA Blocks] ──► [Decoder Blocks (Up)]
                                                                │
[Output Channel Mix] ◄── [Residual Frame Anchor (t) Addition] ◄─┘
```

---

## Dataset Layout

The pipeline consumes multi-channel target sequences natively from structured dataset directories:

```
GoPro/
├── train/
│   └── [Video_ID]/
│       ├── sharp/        # Target high-frequency ground truth images
│       └── blur_gamma/   # Distorted temporal frame representations
└── test/
```

During execution, data loader objects apply synchronized spatial augmentations — Random Crop matrices matching $256 \times 256$ dimensions and Horizontal Flips — consistently across all channels within the triplet structure to prevent temporal misalignment.

---

## Quickstart

### Prerequisites

Configure your environment using the core machine learning and hardware-accelerated processing stack:

```bash
conda install -c conda-forge pytorch torchvision albumentations opencv tqdm matplotlib
```

### Model Evaluation

To evaluate a checkpoint against the test set:

```python
import torch
from torch.utils.data import DataLoader
from deblur_core import UNet, NAFNetBlock, GoProDataset, evaluate

# Hardware mapping configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = UNet(block_type=NAFNetBlock).to(device)
checkpoint = torch.load("./weights/deblur_checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

# Stream test dataset
test_dataset = GoProDataset(data_path="GoPro/test", transform=None)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

# Run evaluation
mean_psnr, mean_ssim = evaluate(model, test_loader, device)
print(f"PSNR = {mean_psnr:.2f} dB, SSIM = {mean_ssim:.4f}")
```

---

## Engineering Context

This repository can operate as a standalone restoration framework or as a preprocessing module for downstream video analytics architectures. By filtering out sudden motion artifacts and exposure noise, it significantly stabilizes inter-frame sequences, preventing false positives in high-speed scene change detection and temporal tracking tasks.
