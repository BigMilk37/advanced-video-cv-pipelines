# Advanced Video CV Pipelines

> **Scene Change Detection · Neural Deblurring · Plenoptic Light-Field Processing**

A collection of three advanced computer vision pipelines for sequential video analysis, each implemented in Python as Jupyter notebooks. The modules target different stages of the video understanding stack — from raw sensor decoding to deep learning restoration to temporal segmentation.

---

## Repository Structure

```
advanced-video-cv-pipelines/
├── scene-change-detection/     # Hybrid heuristic + ML scene segmentation
├── deblurring/                 # NAFNet-based neural video restoration
└── plenoptics/                 # Light-field decoding and multi-view synthesis
```

---

## Modules

### 1. Hybrid Scene Change Detection

Detects abrupt cuts and gradual dissolves in continuous video streams using a two-stage cascading engine: low-level mathematical heuristics filter high-confidence transitions immediately, while near-threshold candidates are passed through a sliding temporal window and classified by a supervised Decision Tree to suppress false positives.

**Features:**
- Frame-to-frame MSE, HSV histogram comparison, and Sobel edge gradients for feature extraction
- Deterministic hard-pass bypass for obvious transitions (latency-preserving)
- Temporal window contextualization ($t-N$ to $t+N$) for ambiguous candidates
- Decision Tree classifier trained to distinguish real cuts from camera pans, flashes, and exposure changes

**Performance:** Cuts F1 = **0.87** · Dissolves F1 = **0.76**

**Dependencies:**
```bash
conda install -c conda-forge opencv tqdm numpy scikit-learn matplotlib
```

---

### 2. Neural Video Restoration (NAFNet)

A multi-frame deblurring engine built on a Non-Linear Activation Free Network (NAFNet) architecture. Processes frame triplets ($t-1$, $t$, $t+1$) concatenated along the channel dimension through a U-Net encoder-decoder, removing motion blur while preserving high-frequency edge structure.

**Features:**
- Non-linear activation-free design using Simple Gate mechanics and Simple Channel Attention (SCA)
- Fault-tolerant training with atomic checkpointing and real-time NaN/Inf gradient validation
- GFLOPs profiling, inference latency benchmarking, and Rate-Distortion curve generation
- Temporal attention visualization mapping cross-frame structural patch correlations

**Performance:** PSNR = **30.12 dB** on the GoPro dataset (2,103 train / 1,111 test frame triplets)

**Dependencies:**
```bash
conda install -c conda-forge pytorch torchvision albumentations opencv tqdm matplotlib thop
```

---

### 3. Plenoptic Light-Field Processing

A full decoding and synthesis pipeline for raw plenoptic (light-field) camera output from Lytro micro-lens array sensors. Converts raw 3280×3280 Bayer-patterned sensor data into a 4D light-field ray grid $L(x, y, u, v)$, then extracts virtual sub-aperture views for parallax animation and stereoscopic rendering.

**Features:**
- Bayer demosaicing via high-quality Malvar linear interpolation kernels
- Hexagonal MLA grid normalization with sub-pixel rotation correction and bilinear sampling
- Sub-aperture view extraction across the full $(u, v)$ angular grid with histogram brightness normalization
- Red/Cyan stereoscopic anaglyph synthesis with adjustable parallax convergence

**Dependencies:**
```bash
conda install -c conda-forge numpy opencv matplotlib scipy scikit-image imageio tqdm
```

---

## Quickstart

Each module is self-contained. Navigate to the relevant subdirectory and open the notebook:

```bash
# Scene change detection
cd scene-change-detection
jupyter notebook

# Neural deblurring
cd deblurring
jupyter notebook

# Plenoptic processing
cd plenoptics
jupyter notebook
```

---

## Dataset Formats

**Scene Change Detection** — JSON ground truth with zero-based frame indices:
```json
[
  {
    "source": "train_dataset/video/03.mp4",
    "scene_change": "train_dataset/gt/03.json",
    "len": 3250
  }
]
```

**Neural Deblurring** — GoPro dataset directory structure:
```
GoPro/
├── train/[Video_ID]/{sharp/, blur_gamma/}
└── test/[Video_ID]/{sharp/, blur_gamma/}
```

**Plenoptics** — Raw `.png` sensor grid + `metadata.json` with micro-lens calibration:
```python
plenoptic_image_config = {
    'block_width': 9.927582228653469,
    'block_height': 8.607538408172838
}
```

---

## Tech Stack

| Library | Used In |
|---------|---------|
| `opencv`, `numpy` | All modules |
| `scikit-learn` | Scene change detection (Decision Tree) |
| `pytorch`, `torchvision` | Neural deblurring (NAFNet) |
| `scipy`, `scikit-image` | Plenoptics (Bayer demosaicing, histogram matching) |
| `imageio` | Plenoptics (GIF / anaglyph output) |
| `thop` / `ptflops` | Deblurring (GFLOPs profiling) |
| `matplotlib` | All modules (visualization) |
