# Temporal Video Analytics

> **Scene Detection · Neural Restoration · Plenoptic Processing**

A high-performance computer vision framework for sequential video data analysis, neural restoration, and multi-view processing. Three specialized modules operate under strict computational constraints and low-level matrix-processing mandates.

---

## Modules

| Module | Description | Performance |
|--------|-------------|-------------|
| [Hybrid Scene Change Detection](#-module-1-hybrid-scene-change-detection) | Cascading heuristic + Decision Tree classifier | Cuts F1: **0.87** · Dissolves F1: **0.76** |
| [Neural Video Restoration (NAFNet)](#-module-2-neural-video-restoration-nafnet) | Edge-preserving deblurring & motion stabilization | — |
| [Plenoptic Light-Field Analysis](#-module-3-plenoptic-light-field-analysis) | Multi-view depth estimation from micro-lens arrays | — |

---

## Module 1: Hybrid Scene Change Detection

A cascading engine that integrates low-level mathematical heuristics with a supervised Decision Tree classifier to isolate abrupt cuts and gradual dissolves.

### Architecture

```
[Continuous Video Stream]
         │
         ├──► 1. Feature Extraction (Pixel MSE, HSV Histograms, Sobel Gradients)
         │
         └──► 2. Heuristic Filter & Confidence Check
                     ├──► High Confidence / Hard Pass ──► [Immediate Cut Flagged]
                     │
                     └──► Low Confidence Candidate ──► 3. Temporal Window [t-N, t+N]
                                                                  │
                                                                  └──► 4. Decision Tree Classifier
                                                                               ├──► Valid Transition
                                                                               └──► False Positive (Pruned)
```

### Feature Extraction

| Feature | Method |
|---------|--------|
| **Pixel Delta** | Frame-to-frame Mean Squared Error (MSE) for gross structural transitions |
| **Color Histograms** | Multi-channel HSV/RGB bin comparison — invariant to local motion & tracking shots |
| **Edge Gradients** | Structural boundary degradation via horizontal/vertical Sobel convolution kernels |

### Hybrid Filtering Strategy

Classical heuristic thresholds are fast but prone to false positives from camera movement, panning, and lighting anomalies. This engine resolves that with a two-stage approach:

- **Deterministic Hard Passes** — Delta values far exceeding baseline variance bypass ML entirely, preserving latency budgets.
- **Temporal Window Contextualization** — Near-threshold candidates are represented as feature vectors spanning a sliding window of $t-N$ to $t+N$ frames.
- **Decision Tree False-Positive Reduction** — A supervised classifier differentiates actual transitions from transient motion noise and sudden exposure changes.

---

## Module 2: Neural Video Restoration (NAFNet)

An implementation of **Non-Linear Activation Free Networks** optimized for real-time video pipelines.

- Removes motion blur and high-frequency noise from raw frame sequences before downstream analysis
- Retains edge definition without introducing synthetic artifacts
- Integrates directly into the Scene Change Detector preprocessing pipeline for low-contrast/shaky footage

---

## Module 3: Plenoptic Light-Field Analysis

A dedicated pipeline for spatial-temporal streams from plenoptic (light-field) camera arrays.

- Resolves multi-view sub-aperture images from raw micro-lens sensor arrays
- Implements passive depth estimation via disparity maps across sub-aperture views
- Enables post-capture refocusing and perspective shifts on sequential video frames

---

## Quickstart

### Prerequisites

```bash
conda install -c conda-forge opencv tqdm numpy scikit-learn matplotlib
```

### Running Scene Detection

```python
import os
from temporal_cv import VideoStreamer, HybridSCD

# Initialize low-level frame streamer
video_path = os.path.join("train_dataset", "video", "03.mp4")
stream = VideoStreamer(video_path)

# Initialize the cascading detection engine
scd_engine = HybridSCD(
    heuristic_threshold=2000,
    model_path="models/decision_tree_filter.pkl"
)

# Process video and retrieve filtered scene change frames
verified_cuts = scd_engine.process(stream)
print(f"Verified scene changes detected at: {verified_cuts}")
```

---

## Dataset Format

Input datasets map continuous video streams to ground-truth index structures:

```json
[
  {
    "source": "train_dataset/video/03.mp4",
    "scene_change": "train_dataset/gt/03.json",
    "len": 3250
  }
]
```

Ground-truth files use **zero-based indexing** with two transition types:

| Key | Description |
|-----|-------------|
| `cut` | Frame indices for instantaneous, sharp visual cuts |
| `dissolve` | Start/end frame index pairs for gradual cross-fading transitions |
