# Plenoptic Light-Field Image Processing & Multi-View Synthesizer

A specialized computer vision pipeline for decoding, transforming, and virtualizing multi-view light fields captured by micro-lens array (MLA) sensors. Specifically targeted at the Lytro Light Field Camera architecture, this module covers raw sensor decoding, Bayer demosaicing, sub-aperture view extraction, and stereoscopic anaglyph synthesis.

---

## Pipeline Features

| Feature | Description |
|---------|-------------|
| Bayer Demosaicing | High-quality linear interpolation across B/G1/G2/R channel masks |
| Hexagonal Grid Normalization | Geometric row-shift correction for honeycomb MLA layouts |
| Sub-Aperture Extraction | Virtual pinhole camera reconstruction across $(x, y, u, v)$ light-field coordinates |
| Histogram Normalization | Cross-frame brightness correction via `match_histograms` |
| Stereoscopic Anaglyph Synthesis | Red/Cyan compositing with adjustable horizontal parallax |

---

## Technical Architecture

Unlike conventional sensors that collapse 4D light rays into a 2D pixel grid $(x, y)$, a plenoptic camera captures both spatial position and angular direction $(x, y, u, v)$ using an array of micro-lenses positioned over the sensor.

### 1. Bayer Demosaicing

Raw Lytro sensor output stores color information in a B/G1/G2/R Bayer pattern. The pipeline:

- Constructs per-channel binary masks tiled across the full 3280×3280 sensor grid.
- Applies high-quality linear interpolation kernels (Malvar et al.) via `scipy.ndimage.convolve` to reconstruct full-resolution RGB from sparse single-channel samples.

### 2. Hexagonal Array Normalization

Lytro micro-lenses are packed in a hexagonal honeycomb formation rather than a Cartesian square grid:

- **Fractional Row Offsets:** Every alternating row is geometrically displaced by half a pixel width.
- **Rotation Correction:** The MLA is slightly rotated relative to the sensor pixel grid; the pipeline tracks micro-lens center coordinates with sub-pixel precision to compensate.
- **Bilinear Interpolation:** Non-integer pixel coordinates under each lens are resolved via bilinear sampling to avoid aliasing.

### 3. Sub-Aperture View Extraction

By sampling the pixel at position $(u, v)$ from beneath every micro-lens across spatial coordinates $(x, y)$, the pipeline reconstructs a standalone virtual sub-aperture view:

- **Multi-View Array Generation:** Extracts a grid of virtual cameras representing continuous horizontal and vertical viewing shifts.
- **Parallax Transition:** Stepping through adjacent sub-aperture images produces smooth scene parallax, as if physically moving the camera laterally.
- **Histogram Matching:** Each extracted frame is brightness-normalized against a reference center view using `skimage.exposure.match_histograms` to compensate for uneven MLA light transmission.

```
[Raw Hexagonal Light-Field Sensor (3280×3280)]
                   │
      ├── Bayer Demosaicing (Malvar Interpolation)
      └── Hexagonal Row-Shift Normalization
                   │
   [4D Light-Field Ray Grid L(x, y, u, v)]
                   │
   ┌───────────────┴───────────────┐
   ▼                               ▼
[Sub-Aperture Slicing]     [Stereo View Selection]
   │                               │
[Multi-View Array + Histogram Norm] [Parallax Shift Δx Calibration]
   │                               │
[Perspective Parallax GIF]  [Red/Cyan Anaglyph Synthesis]
```

### 4. Stereoscopic Anaglyph Synthesis

Sub-aperture views at opposite horizontal offsets mimic human left/right eye separation:

- **Stereo Pairing:** Left and right views are selected at maximum horizontal parallax (e.g., `dx = -4` and `dx = +4`).
- **Histogram Alignment:** The right view is matched to the left view's color distribution before compositing.
- **Color-Space Compositing:** The left-eye tensor is mapped to the Red channel; the right-eye tensor populates the Green and Blue (Cyan) channels.
- **Parallax Adjustment:** The convergence plane can be shifted post-capture by varying `position_shift_x`, controlling which scene depth appears to protrude or recede.

---

## Dataset & File Format

| File | Description |
|------|-------------|
| `*.png` | Demosaiced or raw monochromatic sensor grid (Bayer-patterned) |
| `metadata.json` | Micro-lens pitch, spatial offsets, and sensor-to-lens calibration bounds |

Example config used at runtime:

```python
plenoptic_image_config = {
    'block_width': 9.927582228653469,
    'block_height': 8.607538408172838
}
```

---

## Quickstart

### Prerequisites

```bash
conda install -c conda-forge numpy opencv matplotlib scipy scikit-image imageio tqdm
```

### Running the Pipeline

```python
import numpy as np
import imageio
from skimage import io
from skimage.util import img_as_float
from skimage.exposure import match_histograms
from plenoptic_engine import high_quality_bayer_interpolation, baseline_move

# 1. Load and demosaic raw light-field image
raw_data = img_as_float(io.imread("data/bee.png"))
if len(raw_data.shape) == 3:
    raw_data = raw_data[:, :, 0]

image_rgb = high_quality_bayer_interpolation(raw_data)

# 2. Extract sub-aperture views with histogram normalization
plenoptic_image_config = {'block_width': 9.9275, 'block_height': 8.6075}
reference_frame = baseline_move(image_rgb, plenoptic_image_config, 0, 0)

frames = []
for dx in range(-4, 5):
    for dy in range(-3, 4):
        frame = baseline_move(image_rgb, plenoptic_image_config, dy, dx)
        frames.append(match_histograms(frame, reference_frame, channel_axis=-1))

imageio.mimsave("output/parallax_view.gif", frames, duration=100)

# 3. Synthesize Red/Cyan anaglyph
left_view = baseline_move(image_rgb, plenoptic_image_config, 0, -4)
right_view = baseline_move(image_rgb, plenoptic_image_config, 0, 4)
right_view = match_histograms(right_view, left_view, channel_axis=-1)

anaglyph = np.zeros_like(left_view)
anaglyph[:, :, 0] = left_view[:, :, 0]   # Red  ← left eye
anaglyph[:, :, 1] = right_view[:, :, 1]  # Green ← right eye
anaglyph[:, :, 2] = right_view[:, :, 2]  # Blue  ← right eye

imageio.imwrite("output/stereo_anaglyph.png", (anaglyph * 255).astype("uint8"))
print("Anaglyph rendered successfully.")
```

---

## Engineering Context

This module operates as a standalone light-field processing framework or as the spatial input layer for downstream video analytics pipelines. Sub-aperture view generation provides depth-aware frame sequences that significantly improve passive depth estimation and inter-frame motion tracking in multi-view scene analysis.
