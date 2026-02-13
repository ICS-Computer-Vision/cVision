# Phase 3: Porthole filtering, binary segmentation, bounding boxes

No trained model. Uses **classical (SoTA) methods**:

1. **Filtering** – Separate porthole from road: grayscale, optional CLAHE, light blur.
2. **Binary** – Porthole = **black**, road = **white** (Otsu or adaptive threshold).
3. **Detection** – Contour-based regions + optional **Hough circle** detection; merge; draw bounding boxes.

## Methods (no training)

- **Otsu’s threshold** – Automatic global threshold.
- **Adaptive Gaussian threshold** – Handles uneven lighting.
- **Morphology** – Close/open to clean the binary mask.
- **Contour analysis** – `findContours` → filter by area, circularity, aspect ratio → `boundingRect`.
- **Hough circle detection** (OpenCV) – Robust circle detection; results merged with contour boxes via IoU.

Tuning is done in `config.py` (threshold method, morphology kernels, min area, circularity, Hough params).

## Usage

From project root (default input is `images`, i.e. original frames):

```bash
# Process all frames in images, write to phase_3_output
python -m phase_3.detect_portholes --output phase_3_output

# Or use enhanced images from Phase 2
python -m phase_3.detect_portholes --input images_enhanced --output phase_3_output

# Preview first frame (original, binary, boxes)
python -m phase_3.detect_portholes --preview

# Limit number of frames
python -m phase_3.detect_portholes --output phase_3_output --limit 10
```

From `phase_3` directory:

```bash
python detect_portholes.py --input ../images --output ../phase_3_output --preview
```

## Outputs

- `output_dir/binary/` – Binary images (porthole = black, road = white).
- `output_dir/boxes/` – Original frames with green bounding boxes around portholes.

## Config (`config.py`)

- **FILTER_CONFIG** – CLAHE, blur.
- **BINARY_CONFIG** – `method`: `"otsu"` or `"adaptive"`; `porthole_is_darker`.
- **MORPH_CONFIG** – Close/open kernel sizes.
- **CONTOUR_CONFIG** – `min_porthole_distance` (min diagonal of bounding box in px; smaller = noise), `min_area`, `max_area_ratio`, `min_circularity`, `max_aspect_ratio`.
- **HOUGH_CONFIG** – `dp`, `min_dist_ratio`, radius range, Canny/accumulator thresholds.
- **DETECTION_MODE** – `"contours_only"` | `"hough_only"` | `"contours_and_hough"`.

## Dependencies

Same as project: `opencv-python`, `numpy` (and `matplotlib` for `--preview`).
