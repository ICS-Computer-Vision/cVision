# Sprint 1: Geometry of Formation

Computer vision pipeline that builds a metric top-down (bird's-eye) view from dashboard video. It uses camera calibration and homography to undistort frames and warp them to a world-aligned map with a known scale.

---

## Requirements

- **Python 3**
- **OpenCV** (`opencv-python`)
- **NumPy**

Install dependencies:

```bash
pip install opencv-python numpy
```

---

## How to Run

All steps are run from this folder (`sprint1/`). The recommended way is the menu driver:

```bash
cd sprint1
python3 run_sprint1_menu.py
```

The menu lets you run each step individually (1–7) or execute the full sequence (9). Follow the steps in order the first time.

You can also run scripts directly:

```bash
python3 calibrate_camera.py
python3 calculate_homography.py
python3 fix_resolution.py
# ... then video steps (see Pipeline Steps below)
```

---

## Pipeline Steps

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `calibrate_camera.py` | Estimate lens distortion from checkerboard images. Produces `camera_calibration.pkl` and verification images. |
| 2 | `calculate_homography.py` | Compute top-down perspective transform from a reference image. Produces `geometry_pipeline.pkl` and `verification_3_birdseye.jpg`. |
| 3 | `fix_resolution.py` | Rescale the pipeline for video resolution if it differs from the calibration photos. Produces `geometry_pipeline_video.pkl`. |
| 4 | `test_on_video.py` | Quick check: narrow top-down view; confirms lines are parallel. |
| 5 | `test_on_video_wide.py` | Short preview (about 5 s) of the wide canvas; no file written. |
| 6 | `create_side_by_side.py` | Export split-screen video (raw + map) as `sprint1_demo_reel.mp4`. |
| 7 | `pipeline_sprint1_formation.py` | Process full video and export frame images to `sprint1_frames/` (no video file). |
| 8 | `restoration_engine_sprint2.py` | Restore top-down frames (stabilize, denoise, normalize lighting) to `sprint2_restored_frames/`. |

Steps 4–7 require `road_test.mp4` in this folder.

---

## Inputs

- **`calibration_images/`** — Checkerboard photos (`.jpg`) used for camera calibration. Step 1 expects a subfolder named `calibration_images` with multiple images.
- **`homography_setup.jpg`** — Reference image of the checkerboard on the ground, used to compute the homography (Step 2). Must match the camera and layout used for calibration.
- **`road_test.mp4`** — Dashboard video to process in Steps 4–7. Place it in this folder before running those steps.

---

## Outputs

- **`camera_calibration.pkl`** — Camera matrix and distortion coefficients (Step 1).
- **`geometry_pipeline.pkl`** — Homography and calibration at photo resolution (Step 2).
- **`geometry_pipeline_video.pkl`** — Pipeline scaled for video resolution; used by all video scripts (Step 3).
- **Verification images** — `verification_1_corners_found.jpg`, `verification_2_undistorted.jpg`, `verification_3_birdseye.jpg` (and optionally `debug_corners_full_image.jpg`) for sanity checks.
- **`sprint1_demo_reel.mp4`** — Side-by-side video (Step 6). Output size is scaled (default half resolution) to keep the file smaller.
- **`sprint1_frames/`** — Folder of JPEG frames from the final pipeline (Step 7). Each image is a full square bird's-eye view (e.g. 2000x2000). Frames are exported every Nth video frame (configurable in the script).
- **`sprint2_restored_frames/`** — Restored JPEG frames from Sprint 2 (Step 8), stabilized and light-normalized.

Generated videos and `sprint1_frames/` are listed in `.gitignore` so they are not committed.

---

## Sprint 2: Restoration Engine

The restoration engine consumes the top-down frames from Sprint 1 and outputs cleaner, stabilized road textures.

Run from this folder:

```bash
python3 restoration_engine_sprint2.py
```

Key toggles are at the top of the script:

- `ENABLE_STABILIZATION`
- `ENABLE_DENOISE`
- `ENABLE_ILLUMINATION_NORM`
- `ENABLE_SHADOW_FLATTEN`
- `ENABLE_SHARPEN`

If you want a preview video, set `EXPORT_VIDEO = True`.

---

## Other Scripts

- **`debug_black_screen.py`** — Diagnostic for a blank map view; useful if the warped output is black (often a resolution mismatch; re-run Step 3).

---

## Configuration

Key settings are at the top of each script:

- **`pipeline_sprint1_formation.py`** — `CANVAS_SIZE`, `FRAME_EXPORT_EVERY`, `SHIFT_X` (translation of the road on the canvas).
- **`create_side_by_side.py`** — `OUTPUT_SCALE` (smaller value = smaller file; default 0.5).
- **`calibrate_camera.py`** — `CHECKERBOARD_DIMS`, `SQUARE_SIZE`.
- **`calculate_homography.py`** — `IMAGE_PATH`, `CHECKERBOARD_DIMS`, `SQUARE_SIZE_CM`, `PIXELS_PER_CM`, crop bounds.
- **`fix_resolution.py`** — `PHOTO_W`/`PHOTO_H`, `VIDEO_W`/`VIDEO_H` (must match your calibration image and video resolution).

Adjust these if you change camera, checkerboard, or video resolution.
