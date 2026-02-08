"""
Sprint 2: Restoration Engine

Takes the top-down frames from Sprint 1 and produces stabilized, denoised,
shadow-normalized road textures.
"""

import os
import glob
import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_DIR = "sprint1_frames"
OUTPUT_DIR = "sprint2_restored_frames"

# Set this to True only when you're happy with results (video writing is slow)
EXPORT_VIDEO = False
OUTPUT_VIDEO_PATH = "sprint2_restored_preview.mp4"
OUTPUT_VIDEO_FPS = 20


# Stabilization
ENABLE_STABILIZATION = True
STABILIZE_MOTION = cv2.MOTION_AFFINE
STABILIZE_MAX_ITERS = 100
STABILIZE_EPS = 1e-6
STABILIZE_DOWNSCALE = 0.5  # Use 1.0 for full-res alignment

# Denoising
ENABLE_DENOISE = True
DENOISE_H = 5
DENOISE_H_COLOR = 5
DENOISE_TEMPLATE_WINDOW = 7
DENOISE_SEARCH_WINDOW = 21

# Illumination / shadow normalization
ENABLE_ILLUMINATION_NORM = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
ENABLE_SHADOW_FLATTEN = True
SHADOW_FLATTEN_SIGMA = 25

# Sharpening
ENABLE_SHARPEN = True
SHARPEN_AMOUNT = 0.5
SHARPEN_SIGMA = 2.0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def list_frames(input_dir):
    pattern = os.path.join(input_dir, "*.jpg")
    files = sorted(glob.glob(pattern))
    return files


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def downscale_image(img, scale):
    if scale == 1.0:
        return img
    h, w = img.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def scale_warp_matrix(warp_2x3, scale):
    if scale == 1.0:
        return warp_2x3
    warp_3x3 = np.eye(3, dtype=np.float32)
    warp_3x3[:2, :] = warp_2x3
    scale_m = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float32)
    inv_scale_m = np.array([[1.0 / scale, 0, 0], [0, 1.0 / scale, 0], [0, 0, 1]], dtype=np.float32)
    warp_3x3 = inv_scale_m @ warp_3x3 @ scale_m
    return warp_3x3[:2, :]


def estimate_warp_ecc(prev_gray, curr_gray):
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        STABILIZE_MAX_ITERS,
        STABILIZE_EPS,
    )
    _, warp = cv2.findTransformECC(
        prev_gray, curr_gray, warp, STABILIZE_MOTION, criteria, None, 1
    )
    return warp


def estimate_warp_optical(prev_gray, curr_gray):
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=15,
        blockSize=7,
    )
    if prev_pts is None:
        return np.eye(2, 3, dtype=np.float32)

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None
    )
    if curr_pts is None or status is None:
        return np.eye(2, 3, dtype=np.float32)

    status = status.reshape(-1)
    prev_good = prev_pts[status == 1]
    curr_good = curr_pts[status == 1]

    if len(prev_good) < 10 or len(curr_good) < 10:
        return np.eye(2, 3, dtype=np.float32)

    warp, _ = cv2.estimateAffinePartial2D(
        curr_good, prev_good, method=cv2.RANSAC, ransacReprojThreshold=3
    )
    if warp is None:
        return np.eye(2, 3, dtype=np.float32)
    return warp.astype(np.float32)


def stabilize_frame(frame, prev_gray):
    if prev_gray is None:
        return frame, to_gray(frame)

    curr_gray = to_gray(frame)
    prev_small = downscale_image(prev_gray, STABILIZE_DOWNSCALE)
    curr_small = downscale_image(curr_gray, STABILIZE_DOWNSCALE)

    warp = None
    if ENABLE_STABILIZATION:
        try:
            warp = estimate_warp_ecc(prev_small, curr_small)
        except cv2.error:
            warp = None

        if warp is None:
            warp = estimate_warp_optical(prev_small, curr_small)

        warp = scale_warp_matrix(warp, STABILIZE_DOWNSCALE)
    else:
        warp = np.eye(2, 3, dtype=np.float32)

    h, w = frame.shape[:2]
    stabilized = cv2.warpAffine(
        frame,
        warp,
        (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
    return stabilized, to_gray(stabilized)


def denoise_frame(frame):
    if not ENABLE_DENOISE:
        return frame
    return cv2.fastNlMeansDenoisingColored(
        frame,
        None,
        DENOISE_H,
        DENOISE_H_COLOR,
        DENOISE_TEMPLATE_WINDOW,
        DENOISE_SEARCH_WINDOW,
    )


def normalize_illumination(frame):
    if not ENABLE_ILLUMINATION_NORM:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l = clahe.apply(l)

    if ENABLE_SHADOW_FLATTEN:
        blur = cv2.GaussianBlur(l, (0, 0), SHADOW_FLATTEN_SIGMA)
        l_float = l.astype(np.float32)
        blur_float = blur.astype(np.float32)
        l_flat = (l_float / (blur_float + 1.0)) * 128.0
        l = cv2.normalize(l_flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def sharpen_frame(frame):
    if not ENABLE_SHARPEN:
        return frame
    blurred = cv2.GaussianBlur(frame, (0, 0), SHARPEN_SIGMA)
    return cv2.addWeighted(frame, 1.0 + SHARPEN_AMOUNT, blurred, -SHARPEN_AMOUNT, 0)


def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"Error: '{INPUT_DIR}' not found.")
        return

    files = list_frames(INPUT_DIR)
    if not files:
        print(f"Error: No .jpg frames found in '{INPUT_DIR}'.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize video writer if needed
    video_writer = None

    prev_gray = None
    for idx, path in enumerate(files, start=1):
        frame = cv2.imread(path)
        if frame is None:
            print(f"Warning: Could not read '{path}'. Skipping.")
            continue

        stabilized, prev_gray = stabilize_frame(frame, prev_gray)
        denoised = denoise_frame(stabilized)
        normalized = normalize_illumination(denoised)
        restored = sharpen_frame(normalized)

        # Lazy-init video writer
        if EXPORT_VIDEO and video_writer is None:
            h, w = restored.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, OUTPUT_VIDEO_FPS, (w, h))

        if video_writer is not None:
            video_writer.write(restored)

        out_name = os.path.join(OUTPUT_DIR, os.path.basename(path))
        cv2.imwrite(out_name, restored)

        print(f"[{idx}/{len(files)}] Restored -> {out_name}")

    if video_writer is not None:
        video_writer.release()

    print(f"Done. Restored frames saved to '{OUTPUT_DIR}/'.")
    if EXPORT_VIDEO:
        print(f"Preview video saved to '{OUTPUT_VIDEO_PATH}'.")


if __name__ == "__main__":
    main()
