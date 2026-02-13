"""
Phase 3: Porthole detection without training.
1. Filter frame to separate porthole from road.
2. Binary segmentation: porthole = black, road = white.
3. Draw bounding boxes around portholes.

Uses SoTA classical methods: Otsu/adaptive threshold, morphology, contour analysis,
optional Hough circle detection (OpenCV). No model training.
"""
from pathlib import Path
import argparse
import sys
from typing import List, Tuple

import numpy as np
import cv2

# Load config from phase_3 directory so script works when run from any cwd
_phase3_dir = Path(__file__).resolve().parent
if str(_phase3_dir) not in sys.path:
    sys.path.insert(0, str(_phase3_dir))
import config as cfg


def _ensure_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def apply_filter(gray: np.ndarray, config: dict) -> np.ndarray:
    """Filter to improve porthole vs road separation (no binary yet)."""
    out = gray.copy()
    if config.get("blur_ksize"):
        k = config["blur_ksize"]
        out = cv2.GaussianBlur(out, k, 0)
    if config.get("use_clahe"):
        clahe = cv2.createCLAHE(
            clipLimit=config.get("clahe_clip", 2.0),
            tileGridSize=config.get("clahe_grid", (8, 8)),
        )
        out = clahe.apply(out)
    return out


def to_binary(gray: np.ndarray, config: dict) -> np.ndarray:
    """
    Binary image: porthole = black (0), road = white (255).
    Uses Otsu or adaptive threshold (no training).
    """
    method = config.get("method", "otsu")
    porthole_is_darker = config.get("porthole_is_darker", True)

    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        block = config.get("adaptive_block", 31)
        if block % 2 == 0:
            block += 1
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, config.get("adaptive_c", 8)
        )

    # We want porthole = 0 (black), road = 255 (white).
    # Otsu/adaptive typically give: foreground (darker) = 0 or 255 depending on image.
    # If porthole is darker and binary has porthole as 255, invert.
    if porthole_is_darker and np.median(binary) > 127:
        binary = 255 - binary
    elif not porthole_is_darker and np.median(binary) < 127:
        binary = 255 - binary
    return binary


def remove_small_black_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    """
    Remove black (porthole) connected components smaller than min_area.
    Keeps only large black regions (real portholes); small specks become white (road).
    """
    if min_area <= 0:
        return binary
    # Portholes are 0 (black). ConnectedComponents expects non-zero as foreground.
    inv = 255 - binary
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    out = binary.copy()
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            out[labels == i] = 255  # turn small black regions into road (white)
    return out


def apply_morphology(binary: np.ndarray, config: dict) -> np.ndarray:
    """Clean binary mask: close holes inside portholes, open to remove small black specks."""
    if not config.get("enabled", True):
        return binary
    out = binary.copy()
    ck = config.get("close_kernel", (5, 5))
    ok = config.get("open_kernel", (15, 15))
    kernel_close = np.ones(ck, np.uint8)
    kernel_open = np.ones(ok, np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel_close)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel_open)
    min_black = config.get("min_black_area", 0)
    out = remove_small_black_components(out, min_black)
    return out


def contour_circularity(contour: np.ndarray) -> float:
    """4 * pi * area / perimeter^2; 1.0 for a circle."""
    area = cv2.contourArea(contour)
    if area <= 0:
        return 0.0
    perim = cv2.arcLength(contour, True)
    if perim <= 0:
        return 0.0
    return 4 * np.pi * area / (perim * perim)


def get_contour_boxes(
    binary: np.ndarray,
    config: dict,
) -> List[Tuple[int, int, int, int]]:
    """
    Find contours of black (porthole) regions; filter by area and circularity;
    return list of (x, y, w, h) bounding boxes.
    """
    # OpenCV: findContours expects white = foreground; our portholes are black.
    # So find contours on inverted image (portholes become white).
    inv = 255 - binary
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = binary.shape[:2]
    image_area = w * h
    max_area = image_area * config.get("max_area_ratio", 0.4)
    min_area = config.get("min_area", 200)
    min_circ = config.get("min_circularity", 0.4)
    max_aspect = config.get("max_aspect_ratio", 2.0)
    min_distance = config.get("min_porthole_distance", 0)  # diagonal < this = noise

    boxes = []  # type: List[Tuple[int, int, int, int]]
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        if contour_circularity(c) < min_circ:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw <= 0 or bh <= 0:
            continue
        # Distance between opposite corners of the bounding box (diagonal)
        diagonal = np.sqrt(bw * bw + bh * bh)
        if min_distance > 0 and diagonal < min_distance:
            continue  # treat as noise
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect > max_aspect:
            continue
        boxes.append((int(x), int(y), int(bw), int(bh)))
    return boxes


def get_hough_boxes(
    gray: np.ndarray,
    binary: np.ndarray,
    config: dict,
) -> List[Tuple[int, int, int, int]]:
    """
    Hough circle detection (SoTA for circles without training).
    Returns bounding boxes (x, y, w, h) from (cx, cy, radius).
    """
    if not config.get("enabled", True):
        return []
    h, w = gray.shape[:2]
    min_dim = min(h, w)
    min_dist = max(10, int(min_dim * config.get("min_dist_ratio", 0.05)))
    r_min = max(5, int(min_dim * config.get("radius_min_ratio", 0.03)))
    r_max = int(min_dim * config.get("radius_max_ratio", 0.35))
    if r_max <= r_min:
        r_max = r_min + 20

    # Use smoothed grayscale for Hough
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=config.get("dp", 1.2),
        minDist=min_dist,
        param1=config.get("canny_high", 100),
        param2=config.get("accumulator_threshold", 30),
        minRadius=r_min,
        maxRadius=r_max,
    )
    # Same threshold: diagonal of box must be >= min_porthole_distance (box is 2r x 2r, diagonal = 2r*sqrt(2))
    min_distance = config.get("min_porthole_distance", 0)
    boxes = []  # type: List[Tuple[int, int, int, int]]
    if circles is not None:
        for c in circles[0]:
            cx, cy, r = int(c[0]), int(c[1]), int(np.ceil(c[2]))
            side = 2 * r
            diagonal = side * np.sqrt(2)
            if min_distance > 0 and diagonal < min_distance:
                continue  # treat as noise
            x = max(0, cx - r)
            y = max(0, cy - r)
            boxes.append((x, y, side, side))
    return boxes


def merge_boxes(
    contour_boxes: List[Tuple[int, int, int, int]],
    hough_boxes: List[Tuple[int, int, int, int]],
    iou_thresh: float = 0.3,
) -> List[Tuple[int, int, int, int]]:
    """Merge two lists of boxes; if a Hough box overlaps a contour box (IoU >= thresh), keep one."""
    if not hough_boxes:
        return contour_boxes
    if not contour_boxes:
        return hough_boxes

    def box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, aw, ah = a
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx1, by1, bw, bh = b
        bx2, by2 = bx1 + bw, by1 + bh
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    used_contour = [False] * len(contour_boxes)
    result = list(contour_boxes)
    for hb in hough_boxes:
        overlapping = False
        for i, cb in enumerate(contour_boxes):
            if used_contour[i]:
                continue
            if box_iou(cb, hb) >= iou_thresh:
                overlapping = True
                break
        if not overlapping:
            result.append(hb)
    return result


def detect_portholes(
    img: np.ndarray,
    *,
    filter_config: dict | None = None,
    binary_config: dict | None = None,
    morph_config: dict | None = None,
    contour_config: dict | None = None,
    hough_config: dict | None = None,
    detection_mode: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    Full pipeline: filter -> binary (porthole black, road white) -> morphology
    -> detect boxes. Returns (filtered_gray, binary_mask, list of (x,y,w,h)).
    """
    filter_config = filter_config or cfg.FILTER_CONFIG
    binary_config = binary_config or cfg.BINARY_CONFIG
    morph_config = morph_config or cfg.MORPH_CONFIG
    contour_config = contour_config or cfg.CONTOUR_CONFIG
    hough_config = hough_config or cfg.HOUGH_CONFIG
    detection_mode = detection_mode or cfg.DETECTION_MODE

    gray = _ensure_grayscale(img)
    filtered = apply_filter(gray, filter_config)
    binary = to_binary(filtered, binary_config)
    binary = apply_morphology(binary, morph_config)

    contour_boxes = get_contour_boxes(binary, contour_config)
    hough_cfg = dict(hough_config)
    hough_cfg["min_porthole_distance"] = contour_config.get("min_porthole_distance", 0)
    hough_boxes = get_hough_boxes(filtered, binary, hough_cfg) if hough_config.get("enabled") else []

    if detection_mode == "contours_only":
        boxes = contour_boxes
    elif detection_mode == "hough_only":
        boxes = hough_boxes
    else:
        boxes = merge_boxes(contour_boxes, hough_boxes)

    return filtered, binary, boxes


def draw_boxes(
    img: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on image (BGR)."""
    out = img.copy()
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


def run_on_folder(
    input_dir: Path,
    output_dir: Path,
    *,
    save_binary: bool = True,
    save_boxes: bool = True,
    limit: int = 0,
) -> None:
    """Process all images in input_dir; write binary masks and box overlays to output_dir."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in extensions])
    if limit > 0:
        files = files[:limit]

    (output_dir / "binary").mkdir(parents=True, exist_ok=True)
    (output_dir / "boxes").mkdir(parents=True, exist_ok=True)

    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            print(f"Skip (not read): {path.name}")
            continue
        filtered, binary, boxes = detect_portholes(img)
        name = path.stem + path.suffix
        if save_binary:
            cv2.imwrite(str(output_dir / "binary" / name), binary)
        if save_boxes:
            overlay = draw_boxes(img, boxes)
            cv2.imwrite(str(output_dir / "boxes" / name), overlay)
        print(f"  {path.name} -> {len(boxes)} porthole(s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: Filter (porthole vs road), binary (black/white), bounding boxes. No training."
    )
    parser.add_argument("--input", "-i", default="images", help="Input directory (original or enhanced images)")
    parser.add_argument("--output", "-o", default="phase_3_output", help="Output directory")
    parser.add_argument("--limit", "-n", type=int, default=0, help="Process only first N images (0 = all)")
    parser.add_argument("--no-binary", action="store_true", help="Do not save binary masks")
    parser.add_argument("--no-boxes", action="store_true", help="Do not save box overlay images")
    parser.add_argument("--preview", "-p", action="store_true", help="Show first frame: original, binary, boxes (matplotlib)")
    args = parser.parse_args()

    src = Path(args.input)
    if not src.is_dir():
        print(f"Input directory not found: {src}")
        return

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted([f for f in src.iterdir() if f.suffix.lower() in extensions])
    if not files:
        print(f"No images in {src}")
        return

    if args.preview:
        import matplotlib.pyplot as plt
        path = files[0]
        img = cv2.imread(str(path))
        if img is None:
            print(f"Could not read: {path}")
            return
        filtered, binary, boxes = detect_portholes(img)
        overlay = draw_boxes(img, boxes)
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Input")
        axes[0].axis("off")
        axes[1].imshow(binary, cmap="gray")
        axes[1].set_title("Binary (porthole=black, road=white)")
        axes[1].axis("off")
        axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"Bounding boxes ({len(boxes)})")
        axes[2].axis("off")
        plt.suptitle(path.name)
        plt.tight_layout()
        plt.show()
        return

    dst = Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)
    run_on_folder(
        src,
        dst,
        save_binary=not args.no_binary,
        save_boxes=not args.no_boxes,
        limit=args.limit,
    )
    print(f"Done. Output: {dst}")


if __name__ == "__main__":
    main()
