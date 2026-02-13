"""
Phase 3 config: filtering, binary segmentation, and porthole detection.
SoTA classical methods (no model training): Otsu/adaptive threshold, morphology,
contour analysis, optional Hough circle detection.
"""

# --- Filtering: separate porthole (darker) from road (lighter) ---
FILTER_CONFIG = {
    # Pre-threshold: grayscale + optional contrast (CLAHE) for better separation
    "use_clahe": True,
    "clahe_clip": 2.0,
    "clahe_grid": (8, 8),
    # Light blur to reduce noise before threshold (keeps edges)
    "blur_ksize": (3, 3),
}

# --- Binary: porthole = black (0), road = white (255) ---
BINARY_CONFIG = {
    # "otsu" = Otsu's automatic threshold; "adaptive" = adaptive Gaussian
    "method": "otsu",
    # For adaptive: block size (odd), C subtract constant
    "adaptive_block": 31,
    "adaptive_c": 8,
    # Invert so that darker (porthole) becomes black; set False if your data is opposite
    "porthole_is_darker": True,
}

# --- Morphology: clean binary mask (remove small holes/specks) ---
MORPH_CONFIG = {
    "enabled": True,
    # Close small gaps inside portholes
    "close_kernel": (5, 5),
    # Open = erode then dilate: removes small BLACK specks; keep smaller so real portholes remain
    "open_kernel": (9, 9),
    # Remove black connected components smaller than this (pixels); 0 = disable
    "min_black_area": 400,
}

# --- Contour-based detection (primary): find black regions, filter by shape/size ---
CONTOUR_CONFIG = {
    # Minimum diagonal (pixels) of bounding box; smaller = noise.
    "min_porthole_distance": 40,
    # Min contour area (pixels); small contours = noise.
    "min_area": 500,
    "max_area_ratio": 0.4,  # max area as fraction of image area
    # Circularity: 4*pi*area/perimeter^2; 1 = circle. Allow elongated portholes.
    "min_circularity": 0.25,
    # Aspect ratio of bounding box: avoid very elongated blobs
    "max_aspect_ratio": 2.5,
}

# --- Optional: Hough circle detection (SoTA for circles without training) ---
HOUGH_CONFIG = {
    "enabled": False,  # Off by default; texture can trigger many small circles
    # OpenCV HoughCircles: dp, minDist, param1 (Canny high), param2 (accumulator)
    "dp": 1.2,
    "min_dist_ratio": 0.05,  # min distance between centers as fraction of min dimension
    "canny_high": 100,
    "accumulator_threshold": 30,
    # Radius range as fraction of image min dimension (e.g. 0.05–0.25)
    "radius_min_ratio": 0.03,
    "radius_max_ratio": 0.35,
}

# --- Merge strategy: use contours, and optionally refine with Hough circles ---
# "contours_only" | "hough_only" | "contours_and_hough" (merge both)
DETECTION_MODE = "contours_and_hough"
