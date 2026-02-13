"""Phase 3: Porthole filtering, binary segmentation, and bounding box detection (no training)."""
from .detect_portholes import (
    detect_portholes,
    draw_boxes,
    run_on_folder,
)

__all__ = ["detect_portholes", "draw_boxes", "run_on_folder"]
