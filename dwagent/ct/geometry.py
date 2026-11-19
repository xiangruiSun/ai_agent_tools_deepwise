"""Shared measurement code so that reader contours and detector masks are measured the same way."""
from __future__ import annotations

import numpy as np


def axes_mm(mask2d: np.ndarray, dy: float, dx: float) -> tuple[float, float]:
    """Long and short axis of a 2D region in mm, from the principal axes of its pixels.

    For a filled ellipse the full axis length is 4 standard deviations along that axis, which
    is what is returned; one pixel is added so a single-pixel region is not reported as 0 mm.
    """
    ys, xs = np.nonzero(mask2d)
    if len(ys) == 0:
        return 0.0, 0.0
    if len(ys) < 3:
        return max(dy, dx) * len(ys), min(dy, dx)
    pts = np.stack([ys * dy, xs * dx], axis=1)
    cov = np.cov(pts.T)
    evals = np.sort(np.clip(np.linalg.eigvalsh(cov), 0, None))[::-1]
    pix = (dy + dx) / 2
    return float(4 * np.sqrt(evals[0]) + pix), float(4 * np.sqrt(evals[1]) + pix)


def mean_diameter(long_mm: float, short_mm: float) -> float:
    """Lung-RADS mean diameter: average of long and short axis, to one decimal."""
    return round((long_mm + short_mm) / 2, 1)


def laterality(x_index: float, n_cols: int) -> str:
    """Head-first supine axial images: the patient's right is on the image's left."""
    return "right" if x_index < n_cols / 2 else "left"


def zone(z_mm: float, lung_z_min: float, lung_z_max: float) -> str:
    if lung_z_max <= lung_z_min:
        return "mid"
    f = (z_mm - lung_z_min) / (lung_z_max - lung_z_min)
    return "upper" if f > 2 / 3 else "lower" if f < 1 / 3 else "mid"
