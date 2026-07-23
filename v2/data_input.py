"""Module 1: Data input (images & Cellpose segmentations).

Loads Cellpose `.npy` segmentation output and extracts per-cell geometric
descriptors. Ported from `gui_main.MainWindow.is_edge_cell` /
`bio_extract_cell_info` (src/gui_main.py:944-1092), stripped of all
PyQt5/`self` coupling:

- The 3-site GUI progress trio (`self.show_progress_bar`,
  `self.progress_dialog.setValue`, `QApplication.processEvents()`) collapses
  into one optional `progress_cb(message, percent)` callback.
- `output_path` is optional: pass `None` to get the DataFrame back with zero
  filesystem side effects (useful for tests/standalone debugging).

Bug Fix #4 (see v2/README.md changelog): the v1 code unconditionally
`print()`s every row containing a -1 sentinel value (from a failed ellipse
fit or zero-perimeter cell) on every single call. That's replaced here with a
guarded `logging.debug()` call that only fires when such rows exist.
"""

import logging
import math
import os
import re
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Full 18-column schema written to Cells_info.xlsx. Order matches
# src/gui_main.py:958-965 exactly (downstream modules rely on this order).
CELLS_INFO_COLUMNS = [
    "Cell Area", "Maximum Horizontal Length", "Maximum Vertical Length", "Horizontal/Vertical Length Ratio",
    "Cell Perimeter", "Approximate Polygon Vertex Count",
    "Fitted Ellipse Minor Axis", "Fitted Ellipse Major Axis", "Ellipse Major/Minor Axis Ratio",
    "Fitted Ellipse Angle", "Circumscribed Circle Radius",
    "Inscribed Circle Radius", "Center X Coordinate", "Center Y Coordinate", "Cell Left Boundary",
    "Cell Right Boundary", "Circularity", "P-Value",
]

# The 13-column subset used as the default PCA/K-means feature set (excludes
# raw position/boundary columns: Center X/Y, Left/Right Boundary). Ported
# from FeatureSelectionDialog.features, src/gui_main.py:96-99.
ML_FEATURE_COLUMNS = [
    "Cell Area", "Fitted Ellipse Major Axis", "Fitted Ellipse Minor Axis", "Ellipse Major/Minor Axis Ratio",
    "Fitted Ellipse Angle", "Cell Perimeter", "Maximum Horizontal Length", "Maximum Vertical Length",
    "Horizontal/Vertical Length Ratio", "Approximate Polygon Vertex Count",
    "Circumscribed Circle Radius", "Inscribed Circle Radius", "Circularity",
]


def _extract_frame_number(filename: str) -> float:
    """Largest run of digits in `filename`; `inf` if none (sorts last), matching
    the nested `extract_number` helper in src/gui_main.py:973-975."""
    match = re.search(r"\d+", filename)
    return int(match.group()) if match else float("inf")


def list_segmentation_files(npy_folder: str) -> list[str]:
    """List `.npy` segmentation filenames in `npy_folder`, sorted by the
    largest embedded frame number and filtered to frame number >= 1. Returns
    filenames only (not full paths), matching src/gui_main.py:970-980."""
    npy_files = [f for f in os.listdir(npy_folder) if f.endswith(".npy")]
    npy_files = sorted(npy_files, key=_extract_frame_number)
    npy_files = [f for f in npy_files if _extract_frame_number(f) >= 1]
    return npy_files


def is_edge_cell(cell_pixels: tuple[np.ndarray, np.ndarray], height: int, width: int, margin: int = 3) -> bool:
    """True if a cell's pixel positions (as returned by `np.where(outlines == cell_id)`,
    i.e. `(row_indices, col_indices)`) come within `margin` pixels of any image
    border. Verbatim port of src/gui_main.py:944-954 (already `self`-free)."""
    min_x = np.min(cell_pixels[1])
    max_x = np.max(cell_pixels[1])
    min_y = np.min(cell_pixels[0])
    max_y = np.max(cell_pixels[0])

    if min_x <= margin or max_x >= width - margin or min_y <= margin or max_y >= height - margin:
        return True
    return False


def extract_cell_info(
    npy_folder: str,
    output_path: Optional[str] = None,
    exclude_edge_cells: bool = False,
    progress_cb: Optional[Callable[[str, int], None]] = None,
) -> tuple[pd.DataFrame, int]:
    """Extract the 18-column geometric feature table for every cell across all
    frames in `npy_folder` (Cellpose `.npy` output, each a dict with an
    `'outlines'` labeled mask). Row index = `Cell Index` string
    `Cell{frame}_{cell_id}`.

    If `output_path` is given, writes `Cells_info.xlsx` (sheet `'Cell'`) there;
    if `None`, only the DataFrame is returned (no filesystem writes).

    `progress_cb(message, percent)` is called once per frame if provided.

    Ported from `bio_extract_cell_info`, src/gui_main.py:956-1092.
    Returns `(cells_info, fig_num)` exactly as v1 did.
    """
    cells_info = pd.DataFrame(columns=CELLS_INFO_COLUMNS)
    fig_num = 0

    npy_files = list_segmentation_files(npy_folder)
    total_files = len(npy_files)
    frame_number = 0

    for idx, filename in enumerate(npy_files):
        basename, ext = os.path.splitext(filename)
        if ext != ".npy":
            continue
        dat = np.load(os.path.join(npy_folder, filename), allow_pickle=True).item()
        outlines = dat["outlines"]
        height, width = outlines.shape
        fig_num += 1
        cells_info_fig = pd.DataFrame(columns=CELLS_INFO_COLUMNS)
        regex = re.compile(r"\d+")
        nums = regex.findall(filename)
        fig_id = str(max(map(int, nums))) if nums else str(idx + 1)
        frame_number = int(fig_id) if fig_id else idx + 1

        if progress_cb is not None:
            percent = int((idx + 1) / total_files * 100) if total_files else 100
            progress_cb(f"Extracting cell features for frame {frame_number}...", percent)

        unique_ids = np.unique(outlines)
        unique_ids = unique_ids[unique_ids != 0]

        for cell_id in unique_ids:
            positions = np.where(outlines == cell_id)

            if exclude_edge_cells and is_edge_cell(positions, height, width, margin=3):
                continue

            contour_points = np.array([positions[1], positions[0]]).T.reshape(-1, 1, 2)

            fill_img = np.zeros_like(outlines, dtype=np.uint8)
            cv2.fillPoly(fill_img, [contour_points], 1)
            cell_area = np.sum(fill_img)

            contours, _ = cv2.findContours(fill_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                filled_contour = contours[0]
                cell_arclength = int(cv2.arcLength(filled_contour, True))
                vertices = cv2.approxPolyDP(filled_contour, 2, True)

                _, radius = cv2.minEnclosingCircle(filled_contour)
                min_enclosing_circle_radius = int(radius)

                dist_transform = cv2.distanceTransform(fill_img, cv2.DIST_L2, 5)
                _, max_val, _, _ = cv2.minMaxLoc(dist_transform)
                max_inner_circle_radius = int(max_val)

                if cell_arclength > 0:
                    circularity = (4 * math.pi * cell_area) / (cell_arclength ** 2)
                    p_value = (4 * math.pi) / (circularity ** 2)
                else:
                    circularity = -1
                    p_value = -1
            else:
                vertices = []
                min_enclosing_circle_radius = 0
                max_inner_circle_radius = 0
                cell_arclength = -1
                circularity = -1
                p_value = -1

            if len(contour_points) >= 5:
                cell_ellipse = cv2.fitEllipse(contour_points)
                ratio_long_and_short_axes = (
                    cell_ellipse[1][1] / cell_ellipse[1][0] if cell_ellipse[1][0] != 0 else -1
                )
                cell_ellipse_angle = 180 - cell_ellipse[2] if cell_ellipse[2] > 90 else cell_ellipse[2]
            else:
                cell_ellipse = None
                ratio_long_and_short_axes = -1
                cell_ellipse_angle = -1

            cy = int(np.mean(positions[0]))
            cx = int(np.mean(positions[1]))

            ap = np.max(positions[1]) - np.min(positions[1])
            dv = np.max(positions[0]) - np.min(positions[0])
            aspect_ratio = ap / dv if dv != 0 else -1

            cell_info = [
                cell_area, ap, dv, aspect_ratio, cell_arclength, len(vertices),
                cell_ellipse[1][0] if cell_ellipse is not None else -1,
                cell_ellipse[1][1] if cell_ellipse is not None else -1,
                ratio_long_and_short_axes, cell_ellipse_angle,
                min_enclosing_circle_radius, max_inner_circle_radius,
                cx, cy, np.min(positions[1]), np.max(positions[1]), circularity, p_value,
            ]
            cell_info = np.round(cell_info, 2)
            cell_index = f"Cell{fig_id}_{cell_id}"
            cell_info_df = pd.DataFrame([cell_info], index=[cell_index], columns=CELLS_INFO_COLUMNS)
            cells_info_fig = pd.concat([cells_info_fig, cell_info_df])

        cells_info = pd.concat([cells_info, cells_info_fig])

    invalid_rows = cells_info[(cells_info == -1).any(axis=1)]
    if not invalid_rows.empty:
        logger.debug(
            "%d cell(s) have -1 sentinel values (ellipse-fit or zero-perimeter fallback): %s",
            len(invalid_rows), list(invalid_rows.index),
        )

    cells_info.index.name = "Cell Index"

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        cells_info.to_excel(os.path.join(output_path, "Cells_info.xlsx"), sheet_name="Cell", index=True)

    if progress_cb is not None:
        progress_cb(f"Cell feature extraction for frame {frame_number} completed...", 100)

    return cells_info, fig_num


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standalone debug entry for v2.data_input")
    parser.add_argument("--npy-folder", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--exclude-edge-cells", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    def _print_progress(message: str, percent: int) -> None:
        print(f"[{percent:3d}%] {message}")

    df, n_frames = extract_cell_info(
        args.npy_folder, args.output, args.exclude_edge_cells, progress_cb=_print_progress
    )
    print(f"\nProcessed {n_frames} frame(s); extracted {len(df)} cells total.")
    print(df.head())
    invalid = df[(df == -1).any(axis=1)]
    print(f"\n{len(invalid)} row(s) contain -1 sentinel values.")
