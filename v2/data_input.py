"""Module 1: Data input (images & segmentations).

Loads segmentation output and extracts per-cell geometric descriptors.
`extract_cell_info` was ported from `gui_main.MainWindow.is_edge_cell` /
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

Non-Cellpose input formats (reviewer-requested extension, not in v1)
---------------------------------------------------------------------
v1 only reads Cellpose's `.npy` output (a dict with an `'outlines'` labeled
mask). A peer reviewer pointed out that other commonly-used epithelial
segmentation tools (e.g. EpySeg) export masks or boundary *skeletons* as
plain `.tif`/`.tiff` images instead, and asked for the pipeline to accept a
wider range of input formats. `extract_cell_info` itself is unchanged (it
still just walks a folder of Cellpose-style `.npy` files) -- the new
`detect_mask_format` / `load_tiff_mask` / `convert_tiff_masks_to_npy` /
`prepare_segmentation_folder` functions below are a composable pre-processing
layer: they convert a folder of `.tif`/`.tiff` masks into the same `.npy`
dict convention first, so the existing (already-validated) extraction logic
never needs to know where the segmentation came from. See
`prepare_segmentation_folder` for the one-call "just get me an npy folder"
entry point, and v2/README.md for the full design rationale (including why
distinguishing a boundary skeleton from a filled cell-body mask needs more
than just counting pixel values).
"""

import logging
import math
import os
import re
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt as ndimage_distance_transform_edt
from scipy.ndimage import find_objects as ndimage_find_objects
from scipy.ndimage import label as ndimage_label
from scipy.ndimage import maximum_filter as ndimage_maximum_filter
from scipy.ndimage import minimum_filter as ndimage_minimum_filter

logger = logging.getLogger(__name__)

# 18-column schema written to Cells_info.xlsx (17 until Round 26 added
# `Edge Distance`). The two row builders in
# `extract_cell_info` fill it positionally, so they change with it. Tracking
# weights are resolved BY NAME (`cell_tracking.WEIGHTED_COLUMNS`), so the order
# no longer moves them.
#
# Three columns from v1's 18 were removed on 2026-08-18 because they confused
# users by appearing in the output while being offered nowhere in Classify.
# Two came back in Round 24 (2026-09-11), in their v1 positions, as
# default-off Classify features -- see `pca_kmeans.FEATURE_FAMILIES["basic"]`:
#   `Fitted Ellipse Angle` -- stored FOLDED (`180 - a if a > 90`, so 0-90), a
#       mirror reflection rather than a modulo, so orientations 40 degrees apart
#       share a value (70 and 110 both stored 20.0). Not invertible, so unusable
#       as a direction; for that use `shape_features`' Sin2Theta / Cos2Theta,
#       recomputed from the elliptic-Fourier harmonic-1 SVD. Restored because
#       as a classifier on an axis-aligned specimen it is strong (single-feature
#       AUC 0.917 on `AS_LE_lateral view_E-cad`,
#       `docs/classification-feature-audit.md`).
#   `Circularity` -- `4*pi*A/P^2`, the same quantity as `Shape Index`. Restored
#       so the basic family can offer it without the shape step.
#   `P-Value` -- `4*pi/circularity**2`, a monotone function of a monotone
#       function of `Shape Index`. The same quantity a third time. Stays removed.
# Neither `Circularity` nor `P-Value` was ever a QC signal in its own right:
# dropping both from `segmentation_qc.measurement_failures` detected exactly the
# same 12 failures across 24 frames of `AS_dorsal view_E-cad`.
CELLS_INFO_COLUMNS = [
    "Cell Area", "Maximum Horizontal Length", "Maximum Vertical Length", "Horizontal/Vertical Length Ratio",
    "Cell Perimeter", "Approximate Polygon Vertex Count",
    "Fitted Ellipse Minor Axis", "Fitted Ellipse Major Axis", "Ellipse Major/Minor Axis Ratio",
    "Fitted Ellipse Angle", "Circumscribed Circle Radius",
    "Inscribed Circle Radius", "Center X Coordinate", "Center Y Coordinate", "Cell Left Boundary",
    "Cell Right Boundary", "Circularity",
    # Round 26 B1. APPENDED LAST, deliberately: `common.save_modified_rows` and
    # the tracking-only path below map columns by POSITION, so inserting
    # anywhere else would silently shift every column after it.
    #
    # A distance, not a flag, so the threshold is not baked into the data: a
    # different `margin`, or a Schmitt trigger over it, needs no re-extraction.
    # Like Center X/Y and the Left/Right Boundaries it LOCATES a cell rather
    # than describing it, which is why it stays out of every feature set (see
    # `ML_FEATURE_COLUMNS`).
    "Edge Distance",
]

# Default PCA/K-means feature set: every column except the raw positions
# (Center X/Y, Left/Right Boundary), which locate a cell rather than describe
# it. This is `pca_kmeans.FEATURE_FAMILIES["basic"]`, whose `default_off`
# decides which of these are ticked by default.
#
# **v1's 13-feature list stays retired.** These are the same 13 columns in v1's
# order, but they would not reproduce v1's numbers, since the switch to
# mask-derived geometry moved perimeter by ~-5% and circularity by ~+10%.
# `src/` remains frozen and is the reference for exact v1 reproduction.
ML_FEATURE_COLUMNS = [
    "Cell Area", "Fitted Ellipse Major Axis", "Fitted Ellipse Minor Axis", "Ellipse Major/Minor Axis Ratio",
    "Fitted Ellipse Angle", "Cell Perimeter", "Maximum Horizontal Length", "Maximum Vertical Length",
    "Horizontal/Vertical Length Ratio", "Approximate Polygon Vertex Count",
    "Circumscribed Circle Radius", "Inscribed Circle Radius", "Circularity",
]

# Background margin added around each cell's bounding box before the per-cell
# raster work in `extract_cell_info`. `cv2.distanceTransform` measures distance
# to the nearest zero pixel, so a crop tight to the cell leaves its edge pixels
# with no zero to measure from. Measured on `AS_dorsal view_E-cad`: margin 0
# corrupts the Inscribed Circle Radius of 260/2822 cells (the saturated
# distance overflows the `int()` cast, max error 65532); margin 1 is already
# enough to reproduce the full-frame result exactly. 8 is kept as cheap
# headroom -- the cost scales with the (small) bbox area, not the frame.
_BBOX_MARGIN = 8

# A cell whose stored `outlines` ring has fewer than this fraction of the
# pixels on its mask's external contour -- image-border pixels not counted --
# is measured from that contour instead.
# The mask is the segmentation; `outlines` is derived from it, and Cellpose's
# GUI does not always re-derive it after a manual edit. On `AS_dorsal
# view_E-cad` (every file carries `manual_changes`) 12 cells kept 1-10 px of
# ring against a 99-240 px mask boundary, so every column taken from the ring
# -- centroid, H/V extent, ellipse fit, left/right boundary -- described one
# pixel, and the ellipse fit failed. Those were the "12 measurement failures"
# `segmentation_qc.measurement_failures` had been reporting on that dataset.
# Border pixels are left out of the comparison because the two ring
# conventions differ exactly there: a Cellpose ring IS the external contour
# (pixel-identical on 3,761 cells checked), while `thin_label_outlines`, which
# writes the TIFF-converted rings, has no ring pixel along the image border. With
# the border counted, `Demo_Seg_QC`'s corner cells read 0.42-0.48 and were
# wrongly rebuilt. Measured ratio over all seven test datasets (43,983 cells):
# those 12 sit at 0.00-0.08; the next-lowest cell anywhere is 0.97, bar one
# puzzle cell at exactly 0.5 (`Cell1564_255`, a 6 px cell plus a stray mask
# pixel -- see `_MIN_ELLIPSE_AXIS_PX`), which this deliberately leaves alone.
_STALE_OUTLINE_FRACTION = 0.5

# Smallest ellipse semi-axis, in pixels, that `cv2.fitEllipse` is trusted to
# have measured. Below this the fit is degenerate and its output is not even
# reproducible.
#
# Found while regression-testing: `extract_cell_info` was not deterministic.
# Four consecutive runs over `cp_dataset_puzzle-like_cells` in one process gave
# different tables, always differing on `Cell1564_255` -- a **6-pixel** cell
# whose major axis came back 7.48 on one run and 15.11 on the next. Confirmed
# pre-existing by reproducing it on the unmodified code.
#
# The reproducibility was the symptom; the damage was to the feature. When the
# minor axis was exactly 0 the old `!= 0` guard caught it, but a tiny nonzero
# axis slipped through and made `Ellipse Major/Minor Axis Ratio` enormous:
# 5443 on `AS_LE_lateral view_E-cad` (median 3.15) and 13421 on the puzzle set.
# One such value inflated that column's standard deviation to 54.65, so after
# `StandardScaler` 99% of cells were compressed into z = [-0.055, 0.134] while
# the artifact sat at z = 99.5 -- real elongation variation (median 3.15,
# p99 10.14) squeezed into a fifth of a standard deviation, leaving PCA almost
# nothing but the artifact on an axis that carries 93% of PC3's loading.
#
# Threshold chosen by sweeping it against determinism over all six test
# datasets, five interleaved runs each:
#
#   1.0 px -> `AS_LE_lateral` still flaky   33/23994 cells sentinelled (0.14%)
#   1.5 px -> deterministic                 35/23994 (0.15%)
#   2.0 px -> deterministic                 39/23994 (0.16%)
#   3.0 px -> deterministic                 52/23994 (0.22%)
#
# 1.0 px is knife-edge: `Cell71_124` (5 px, 2 polygon vertices -- a line of
# pixels) measured a minor axis of 1.00 on one run and 1.01 on the next, so it
# fell on either side of the cut depending on the run. 1.5 px is already
# enough, and 2.0 px buys margin for four more cells out of twenty-four
# thousand. The 1st percentile of genuine minor axes is 7-12 px across these
# datasets, so nothing real is anywhere near the threshold.
_MIN_ELLIPSE_AXIS_PX = 2.0


def _extract_frame_number(filename: str) -> float:
    """The LAST run of digits in `filename`; `inf` if none (sorts last).

    Bug fix (found while testing `test_data/embryos_GBE_GAP43`, whose mask
    filenames look like `Ctrl_GAP43-mChCtrl_GAP43-mCh0000_handCorrection.tif`):
    the original port of v1's nested `extract_number` helper
    (src/gui_main.py:973-975) used `re.search` (the FIRST digit run), which
    for that dataset picks up "43" from the "GAP43" gene name -- appearing
    before the actual zero-padded frame counter -- giving every one of the
    31 frames the same "frame number" and silently collapsing them into one.
    `extract_cell_info`'s own `fig_id` computation had a second, different
    bug in the same direction (used the LARGEST digit run, which also
    resolves to "43" > any of "0000".."0030"). Both are fixed here by using
    the LAST digit run consistently, which is the actual frame counter for
    every filename pattern seen in this project's test datasets (including
    ones with multiple embedded numbers, like a gene name plus a frame
    counter) -- verified against every `test_data/*/images` and `*/masks`
    filename pattern in the repo before making this the default.
    """
    matches = re.findall(r"\d+", filename)
    return int(matches[-1]) if matches else float("inf")


def warn_if_not_a_movie(folder: str, masks_by_frame: Optional[dict] = None) -> Optional[dict]:
    """Warn -- never refuse -- when a folder looks like separate specimens
    rather than a time series.

    Returns `None` when the folder looks like a movie, otherwise
    `{"reason", "message", "detail"}`.

    **Why a warning and not a refusal.** Only the user can distinguish a
    legitimate two-frame movie from a two-specimen folder; the filenames cannot.
    So this informs and never blocks.

    **Why it exists.** `resolve_frame_numbers` assigns frame numbers from
    trailing digits, so every one of these silently becomes "a movie", and each
    then fed a cross-frame operation in this project:

    - `pupae_E-cad`'s `..._P1` / `..._P2` -- two different pupae;
    - `cp_dataset_puzzle-like_cells` -- twelve unrelated images, 10 of whose 11
      consecutive pairs have DIFFERENT dimensions;
    - `Demo_Seg_QC`'s `1_Demo_OE` / `2_Demo_WT` / `3_Demo_WT_handCorrection` --
      two specimens plus a hand-corrected copy of one of them. That last one is
      the dangerous shape: its 2->3 pair has a median best IoU of **1.000**, so
      any tracking score computed on it is ~100% by construction and a broken
      tracker would still pass.

    Differing array shapes is the cheapest and least ambiguous trigger, so it is
    checked first and reported alone when it fires.
    """
    shapes = {}
    if masks_by_frame:
        shapes = {f: tuple(a.shape[:2]) for f, a in masks_by_frame.items() if a is not None}
    if len(set(shapes.values())) > 1:
        listed = ", ".join(f"frame {f}: {h}x{w}" for f, (h, w) in sorted(shapes.items())[:6])
        return {
            "reason": "shape_mismatch",
            "message": (f"{os.path.basename(folder) or folder}: frames have different image "
                        f"dimensions, so this is probably a set of separate images rather than a "
                        f"movie. Cross-frame operations (tracking, T1, drift) are not meaningful "
                        f"here; per-frame ones (extraction, QC, shape) are fine."),
            "detail": listed,
        }

    stems = [os.path.splitext(n)[0] for n in data_input_listing(folder)]
    if len(stems) > 1:
        prefix = os.path.commonprefix(stems)
        # A movie's names share a long prefix and differ only in a counter. Two
        # specimens share little, or differ by more than digits.
        tails = [t[len(prefix):] for t in stems]
        if any(not re.fullmatch(r"\d*[^0-9]*\d*", t) for t in tails) and len(prefix) < 3:
            return {
                "reason": "unrelated_stems",
                "message": (f"{os.path.basename(folder) or folder}: filenames do not look like a "
                            f"shared prefix plus an incrementing index, so this may be a set of "
                            f"separate images rather than a movie."),
                "detail": ", ".join(stems[:6]),
            }
    return None


def data_input_listing(folder: str) -> list:
    """Segmentation filenames in `folder`, for the movie-shape guard.

    Deliberately not `list_segmentation_files`: that one lists only `.npy` and
    returns nothing at all for a TIFF project, which would make the guard
    silently pass on exactly the datasets it exists to catch.
    """
    if not folder or not os.path.isdir(folder):
        return []
    return sorted(
        name for name in os.listdir(folder)
        if os.path.splitext(name)[1].lower() in (".npy", ".tif", ".tiff", ".png")
    )


def resolve_frame_numbers(filenames: list[str]) -> dict[str, int]:
    """Map each filename to its frame number, falling back to 1-based
    position when the filenames carry no digits at all.

    `_extract_frame_number` returns `inf` for a digit-less name, which is
    fine for sorting but useless as an identity. Datasets of *unrelated
    samples* rather than a time series have exactly that shape --
    `test_data/Demo_Seg_QC` holds `WT_mask.tif`/`OE_mask.tif` with no
    counter anywhere -- and every consumer that looked a frame up by number
    (`_npy_path_for_frame`, `_find_matching_image`) silently found nothing,
    so previews came back blank. `extract_cell_info` meanwhile had its own
    private `str(idx + 1)` fallback, so it happily emitted `Cell1_*`/`Cell2_*`
    rows that nothing else could resolve back to a file. This helper is the
    single place that decision is made, so those two can no longer disagree.

    Numbering is all-or-nothing on purpose: if *every* name has digits the
    embedded numbers win (the normal time-series case, unchanged), otherwise
    *all* names are numbered positionally. Mixing the two per-file would let
    a positional 1 collide with a digit-derived 1 and silently merge two
    different frames.

    Ordering is `(embedded number, lowercased name)` so the positional case
    is deterministic -- sorting digit-less names by `inf` alone leaves them
    in `os.listdir` order, which is arbitrary and would let frames 1 and 2
    swap identity between runs on the same folder.
    """
    ordered = sorted(filenames, key=lambda f: (_extract_frame_number(f), f.lower()))
    if all(_extract_frame_number(f) != float("inf") for f in ordered):
        return {f: int(_extract_frame_number(f)) for f in ordered}
    return {f: i + 1 for i, f in enumerate(ordered)}


def list_segmentation_files(npy_folder: str) -> list[str]:
    r"""List `.npy` segmentation filenames in `npy_folder`, ordered by frame
    number. Returns filenames only (not full paths). **Every `.npy` in the
    folder comes back** -- the `>= 0` test below filters nothing, because
    `_extract_frame_number` reads its number with `re.findall(r"\d+")` and
    can never return a negative one.

    It used to read `>= 1`, copied from src/gui_main.py:970-980, where it
    guarded against a digit-less filename falling back to `inf`. That case
    has since been taken over by `resolve_frame_numbers`' all-or-nothing
    positional numbering, and `inf >= 1` passed the guard anyway -- so the
    only thing the filter still did was drop **frame 0**.

    It dropped one on real data: `test_data/embryos_GBE_GAP43` numbers its
    masks from `handCorrection0000.tif`, and `0 >= 1` is false, so that frame
    never entered extraction, tracking or any downstream count. Silently: no
    warning, and the frame count simply read one lower (30, not 31).

    The comparison is kept rather than deleted so this docstring has
    something to hang on, and the assertion below is the check that actually
    matters -- all four consumers share this function, so they cannot
    disagree with each other about a dropped frame; only the directory can."""
    npy_files = [f for f in os.listdir(npy_folder) if f.endswith(".npy")]
    frame_numbers = resolve_frame_numbers(npy_files)
    kept = [f for f in frame_numbers if frame_numbers[f] >= 0]
    if len(kept) != len(npy_files):
        dropped = sorted(set(npy_files) - set(kept))
        raise AssertionError(
            "list_segmentation_files dropped %d of %d .npy file(s) in %s: %s"
            % (len(npy_files) - len(kept), len(npy_files), npy_folder,
               ", ".join(dropped))
        )
    return kept


def detect_mask_format(mask_folder: str) -> str:
    """Inspect `mask_folder` and report which segmentation format it holds:
    `'npy'` (Cellpose-style, at least one `.npy` and no `.tif`/`.tiff`),
    `'tiff'` (at least one `.tif`/`.tiff` and no `.npy`), `'mixed'` (both --
    ambiguous, caller should treat as an error), or `'empty'` (neither).
    """
    files = os.listdir(mask_folder)
    has_npy = any(f.endswith(".npy") for f in files)
    has_tiff = any(f.lower().endswith((".tif", ".tiff")) for f in files)
    if has_npy and has_tiff:
        return "mixed"
    if has_npy:
        return "npy"
    if has_tiff:
        return "tiff"
    return "empty"


def _collapse_to_label_array(mask_arr: np.ndarray) -> np.ndarray:
    """Collapse a loaded TIFF array (`(H, W)` grayscale/labeled, or `(H, W, C)`
    RGB/RGBA) into a single 2D integer array. A grayscale/single-channel
    array is used directly. An RGB array where every pixel's channels are
    identical (i.e. it's really grayscale saved as RGB) is collapsed to one
    channel. A genuinely multi-color RGB array (e.g. a hand-labeled QC export
    where each cell was painted a distinct color) is treated as an
    already-labeled instance mask: each unique color becomes one integer
    label, with the single most frequent color treated as background (0).
    """
    if mask_arr.ndim == 2:
        return mask_arr.astype(np.int64)

    if mask_arr.ndim == 3:
        channels_equal = np.all(mask_arr == mask_arr[..., :1], axis=-1)
        if channels_equal.all():
            return mask_arr[..., 0].astype(np.int64)

        flat = mask_arr.reshape(-1, mask_arr.shape[-1])
        colors, inverse = np.unique(flat, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        background_color_idx = np.argmax(counts)

        remap = np.zeros(len(colors), dtype=np.int64)
        next_label = 1
        for color_idx in range(len(colors)):
            if color_idx == background_color_idx:
                continue
            remap[color_idx] = next_label
            next_label += 1

        return remap[inverse].reshape(mask_arr.shape[:2])

    raise ValueError(f"Unsupported mask array shape {mask_arr.shape}; expected 2D or 3D (H, W[, C]).")


def _fragmentation_ratio(binary_mask: np.ndarray) -> float:
    """How many more pieces `binary_mask` breaks into under 4-connectivity
    than under 8-connectivity. A thin, 1-pixel-wide boundary/skeleton
    network fragments heavily going from 8- to 4-connectivity (diagonal-only
    junctions get cut), while solid cell-body blobs do not. Used by
    `_resolve_skeleton_classes` to tell the two apart without relying on
    which class happens to have fewer pixels (a skeleton isn't always the
    minority class, and a mask of small/sparse cells isn't always the
    majority)."""
    if not binary_mask.any():
        return 0.0
    _, n4 = ndimage_label(binary_mask)
    _, n8 = ndimage_label(binary_mask, structure=np.ones((3, 3), dtype=int))
    return n4 / max(n8, 1)


def _resolve_skeleton_classes(class_a: np.ndarray, class_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given the two complementary binary masks of a 2-value TIFF, decide
    which one is the cell-body region and which is the boundary/skeleton
    network, and return `(cell_region_mask, boundary_mask)`. See
    `_fragmentation_ratio` for the underlying topology check.
    """
    if _fragmentation_ratio(class_a) > _fragmentation_ratio(class_b):
        return class_b, class_a
    return class_a, class_b


# A median needs enough components to mean anything; below this the filter
# stands down rather than guessing what "typical" is.
_MIN_COMPONENTS_FOR_MEDIAN = 5


def _drop_oversized_components(labeled: np.ndarray, background_area_ratio: float) -> np.ndarray:
    """Relabel to background (0) any connected component whose area exceeds
    `background_area_ratio` times the MEDIAN component area.

    Connected-component labeling of a skeleton's complement doesn't know which
    components are real cells and which are unsegmented "outside the tissue"
    background -- if the boundary network doesn't fully enclose the image,
    everything outside the tissue merges into one giant blob that dwarfs
    typical cells.

    **This used to compare against the MAXIMUM component area, which made it
    unsatisfiable.** The runaway background component IS the maximum, so
    `areas > ratio * max_area` can never hold for any `ratio >= 1`, and the
    function changed nothing at the default of 10.0 -- measured on all 31
    frames of `test_data/embryos_GBE_GAP43`: zero pixels altered. The median
    is what the caller-facing docstring in `load_tiff_mask` always promised,
    and it separates cleanly here: that dataset's background component runs
    78-101x the median cell area in every frame, so a default of 10 catches it
    with a wide margin on both sides.

    The consequences of leaving it in were not cosmetic. The giant label
    survived into `masks`, so it was rendered, measured into
    `Cells_info.xlsx` as a ~79x-median "cell", fed to the neighbour graph --
    and, worst, to tracking, where its 426px extent made the displacement gate
    vacuous and its circumscribed radius set the global matching sentinel.
    Removing it cut mis-tracked links on that dataset from 23.6% to 1.6%.
    """
    areas = np.bincount(labeled.ravel())
    areas[0] = 0
    nonzero_areas = areas[areas > 0]
    if len(nonzero_areas) < _MIN_COMPONENTS_FOR_MEDIAN:
        return labeled

    median_area = float(np.median(nonzero_areas))
    if median_area <= 0:
        return labeled

    oversized_labels = np.nonzero(areas > background_area_ratio * median_area)[0]
    if len(oversized_labels) == 0:
        return labeled

    logger.info(
        "Dropped %d oversized component(s) as unsegmented background "
        "(area > %.1fx the median cell area of %.0f px): sizes %s",
        len(oversized_labels), background_area_ratio, median_area,
        sorted(int(areas[l]) for l in oversized_labels),
    )
    result = labeled.copy()
    result[np.isin(labeled, oversized_labels)] = 0
    return result


# A drawn cell boundary is 1-3 px wide, so every one of its pixels lies within
# a couple of pixels of a cell. Anything further away is not a separator
# between two cells -- see `_close_label_gaps`.
_MAX_GAP_FILL_PX = 3


def _close_label_gaps(
    labeled: np.ndarray, fillable: np.ndarray, max_distance_px: int = _MAX_GAP_FILL_PX,
) -> np.ndarray:
    """Assign every `fillable` pixel to the nearest labeled cell, so cells
    that a drawn boundary line separated end up directly adjacent (Cellpose's
    convention, where the border between two cells is just a label change).

    `scipy.ndimage.distance_transform_edt(..., return_indices=True)` gives the
    coordinates of the nearest non-background pixel for the whole image in one
    pass, which is the nearest-label assignment we want. (`skimage`'s
    `expand_labels` would be the obvious tool, but scikit-image is not
    installed and is referenced nowhere in this repo -- scipy already is.)

    `fillable` should be only the pixels known to be a separator, and even
    then only those actually lying BETWEEN cells -- `max_distance_px` enforces
    that second half.

    Without the distance bound this silently teleports far-away separator
    pixels onto whichever cell happens to be nearest. Measured on
    `test_data/embryos_GBE_GAP43`, whose TIFF masks carry a closed line around
    the entire image perimeter: after `_drop_oversized_components`
    correctly deletes the unsegmented band inside that line, all 1,468
    perimeter pixels (exactly `2*(H+W)`) were handed to peripheral cells a
    median of 17 px away. That gave 1,303 cells a second, disconnected
    fragment sitting on row 0 -- and since `is_edge_cell` is bounding-box
    based, one such pixel condemned the whole cell, so `exclude_edge_cells`
    then dropped 17.5% of every frame before tracking ever saw them. (Since
    Round 26 that filter no longer drops rows at all, but the bounding-box
    sensitivity this paragraph describes is why the pixels must not be
    teleported in the first place.)

    A real drawn boundary is 1-3 px wide, so each of its pixels is within a
    couple of pixels of the cell it belongs to; the default bound closes
    genuine gaps while leaving an orphaned perimeter line where it is.

    **That perimeter line is a property of the TIFF input path, not of anyone's
    curation.** Round 7 attributed it to hand-correction; checked across this
    project's datasets, both TIFF-mask datasets have a complete image border
    (`embryos_GBE_GAP43` 1,468 px = 2(300+434); `Demo_Seg_QC` 836 px =
    2(209+209)) and none of the three native-Cellpose `.npy` datasets has one
    at all. A skeleton/outline TIFF writes the image edge as part of the
    boundary network, so the line arrives with the file. `_handCorrection` in
    those filenames is the exporting tool's naming convention and is NOT
    evidence that a human edited anything.
    """
    if not fillable.any():
        return labeled
    distance, (nearest_y, nearest_x) = ndimage_distance_transform_edt(
        labeled == 0, return_indices=True,
    )
    fillable = fillable & (distance <= max_distance_px)
    if not fillable.any():
        return labeled
    closed = labeled.copy()
    closed[fillable] = labeled[nearest_y[fillable], nearest_x[fillable]]
    return closed


def thin_label_outlines(labeled: np.ndarray) -> np.ndarray:
    """The 1-px inner boundary ring of each cell, keeping that cell's label
    and zeroing its interior -- i.e. Cellpose's `'outlines'` convention.

    A pixel is on the ring when it belongs to a cell and any 4-neighbour
    carries a different label (another cell, or background). Computed with
    array shifts rather than a per-cell erosion so it costs one pass over the
    image regardless of cell count.
    """
    differs = np.zeros(labeled.shape, dtype=bool)
    differs[:-1, :] |= labeled[:-1, :] != labeled[1:, :]
    differs[1:, :] |= labeled[1:, :] != labeled[:-1, :]
    differs[:, :-1] |= labeled[:, :-1] != labeled[:, 1:]
    differs[:, 1:] |= labeled[:, 1:] != labeled[:, :-1]

    ring = np.zeros_like(labeled)
    on_ring = differs & (labeled > 0)
    ring[on_ring] = labeled[on_ring]
    return ring


def _vertex_pockets(labeled: np.ndarray) -> np.ndarray:
    """Pixels of every label that has NO interior pixel -- none whose whole
    3x3 neighbourhood is that label (a pixel on the image edge never
    qualifies, as with `binary_erosion`'s zero border).

    On a skeleton mask these are the background slivers the drawn network
    encloses at a vertex, not cells. Tissue Analyzer leaves them where three
    boundary lines meet: on `embryos_GBE_OE_GAP43` 201 of them (189 at 2 px,
    12 at 3 px) became labels, and at `min_cell_area_px = 3` -- which drops
    `count < 3` -- the twelve 3 px ones reached `Cells_info` with a sentinel
    ellipse and one took a real link in tracking (`Cell25_17 -> Cell26_43`, a
    73 px jump). The test is shape, not size: a pocket is at most 2 px thick
    everywhere, so it cannot hold a 3x3 block, while the smallest real label
    on the three TIFF test datasets (64 px, `Demo_Seg_QC`; 114 px on OE once
    the pockets are gone) has plenty. Measured: exactly those 201 labels and
    nothing else on all three (`embryos_GBE_GAP43`, `embryos_GBE_OE_GAP43`,
    `Demo_Seg_QC`). No pixel count to guess, and it ports across
    magnifications.
    """
    low = ndimage_minimum_filter(labeled, size=3, mode="constant", cval=0)
    high = ndimage_maximum_filter(labeled, size=3, mode="constant", cval=0)
    has_interior = np.zeros(int(labeled.max()) + 1, dtype=bool)
    has_interior[labeled[(low == high) & (labeled > 0)]] = True
    return (labeled > 0) & ~has_interior[labeled]


def _label_binary_mask(
    label_array: np.ndarray, mask_type: str, background_area_ratio: Optional[float] = 10.0,
) -> np.ndarray:
    """Turn a 2-valued (binary) `label_array` into a labeled instance mask
    (each connected cell region gets a unique positive integer, 0 =
    background/boundary), per `mask_type`:

    - `'skeleton'`: one value is a boundary/skeleton network to be discarded;
      the other is split into cells via connected-component labeling. Which
      value is which is resolved automatically (see `_resolve_skeleton_classes`)
      since a skeleton isn't reliably "the minority value" across tools.
    - `'binary_mask'`: the non-background value already directly represents
      cell interiors (no boundary lines to remove); background is the lower
      (darker) value. Each connected foreground region becomes one cell.
    - `'auto'`: same as `'skeleton'` -- for a 2-valued TIFF, a thin boundary
      network is the far more common convention (and what motivated the
      reviewer's request), so auto-detection defaults to it. Pass an
      explicit `mask_type` if your tool's binary masks mean "foreground =
      cell body" instead.

    After labeling, `background_area_ratio` (default 10x the maximum
    component area; pass `None` to disable) discards oversized components --
    see `_drop_oversized_components`. This cleanup only applies to labels
    derived here from a binary image; it is never applied to an
    already-labeled instance mask (`mask_type='labeled'` in `load_tiff_mask`),
    since a genuinely large but valid cell in a trusted pre-existing mask
    shouldn't be second-guessed.
    """
    values = np.unique(label_array)
    if len(values) <= 1:
        return np.zeros_like(label_array, dtype=np.int64)

    class_a = label_array == values[0]
    class_b = label_array == values[1]

    boundary = None
    if mask_type == "binary_mask":
        # The foreground (higher) value is the cell body, matching this
        # mask_type's documented meaning and the UI's own label
        # ("binary_mask (foreground = cell body)").
        #
        # This used to pick whichever class covered FEWER pixels, which is
        # backwards whenever cells outnumber the gaps between them -- and on
        # skeleton-like input it selected the skeleton network itself,
        # labeling 4,668 line fragments as "cells" instead of finding the 269
        # real ones. Cell/boundary disambiguation belongs to the skeleton
        # path (`_resolve_skeleton_classes`), which is what `'auto'` uses;
        # `'binary_mask'` is the explicit "trust me, foreground is cells"
        # escape hatch and should not second-guess the caller.
        cell_region = class_b
    else:  # 'auto' or 'skeleton'
        cell_region, boundary = _resolve_skeleton_classes(class_a, class_b)

    labeled, _ = ndimage_label(cell_region)
    labeled = labeled.astype(np.int64)

    if background_area_ratio is not None:
        labeled = _drop_oversized_components(labeled, background_area_ratio)

    if boundary is not None:
        # A vertex pocket goes back to the network it is a gap in, not to
        # background, so the gap-closing below hands its pixels to the cells
        # around it exactly as it does the drawn line. After the oversize
        # drop, not before: that test is relative to the MEDIAN component area,
        # which removing ~1% of tiny components would nudge. Labels are zeroed,
        # not renumbered, so every other cell keeps its ID.
        pockets = _vertex_pockets(labeled)
        if pockets.any():
            logger.info(
                "Returned %d vertex pocket(s) (labels with no interior pixel) to the skeleton",
                len(np.unique(labeled[pockets])),
            )
            labeled[pockets] = 0
            boundary = boundary | pockets

        # Hand the drawn boundary line back to the cells it separates.
        #
        # Without this, the 1-px skeleton belongs to no cell and every pair of
        # neighbours is left with a permanent gap between them -- measured on
        # test_data/embryos_GBE_GAP43: 8.03% of the image (exactly the
        # skeleton pixel count) sat as background and NOT ONE of the 269 cells
        # was 4-adjacent to another, where native Cellpose output has 16,107
        # adjacent pixel pairs. That eroded rim understates every cell's area
        # by ~9% and, worse for QC, pulls each cell's boundary one pixel
        # inside the membrane signal that `segmentation_qc.boundary_contrast`
        # is trying to sample.
        #
        # Restricted to the resolved boundary class, never a blanket fill of
        # all background, so a genuinely unsegmented region can't be absorbed
        # into a neighbouring cell. Not applied to `'binary_mask'`, whose
        # background may legitimately be empty space rather than a drawn line.
        labeled = _close_label_gaps(labeled, boundary)

    return labeled


def load_tiff_mask(
    tiff_path: str, mask_type: str = "auto", background_area_ratio: Optional[float] = 10.0,
) -> np.ndarray:
    """Load a single segmentation TIFF (`tiff_path`) and return a labeled 2D
    integer array in the same convention as Cellpose's `'outlines'` (each
    cell a unique positive integer, 0 = background), ready to drop into an
    `.npy` dict for `extract_cell_info`.

    `mask_type`:
    - `'auto'` (default): if the file has more than 2 distinct values, it's
      treated as an already-labeled instance mask (values used as-is,
      remapping colors to integers first if it's a multi-color RGB export).
      If it has 2 values, treated as `'skeleton'` (see below) -- the more
      common convention for tools that export boundary lines rather than
      filled instances.
    - `'labeled'`: force treating the file as an already-labeled instance
      mask regardless of how many distinct values it has.
    - `'skeleton'`: force treating a 2-valued file as a boundary/skeleton
      network to discard, splitting the remaining region into cells.
    - `'binary_mask'`: force treating a 2-valued file as a direct
      foreground-is-cells mask (no boundary lines to remove).

    `background_area_ratio` (default 10, `None` or 0 to disable) discards
    connected components more than that many times larger than the MEDIAN
    component -- catches unsegmented background regions outside the tissue
    that connected-component labeling would otherwise mistake for one giant
    cell. Only applies to `'skeleton'`/`'binary_mask'`/2-value-`'auto'`
    labeling (self-derived here), never to an already-labeled mask.

    Raises `ValueError` for an unrecognized `mask_type`, or if a `'labeled'`/
    `'skeleton'`/`'binary_mask'` file doesn't match what that mode expects
    (e.g. `'skeleton'`/`'binary_mask'` on a file with more than 2 values).
    """
    if mask_type not in ("auto", "labeled", "skeleton", "binary_mask"):
        raise ValueError(f"Unknown mask_type: {mask_type!r}")

    mask_arr = np.array(Image.open(tiff_path))
    label_array = _collapse_to_label_array(mask_arr)
    unique_values = np.unique(label_array)

    if mask_type == "labeled":
        return label_array

    if mask_type in ("skeleton", "binary_mask"):
        if len(unique_values) > 2:
            raise ValueError(
                f"{tiff_path}: mask_type={mask_type!r} expects a binary (<=2 value) mask, "
                f"but found {len(unique_values)} distinct values. Use mask_type='labeled' instead "
                "if this is already an instance-labeled mask."
            )
        return _label_binary_mask(label_array, mask_type, background_area_ratio)

    # 'auto'
    if len(unique_values) <= 2:
        return _label_binary_mask(label_array, "skeleton", background_area_ratio)
    return label_array


def _find_matching_image(
    image_folder: str, frame_number: float, reference_name: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Find a same-frame-numbered image in `image_folder` (matching the
    fallback-image lookup pattern used across the v1 GUI windows, e.g.
    src/gui_comparison.py:416-425) and return it as an array, or `None` if
    not found."""
    candidates = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".tif", ".tiff", ".jpeg", ".jpg", ".png"))
    ]

    # Prefer pairing by name when the caller knows which mask it's resolving.
    # Position alone is not enough: sorted order can disagree between the two
    # folders. test_data/Demo_Seg_QC sorts its masks
    # `Demo_OE_mask`, `Demo_WT_handCorrection_mask`, `Demo_WT_mask` but its
    # images `Demo_OE`, `Demo_WT`, `Demo_WT_handCorrection` -- so frames 2 and
    # 3 silently receive each other's image, which would corrupt every
    # intensity-based QC metric while looking perfectly fine.
    if reference_name is not None:
        matched = _match_image_by_stem(candidates, reference_name)
        if matched is not None:
            return np.array(Image.open(os.path.join(image_folder, matched)))

    # Resolved the same way as the mask folder, so a dataset whose filenames
    # carry no counter still pairs image N with mask N by sorted position.
    for file_name, number in resolve_frame_numbers(candidates).items():
        if number == frame_number:
            return np.array(Image.open(os.path.join(image_folder, file_name)))
    return None


# Tokens a segmentation file's name tends to carry that its source image does
# not. Stripped one at a time, longest-form-first, so the most specific match
# wins (`Demo_WT_handCorrection_mask` should find `Demo_WT_handCorrection`,
# not `Demo_WT`).
_MASK_NAME_SUFFIXES = (
    "_handcorrection", "_cp_masks", "_masks", "_mask", "_seg", "_labels", "_label", "_outlines",
)


def _match_image_by_stem(image_files: list[str], mask_filename: str) -> Optional[str]:
    """The image in `image_files` whose stem matches `mask_filename`'s once
    segmentation-specific suffixes are stripped, or `None` if none does.

    Case-insensitive, and tries progressively shorter forms so a mask named
    `X_handCorrection_mask.tif` prefers image `X_handCorrection.tif` over
    `X.tif`.
    """
    by_stem = {os.path.splitext(f)[0].lower(): f for f in image_files}

    stem = os.path.splitext(mask_filename)[0].lower()
    while True:
        if stem in by_stem:
            return by_stem[stem]
        for suffix in _MASK_NAME_SUFFIXES:
            if stem.endswith(suffix) and len(stem) > len(suffix):
                stem = stem[: -len(suffix)]
                break
        else:
            return None


def convert_tiff_masks_to_npy(
    mask_folder: str,
    npy_output_folder: str,
    image_folder: Optional[str] = None,
    mask_type: str = "auto",
    background_area_ratio: Optional[float] = 10.0,
    progress_cb: Optional[Callable[[str, int], None]] = None,
) -> list[str]:
    """Convert every `.tif`/`.tiff` segmentation file in `mask_folder` into a
    Cellpose-compatible `.npy` file written to `npy_output_folder`. Returns
    the list of `.npy` paths written, ordered by frame number. See
    `load_tiff_mask` for `mask_type`/`background_area_ratio`.

    Each dict carries `'masks'` (the filled instance-label array),
    `'outlines'` (the 1-px boundary ring of each cell), and `'img'` when a
    matching image is found -- i.e. exactly Cellpose's own key convention.

    `'masks'` is what makes the file loadable by the Cellpose GUI, which is
    the point of writing these at all -- a user converts TIFF masks here,
    hand-corrects them in Cellpose, and feeds them back. Cellpose's
    `io._load_seg` indexes `dat["masks"]` unconditionally and raised
    `KeyError: 'masks'` on the `{'outlines': ...}`-only dicts this used to
    write; it then reports "no image file found and no image in npy" unless
    `'img'` is embedded, since it looks for the source image beside the
    `.npy` and the converted folder is a subdirectory of the masks folder.

    `'outlines'` holds a genuine thin ring (`thin_label_outlines`). It used to
    hold the same filled array as `'masks'`, which made the UI's "Filled mask
    / Outlines" toggle a no-op on TIFF-derived data -- both modes painted the
    identical array, so the preview appeared stuck on the filled mask in every
    render step. It also changes what `extract_cell_info` measures from (it
    reads `'outlines'`): the ellipse is fitted to the raw stored array, so a
    ring now fits the actual outline instead of a filled point cloud biasing
    it small. That makes TIFF input consistent with the Cellpose path, which
    has always fitted to a ring, at the cost of larger absolute ellipse axes
    (~+17% on `embryos_GBE_GAP43`; the axis *ratio* moves ~1.5%).

    To hand-correct one of these in Cellpose, copy it next to its image and
    rename it `<image-stem>_seg.npy`, which is where Cellpose looks; the
    embedded `'img'` also lets it open standalone.
    """
    os.makedirs(npy_output_folder, exist_ok=True)
    tiff_files = [f for f in os.listdir(mask_folder) if f.lower().endswith((".tif", ".tiff"))]
    frame_number_by_file = resolve_frame_numbers(tiff_files)
    tiff_files = list(frame_number_by_file)
    total = len(tiff_files)
    written_paths = []

    for idx, filename in enumerate(tiff_files):
        labeled = load_tiff_mask(
            os.path.join(mask_folder, filename), mask_type=mask_type, background_area_ratio=background_area_ratio,
        )
        # uint16 matches what Cellpose itself saves and what its GUI expects;
        # `load_tiff_mask` returns a wider int dtype from `ndimage.label`.
        cast = np.uint16 if labeled.max() < 2 ** 16 - 1 else np.uint32
        npy_dict = {"masks": labeled.astype(cast), "outlines": thin_label_outlines(labeled).astype(cast)}

        if image_folder is not None:
            img_array = _find_matching_image(
                image_folder, frame_number_by_file[filename], reference_name=filename,
            )
            if img_array is not None:
                npy_dict["img"] = img_array

        out_path = os.path.join(npy_output_folder, f"{os.path.splitext(filename)[0]}.npy")
        np.save(out_path, npy_dict, allow_pickle=True)
        written_paths.append(out_path)

        if progress_cb is not None:
            percent = int((idx + 1) / total * 100) if total else 100
            progress_cb(f"Converted {filename} to .npy...", percent)

    return written_paths


def load_outline_coordinates(path: str) -> Optional[np.ndarray]:
    """One cell outline from a whitespace-delimited `x y` text file, as an
    open `(N, 2)` float array (a repeated closing point is dropped).

    Handles both variants found in the Vofely et al. 2019 Dryad deposit
    (`10.5061/dryad.g4q6pv3`), surveyed across all 11,346 of its files:
    6,235 have no header and space-separated floats, 5,111 carry an `x\\ty`
    header and tab-separated integers. Returns `None` for anything
    unparseable or shorter than 10 points (its `README.txt` included).
    """
    if os.path.basename(path).lower() == "readme.txt":
        return None
    try:
        with open(path, errors="replace") as handle:
            first = handle.readline()
        try:
            [float(token) for token in first.split()]
            skip_rows = 0
        except ValueError:
            skip_rows = 1
        coords = np.loadtxt(path, skiprows=skip_rows)
    except Exception:
        return None
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 10:
        return None
    if np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    return coords.astype(np.float64)


def _coords_share_a_frame(cells: list[np.ndarray]) -> bool:
    """Whether a group of outlines is laid out in one shared image frame or
    each centred on its own centroid.

    Both occur in the Vofely deposit and they need different handling: the
    `gp`/`JH` files are in shared image-pixel coordinates and keep the field
    of view (so neighbour/tissue analyses are meaningful), while the numeric
    files are individually centred and carry no spatial relationship at all.
    Decided by comparing the spread of centroids against typical cell size --
    independently centred cells all sit on top of each other.
    """
    if len(cells) < 2:
        return False
    centroids = np.array([c.mean(axis=0) for c in cells])
    centroid_spread = np.ptp(centroids, axis=0).max()
    typical_cell = np.median([max(np.ptp(c[:, 0]), np.ptp(c[:, 1])) for c in cells])
    return bool(centroid_spread > typical_cell)


def convert_outline_coords_to_npy(
    coords_folder: str,
    npy_output_folder: str,
    px_per_unit: float = 1.0,
    group_pattern: str = r"^(.*?)[-.]\d+$",
    margin: int = 8,
    max_overlap_fraction: float = 0.01,
    progress_cb: Optional[Callable[[str, int], None]] = None,
) -> list[str]:
    """Convert a folder of per-cell outline coordinate `.txt` files into
    Cellpose-style `.npy` frames for `extract_cell_info`. Returns the written
    paths, ordered by group name.

    Built for the Vofely et al. 2019 Dryad pavement-cell deposit, whose cells
    are hand-traced outlines rather than a segmentation, one file per cell.
    Files are grouped into frames by `group_pattern` (default: the stem minus
    its trailing `-N`/`.N` cell number, which matches both of that deposit's
    naming conventions).

    `px_per_unit` scales the coordinates before rasterizing. **Do not go below
    1.0**: measured against solidity computed directly from the float
    coordinates, 1.0 and 2.0 px/unit reproduce it exactly (bias 0.0000,
    max error 0.0000, r = 1.00000 over 441 cells), while 0.5 px/unit
    introduces a -0.0056 bias and errors up to 0.031.

    Groups already in shared image coordinates keep their layout; groups whose
    cells are each centred on their own centroid are tiled onto a grid, since
    they carry no true spatial relationship (see `_coords_share_a_frame`).
    Grid-tiled frames must not be used for neighbour or tissue-level analyses.
    """
    os.makedirs(npy_output_folder, exist_ok=True)
    pattern = re.compile(group_pattern)

    groups: dict[str, list[str]] = {}
    for filename in sorted(os.listdir(coords_folder)):
        if not filename.endswith(".txt"):
            continue
        stem = filename[:-4]
        match = pattern.match(stem)
        groups.setdefault(match.group(1) if match else stem, []).append(
            os.path.join(coords_folder, filename)
        )

    written_paths = []
    total = len(groups)
    for idx, (group_name, paths) in enumerate(sorted(groups.items())):
        cells = [c for c in (load_outline_coordinates(p) for p in paths) if c is not None]
        if not cells:
            continue

        scaled = [c * px_per_unit for c in cells]
        placed = None
        if _coords_share_a_frame(scaled):
            origin = np.array([min(c[:, 0].min() for c in scaled), min(c[:, 1].min() for c in scaled)])
            shared = [c - origin + margin for c in scaled]
            width = int(max(c[:, 0].max() for c in shared)) + margin + 1
            height = int(max(c[:, 1].max() for c in shared)) + margin + 1

            # Hand-traced outlines are not a segmentation: nothing forced the
            # tracer to keep them disjoint, and a label image cannot represent
            # a pixel belonging to two cells. Painting them in sequence would
            # let each cell silently erase parts of the ones before it.
            # Measured on the Vofely deposit: `gp15-043` overlaps 0.0%, but
            # `gp15-046` overlaps 14.9% of total cell area and the puzzle-like
            # `gp17-003` fern 25.2%, stacking up to 5 deep. So the layout is
            # only kept when the overlap is negligible.
            coverage = np.zeros((height, width), dtype=np.int32)
            total_area = 0
            for c in shared:
                one = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(one, [np.round(c).astype(np.int32).reshape(-1, 1, 2)], 1)
                coverage += one
                total_area += int(one.sum())
            overlap = int((coverage > 1).sum())
            if total_area and overlap / total_area > max_overlap_fraction:
                logger.warning(
                    "%s: traced outlines overlap over %.1f%% of their total area (max %d cells deep); "
                    "tiling to a grid instead of keeping the image layout, so no cell is truncated. "
                    "Neighbor/tissue-level analyses are not meaningful for this group.",
                    group_name, 100 * overlap / total_area, int(coverage.max()),
                )
            else:
                placed = shared

        if placed is None:
            cell_w = int(max(np.ptp(c[:, 0]) for c in scaled)) + 2 * margin + 1
            cell_h = int(max(np.ptp(c[:, 1]) for c in scaled)) + 2 * margin + 1
            columns = int(np.ceil(np.sqrt(len(scaled))))
            rows = int(np.ceil(len(scaled) / columns))
            width, height = columns * cell_w, rows * cell_h
            placed = []
            for i, c in enumerate(scaled):
                row, column = divmod(i, columns)
                placed.append(
                    c - [c[:, 0].min(), c[:, 1].min()] + [column * cell_w + margin, row * cell_h + margin]
                )

        cast = np.uint16 if len(placed) < 2 ** 16 - 1 else np.uint32
        labeled = np.zeros((height, width), dtype=cast)
        for i, c in enumerate(placed, 1):
            cv2.fillPoly(labeled, [np.round(c).astype(np.int32).reshape(-1, 1, 2)], int(i))

        out_path = os.path.join(npy_output_folder, f"{group_name}.npy")
        np.save(
            out_path,
            {"masks": labeled, "outlines": thin_label_outlines(labeled).astype(cast)},
            allow_pickle=True,
        )
        written_paths.append(out_path)

        if progress_cb is not None:
            progress_cb(f"Converted {group_name} ({len(placed)} cells)...", int((idx + 1) / total * 100))

    return written_paths


# --------------------------------------------------------------------------
# The one place a label array is read (Round 19 D2)
# --------------------------------------------------------------------------
#
# Six modules used to index `dat["masks"]` for themselves: this one,
# `webview/api._load_frame`, `cell_tracking._FrameLabelLoader`,
# `tracking_benchmark.load_masks`, `shape_features` and `tissue_features`.
# Six readers is six chances to disagree about what a frame contains, and the
# whole of Rounds 17-19 has been about inputs diverging from what a run thought
# it read.
#
# A minimum-cell-area filter makes that concrete. `exclude_edge_cells` used to
# work by `continue`-ing inside the per-cell loop -- dropping the ROW and
# leaving the label in the mask, which is harmless for an edge cell (and since
# Round 26 it drops nothing at all). For a 1 px vertex artefact it is not: every consumer downstream reads the MASK, so the label
# survives as a fifth neighbour in `adjacency.build_neighbor_graph` (which can
# stop the four real cells around a vertex being recorded as neighbours of one
# another -- corrupting T1 detection exactly where T1s happen), is seen by
# `mask_overlap_fractions`, and renders as a permanently unmatched cell.
#
# So the removal happens on the mask, here, once.

#: Minimum area in pixels for a label to survive loading; 0 disables. Module
#: state rather than a parameter threaded through six call sites: the property
#: that matters is that every consumer sees the SAME labels, and six parameters
#: would be six chances to pass a different one. Set once per project via
#: `set_min_cell_area`.
_MIN_CELL_AREA_PX = 0

#: Chosen from the data, not from taste (Round 19 D3). On
#: `embryos_GBE_GAP43` the three smallest labels are 2, 3 and 3 px and the next
#: smallest is 119 px -- a 116 px gap with nothing in it. On
#: `AS_dorsal view_E-cad` the smallest is 139 px and there are no artefacts at
#: all. 19 is the geometric midpoint of that observed gap (sqrt(3 x 119) =
#: 18.9): an order of magnitude above the artefacts, six times below the
#: smallest real cell. It is a suggestion the UI offers, never a default --
#: `_MIN_CELL_AREA_PX` ships 0.
#:
#: **The bound is inclusive on the kept side: a label of exactly N px SURVIVES**
#: (`apply_min_cell_area` drops `count < N`). "Minimum area 3" therefore keeps
#: 3 px labels; to drop them, set 4. That is what let twelve 3 px vertex
#: pockets through on `embryos_GBE_OE_GAP43` at a setting of 3 (Round 24), so
#: the UI says "keep >= N px" rather than leaving the boundary to be guessed.
#:
#: Since Round 24 (2026-09-11) the numbers above are history: WT GBE's masks,
#: hand-corrected since, no longer hold the 2/3/3 px labels (smallest 119 px),
#: and a skeleton TIFF's vertex pockets are returned to the network at load by
#: shape, whatever their size (`_vertex_pockets`). On that input this filter is
#: a second line, not the fix.
SUGGESTED_MIN_CELL_AREA_PX = 19


def set_min_cell_area(area_px: int) -> int:
    """Set the minimum label area every loader applies. 0 disables."""
    global _MIN_CELL_AREA_PX
    _MIN_CELL_AREA_PX = max(0, int(area_px or 0))
    return _MIN_CELL_AREA_PX


def get_min_cell_area() -> int:
    return _MIN_CELL_AREA_PX


#: Whether the ANALYSIS consumers drop cells near the image border (Round 26).
#: Module state for the same reason the minimum area is: the property that
#: matters is that every consumer, and every provenance writer, reads the SAME
#: answer -- threading it through six call sites is six chances to pass a
#: different one. Like the area it is not persisted by itself; the project
#: settings file restores it (`webview/api._load_project_settings`).
_EXCLUDE_EDGE_CELLS = False


def set_edge_exclusion(enabled: bool) -> bool:
    """Turn the border-cell exclusion on or off for the analysis consumers.

    It never affects tracking. See `drop_edge_cells`.
    """
    global _EXCLUDE_EDGE_CELLS
    _EXCLUDE_EDGE_CELLS = bool(enabled)
    return _EXCLUDE_EDGE_CELLS


def get_edge_exclusion() -> bool:
    return _EXCLUDE_EDGE_CELLS


def apply_min_cell_area(labels, area_px: Optional[int] = None):
    """Zero out labels smaller than `area_px`. Returns `(labels, removed)`.

    **Background, not a neighbour.** A 1 px vertex gap is not part of any cell,
    and donating it to a neighbour would change that cell's measured area for
    no gain. A background pocket this small is bridged by
    `build_neighbor_graph`'s `dilation_px=2` anyway.
    """
    threshold = _MIN_CELL_AREA_PX if area_px is None else int(area_px or 0)
    if threshold <= 0 or labels is None:
        return labels, []
    values, counts = np.unique(labels[labels > 0], return_counts=True)
    small = [int(v) for v, c in zip(values, counts) if c < threshold]
    if not small:
        return labels, []
    cleaned = labels.copy()
    cleaned[np.isin(cleaned, small)] = 0
    return cleaned, small


def looks_filled(labels) -> bool:
    """Is this a filled label image, or a 1-px boundary array?

    Cellpose's `outlines` key is the boundary line; some older output of
    `convert_tiff_masks_to_npy` wrote a filled array under the same name. They
    cannot be told apart by key, so tell them apart by shape: a filled cell
    occupies most of its bounding box, a ring occupies a thin fraction of it.
    Checked on the largest label, which is the least noisy one.

    Lived in `cell_tracking` until Round 22, where `load_label_array` needed
    it to make its own docstring true -- it claimed the `outlines` fallback
    was conditional on this test and did it unconditionally, so a ring array
    could come back as if it were a label image.
    """
    if labels is None or not hasattr(labels, "shape") or labels.ndim != 2:
        return False
    values, counts = np.unique(labels[labels > 0], return_counts=True)
    if not len(values):
        return False
    label = values[int(np.argmax(counts))]
    ys, xs = np.nonzero(labels == label)
    box = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    return box > 0 and (len(ys) / box) > 0.35


def load_label_array(npy_path_or_dict, area_px: Optional[int] = None,
                     with_removed: bool = False):
    """**The** way to get a frame's filled label array. Every consumer goes
    through here so they cannot disagree about what the frame contains.

    Prefers `masks`; falls back to `outlines` only when that array passes
    `looks_filled` (older `convert_tiff_masks_to_npy` output wrote a filled
    array under that name). Returns `None` when neither key yields a filled
    array -- a genuine boundary ring is not a label image, and filtering one
    by area removes every label in it, which is exactly the mistake Round 21
    made. Applies the minimum-area filter last, so the label set returned is
    the label set every downstream step sees.

    The `looks_filled` test used to be applied by `cell_tracking` only, so on
    an old `outlines`-only file tracking filtered and extraction did not
    (Round 22 D) -- the same split Round 21 §A fixed one branch of.
    """
    data = (npy_path_or_dict if isinstance(npy_path_or_dict, dict)
            else np.load(npy_path_or_dict, allow_pickle=True).item())
    labels = data.get("masks")
    if labels is None and looks_filled(data.get("outlines")):
        labels = data.get("outlines")
    labels, removed = apply_min_cell_area(labels, area_px)
    return (labels, removed) if with_removed else labels


def prepare_segmentation_folder(
    mask_folder: str,
    npy_output_folder: Optional[str] = None,
    image_folder: Optional[str] = None,
    mask_type: str = "auto",
    background_area_ratio: Optional[float] = 10.0,
    progress_cb: Optional[Callable[[str, int], None]] = None,
) -> str:
    """Return a folder of Cellpose-style `.npy` segmentation files ready for
    `extract_cell_info`, regardless of whether `mask_folder` already holds
    `.npy` files or needs converting from `.tif`/`.tiff` masks/skeletons
    first. This is the one-call entry point for the format flexibility
    requested in review (supporting segmentation tools other than Cellpose,
    e.g. EpySeg, that export TIFF masks or boundary skeletons):

    - Already `.npy`: returned unchanged (no conversion, no copying).
    - `.tif`/`.tiff` only: converted via `convert_tiff_masks_to_npy` into
      `npy_output_folder` (defaults to `<mask_folder>/npy_converted`), which
      is then returned. See `load_tiff_mask` for `mask_type`/
      `background_area_ratio`.

    Raises `ValueError` if `mask_folder` has neither format (`'empty'`) or
    both (`'mixed'`, ambiguous -- keep one segmentation format per folder).
    """
    fmt = detect_mask_format(mask_folder)

    if fmt == "npy":
        return mask_folder

    if fmt == "tiff":
        output_folder = npy_output_folder or os.path.join(mask_folder, "npy_converted")
        convert_tiff_masks_to_npy(
            mask_folder, output_folder, image_folder=image_folder, mask_type=mask_type,
            background_area_ratio=background_area_ratio, progress_cb=progress_cb,
        )
        return output_folder

    if fmt == "empty":
        raise ValueError(f"No .npy or .tif/.tiff segmentation files found in {mask_folder}")

    raise ValueError(
        f"Both .npy and .tif/.tiff files found in {mask_folder} -- ambiguous input; "
        "keep only one segmentation format per folder."
    )


#: The margin `is_edge_cell` has always used, named rather than written out at
#: each call site. Round 26 deliberately does NOT change it: with the exclusion
#: moved off the tracking path, what the right number is -- and whether it wants
#: a Schmitt trigger over it, as the junction-length minimum has -- is a
#: separate question, to be decided on the numbers Round 26 produces.
DEFAULT_EDGE_MARGIN_PX = 3


def edge_distance(cell_pixels: tuple[np.ndarray, np.ndarray], height: int, width: int) -> int:
    """How far a cell's bounding box stays clear of the image border, in pixels.

    `min(min_x, min_y, width - max_x, height - max_y)` -- exactly the four
    quantities `is_edge_cell` compares against its margin, so

        is_edge_cell(pixels, h, w, margin) == (edge_distance(pixels, h, w) <= margin)

    holds for every margin, and the self-test asserts it over a grid of cases.

    **The far edges are measured as `width - max_x`, not `width - 1 - max_x`.**
    That is one more than the geometric distance -- a cell occupying the last
    column scores 1, not 0 -- and it is kept because the predicate it has to
    agree with is a verbatim port of v1 and already asymmetric. Changing the
    convention here would silently move which cells a given margin excludes,
    which is a separate decision from Round 26's (see `v2/rounds/Round 26.md`
    §E: `margin=3` is not being touched this round).
    """
    return int(min(np.min(cell_pixels[1]), np.min(cell_pixels[0]),
                   width - np.max(cell_pixels[1]), height - np.max(cell_pixels[0])))


def is_edge_cell(cell_pixels: tuple[np.ndarray, np.ndarray], height: int, width: int,
                 margin: int = DEFAULT_EDGE_MARGIN_PX) -> bool:
    """True if a cell's pixel positions (as returned by `np.where(outlines == cell_id)`,
    i.e. `(row_indices, col_indices)`) come within `margin` pixels of any image
    border. Verbatim port of src/gui_main.py:944-954 (already `self`-free).

    **Since Round 26 this no longer decides whether a cell gets a row.**
    Extraction records `edge_distance` for every cell and the filter moved to
    the consumers that need it (morphometrics, PCA/K-means) -- tracking does
    not filter, because a truncated cell's *geometry* is unusable while its
    centroid and mask are fine. Run per frame on a bounding box, this predicate
    flipped 136 times across 85 frame pairs on `AS_LE_lateral view_E-cad`, and
    each flip minted an orphan cell and a new track
    (`docs/edge-flicker-diagnosis.md`).
    """
    return edge_distance(cell_pixels, height, width) <= margin


def drop_edge_cells(frame: "pd.DataFrame", margin: Optional[int] = None) -> "pd.DataFrame":
    """Rows whose cell stays more than `margin` px clear of the image border.

    **The one place the exclusion is applied**, so the consumers that want it
    cannot disagree about what it means (Round 26 B2). Tracking does not call
    this: a truncated cell's geometry is wrong, which is what the exclusion is
    for, but its centroid and mask -- all tracking reads -- are fine.

    A table extracted before Round 26 has no `Edge Distance` column. It is
    returned unchanged, because such a table had the edge cells dropped at
    extraction already: filtering again would be right, and raising would make
    an old project unopenable, but *silently* returning fewer rows than the
    caller's own count is what neither would do.
    """
    if "Edge Distance" not in frame.columns:
        return frame
    limit = DEFAULT_EDGE_MARGIN_PX if margin is None else int(margin)
    return frame[frame["Edge Distance"] > limit]


# Arc-length spacing, in pixels, at which a traced boundary's perimeter is
# measured. See `contour_perimeter`.
PERIMETER_SPACING_PX = 3.0


def contour_perimeter(
    contour: np.ndarray, spacing: float = PERIMETER_SPACING_PX, min_points: int = 32
) -> float:
    """Perimeter measured on the boundary resampled to a fixed arc-length
    `spacing`, rather than by summing the raw pixel chain.

    Summing a pixel chain overestimates arc length -- a staircase is longer
    than the diagonal it approximates -- and the error depends on the
    boundary's orientation, so it cannot be removed by a constant. Measured
    against exact perimeters on rasterized shapes (circles r=30..200, axis-
    aligned squares, the same squares at 30 degrees, and 6/8-lobed cells):

        estimator                mean |err|   max |err|
        raw cv2.arcLength           4.42%       7.29%
        Kulpa x0.9481               1.64%       5.19%
        Vossepoel-Smeulders         0.74%       2.15%
        resample @2px               0.80%       1.63%
        resample @3px (this)        0.31%       0.70%

    Kulpa's constant looks appealing because `arcLength` is a near-uniform
    +5.5% on curved boundaries, but an axis-aligned square has no staircase at
    all and is already exact, so the correction makes it 5.2% too short.
    Resampling has no such blind spot: it reads the square exactly and the
    circle to 0.6%.

    This matters beyond tidiness. The shape index p = P/sqrt(A) is compared
    against the rigidity threshold p* ~ 3.81 of Bi et al.; the raw estimator
    reports a perfect circle as 3.743 instead of 3.545, which is enough to move
    solid-like tissue across that threshold.

    `min_points` keeps small cells from being under-sampled, at the cost of a
    finer spacing than requested for them.
    """
    closed = np.vstack([contour, contour[0]])
    step = np.hypot(*np.diff(closed, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(step)])
    total = arc[-1]
    if total <= 0:
        return 0.0
    n_points = max(min_points, int(total / spacing))
    target = np.linspace(0.0, total, n_points, endpoint=False)
    resampled = np.column_stack([np.interp(target, arc, closed[:, 0]), np.interp(target, arc, closed[:, 1])])
    return float(np.hypot(*np.diff(np.vstack([resampled, resampled[0]]), axis=0).T).sum())


def extract_cell_info(
    npy_folder: str,
    output_path: Optional[str] = None,
    exclude_edge_cells: bool = False,
    progress_cb: Optional[Callable[[str, int], None]] = None,
    tracking_only: bool = False,
    frames: Optional[set] = None,
) -> tuple[pd.DataFrame, int]:
    """Extract the 19-column geometric feature table for every cell across all
    frames in `npy_folder` (Cellpose `.npy` output, each a dict with an
    `'outlines'` labeled mask). Row index = `Cell Index` string
    `Cell{frame}_{cell_id}`.

    If `output_path` is given, writes `Cells_info.xlsx` (sheet `'Cell'`) there;
    if `None`, only the DataFrame is returned (no filesystem writes).

    `progress_cb(message, percent)` is called once per frame if provided.

    `frames`, when given, restricts extraction to those frame numbers. Every
    column is a function of that frame's own mask alone, so the rows produced
    are identical to the ones a whole-movie run would produce for them -- which
    is what makes a partial re-extraction after a hand-correction exact rather
    than approximate.

    `tracking_only=True` computes only the four columns
    `cell_tracking.build_cost_matrix` actually reads and leaves the rest `NaN`,
    for users who only want to track and never classify. The column set and
    order are unchanged either way -- `Edge Distance` is filled in both modes,
    since it costs one `min` over pixels already in hand.

    **`exclude_edge_cells` no longer drops rows here (Round 26 B).** Every cell
    gets a row and an `Edge Distance`; `exclude_edge_cells <= margin` is applied
    by the consumers for which a truncated geometry is wrong -- morphometrics
    and PCA/K-means -- and NOT by tracking, which reads only the centroid and
    the mask. The parameter is kept so callers and stored settings stay valid,
    and because which semantics a run used has to be recoverable afterwards; it
    is recorded in `extract_settings.json` and in the tracking provenance. A
    `True` here logs that it no longer filters rather than silently doing
    nothing.

    Ported from `bio_extract_cell_info`, src/gui_main.py:956-1092.
    Returns `(cells_info, fig_num)` exactly as v1 did.
    """
    # Rows are accumulated as plain lists and turned into a DataFrame once at
    # the end. The previous version built a one-row DataFrame per cell and
    # `pd.concat`-ed it onto an accumulator, which is quadratic and, after the
    # bounding-box change above, had become ~60% of total extraction time
    # (measured with cProfile on `AS_dorsal view_E-cad`) -- more than all the
    # image processing combined.
    if exclude_edge_cells:
        logger.info(
            "exclude_edge_cells=True no longer drops rows at extraction (Round 26): every cell "
            "gets a row and an 'Edge Distance' column, and the filter is applied by the "
            "morphometric and PCA/K-means consumers instead. Tracking deliberately does not filter."
        )

    row_index: list[str] = []
    row_values: list[np.ndarray] = []
    fig_num = 0

    npy_files = list_segmentation_files(npy_folder)
    frame_number_by_file = resolve_frame_numbers(npy_files)
    # `frames` limits the work to a range without changing what is computed for
    # the frames inside it: every column here depends only on that frame's own
    # mask, so a range is exact rather than an approximation (Round 19 C3).
    if frames is not None:
        wanted = {int(f) for f in frames}
        npy_files = [f for f in npy_files if frame_number_by_file.get(f) in wanted]
    total_files = len(npy_files)
    frame_number = 0

    for idx, filename in enumerate(npy_files):
        basename, ext = os.path.splitext(filename)
        if ext != ".npy":
            continue
        dat = np.load(os.path.join(npy_folder, filename), allow_pickle=True).item()
        outlines = dat["outlines"]
        # The FILLED per-cell label array. Everything the old code derived by
        # re-filling the outline ring with `cv2.fillPoly` is now measured on
        # this instead -- see `_MASK_DERIVED_NOTE`. Legacy `.npy` files written
        # before Cellpose stored `'masks'` fall back to the old reconstruction.
        # Through the shared loader (`load_label_array`), so extraction sees
        # exactly the label set every other consumer sees (Round 19 D2). Round
        # 19 wrote that comment and then called `apply_min_cell_area` directly,
        # which is not the shared loader: on an old `outlines`-only file
        # `dat.get("masks")` is None, so nothing was removed and nothing said
        # so, while `cell_tracking` filtered the same file (Round 22 D).
        #
        # **Both arrays, and that is the whole point (Round 21 A1).** Round 19
        # filtered only `masks` -- but the per-cell loop below enumerates
        # `outlines`, so extraction kept emitting a row for every removed
        # label: at a 200 px threshold on GBE the mask dropped to 261 labels
        # while `Cells_info` still had 262 rows, and tracking, which is driven
        # by `Cells_info`, then linked all 123 removed cells across the movie.
        # Round 19's assertion missed it because it compared the four consumers
        # that read `masks` and never checked extraction's own row set.
        fill_labels, _removed_small = load_label_array(dat, with_removed=True)
        if fill_labels is None and get_min_cell_area() > 0:
            # Not "nothing to remove": nothing to decide ON. Say so rather
            # than reporting a removal count of zero for this frame.
            logger.warning(
                "Frame %s (%s) has no filled label array (keys: %s); the "
                "minimum-area filter of %d px could not be applied to it",
                frame_number_by_file.get(filename), filename,
                sorted(dat)[:6], get_min_cell_area(),
            )
        # The DECISION is made once, on the filled array, and then applied to
        # both. `outlines` must not be filtered by its own pixel counts: it is
        # a ~2-5%-of-pixels boundary ring, so an area threshold meant for
        # filled cells removes every label (at 200 px on GBE it took all 262).
        if _removed_small:
            outlines = outlines.copy()
            outlines[np.isin(outlines, _removed_small)] = 0
        height, width = outlines.shape
        fig_num += 1
        # Via `resolve_frame_numbers` rather than a private regex, so the
        # `Cell{frame}_{id}` indices written here always resolve back to a
        # file for `_npy_path_for_frame`/`_find_matching_image`. The old
        # inline fallback numbered digit-less files positionally while those
        # lookups saw `inf`, which is what left such datasets previewing blank.
        frame_number = frame_number_by_file[filename]
        fig_id = str(frame_number)

        if progress_cb is not None:
            percent = int((idx + 1) / total_files * 100) if total_files else 100
            progress_cb(f"Extracting cell features for frame {frame_number}...", percent)

        unique_ids = np.unique(outlines)
        unique_ids = unique_ids[unique_ids != 0]

        # Per-cell bounding boxes, so the per-cell work below scans only each
        # cell's own neighborhood instead of the whole frame. Profiling the
        # previous full-frame version on `AS_dorsal view_E-cad`: `np.where`
        # over the whole array was 55.8% of extraction time and
        # `distanceTransform` over the whole array another 35.4% -- both
        # O(n_cells * frame_area). `find_objects` returns a list positionally
        # indexed by `label - 1` (None for absent labels), which is why it is
        # indexed rather than iterated.
        bboxes = ndimage_find_objects(outlines)
        # The crop must hold the whole MASK, not only the ring: the raster
        # measurements below read `fill_labels` through it, and a ring that has
        # gone stale (see `_STALE_OUTLINE_FRACTION`) can be far smaller than
        # its cell -- on `AS_dorsal` a 1826 px cell was measured as the 13 px of
        # it that fell inside its 1 px ring's box. The union keeps a ring pixel
        # outside the mask (none found in any test dataset) inside the crop too.
        mask_bboxes = ndimage_find_objects(fill_labels) if fill_labels is not None else []
        stale_outlines = []

        for cell_id in unique_ids:
            # Indexed, not guarded: `cell_id` came from `np.unique(outlines)`,
            # so the label is present and `find_objects` cannot return None for
            # it. A `None` here would mean the array is not what this function
            # documents, and raising says so rather than silently dropping a
            # cell from the table.
            bbox = bboxes[cell_id - 1]
            mask_bbox = mask_bboxes[cell_id - 1] if cell_id <= len(mask_bboxes) else None
            y0, y1, x0, x1 = bbox[0].start, bbox[0].stop, bbox[1].start, bbox[1].stop
            if mask_bbox is not None:
                y0, y1 = min(y0, mask_bbox[0].start), max(y1, mask_bbox[0].stop)
                x0, x1 = min(x0, mask_bbox[1].start), max(x1, mask_bbox[1].stop)
            y0 = max(0, y0 - _BBOX_MARGIN)
            y1 = min(height, y1 + _BBOX_MARGIN)
            x0 = max(0, x0 - _BBOX_MARGIN)
            x1 = min(width, x1 + _BBOX_MARGIN)
            window = outlines[y0:y1, x0:x1]

            local = np.where(window == cell_id)
            if mask_bbox is not None:
                # Cellpose's own ring convention (`masks_to_outlines`): every
                # external contour of the cell's mask, all pixels.
                own = (fill_labels[y0:y1, x0:x1] == cell_id).astype(np.uint8)
                mask_ring = np.zeros_like(own)
                for contour in cv2.findContours(own, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]:
                    mask_ring[contour[:, 0, 1], contour[:, 0, 0]] = 1
                interior = mask_ring.copy()
                if y0 == 0:
                    interior[0, :] = 0
                if y1 == height:
                    interior[-1, :] = 0
                if x0 == 0:
                    interior[:, 0] = 0
                if x1 == width:
                    interior[:, -1] = 0
                if local[0].size < _STALE_OUTLINE_FRACTION * np.count_nonzero(interior):
                    stale_outlines.append(int(cell_id))
                    local = np.where(mask_ring > 0)
            # Absolute frame coordinates, matching the pre-bbox behaviour that
            # every column below (centroid, left/right boundary, edge test) is
            # defined in.
            positions = (local[0] + y0, local[1] + x0)

            # Measured for EVERY cell, and it no longer decides whether the cell
            # gets a row (Round 26 B1). The exclusion is scientifically
            # meaningful for morphometrics -- a truncated cell's geometry is
            # wrong -- and meaningless for tracking, which reads only the
            # centroid and the mask, both of which a border cell has intact.
            # Applied here, per frame and on a bounding box, it flipped 136
            # times over 85 frame pairs on `AS_LE_lateral view_E-cad`, and each
            # flip orphaned a cell and started a new track
            # (`docs/edge-flicker-diagnosis.md`). Filtering now happens where
            # the geometry is actually used.
            #
            # Measured on the FILLED mask, not on `positions` (which come from
            # `outlines`). `thin_label_outlines` marks a pixel only where a
            # 4-neighbour INSIDE the array carries a different label, so a cell
            # running to the image border has no ring pixel along that border
            # and its ring can sit many pixels inside the frame -- on
            # `embryos_GBE_GAP43` the runaway background component's ring
            # stopped 4-19 px short of every edge, so a `margin=3` test missed
            # it in 21 of 31 frames. The filled mask has no such blind spot.
            edge_source = fill_labels if fill_labels is not None else outlines
            edge_positions = np.where(edge_source[y0:y1, x0:x1] == cell_id)
            if edge_positions[0].size == 0:
                edge_positions = local
            cell_edge_distance = edge_distance(
                (edge_positions[0] + y0, edge_positions[1] + x0), height, width,
            )

            if tracking_only:
                # Only the four columns `cell_tracking.build_cost_matrix`
                # actually reads: Maximum Horizontal/Vertical Length (its
                # displacement gate) and Center X/Y Coordinate (the only
                # nonzero entries in DEFAULT_TRACKING_WEIGHTS). Everything
                # else is left NaN -- the column set and its order are
                # unchanged, since downstream code maps columns positionally.
                # Float, not int: truncating quantises the very quantity the
                # tracking cost compares. On `embryos_GBE_GAP43` the median
                # margin between a cell's true continuation and its nearest
                # impostor is 5.4 px, and ~20% of cells have a runner-up within
                # 2 px -- against which a 1 px truncation bias is not noise you
                # can ignore.
                cy = float(np.mean(positions[0]))
                cx = float(np.mean(positions[1]))
                ap = np.max(positions[1]) - np.min(positions[1])
                dv = np.max(positions[0]) - np.min(positions[0])
                cell_info = [
                    np.nan, ap, dv, ap / dv if dv != 0 else -1, np.nan, np.nan,
                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                    cx, cy, np.min(positions[1]), np.max(positions[1]), np.nan,
                    cell_edge_distance,
                ]
                # Rounded like the full path so the populated columns come out
                # float64 rather than int64; without it the two modes produce
                # numerically equal tables with different dtypes.
                row_index.append(f"Cell{fig_id}_{cell_id}")
                row_values.append(np.round(cell_info, 2))
                continue

            # Kept in absolute coordinates because `cv2.fitEllipse` consumes it
            # further down; only the raster work below is done in the crop.
            contour_points = np.array([positions[1], positions[0]]).T.reshape(-1, 1, 2)

            # Local coordinates for the raster work: area, arc length, vertex
            # count and both radii are all translation-invariant, so cropping
            # cannot change them.
            if fill_labels is not None:
                fill_img = (fill_labels[y0:y1, x0:x1] == cell_id).astype(np.uint8)
            else:
                local_points = np.array([local[1], local[0]]).T.reshape(-1, 1, 2)
                fill_img = np.zeros(window.shape, dtype=np.uint8)
                cv2.fillPoly(fill_img, [local_points], 1)
            cell_area = np.sum(fill_img)

            contours, _ = cv2.findContours(fill_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) > 0:
                filled_contour = max(contours, key=cv2.contourArea)
                # Measured on the resampled boundary rather than by summing the
                # pixel chain: `cv2.arcLength` overestimates a digitized outline
                # by ~5.5% and the error depends on its orientation, so it
                # cannot be removed by a constant. See `contour_perimeter`.
                cell_arclength = int(round(contour_perimeter(filled_contour.reshape(-1, 2).astype(np.float64))))
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

            cell_ellipse = cv2.fitEllipse(contour_points) if len(contour_points) >= 5 else None
            if cell_ellipse is not None and min(cell_ellipse[1]) < _MIN_ELLIPSE_AXIS_PX:
                # Degenerate fit: fall through to the same `-1` sentinel a
                # failed fit already uses. See `_MIN_ELLIPSE_AXIS_PX`.
                cell_ellipse = None
            if cell_ellipse is not None:
                ratio_long_and_short_axes = cell_ellipse[1][1] / cell_ellipse[1][0]
                cell_ellipse_angle = 180 - cell_ellipse[2] if cell_ellipse[2] > 90 else cell_ellipse[2]
            else:
                ratio_long_and_short_axes = -1
                cell_ellipse_angle = -1

            # Float, not int -- see the note on the tracking-only path above.
            cy = float(np.mean(positions[0]))
            cx = float(np.mean(positions[1]))

            ap = np.max(positions[1]) - np.min(positions[1])
            dv = np.max(positions[0]) - np.min(positions[0])
            aspect_ratio = ap / dv if dv != 0 else -1

            cell_info = [
                cell_area, ap, dv, aspect_ratio, cell_arclength, len(vertices),
                cell_ellipse[1][0] if cell_ellipse is not None else -1,
                cell_ellipse[1][1] if cell_ellipse is not None else -1,
                ratio_long_and_short_axes, cell_ellipse_angle,
                min_enclosing_circle_radius, max_inner_circle_radius,
                cx, cy, np.min(positions[1]), np.max(positions[1]), circularity,
                cell_edge_distance,
            ]
            cell_info = np.round(cell_info, 2)
            row_index.append(f"Cell{fig_id}_{cell_id}")
            row_values.append(cell_info)

        if stale_outlines:
            logger.warning(
                "Frame %s (%s): %d cell(s) have a stored outline covering less than half "
                "of their mask's boundary (stale after a manual edit?) and were measured "
                "from the mask instead: labels %s",
                frame_number, filename, len(stale_outlines), stale_outlines,
            )

    cells_info = pd.DataFrame(row_values, index=row_index, columns=CELLS_INFO_COLUMNS)

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

    def _self_test() -> None:
        """Mask-interpretation checks that need no dataset on disk."""
        import tempfile

        def ncells(x):
            return len(set(np.unique(x)) - {0})

        def adjacent_pairs(x):
            total = 0
            for a, b in ((x[:-1, :], x[1:, :]), (x[:, :-1], x[:, 1:])):
                total += int(((a != b) & (a > 0) & (b > 0)).sum())
            return total

        tmp = tempfile.mkdtemp()

        # A 1-px white grid on black: 9 cells whose only separator is the
        # drawn line. This is the TA/EpySeg shape that used to come out eroded.
        grid = np.zeros((31, 31), np.uint8)
        for i in (0, 10, 20, 30):
            grid[i, :] = 255
            grid[:, i] = 255
        skeleton_path = os.path.join(tmp, "skeleton1.tif")
        Image.fromarray(grid).save(skeleton_path)

        labeled = load_tiff_mask(skeleton_path, mask_type="auto")
        print("skeleton/auto: cells=%d background=%.1f%% adjacent_pairs=%d"
              % (ncells(labeled), 100 * (labeled == 0).mean(), adjacent_pairs(labeled)))
        assert ncells(labeled) == 9, ncells(labeled)
        assert (labeled == 0).sum() == 0, "the drawn boundary must be handed back to the cells"
        assert adjacent_pairs(labeled) > 0, "neighbouring cells must end up touching, as Cellpose's do"

        ring = thin_label_outlines(labeled)
        print("thin outlines: %.1f%% of pixels, label set preserved=%s"
              % (100 * (ring > 0).mean(), set(np.unique(ring)) - {0} == set(np.unique(labeled)) - {0})),
        assert (ring > 0).sum() < (labeled > 0).sum(), "a ring must be smaller than the filled mask"
        assert set(np.unique(ring)) - {0} == set(np.unique(labeled)) - {0}

        # --- R24: a vertex pocket is not a cell. The same grid with the vertex
        # at (10, 10) drawn as a thick knot that encloses a 3-px L -- the shape
        # Tissue Analyzer left 12 times on `embryos_GBE_OE_GAP43` (frame 30
        # label 17). It must go back to the network, not become a 10th cell.
        knot = grid.copy()
        knot[9:13, 9:13] = 255
        for y, x in ((10, 10), (11, 10), (11, 11)):
            knot[y, x] = 0
        knot_path = os.path.join(tmp, "skeleton_pocket.tif")
        Image.fromarray(knot).save(knot_path)
        knot_labels = load_tiff_mask(knot_path, mask_type="auto")
        assert ncells(knot_labels) == 9, ("the enclosed 3-px pocket became a cell", ncells(knot_labels))
        knot_areas = np.bincount(knot_labels.ravel())[1:]
        assert knot_areas[knot_areas > 0].min() > 50, knot_areas   # zeroed, not renumbered: gaps are expected
        # The shape test at its boundary: a 3x3 block has one interior pixel
        # and stays; a 2-px-thick strip of any length has none and goes.
        shapes = np.zeros((12, 30), dtype=np.int64)
        shapes[1:4, 1:4] = 1
        shapes[6:8, 1:29] = 2
        flagged = set(np.unique(shapes[_vertex_pockets(shapes)]))
        assert flagged == {2}, flagged
        print("R24: an enclosed 3-px vertex pocket is returned to the skeleton (9 cells,")
        print("  not 10); a 3x3 block is kept, a 2 x 28 strip is not.")

        # binary_mask means "foreground IS the cell body" -- three solid blobs
        # on empty background. The background here is real empty space, so it
        # must NOT be gap-closed into the cells.
        blobs = np.zeros((31, 31), np.uint8)
        blobs[2:14, 2:14] = 255
        blobs[17:29, 2:14] = 255
        blobs[2:14, 17:29] = 255
        blob_path = os.path.join(tmp, "binary1.tif")
        Image.fromarray(blobs).save(blob_path)

        blob_labels = load_tiff_mask(blob_path, mask_type="binary_mask")
        print("binary_mask: cells=%d background=%.1f%%"
              % (ncells(blob_labels), 100 * (blob_labels == 0).mean()))
        assert ncells(blob_labels) == 3, "must label the foreground blobs, not the background"
        assert (blob_labels == 0).sum() > 0, "binary_mask background is real empty space, not a separator"

        # Coordinate-outline import. A lobed star and a plain square, written
        # in both on-disk variants (headerless floats / `x\ty` header), far
        # enough apart to share a frame without overlapping.
        coords_dir = os.path.join(tmp, "coords")
        os.makedirs(coords_dir, exist_ok=True)
        angle = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        radius = 40 * (1 + 0.35 * np.cos(6 * angle))
        star = np.column_stack([radius * np.cos(angle) + 100, radius * np.sin(angle) + 100])
        square = np.column_stack(
            [np.r_[np.linspace(-30, 30, 100), np.full(100, 30.0), np.linspace(30, -30, 100), np.full(100, -30.0)] + 400,
             np.r_[np.full(100, -30.0), np.linspace(-30, 30, 100), np.full(100, 30.0), np.linspace(30, -30, 100)] + 100]
        )
        np.savetxt(os.path.join(coords_dir, "sample-1.txt"), star)
        with open(os.path.join(coords_dir, "sample-2.txt"), "w") as handle:
            handle.write("x\ty\n")
            for x, y in square:
                handle.write(f"{x}\t{y}\n")

        coord_out = os.path.join(tmp, "coords_npy")
        written = convert_outline_coords_to_npy(coords_dir, coord_out)
        assert len(written) == 1, f"both cells belong to one group, got {len(written)} frames"
        coord_masks = np.load(written[0], allow_pickle=True).item()["masks"]
        assert ncells(coord_masks) == 2, f"expected 2 cells, got {ncells(coord_masks)}"
        for label in (1, 2):
            _, parts = ndimage_label(coord_masks == label)
            assert parts == 1, f"cell {label} was split into {parts} pieces"
        # The star is lobed, the square is convex: solidity must separate them,
        # and the square's must be ~1.
        solidity = {}
        for label in (1, 2):
            found, _ = cv2.findContours((coord_masks == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            outline = max(found, key=cv2.contourArea)
            solidity[label] = cv2.contourArea(outline) / cv2.contourArea(cv2.convexHull(outline))
        print("coordinate import: cells=%d solidity star=%.3f square=%.3f"
              % (ncells(coord_masks), min(solidity.values()), max(solidity.values())))
        assert max(solidity.values()) > 0.99, "the convex square must have solidity ~1"
        assert min(solidity.values()) < 0.75, "the 6-lobed star must have clearly sub-1 solidity"

        # --- the movie-shape guard: warns, never refuses
    mismatched = warn_if_not_a_movie("mock", {1: np.zeros((209, 209)), 2: np.zeros((210, 210))})
    assert mismatched and mismatched["reason"] == "shape_mismatch", mismatched
    assert warn_if_not_a_movie("mock", {1: np.zeros((100, 100)), 2: np.zeros((100, 100))}) is None
    print("\nmovie guard: differing frame shapes warn (%s); equal shapes do not."
          % mismatched["reason"])
    print("  KNOWN LIMIT: `pupae_E-cad`'s `_P1`/`_P2` are two specimens but share a long prefix")
    print("  and differ by one digit, so the filename heuristic cannot see them. Only the shape")
    print("  check could, and only if the images differ in size. This informs; it is not a")
    print("  classifier, and it never blocks.")

    # --- D (R19): the minimum-area filter, and its interaction with
    # `_close_label_gaps`.
    grid = np.zeros((40, 60), dtype=int)
    grid[5:25, 5:25] = 1          # a real cell
    grid[5:25, 30:50] = 2         # another
    grid[30, 30] = 3              # a 1 px vertex artefact
    kept, removed = apply_min_cell_area(grid, 19)
    assert removed == [3], removed
    assert sorted(set(kept.ravel()) - {0}) == [1, 2], sorted(set(kept.ravel()))
    assert (kept[30, 30] == 0), "a removed label must become BACKGROUND"
    assert int((kept == 1).sum()) == int((grid == 1).sum()), \
        "removing an artefact must not donate its pixels to a neighbour"
    print("\nD: a 1 px vertex artefact is removed to BACKGROUND, and the neighbouring")
    print("  cells' areas are unchanged -- donating the pixel would move measured areas.")

    # A1 (R21): the filter must remove the ROW too, not only the mask label.
    # Round 19 filtered `masks` while the per-cell loop enumerated `outlines`,
    # so extraction kept emitting a row for every removed label -- and tracking,
    # which is driven by `Cells_info`, then linked all of them. And `outlines`
    # must NOT be filtered by its own pixel counts: it is a boundary ring, so an
    # area threshold meant for filled cells removes everything.
    import tempfile as _tmp
    holder = _tmp.mkdtemp()
    filled = np.zeros((40, 90), dtype=int)
    filled[5:25, 5:25] = 1          # 400 px
    filled[5:25, 30:50] = 2         # 400 px
    filled[30, 60] = 3              # 1 px artefact
    rings = thin_label_outlines(filled)
    np.save(os.path.join(holder, "f001_seg.npy"),
            {"masks": filled, "outlines": rings}, allow_pickle=True)

    set_min_cell_area(0)
    all_rows, _ = extract_cell_info(holder, output_path=None)
    set_min_cell_area(19)
    filtered_rows, _ = extract_cell_info(holder, output_path=None)
    set_min_cell_area(0)
    assert len(all_rows) == 3, len(all_rows)
    assert len(filtered_rows) == 2, (
        "the filter must drop the ROW as well as the label: %d rows" % len(filtered_rows))
    assert not any(str(i).endswith("_3") for i in filtered_rows.index), list(filtered_rows.index)
    print("  the removed label loses its Cells_info ROW too, not just its mask pixels")
    print("     (3 rows -> 2), so tracking cannot resurrect it.")

    # An area threshold applied to the boundary ring would take every cell.
    _, ring_removed = apply_min_cell_area(rings, 200)
    assert len(ring_removed) == 3, ring_removed
    set_min_cell_area(200)
    survivors, _ = extract_cell_info(holder, output_path=None)
    set_min_cell_area(0)
    assert len(survivors) == 2, (
        "the two 400 px cells must survive a 200 px threshold; filtering the boundary ring by "
        "its own pixel counts would have removed all three")
    print("     and the decision is made on the FILLED array, then applied to both:")
    print("     at 200 px the two 400 px cells survive the ring test that would have")
    print("     removed them (%d ring labels below 200 px)." % len(ring_removed))

    off, none_removed = apply_min_cell_area(grid, 0)
    assert off is grid and none_removed == [], "0 must be a true no-op"
    # The UI promises "keep >= N px": a label of exactly N px survives N.
    three_px = np.zeros((5, 5), dtype=int)
    three_px[1, 1:4] = 7
    assert apply_min_cell_area(three_px, 3)[1] == [] and apply_min_cell_area(three_px, 4)[1] == [7]
    print("  0 px is a no-op that returns the same array object, so 'off' cannot drift.")

    # `_close_label_gaps` fills separator pixels up to `_MAX_GAP_FILL_PX`. It
    # runs at TIFF->npy CONVERSION and the filter runs at LOAD, so the filter
    # cannot be undone by it -- but assert the composition rather than argue it.
    boundary = np.zeros_like(grid, dtype=bool)
    boundary[:, 25:30] = True
    closed = _close_label_gaps(kept.copy(), boundary)
    assert closed[30, 30] == 0, (
        "_close_label_gaps must not refill a removed artefact's pixel: that would silently "
        "undo the removal and move a neighbour's measured area")
    assert sorted(set(closed.ravel()) - {0}) == [1, 2], sorted(set(closed.ravel()))
    print("  and _close_label_gaps does not refill the removed pixel afterwards.")

    # --- E (R22): frame 0 enters the pipeline.
    zero_based = _tmp.mkdtemp()
    for f in (0, 1, 2):
        one = np.zeros((30, 60), dtype=int)
        one[5:20, 5:20] = 1
        one[5:20, 30:45] = 2
        np.save(os.path.join(zero_based, "f%04d_seg.npy" % f),
                {"masks": one, "outlines": thin_label_outlines(one)}, allow_pickle=True)

    # E1: a 0-numbered folder loses no frame.
    listed = list_segmentation_files(zero_based)
    assert len(listed) == 3, listed
    assert set(resolve_frame_numbers(listed).values()) == {0, 1, 2}, \
        resolve_frame_numbers(listed)
    print("\nE1 (R22): a 0-numbered folder yields 3 files and frames {0, 1, 2}.")
    print("  `>= 1` silently dropped frame 0 -- on `embryos_GBE_GAP43` that was")
    print("  `handCorrection0000`, and the movie read as 30 frames instead of 31.")

    # E2: the conservation assertion. This is the only check that can catch a
    # dropped frame: all four consumers call `list_segmentation_files`, so they
    # agree with each other no matter what it drops. Only the DIRECTORY
    # disagrees. Assert it fires, and that it names the file it lost.
    _real_resolve = resolve_frame_numbers
    try:
        globals()["resolve_frame_numbers"] = (
            lambda names: {n: i for n, i in _real_resolve(names).items() if i >= 1})
        try:
            list_segmentation_files(zero_based)
            raise SystemExit("E2: the conservation assertion did not fire")
        except AssertionError as exc:
            assert "f0000_seg.npy" in str(exc), str(exc)
            assert "1 of 3" in str(exc), str(exc)
    finally:
        globals()["resolve_frame_numbers"] = _real_resolve
    print("E2 (R22): dropping a frame raises, and the message NAMES f0000_seg.npy")
    print("  rather than only reporting that the counts disagree.")

    # E3: frame 0 really produces rows.
    set_min_cell_area(0)
    zero_rows, zero_frames = extract_cell_info(zero_based, output_path=None)
    assert zero_frames == 3, zero_frames
    assert any(str(i).startswith("Cell0_") for i in zero_rows.index), list(zero_rows.index)
    assert len(zero_rows) == 6, len(zero_rows)
    print("E3 (R22): extraction over that folder emits Cell0_* rows (3 frames, 6 cells).")

    # E4 (R22 D): the shared loader is what decides, and a boundary ring is not
    # a label array. Extraction used to read `dat.get("masks")` directly, so an
    # old outlines-only file was silently left unfiltered while `cell_tracking`
    # filtered it.
    ring_only = _tmp.mkdtemp()
    np.save(os.path.join(ring_only, "r0001_seg.npy"),
            {"outlines": rings}, allow_pickle=True)
    assert load_label_array(os.path.join(ring_only, "r0001_seg.npy")) is None, \
        "a boundary ring must not be returned as a filled label array"
    filled_only = _tmp.mkdtemp()
    np.save(os.path.join(filled_only, "g0001_seg.npy"),
            {"outlines": filled}, allow_pickle=True)
    recovered, ring_removed_2 = load_label_array(
        os.path.join(filled_only, "g0001_seg.npy"), 19, with_removed=True)
    assert recovered is not None and ring_removed_2 == [3], ring_removed_2
    print("E4 (R22 D): `load_label_array` falls back to `outlines` only when it is")
    print("  filled -- a ring returns None, an old filled-under-outlines file filters.")

    # --- R24: the two restored columns land in their named slots, and the
    # wider sheet survives the Excel round trip the product does: written by
    # Extract, re-read by "load existing extraction", then the leading-edge
    # column written back into the same file. The row builders are positional,
    # so a value in the wrong slot would pass every length check -- assert the
    # values by NAME against quantities they must equal.
    import manual_correction_classification
    set_min_cell_area(0)
    xlsx_dir = _tmp.mkdtemp()
    written, _ = extract_cell_info(holder, output_path=xlsx_dir)
    assert list(written.columns) == CELLS_INFO_COLUMNS, list(written.columns)
    squares = written.loc[["Cell1_1", "Cell1_2"]]
    expected_c = (4 * math.pi * squares["Cell Area"] / squares["Cell Perimeter"] ** 2).round(2)
    assert np.allclose(squares["Circularity"], expected_c, atol=0.006), (squares["Circularity"], expected_c)
    assert squares["Fitted Ellipse Angle"].between(0, 90).all(), squares["Fitted Ellipse Angle"]
    # The neighbours of both inserted slots: a 20x20 square has an inscribed
    # radius of 10, and a circumscribed radius of 13 -- the contour runs through
    # pixel centres, 19 px apart, so the half-diagonal is 13.4, truncated.
    assert (squares["Circumscribed Circle Radius"] == 13).all(), squares["Circumscribed Circle Radius"]
    assert (squares["Inscribed Circle Radius"] == 10).all(), squares["Inscribed Circle Radius"]
    assert (squares["Cell Right Boundary"] - squares["Cell Left Boundary"] == 19).all()

    path = os.path.join(xlsx_dir, "Cells_info.xlsx")
    loaded = pd.read_excel(path, sheet_name="Cell")          # as `load_existing_extraction`
    assert list(loaded.columns) == ["Cell Index"] + CELLS_INFO_COLUMNS, list(loaded.columns)
    pd.testing.assert_frame_equal(loaded.set_index("Cell Index"), written, check_dtype=False)
    loaded["leading_edge"] = [1.0, -1.0, 1.0][: len(loaded)]
    manual_correction_classification.save_leading_edge_column(loaded, path)
    again = pd.read_excel(path, sheet_name="Cell")
    assert list(again.columns) == ["Cell Index"] + CELLS_INFO_COLUMNS + ["leading_edge"], list(again.columns)
    pd.testing.assert_frame_equal(again.drop(columns="leading_edge"), loaded.drop(columns="leading_edge"),
                                  check_dtype=False)
    assert list(again["leading_edge"]) == list(loaded["leading_edge"]), again["leading_edge"]
    # --- R24 stale outlines. The mask is the segmentation; a stored ring that
    # has lost most of its pixels (Cellpose's GUI does not always re-derive it
    # after a manual edit) must not decide the row. Cell 1 keeps ONE ring pixel
    # of its 76; its row must come out exactly as from an intact file. Cell 2 is
    # a full-width strip along the top edge in the TIFF convention
    # (`thin_label_outlines` has no ring pixel on the image border), so its
    # stored ring is 60 of a 136 px contour -- under half, yet not stale. It must
    # be left alone; counting the border, it was rebuilt (Demo_Seg_QC corners).
    labels_r = np.zeros((60, 60), dtype=np.int32)
    labels_r[0:10, :] = 2
    labels_r[20:40, 20:40] = 1
    intact_dir, stale_dir = _tmp.mkdtemp(), _tmp.mkdtemp()
    ring_r = thin_label_outlines(labels_r)
    np.save(os.path.join(intact_dir, "s0001_seg.npy"), {"masks": labels_r, "outlines": ring_r}, allow_pickle=True)
    stale_ring = ring_r.copy()
    stale_ring[stale_ring == 1] = 0
    stale_ring[20, 30] = 1
    np.save(os.path.join(stale_dir, "s0001_seg.npy"), {"masks": labels_r, "outlines": stale_ring}, allow_pickle=True)
    intact_rows, _ = extract_cell_info(intact_dir)
    stale_rows, _ = extract_cell_info(stale_dir)
    pd.testing.assert_frame_equal(stale_rows, intact_rows)
    assert stale_rows.loc["Cell1_1", "Cell Area"] == 400, stale_rows.loc["Cell1_1"]
    assert not (stale_rows.loc["Cell1_1"] == -1).any(), stale_rows.loc["Cell1_1"]
    # Both files store the same ring for the strip, so the equality above cannot
    # see it being rebuilt -- assert its values. Its stored ring is the bottom
    # row alone (y = 9); rebuilt from the contour, its centroid moves to ~4.5.
    strip = stale_rows.loc["Cell1_2"]
    assert strip["Center Y Coordinate"] == 9.0 and strip["Maximum Vertical Length"] == 0, strip
    print("\nR24: a 1-px stale ring is measured from the mask (area 400, no sentinel), row-")
    print("  identical to the intact file; a TIFF-convention strip on the image border")
    print("  (stored ring 60 of a 136 px contour) is left alone.")

    # --- R26 B1: the edge test became a measurement.
    # The predicate is a verbatim v1 port and stays, but it is now derived from
    # the column, so the two cannot drift apart. Asserted over a grid rather
    # than one case, because the far edges use `width - max_x` (one more than
    # the geometric distance) and a hand-picked example would hide that.
    for h, w in ((40, 60), (353, 422)):
        for y0_, y1_, x0_, x1_ in ((0, 5, 0, 5), (3, 9, 10, 20), (4, 9, 10, 20),
                                   (10, 20, w - 4, w - 1), (10, 20, 10, 20)):
            pix = (np.array([y0_, y1_]), np.array([x0_, x1_]))
            d = edge_distance(pix, h, w)
            for margin in range(0, 7):
                assert is_edge_cell(pix, h, w, margin=margin) == (d <= margin), (
                    pix, h, w, margin, d)
    print("\nR26: `is_edge_cell(..., margin)` == `edge_distance(...) <= margin` over a grid")
    print("  of boxes and margins, so the column and the ported predicate cannot diverge.")

    # The flicker itself: the same cell 3 px from the top in one frame and 4 px
    # in the next. Both frames used to give it opposite fates -- excluded, then
    # kept -- which minted an orphan and a new track. Now both get a row, and
    # the distance says which side of any margin each one is on.
    flicker = _tmp.mkdtemp()
    for frame, top in ((1, 3), (2, 4)):
        lab = np.zeros((60, 60), dtype=np.int32)
        lab[top:30, 10:30] = 1          # bbox starts at row `top`
        lab[40:50, 40:50] = 2           # an interior cell, far from every border
        np.save(os.path.join(flicker, "f%04d_seg.npy" % frame),
                {"masks": lab, "outlines": thin_label_outlines(lab)}, allow_pickle=True)
    kept, _ = extract_cell_info(flicker, exclude_edge_cells=True)
    assert list(kept.index) == ["Cell1_1", "Cell1_2", "Cell2_1", "Cell2_2"], list(kept.index)
    assert kept.loc["Cell1_1", "Edge Distance"] == 3, kept.loc["Cell1_1", "Edge Distance"]
    assert kept.loc["Cell2_1", "Edge Distance"] == 4, kept.loc["Cell2_1", "Edge Distance"]
    # 11, not 10: rows 40-49 of a 60-row frame are 10 px clear of the bottom,
    # and the far edges count `height - max_y` (see `edge_distance`). Asserted
    # as 11 rather than quietly loosened -- the asymmetry is inherited from the
    # v1 predicate this column has to agree with.
    assert kept.loc["Cell1_2", "Edge Distance"] == 11, kept.loc["Cell1_2", "Edge Distance"]
    # The old behaviour is still expressible, but now at the consumer's choice.
    assert list(kept.index[kept["Edge Distance"] > 3]) == ["Cell1_2", "Cell2_1", "Cell2_2"]
    tracking_rows, _ = extract_cell_info(flicker, tracking_only=True)
    assert list(tracking_rows["Edge Distance"]) == [3, 11, 4, 11], list(tracking_rows["Edge Distance"])
    print("R26: a cell 3 px from the border in frame 1 and 4 px in frame 2 keeps a row in")
    print("  BOTH (it used to flip between dropped and kept, orphaning a cell each time);")
    print("  the tracking-only path fills the column too.")

    # B3: the new column must not reach any feature set.
    assert CELLS_INFO_COLUMNS[-1] == "Edge Distance", CELLS_INFO_COLUMNS[-1]
    assert "Edge Distance" not in ML_FEATURE_COLUMNS, "Edge Distance leaked into ML_FEATURE_COLUMNS"
    print("  and it is the LAST column (positional writers) and in no feature set.")

    print("\nR24: Circularity/Fitted Ellipse Angle land in their named slots (C = 4*pi*A/P^2,")
    print("  angle in 0-90, neighbours intact), and the %d-column sheet round-trips through"
          % len(CELLS_INFO_COLUMNS))
    print("  load-existing and the leading-edge write-back with every column unchanged.")

    print("\nAll data_input self-tests passed.")

    parser = argparse.ArgumentParser(description="Standalone debug entry for v2.data_input")
    parser.add_argument(
        "--self-test", action="store_true", help="Run the mask-interpretation checks and exit.",
    )
    parser.add_argument(
        "--mask-folder",
        help="Folder of segmentation files -- either Cellpose .npy output, or .tif/.tiff masks/skeletons "
             "(auto-detected and converted to .npy via prepare_segmentation_folder if needed).",
    )
    parser.add_argument("--image-folder", default=None, help="Only used when converting .tif/.tiff masks.")
    parser.add_argument(
        "--mask-type", default="auto", choices=["auto", "labeled", "skeleton", "binary_mask"],
        help="How to interpret .tif/.tiff masks; ignored for .npy input.",
    )
    parser.add_argument(
        "--background-area-ratio", type=float, default=10.0,
        help="Discard connected components more than this many times the maximum cell area (unsegmented "
             "background cleanup); pass 0 or negative to disable. Ignored for .npy input.", 
    )
    parser.add_argument("--npy-output-folder", default=None, help="Where converted .npy files are written.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--exclude-edge-cells", action="store_true")
    args = parser.parse_args()

    if args.self_test or not args.mask_folder:
        _self_test()
        raise SystemExit(0)

    logging.basicConfig(level=logging.DEBUG)

    def _print_progress(message: str, percent: int) -> None:
        print(f"[{percent:3d}%] {message}")

    detected_format = detect_mask_format(args.mask_folder)
    print(f"Detected mask format in {args.mask_folder}: {detected_format!r}")

    npy_folder = prepare_segmentation_folder(
        args.mask_folder,
        npy_output_folder=args.npy_output_folder,
        image_folder=args.image_folder,
        mask_type=args.mask_type,
        background_area_ratio=args.background_area_ratio if args.background_area_ratio > 0 else None,
        progress_cb=_print_progress,
    )
    if npy_folder != args.mask_folder:
        print(f"Converted .tif/.tiff masks -> {npy_folder}")

    df, n_frames = extract_cell_info(
        npy_folder, args.output, args.exclude_edge_cells, progress_cb=_print_progress
    )
    print(f"\nProcessed {n_frames} frame(s); extracted {len(df)} cells total.")
    print(df.head())
    invalid = df[(df == -1).any(axis=1)]
    print(f"\n{len(invalid)} row(s) contain -1 sentinel values.")
