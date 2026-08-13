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
from scipy.ndimage import label as ndimage_label

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
    """List `.npy` segmentation filenames in `npy_folder`, ordered by frame
    number and filtered to frame number >= 1. Returns filenames only (not
    full paths), matching src/gui_main.py:970-980.

    Digit-less filenames are kept (and numbered positionally by
    `resolve_frame_numbers`) rather than dropped by the `>= 1` filter, which
    `inf` used to pass by accident anyway."""
    npy_files = [f for f in os.listdir(npy_folder) if f.endswith(".npy")]
    frame_numbers = resolve_frame_numbers(npy_files)
    return [f for f in frame_numbers if frame_numbers[f] >= 1]


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


def _drop_oversized_components(labeled: np.ndarray, background_area_ratio: float) -> np.ndarray:
    """Relabel to background (0) any connected component whose area exceeds
    `background_area_ratio` times the MAXIMUM component area.

    Connected-component labeling of a skeleton's complement (or of a raw
    binary mask) doesn't know which components are real cells and which are
    unsegmented "outside the tissue" background -- if the boundary network
    doesn't fully enclose the image, everything outside the tissue can merge
    into one giant blob that dwarfs typical cells. Filtering by a large
    multiple of the maximum observed cell area is a conservative test that
    compares each component against the largest detected cell (instead of the
    median), which may be preferred in datasets where cell sizes vary widely.
    Note: when an unsegmented background region itself becomes the largest
    component, this max-based test will be less aggressive than a median-based
    test and may fail to remove that background; choose `background_area_ratio`
    accordingly.
    """
    areas = np.bincount(labeled.ravel())
    areas[0] = 0
    nonzero_areas = areas[areas > 0]
    if len(nonzero_areas) == 0:
        return labeled

    max_area = np.max(nonzero_areas)
    oversized_labels = np.nonzero(areas > background_area_ratio * max_area)[0]
    if len(oversized_labels) == 0:
        return labeled

    result = labeled.copy()
    result[np.isin(labeled, oversized_labels)] = 0
    return result


def _close_label_gaps(labeled: np.ndarray, fillable: np.ndarray) -> np.ndarray:
    """Assign every `fillable` pixel to the nearest labeled cell, so cells
    that a drawn boundary line separated end up directly adjacent (Cellpose's
    convention, where the border between two cells is just a label change).

    `scipy.ndimage.distance_transform_edt(..., return_indices=True)` gives the
    coordinates of the nearest non-background pixel for the whole image in one
    pass, which is the nearest-label assignment we want. (`skimage`'s
    `expand_labels` would be the obvious tool, but scikit-image is not
    installed and is referenced nowhere in this repo -- scipy already is.)

    `fillable` should be only the pixels known to be a separator; passing the
    whole background would flood genuinely unsegmented regions into cells.
    """
    if not fillable.any():
        return labeled
    _distance, (nearest_y, nearest_x) = ndimage_distance_transform_edt(
        labeled == 0, return_indices=True,
    )
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

    `background_area_ratio` (default 10, `None` to disable) discards
    connected components more than that many times larger than the median
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
    frame_number_by_file = resolve_frame_numbers(npy_files)

    for idx, filename in enumerate(npy_files):
        basename, ext = os.path.splitext(filename)
        if ext != ".npy":
            continue
        dat = np.load(os.path.join(npy_folder, filename), allow_pickle=True).item()
        outlines = dat["outlines"]
        height, width = outlines.shape
        fig_num += 1
        cells_info_fig = pd.DataFrame(columns=CELLS_INFO_COLUMNS)
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
