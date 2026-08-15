"""No-reference segmentation quality metrics, for the Extract step's QC view.

"No-reference" because there is no ground-truth segmentation to compare
against -- these judge a mask by its own internal consistency and its
agreement with the raw image. The best-known cell-specific work in this space
(Chen & Murphy, "Evaluation of cell segmentation methods without reference
segmentations", MBoC 2022) PCA-combines 14 metrics, but assumes *multichannel*
images containing *several cell types with differing expression* -- neither
holds for a single-channel junction marker like Ecad-GFP, so its weighting
would not transfer. The metrics here are chosen for that kind of image
instead, and are kept separate and individually selectable so a flagged cell
comes with a reason rather than an opaque number.

Cellpose's own `cellprob` is deliberately NOT used even though it is present
in this project's `.npy` files (`flows[4][2]`). Two reasons: the `flows`
layout differs between Cellpose versions *within this project* (one dataset
stores `flows[0]` as an ndarray, another as a list), and more importantly it
records Cellpose's original guess, so it is stale for exactly the cells a
human later hand-corrected -- which in this project's data is 8-43 cells per
frame. Everything here is computed from the raw image and the final mask, so
it stays correct after hand-correction and independent of Cellpose's version.

Every function returns `{raw_cell_label: score}` with **score in [0, 1] where
1 is good and 0 is suspicious**, matching `adjacency.environment_scores`'
shape and direction so `rendering.continuous_lut_colors` consumes them
unchanged and "dark = inspect this" means the same thing in both views.
Cells that cannot be judged are omitted rather than scored 0, the same
convention (and for the same reason) as the environment score.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import ndimage

import adjacency

# Metric keys the UI offers, in display order.
QC_METRIC_NAMES = [
    "junction_contrast", "boundary_contrast", "morphology", "neighbor_count", "measurement_failures",
]

# Fallback minimum interface size, used only when a frame has too few
# junctions to take a percentile from. Normally the threshold is DERIVED per
# project by `suggest_min_interface_px` -- junction length scales with image
# resolution, and the 25th percentile spans 14-40 px across this project's
# datasets, so no single constant serves them all.
#
# Whatever the value, it is not cosmetic: on `Demo_Seg_QC`'s WT frame the two
# junctions that out-ranked the real segmentation error were 15 and 18 band px
# stubs beside vertices. Excluding them moved the genuine error from 3rd-worst
# to worst-of-159, with the next junction 0.065 away.
MIN_INTERFACE_PX = 20

# Percentile of a frame's own junction lengths used as the default minimum.
# 25 rather than a lower value on measured grounds: at the 5th percentile the
# *corrected* Demo_Seg_QC frame still reports 2 flagged junctions (the vertex
# stubs above, one of which visibly traces a real but dim membrane), while at
# the 25th the error frame flags exactly its one real error and the corrected
# frame flags none. Clean datasets flag nothing either way, so a lower
# percentile buys only false positives.
DEFAULT_MIN_INTERFACE_PERCENTILE = 25

# How far a cell may be grown into background to meet a neighbour before their
# shared border is measured. See `_bridge_gaps`.
MAX_GAP_PX = 3

# Per-metric display window and triage threshold. These metrics are all
# "1 = good", but they do NOT share a scale: `boundary_contrast` is a physical
# intensity ratio whose realistic range on epithelial junction data is roughly
# 0.05-0.6 (measured across this project's datasets), while the other three are
# constructed to span the full 0-1. Rendering them all against a fixed 0-1 LUT
# would push every cell into the dark half of the colormap and a shared 0.5
# threshold would flag ~85% of cells as suspect.
#
# `vmin`/`vmax` window the LUT only -- they do NOT rescale per frame, so the
# cross-frame comparability that `rendering.continuous_lut_colors` deliberately
# preserves still holds. `flag_below` is a triage cutoff for the frame summary;
# nothing is ever discarded on its basis.
QC_METRIC_DISPLAY = {
    # vmin is pinned at 0 on physical grounds, not fitted: a real membrane
    # must be BRIGHTER than the cytoplasm it separates, so contrast <= 0 means
    # "there is no membrane here" regardless of dataset or exposure. Anything
    # at or below 0 therefore renders at the very bottom of the LUT.
    "junction_contrast": {"vmin": 0.0, "vmax": 0.25, "flag_below": 0.02},
    "boundary_contrast": {"vmin": 0.0, "vmax": 0.6, "flag_below": 0.25},
    "morphology": {"vmin": 0.0, "vmax": 1.0, "flag_below": 0.5},
    "neighbor_count": {"vmin": 0.0, "vmax": 1.0, "flag_below": 0.5},
    "measurement_failures": {"vmin": 0.0, "vmax": 1.0, "flag_below": 0.5},
}

# Neighbor count a cell in a well-packed epithelium tends toward. Deviation
# from it is a standard packing/segmentation signal: a merged cell has too
# many neighbors, a fragment too few.
_HEXAGONAL_NEIGHBORS = 6


def fit_display_window(metric: str, scores: Dict[int, float]) -> dict:
    """Adapt `metric`'s LUT window to the scores actually seen in a project,
    falling back to `QC_METRIC_DISPLAY` when there's nothing to fit.

    Only the two *contrast* metrics are refit. The other three are constructed
    to span 0-1 by definition, but a contrast is a physical intensity ratio
    whose achievable range depends on the marker, exposure and background of
    the acquisition, and it varies far more than a single hardcoded window can
    cover: median `junction_contrast` is 0.14 on `Demo_Seg_QC` but 0.52 on
    `AS_dorsal view_E-cad` and 0.45 on `pupae_E-cad`. A window sized for one
    saturates the others -- every junction pinned to the top of the colormap,
    or (as with the original 0.05-0.60 `boundary_contrast` default against
    `Demo_Seg_QC`'s 0.07 ceiling) every cell pinned to the bottom and flagged.

    **`vmin` stays at 0 and is never fitted.** For a contrast that is a
    physically meaningful zero, not a convenience: a real membrane is brighter
    than the cytoplasm it separates, so <= 0 means "there is no membrane
    here". Fitting `vmin` would turn that absolute statement into a relative
    one and let the worst junction in a clean frame look as alarming as a
    genuine false junction.

    The window is still FIXED per project, not per frame -- the caller fits
    it once and reuses it -- so frames stay comparable to one another, which
    is the property the fixed-range LUT exists to protect. `flag_below`
    keeps the same proportion of the window as the measured default, so the
    triage rate stays comparable across datasets too.
    """
    base = dict(QC_METRIC_DISPLAY.get(metric, {"vmin": 0.0, "vmax": 1.0, "flag_below": 0.5}))
    if metric not in ("boundary_contrast", "junction_contrast") or not scores:
        return base

    values = np.array(list(scores.values()), dtype=float)
    # 98th percentile, not max: one cell sitting on an unusually bright
    # junction shouldn't compress everything else.
    vmax = float(np.percentile(values, 98))
    if not np.isfinite(vmax) or vmax <= 0:
        return base

    vmax = float(np.clip(vmax, 0.05, 1.0))
    flag_ratio = base["flag_below"] / base["vmax"] if base.get("vmax") else 0.0
    return {"vmin": 0.0, "vmax": vmax, "flag_below": vmax * flag_ratio}


def _cells_info_for_frame(cells_info: pd.DataFrame, frame_number: int) -> pd.DataFrame:
    """Rows of `cells_info` belonging to `frame_number`, indexed by raw cell
    label. Accepts `cells_info` either indexed by `Cell Index` or carrying it
    as a plain column (`api.py` holds the `.reset_index()`ed form)."""
    frame = cells_info
    if "Cell Index" in frame.columns:
        frame = frame.set_index("Cell Index")
    frame = frame[frame.index.str.startswith(f"Cell{frame_number}_")]
    labels = frame.index.str.extract(rf"Cell{frame_number}_(\d+)$")[0]
    frame = frame[labels.notna().values]
    return frame.set_index(labels.dropna().astype(int).values)


def boundary_contrast(
    raw_image: np.ndarray, fill_labels: np.ndarray, ring_px: int = 2,
) -> Dict[int, float]:
    """How well each cell's outline sits on the bright membrane signal:
    `(mean ring - mean interior) / (mean ring + mean interior)`, clipped to
    `[0, 1]`, where the "ring" is the cell's own border band (its mask minus
    an `ring_px` erosion) and the interior is what remains.

    For a junction marker (Ecad-GFP, and the boundary-skeleton TIFFs other
    tools export) a correctly placed outline lies on the bright membrane, so
    a high value means the segmentation agrees with the image and a low one
    means the boundary was drawn where there is no membrane -- a merged pair,
    a spurious division of one cell, or a boundary that drifted.

    `raw_image` must be the **unmodified** array, not
    `rendering.load_base_image`'s output: that helper max-normalizes per
    frame to uint8, which makes values incomparable between frames and
    discards 8 bits of a 16-bit acquisition. `data_input._find_matching_image`
    returns the array untouched. A colour image is reduced to its mean
    channel here.

    Cells too small to have both a ring and an interior at this erosion depth
    are omitted (unjudgeable, not bad).
    """
    image = raw_image.astype(np.float32)
    if image.ndim == 3:
        image = image.mean(axis=2)

    scores: Dict[int, float] = {}
    height, width = fill_labels.shape
    for index, bbox in enumerate(ndimage.find_objects(fill_labels)):
        if bbox is None:
            continue
        label = index + 1
        r0 = max(bbox[0].start - ring_px, 0)
        r1 = min(bbox[0].stop + ring_px, height)
        c0 = max(bbox[1].start - ring_px, 0)
        c1 = min(bbox[1].stop + ring_px, width)

        mask = fill_labels[r0:r1, c0:c1] == label
        window = image[r0:r1, c0:c1]
        interior = ndimage.binary_erosion(mask, iterations=ring_px)
        ring = mask & ~interior
        if interior.sum() < 5 or ring.sum() < 5:
            continue

        ring_mean = float(window[ring].mean())
        interior_mean = float(window[interior].mean())
        total = ring_mean + interior_mean
        if total <= 0:
            continue
        scores[label] = float(np.clip((ring_mean - interior_mean) / total, 0.0, 1.0))

    return scores


def _bridge_gaps(fill_labels: np.ndarray, max_gap_px: int) -> np.ndarray:
    """Grow each cell into the background by up to `max_gap_px`, so cells that
    a thin gap keeps apart end up sharing a border.

    Needed because "adjacent" is not a given even in Cellpose output: on
    `test_data/pupae_E-cad` **no two of the 826 cells touch anywhere** (0
    4-adjacent pixel pairs), so a strictly touching-only interface search
    finds nothing at all. The gap between two such cells is exactly where the
    membrane sits, so bridging it puts the interface band on the signal we
    want to measure.

    Bounded on purpose: a distance limit keeps cells from reaching across the
    genuinely unsegmented background that surrounds the tissue (67-71% of the
    frame in this project's Cellpose datasets).
    """
    background = fill_labels == 0
    if not background.any() or max_gap_px <= 0:
        return fill_labels
    distance, (nearest_y, nearest_x) = ndimage.distance_transform_edt(background, return_indices=True)
    within_reach = background & (distance <= max_gap_px)
    bridged = fill_labels.copy()
    bridged[within_reach] = fill_labels[nearest_y[within_reach], nearest_x[within_reach]]
    return bridged


def _interface_pixels(fill_labels: np.ndarray) -> Dict[tuple, np.ndarray]:
    """`{(low_label, high_label): flat pixel indices}` for every pair of
    touching cells -- the 2-px band straddling their shared border.

    One vectorised pass over the two shift directions rather than a
    boolean-mask sweep per pair: a frame with ~400 interfaces would otherwise
    walk the whole image 400 times.
    """
    flat_index = np.arange(fill_labels.size).reshape(fill_labels.shape)
    stride = int(fill_labels.max()) + 1

    keys, indices = [], []
    for first, second in (
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
    ):
        left, right = fill_labels[first], fill_labels[second]
        touching = (left != right) & (left > 0) & (right > 0)
        if not touching.any():
            continue
        a = left[touching].astype(np.int64)
        b = right[touching].astype(np.int64)
        key = np.minimum(a, b) * stride + np.maximum(a, b)
        # Both sides of the border belong to the band.
        keys.append(np.concatenate([key, key]))
        indices.append(np.concatenate([flat_index[first][touching], flat_index[second][touching]]))

    if not keys:
        return {}

    key = np.concatenate(keys)
    index = np.concatenate(indices)
    order = np.argsort(key, kind="stable")
    key, index = key[order], index[order]
    boundaries = np.flatnonzero(np.diff(key)) + 1

    interfaces = {}
    for chunk_key, chunk_index in zip(np.split(key, boundaries), np.split(index, boundaries)):
        pair = (int(chunk_key[0] // stride), int(chunk_key[0] % stride))
        interfaces[pair] = np.unique(chunk_index)
    return interfaces


def suggest_min_interface_px(
    fill_labels: np.ndarray,
    percentile: float = DEFAULT_MIN_INTERFACE_PERCENTILE,
    max_gap_px: int = MAX_GAP_PX,
) -> int:
    """A minimum interface size derived from this frame's own junctions: the
    `percentile`-th of their band-pixel lengths.

    Junction length in pixels scales with image resolution, so a fixed
    threshold is either too strict on a small frame or too permissive on a
    large one -- measured across this project the 25th percentile is 14 px on
    `pupae_E-cad`, 18 on `embryos_GBE_GAP43`, 20 on `Demo_Seg_QC` and
    `AS_LE_lateral`, and 40 on `AS_dorsal view_E-cad`. Deriving it makes the
    metric resolution-independent without asking the user for a pixel size.

    Falls back to `MIN_INTERFACE_PX` when there are too few junctions for a
    percentile to mean anything.
    """
    sizes = np.array([pixels.size for pixels in _interface_pixels(_bridge_gaps(fill_labels, max_gap_px)).values()])
    if sizes.size < 8:
        return MIN_INTERFACE_PX
    return max(1, int(np.percentile(sizes, percentile)))


def _resolve_min_interface_px(
    fill_labels: np.ndarray, min_interface_px: Optional[int], max_gap_px: int,
) -> int:
    return suggest_min_interface_px(fill_labels, max_gap_px=max_gap_px) if min_interface_px is None \
        else int(min_interface_px)


def _interior_levels(raw_image: np.ndarray, fill_labels: np.ndarray, erode_px: int = 2) -> Dict[int, float]:
    """`{label: median interior intensity}` -- the cytoplasm level each cell's
    junctions are judged against."""
    levels = {}
    for index, bbox in enumerate(ndimage.find_objects(fill_labels)):
        if bbox is None:
            continue
        label = index + 1
        mask = fill_labels[bbox] == label
        window = raw_image[bbox]
        interior = ndimage.binary_erosion(mask, iterations=erode_px)
        levels[label] = float(np.median(window[interior if interior.sum() >= 5 else mask]))
    return levels


def junction_contrast(
    raw_image: np.ndarray, fill_labels: np.ndarray, min_interface_px: Optional[int] = None,
    max_gap_px: int = MAX_GAP_PX,
) -> Dict[tuple, float]:
    """`{(cell_a, cell_b): contrast}` for each shared border: how much
    brighter the border is than the cytoplasm on either side of it, as
    `(border - interior) / (border + interior)`.

    This is the metric that catches a *false* junction -- a boundary drawn
    across a cell where no membrane exists. `boundary_contrast` averages over
    a cell's whole perimeter, so one bad edge among six good ones is diluted
    away: on `Demo_Seg_QC`'s WT frame the two cells split by a spurious line
    scored at the 25th and 57th percentile, i.e. entirely unremarkable.
    Measured per interface instead, that same line is the worst of 159 and the
    only one in the frame with contrast <= 0.

    **Contrast <= 0 is the meaningful threshold**, and it is absolute rather
    than relative: a real membrane is brighter than the cytoplasm it
    separates, so a border that is not means there is no membrane there.

    Medians, not means, along the border -- the vertices at each end are
    genuinely bright and a mean lets them mask a dark span between them.
    Interfaces smaller than `min_interface_px` are skipped; passing `None`
    derives that threshold from the frame (see `suggest_min_interface_px`).
    """
    image = raw_image.astype(np.float32)
    if image.ndim == 3:
        image = image.mean(axis=2)
    flat_image = image.ravel()

    min_interface_px = _resolve_min_interface_px(fill_labels, min_interface_px, max_gap_px)
    interiors = _interior_levels(image, fill_labels)

    scores = {}
    for (a, b), pixels in _interface_pixels(_bridge_gaps(fill_labels, max_gap_px)).items():
        if pixels.size < min_interface_px or a not in interiors or b not in interiors:
            continue
        border = float(np.median(flat_image[pixels]))
        interior = (interiors[a] + interiors[b]) / 2.0
        total = border + interior
        if total <= 0:
            continue
        scores[(a, b)] = float((border - interior) / total)
    return scores


def junction_label_array(
    fill_labels: np.ndarray, min_interface_px: Optional[int] = None, max_gap_px: int = MAX_GAP_PX,
) -> tuple:
    """`(labels, {junction_id: (cell_a, cell_b)})` where `labels` paints each
    scored interface with its own id and everything else 0.

    Lets a junction map render through the existing
    `rendering.render_score_overlay` unchanged -- it only ever asks "which
    pixels carry this key" -- so a bad *edge* can be shown in place instead of
    only tinting the two cells that share it.
    """
    min_interface_px = _resolve_min_interface_px(fill_labels, min_interface_px, max_gap_px)
    painted = np.zeros(fill_labels.shape, dtype=np.int32)
    mapping = {}
    for junction_id, (pair, pixels) in enumerate(
        sorted(_interface_pixels(_bridge_gaps(fill_labels, max_gap_px)).items()), start=1,
    ):
        if pixels.size < min_interface_px:
            continue
        painted.ravel()[pixels] = junction_id
        mapping[junction_id] = pair
    return painted, mapping


def worst_junction_per_cell(
    raw_image: np.ndarray, fill_labels: np.ndarray, min_interface_px: Optional[int] = None,
    max_gap_px: int = MAX_GAP_PX,
) -> Dict[int, float]:
    """`{label: worst contrast among that cell's junctions}` -- the per-cell
    view of `junction_contrast`.

    Worst rather than mean, deliberately: a cell with one false edge and five
    good ones is exactly the case `boundary_contrast` misses, and only the
    minimum surfaces it. Cells with no interface large enough to judge are
    omitted rather than scored 0.
    """
    worst: Dict[int, float] = {}
    for (a, b), value in junction_contrast(raw_image, fill_labels, min_interface_px, max_gap_px).items():
        for label in (a, b):
            if label not in worst or value < worst[label]:
                worst[label] = value
    return worst


def _robust_z(values: np.ndarray) -> np.ndarray:
    """|value - median| / MAD, a spread measure that a few merged or
    fragmented cells can't inflate the way a mean/std z-score would.

    Falls back to the mean absolute deviation when the MAD is exactly 0.
    That happens whenever more than half the cells share a value -- and
    dividing by it, or returning zeros, would hide precisely the outliers
    that make the MAD 0 in the first place. Only when every value is
    identical (both spreads 0) is the frame genuinely featureless, and then
    zeros are correct.
    """
    deviation = np.abs(values - np.median(values))
    mad = np.median(deviation)
    if mad > 0:
        return deviation / (1.4826 * mad)
    mean_deviation = deviation.mean()
    if mean_deviation > 0:
        return deviation / mean_deviation
    return np.zeros_like(values, dtype=float)


def morphology_outlier(
    cells_info: pd.DataFrame, frame_number: int, z_cap: float = 5.0,
) -> Dict[int, float]:
    """How typical each cell's shape is versus the rest of its own frame,
    from measurements `extract_cell_info` already stores -- no new
    measurement, and no assumption about absolute cell size (which varies by
    magnification and by tissue).

    Combines robust z-scores of `Cell Area`, `Circularity` and the
    inscribed/circumscribed radius ratio (a compactness proxy: a sliver or a
    ragged fragment has a small inscribed circle relative to its enclosing
    one). The worst of the three drives the score, so a cell that is
    plausible on two axes and wildly off on the third is still flagged.
    Score `1` = typical, `0` = at or beyond `z_cap` median-absolute
    deviations.

    Cells carrying `-1` sentinels in these columns are omitted -- those are a
    measurement failure, which `measurement_failures` reports properly rather
    than letting a sentinel masquerade as an extreme shape.
    """
    frame = _cells_info_for_frame(cells_info, frame_number)
    if frame.empty:
        return {}

    area = frame["Cell Area"].astype(float).to_numpy()
    circularity = frame["Circularity"].astype(float).to_numpy()
    inscribed = frame["Inscribed Circle Radius"].astype(float).to_numpy()
    circumscribed = frame["Circumscribed Circle Radius"].astype(float).to_numpy()

    valid = (area > 0) & (circularity > 0) & (inscribed > 0) & (circumscribed > 0)
    if valid.sum() < 3:  # too few comparable cells for a spread to mean anything
        return {}

    compactness = np.divide(inscribed, circumscribed, out=np.zeros_like(inscribed, dtype=float), where=circumscribed > 0)
    worst = np.maximum.reduce([
        _robust_z(area[valid]), _robust_z(circularity[valid]), _robust_z(compactness[valid]),
    ])

    labels = frame.index.to_numpy()[valid]
    return {int(l): float(np.clip(1.0 - z / z_cap, 0.0, 1.0)) for l, z in zip(labels, worst)}


def neighbor_count_deviation(
    fill_labels: np.ndarray,
    neighbor_graph: Optional[dict] = None,
    max_deviation: int = 4,
) -> Dict[int, float]:
    """How far each cell's neighbor count is from the hexagonal packing an
    epithelium tends toward: score `1` at exactly 6 neighbors, falling to `0`
    at `max_deviation` away. A merged pair reads as too many neighbors, a
    fragment or an over-split cell as too few.

    Pass `neighbor_graph` to reuse an already-built
    `adjacency.build_neighbor_graph` (the webview caches one per frame);
    otherwise it is built here. `fill_labels` must be the FILLED mask -- the
    thin boundary array yields near-empty neighbor sets.

    Cells at the image border are omitted: their neighbor count is truncated
    by the field of view, not by segmentation quality, so scoring them would
    flag the entire perimeter of every frame.
    """
    graph = neighbor_graph if neighbor_graph is not None else adjacency.build_neighbor_graph(fill_labels)
    height, width = fill_labels.shape

    border_labels = set(fill_labels[0, :]) | set(fill_labels[-1, :]) | set(fill_labels[:, 0]) | set(fill_labels[:, -1])
    border_labels.discard(0)

    scores: Dict[int, float] = {}
    for label, neighbors in graph.items():
        if label in border_labels:
            continue
        deviation = abs(len(neighbors) - _HEXAGONAL_NEIGHBORS)
        scores[int(label)] = float(np.clip(1.0 - deviation / max_deviation, 0.0, 1.0))
    return scores


def measurement_failures(cells_info: pd.DataFrame, frame_number: int) -> Dict[int, float]:
    """Cells whose feature extraction fell back to a sentinel: `0` for a
    failed cell, `1` for a clean one.

    `extract_cell_info` writes `-1` for a failed ellipse fit (fewer than 5
    contour points), a zero perimeter, or zero detected contours -- but that
    last branch writes **`0`**, not `-1`, for the vertex count and both radii,
    so scanning for `-1` alone misses it. Both are checked here.

    These failures are currently only visible via a `logger.debug` line, yet
    the affected rows are fed to PCA/K-means as though the sentinels were
    real measurements. Surfacing them is the point of this metric.
    """
    frame = _cells_info_for_frame(cells_info, frame_number)
    if frame.empty:
        return {}

    sentinel_negative = ["Circularity", "P-Value", "Cell Perimeter", "Fitted Ellipse Major Axis",
                         "Fitted Ellipse Minor Axis", "Ellipse Major/Minor Axis Ratio", "Fitted Ellipse Angle"]
    sentinel_zero = ["Approximate Polygon Vertex Count", "Circumscribed Circle Radius", "Inscribed Circle Radius"]

    present_negative = [c for c in sentinel_negative if c in frame.columns]
    present_zero = [c for c in sentinel_zero if c in frame.columns]

    failed = pd.Series(False, index=frame.index)
    if present_negative:
        failed |= (frame[present_negative].astype(float) == -1).any(axis=1)
    if present_zero:
        failed |= (frame[present_zero].astype(float) == 0).any(axis=1)

    return {int(label): (0.0 if bad else 1.0) for label, bad in failed.items()}


def summarize(scores: Dict[int, float], flag_below: float = 0.5) -> dict:
    """`{"scored": n, "flagged": n, "min": x, "median": x}` for one frame's
    scores -- lets the UI point at the worst frames without stepping through
    the whole movie. `flag_below` is a display threshold only; nothing here
    or in the metrics themselves discards a cell."""
    if not scores:
        return {"scored": 0, "flagged": 0, "min": None, "median": None}
    values = np.array(list(scores.values()), dtype=float)
    return {
        "scored": int(values.size),
        "flagged": int((values < flag_below).sum()),
        "min": round(float(values.min()), 3),
        "median": round(float(np.median(values)), 3),
    }


if __name__ == "__main__":
    # --- boundary_contrast: a cell WITH a bright membrane ring vs one without.
    labels = np.zeros((40, 80), dtype=int)
    labels[5:25, 5:25] = 1   # will get a bright ring
    labels[5:25, 45:65] = 2  # uniform, no ring
    image = np.zeros((40, 80), dtype=np.float32)
    image[labels == 1] = 10.0
    ring1 = (labels == 1) & ~ndimage.binary_erosion(labels == 1, iterations=2)
    image[ring1] = 200.0
    image[labels == 2] = 100.0  # flat: ring and interior identical

    contrast = boundary_contrast(image, labels)
    print("boundary_contrast:", {k: round(v, 3) for k, v in contrast.items()})
    assert contrast[1] > 0.8, contrast[1]
    assert contrast[2] == 0.0, contrast[2]
    print("  -> membrane-aligned cell scores high, flat cell scores 0.")

    # --- junction_contrast: four tiled cells whose shared borders all sit on
    # bright membrane, except one "junction" drawn across dark cytoplasm.
    tiles = np.zeros((60, 60), dtype=int)
    tiles[0:30, 0:30] = 1
    tiles[0:30, 30:60] = 2
    tiles[30:60, 0:30] = 3
    tiles[30:60, 30:60] = 4
    membrane = np.zeros((60, 60), dtype=np.float32) + 100.0   # cytoplasm
    membrane[28:32, :] = 900.0                                 # real horizontal membrane
    membrane[:, 28:32] = 900.0                                 # real vertical membrane
    # Cell 1 is split by a line drawn where there is NO membrane signal.
    tiles[0:15, 0:30] = 1
    tiles[15:30, 0:30] = 5

    junctions = junction_contrast(membrane, tiles, min_interface_px=20)
    false_edge = junctions[(1, 5)]
    real_edges = [v for k, v in junctions.items() if k != (1, 5)]
    print("\njunction_contrast: false edge=%.3f   real edges min=%.3f"
          % (false_edge, min(real_edges)))
    assert false_edge <= 0, "an edge drawn across cytoplasm must not score positive"
    assert min(real_edges) > 0.5, real_edges
    assert false_edge < min(real_edges)
    print("  -> the fabricated junction is the only non-positive edge.")

    worst = worst_junction_per_cell(membrane, tiles, min_interface_px=20)
    assert worst[1] <= 0 and worst[5] <= 0, worst
    assert all(worst[c] > 0.5 for c in (2, 3, 4)), worst
    print("  -> both cells sharing it are flagged; the untouched cells are not.")

    derived = suggest_min_interface_px(tiles)
    print("  -> derived min interface = %d px (p%d of this frame's junctions)"
          % (derived, DEFAULT_MIN_INTERFACE_PERCENTILE))
    assert derived > 0
    assert junction_contrast(membrane, tiles) == junction_contrast(membrane, tiles, min_interface_px=derived), \
        "min_interface_px=None must be equivalent to passing the derived value"

    painted, mapping = junction_label_array(tiles, min_interface_px=20)
    assert set(mapping.values()) >= {(1, 5)}, mapping
    assert (painted > 0).sum() < (tiles > 0).sum()
    print("  -> junction map paints edges only, ready for the score overlay.")

    # Cells separated by a gap (Cellpose leaves every pupae_E-cad cell
    # detached) must still produce interfaces.
    detached = tiles.copy()
    detached[29:31, :] = 0
    detached[:, 29:31] = 0
    assert junction_contrast(membrane, detached, min_interface_px=20), "gap-separated cells must still be scored"
    print("  -> cells separated by a thin gap are still bridged and scored.")

    # --- morphology_outlier: naturally-varying cells plus one merged blob.
    rng = np.random.default_rng(0)
    rows = []
    for i in range(1, 21):
        area = float(rng.normal(100, 8)) if i < 20 else 5000.0  # cell 20 is merged
        rows.append({"Cell Index": f"Cell1_{i}", "Cell Area": area, "Circularity": 0.8,
                     "Inscribed Circle Radius": 5, "Circumscribed Circle Radius": 7})
    morph = morphology_outlier(pd.DataFrame(rows), 1)
    normals = [morph[i] for i in range(1, 20)]
    print("\nmorphology_outlier (19 normal + 1 merged): merged=%.2f  normal min=%.2f mean=%.2f"
          % (morph[20], min(normals), sum(normals) / len(normals)))
    assert morph[20] == 0.0, morph[20]
    assert min(normals) > morph[20], "the merged blob must be separated from every normal cell"
    assert sum(normals) / len(normals) > 0.8, normals
    print("  -> the merged blob scores 0; normal variation averages near 1.")

    # Degenerate frame: most cells identical makes the MAD exactly 0, which
    # must not hide the one cell that differs.
    tied = [{"Cell Index": f"Cell1_{i}", "Cell Area": 100.0, "Circularity": 0.8,
             "Inscribed Circle Radius": 5, "Circumscribed Circle Radius": 7} for i in range(1, 10)]
    tied.append({"Cell Index": "Cell1_10", "Cell Area": 5000.0, "Circularity": 0.8,
                 "Inscribed Circle Radius": 5, "Circumscribed Circle Radius": 7})
    tied_scores = morphology_outlier(pd.DataFrame(tied), 1)
    assert tied_scores[10] == 0.0, tied_scores
    assert tied_scores[1] == 1.0, tied_scores
    print("  -> zero-MAD frame still flags its outlier (mean-deviation fallback).")

    # --- neighbor_count_deviation: a 3x3 block of touching cells. The centre
    # has 8 neighbours (|8-6|=2 -> 0.5); the border ring is skipped entirely.
    grid = np.zeros((40, 40), dtype=int)
    label_id = 1
    for r in range(3):
        for c in range(3):
            grid[4 + r * 10: 13 + r * 10, 4 + c * 10: 13 + c * 10] = label_id  # 9px cells, 1px gaps
            label_id += 1
    neigh = neighbor_count_deviation(grid)
    print("\nneighbor_count_deviation:", {k: round(v, 2) for k, v in neigh.items()})
    assert 5 in neigh, "the centre cell should be scored"
    assert abs(neigh[5] - 0.5) < 1e-9, neigh[5]
    print("  -> centre cell (8 neighbours) scores 0.5; border cells omitted.")

    # --- measurement_failures: both sentinel conventions.
    fails = pd.DataFrame([
        {"Cell Index": "Cell1_1", "Circularity": 0.8, "Approximate Polygon Vertex Count": 6},
        {"Cell Index": "Cell1_2", "Circularity": -1.0, "Approximate Polygon Vertex Count": 6},   # -1 branch
        {"Cell Index": "Cell1_3", "Circularity": 0.8, "Approximate Polygon Vertex Count": 0},    # zero-contour branch
    ])
    flags = measurement_failures(fails, 1)
    print("\nmeasurement_failures:", flags)
    assert flags == {1: 1.0, 2: 0.0, 3: 0.0}, flags
    print("  -> both the -1 and the 0 sentinel conventions are caught.")

    print("\nsummary of a mixed frame:", summarize({1: 1.0, 2: 0.2, 3: 0.4, 4: 0.9}))
    print("\nAll segmentation_qc smoke tests passed.")
