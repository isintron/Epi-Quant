"""Per-cell shape descriptors for convoluted / puzzle-like cells.

Answers reviewer 2's request for "features [...] relevant when looking at more
convolve and complex cell shape (e.g. puzzle like cells)". Three families, all
dimensionless or explicitly scale-normalized so cells imaged at different
magnifications stay comparable:

- **convex hull**: solidity, convexity, shape index
- **hull deviation**: lobe count, concavity depth, neck width -- the properties
  that actually distinguish a jigsaw cell from a rounded one
- **elliptic Fourier**: orientation and elongation from harmonic 1, plus a
  scalar irregularity (see `efd.py`), and optionally the raw coefficients

Curvature-based scalars are computed here but deliberately **not** shipped as
descriptors; `CELLS_SHAPE_COLUMNS` explains why.

**Contours come from `dat['masks']`, never from `data_input.extract_cell_info`'s
reconstruction.** That function takes `np.where(outlines == id)` -- scanline
order -- and passes it to `cv2.fillPoly`, which bridges across every concavity.
On near-convex cells the error is invisible (+0.04% area on Tabernaemontana,
solidity 0.97); on the puzzle-like Adiantum fern (solidity 0.53) the median
area error is +67% and reaches +617%, with the lobes collapsing into a slab.
See `README.md`. Nothing here reuses that path.

**No code from LobeFinder or PaCeQuant is used.** PaCeQuant (in MiToBo) is
GPL-3.0 and LobeFinder carries no licence at all, so both are implemented from
their published method descriptions:

- Wu, Belteton, Pack, Szymanski & Umulis (2016), "LobeFinder: A Convex
  Hull-Based Method for Quantitative Boundary Analyses of Lobed Plant Cells",
  *Plant Physiology* 171:2331-2342.
- Moller, Poeschl, Plotner & Baurle (2017), "PaCeQuant: A Tool for
  High-Throughput Quantification of Pavement Cell Shape Characteristics",
  *Plant Physiology* 175:998-1017.

The shape index is the established structural order parameter of Bi, Lopez,
Schwarz & Manning (2015), *Nature Physics* 11:1074, and Bi, Yang, Marchetti &
Manning (2016), *Physical Review X* 6:021011 -- not a new quantity.

Failures are reported as `NaN`, never `-1`: `segmentation_qc.measurement_failures`
scans named columns for `-1` sentinels and `extract_cell_info`'s own
`(cells_info == -1).any(axis=1)` check is column-blind, so reusing `-1` here
would make a short contour indistinguishable from a real measurement failure.
"""

import logging
import math
import os
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

import data_input
import efd

# Owned by `data_input` so that module can use it too: `shape_features` imports
# `data_input`, so the dependency cannot point the other way. Re-exported here
# because this is where the perimeter reasoning is documented and where callers
# expect it.
PERIMETER_SPACING_PX = data_input.PERIMETER_SPACING_PX
contour_perimeter = data_input.contour_perimeter

logger = logging.getLogger(__name__)

# Scalar descriptors, in output order.
#
# Every one of these is stable against the smoothing width -- verified on
# `AS_dorsal view_E-cad` by recomputing at sigma = 0.01/0.02/0.04/0.08 of the
# contour, where each moved by 0.0%.
#
# **Curvature-derived scalars were deliberately dropped from this list.** A
# first version also emitted a curvature-based lobe count, concave fraction,
# bending energy and curvature standard deviation. On synthetic lobed shapes
# they are exact (the self-test still checks that), but on real segmentations
# the same sweep moved them by 70-100%: the median AS cell reported 10, 5, 2
# and finally 0 lobes as smoothing increased. Curvature is a second derivative,
# so it amplifies the pixel-scale waviness of a segmented boundary, and no
# threshold fixes it (sweeping a curvature-magnitude cutoff from 0.0 to 0.5 left
# the real-data instability untouched while the synthetics stayed exact).
# Feeding features like that into PCA is precisely the failure this work exists
# to correct, so they are not shipped.
#
# The hull- and Fourier-based descriptors have no such problem: a convex hull is
# a global construct that local noise cannot move, and truncating the Fourier
# series at order 10 is an explicit low-pass. `Lobe Count` is therefore the
# convex-hull-defect count -- which is also what LobeFinder's published method
# actually uses.
CELLS_SHAPE_COLUMNS = [
    # convex hull
    "Solidity", "Convexity", "Shape Index", "Hull Area", "Mask Area", "Mask Perimeter",
    # lobing, from hull deviation
    "Lobe Count", "Max Concavity Depth", "Mean Neck Width",
    # elliptic Fourier
    "Elongation", "Sin2Theta", "Cos2Theta", "EFD Irregularity", "Contour Points",
]

# Dimensionless subset, safe to pool across images of different magnification.
SCALE_INVARIANT_SHAPE_COLUMNS = [
    "Solidity", "Convexity", "Shape Index", "Lobe Count", "Max Concavity Depth",
    "Mean Neck Width", "Elongation", "Sin2Theta", "Cos2Theta", "EFD Irregularity",
]




def efd_coefficient_columns(order: int = efd.EFD_ORDER_DEFAULT) -> list[str]:
    """`efd_0_a ... efd_{order-1}_d`.

    Deliberately the exact naming used by
    `figures/Epiquant_ShapeAnalysis_allMutants.ipynb`, so that notebook can
    read a `Cells_shape.xlsx` with no edits -- giving a second, independent
    implementation to cross-check against.
    """
    return [f"efd_{h}_{c}" for h in range(order) for c in "abcd"]


def contour_from_mask(cell_mask: np.ndarray, offset: tuple[int, int] = (0, 0)) -> Optional[np.ndarray]:
    """Largest external contour of a filled binary mask, as an open `(N, 2)`
    float array of xy in frame coordinates.

    `CHAIN_APPROX_NONE` because curvature and the Fourier transform need every
    boundary pixel. (For `arcLength` and `approxPolyDP` the compressed
    `CHAIN_APPROX_SIMPLE` is provably equivalent, which is why
    `data_input.extract_cell_info` is left alone.)
    """
    found, _ = cv2.findContours(cell_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not found:
        return None
    largest = max(found, key=cv2.contourArea)
    return largest.reshape(-1, 2).astype(np.float64) + np.asarray(offset, dtype=np.float64)


def resample_contour(contour: np.ndarray, n_points: int = 256) -> np.ndarray:
    """Resample a closed contour to `n_points` equally spaced along arclength.

    Raw pixel chains are spaced anisotropically -- 1 px between axial
    neighbours but 1.414 px between diagonal ones -- which biases finite
    differences. Equal spacing also makes the smoothing width below mean the
    same thing on every cell.
    """
    closed = np.vstack([contour, contour[0]])
    step = np.hypot(*np.diff(closed, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(step)])
    if arc[-1] <= 0:
        return np.repeat(contour[:1], n_points, axis=0)
    target = np.linspace(0.0, arc[-1], n_points, endpoint=False)
    return np.column_stack([np.interp(target, arc, closed[:, 0]), np.interp(target, arc, closed[:, 1])])


def smooth_contour(contour: np.ndarray, sigma_points: float) -> np.ndarray:
    """Gaussian smoothing with `wrap` mode, since a cell outline is periodic --
    the default `reflect` would pin the seam and create a false curvature spike
    wherever OpenCV happened to start tracing."""
    if sigma_points <= 0:
        return contour
    return np.column_stack([
        ndimage.gaussian_filter1d(contour[:, 0], sigma_points, mode="wrap"),
        ndimage.gaussian_filter1d(contour[:, 1], sigma_points, mode="wrap"),
    ])


def hull_deviation(contour: np.ndarray, hull: np.ndarray) -> np.ndarray:
    """Perpendicular distance from every contour point to the convex hull
    boundary -- 0 where the contour lies on the hull, larger inside a
    concavity. Vectorized point-to-segment distance over the hull's edges.

    Used instead of `cv2.convexityDefects`, which requires the hull's vertex
    indices to be monotonic along the contour and raises otherwise. That is not
    an edge case: a cell with a one-pixel neck gets traced out-and-back, so the
    same coordinate appears twice at non-adjacent positions and the hull
    ordering becomes ambiguous. It happened on 4 of 2822 cells in
    `AS_dorsal view_E-cad`, and deduplicating points does not fix it because the
    repeats are not adjacent. Measuring distance to the hull needs no ordering
    at all, so it simply cannot fail that way.
    """
    start = hull
    end = np.roll(hull, -1, axis=0)
    edge = end - start                                  # (H, 2)
    length_sq = np.einsum("ij,ij->i", edge, edge)
    length_sq[length_sq == 0] = 1.0
    delta = contour[:, None, :] - start[None, :, :]     # (N, H, 2)
    t = np.clip(np.einsum("nhj,hj->nh", delta, edge) / length_sq, 0.0, 1.0)
    closest = start[None, :, :] + t[:, :, None] * edge[None, :, :]
    return np.min(np.linalg.norm(contour[:, None, :] - closest, axis=2), axis=1)


def _polygon_area(contour: np.ndarray) -> float:
    """Shoelace area; the sign carries the winding direction."""
    x, y = contour[:, 0], contour[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def signed_curvature(contour: np.ndarray) -> np.ndarray:
    """Signed curvature per point, oriented so **convex is positive**.

    The raw formula's sign follows the winding direction, and image coordinates
    are y-down, so without the correction a perfectly convex cell can come back
    100% concave depending only on which way OpenCV traced it. Multiplying by
    the sign of the shoelace area removes that dependence.
    """
    dx = np.gradient(contour[:, 0])
    dy = np.gradient(contour[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = np.power(dx * dx + dy * dy, 1.5)
    denom[denom == 0] = np.nan
    curvature = (dx * ddy - dy * ddx) / denom
    orientation = 1.0 if _polygon_area(contour) >= 0 else -1.0
    return np.nan_to_num(curvature * orientation)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True in a *circular* boolean array, as (start, length)."""
    n = len(mask)
    if mask.all():
        return [(0, n)]
    if not mask.any():
        return []
    start = int(np.argmax(~mask))  # rotate so index 0 is False -> no run wraps
    rolled = np.roll(mask, -start)
    edges = np.flatnonzero(np.diff(rolled.astype(np.int8)) != 0) + 1
    bounds = np.concatenate([[0], edges, [n]])
    out = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if rolled[lo]:
            out.append(((lo + start) % n, hi - lo))
    return out


def curvature_lobe_count(curvature: np.ndarray, min_run_fraction: float = 0.02) -> int:
    """Lobes counted as runs of convex boundary separated by concave runs.

    **Not part of the shipped descriptor set** -- see `CELLS_SHAPE_COLUMNS` for
    why (it is exact on clean synthetic outlines but 70-100% smoothing-dependent
    on real segmentations). It is kept because it is a genuinely independent
    estimator of the same quantity as the hull-defect count, derived from local
    differential geometry rather than global hull deviation, which makes it a
    strong cross-check: the self-test asserts the two agree exactly on every
    synthetic lobed shape.

    **A convex contour must score 0, not 1.** Requiring a genuine intervening
    concave run is what enforces that; thresholding curvature alone reports a
    circle as a single lobe.
    """
    n = len(curvature)
    min_run = max(2, int(round(min_run_fraction * n)))
    convex_runs = [r for r in _runs(curvature > 0) if r[1] >= min_run]
    concave_runs = [r for r in _runs(curvature < 0) if r[1] >= min_run]
    return len(convex_runs) if concave_runs else 0


def hull_features(contour: np.ndarray, perimeter_spacing: float = PERIMETER_SPACING_PX) -> dict:
    """Convex-hull descriptors plus the concavity geometry behind them.

    Area and hull area are both taken with `cv2.contourArea` on the *same*
    representation. Mixing a pixel count with a polygon area instead inflates
    solidity by roughly perimeter/2 pixels -- enough to push it above 1.0 on
    real data (medians of 1.006-1.174 were measured that way) and, worse, to
    smuggle a size dependence back into a supposedly dimensionless feature.
    """
    points = np.round(contour).astype(np.int32).reshape(-1, 1, 2)
    area = cv2.contourArea(points)
    perimeter = contour_perimeter(contour, spacing=perimeter_spacing)
    hull = cv2.convexHull(points)
    hull_area = cv2.contourArea(hull)
    # The hull is a polygon through real vertices with straight edges between
    # them, so its length is already exact -- no staircase to correct.
    hull_perimeter = cv2.arcLength(hull, True)

    out = {
        "Mask Area": float(area),
        "Mask Perimeter": float(perimeter),
        "Hull Area": float(hull_area),
        "Solidity": float(area / hull_area) if hull_area > 0 else np.nan,
        "Convexity": float(hull_perimeter / perimeter) if perimeter > 0 else np.nan,
        "Shape Index": float(perimeter / math.sqrt(area)) if area > 0 else np.nan,
        "Lobe Count": np.nan,
        "Max Concavity Depth": np.nan,
        "Mean Neck Width": np.nan,
    }

    # Lobing as deviation from the convex hull: a lobe is a protrusion, so it
    # is bounded by concavities where the boundary dips inside the hull. Depths
    # are expressed as a fraction of the equivalent radius, which is what makes
    # the descriptors scale-free.
    equivalent_radius = math.sqrt(area / math.pi) if area > 0 else 0.0
    if equivalent_radius > 0 and len(hull) >= 3:
        depth = hull_deviation(contour, hull.reshape(-1, 2).astype(np.float64)) / equivalent_radius
        # A concavity has to be deeper than 5% of the equivalent radius to
        # count, which is what keeps segmentation waviness out -- the same role
        # the depth filter plays in LobeFinder.
        concavities = [r for r in _runs(depth > 0.05) if r[1] >= 2]
        out["Lobe Count"] = float(len(concavities))
        out["Max Concavity Depth"] = float(depth.max())
        if concavities:
            # Neck width: the chord across each concavity's mouth.
            widths = [
                float(np.hypot(*(contour[(start + length - 1) % len(contour)] - contour[start])))
                for start, length in concavities
            ]
            out["Mean Neck Width"] = float(np.mean(widths) / equivalent_radius)
        else:
            out["Mean Neck Width"] = 0.0
    return out


def cell_shape_features(
    cell_mask: np.ndarray,
    offset: tuple[int, int] = (0, 0),
    efd_order: int = efd.EFD_ORDER_DEFAULT,
    n_contour_points: int = 256,
    smoothing_sigma_fraction: float = 0.02,
    min_run_fraction: float = 0.02,
    with_coefficients: bool = False,
) -> Optional[dict]:
    """Every scalar in `CELLS_SHAPE_COLUMNS` for one cell's filled binary mask,
    or `None` if no contour could be traced.

    `smoothing_sigma_fraction` and `min_run_fraction` are fractions of the
    contour, not pixel counts, so lobe detection behaves the same at any
    magnification.
    """
    contour = contour_from_mask(cell_mask, offset=offset)
    if contour is None or len(contour) < 8:
        return None

    values = {name: np.nan for name in CELLS_SHAPE_COLUMNS}
    values["Contour Points"] = float(len(contour))
    values.update(hull_features(contour))

    resampled = resample_contour(contour, n_points=n_contour_points)

    coefficients = None
    if len(contour) >= 2 * efd_order + 1:
        try:
            coefficients = efd.elliptic_fourier_descriptors(resampled, order=efd_order)
        except ValueError:
            coefficients = None
    if coefficients is not None:
        phase = efd.phase_normalize(coefficients)
        elongation, psi = efd.ellipse_from_harmonic1(phase)
        values["Elongation"] = float(elongation[0])
        values["Sin2Theta"] = float(np.sin(2 * psi[0]))
        values["Cos2Theta"] = float(np.cos(2 * psi[0]))
        values["EFD Irregularity"] = float(efd.irregularity(efd.efd_features(coefficients)))

    if with_coefficients:
        flat = coefficients.ravel() if coefficients is not None else np.full(efd_order * 4, np.nan)
        values.update(dict(zip(efd_coefficient_columns(efd_order), flat.tolist())))
    return values


def extract_shape_features(
    npy_folder: str,
    output_path: Optional[str] = None,
    efd_order: int = efd.EFD_ORDER_DEFAULT,
    n_contour_points: int = 256,
    smoothing_sigma_fraction: float = 0.02,
    min_run_fraction: float = 0.02,
    exclude_edge_cells: bool = False,
    with_coefficients: bool = False,
    progress_cb: Optional[Callable[[str, int], None]] = None,
    frames: Optional[set] = None,
) -> pd.DataFrame:
    """Shape descriptors for every cell in every frame of `npy_folder`.

    Indexed by the same `Cell{frame}_{cell_id}` strings
    `data_input.extract_cell_info` produces, so the two tables join directly.
    Written to `Cells_shape.xlsx` (sheet `'Shape'`) when `output_path` is given
    -- a **separate file** from `Cells_info.xlsx` rather than extra columns,
    because `common.save_modified_rows` maps DataFrame column position to Excel
    column position and would silently write into the wrong columns of an
    existing file if the schema grew.
    """
    columns = list(CELLS_SHAPE_COLUMNS)
    if with_coefficients:
        columns += efd_coefficient_columns(efd_order)

    npy_files = data_input.list_segmentation_files(npy_folder)
    frame_number_by_file = data_input.resolve_frame_numbers(npy_files)
    # `frames` limits the work to a range. Every value here is a function of
    # that cell's own mask and its own frame's neighbours, so a range is exact
    # (Round 19 C3).
    if frames is not None:
        wanted = {int(f) for f in frames}
        npy_files = [f for f in npy_files if frame_number_by_file.get(f) in wanted]
    total = len(npy_files)

    row_index: list[str] = []
    row_values: list[list[float]] = []

    for idx, filename in enumerate(npy_files):
        if os.path.splitext(filename)[1] != ".npy":
            continue
        loaded = np.load(os.path.join(npy_folder, filename), allow_pickle=True).item()
        # The filled array, per `webview/api.py:_load_frame`'s convention. The
        # `outlines` fallback exists only for files that predate `masks`.
        labels = data_input.load_label_array(loaded)
        height, width = labels.shape
        frame_id = str(frame_number_by_file[filename])

        if progress_cb is not None:
            progress_cb(
                f"Computing shape descriptors for frame {frame_id}...",
                int((idx + 1) / total * 100) if total else 100,
            )

        boxes = ndimage.find_objects(labels)
        for cell_id in np.unique(labels):
            if cell_id == 0:
                continue
            box = boxes[cell_id - 1]
            if box is None:
                continue
            sub = labels[box] == cell_id
            if exclude_edge_cells:
                pixels = np.where(labels == cell_id)
                if data_input.is_edge_cell(pixels, height, width, margin=3):
                    continue
            values = cell_shape_features(
                sub,
                offset=(box[1].start, box[0].start),
                efd_order=efd_order,
                n_contour_points=n_contour_points,
                smoothing_sigma_fraction=smoothing_sigma_fraction,
                min_run_fraction=min_run_fraction,
                with_coefficients=with_coefficients,
            )
            if values is None:
                continue
            row_index.append(f"Cell{frame_id}_{cell_id}")
            row_values.append([values[name] for name in columns])

    shape_info = pd.DataFrame(row_values, index=row_index, columns=columns)
    shape_info.index.name = "Cell Index"

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        shape_info.to_excel(os.path.join(output_path, "Cells_shape.xlsx"), sheet_name="Shape", index=True)

    return shape_info


# ---------------------------------------------------------------------------


def _self_test() -> None:
    """Synthetic shapes with analytically known answers."""

    def render(contour: np.ndarray, pad: int = 20) -> np.ndarray:
        shifted = contour - contour.min(axis=0) + pad
        size = (int(shifted[:, 1].max()) + pad, int(shifted[:, 0].max()) + pad)
        canvas = np.zeros(size, np.uint8)
        cv2.fillPoly(canvas, [np.round(shifted).astype(np.int32).reshape(-1, 1, 2)], 1)
        return canvas

    def lobed(k: int, amplitude: float, radius: float = 90.0, n: int = 1200) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = radius * (1 + amplitude * np.cos(k * t))
        return np.column_stack([r * np.cos(t), r * np.sin(t)])

    t = np.linspace(0, 2 * np.pi, 1200, endpoint=False)
    circle = np.column_stack([90 * np.cos(t), 90 * np.sin(t)])
    square = np.column_stack([
        np.r_[np.linspace(-70, 70, 300), np.full(300, 70.0), np.linspace(70, -70, 300), np.full(300, -70.0)],
        np.r_[np.full(300, -70.0), np.linspace(-70, 70, 300), np.full(300, 70.0), np.linspace(70, -70, 300)],
    ])

    # 1. Convex shapes: 0 lobes (not 1), solidity ~1, known shape index.
    for name, contour, expected_index in (("circle", circle, 2 * math.sqrt(math.pi)), ("square", square, 4.0)):
        f = cell_shape_features(render(contour))
        assert f["Lobe Count"] == 0, f"{name} must have 0 lobes, got {f['Lobe Count']}"
        assert f["Solidity"] > 0.99, f"{name} solidity {f['Solidity']:.3f}"
        assert abs(f["Shape Index"] - expected_index) < 0.05, (
            f"{name} shape index {f['Shape Index']:.3f} != {expected_index:.3f}"
        )
        print(f"  {name:8} lobes={f['Lobe Count']:.0f} solidity={f['Solidity']:.3f} "
              f"shape_index={f['Shape Index']:.3f} (exact {expected_index:.3f})")

    # 2. Lobed shapes: the curvature route must recover k exactly, and the
    #    independent convexity-defect route must agree with it.
    # The shipped `Lobe Count` comes from convex-hull deviation. The curvature
    # route is an entirely independent estimator of the same quantity (local
    # differential geometry vs global hull deviation), so requiring them to
    # agree exactly is a strong check that neither is quietly wrong.
    print(f"  {'shape':>12} {'hull lobes':>11} {'curv lobes':>11} {'solidity':>9} {'shape_idx':>10} {'irreg':>7}")
    for k in (4, 5, 6, 8):
        for amplitude in (0.20, 0.45):
            mask = render(lobed(k, amplitude))
            f = cell_shape_features(mask)
            resampled = resample_contour(contour_from_mask(mask), 256)
            curvature_lobes = curvature_lobe_count(signed_curvature(smooth_contour(resampled, 0.02 * 256)))
            assert f["Lobe Count"] == k, f"k={k} amp={amplitude}: hull defects gave {f['Lobe Count']:.0f}"
            assert curvature_lobes == k, (
                f"k={k} amp={amplitude}: curvature gave {curvature_lobes}, hull defects gave "
                f"{f['Lobe Count']:.0f} -- the two independent estimators disagree"
            )
            assert f["Solidity"] < 0.99, f"a lobed cell must not read as convex ({f['Solidity']:.3f})"
            print(f"  k={k} amp={amplitude:.2f}  {f['Lobe Count']:11.0f} {curvature_lobes:11d} "
                  f"{f['Solidity']:9.3f} {f['Shape Index']:10.3f} {f['EFD Irregularity']:7.4f}")

    # 3. Deeper lobes must read as monotonically less solid / more irregular.
    shallow = cell_shape_features(render(lobed(6, 0.15)))
    deep = cell_shape_features(render(lobed(6, 0.45)))
    assert deep["Solidity"] < shallow["Solidity"], "deeper lobes must lower solidity"
    assert deep["EFD Irregularity"] > shallow["EFD Irregularity"], "deeper lobes must raise irregularity"
    assert deep["Shape Index"] > shallow["Shape Index"], "deeper lobes must raise the shape index"

    # 4. Scale invariance: the same shape at 3x must give the same
    #    dimensionless descriptors. This is what lets cells from images of
    #    different magnification be pooled.
    small = cell_shape_features(render(lobed(6, 0.35, radius=60)))
    large = cell_shape_features(render(lobed(6, 0.35, radius=180)))
    for name in ("Solidity", "Shape Index", "Lobe Count", "Max Concavity Depth", "EFD Irregularity"):
        assert abs(small[name] - large[name]) < 0.05 * max(1.0, abs(large[name])), (
            f"{name} is not scale invariant: {small[name]:.4f} at r=60 vs {large[name]:.4f} at r=180"
        )
    print(f"  scale 60->180px: solidity {small['Solidity']:.3f}/{large['Solidity']:.3f}  "
          f"shape_index {small['Shape Index']:.3f}/{large['Shape Index']:.3f}  "
          f"irreg {small['EFD Irregularity']:.4f}/{large['EFD Irregularity']:.4f}")

    # 5. Rotation invariance of the shape descriptors, and correct tracking of
    #    orientation by (sin 2psi, cos 2psi).
    ellipse = np.column_stack([120 * np.cos(t), 30 * np.sin(t)])
    base = cell_shape_features(render(ellipse))
    for degrees in (25.0, 65.0, 130.0):
        angle = np.deg2rad(degrees)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated = cell_shape_features(render(ellipse @ rotation.T))
        assert abs(rotated["Solidity"] - base["Solidity"]) < 0.02, "solidity must not depend on orientation"
        assert abs(rotated["Elongation"] - base["Elongation"]) < 0.1, "elongation must not depend on orientation"
    # psi and psi+180 are the same director, so sin/cos of 2*psi must match.
    flipped = cell_shape_features(render(ellipse @ np.array([[-1.0, 0.0], [0.0, -1.0]])))
    assert abs(flipped["Sin2Theta"] - base["Sin2Theta"]) < 1e-2, "a 180-degree flip is the same director"
    assert abs(flipped["Cos2Theta"] - base["Cos2Theta"]) < 1e-2, "a 180-degree flip is the same director"
    print(f"  ellipse 120x30: elongation={base['Elongation']:.2f} "
          f"sin2psi={base['Sin2Theta']:+.3f} cos2psi={base['Cos2Theta']:+.3f}; "
          f"180-degree flip reproduces both")

    print("\nshape_features.py self-test: OK")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standalone debug entry for v2.shape_features")
    parser.add_argument("--self-test", action="store_true", help="run synthetic-shape checks")
    parser.add_argument("--npy-folder", help="segmentation folder to summarize")
    parser.add_argument("--output-path", help="write Cells_shape.xlsx here")
    parser.add_argument("--with-coefficients", action="store_true", help="also emit raw EFD coefficients")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
    elif args.npy_folder:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        frame = extract_shape_features(
            data_input.prepare_segmentation_folder(args.npy_folder),
            output_path=args.output_path,
            with_coefficients=args.with_coefficients,
            progress_cb=lambda message, percent: print(f"[{percent:3d}%] {message}"),
        )
        print(f"\n{len(frame)} cells")
        print(frame[SCALE_INVARIANT_SHAPE_COLUMNS].describe().round(3).to_string())
    else:
        parser.print_help()
