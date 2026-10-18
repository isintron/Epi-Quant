"""Module 3: Hungarian-algorithm-based cell tracking.

Ported from `gui_main.MainWindow.run_cell_tracking`
(src/gui_main.py:1601-1803), `load_and_process_data` (1805-1810), and the
non-GUI parts of `perform_cell_tracking_and_generate_images` (763-848).

`build_cost_matrix`/`match_frames` are a NEW decomposition of
`run_cell_tracking`'s per-frame-pair inner loop (v1 has this all inline) so
each piece is independently unit-testable with small synthetic frames.
`track_cells` recombines them to reproduce v1's exact output file schema and
numbering (needed for parity testing against real v1 runs).

Bug Fix #3 (see v2/README.md changelog): `split_by_leading_edge` uses
`<= 0` (mes) / `> 0` (epi) consistently, matching `track_cells`'s own
internal population-detection convention (which already used `<=0`/`>0`).
v1's `perform_cell_tracking_and_generate_images` used strict `< 0` / `> 0`
for this same split, silently dropping any cell with `leading_edge == 0`
from tracking entirely.
"""

import logging
import os
import re
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import data_input

logger = logging.getLogger(__name__)

# v1 indexed its feature list POSITIONALLY for both the displacement gate and
# the weights, which made the column order silently load-bearing. The gate is
# resolved by name (`_gate_indices`); the weights were NOT until Round 14 --
# `DEFAULT_TRACKING_WEIGHTS` is a positional list, so a caller passing custom
# `feature_columns` without matching `weights` got them silently misaligned.
# `resolve_weights` fixes that, and `build_cost_matrix` now refuses an all-zero
# resolution rather than returning an arbitrary permutation.

# Columns the plausible-displacement gate reads: a match is rejected when the
# centroid moved further than the cell's own bounding-box extent.
GATE_EXTENT_COLUMNS = ("Maximum Horizontal Length", "Maximum Vertical Length")
GATE_CENTER_COLUMNS = ("Center X Coordinate", "Center Y Coordinate")

# The Cells_info.xlsx columns the tracking cost matrix reads: the gate's four,
# which include the only two that carry weight (`WEIGHTED_COLUMNS`).
#
# This used to be the whole `data_input.CELLS_INFO_COLUMNS`. Every other column
# had weight 0 and is skipped by `build_cost_matrix`, so it contributed nothing
# -- but it still had to EXIST: when Round 24 restored two Classify columns, a
# project extracted before it (15 columns on disk) would have raised KeyError
# at tracking. The cost is bit-identical either way; the self-test asserts it.
DEFAULT_TRACKING_FEATURES = list(GATE_EXTENT_COLUMNS + GATE_CENTER_COLUMNS)


def _gate_indices(feature_columns: list) -> tuple:
    """Positions of the gate's four columns within `feature_columns`."""
    try:
        return tuple(feature_columns.index(c) for c in GATE_EXTENT_COLUMNS + GATE_CENTER_COLUMNS)
    except ValueError as missing:
        raise ValueError(
            f"feature_columns is missing a column the displacement gate needs: {missing}"
        ) from None

# Only Center X/Y actually contribute -- this reduces to pure centroid
# Euclidean distance, matching the manuscript's stated tracking method. Kept as
# behaviour; exposed as a real parameter rather than a hardcoded literal.
#
# Built by NAME rather than written out positionally. The previous literal
# `[0]*12 + [1, 1] + [0]*4` silently encoded that Center X/Y sat at indices
# 12/13; when three columns were dropped from `CELLS_INFO_COLUMNS` on
# 2026-08-18 they moved to 11/12, and a positional literal would have started
# weighting `Inscribed Circle Radius` and `Center X` instead -- with no error,
# just wrong tracks.
# The columns that actually carry weight, BY NAME. `DEFAULT_TRACKING_WEIGHTS`
# below is the positional projection of this onto `DEFAULT_TRACKING_FEATURES`,
# kept because callers pass weight lists; `resolve_weights` is what any new code
# should use.
WEIGHTED_COLUMNS = {"Center X Coordinate": 1, "Center Y Coordinate": 1}

DEFAULT_TRACKING_WEIGHTS = [
    WEIGHTED_COLUMNS.get(column, 0) for column in DEFAULT_TRACKING_FEATURES
]


def resolve_weights(feature_columns: list) -> list:
    """Weights for `feature_columns`, resolved BY NAME.

    The module docstring has always claimed both the gate and the weights are
    "resolved by NAME ... so adding or removing a column cannot quietly repoint
    them at the wrong data". That was true of `_gate_indices` only. `weights`
    defaulted to `DEFAULT_TRACKING_WEIGHTS` -- built against
    `DEFAULT_TRACKING_FEATURES` -- and was then indexed **positionally** against
    whatever `feature_columns` the caller actually passed.

    Measured consequence: calling `track_cells(..., feature_columns=[
    "Maximum Horizontal Length", "Maximum Vertical Length",
    "Center X Coordinate", "Center Y Coordinate"])` -- the four columns tracking
    genuinely needs -- left every resolved weight at 0, so the distance term was
    identically zero, the whole cost matrix was zeros, and
    `linear_sum_assignment` returned an **arbitrary permutation**. No exception
    and no warning: the both-terms-disabled guard does not fire, because
    `centroid_weight` is still 1.0. The mirror case raises `IndexError`.
    """
    return [WEIGHTED_COLUMNS.get(column, 0) for column in feature_columns]

# Weight of the mask-overlap term, in units of the cell's own diameter (see
# `build_cost_matrix`). 1.0 means a completely non-overlapping candidate is
# penalised by one cell-diameter of apparent displacement -- enough to lose to
# any genuinely overlapping rival, without hard-rejecting a cell that
# legitimately has no predecessor (a division, or one entering the field).
DEFAULT_OVERLAP_WEIGHT = 1.0


def mask_overlap_fractions(labels1: np.ndarray, labels2: np.ndarray) -> dict:
    """`{(label1, label2): IoU}` for every pair of cells whose masks overlap
    between two consecutive frames.

    One vectorised pass rather than a per-pair intersection: pairing the two
    label arrays into a single key array and counting with `np.unique` costs
    one traversal regardless of cell count (measured well under a second even
    at 824x1446 with ~800 cells).

    Only overlapping pairs appear; a missing key means IoU 0.
    """
    stride = int(labels2.max()) + 1
    both = (labels1 > 0) & (labels2 > 0)
    if not both.any():
        return {}

    pair_key = labels1[both].astype(np.int64) * stride + labels2[both].astype(np.int64)
    keys, counts = np.unique(pair_key, return_counts=True)
    area1 = np.bincount(labels1.ravel())
    area2 = np.bincount(labels2.ravel())

    overlaps = {}
    for key, intersection in zip(keys, counts):
        a, b = int(key // stride), int(key % stride)
        union = area1[a] + area2[b] - intersection
        if union > 0:
            overlaps[(a, b)] = float(intersection / union)
    return overlaps


def mask_coverage_fractions(labels1: np.ndarray, labels2: np.ndarray) -> dict:
    """`{(label1, label2): |A n B| / |A|}` -- how much of each frame-1 cell each
    frame-2 cell covers. The sibling of `mask_overlap_fractions`, normalised by
    the FIRST cell's area only.

    **Divided by `|A|`, deliberately, and not the raw intersection.** The raw
    intersection favours big candidates: a large neighbour's absolute overlap
    with the anchor can exceed the true successor's, purely because it has more
    pixels to overlap with. Normalising by the anchor makes the score
    insensitive to how big the candidate is, which is the one property IoU does
    not have.

    Where this differs from IoU, algebraically (`v2/docs/census.md` §8.8 has the
    worked pair):

    - the true successor is much **larger** than the anchor (it absorbed a
      neighbour, or the anchor was under-segmented): IoU nearly loses the
      ability to separate it from a same-size neighbour, coverage does not;
    - the anchor **divides** into two smaller daughters: IoU separates the
      daughter from a neighbour better than coverage does.

    So the two fail in opposite directions, which is the argument for offering
    both and not for replacing one with the other. It is an argument, not a
    measurement -- which source a human actually picks is recorded per verdict
    (`chosen_reason`), and that is what decides whether this one stays.

    Only overlapping pairs appear; a missing key means 0.
    """
    stride = int(labels2.max()) + 1
    both = (labels1 > 0) & (labels2 > 0)
    if not both.any():
        return {}
    pair_key = labels1[both].astype(np.int64) * stride + labels2[both].astype(np.int64)
    keys, counts = np.unique(pair_key, return_counts=True)
    area1 = np.bincount(labels1.ravel())
    coverage = {}
    for key, intersection in zip(keys, counts):
        a, b = int(key // stride), int(key % stride)
        if area1[a] > 0:
            coverage[(a, b)] = float(intersection / area1[a])
    return coverage


def _warn_if_overlap_unreachable(loader: "_FrameLabelLoader", npy_folder, overlap_weight,
                                 drift_mode, entry: str) -> None:
    """Say so when a weighted cost term has nothing to read.

    The failure this exists for is not a crash: the run completes, reports
    success, and is quietly a different tracking than the one that was asked
    for. `CLAUDE.md` rule 7 -- check that the inputs were FOUND, not that a
    path was supplied.
    """
    if loader:
        return
    wants_overlap = overlap_weight and overlap_weight > 0
    wants_drift = drift_mode and drift_mode != "none"
    if not (wants_overlap or wants_drift):
        return
    if npy_folder and os.path.isdir(npy_folder):
        detail = ("%r exists but holds no .npy segmentation -- on a TIFF dataset the arrays are in "
                  "its 'npy_converted' subfolder, not in the mask folder itself" % npy_folder)
    else:
        detail = "no npy_folder was given" if not npy_folder else "%r is not a directory" % npy_folder
    wanted = []
    if wants_overlap:
        wanted.append("the mask-overlap term (overlap_weight=%s)" % overlap_weight)
    if wants_drift:
        wanted.append("drift estimation (drift_mode=%r)" % drift_mode)
    logger.warning(
        "%s: %s, so %s cannot be computed and this run is centroid-only. The result will differ "
        "from the same call with the arrays reachable.", entry, detail, " and ".join(wanted),
    )


class _FrameLabelLoader:
    """Loads a frame's filled label array from a segmentation folder, once.

    Tracking is driven by `Cells_info`, which has no pixels in it, so the
    overlap term needs the masks back. Each frame is read once and reused for
    both of the frame-pairs it participates in.

    Returns `None` -- never a substitute array -- when the frame has no
    `masks`. The caller then tracks that pair on centroids alone, which is a
    stated fallback rather than a silent one.
    """

    def __init__(self, npy_folder: Optional[str]):
        self.npy_folder = npy_folder
        self._cache: dict = {}
        self._paths: Optional[dict] = None
        # The minimum-area threshold this loader actually applied. Read from
        # `data_input`'s process state at construction and REPORTED, so a run
        # can say which threshold it used instead of leaving it to be inferred
        # -- the dependence was previously invisible (Round 21 A2).
        self.min_cell_area_px = data_input.get_min_cell_area()

    def _resolve(self) -> dict:
        """`{frame_number: path}`, resolved once. Empty when the folder holds
        no segmentation arrays."""
        if self._paths is None:
            if not self.npy_folder or not os.path.isdir(self.npy_folder):
                self._paths = {}
            else:
                files = data_input.list_segmentation_files(self.npy_folder)
                self._paths = {
                    number: os.path.join(self.npy_folder, name)
                    for name, number in data_input.resolve_frame_numbers(files).items()
                }
        return self._paths

    def __bool__(self) -> bool:
        """True when arrays were **found**, not when a path was supplied.

        This used to be `isdir(npy_folder)`, and the difference has cost twice
        (Round 15 §A1, Round 16's centroid-only baseline). A TIFF dataset's own
        `masks/` folder holds only `.tif` -- the arrays live in
        `masks/npy_converted/` beneath it -- so `isdir` passed, the loader then
        found nothing, and a cost term weighted 1.0 contributed nothing while
        the run reported success. `CLAUDE.md` rule 7 case 2.
        """
        return bool(self._resolve())

    def get(self, frame_number: int) -> Optional[np.ndarray]:
        path = self._resolve().get(frame_number)
        labels = None
        if path is not None:
            try:
                dat = np.load(path, allow_pickle=True).item()
                # Through the shared loader: it prefers 'masks', accepts an
                # older `convert_tiff_masks_to_npy` file whose FILLED array sat
                # under 'outlines' (see `_load_frame`'s docstring) only after
                # `looks_filled` confirms it is one, and applies the same
                # minimum-area filter extraction applies. The three-branch
                # version that lived here was the only place that test was
                # made, which is why extraction did not make it (Round 22 D).
                labels = data_input.load_label_array(dat, self.min_cell_area_px)
                if labels is None:
                    # A genuine 'outlines' array is the ~2-5%-of-pixels
                    # boundary line. Two boundary rings essentially never
                    # intersect, so every IoU came out ~0 and the overlap term
                    # degenerated into a near-constant penalty -- the same
                    # ranking as centroids alone, but reported as if the mask
                    # had been used. Returning None says so instead.
                    logger.warning(
                        "Frame %s has no filled label array (keys: %s); tracking that pair on "
                        "centroids alone rather than on boundary outlines, which carry no "
                        "usable overlap", frame_number, sorted(dat)[:6],
                    )
            except Exception:
                logger.exception("Could not load masks for frame %s; tracking that pair on centroids alone", frame_number)
        self._cache[frame_number] = labels
        return labels


# Parameters that exist on the cost primitives but are DELIBERATELY not
# threaded into the pipeline entry points. Each needs a reason, and the
# reachability self-test fails if a parameter is missing from `track_cells` /
# `retrack_frame_range` without being listed here -- so a gap is a decision on
# the record rather than an omission nobody noticed.
#
# This class of defect has now shipped three times: Round 10's LAP parameters,
# Round 11's registration, and registration again in `retrack_frame_range`.
# Each was "implemented but unreachable from the path users run", and each was
# found a round later.
UNTHREADED_PARAMETERS = {
    "cost_form": "measured: influence falls to 0.0-0.5% once registration is on",
    "overlap_power": "measured: influence falls to 0.0-0.5% once registration is on",
    "no_link_ratio": "measured: influence falls to 0.0-0.5% once registration is on",
    "four_quadrant": "measured: influence falls to 0.0-0.5% once registration is on",
}


DRIFT_MODES = ("none", "centroid", "mask")


def _validate_drift_mode(drift_mode: str) -> str:
    """Reject an unknown `drift_mode` instead of silently running another one.

    Only `!= "none"` and `== "mask"` were ever tested, so `"Mask"`, `"masks"` or
    any typo fell through to the CENTROID experiment while the caller believed
    mask registration was on. `tracking_benchmark._solve_pairs` already raises;
    this matches it.
    """
    if drift_mode not in DRIFT_MODES:
        raise ValueError(f"drift_mode must be one of {DRIFT_MODES}, got {drift_mode!r}")
    return drift_mode


def _shift_labels(labels: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate a label image by whole pixels, filling the vacated edge with 0.

    Mirrors `tracking_benchmark.shift_labels`; kept here so `cell_tracking` does
    not import the benchmark (the dependency runs the other way).
    """
    shifted = np.zeros_like(labels)
    height, width = labels.shape[:2]
    dy, dx = int(round(dy)), int(round(dx))
    rows0, rows1 = max(0, dy), min(height, height + dy)
    cols0, cols1 = max(0, dx), min(width, width + dx)
    if rows0 >= rows1 or cols0 >= cols1:
        return shifted
    shifted[rows0:rows1, cols0:cols1] = labels[rows0 - dy:rows1 - dy, cols0 - dx:cols1 - dx]
    return shifted


def _estimate_drift(label_loader, frame_number: int) -> tuple:
    """Global `(dx, dy)` from mutual-best-IoU displacement, masks only."""
    if not label_loader:
        return (0.0, 0.0)
    from scipy import ndimage
    a, b = label_loader.get(frame_number), label_loader.get(frame_number + 1)
    if a is None or b is None:
        return (0.0, 0.0)
    overlaps = mask_overlap_fractions(a, b)
    forward, backward = {}, {}
    for (i, j), value in overlaps.items():
        if value > forward.get(i, (0.0, None))[0]:
            forward[i] = (value, j)
        if value > backward.get(j, (0.0, None))[0]:
            backward[j] = (value, i)
    def centres(labels):
        ids = np.unique(labels); ids = ids[ids > 0]
        if not len(ids):
            return {}
        found = ndimage.center_of_mass(labels > 0, labels, ids)
        return {int(k): (float(c[1]), float(c[0])) for k, c in zip(ids, found)}
    here, there = centres(a), centres(b)
    deltas = [(there[j][0] - here[i][0], there[j][1] - here[i][1])
              for i, (_, j) in forward.items()
              if backward.get(j, (0.0, None))[1] == i and i in here and j in there]
    if not deltas:
        return (0.0, 0.0)
    return (float(np.median([d[0] for d in deltas])), float(np.median([d[1] for d in deltas])))


def _no_match_sentinel(cost_matrix: pd.DataFrame, fallback: float, no_link_ratio: Optional[float] = None) -> float:
    """The cost of "no match", which must be worse than EVERY real candidate.

    Note what the old global constant (the largest circumscribed-cell radius
    in the movie) was really doing: because the solver prefers a dummy over
    anything dearer, and the `!= sentinel` filter then deletes that row, it
    silently acted as a **maximum-acceptable-cost ceiling** as well as a fill
    value. That ceiling was load-bearing -- removing it and letting the solver
    match everything took wrong links from 3.3% to 20.0% on
    `embryos_GBE_GAP43`.

    So the ceiling is kept, but as an explicit displacement criterion on the
    centroid term (`max_centroid_distance` in `build_cost_matrix`) rather than
    as a side effect of the fill value. That separation is what lets the
    overlap term exist at all: its penalty is in pixel units and can reach a
    whole cell diameter, which would otherwise push genuine matches over a
    ceiling meant for displacement alone. Here the sentinel does only its
    named job, and simply has to sit above every finite entry.

    `no_link_ratio` switches the additive `max + 1` for TrackMate's
    MULTIPLICATIVE form, `C_alt = ratio * max(C)` (TrackMate uses 1.05).
    The additive constant is not scale-free: +1.0 is +3.1% above the sentinel
    on `embryos_GBE_GAP43` (32.0) but only +1.2% on `AS_dorsal` (85.0), so the
    same line of code means a different propensity to drop a link on different
    datasets. Sentinel magnitude was measured to move about 2% of links
    (7,013 -> 7,152 going from max+1 to max+1000), so this is not a
    cosmetic knob. `None` (the default) reproduces the additive behaviour
    exactly.
    """
    finite = cost_matrix.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float(fallback)
    highest = float(max(finite.max(), fallback))
    if no_link_ratio is None:
        return highest + 1.0
    if no_link_ratio <= 1.0:
        # At or below 1.0 the sentinel lands at or under the largest real cost,
        # so the genuine maximum-cost link is filtered out as a non-link -- the
        # solver silently drops the very link the ceiling was meant to keep.
        raise ValueError(
            f"no_link_ratio must be > 1.0 (TrackMate uses 1.05), got {no_link_ratio!r}: "
            "at or below 1.0 the sentinel sits under the largest real cost."
        )
    return highest * float(no_link_ratio)


def _raw_label(cell_index: str) -> int:
    """`'Cell12_47'` -> `47`, the per-frame mask label."""
    return int(str(cell_index).rsplit("_", 1)[1])


def predict_centroids(
    prev_matches: Optional[dict],
    cells_prev: Optional[pd.DataFrame],
    cells_current: pd.DataFrame,
    feature_columns: Optional[list[str]] = None,
) -> dict:
    """Where each cell in `cells_current` is expected to be one frame later:
    `{cell_index: (x, y)}`, for `build_cost_matrix`'s `predicted_centroids`.

    `prev_matches` maps a `cells_prev` index to a `cells_current` index -- the
    accepted matches of the PREVIOUS frame pair. A cell that was matched gets
    a constant-velocity prediction from its own last displacement; every other
    cell gets the frame-wide median displacement.

    **Round 9's residuals (7.9 / 4.0 / 2.3 px) are withdrawn as evidence, and
    so is the sentence "that is the whole mechanism" that used to stand here.**
    They were measured only over cells already tracked correctly across three
    frames -- the subset where the prediction was not needed -- so they
    describe the easy cases and say nothing about the hard ones. The README was
    corrected in Round 10; this docstring was not, until Round 11.

    What IS established (Round 11): the motion on `embryos_GBE_GAP43` is
    coherent, dominated by a whole-field drift of about 6.9 px per frame
    against a 13 px median cell radius. But per-cell velocity prediction is not
    the way to exploit that -- registering the masks by the global drift is
    (`tracking_benchmark.registration_decision`), because a linear assignment
    treats a per-cell prediction as CERTAIN and so converts prediction error
    directly into a displaced search centre. This function stays default-off
    and undecided pending ground truth.

    The fallback is deliberately the frame-wide median rather than a local
    neighbourhood one. It only applies to cells with no previous link -- a
    small minority, since track breaks run at 0.27% -- and the median already
    recovers most of the gain (7.9 -> 4.0 px) for them without needing an
    adjacency graph this function has no other reason to build.
    """
    if feature_columns is None:
        feature_columns = DEFAULT_TRACKING_FEATURES
    _, _, center_x, center_y = _gate_indices(list(feature_columns))
    x_col, y_col = feature_columns[center_x], feature_columns[center_y]

    velocities = {}
    if prev_matches and cells_prev is not None:
        for prev_id, current_id in prev_matches.items():
            if prev_id in cells_prev.index and current_id in cells_current.index:
                velocities[current_id] = (
                    float(cells_current.loc[current_id, x_col]) - float(cells_prev.loc[prev_id, x_col]),
                    float(cells_current.loc[current_id, y_col]) - float(cells_prev.loc[prev_id, y_col]),
                )

    if velocities:
        known = np.asarray(list(velocities.values()), dtype=float)
        fallback = (float(np.median(known[:, 0])), float(np.median(known[:, 1])))
    else:
        fallback = (0.0, 0.0)

    predictions = {}
    for cell_id in cells_current.index:
        dx, dy = velocities.get(cell_id, fallback)
        predictions[cell_id] = (
            float(cells_current.loc[cell_id, x_col]) + dx,
            float(cells_current.loc[cell_id, y_col]) + dy,
        )
    return predictions


def build_cost_matrix(
    cells_frame1: pd.DataFrame,
    cells_frame2: pd.DataFrame,
    feature_columns: Optional[list[str]] = None,
    weights: Optional[list[float]] = None,
    overlaps: Optional[dict] = None,
    overlap_weight: float = DEFAULT_OVERLAP_WEIGHT,
    max_centroid_distance: Optional[float] = None,
    centroid_weight: float = 1.0,
    predicted_centroids: Optional[dict] = None,
    cost_form: str = "additive",
    overlap_power: float = 1.0,
) -> pd.DataFrame:
    """Build the dissimilarity/cost matrix between every cell in
    `cells_frame1` (rows) and `cells_frame2` (columns), both indexed by
    `Cell Index` string with `feature_columns` present. A pair is marked
    implausible (`NaN`) if the frame-2 cell's centroid moved further than
    `max(cell's own Max Horizontal Length, Max Vertical Length)` in either
    axis; otherwise the weighted Euclidean distance over `feature_columns` is
    computed. Ported from src/gui_main.py:1694-1727.

    `overlaps` optionally adds a mask-overlap term (from
    `mask_overlap_fractions`). It is kept in PIXEL units so the existing
    displacement gate and the sentinel in `track_cells` keep their meaning:

        cost = centroid_weight * distance + overlap_weight * (1 - IoU) * max(H, V)

    where `max(H, V)` is the frame-1 cell's own extent, already computed for
    the gate. A perfect overlap adds nothing; no overlap adds one
    cell-diameter of penalty. Cells in a dense epithelium overlap themselves
    heavily between frames (median best-IoU 0.42-0.59 on this project's
    datasets) while overlapping a neighbour barely at all, so this
    discriminates exactly the near-ties where centroid distance alone has no
    information left -- roughly a fifth of cells on `embryos_GBE_GAP43` have a
    runner-up within 2 px.

    Passing `overlaps=None` (or `overlap_weight=0`) reproduces the
    centroid-only cost exactly, which is what keeps the previously-published
    baseline reproducible for comparison. `centroid_weight=0` with a non-zero
    `overlap_weight` is the mask-overlap-only cost; both weights zero is
    rejected rather than silently returning an all-zero matrix.

    **The two gates are deliberately NOT scaled by `centroid_weight`.** They
    are a plausibility filter, not part of the cost -- the separation
    `_no_match_sentinel` describes -- so they keep rejecting implausible jumps
    in every mode. Without that, `centroid_weight=0` would let a distant cell
    win on a sliver of overlap.

    `cost_form` selects between two shapes for combining distance and overlap:

    - `"additive"` (default, unchanged): `centroid_weight * d +
      overlap_weight * (1 - IoU)^p * max(H, V)`. The `max(H, V)` factor is a
      *bridge* that converts a dimensionless overlap deficit into pixels, and
      it has a consequence worth knowing: the total can exceed the
      displacement ceiling, which turns the sentinel from a movie-wide
      constant (32.0 on `embryos_GBE_GAP43` at `overlap_weight=0`) into a
      per-frame-pair quantity (64.8-71.9 at weight 1).
    - `"multiplicative"`: TrackMate's shape, `P = 1 + overlap_weight *
      (1 - IoU)^p` and `cost = (d * P)^2`. The penalty is dimensionless, so no
      bridge is needed and the ceiling keeps its meaning.

    `overlap_power` raises the overlap deficit to a power, as ultrack does
    (p >= 2 favours fewer, better-overlapping matches). Default 1 is linear,
    i.e. unchanged.

    `predicted_centroids` (`{cell_index: (x, y)}`) replaces each frame-1 cell's
    own centroid with where it is *expected* to be in frame 2, so the distance
    and the gates both measure the residual from that prediction. Two very
    different things use it: `predict_centroids` (per-cell velocity, default
    off and undecided) and a global-drift shift, which is the registration
    Tissue Analyzer applies before matching. See
    `tracking_benchmark.registration_decision`; the two are not interchangeable
    and were measured separately in Round 11.
    """
    # Only a matrix with something in it can be degenerate. `track_cells`
    # iterates to the last frame, whose "next frame" is empty and has no masks
    # to overlap -- that pair has nothing to match and must not be treated as a
    # misconfiguration.
    if cost_form not in ("additive", "multiplicative"):
        raise ValueError(f"Unknown cost_form {cost_form!r}; expected 'additive' or 'multiplicative'.")
    if overlap_power <= 0:
        # (1 - IoU)**0 == 1 even for a PERFECT overlap, so a zero power inverts
        # the term: the best possible match pays the full penalty.
        raise ValueError(f"overlap_power must be > 0, got {overlap_power!r}.")
    if (centroid_weight == 0 and (overlap_weight == 0 or not overlaps)
            and not cells_frame1.empty and not cells_frame2.empty):
        raise ValueError(
            "centroid_weight and the overlap term cannot both be disabled -- that leaves no cost "
            "to minimise and the assignment would be arbitrary."
        )
    if feature_columns is None:
        feature_columns = DEFAULT_TRACKING_FEATURES
    if weights is None:
        weights = resolve_weights(feature_columns)
    if len(weights) != len(feature_columns):
        raise ValueError(
            f"weights has {len(weights)} entries for {len(feature_columns)} feature_columns; "
            "they are matched positionally, so the lengths must agree."
        )
    if not any(weights):
        raise ValueError(
            "Every resolved weight is zero, so the distance term would be identically 0 and the "
            "assignment arbitrary. feature_columns must include at least one of "
            f"{sorted(WEIGHTED_COLUMNS)}; got {list(feature_columns)}."
        )

    # Zero-weight columns are skipped rather than multiplied by zero. With all
    # but two of the default columns weighted 0, `0 * value` was already pure
    # overhead; more importantly `0 * np.nan` is `np.nan`, so a single
    # unpopulated column silently turned every cost into NaN and
    # `linear_sum_assignment` then rejected the whole matrix. That made it
    # impossible to track from a table where irrelevant columns are absent
    # (e.g. `data_input.extract_cell_info(..., tracking_only=True)`).
    # Results for a fully-populated table are unchanged: the skipped terms
    # contributed exactly 0.
    scored = [k for k in range(len(feature_columns)) if weights[k] != 0]
    extent_x, extent_y, center_x, center_y = _gate_indices(list(feature_columns))

    idx1 = list(cells_frame1.index)
    idx2 = list(cells_frame2.index)
    use_overlap = bool(overlaps) and overlap_weight != 0
    if not idx1 or not idx2:
        return pd.DataFrame(np.zeros((len(idx1), len(idx2))), index=idx1, columns=idx2)

    # Vectorised over the whole matrix. This was a double Python loop doing one
    # `cells_frame2.loc[id2, feature_columns]` per CANDIDATE PAIR -- about
    # 62,500 pandas lookups for a 250x250 frame pair, which made a full 29-pair
    # sweep take tens of minutes and was the direct cause of two abandoned
    # measurements in Round 10. Broadcasting gives the identical matrix in
    # seconds; `_build_cost_matrix_reference` below keeps the original loop and
    # the self-test asserts the two agree exactly.
    features1 = cells_frame1[feature_columns].to_numpy(dtype=float, copy=True)
    features2 = cells_frame2[feature_columns].to_numpy(dtype=float)

    if predicted_centroids is not None:
        # Measure from where each cell is EXPECTED to be, not from where it was.
        # Substituting into the feature matrix rather than at each comparison
        # means the gates below move with the prediction too, which is the
        # point: a fast but correctly-predicted cell must not be gated out for
        # having moved.
        for i, id1 in enumerate(idx1):
            predicted = predicted_centroids.get(id1)
            if predicted is not None:
                features1[i, center_x] = float(predicted[0])
                features1[i, center_y] = float(predicted[1])

    max_distance = np.maximum(features1[:, extent_x], features1[:, extent_y])[:, None]

    delta_x = features2[None, :, center_x] - features1[:, None, center_x]
    delta_y = features2[None, :, center_y] - features1[:, None, center_y]
    implausible = (np.abs(delta_x) > max_distance) | (np.abs(delta_y) > max_distance)

    # Accumulated in the same column order as the original loop so the
    # floating-point result is bit-identical, not merely close.
    distance_sq = np.zeros((len(idx1), len(idx2)), dtype=float)
    for k in scored:
        distance_sq += weights[k] * (features2[None, :, k] - features1[:, None, k]) ** 2
    distance = np.sqrt(distance_sq)

    # The displacement ceiling applies to the CENTROID term only, so the overlap
    # penalty below can never push an otherwise-plausible match over it. See
    # `_no_match_sentinel`. Both this and the gate above read the UNWEIGHTED
    # distance, so lowering `centroid_weight` changes what the solver prefers
    # but never what it is allowed to consider.
    if max_centroid_distance is not None:
        implausible |= distance > max_centroid_distance

    deficit = np.zeros((len(idx1), len(idx2)), dtype=float)
    if use_overlap:
        position1 = {label: i for i, label in enumerate(_raw_label(x) for x in idx1)}
        position2 = {label: j for j, label in enumerate(_raw_label(x) for x in idx2)}
        iou = np.zeros((len(idx1), len(idx2)), dtype=float)
        for (label_a, label_b), value in overlaps.items():
            i, j = position1.get(label_a), position2.get(label_b)
            if i is not None and j is not None:
                iou[i, j] = value
        deficit = (1.0 - iou) ** overlap_power

    if cost_form == "multiplicative":
        cells_relation = (distance * (1.0 + overlap_weight * deficit)) ** 2
    else:
        cells_relation = centroid_weight * distance + overlap_weight * deficit * max_distance

    cells_relation = np.where(implausible, np.nan, cells_relation)

    return pd.DataFrame(cells_relation, index=idx1, columns=idx2)


def _build_cost_matrix_reference(
    cells_frame1, cells_frame2, feature_columns=None, weights=None, overlaps=None,
    overlap_weight=DEFAULT_OVERLAP_WEIGHT, max_centroid_distance=None,
    centroid_weight=1.0, predicted_centroids=None, cost_form="additive", overlap_power=1.0,
) -> pd.DataFrame:
    """The original per-pair loop `build_cost_matrix` replaced, kept verbatim.

    Not used in the pipeline -- it exists so the vectorised version can be
    asserted equal to it rather than merely believed to be. It is slow on
    purpose: one `.loc` per candidate pair is exactly what made the sweep
    unaffordable, and that is the property being removed.
    """
    if feature_columns is None:
        feature_columns = DEFAULT_TRACKING_FEATURES
    if weights is None:
        weights = resolve_weights(feature_columns)
    scored = [k for k in range(len(feature_columns)) if weights[k] != 0]
    extent_x, extent_y, center_x, center_y = _gate_indices(list(feature_columns))

    idx1, idx2 = list(cells_frame1.index), list(cells_frame2.index)
    cells_relation = np.zeros((len(idx1), len(idx2)))
    use_overlap = bool(overlaps) and overlap_weight != 0
    labels1 = [_raw_label(i) for i in idx1] if use_overlap else None
    labels2 = [_raw_label(i) for i in idx2] if use_overlap else None

    for i, id1 in enumerate(idx1):
        info1 = list(cells_frame1.loc[id1, feature_columns])
        if predicted_centroids is not None:
            predicted = predicted_centroids.get(id1)
            if predicted is not None:
                info1[center_x], info1[center_y] = float(predicted[0]), float(predicted[1])
        for j, id2 in enumerate(idx2):
            info2 = list(cells_frame2.loc[id2, feature_columns])
            max_distance = max(info1[extent_x], info1[extent_y])
            if (abs(info2[center_x] - info1[center_x]) > max_distance
                    or abs(info2[center_y] - info1[center_y]) > max_distance):
                cells_relation[i, j] = np.nan
                continue
            distance_sq = 0.0
            for k in scored:
                distance_sq += weights[k] * (info2[k] - info1[k]) ** 2
            distance = np.sqrt(distance_sq)
            if max_centroid_distance is not None and distance > max_centroid_distance:
                cells_relation[i, j] = np.nan
                continue
            deficit = 0.0
            if use_overlap:
                iou = overlaps.get((labels1[i], labels2[j]), 0.0)
                deficit = (1.0 - iou) ** overlap_power
            if cost_form == "multiplicative":
                cost = (distance * (1.0 + overlap_weight * deficit)) ** 2
            else:
                cost = centroid_weight * distance + overlap_weight * deficit * max_distance
            cells_relation[i, j] = cost
    return pd.DataFrame(cells_relation, index=idx1, columns=idx2)


def match_frames(cost_matrix: pd.DataFrame, sentinel: float, four_quadrant: bool = False) -> pd.DataFrame:
    """Fill NaNs in `cost_matrix` with `sentinel`, solve the assignment
    problem via the Hungarian algorithm, and return the RAW (not yet
    row-numbered or sentinel-filtered) matched-pairs table with columns
    `frame1_cell`, `frame2_cell`, `Difference`.

    Index numbering and sentinel-based filtering are deliberately left to
    the caller (`track_cells`) rather than done here: v1 assigns row numbers
    over the FULL match set before filtering out sentinel-only matches, so
    surviving rows keep non-contiguous numbers — collapsing that ordering
    into this function would break exact parity with v1's `Index` column.
    Ported from src/gui_main.py:1738-1745.

    `four_quadrant=True` builds TrackMate's `(n+m) x (n+m)` block matrix
    instead of relying on scipy's rectangular handling:

        [ link (n x m)        | termination (n x n) ]
        [ initiation (m x m)  | auxiliary   (m x n) ]

    **Why this is worth having as an option.** The rectangular form produces
    TWO different unmatched paths that are not treated alike: a row matched to
    a sentinel cell (filtered out afterwards, and it leaves a record), and --
    when `m > n` -- a column that is simply never emitted, with no cost and no
    record at all. Measured on `embryos_GBE_GAP43`: 139 sentinel-filtered
    against 82 never emitted. The block form gives every cell an explicit
    termination or initiation cost, so both paths become visible and priced.
    It is also the prerequisite for a second-stage segment LAP
    (see `close_track_gaps`).

    Off by default and asserted to reproduce the rectangular result exactly on
    the shipped cost.
    """
    df_filled = cost_matrix.fillna(sentinel)
    if not four_quadrant:
        row_ind, col_ind = linear_sum_assignment(df_filled.values)
        return pd.DataFrame({
            "frame1_cell": [cost_matrix.index[r] for r in row_ind],
            "frame2_cell": [cost_matrix.columns[c] for c in col_ind],
            "Difference": [df_filled.iat[r, c] for r, c in zip(row_ind, col_ind)],
        })

    values = df_filled.values
    n, m = values.shape
    blocked = np.full((n + m, n + m), np.inf)
    blocked[:n, :m] = values
    # Termination: row i may end, at the sentinel price, on its own diagonal.
    blocked[:n, m:][np.diag_indices(n)] = sentinel
    # Initiation: column j may begin, likewise.
    blocked[n:, :m][np.diag_indices(m)] = sentinel
    # Auxiliary block. TrackMate mirrors the link block transposed so the
    # matrix stays feasible; a finite entry here costs the minimum, which lets
    # the solver pair a termination with its matching initiation for free.
    lowest = float(values.min()) if values.size else 0.0
    auxiliary = np.where(np.isfinite(values.T), lowest, np.inf)
    blocked[n:, m:] = auxiliary

    row_ind, col_ind = linear_sum_assignment(blocked)
    pairs = [(r, c) for r, c in zip(row_ind, col_ind) if r < n and c < m]
    return pd.DataFrame({
        "frame1_cell": [cost_matrix.index[r] for r, _ in pairs],
        "frame2_cell": [cost_matrix.columns[c] for _, c in pairs],
        "Difference": [df_filled.iat[r, c] for r, c in pairs],
    })


def load_and_process_data(file_path: str) -> pd.DataFrame:
    """Read a `*_Matching_Results_new_*.xlsx` file and drop its `Difference`
    column. Verbatim port of src/gui_main.py:1805-1810."""
    data = pd.read_excel(file_path)
    data.drop(columns=["Difference"], inplace=True)
    return data


def _merge_tracking_pairs(
    frame_output_dir: str, folder_prefix: str, start_index: int,
    produced_frames: Optional[list] = None,
) -> pd.DataFrame:
    """Stitch all `{folder_prefix}Matching_Results_new_{i}_{i+1}.xlsx` files
    in `frame_output_dir` into one wide `Index, Frame{start}..Frame{last}`
    DataFrame, writes `{folder_prefix}merged_tracking_results.xlsx`, and
    returns it.

    Incremental "open tracks" stitch — walks pair files in ascending frame
    order, keeping a mapping of "cell string at the current end of an open
    track" -> that track's row. A cell that continues an open track extends
    that row; a cell with no open track (including every cell at the very
    first pair, where no tracks are open yet) starts a brand-new track.

    Replaces the originally-ported "seed a table from the first pair file,
    then look up later pairs by value" approach (src/gui_main.py:1762-1803),
    which had two bugs: (1) a cell whose first valid match was at frame
    i > start had no row to attach to yet and was silently dropped — cells
    first appearing mid-movie were never tracked; (2) `Index` numbering was
    inherited from the first pair file's own provisional numbers, which
    only ever covered frame-1-onward tracks. `Index` is assigned once here,
    over the complete stitched set, and duplicate/ambiguous claims on a
    single physical cell can't arise by construction: each pair file's own
    second-frame values are already unique (Hungarian assignment gives each
    column at most one row), and `open_tracks` is consumed via `.pop()`, so
    no cell string can ever be attached to two different tracks at once.
    """
    matching_files = [
        f for f in os.listdir(frame_output_dir)
        if re.match(rf"{folder_prefix}Matching_Results_new_\d+_\d+\.xlsx", f)
    ]
    matching_frames = sorted(
        set(int(re.search(r"\d+", f).group(0)) for f in matching_files if re.search(r"\d+", f))
    )
    # Only the pairs THIS run produced. Listing the directory stitches in files
    # left by a previous run, and `track_cells` never clears it: measured, a
    # 5-frame movie re-tracked as 2 frames into the same output directory
    # produced a phantom track assembled entirely from the old files, describing
    # cells that no longer exist. That is exactly the hand-correction ->
    # re-track workflow this module is built around.
    if produced_frames is not None:
        matching_frames = [f for f in matching_frames if f in set(produced_frames)]
    if not matching_frames:
        data_final = pd.DataFrame()
        data_final.to_excel(os.path.join(frame_output_dir, f"{folder_prefix}merged_tracking_results.xlsx"), index=False)
        return data_final

    start_frame = min(matching_frames)

    tracks: list[dict[str, str]] = []
    open_tracks: dict[str, int] = {}

    # Iterate the frame numbers that actually EXIST, not a count treated as a
    # span. `range(start, start + len(matching_frames))` is correct only when
    # the frame numbers are contiguous; measured on a movie with frames
    # 1,2,5,6,9,10 it wrote `Matching_Results_new_9_10.xlsx` correctly and then
    # never opened it, so frames 8-10 vanished from the merged table with no
    # error and no log line. Frame holes are normal -- a frame whose
    # segmentation yields no cells, or a partially segmented movie.
    for i in matching_frames:
        pair_path = os.path.join(frame_output_dir, f"{folder_prefix}Matching_Results_new_{i}_{i + 1}.xlsx")
        if not os.path.exists(pair_path):
            continue
        data_next = load_and_process_data(pair_path)
        data_next.columns = ["Index", f"Frame{i}", f"Frame{i + 1}"]
        frame_i_col, frame_i1_col = f"Frame{i}", f"Frame{i + 1}"

        for _, row in data_next.iterrows():
            cell_i, cell_i1 = row[frame_i_col], row[frame_i1_col]
            if cell_i in open_tracks:
                track_idx = open_tracks.pop(cell_i)
            else:
                track_idx = len(tracks)
                tracks.append({})
            tracks[track_idx][frame_i_col] = cell_i
            tracks[track_idx][frame_i1_col] = cell_i1
            open_tracks[cell_i1] = track_idx

    frame_columns = [f"Frame{n}" for n in sorted(set(matching_frames) | {f + 1 for f in matching_frames})]
    data_final = pd.DataFrame(tracks, columns=frame_columns)
    data_final.insert(0, "Index", range(start_index, start_index + len(data_final)))

    data_final.to_excel(os.path.join(frame_output_dir, f"{folder_prefix}merged_tracking_results.xlsx"), index=False)
    return data_final


def dedupe_tracking_data(tracking_data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Repair a tracking table that may have duplicate/ambiguous claims on a
    single physical cell (possible in files written before the
    `_merge_tracking_pairs` fix above; fresh tracking runs can't produce
    these). For every `Frame{n}` column, keeps only the lowest-`Index` row's
    claim on each cell string and nulls the rest. Returns
    `(tracking_data, duplicates_removed_count)` -- `tracking_data` is
    mutated in place and also returned for convenience; the count lets the
    caller decide whether to surface a notice."""
    id_col = tracking_data.columns[0]
    frame_cols = [c for c in tracking_data.columns if c.startswith("Frame")]
    removed = 0

    for col in frame_cols:
        seen: dict[str, int] = {}  # cell string -> its lowest-Index row's positional index
        for pos in tracking_data.index:
            value = tracking_data.at[pos, col]
            if pd.isna(value):
                continue
            if value not in seen:
                seen[value] = pos
                continue
            # Duplicate claim: keep whichever row has the lower Index, null the other.
            keep_pos = seen[value]
            if tracking_data.at[pos, id_col] < tracking_data.at[keep_pos, id_col]:
                tracking_data.at[keep_pos, col] = None
                seen[value] = pos
            else:
                tracking_data.at[pos, col] = None
            removed += 1

    return tracking_data, removed


def track_cells(
    cells_info: pd.DataFrame,
    output_dir: str,
    start_index: Optional[int] = None,
    feature_columns: Optional[list[str]] = None,
    weights: Optional[list[float]] = None,
    progress_cb: Optional[Callable[[str, int], None]] = None,
    npy_folder: Optional[str] = None,
    overlap_weight: float = DEFAULT_OVERLAP_WEIGHT,
    centroid_weight: float = 1.0,
    motion_compensation: bool = False,
    max_frame_gap: int = 0,
    cost_form: str = "additive",
    overlap_power: float = 1.0,
    no_link_ratio: Optional[float] = None,
    four_quadrant: bool = False,
    drift_mode: str = "none",
    drift_by_frame: Optional[dict] = None,
) -> pd.DataFrame:
    """Track cells frame-to-frame across all frames present in `cells_info`
    (indexed by `Cell Index` string; optionally pre-split to a single
    population via `split_by_leading_edge`). Writes per-frame-pair
    `{prefix}Cell_Relation_*.xlsx` / `{prefix}Matching_Results_new_*.xlsx`
    plus a stitched `{prefix}merged_tracking_results.xlsx` under
    `output_dir/{mes,epi,All}_cell_tracking/`, and returns the merged
    DataFrame. Ported from `run_cell_tracking`, src/gui_main.py:1601-1803.

    `output_dir` corresponds to v1's `config.cell_track_output_path` (the
    parent tracking directory, NOT the per-population subfolder).

    If `start_index` is `None` (the default), it's auto-detected exactly as
    v1 did: continue numbering from the sibling population's already-tracked
    max index, if that population was tracked first and its merged file
    already exists under `output_dir`.

    `drift_mode` applies a measured global drift, and is three-way rather than
    a boolean because the two ways of using the same drift are different
    experiments (see `tracking_benchmark._solve_pairs`):

    - `"none"` (default) -- the published baseline, byte-reproducible.
    - `"centroid"` -- shift the predicted centroid only; the IoU term still
      sees unregistered masks.
    - `"mask"` -- Tissue Analyzer's convention: register the frame-f label
      image AND shift the centroid, so both terms are measured in the
      registered frame (`match_by_max_overlap(..., pre_register=True)`).

    `drift_by_frame` is `{frame: (dx, dy)}`, normally from
    `tracking_benchmark.registration_decision`; omitted, the drift is estimated
    per pair from mutual-best-IoU displacement.

    **This was Round 11's central finding and it was unreachable from here** --
    it existed only on the benchmark's own `track_in_memory`, so the setting
    that scores best could not be produced by the pipeline users run.

    `cost_form`, `overlap_power`, `no_link_ratio` and `four_quadrant` are the
    LAP-structure alternatives documented on `build_cost_matrix`,
    `_no_match_sentinel` and `match_frames`. They existed there but were
    unreachable from here, which made Round 10's own requirement -- "report its
    effect under both judges" -- unsatisfiable. Every default reproduces
    today's behaviour exactly.

    `max_frame_gap` > 0 runs `close_track_gaps` after the merge, so a track
    that loses its cell for a frame re-attaches instead of ending. Default 0,
    which leaves the merged table byte-identical to before this existed.

    `motion_compensation` predicts each frame's cells forward from the
    previous pair's accepted matches (see `predict_centroids`) instead of
    assuming they stay put. It defaults **off here** so the published
    cell-loss baseline stays reachable from this function and the self-tests
    can pin it; the webview turns it on by default, which is a deliberate
    divergence -- it changes which cells link, and therefore track counts and
    cell-loss %.
    """
    if feature_columns is None:
        feature_columns = DEFAULT_TRACKING_FEATURES
    if weights is None:
        weights = resolve_weights(feature_columns)

    cells_info = cells_info.copy()
    cells_info.index = cells_info.index.map(str)

    auto_start_index = 1
    if "leading_edge" not in cells_info.columns:
        folder_prefix, output_subfolder, name = "All_cell_", "All_cell_tracking", "all_cell"
    elif (cells_info["leading_edge"] <= 0).all():
        folder_prefix, output_subfolder, name = "mes_cell_", "mes_cell_tracking", "mes_cell"
        sibling_file = os.path.join(output_dir, "epi_cell_tracking", "epi_cell_merged_tracking_results.xlsx")
        if os.path.exists(sibling_file):
            sibling_df = pd.read_excel(sibling_file, header=0)
            if not sibling_df.empty:
                auto_start_index = sibling_df.iloc[:, 0].max() + 1
    elif (cells_info["leading_edge"] > 0).all():
        folder_prefix, output_subfolder, name = "epi_cell_", "epi_cell_tracking", "epi_cell"
        sibling_file = os.path.join(output_dir, "mes_cell_tracking", "mes_cell_merged_tracking_results.xlsx")
        if os.path.exists(sibling_file):
            sibling_df = pd.read_excel(sibling_file, header=0)
            if not sibling_df.empty:
                auto_start_index = sibling_df.iloc[:, 0].max() + 1
    else:
        raise ValueError(
            "cells_info['leading_edge'] must be entirely <= 0, entirely > 0, or the column must be "
            "absent entirely -- a mixed-sign column within one track_cells() call is not a case v1 handled "
            "(split populations with split_by_leading_edge before calling)."
        )

    if start_index is None:
        start_index = auto_start_index
    merge_start_index = start_index  # `start_index` below is mutated per-pair; the merge needs the original seed.

    frame_output_dir = os.path.join(output_dir, output_subfolder)
    os.makedirs(frame_output_dir, exist_ok=True)

    fig_ids = sorted(set(
        int(re.search(r"Cell(\d+)_", idx).group(1))
        for idx in cells_info.index if re.search(r"Cell(\d+)_", idx)
    ))
    total_figs = len(fig_ids)
    if total_figs == 0:
        logger.warning("No valid frames found for tracking (%s).", name)
        return pd.DataFrame()

    label_loader = _FrameLabelLoader(npy_folder)
    _warn_if_overlap_unreachable(label_loader, npy_folder, overlap_weight, drift_mode, "track_cells")
    # The value the old sentinel implicitly enforced as a displacement ceiling.
    radius_ceiling = cells_info["Circumscribed Circle Radius"].max()

    _validate_drift_mode(drift_mode)
    if drift_mode != "none" and not drift_by_frame and not (npy_folder and os.path.isdir(npy_folder)):
        # `_estimate_drift` returns (0, 0) with no masks to measure from, so the
        # run is silently UNREGISTERED while the caller believes otherwise.
        logger.warning(
            "drift_mode=%r was requested but no npy_folder and no drift_by_frame were given, so the "
            "drift is (0, 0) and this run is effectively unregistered.", drift_mode,
        )

    # Which frame pairs THIS run actually wrote, so the merge cannot stitch in a
    # previous run's leftovers (see `_merge_tracking_pairs`).
    produced_frames: list = []
    # Carried across iterations so each pair can predict from the last one's
    # answer; `None` on the first pair, where no velocity exists yet.
    previous_cells, previous_matches = None, None

    for idx, fig_id in enumerate(fig_ids):
        if progress_cb is not None:
            progress_cb(f"Tracking frame {fig_id} for {name}...", int((idx + 1) / total_figs * 100))

        cells_info_fig1 = cells_info[cells_info.index.str.contains(f"Cell{fig_id}_")]
        cells_info_fig2 = cells_info[cells_info.index.str.contains(f"Cell{fig_id + 1}_")]

        shift = (0.0, 0.0)
        if drift_mode != "none":
            shift = (drift_by_frame or {}).get(fig_id) or _estimate_drift(label_loader, fig_id)

        overlaps = None
        if label_loader and overlap_weight != 0:
            labels1, labels2 = label_loader.get(fig_id), label_loader.get(fig_id + 1)
            if labels1 is not None and labels2 is not None:
                if drift_mode == "mask" and (shift[0] or shift[1]):
                    labels1 = _shift_labels(labels1, shift[0], shift[1])
                overlaps = mask_overlap_fractions(labels1, labels2)

        # Mask-overlap-only mode has nothing to minimise on a pair whose masks
        # could not be read. Degrade that pair to the centroid cost rather than
        # failing the whole run -- one unreadable frame should not cost the
        # movie its tracking.
        pair_centroid_weight = centroid_weight
        if centroid_weight == 0 and not overlaps and not cells_info_fig2.empty:
            logger.warning(
                "No mask overlaps available for frames %s-%s; falling back to the centroid cost "
                "for this pair.", fig_id, fig_id + 1,
            )
            pair_centroid_weight = 1.0

        predicted = None
        if motion_compensation:
            predicted = predict_centroids(
                previous_matches, previous_cells, cells_info_fig1, feature_columns,
            )
        if drift_mode != "none" and (shift[0] or shift[1]) and not cells_info_fig1.empty:
            _, _, cx, cy = _gate_indices(list(feature_columns))
            x_col, y_col = feature_columns[cx], feature_columns[cy]
            base = predicted or {}
            predicted = {
                index: (base.get(index, (float(cells_info_fig1.at[index, x_col]),
                                         float(cells_info_fig1.at[index, y_col])))[0] + shift[0],
                        base.get(index, (float(cells_info_fig1.at[index, x_col]),
                                         float(cells_info_fig1.at[index, y_col])))[1] + shift[1])
                for index in cells_info_fig1.index
            }

        cost_df = build_cost_matrix(
            cells_info_fig1, cells_info_fig2, feature_columns, weights,
            overlaps=overlaps, overlap_weight=overlap_weight,
            max_centroid_distance=radius_ceiling,
            centroid_weight=pair_centroid_weight, predicted_centroids=predicted,
            cost_form=cost_form, overlap_power=overlap_power,
        )
        cost_df.to_excel(os.path.join(frame_output_dir, f"{folder_prefix}Cell_Relation_{fig_id}_{fig_id + 1}.xlsx"))

        sentinel = _no_match_sentinel(
            cost_df, cells_info["Circumscribed Circle Radius"].max(), no_link_ratio=no_link_ratio,
        )
        matching_result = match_frames(cost_df, sentinel, four_quadrant=four_quadrant)
        matching_result = matching_result.rename(
            columns={"frame1_cell": f"Frame{fig_id}", "frame2_cell": f"Frame{fig_id + 1}"}
        )
        matching_result.insert(0, "Index", range(start_index, start_index + len(matching_result)))
        matching_result = matching_result[matching_result["Difference"] != sentinel]

        produced_frames.append(fig_id)
        previous_cells = cells_info_fig1
        previous_matches = dict(zip(
            matching_result[f"Frame{fig_id}"], matching_result[f"Frame{fig_id + 1}"]
        ))

        matching_result.to_excel(
            os.path.join(frame_output_dir, f"{folder_prefix}Matching_Results_new_{fig_id}_{fig_id + 1}.xlsx"),
            index=False,
        )
        start_index += len(matching_result)

    merged = _merge_tracking_pairs(
        frame_output_dir, folder_prefix, start_index=merge_start_index,
        produced_frames=produced_frames,
    )
    if max_frame_gap:
        merged, joins = close_track_gaps(
            merged, cells_info, max_frame_gap=max_frame_gap,
            feature_columns=feature_columns, max_centroid_distance=radius_ceiling,
        )
        if joins:
            logger.info("Gap closing joined %d track segment(s) across up to %d skipped frame(s).",
                        len(joins), max_frame_gap)
            merged.to_excel(
                os.path.join(frame_output_dir, f"{folder_prefix}merged_tracking_results.xlsx"), index=False,
            )
    return merged


def _join_hole(join: dict) -> tuple:
    """The frames a gap-join bridges, as `(first_missing, last_missing)`.

    Read from the join's own `seam_frame` rather than re-derived from the
    table: after the first link of a chain the surviving row's last frame is no
    longer the seam, so re-deriving would mis-scope every later link.
    """
    seam = int(join["seam_frame"])
    return (seam + 1, seam + int(join["gap"]))


def _apply_selected_joins(tracking_data: pd.DataFrame, id_col: str, joins: list) -> pd.DataFrame:
    """Apply only `joins`, leaving every other track alone."""
    merged = tracking_data.copy(deep=True)
    frame_cols = [c for c in merged.columns if c.startswith("Frame")]
    absorbed = set()
    for join in joins:
        into_rows = merged.index[merged[id_col] == int(join["into_track_id"])]
        from_rows = merged.index[merged[id_col] == int(join["from_track_id"])]
        if not len(into_rows) or not len(from_rows):
            continue
        into_row, from_row = into_rows[0], from_rows[0]
        for col in frame_cols:
            value = merged.at[from_row, col]
            if pd.notna(value):
                merged.at[into_row, col] = value
                merged.at[from_row, col] = None
        absorbed.add(int(join["from_track_id"]))
    if absorbed:
        keep = [p for p in merged.index if int(merged.at[p, id_col]) not in absorbed]
        merged = merged.loc[keep].reset_index(drop=True)
    return merged


def close_track_gaps(
    tracking_data: pd.DataFrame,
    cells_info: pd.DataFrame,
    max_frame_gap: int = 1,
    feature_columns: Optional[list[str]] = None,
    max_centroid_distance: Optional[float] = None,
) -> tuple:
    """Second-stage LAP: join a track that ENDS at frame `f` to one that BEGINS
    at `f + 1 + g` (`g` frames skipped, `1 <= g <= max_frame_gap`), so a cell
    momentarily lost to segmentation re-attaches instead of minting a new id.

    Returns `(tracking_data, joins)` where `joins` is a list of
    `{"from_track_id", "into_track_id", "gap", "seam_frame", "distance"}`. The
    table is rebuilt, not mutated. A chain of segments (A ends, B fills one gap
    later, C one gap after B) is closed **whole**: every link reports the same
    `into_track_id` -- the surviving root -- and `seam_frame` is the last frame
    of the segment that link bridges from.

    **This is a second assignment problem over track SEGMENTS, not a special
    case inside the frame-to-frame one** -- the structure u-track (Jaqaman
    2008), TrackMate and LapTrack all share: step 1 links frame to frame and
    produces tracklets, step 2 links tracklet ends to tracklet starts to close
    gaps (and, in those tools, to split and merge). It is the same
    `linear_sum_assignment`, no ILP and no flow network, and putting it here
    rather than inside `build_cost_matrix` keeps step 1's cost exactly what it
    was.

    Cost is centroid distance between the two ends, gated at
    `max_centroid_distance * (gap + 1)` -- a cell allowed `d` px in one frame
    is allowed `2d` across a skipped frame. The overlap term is deliberately
    NOT used: the two cells are by definition in non-adjacent frames, so their
    masks need not overlap at all and an IoU of 0 would carry no information.

    **Default off at every layer.** Closing gaps changes track counts and
    therefore cell-loss %, which is the manuscript's 10%-vs-60% benchmark.
    Report its effect under BOTH judges in `tracking_benchmark` before turning
    it on; one judge is not enough (see that module's docstring).
    """
    if max_frame_gap < 1 or tracking_data is None or tracking_data.empty:
        return tracking_data, []
    if feature_columns is None:
        feature_columns = DEFAULT_TRACKING_FEATURES
    _, _, centre_x, centre_y = _gate_indices(list(feature_columns))
    x_col, y_col = feature_columns[centre_x], feature_columns[centre_y]
    if max_centroid_distance is None:
        max_centroid_distance = float(cells_info["Circumscribed Circle Radius"].max())

    id_col = tracking_data.columns[0]
    frame_cols = sorted(
        ((int(c[len("Frame"):]), c) for c in tracking_data.columns if c.startswith("Frame")),
    )

    ends, starts = {}, {}
    for position, row in tracking_data.iterrows():
        claims = [(number, row[col]) for number, col in frame_cols if pd.notna(row[col])]
        if not claims:
            continue
        track_id = int(row[id_col])
        starts[track_id] = (claims[0][0], claims[0][1], position)
        ends[track_id] = (claims[-1][0], claims[-1][1], position)

    def centre(cell_index):
        if cell_index not in cells_info.index:
            return None
        return (float(cells_info.at[cell_index, x_col]), float(cells_info.at[cell_index, y_col]))

    tail_ids = sorted(ends)
    head_ids = sorted(starts)
    cost = np.full((len(tail_ids), len(head_ids)), np.nan)
    for i, tail in enumerate(tail_ids):
        end_frame, end_cell, _ = ends[tail]
        a = centre(end_cell)
        if a is None:
            continue
        for j, head in enumerate(head_ids):
            if head == tail:
                continue
            start_frame, start_cell, _ = starts[head]
            gap = start_frame - end_frame - 1
            if gap < 1 or gap > max_frame_gap:
                continue
            b = centre(start_cell)
            if b is None:
                continue
            distance = float(np.hypot(b[0] - a[0], b[1] - a[1]))
            if distance > max_centroid_distance * (gap + 1):
                continue
            cost[i, j] = distance

    if not np.isfinite(cost).any():
        return tracking_data, []

    sentinel = float(np.nanmax(cost[np.isfinite(cost)])) + 1.0
    size = max(cost.shape)
    square = np.full((size, size), sentinel)
    square[:cost.shape[0], :cost.shape[1]] = np.where(np.isfinite(cost), cost, sentinel)
    rows, columns = linear_sum_assignment(square)

    # Collect the accepted links first, then apply them as CHAINS.
    #
    # A5 (R15): the old loop applied a link only if neither end had been touched
    # (`if tail in absorbed or head in absorbed: continue`), so an A->B->C chain
    # kept whichever of its two links `linear_sum_assignment` happened to emit
    # first -- and which one that was depended on row order. Three segments one
    # gap apart came back as two tracks instead of one, understating by a
    # run-order-dependent amount the very cell-loss figure this feature exists
    # to improve. The links form disjoint paths (each tail links out once, each
    # head is absorbed once, and a head always starts after its tail ends, so
    # there are no cycles), so walking each path from its root closes the whole
    # chain deterministically.
    links = {}
    for i, j in zip(rows, columns):
        if i >= cost.shape[0] or j >= cost.shape[1] or square[i, j] == sentinel:
            continue
        links[tail_ids[i]] = (head_ids[j], float(cost[i, j]))

    merged = tracking_data.copy()
    row_of = {int(merged.at[p, id_col]): p for p in merged.index}
    joins = []
    absorbed = set()
    heads = {head for head, _ in links.values()}
    for root in sorted(t for t in links if t not in heads):
        current = root
        while current in links:
            head, distance = links[current]
            if head in absorbed:
                break
            root_row, head_row = row_of[root], row_of[head]
            for _, col in frame_cols:
                value = merged.at[head_row, col]
                if pd.notna(value):
                    merged.at[root_row, col] = value
                    merged.at[head_row, col] = None
            joins.append({
                "from_track_id": head, "into_track_id": root,
                "gap": starts[head][0] - ends[current][0] - 1,
                # The last frame held by the segment this link bridges FROM.
                # Recorded here rather than re-derived later, because after the
                # first link of a chain the surviving row's last frame is no
                # longer the seam -- `retrack_frame_range` needs the true one to
                # decide whether a join falls inside its range.
                "seam_frame": ends[current][0],
                "distance": round(distance, 2),
            })
            absorbed.add(head)
            current = head

    if absorbed:
        keep = [p for p in merged.index if int(merged.at[p, id_col]) not in absorbed]
        merged = merged.loc[keep].reset_index(drop=True)
    return merged, joins


def retrack_frame_range(
    cells_info: pd.DataFrame,
    tracking_data: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    feature_columns: Optional[list[str]] = None,
    weights: Optional[list[float]] = None,
    npy_folder: Optional[str] = None,
    overlap_weight: float = DEFAULT_OVERLAP_WEIGHT,
    centroid_weight: float = 1.0,
    motion_compensation: bool = False,
    cost_form: str = "additive",
    overlap_power: float = 1.0,
    no_link_ratio: Optional[float] = None,
    four_quadrant: bool = False,
    drift_mode: str = "none",
    drift_by_frame: Optional[dict] = None,
    max_frame_gap: int = 0,
) -> dict:
    """Re-solve tracking for frames `start_frame..end_frame` only, leaving
    every other frame's identities untouched. For use after hand-correcting
    segmentation: the masks changed, so the stored links for those frames are
    stale, but re-tracking the whole movie would renumber tracks and discard
    manual tracking corrections elsewhere.

    Returns `{"reconciled": [...], "new_tracks": [...], "frames": [...],
    "relinked": int, "tracking_data": DataFrame, "gap_joins": [...],
    "gap_joins_rejected_out_of_range": int}`.

    **Always use `result["tracking_data"]`.** The docstring used to have this
    backwards: appending a new track's row (`tracking_data.loc[max + 1] = ...`)
    is done *in place* and does NOT produce a new object, while the calls that
    DO rebind are `merge_tracks_from_frame` (seam reconciliation) and the
    gap-join application. So the caller's own reference is stale exactly when
    the docstring said it was current, and vice versa. Treat the argument as
    an input whose mutation is unspecified.

    **This is not the "local re-track" that was built and removed earlier.**
    That one re-ran Hungarian matching in a spatial neighborhood around a
    *tracking* correction -- re-running the algorithm that produced the wrong
    link, which unsurprisingly reproduced it. Here the *segmentation input
    itself changed*, so the old links describe cells that no longer exist and
    recomputing them is the only correct response.

    How identity survives the two seams:

    - **Leading seam.** Frame `start_frame - 1` is untouched, so each of its
      cell strings still resolves to a track row. The re-solved
      `(start-1, start)` match therefore says which existing track each
      re-tracked cell continues, and that cell is written into that same
      row. Identity is preserved across the seam for free. (This is the one
      idea worth taking from EPySeg, whose `match_by_max_overlap` propagates
      the previous frame's identities off disk -- minus its unconditional
      re-mint of the first frame, which is exactly what makes partial
      re-tracking impossible there.)
    - **Trailing seam.** Frame `end_frame + 1` still holds its OLD identities,
      which may disagree with the re-tracked chain. The re-solved
      `(end, end+1)` match says which `end+1` cell genuinely continues each
      re-tracked track; where that cell is currently claimed by a different
      track, that track's remaining future is spliced onto the re-tracked one
      via `merge_tracks_from_frame` and the swap is recorded in
      `"reconciled"`. EPySeg has no equivalent -- it simply leaves the seam
      broken.

    Cells inside the range with no anchor (a cell that appears mid-range)
    start a new track numbered `max(Index) + 1`, the same collision-free
    scheme the correction path already uses.
    """
    import manual_correction_tracking  # local: avoids a circular import at module load

    if start_frame > end_frame:
        raise ValueError(f"start_frame {start_frame} is after end_frame {end_frame}.")

    # Normalised exactly as `track_cells` does. Without it `.str.contains` below
    # raises on a non-string index that `track_cells` accepts -- a divergence
    # that the signature-reachability test cannot see, because it compares
    # keyword names rather than behaviour.
    cells_info = cells_info.copy()
    cells_info.index = cells_info.index.map(str)

    # A re-track must refuse what `track_cells` refuses. `track_cells` raises on
    # a mixed-sign `leading_edge` because one cost matrix over two populations
    # is not the problem it solves; this function used to build exactly that
    # matrix and return silently -- the divergence was on the quiet side.
    if "leading_edge" in cells_info.columns:
        edge = cells_info["leading_edge"]
        if not ((edge <= 0).all() or (edge > 0).all()):
            raise ValueError(
                "cells_info['leading_edge'] must be entirely <= 0, entirely > 0, or the column must be "
                "absent entirely -- re-tracking one cost matrix over both populations is not the problem "
                "track_cells solves (split populations with split_by_leading_edge before calling)."
            )

    id_col = tracking_data.columns[0]
    frame_numbers = sorted(set(
        int(match.group(1)) for idx in cells_info.index
        for match in [re.search(r"Cell(\d+)_", str(idx))] if match
    ))
    if not frame_numbers:
        raise ValueError("cells_info contains no recognizable Cell{frame}_ indices.")
    first_frame, last_frame = min(frame_numbers), max(frame_numbers)

    # Include the seam pairs: they are what let identity carry across the
    # boundaries instead of the range floating free.
    pair_start = max(start_frame - 1, first_frame)
    pair_end = min(end_frame, last_frame - 1)
    radius_fallback = cells_info["Circumscribed Circle Radius"].max()

    # The cost has to be the one `track_cells` uses. It was not: this built a
    # bare centroid cost with no overlap term and no displacement ceiling, so
    # re-tracking a range solved a DIFFERENT problem than the full run had for
    # those same frames, and could "repair" a link into something the full run
    # would never have chosen.
    #
    # That was fixed once and then broken TWICE MORE, because every parameter
    # added to `track_cells` had to be threaded here by hand and nothing
    # checked. Measured cost the second time: 15.96% of links differing between
    # a registered full run and a re-track, 40.95% on the worst frame pair.
    # The invariant is now enforced by a signature-reachability test and a
    # link-equivalence test in this module's self-test -- a comment is not an
    # enforcement mechanism.
    if feature_columns is None:
        feature_columns = DEFAULT_TRACKING_FEATURES
    if weights is None:
        weights = resolve_weights(feature_columns)

    _validate_drift_mode(drift_mode)
    if drift_mode != "none" and not drift_by_frame and not (npy_folder and os.path.isdir(npy_folder)):
        logger.warning(
            "drift_mode=%r was requested but no npy_folder and no drift_by_frame were given, so the "
            "drift is (0, 0) and this re-track is effectively unregistered.", drift_mode,
        )

    label_loader = _FrameLabelLoader(npy_folder)
    _warn_if_overlap_unreachable(label_loader, npy_folder, overlap_weight, drift_mode,
                                 "retrack_frame_range")
    links: dict[int, dict[str, str]] = {}
    previous_cells, previous_matches = None, None
    for frame in range(pair_start, pair_end + 1):
        cells_a = cells_info[cells_info.index.str.contains(f"Cell{frame}_")]
        cells_b = cells_info[cells_info.index.str.contains(f"Cell{frame + 1}_")]
        if cells_a.empty or cells_b.empty:
            continue

        shift = (0.0, 0.0)
        if drift_mode != "none":
            shift = (drift_by_frame or {}).get(frame) or _estimate_drift(label_loader, frame)

        overlaps = None
        if label_loader and overlap_weight != 0:
            labels_a, labels_b = label_loader.get(frame), label_loader.get(frame + 1)
            if labels_a is not None and labels_b is not None:
                if drift_mode == "mask" and (shift[0] or shift[1]):
                    labels_a = _shift_labels(labels_a, shift[0], shift[1])
                overlaps = mask_overlap_fractions(labels_a, labels_b)

        predicted = None
        if motion_compensation:
            predicted = predict_centroids(previous_matches, previous_cells, cells_a, feature_columns)
        if drift_mode != "none" and (shift[0] or shift[1]) and not cells_a.empty:
            _, _, cx, cy = _gate_indices(list(feature_columns))
            x_col, y_col = feature_columns[cx], feature_columns[cy]
            base = predicted or {}
            predicted = {
                index: (base.get(index, (float(cells_a.at[index, x_col]),
                                         float(cells_a.at[index, y_col])))[0] + shift[0],
                        base.get(index, (float(cells_a.at[index, x_col]),
                                         float(cells_a.at[index, y_col])))[1] + shift[1])
                for index in cells_a.index
            }

        cost = build_cost_matrix(
            cells_a, cells_b, feature_columns, weights,
            overlaps=overlaps, overlap_weight=overlap_weight,
            max_centroid_distance=radius_fallback,
            centroid_weight=centroid_weight, predicted_centroids=predicted,
            cost_form=cost_form, overlap_power=overlap_power,
        )
        sentinel = _no_match_sentinel(cost, radius_fallback, no_link_ratio=no_link_ratio)
        matched = match_frames(cost, sentinel, four_quadrant=four_quadrant)
        matched = matched[matched["Difference"] != sentinel]
        links[frame] = dict(zip(matched["frame1_cell"], matched["frame2_cell"]))
        previous_cells, previous_matches = cells_a, links[frame]

    def _row_holding(frame: int, cell_index: str):
        col = f"Frame{frame}"
        if col not in tracking_data.columns:
            return None
        rows = tracking_data.index[tracking_data[col] == cell_index]
        return rows[0] if len(rows) else None

    # Walk each re-solved link forward, writing continuations into the row
    # that already owns the chain's first cell.
    result = {"reconciled": [], "new_tracks": [], "frames": list(range(start_frame, end_frame + 1)), "relinked": 0}
    claimed_this_run: set = set()

    for frame in sorted(links):
        for cell_a, cell_b in links[frame].items():
            row = _row_holding(frame, cell_a)
            target_col = f"Frame{frame + 1}"
            if target_col not in tracking_data.columns:
                continue

            if row is None:
                # No existing track owns this chain's start -- a cell that
                # appeared inside the range. Give it a fresh, non-colliding id.
                new_id = int(tracking_data[id_col].max()) + 1
                blank = {c: None for c in tracking_data.columns}
                blank[id_col] = new_id
                blank[f"Frame{frame}"] = cell_a
                tracking_data.loc[tracking_data.index.max() + 1] = blank
                row = tracking_data.index[tracking_data[id_col] == new_id][0]
                result["new_tracks"].append({"track_id": new_id, "from_frame": frame, "cell_index": cell_a})

            existing = tracking_data.at[row, target_col]
            if pd.notna(existing) and existing == cell_b:
                continue  # already links this way; nothing to change

            # Whoever else currently claims cell_b must give it up first.
            other = _row_holding(frame + 1, cell_b)
            if other is not None and other != row:
                other_id = int(tracking_data.at[other, id_col])
                this_id = int(tracking_data.at[row, id_col])
                if frame + 1 > end_frame:
                    # Trailing seam: cell_b belongs to the untouched future, so
                    # splice that whole future onto the re-tracked track rather
                    # than orphaning it at the boundary.
                    tracking_data, _displaced = manual_correction_tracking.merge_tracks_from_frame(
                        tracking_data, frame + 1, other_id, this_id,
                    )
                    result["reconciled"].append(
                        {"frame": frame + 1, "from_track_id": other_id, "into_track_id": this_id, "cell_index": cell_b},
                    )
                    row = tracking_data.index[tracking_data[id_col] == this_id][0]
                    continue
                tracking_data.at[other, target_col] = None

            tracking_data.at[row, target_col] = cell_b
            claimed_this_run.add(cell_b)
            result["relinked"] += 1

    # Any cell inside the range that no link reached is genuinely unmatched
    # now; clear stale claims on cells that no longer exist in cells_info.
    for frame in range(start_frame, end_frame + 1):
        col = f"Frame{frame}"
        if col not in tracking_data.columns:
            continue
        present = set(cells_info.index[cells_info.index.str.startswith(f"Cell{frame}_")])
        for pos in tracking_data.index:
            value = tracking_data.at[pos, col]
            if pd.notna(value) and value not in present:
                tracking_data.at[pos, col] = None

    if max_frame_gap:
        # `track_cells` gap-closes after its merge, so a re-track that skipped
        # it would leave the range structurally different from the rest of the
        # movie. But `close_track_gaps` sees the WHOLE table, and the joins it
        # proposes are not all inside the range the caller asked for: a
        # re-track bounded to [1,2] was observed merging a track spanning
        # frames 8-9 into one spanning 5-6 -- frames the caller excluded, and
        # exactly the manual corrections this function exists to preserve.
        # So: let it solve over everything (a segment's best partner may sit
        # outside the range), then apply only the joins whose bridged hole
        # lies wholly within [start_frame, end_frame].
        id_col = tracking_data.columns[0]
        _, proposed = close_track_gaps(
            tracking_data.copy(deep=True), cells_info, max_frame_gap=max_frame_gap,
            feature_columns=feature_columns, max_centroid_distance=radius_fallback,
        )
        # A chain is all-or-nothing: its links share one surviving root and
        # must be applied in order, so applying only its tail links would merge
        # a later segment into a root it was never adjacent to.
        out_of_range_roots = {
            int(j["into_track_id"]) for j in proposed
            if not (start_frame <= _join_hole(j)[0] and _join_hole(j)[1] <= end_frame)
        }
        joins, rejected = [], 0
        for join in proposed:
            if int(join["into_track_id"]) in out_of_range_roots:
                rejected += 1
                continue
            joins.append(join)
        if joins:
            tracking_data = _apply_selected_joins(tracking_data, id_col, joins)
        result["gap_joins"] = joins
        result["gap_joins_rejected_out_of_range"] = rejected

    result["tracking_data"] = tracking_data
    return result


def split_by_leading_edge(cells_info: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split `cells_info` (must have a `leading_edge` column and a plain
    `'Cell Index'` column, not yet indexed) into (mes, epi) populations.

    Bug Fix #3: uses `<= 0` (mes) / `> 0` (epi), consistently with
    `track_cells`'s own internal population-detection convention. v1's
    `perform_cell_tracking_and_generate_images` (src/gui_main.py:796,800)
    used strict `< 0` / `> 0`, silently excluding any `leading_edge == 0`
    cell from tracking entirely.
    """
    mes_cells_info = cells_info[cells_info["leading_edge"] <= 0].copy()
    epi_cells_info = cells_info[cells_info["leading_edge"] > 0].copy()
    return mes_cells_info, epi_cells_info


def track_all_populations(
    cells_info: pd.DataFrame,
    output_dir: str,
    progress_cb: Optional[Callable[[str, int], None]] = None,
    npy_folder: Optional[str] = None,
    overlap_weight: float = DEFAULT_OVERLAP_WEIGHT,
    centroid_weight: float = 1.0,
    motion_compensation: bool = False,
    max_frame_gap: int = 0,
    cost_form: str = "additive",
    overlap_power: float = 1.0,
    no_link_ratio: Optional[float] = None,
    four_quadrant: bool = False,
    drift_mode: str = "none",
    drift_by_frame: Optional[dict] = None,
    feature_columns: Optional[list[str]] = None,
    weights: Optional[list[float]] = None,
) -> pd.DataFrame:
    """Orchestrates tracking for one or two populations (mes/epi, if
    `cells_info` has a `leading_edge` column; otherwise a single
    unclassified population), continuing cell-index numbering across both,
    concatenating results, and writing the final
    `all_cell_tracking/all_cell_merged_tracking_results.xlsx` under
    `output_dir`. Ported from the non-GUI parts of
    `perform_cell_tracking_and_generate_images`, src/gui_main.py:763-848
    (excludes the `ComparisonMainWindow` tracking-image-generation tail at
    856-874, which stays GUI-only).

    `cells_info` should have `'Cell Index'` as a plain column (not the
    index) -- matching v1's runtime shape (loaded via plain
    `pd.read_excel(cells_info_path)`).

    Track `Index` numbering is scoped to this call: the first population starts
    at 1 and the second continues from it, regardless of what a previous run
    left in `output_dir`. Re-running is therefore idempotent in the ids, which
    is what any manual correction keyed by `Index` depends on.
    """
    has_leading_edge = "leading_edge" in cells_info.columns
    all_tracking_data = pd.DataFrame()

    if has_leading_edge:
        mes_cells_info, epi_cells_info = split_by_leading_edge(cells_info)
        mes_cells_info.to_excel(os.path.join(output_dir, "mes_cells_info.xlsx"), sheet_name="mesCells", index=False)
        epi_cells_info.to_excel(os.path.join(output_dir, "epi_cells_info.xlsx"), sheet_name="epiCells", index=False)
        mes_cells_info = mes_cells_info.set_index("Cell Index")
        epi_cells_info = epi_cells_info.set_index("Cell Index")

        # A5 (R15): scope the id numbering to THIS run. Left to auto-detect,
        # each `track_cells` reads the sibling population's Excel file off
        # disk -- which, on a re-run into the same output directory, is the
        # PREVIOUS run's. Three consecutive calls returned `Index` [1,2], then
        # [3,4], then [5,6], so any manual correction keyed by `Index`
        # misaligned after a re-run. Seeding explicitly makes the numbering a
        # function of this call and nothing else. (`track_cells` keeps its
        # file-reading fallback for direct callers, which is v1's behaviour.)
        epi_start_index = 1
        if not mes_cells_info.empty:
            mes_tracking = track_cells(mes_cells_info, output_dir, start_index=1, progress_cb=progress_cb, npy_folder=npy_folder, overlap_weight=overlap_weight,
                                       centroid_weight=centroid_weight, motion_compensation=motion_compensation,
                                       max_frame_gap=max_frame_gap, cost_form=cost_form,
                                       overlap_power=overlap_power, no_link_ratio=no_link_ratio,
                                       four_quadrant=four_quadrant, drift_mode=drift_mode,
                                       drift_by_frame=drift_by_frame,
                                       feature_columns=feature_columns, weights=weights)
            all_tracking_data = pd.concat([all_tracking_data, mes_tracking], axis=0, ignore_index=True)
            if not mes_tracking.empty:
                epi_start_index = int(mes_tracking.iloc[:, 0].max()) + 1
        if not epi_cells_info.empty:
            epi_tracking = track_cells(epi_cells_info, output_dir, start_index=epi_start_index, progress_cb=progress_cb, npy_folder=npy_folder, overlap_weight=overlap_weight,
                                       centroid_weight=centroid_weight, motion_compensation=motion_compensation,
                                       max_frame_gap=max_frame_gap, cost_form=cost_form,
                                       overlap_power=overlap_power, no_link_ratio=no_link_ratio,
                                       four_quadrant=four_quadrant, drift_mode=drift_mode,
                                       drift_by_frame=drift_by_frame,
                                       feature_columns=feature_columns, weights=weights)
            all_tracking_data = pd.concat([all_tracking_data, epi_tracking], axis=0, ignore_index=True)
    else:
        cells_info = cells_info.set_index("Cell Index")
        all_tracking_data = track_cells(cells_info, output_dir, start_index=1, progress_cb=progress_cb, npy_folder=npy_folder, overlap_weight=overlap_weight,
                                       centroid_weight=centroid_weight, motion_compensation=motion_compensation,
                                       max_frame_gap=max_frame_gap, cost_form=cost_form,
                                       overlap_power=overlap_power, no_link_ratio=no_link_ratio,
                                       four_quadrant=four_quadrant, drift_mode=drift_mode,
                                       drift_by_frame=drift_by_frame,
                                       feature_columns=feature_columns, weights=weights)

    if all_tracking_data is not None and not all_tracking_data.empty:
        all_cell_dir = os.path.join(output_dir, "all_cell_tracking")
        os.makedirs(all_cell_dir, exist_ok=True)
        all_tracking_data.to_excel(
            os.path.join(all_cell_dir, "all_cell_merged_tracking_results.xlsx"), index=False
        )

    return all_tracking_data


if __name__ == "__main__":
    import tempfile

    import argparse

    parser = argparse.ArgumentParser(description="Standalone debug entry for v2.cell_tracking")
    parser.add_argument("--cells-info", help="Cells_info.xlsx path (omit to run the self-test)")
    parser.add_argument("--output-dir")
    # Accepted so every module in v2/ answers the same flag. Here the self-test
    # is what runs when `--cells-info` is omitted, so this is a no-op -- but a
    # preflight checklist should not have to remember which modules take it and
    # which reject it. Round 19 A1: the first real click-test lost its §0 step
    # to exactly that.
    parser.add_argument("--self-test", action="store_true",
                        help="run the self-test (the default when --cells-info is omitted)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not args.cells_info:
        # Self-test: a 5-frame, 3-cell movie. Cells sit far apart so the
        # Hungarian match is unambiguous and the assertions test the seam
        # bookkeeping rather than the matcher.
        def _cell(name, x, y):
            row = {c: 0.0 for c in DEFAULT_TRACKING_FEATURES}
            row["Center X Coordinate"], row["Center Y Coordinate"] = x, y
            row["Maximum Horizontal Length"] = row["Maximum Vertical Length"] = 30.0
            row["Circumscribed Circle Radius"] = 10.0
            return pd.Series(row, name=name)

        rows = []
        for f in range(1, 6):
            rows.append(_cell(f"Cell{f}_1", 10.0, 10.0))
            rows.append(_cell(f"Cell{f}_2", 200.0, 10.0))
            if f in (3, 4):  # a cell that only exists mid-range
                rows.append(_cell(f"Cell{f}_9", 10.0, 200.0))
        info = pd.DataFrame(rows)

        tracking = pd.DataFrame({
            "Index": [1, 2],
            **{f"Frame{f}": [f"Cell{f}_1", f"Cell{f}_2"] for f in range(1, 6)},
        })
        # Break the trailing seam: track 2's frame-5 claim is moved to a third
        # track, so re-tracking [2,4] must reconcile it rather than orphan it.
        tracking.loc[2] = {"Index": 99, **{f"Frame{f}": None for f in range(1, 6)}}
        tracking.at[2, "Frame5"] = "Cell5_2"
        tracking.at[1, "Frame5"] = None

        before_ids = set(tracking["Index"])
        out = retrack_frame_range(info, tracking, start_frame=2, end_frame=4)
        tracking = out["tracking_data"]
        print("retrack_frame_range ->", {k: v for k, v in out.items() if k != "tracking_data"})
        print(tracking, "\n")

        row1 = tracking.index[tracking["Index"] == 1][0]
        assert all(tracking.at[row1, f"Frame{f}"] == f"Cell{f}_1" for f in range(1, 6)), "track 1 must be intact"
        assert 1 in before_ids and 2 in before_ids, "pre-existing ids must not be renumbered"

        assert out["reconciled"], "the deliberately-broken trailing seam should have been reconciled"
        rec = out["reconciled"][0]
        assert rec["from_track_id"] == 99 and rec["cell_index"] == "Cell5_2", rec
        row2 = tracking.index[tracking["Index"] == 2][0]
        assert tracking.at[row2, "Frame5"] == "Cell5_2", "seam splice must land on the re-tracked track"

        assert out["new_tracks"], "the mid-range cell should have started a new track"
        new_id = out["new_tracks"][0]["track_id"]
        assert new_id not in before_ids, "a new track must not collide with an existing id"
        rown = tracking.index[tracking["Index"] == new_id][0]
        assert tracking.at[rown, "Frame3"] == "Cell3_9" and tracking.at[rown, "Frame4"] == "Cell4_9", tracking.loc[rown]
        assert pd.isna(tracking.at[rown, "Frame1"]), "the new track must not claim frames before it appeared"

        # --- mask overlap: two cells that swap centroid proximity, where only
        # overlap disambiguates them.
        a = np.zeros((40, 40), dtype=np.int32)
        a[5:15, 5:35] = 1          # wide flat cell
        a[20:35, 5:15] = 2         # tall narrow cell
        b = np.zeros((40, 40), dtype=np.int32)
        b[6:16, 5:35] = 1          # same shapes, shifted a little
        b[21:36, 5:15] = 2
        iou = mask_overlap_fractions(a, b)
        print("\nmask_overlap_fractions: self-overlaps=%.2f/%.2f  cross=%.2f"
              % (iou[(1, 1)], iou[(2, 2)], iou.get((1, 2), 0.0)))
        assert iou[(1, 1)] > 0.8 and iou[(2, 2)] > 0.8, iou
        assert iou.get((1, 2), 0.0) == 0.0, "cells that never touch must not overlap"

        # overlap_weight=0 must reproduce the centroid-only cost exactly
        frame1 = pd.DataFrame([_cell("Cell1_1", 10.0, 10.0), _cell("Cell1_2", 200.0, 10.0)])
        frame2 = pd.DataFrame([_cell("Cell2_1", 12.0, 10.0), _cell("Cell2_2", 202.0, 10.0)])
        plain = build_cost_matrix(frame1, frame2)
        off = build_cost_matrix(frame1, frame2, overlaps={(1, 1): 0.9}, overlap_weight=0)
        assert plain.equals(off), "overlap_weight=0 must be identical to no overlaps at all"
        withov = build_cost_matrix(frame1, frame2, overlaps={(1, 1): 1.0}, overlap_weight=1.0)
        assert withov.iat[0, 0] == plain.iat[0, 0], "a perfect overlap must add no penalty"
        partial = build_cost_matrix(frame1, frame2, overlaps={(1, 1): 0.0}, overlap_weight=1.0)
        assert partial.iat[0, 0] > plain.iat[0, 0], "zero overlap must be penalised"
        print("  -> overlap term is off by default weight 0, free when IoU=1, penalised when IoU=0.")

        # --- centroid_weight: the cost scales, the GATES do not.
        half = build_cost_matrix(frame1, frame2, centroid_weight=0.5)
        assert abs(half.iat[0, 0] - 0.5 * plain.iat[0, 0]) < 1e-9, "centroid_weight must scale the distance"
        # Cell1_1 -> Cell2_2 is 190 px apart, far beyond the 30 px extent: it is
        # NaN at full weight and must stay NaN at zero weight, or a distant cell
        # could win on a sliver of overlap.
        assert pd.isna(plain.iat[0, 1]), "the far pair must be gated at full weight"
        iou_only = build_cost_matrix(
            frame1, frame2, overlaps={(1, 1): 0.5, (1, 2): 1.0}, overlap_weight=1.0, centroid_weight=0.0,
        )
        assert pd.isna(iou_only.iat[0, 1]), "the gate must survive centroid_weight=0"
        assert iou_only.iat[0, 0] > 0, "IoU-only must still cost something for imperfect overlap"
        try:
            build_cost_matrix(frame1, frame2, centroid_weight=0.0)
        except ValueError:
            pass
        else:
            raise AssertionError("disabling both terms must be rejected, not silently return zeros")
        print("  -> centroid_weight scales the cost; the plausibility gates ignore it.")

        # --- motion compensation: a cell moving a steady 8 px/frame is
        # predicted onto its own next position, so the residual collapses.
        prev = pd.DataFrame([_cell("Cell1_1", 10.0, 10.0)])
        cur = pd.DataFrame([_cell("Cell2_1", 18.0, 10.0)])
        nxt = pd.DataFrame([_cell("Cell3_1", 26.0, 10.0)])
        predicted = predict_centroids({"Cell1_1": "Cell2_1"}, prev, cur)
        assert predicted["Cell2_1"] == (26.0, 10.0), predicted
        naive = build_cost_matrix(cur, nxt)
        compensated = build_cost_matrix(cur, nxt, predicted_centroids=predicted)
        print("  cost to the true successor: naive=%.1f  motion-compensated=%.1f"
              % (naive.iat[0, 0], compensated.iat[0, 0]))
        assert naive.iat[0, 0] == 8.0 and compensated.iat[0, 0] == 0.0, (naive.iat[0, 0], compensated.iat[0, 0])
        # A cell with no previous link falls back to the frame-wide median,
        # which is what the moving cell supplies here.
        cur_plus = pd.DataFrame([_cell("Cell2_1", 18.0, 10.0), _cell("Cell2_7", 200.0, 10.0)])
        fallback = predict_centroids({"Cell1_1": "Cell2_1"}, prev, cur_plus)
        assert fallback["Cell2_7"] == (208.0, 10.0), fallback
        print("  -> unlinked cells inherit the frame-wide median displacement, not zero.")

        # --- gap closing: a track that loses its cell for one frame must
        # re-attach rather than mint a new id, and must NOT do so when off.
        gap_info = pd.DataFrame([
            _cell("Cell1_1", 10.0, 10.0), _cell("Cell2_1", 12.0, 10.0),
            _cell("Cell4_1", 16.0, 10.0), _cell("Cell5_1", 18.0, 10.0),
        ])
        gap_tracks = pd.DataFrame({
            "Index": [1, 2],
            "Frame1": ["Cell1_1", None], "Frame2": ["Cell2_1", None],
            "Frame3": [None, None], "Frame4": [None, "Cell4_1"], "Frame5": [None, "Cell5_1"],
        })
        joined, joins = close_track_gaps(gap_tracks.copy(), gap_info, max_frame_gap=1)
        print("\ngap closing:", joins)
        assert len(joins) == 1 and joins[0]["gap"] == 1, joins
        assert len(joined) == 1, "the two segments must become one track"
        row = joined.index[0]
        assert joined.at[row, "Frame1"] == "Cell1_1" and joined.at[row, "Frame5"] == "Cell5_1", joined
        off, no_joins = close_track_gaps(gap_tracks.copy(), gap_info, max_frame_gap=0)
        assert not no_joins and len(off) == 2, "max_frame_gap=0 must change nothing"
        # A 2-frame hole must not be closed when only 1 is allowed.
        far_info = pd.DataFrame([_cell("Cell1_1", 10.0, 10.0), _cell("Cell5_1", 18.0, 10.0)])
        far_tracks = pd.DataFrame({
            "Index": [1, 2], "Frame1": ["Cell1_1", None], "Frame2": [None, None],
            "Frame3": [None, None], "Frame4": [None, None], "Frame5": [None, "Cell5_1"],
        })
        _, far_joins = close_track_gaps(far_tracks, far_info, max_frame_gap=1)
        assert not far_joins, "a 3-frame hole must not be closed at max_frame_gap=1"
        print("  -> gaps close only within the allowed span, and not at all when off.")

        # --- LAP cost-structure alternatives (all default-off).
        assert _no_match_sentinel(plain, 10.0) == max(plain.max().max(), 10.0) + 1.0
        ratio = _no_match_sentinel(plain, 10.0, no_link_ratio=1.05)
        assert abs(ratio - 1.05 * max(plain.max().max(), 10.0)) < 1e-9, ratio
        print("\nsentinel: additive %.3f  vs TrackMate ratio 1.05 -> %.3f"
              % (_no_match_sentinel(plain, 10.0), ratio))

        power = build_cost_matrix(frame1, frame2, overlaps={(1, 1): 0.5}, overlap_weight=1.0, overlap_power=2.0)
        linear = build_cost_matrix(frame1, frame2, overlaps={(1, 1): 0.5}, overlap_weight=1.0)
        # (1-0.5)^2 = 0.25 < 0.5, so squaring the deficit must REDUCE the penalty here.
        assert power.iat[0, 0] < linear.iat[0, 0], (power.iat[0, 0], linear.iat[0, 0])
        assert build_cost_matrix(frame1, frame2, overlaps={(1, 1): 0.5}, overlap_power=1.0).equals(linear)
        multiplicative = build_cost_matrix(
            frame1, frame2, overlaps={(1, 1): 0.5}, overlap_weight=1.0, cost_form="multiplicative")
        assert abs(multiplicative.iat[0, 0] - (2.0 * 1.5) ** 2) < 1e-9, multiplicative.iat[0, 0]
        print("  overlap_power=1 is unchanged; multiplicative form gives (d*P)^2 = %.1f"
              % multiplicative.iat[0, 0])

        # Four-quadrant must reproduce the rectangular answer on the shipped cost.
        sentinel_value = _no_match_sentinel(plain, 10.0)
        rectangular = match_frames(plain, sentinel_value)
        blocked = match_frames(plain, sentinel_value, four_quadrant=True)
        rect_links = {(a, b) for a, b, d in rectangular.itertuples(index=False) if d != sentinel_value}
        block_links = {(a, b) for a, b, d in blocked.itertuples(index=False) if d != sentinel_value}
        assert rect_links == block_links, (rect_links, block_links)
        print("  four-quadrant reproduces the rectangular links exactly (%d links)." % len(rect_links))
        print("  -> every LAP alternative is off by default and verified inert when off.")

        # --- vectorised build_cost_matrix must equal the original loop EXACTLY,
        # across every option, including the NaN pattern from both gates.
        rng = np.random.default_rng(0)
        many1 = pd.DataFrame([_cell(f"Cell1_{n}", *rng.uniform(0, 120, 2)) for n in range(1, 26)])
        many2 = pd.DataFrame([_cell(f"Cell2_{n}", *rng.uniform(0, 120, 2)) for n in range(1, 31)])
        fake_overlaps = {(a, b): float(rng.uniform(0, 1)) for a in range(1, 26) for b in range(1, 31)}
        predictions = {i: (float(rng.uniform(0, 120)), float(rng.uniform(0, 120))) for i in many1.index}
        cases = [
            dict(),
            dict(overlaps=fake_overlaps, overlap_weight=1.0),
            dict(overlaps=fake_overlaps, overlap_weight=2.5, overlap_power=3.0),
            dict(overlaps=fake_overlaps, overlap_weight=1.0, cost_form="multiplicative"),
            dict(overlaps=fake_overlaps, overlap_weight=1.0, centroid_weight=0.0),
            dict(max_centroid_distance=25.0),
            dict(predicted_centroids=predictions, max_centroid_distance=40.0),
            dict(overlaps=fake_overlaps, overlap_weight=1.0, predicted_centroids=predictions,
                 max_centroid_distance=40.0, overlap_power=2.0),
        ]
        # What is asserted, and why not bit-identity: numpy's vectorised sqrt
        # does not always take the same code path as scalar sqrt, so one entry
        # in 133 differed by 3.55e-15 -- a single ULP. That is inherent to
        # vectorising and cannot be removed without giving the speed back. So
        # assert the three things that actually decide behaviour: the NaN
        # pattern (both gates) exactly, the values to far below any meaningful
        # tolerance, and -- the binding one -- that the resulting ASSIGNMENT is
        # identical. Byte-identical tracking on real datasets is verified
        # separately against the stored tables.
        worst = 0.0
        for case in cases:
            fast = build_cost_matrix(many1, many2, **case)
            slow = _build_cost_matrix_reference(many1, many2, **case)
            a, b = fast.to_numpy(), slow.to_numpy()
            assert np.array_equal(np.isnan(a), np.isnan(b)), ("NaN pattern differs", case)
            assert list(fast.index) == list(slow.index) and list(fast.columns) == list(slow.columns)
            finite = ~np.isnan(a)
            if finite.any():
                scale = np.maximum(np.abs(b[finite]), 1.0)
                worst = max(worst, float(np.max(np.abs(a[finite] - b[finite]) / scale)))
            sentinel_value = _no_match_sentinel(fast, 10.0)
            links_fast = set(map(tuple, match_frames(fast, sentinel_value)[["frame1_cell", "frame2_cell"]].values))
            links_slow = set(map(tuple, match_frames(slow, sentinel_value)[["frame1_cell", "frame2_cell"]].values))
            assert links_fast == links_slow, ("assignment differs", case)
        assert worst < 1e-12, worst
        print("\nvectorised cost matrix vs reference loop over %d option combinations:" % len(cases))
        print("  identical NaN pattern; worst relative difference %.2e (one ULP of sqrt);" % worst)
        print("  and the resulting assignment is identical in every case.")
        gated = int(np.isnan(build_cost_matrix(many1, many2, max_centroid_distance=25.0).to_numpy()).sum())
        assert 0 < gated < many1.shape[0] * many2.shape[0], gated
        print("  (%d of %d pairs gated in the ceiling case, so the NaN paths are actually exercised)"
              % (gated, many1.shape[0] * many2.shape[0]))

        # --- R24: the Cells_info schema is not tracking's business. The default
        # cost must be the same matrix whether the table carries the whole
        # schema -- 17 columns when this was written, 18 since Round 26 added
        # `Edge Distance`, with -1 and NaN in the unweighted columns, as
        # `extract_cell_info` writes them -- or only the four gate columns, or
        # the pre-Round-24 15 columns an old project has on disk. When the
        # default was the whole schema, the last case raised KeyError.
        assert "Fitted Ellipse Angle" not in DEFAULT_TRACKING_FEATURES
        assert set(WEIGHTED_COLUMNS) <= set(DEFAULT_TRACKING_FEATURES)
        assert resolve_weights(DEFAULT_TRACKING_FEATURES) == DEFAULT_TRACKING_WEIGHTS
        full1, full2 = many1.copy(), many2.copy()
        for frame_ in (full1, full2):
            for column in data_input.CELLS_INFO_COLUMNS:
                if column not in frame_.columns:
                    frame_[column] = rng.uniform(1, 50, len(frame_))
            frame_.iloc[::4, frame_.columns.get_loc("Fitted Ellipse Angle")] = -1.0
            frame_.iloc[1::4, frame_.columns.get_loc("Circularity")] = np.nan
        old1 = full1.drop(columns=["Fitted Ellipse Angle", "Circularity"])
        old2 = full2.drop(columns=["Fitted Ellipse Angle", "Circularity"])
        for case in cases:
            ref = build_cost_matrix(many1, many2, **case).to_numpy()
            for a1, a2 in ((full1, full2), (old1, old2)):
                got = build_cost_matrix(a1, a2, **case).to_numpy()
                assert np.array_equal(got, ref, equal_nan=True), case
            schema_wide = build_cost_matrix(full1, full2, feature_columns=list(data_input.CELLS_INFO_COLUMNS),
                                            **case).to_numpy()
            assert np.array_equal(schema_wide, ref, equal_nan=True), case
        print("R24: default cost identical (bit-for-bit) on the full-schema table with -1/NaN in the")
        print("  restored columns, on a 15-column pre-R24 table, and with the full schema passed")
        print("  as feature_columns (weights resolved by name).")

        # --- A3: the population split. NO dataset in test_data/ carries a
        # `leading_edge` column (it is added by the Classify + leading-edge
        # step, which none of the stored projects ran), so this path can only
        # be exercised synthetically -- which is a reason to test it here, not
        # a reason to skip it.
        #
        # Two populations laid out so the ONLY plausible partner for a mes cell
        # is an epi cell: if the two are solved as one problem they link across
        # populations, which the full run cannot do.
        def _pop(name, x, y, edge):
            row = {c: 0.0 for c in DEFAULT_TRACKING_FEATURES}
            row["Center X Coordinate"], row["Center Y Coordinate"] = x, y
            row["Maximum Horizontal Length"] = row["Maximum Vertical Length"] = 30.0
            row["Circumscribed Circle Radius"] = 10.0
            row["leading_edge"] = edge
            return pd.Series(row, name=name)

        split_rows = []
        for f in (1, 2, 3):
            split_rows.append(_pop(f"Cell{f}_1", 10.0 + f, 10.0, -1.0))   # mes
            split_rows.append(_pop(f"Cell{f}_2", 12.0 + f, 10.0, 1.0))    # epi, right beside it
        split_info = pd.DataFrame(split_rows)
        split_info.index.name = "Cell Index"
        flat = split_info.reset_index()
        mes, epi = split_by_leading_edge(flat)
        assert len(mes) == 3 and len(epi) == 3, (len(mes), len(epi))
        # Each population, solved on its own, can only link within itself.
        for part in (mes, epi):
            indexed = part.set_index("Cell Index")
            tracks = pd.DataFrame({"Index": [1], **{f"Frame{f}": [indexed.index[f - 1]] for f in (1, 2, 3)}})
            out = retrack_frame_range(indexed, tracks, 2, 2)
            claimed = [v for k, v in out["tracking_data"].iloc[0].items() if str(k).startswith("Frame")]
            labels = {str(c).rsplit("_", 1)[1] for c in claimed if pd.notna(c)}
            assert len(labels) == 1, f"a population must not link across to the other: {claimed}"
        print("\npopulation split: mes and epi re-track as two independent problems")
        print("  (no dataset in test_data/ has a leading_edge column, so this is synthetic on purpose)")

        # --- B: signature reachability. A knob on a cost primitive that no
        # pipeline entry point accepts is not implemented, whatever its
        # docstring says.
        import inspect
        def keywords(fn):
            return {n for n, p in inspect.signature(fn).parameters.items()
                    if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and p.default is not p.empty}
        primitives = {}
        for fn in (build_cost_matrix, _no_match_sentinel, match_frames):
            primitives |= {k: fn.__name__ for k in keywords(fn)}
        # Arguments the entry points supply themselves rather than accept.
        supplied = {"overlaps", "max_centroid_distance", "predicted_centroids",
                    "fallback", "sentinel", "cost_matrix", "no_link_ratio"}
        for entry in (track_cells, retrack_frame_range):
            have = keywords(entry)
            for name, owner in primitives.items():
                if name in supplied or name in have:
                    continue
                assert name in UNTHREADED_PARAMETERS, (
                    f"{owner}'s '{name}' is not reachable from {entry.__name__} and is not listed "
                    f"in UNTHREADED_PARAMETERS with a reason"
                )
        missing = keywords(track_cells) - keywords(track_all_populations) - {
            "start_index", "progress_cb", "output_dir"}
        assert not missing, f"track_cells parameters unreachable from track_all_populations: {missing}"
        for name in UNTHREADED_PARAMETERS:
            assert UNTHREADED_PARAMETERS[name], f"{name} needs a stated reason"
        print("\nreachability: every cost-primitive keyword reaches track_cells AND")
        print("  retrack_frame_range, or is listed in UNTHREADED_PARAMETERS with a reason")
        print("  (%d excluded: %s)" % (len(UNTHREADED_PARAMETERS), ", ".join(sorted(UNTHREADED_PARAMETERS))))

        # --- C1: a frame COUNT is not a frame SPAN.
        import tempfile as _tmp
        holed = pd.DataFrame([_cell(f"Cell{f}_{i}", 10.0 if i == 1 else 200.0, 10.0)
                              for f in (1, 2, 5, 6, 9, 10) for i in (1, 2)])
        out_dir = _tmp.mkdtemp()
        merged_holed = track_cells(holed, out_dir)
        produced = sorted(int(re.search(r"new_(\d+)_", f).group(1))
                          for f in os.listdir(os.path.join(out_dir, "All_cell_tracking"))
                          if "Matching_Results" in f)
        columns = {int(c[len("Frame"):]) for c in merged_holed.columns if c.startswith("Frame")}
        assert max(produced) + 1 in columns, (produced, sorted(columns))
        print("\nC1: non-contiguous frames %s survive the merge; columns reach %d"
              % (produced, max(columns)))

        # --- C2: a second run must not stitch in the first run's pair files.
        dirty = _tmp.mkdtemp()
        track_cells(pd.DataFrame([_cell(f"Cell{f}_{i}", 10.0 if i == 1 else 200.0, 10.0)
                                  for f in (1, 2, 3, 4, 5) for i in (1, 2)]), dirty)
        shrunk = track_cells(pd.DataFrame([_cell(f"Cell{f}_1", 10.0, 10.0) for f in (1, 2)]), dirty)
        ghosts = [v for v in shrunk.values.ravel()
                  if isinstance(v, str) and v.startswith(("Cell3_", "Cell4_", "Cell5_"))]
        assert not ghosts, ghosts
        print("C2: re-tracking 2 frames into a dirty output dir yields %d track(s), no phantoms"
              % len(shrunk))

        # --- D: weights resolved by NAME, and an all-zero resolution refused.
        four = ["Maximum Horizontal Length", "Maximum Vertical Length",
                "Center X Coordinate", "Center Y Coordinate"]
        assert resolve_weights(four) == [0, 0, 1, 1], resolve_weights(four)
        narrowed = build_cost_matrix(frame1[four], frame2[four], feature_columns=four)
        assert np.nansum(np.abs(narrowed.to_numpy())) > 0, narrowed
        try:
            build_cost_matrix(frame1, frame2, feature_columns=["Cell Area", "Cell Perimeter"])
        except ValueError:
            pass
        else:
            raise AssertionError("an all-zero weight resolution must raise, not return a permutation")
        print("D: custom feature_columns resolve by name (%s); an all-zero resolution raises."
              % resolve_weights(four))
        print("  (before: weights were indexed positionally, so those four columns gave a")
        print("   zero cost matrix and linear_sum_assignment returned an arbitrary permutation.)")

        # --- A2 (R15): re-track must not touch identities outside its range.
        # `close_track_gaps` used to be handed the WHOLE table, so a re-track
        # bounded to [1,2] merged a track spanning frames 8-9 into one spanning
        # 5-6 -- exactly the manual corrections this function exists to preserve.
        far = pd.DataFrame([_cell(f"Cell{f}_1", 10.0, 10.0) for f in (1, 2, 3)] +
                           [_cell(f"Cell{f}_9", 300.0, 300.0) for f in (5, 6, 8, 9)])
        far_tracks = pd.DataFrame({
            "Index": [1, 2, 3],
            **{f"Frame{f}": [f"Cell{f}_1" if f <= 3 else None,
                             f"Cell{f}_9" if f in (5, 6) else None,
                             f"Cell{f}_9" if f in (8, 9) else None] for f in range(1, 10)},
        })
        before_far = far_tracks.copy(deep=True)
        out = retrack_frame_range(far, far_tracks.copy(deep=True), 1, 2, max_frame_gap=1)
        after = out["tracking_data"]
        untouched = all(
            sorted(str(v) for v in before_far.loc[before_far["Index"] == t].values.ravel()
                   if isinstance(v, str))
            == sorted(str(v) for v in after.loc[after["Index"] == t].values.ravel()
                      if isinstance(v, str))
            for t in (2, 3) if t in set(after["Index"])
        )
        assert untouched, after.to_string()
        assert out.get("gap_joins_rejected_out_of_range", 0) >= 0
        print("\nA2 (R15): a re-track bounded to [1,2] with gap closing on leaves")
        print("  frames 5-9 untouched (%d out-of-range join(s) rejected)."
              % out.get("gap_joins_rejected_out_of_range", 0))

        # A2c: a mixed-dtype index (one stray non-string label, as a merged or
        # partially re-indexed frame can produce) used to break the `.str`
        # accessor here while `track_cells` tolerated it.
        mixed = far.copy()
        mixed.index = pd.Index(list(far.index[:-1]) + [0], dtype=object)
        try:
            retrack_frame_range(mixed, far_tracks.copy(deep=True), 1, 2)
            print("  A2c: a mixed-dtype cells_info index is normalised, not an AttributeError.")
        except (AttributeError, TypeError, ValueError) as exc:
            raise AssertionError("index normalisation missing in retrack_frame_range: %s" % exc)

        # --- A5 (R15): re-running into the same output directory must return
        # the same track ids. Auto-detection read the sibling population's
        # Excel file off disk, which on a re-run is the PREVIOUS run's, so
        # three consecutive calls gave [1,2,3,4], [5,6,7,8], [9,10,11,12] and
        # every manual correction keyed by `Index` misaligned.
        edge_rows = []
        for f in (1, 2):
            for i, (x, edge) in enumerate([(10.0, -1.0), (200.0, -1.0),
                                           (400.0, 1.0), (600.0, 1.0)], start=1):
                row = {col: 1.0 for col in DEFAULT_TRACKING_FEATURES}
                row.update({"Cell Index": f"Cell{f}_{i}", "leading_edge": edge,
                            "Center X Coordinate": x + 2 * f, "Center Y Coordinate": 10.0,
                            "Circumscribed Circle Radius": 10.0,
                            "Maximum Horizontal Length": 10.0, "Maximum Vertical Length": 10.0})
                edge_rows.append(row)
        rerun_dir = tempfile.mkdtemp()
        seen = []
        for _ in range(3):
            table = track_all_populations(pd.DataFrame(edge_rows), rerun_dir)
            seen.append(sorted(int(v) for v in table.iloc[:, 0]))
        assert seen[0] == seen[1] == seen[2] == [1, 2, 3, 4], seen
        print("  A5: three runs into one output directory give the same ids %s" % (seen[0],))
        print("     (before: %s, then [5, 6, 7, 8], then [9, 10, 11, 12])." % (seen[0],))

        # --- A5 (R15): an A->B->C chain of gaps closes WHOLE. The old loop
        # refused any link touching an already-absorbed track, so it kept
        # exactly ONE of the chain's two links and returned two tracks where
        # one is correct -- measured here on the pre-change code, under all
        # three row orders below. (The plan called the survivor "row-order
        # dependent"; it is not -- `tail_ids`/`head_ids` are sorted by track
        # id, so the survivor is deterministic. It is the LOWEST-id link that
        # wins, not the best one, which is arbitrary rather than unstable.)
        chain_info = pd.DataFrame(
            [_cell("Cell1_1", 10.0, 10.0), _cell("Cell2_1", 12.0, 10.0),
             _cell("Cell4_1", 16.0, 10.0), _cell("Cell5_1", 18.0, 10.0),
             _cell("Cell7_1", 22.0, 10.0), _cell("Cell8_1", 24.0, 10.0)]
        )
        chain_rows = [
            {"Index": 1, **{f"Frame{f}": (f"Cell{f}_1" if f in (1, 2) else None) for f in range(1, 9)}},
            {"Index": 2, **{f"Frame{f}": (f"Cell{f}_1" if f in (4, 5) else None) for f in range(1, 9)}},
            {"Index": 3, **{f"Frame{f}": (f"Cell{f}_1" if f in (7, 8) else None) for f in range(1, 9)}},
        ]
        survivors = set()
        for order in ([0, 1, 2], [2, 1, 0], [1, 2, 0]):
            table = pd.DataFrame([chain_rows[k] for k in order]).reset_index(drop=True)
            closed, chain_joins = close_track_gaps(
                table, chain_info, max_frame_gap=1, max_centroid_distance=20.0)
            assert len(closed) == 1, closed.to_string()
            held = sorted(int(c[len("Frame"):]) for c in closed.columns
                          if c.startswith("Frame") and pd.notna(closed.iat[0, closed.columns.get_loc(c)]))
            assert held == [1, 2, 4, 5, 7, 8], held
            assert len(chain_joins) == 2, chain_joins
            survivors.add(int(closed.iat[0, 0]))
        assert survivors == {1}, survivors
        print("\nA5: an A->B->C gap chain closes to ONE track (frames 1,2,4,5,7,8) under")
        print("  every row order, and the surviving id is always the earliest segment's.")

        # A2a: re-track refuses a mixed-sign leading_edge, as track_cells does.
        mixed_edge = far.copy()
        mixed_edge["leading_edge"] = [1.0] * 3 + [-1.0] * 4
        try:
            retrack_frame_range(mixed_edge, far_tracks.copy(deep=True), 1, 2)
            raise AssertionError("retrack_frame_range must refuse a mixed-sign leading_edge")
        except ValueError as exc:
            assert "leading_edge" in str(exc), exc
        print("  A2a: a mixed-sign leading_edge raises here too, not one mixed cost matrix.")

        # B2 (R17): a folder that exists but holds no arrays is FALSE, not
        # "configured". `__bool__` used to be `isdir(...)`, which passed on a
        # TIFF dataset's own masks/ folder (the arrays are in npy_converted/
        # beneath it) and then found nothing -- so a cost term weighted 1.0
        # contributed nothing while the run reported success. It has cost twice.
        empty_folder = tempfile.mkdtemp()
        assert not _FrameLabelLoader(empty_folder), \
            "a folder with no arrays must be falsy: 'path exists' is not 'inputs found'"
        assert _FrameLabelLoader(empty_folder).get(1) is None
        assert not _FrameLabelLoader(None), "no folder at all must be falsy"

        # The exact TIFF-dataset shape: masks/ holds only .tif, and the arrays
        # sit in masks/npy_converted/ beneath it.
        tiff_root = tempfile.mkdtemp()
        tiff_masks = os.path.join(tiff_root, "masks")
        converted = os.path.join(tiff_masks, "npy_converted")
        os.makedirs(converted)
        open(os.path.join(tiff_masks, "frame001.tif"), "wb").close()
        np.save(os.path.join(converted, "frame001_seg.npy"),
                {"masks": np.ones((8, 8), dtype=int)}, allow_pickle=True)
        assert not _FrameLabelLoader(tiff_masks), \
            "the TIFF dataset's mask folder holds no arrays and must be falsy"
        assert _FrameLabelLoader(converted), "npy_converted/ holds the arrays and must be truthy"
        print("  B2: a folder that exists but holds no arrays is falsy -- including a TIFF")
        print("     dataset's own masks/, whose arrays live in npy_converted/ beneath it.")

        # A4c: a frame with only `outlines` yields None, not 1-px boundaries.
        # As a fallback those gave IoU ~0 for every pair, so the overlap term
        # became a near-constant penalty that ranked exactly like centroids
        # alone -- while reporting that the mask had been used.
        holder = tempfile.mkdtemp()
        ring = np.zeros((40, 40), dtype=int)
        ring[10:30, 10] = 1; ring[10:30, 29] = 1; ring[10, 10:30] = 1; ring[29, 10:30] = 1
        np.save(os.path.join(holder, "f001_seg.npy"), {"outlines": ring}, allow_pickle=True)
        # ...but an older converter's FILLED array under the same key is a real
        # label image and must still be accepted (see `_load_frame`'s docstring).
        filled = np.zeros((40, 40), dtype=int)
        filled[10:30, 10:30] = 1
        np.save(os.path.join(holder, "f002_seg.npy"), {"outlines": filled}, allow_pickle=True)
        loader = _FrameLabelLoader(holder)
        assert loader.get(1) is None, "a 1-px boundary array must not pass as a label image"
        assert loader.get(2) is not None, "a filled array under 'outlines' must still be used"
        # --- B (R23): IoU and coverage fail in OPPOSITE directions.
        # Two configurations, not one -- B and N cannot both sit over the same
        # pixels of A in one label image, so each pair is scored on its own
        # frame. These numbers are algebra, pinned here so the claim in
        # `mask_coverage_fractions` and `docs/census.md` §8.8 cannot drift away
        # from the code; they are NOT a measurement of anything.
        anchor_a = np.zeros((60, 60), dtype=int)
        anchor_a[10:30, 10:30] = 1                       # |A| = 400

        divided = np.zeros((60, 60), dtype=int)
        divided[10:30, 10:20] = 5                        # daughter, 200 px, inside A
        neighbour = np.zeros((60, 60), dtype=int)
        neighbour[10:30, 26:46] = 6                      # 400 px, overlaps A by 80

        iou_d = mask_overlap_fractions(anchor_a, divided)[(1, 5)]
        cov_d = mask_coverage_fractions(anchor_a, divided)[(1, 5)]
        iou_n = mask_overlap_fractions(anchor_a, neighbour)[(1, 6)]
        cov_n = mask_coverage_fractions(anchor_a, neighbour)[(1, 6)]
        assert abs(iou_d - 0.50) < 1e-9 and abs(iou_n - 1 / 9) < 1e-9, (iou_d, iou_n)
        assert abs(cov_d - 0.50) < 1e-9 and abs(cov_n - 0.20) < 1e-9, (cov_d, cov_n)
        assert (iou_d / iou_n) > (cov_d / cov_n), "IoU must separate a DAUGHTER better"

        small = np.zeros((60, 60), dtype=int)
        small[10:20, 10:20] = 1                          # |A| = 100
        grown = np.zeros((60, 60), dtype=int)
        grown[8:23, 8:28] = 7                            # 300 px, contains A
        side = np.zeros((60, 60), dtype=int)
        side[10:20, 16:26] = 8                           # 100 px, overlaps A by 40

        iou_g = mask_overlap_fractions(small, grown)[(1, 7)]
        cov_g = mask_coverage_fractions(small, grown)[(1, 7)]
        iou_s = mask_overlap_fractions(small, side)[(1, 8)]
        cov_s = mask_coverage_fractions(small, side)[(1, 8)]
        assert abs(iou_g - 1 / 3) < 1e-9 and abs(iou_s - 0.25) < 1e-9, (iou_g, iou_s)
        assert abs(cov_g - 1.00) < 1e-9 and abs(cov_s - 0.40) < 1e-9, (cov_g, cov_s)
        assert (cov_g / cov_s) > (iou_g / iou_s), "coverage must separate a GROWN successor better"
        print("  B (R23): a daughter -- IoU %.2f vs %.2f (%.1fx), coverage %.2f vs %.2f (%.1fx);"
              % (iou_d, iou_n, iou_d / iou_n, cov_d, cov_n, cov_d / cov_n))
        print("     a successor 3x the anchor -- IoU %.2f vs %.2f (%.1fx), coverage %.2f vs %.2f (%.1fx)."
              % (iou_g, iou_s, iou_g / iou_s, cov_g, cov_s, cov_g / cov_s))
        print("     Opposite directions, which is the argument for offering BOTH.")
        print("     Algebra, not a measurement -- `chosen_reason` is the measurement.")

        print("  A4c: a 1-px 'outlines' boundary loads as None (IoU over rings is ~0 for every")
        print("     pair); an older converter's FILLED array under the same key still works.")

        # --- A1 (R15): the four validations each refuse the bad value.
        for label, call in (
            ("drift_mode typo", lambda: _validate_drift_mode("Mask")),
            ("no_link_ratio<=1", lambda: _no_match_sentinel(plain, 1.0, no_link_ratio=1.0)),
            ("overlap_power=0", lambda: build_cost_matrix(frame1, frame2, overlap_power=0.0)),
        ):
            try:
                call()
            except ValueError:
                continue
            raise AssertionError("%s must raise" % label)
        print("  A1: drift_mode typos, no_link_ratio<=1.0 and overlap_power=0 all raise.")

        print("All cell_tracking self-tests passed.")
        raise SystemExit(0)

    os.makedirs(args.output_dir, exist_ok=True)

    cells_info_df = pd.read_excel(args.cells_info)

    def _print_progress(message: str, percent: int) -> None:
        print(f"[{percent:3d}%] {message}")

    merged = track_all_populations(cells_info_df, args.output_dir, progress_cb=_print_progress)
    print(f"\nMerged tracking shape: {merged.shape}")
    print(f"Columns: {list(merged.columns)}")
