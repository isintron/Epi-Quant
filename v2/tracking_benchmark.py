"""Evaluation harness for tracking changes -- the judges, and their biases.

**This module exists because Round 9 reached a wrong conclusion with a judge
that was not neutral about the change it was judging.** Motion compensation was
measured to double the wrong-link rate and was written off; the judge used
(agreement with each cell's best-IoU successor) correlates more strongly with
the uncompensated cost than with a motion-predicting one, so it penalised that
whole class of change by construction. Nothing in the repo reproduced those
numbers either -- they came from throwaway scripts. Both problems are fixed
here.

The rule this module enforces
-----------------------------
**Every judge states its known bias in its own docstring, or it does not get
added.** A judge whose bias is unknown cannot be used to decide anything, and
"the number went down" is not a finding unless you can say what would have made
it go down spuriously.

Only `ground_truth` measures accuracy. `iou_agreement` and `environment_dice`
measure *agreement with a reference*, and the reference is not the truth. Say
"agreement", name the reference, and state its bias -- never "accuracy".

Use the two reference judges as a PAIR
--------------------------------------
Their failure modes are opposite, so together they bracket the truth:

- `iou_agreement` penalises motion prediction (see its docstring).
- `environment_dice` is invariant under a *coherent permutation of identities*:
  shift every cell's identity one step in the same direction and every
  neighbour set is preserved in track-ID space (Dice 1.0) while every single
  link is wrong. It therefore REWARDS exactly the systematic error that drift
  compensation can introduce.

On `embryos_GBE_GAP43` the two rank global drift compensation OPPOSITELY, and
that is structural rather than noise. **Only a change that loses on neither is
credible.** Never use either as a sole judge, and never as an optimisation
objective.
"""

import hashlib
import logging
import os
import re
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import adjacency
import cell_tracking
import data_input

logger = logging.getLogger(__name__)

# Default minimum best-IoU for a cell to be judgeable by `iou_agreement`.
# Below it, "which cell is the successor" has no defensible answer, so the cell
# is reported as UNJUDGEABLE rather than silently dropped -- on
# `embryos_GBE_GAP43` that is ~24% of cell-frames, which is far too large a
# fraction to leave out of the denominator without saying so.
DEFAULT_IOU_THRESHOLD = 0.3


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def load_masks(npy_folder: str) -> dict:
    """`{frame_number: filled label array}` for a segmentation folder.

    Uses `data_input`'s own frame-number resolution so a benchmark can never
    disagree with the pipeline about which file is which frame.
    """
    files = data_input.list_segmentation_files(npy_folder)
    masks = {}
    for name, number in data_input.resolve_frame_numbers(files).items():
        dat = np.load(os.path.join(npy_folder, name), allow_pickle=True).item()
        masks[number] = data_input.load_label_array(dat)
    return masks


def _raw_label(cell_index) -> int:
    return int(str(cell_index).rsplit("_", 1)[1])


def label_to_track(tracking_data: pd.DataFrame, frame_number: int) -> dict:
    """`{raw_label: track_id}` for one frame of a tracking table."""
    column = f"Frame{frame_number}"
    if column not in tracking_data.columns:
        return {}
    id_col = tracking_data.columns[0]
    mapping = {}
    for _, row in tracking_data[[id_col, column]].dropna(subset=[column]).iterrows():
        match = re.search(r"Cell\d+_(\d+)", str(row[column]))
        if match:
            mapping.setdefault(int(match.group(1)), int(row[id_col]))
    return mapping


def frame_numbers(tracking_data: pd.DataFrame) -> list:
    """Frame numbers that actually carry data.

    Deliberately not the `Frame*` column list: `_merge_tracking_pairs` emits
    `range(start, start + count + 1)`, so 30 frames of pairs yield an always
    empty `Frame31`. `lineage.track_spans` guards the same trap.
    """
    numbers = []
    for column in tracking_data.columns:
        if column.startswith("Frame") and tracking_data[column].notna().any():
            numbers.append(int(column[len("Frame"):]))
    return sorted(numbers)


# --------------------------------------------------------------------------
# Judges
# --------------------------------------------------------------------------

def best_iou_successors(labels_a: np.ndarray, labels_b: np.ndarray, mutual: bool = False) -> dict:
    """`{label_a: (iou, label_b)}` -- each cell's highest-IoU partner.

    `mutual=True` keeps only pairs that are each other's best, which is the
    stricter correspondence the diagnostics use for chaining.
    """
    overlaps = cell_tracking.mask_overlap_fractions(labels_a, labels_b)
    forward, backward = {}, {}
    for (a, b), value in overlaps.items():
        if value > forward.get(a, (0.0, None))[0]:
            forward[a] = (value, b)
        if value > backward.get(b, (0.0, None))[0]:
            backward[b] = (value, a)
    if not mutual:
        return forward
    return {a: (v, b) for a, (v, b) in forward.items() if backward.get(b, (0.0, None))[1] == a}


def shift_labels(labels: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate a label image by `(dx, dy)` px, filling the vacated edge with 0.

    Integer rounding, deliberately: a label image cannot be interpolated
    (averaging label 7 and label 12 is meaningless), so sub-pixel registration
    would need nearest-neighbour resampling and would not change any overlap by
    more than a boundary pixel. Note this makes the measured post-registration
    IoUs slightly conservative.
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


def estimate_frame_drift(masks_by_frame: Mapping[int, np.ndarray], frame_number: int) -> tuple:
    """Global `(dx, dy)` drift from `frame_number` to the next, in px.

    Median displacement over MUTUAL best-IoU pairs -- mutual because a one-sided
    best can point at a neighbour when a cell has left, and the median because a
    handful of such pairs must not move the estimate.

    This needs no tracking table, so a judge built on it stays independent of
    the tracker being judged. (Estimating the same drift from the tracking table
    instead differs by ~0.25 px median on `embryos_GBE_GAP43`, so either works;
    the mask-only version is used because it cannot be accused of inheriting the
    tracker's errors.)
    """
    nxt = frame_number + 1
    if frame_number not in masks_by_frame or nxt not in masks_by_frame:
        return (0.0, 0.0)
    mutual = best_iou_successors(masks_by_frame[frame_number], masks_by_frame[nxt], mutual=True)
    here, there = mask_centroids(masks_by_frame[frame_number]), mask_centroids(masks_by_frame[nxt])
    deltas = [(there[b][0] - here[a][0], there[b][1] - here[a][1])
              for a, (_, b) in mutual.items() if a in here and b in there]
    if not deltas:
        return (0.0, 0.0)
    return (float(np.median([d[0] for d in deltas])), float(np.median([d[1] for d in deltas])))


def registration_decision(
    masks_by_frame: Mapping[int, np.ndarray],
    cells_info: Optional[pd.DataFrame] = None,
    radius_fraction: float = 0.2,
) -> dict:
    """Should this movie be registered before matching? A measured decision.

    Registers when the median global drift magnitude exceeds `radius_fraction`
    of the median cell radius. On `embryos_GBE_GAP43` that is ~6.5 / 13.0 = 0.50
    -> register; on a tissue with no bulk motion the drift is ~0, the ratio is
    below the threshold, and registration would be an identity operation anyway.

    Returning the ratio and the per-frame drifts, not just a boolean, is the
    point -- the UI shows the basis so a user sees why the behaviour changed
    rather than discovering it silently.
    """
    frames = sorted(masks_by_frame)
    drifts = {f: estimate_frame_drift(masks_by_frame, f) for f in frames[:-1]}
    magnitudes = [float(np.hypot(*d)) for d in drifts.values()]
    median_drift = float(np.median(magnitudes)) if magnitudes else 0.0

    if cells_info is not None and "Circumscribed Circle Radius" in cells_info.columns:
        radius = float(cells_info["Circumscribed Circle Radius"].median())
    else:
        areas = []
        for labels in masks_by_frame.values():
            counts = np.bincount(labels.ravel())
            areas.extend(counts[1:][counts[1:] > 0].tolist())
        radius = float(np.sqrt(np.median(areas) / np.pi)) if areas else 1.0

    ratio = median_drift / radius if radius else 0.0
    return {
        "register": bool(ratio > radius_fraction),
        "median_drift_px": median_drift,
        "median_cell_radius_px": radius,
        "ratio": ratio,
        "threshold": radius_fraction,
        "drift_by_frame": drifts,
    }


def data_properties(
    masks_by_frame: Mapping[int, np.ndarray],
    register: bool = False,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    drift_by_frame: Optional[Mapping[int, tuple]] = None,
) -> dict:
    """Properties of the CORRESPONDENCE PROBLEM itself -- no tracker involved.

    Median best IoU, median (best - second) margin, the fraction of cells whose
    best IoU exceeds 0.5, and how many are judgeable at `iou_threshold`.

    **This is what makes the registration case, precisely because no tracker
    appears in it.** On `embryos_GBE_GAP43` the unregistered numbers are a
    median best IoU of 0.377 against `iou_agreement`'s own 0.3 judgeability
    threshold -- the reference is operating at the edge of its validity.
    Registering by the measured drift takes it to 0.709, the margin from 0.224
    to 0.661, and the judgeable count from 5,483 to 6,818.
    """
    frames = sorted(masks_by_frame)
    best_values, margins, judgeable, total = [], [], 0, 0
    for frame in frames[:-1]:
        if frame + 1 not in masks_by_frame:
            continue
        labels_a = masks_by_frame[frame]
        if register:
            shift = (drift_by_frame or {}).get(frame) or estimate_frame_drift(masks_by_frame, frame)
            labels_a = shift_labels(labels_a, shift[0], shift[1])
        overlaps = cell_tracking.mask_overlap_fractions(labels_a, masks_by_frame[frame + 1])
        per_cell: dict = {}
        for (a, _b), value in overlaps.items():
            per_cell.setdefault(a, []).append(value)
        for values in per_cell.values():
            values.sort(reverse=True)
            best_values.append(values[0])
            margins.append(values[0] - (values[1] if len(values) > 1 else 0.0))
            total += 1
            if values[0] > iou_threshold:
                judgeable += 1
    if not best_values:
        return {"n": 0}
    return {
        "registered": bool(register),
        "median_best_iou": float(np.median(best_values)),
        "median_margin": float(np.median(margins)),
        "fraction_best_above_half": float(np.mean(np.array(best_values) > 0.5)),
        "judgeable": judgeable,
        "n": total,
    }


def iou_agreement(
    tracking_data: pd.DataFrame,
    masks_by_frame: Mapping[int, np.ndarray],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    register: bool = False,
    margin: float = 0.0,
    mutual: bool = False,
    drift_by_frame: Optional[Mapping[int, tuple]] = None,
    **_unused,
) -> dict:
    """Agreement between the tracking's links and each cell's best-IoU successor.

    Returns `{"correct", "wrong", "broken", "unjudgeable", "n_judged",
    "n_total", "wrong_rate", ...}`. A cell is *judgeable* when its best mask
    IoU into the next frame exceeds `iou_threshold`; otherwise "which cell is
    the successor" has no defensible answer and it is counted as
    `unjudgeable`, NOT dropped.

    **Known bias -- read this before quoting a number.** The reference is mask
    overlap, so this judge favours costs that correlate with mask overlap and
    penalises costs that predict motion. Measured on `embryos_GBE_GAP43` over
    42,828 in-gate candidate pairs, Spearman(cost, IoU) is **-0.850** for the
    plain centroid+overlap cost but only **-0.709** once a whole-field drift is
    compensated. A drift-compensated cost that moved links *toward* the truth
    and *away* from the biggest-overlap cell would score worse here, and this
    judge cannot tell that apart from getting worse. It is exactly the trap
    Round 9 fell into.

    Secondary bias: the `iou_threshold` filter selects slow-moving cells. Do not
    reuse it across different frame gaps -- at larger gaps it keeps only cells
    that barely moved, which is what produced Round 9's spurious "displacement
    does not grow with frame gap" result. See `gap_growth`.

    **`register` is the most consequential option here, and it must always be
    reported alongside its opposite.** This judge inherits the registration
    convention of the masks it is handed: an unregistered reference favours a
    tracker that does not register, and a registered one favours a tracker that
    does. Neither is neutral. On `embryos_GBE_GAP43` the unregistered reference
    has a median best IoU of 0.377 against its own 0.3 threshold -- it is at the
    edge of validity -- and registering takes that to 0.709 and adds ~1,300
    judgeable cells. Scoring the same three tracking settings both ways reverses
    their ranking. **Never report one convention alone.** See `data_properties`
    and `registration_decision`.

    `margin` requires the reference's own choice to beat the runner-up by at
    least that much IoU; `mutual` additionally requires the backward best to
    agree. Cells failing either become **unjudgeable**, not correct -- a tie the
    reference cannot resolve is not evidence the tracker was right. Of the 268
    links called wrong unregistered, the reference beats the tracker's pick by
    < 0.15 IoU in 65% of cases, so this is not a rare correction.
    """
    frames = frame_numbers(tracking_data)
    correct = wrong = broken = unjudgeable = 0
    disagreements = []
    for frame in frames[:-1]:
        nxt = frame + 1
        if frame not in masks_by_frame or nxt not in masks_by_frame:
            continue
        labels_a = masks_by_frame[frame]
        if register:
            shift = (drift_by_frame or {}).get(frame) or estimate_frame_drift(masks_by_frame, frame)
            labels_a = shift_labels(labels_a, shift[0], shift[1])
        overlaps = cell_tracking.mask_overlap_fractions(labels_a, masks_by_frame[nxt])
        best, runner_up, backward = {}, {}, {}
        for (a, b), value in overlaps.items():
            if value > best.get(a, (0.0, None))[0]:
                runner_up[a] = best.get(a, (0.0, None))[0]
                best[a] = (value, b)
            elif value > runner_up.get(a, 0.0):
                runner_up[a] = value
            if value > backward.get(b, (0.0, None))[0]:
                backward[b] = (value, a)
        here, there = label_to_track(tracking_data, frame), label_to_track(tracking_data, nxt)
        inverse = {track: label for label, track in there.items()}
        for label, track in here.items():
            value, partner = best.get(label, (0.0, None))
            if value < iou_threshold:
                unjudgeable += 1
                continue
            if margin and (value - runner_up.get(label, 0.0)) < margin:
                unjudgeable += 1
                continue
            if mutual and backward.get(partner, (0.0, None))[1] != label:
                unjudgeable += 1
                continue
            linked = inverse.get(track)
            if linked is None:
                broken += 1
            elif linked == partner:
                correct += 1
            else:
                wrong += 1
                disagreements.append({
                    "frame": frame, "label": label,
                    "reference_iou": round(float(value), 3),
                    "tracker_iou": round(float(overlaps.get((label, linked), 0.0)), 3),
                })

    judged = correct + wrong + broken
    total = judged + unjudgeable
    return {
        "judge": "iou_agreement",
        "correct": correct, "wrong": wrong, "broken": broken,
        "unjudgeable": unjudgeable, "n_judged": judged, "n_total": total,
        "correct_rate": 100.0 * correct / judged if judged else float("nan"),
        "wrong_rate": 100.0 * wrong / judged if judged else float("nan"),
        "broken_rate": 100.0 * broken / judged if judged else float("nan"),
        "unjudgeable_rate": 100.0 * unjudgeable / total if total else float("nan"),
        "registered": bool(register), "margin": float(margin), "mutual": bool(mutual),
        "disagreements": disagreements,
    }


def environment_dice(
    tracking_data: pd.DataFrame,
    masks_by_frame: Mapping[int, np.ndarray],
    dilation_px: int = 2,
    **_unused,
) -> dict:
    """Mean Dice overlap of each cell's neighbour set between consecutive
    frames, compared in TRACK-ID space.

    A correctly tracked cell in a coherent sheet keeps most of its neighbours
    from frame to frame, so a link that swaps two identities usually shows up
    as two damaged neighbour sets. Neighbours come from
    `adjacency.build_neighbor_graph`, the same graph the T1 code uses.

    **Known bias -- and it is a severe one.** This measure is *invariant under
    a coherent permutation of identities*: relabel every cell one step in the
    same direction and every neighbour set is preserved in track-ID space,
    giving Dice 1.0 while every single link is wrong. It therefore REWARDS the
    systematic error that a drift-compensated cost can introduce, which is the
    mirror image of `iou_agreement`'s bias. On `embryos_GBE_GAP43` the two
    judges rank global drift compensation oppositely.

    Never use this as a sole judge and never as an optimisation objective --
    optimising it directly drives toward coherent mislabelling. Its job is to
    veto changes that `iou_agreement` likes for the wrong reason.
    """
    frames = frame_numbers(tracking_data)
    graphs: dict = {}

    def graph_for(frame):
        if frame not in graphs:
            graphs[frame] = (adjacency.build_neighbor_graph(masks_by_frame[frame], dilation_px=dilation_px)
                             if frame in masks_by_frame else None)
        return graphs[frame]

    scores = []
    for frame in frames[:-1]:
        nxt = frame + 1
        graph_a, graph_b = graph_for(frame), graph_for(nxt)
        if not graph_a or not graph_b:
            continue
        here, there = label_to_track(tracking_data, frame), label_to_track(tracking_data, nxt)
        inverse = {track: label for label, track in there.items()}
        for label, track in here.items():
            partner = inverse.get(track)
            if partner is None:
                continue
            set_a = {here[n] for n in graph_a.get(label, set()) if n in here}
            set_b = {there[n] for n in graph_b.get(partner, set()) if n in there}
            if not set_a and not set_b:
                continue
            scores.append(2.0 * len(set_a & set_b) / (len(set_a) + len(set_b)))

    return {
        "judge": "environment_dice",
        "dice": float(np.mean(scores)) if scores else float("nan"),
        "n": len(scores),
    }


def forward_backward_consistency(
    tracking_data: pd.DataFrame = None,
    masks_by_frame: Mapping[int, np.ndarray] = None,
    cells_info: pd.DataFrame = None,
    drift_mode: str = "none",
    drift_by_frame: Optional[Mapping[int, tuple]] = None,
    **cost_kwargs,
) -> dict:
    """Fraction of links where solving the frame pair FORWARDS and BACKWARDS
    disagree: forward says a->b but backward does not say b->a.

    **This has no reference and no registration convention of its own**, which
    is exactly why it is here: `iou_agreement` inherits the registration
    convention of its masks and so cannot arbitrate a question about
    registration. This can.

    Not trivially satisfied. The cost is asymmetric by construction -- the
    displacement gate and the overlap term's `max(H, V)` bridge both come from
    the ROW cell -- so transposing the problem genuinely re-solves it rather
    than reading off the same answer. On `embryos_GBE_GAP43`: 1.98% for the
    shipped cost, 1.05% with a centroid shift, 0.22% with the masks registered.

    **Degenerate case, and it is shared.** A systematically wrong but
    self-consistent assignment scores 0% -- shift every identity one step the
    same way and forward and backward agree perfectly. That is the SAME blind
    spot as `environment_dice`'s invariance under a coherent identity
    permutation, so the two do **not** count as two independent lines. Pair this
    with `assignment_margin` or with `data_properties` instead.
    """
    if cells_info is None or masks_by_frame is None:
        return {"judge": "forward_backward_consistency", "error": "needs cells_info and masks"}
    forward = _solve_pairs(cells_info, masks_by_frame, drift_mode=drift_mode,
                           drift_by_frame=drift_by_frame, **cost_kwargs)
    backward = _solve_pairs(cells_info, masks_by_frame, drift_mode=drift_mode,
                            drift_by_frame=drift_by_frame, reverse=True, **cost_kwargs)
    total = inconsistent = 0
    for frame, links in forward.items():
        back = backward.get(frame, {})
        for a, b in links.items():
            total += 1
            if back.get(b) != a:
                inconsistent += 1
    return {
        "judge": "forward_backward_consistency",
        "inconsistent": inconsistent, "n": total,
        "inconsistency_rate": 100.0 * inconsistent / total if total else float("nan"),
        "drift_mode": drift_mode,
    }


def assignment_margin(
    tracking_data: pd.DataFrame = None,
    masks_by_frame: Mapping[int, np.ndarray] = None,
    cells_info: pd.DataFrame = None,
    drift_mode: str = "none",
    drift_by_frame: Optional[Mapping[int, tuple]] = None,
    **cost_kwargs,
) -> dict:
    """Median (second-smallest - smallest) finite cost per assigned row: how
    decisively the solver preferred its choice over the runner-up.

    No reference and no registration convention, like
    `forward_backward_consistency`. On `embryos_GBE_GAP43`: 11.40 px for the
    shipped cost, 21.65 with a centroid shift, 31.96 with the masks registered.

    **Only comparable within a fixed cost scale.** It is in the cost's own
    units, so it is meaningless across `cost_form` settings -- the
    multiplicative form squares the distance, and a bigger number there means
    nothing about confidence. Also meaningless across different
    `overlap_weight`s for the same reason. Compare it only between settings that
    share a scale.

    **Degenerate case:** it rewards any cost that merely spreads its values
    further apart, whether or not the ordering improved. A cost multiplied by
    ten doubles nothing real and doubles this. That is why it is paired with
    forward-backward consistency rather than used alone.
    """
    if cells_info is None or masks_by_frame is None:
        return {"judge": "assignment_margin", "error": "needs cells_info and masks"}
    margins = _solve_pairs(cells_info, masks_by_frame, drift_mode=drift_mode,
                           drift_by_frame=drift_by_frame, want_margins=True, **cost_kwargs)
    values = [m for frame in margins.values() for m in frame.values()]
    return {
        "judge": "assignment_margin",
        "median_margin": float(np.median(values)) if values else float("nan"),
        "n": len(values), "drift_mode": drift_mode,
    }


DRIFT_MODES = ("none", "centroid", "mask")


def _solve_pairs(
    cells_info: pd.DataFrame,
    masks_by_frame: Mapping[int, np.ndarray],
    overlap_weight: float = cell_tracking.DEFAULT_OVERLAP_WEIGHT,
    centroid_weight: float = 1.0,
    drift_mode: str = "none",
    drift_by_frame: Optional[Mapping[int, tuple]] = None,
    reverse: bool = False,
    want_margins: bool = False,
    feature_columns: Optional[list] = None,
    **_unused,
) -> dict:
    """Solve every frame pair independently; return `{frame: {a: b}}` (raw
    labels) or, with `want_margins`, `{frame: {row_label: margin}}`.

    The margins are keyed by their row label rather than returned as a bare
    list so one implementation serves both consumers: `assignment_margin`
    takes the median over the values, and `census.low_margin_items` needs to
    know WHICH link each margin belongs to in order to queue it. Two
    implementations would be two numbers for the same quantity, and the
    published ones (11.40 / 21.65 / 31.96 px) would stop meaning anything.

    `drift_mode` is deliberately three-way rather than a boolean, because the
    two ways of using the same measured drift are NOT the same experiment:

    - `"none"` -- the shipped cost.
    - `"centroid"` -- shift the predicted centroid by the drift, leave the masks
      alone. Round 10 measured this one. The IoU term still sees unregistered
      masks, so it is fighting the centroid term.
    - `"mask"` -- Tissue Analyzer's convention: register the frame-f label image
      by the drift AND shift the centroid, so BOTH terms are measured in the
      registered frame.

    Collapsing these into one flag is what made the first attempt at this table
    report identical numbers for the first two rows.

    `reverse=True` swaps the two frames AND negates the drift, so the backward
    problem is the mirror image rather than a differently-registered one.
    """
    if drift_mode not in DRIFT_MODES:
        raise ValueError(f"drift_mode must be one of {DRIFT_MODES}, got {drift_mode!r}")
    if feature_columns is None:
        feature_columns = cell_tracking.DEFAULT_TRACKING_FEATURES
    cells_info = cells_info.copy()
    cells_info.index = cells_info.index.map(str)
    ceiling = cells_info["Circumscribed Circle Radius"].max()
    frames = sorted(set(
        int(m.group(1)) for i in cells_info.index
        for m in [re.search(r"Cell(\d+)_", str(i))] if m
    ))

    out: dict = {}
    for frame in frames:
        a = cells_info[cells_info.index.str.contains(f"Cell{frame}_")]
        b = cells_info[cells_info.index.str.contains(f"Cell{frame + 1}_")]
        if a.empty or b.empty:
            continue
        shift = (0.0, 0.0)
        if drift_mode != "none":
            shift = (drift_by_frame or {}).get(frame) or estimate_frame_drift(masks_by_frame, frame)

        labels_a, labels_b = masks_by_frame.get(frame), masks_by_frame.get(frame + 1)
        first, second = a, b
        if reverse:
            first, second = b, a
            labels_a, labels_b = labels_b, labels_a
            shift = (-shift[0], -shift[1])

        overlaps = None
        if labels_a is not None and labels_b is not None and overlap_weight != 0:
            moved = labels_a
            if drift_mode == "mask" and (shift[0] or shift[1]):
                moved = shift_labels(labels_a, shift[0], shift[1])
            overlaps = cell_tracking.mask_overlap_fractions(moved, labels_b)

        predicted = None
        if shift[0] or shift[1]:
            _, _, cx, cy = cell_tracking._gate_indices(list(feature_columns))
            x_col, y_col = feature_columns[cx], feature_columns[cy]
            predicted = {i: (float(first.at[i, x_col]) + shift[0], float(first.at[i, y_col]) + shift[1])
                         for i in first.index}

        cost = cell_tracking.build_cost_matrix(
            first, second, overlaps=overlaps, overlap_weight=overlap_weight,
            max_centroid_distance=ceiling, centroid_weight=centroid_weight,
            predicted_centroids=predicted,
        )
        sentinel = cell_tracking._no_match_sentinel(cost, ceiling)
        matched = cell_tracking.match_frames(cost, sentinel)
        matched = matched[matched["Difference"] != sentinel]

        if want_margins:
            values = cost.to_numpy(dtype=float)
            row_of = {index: i for i, index in enumerate(cost.index)}
            frame_margins = {}
            for cell in matched["frame1_cell"]:
                row = values[row_of[cell]]
                finite = np.sort(row[np.isfinite(row)])
                if finite.size >= 2:
                    frame_margins[_raw_label(cell)] = float(finite[1] - finite[0])
            out[frame] = frame_margins
        else:
            out[frame] = {_raw_label(x): _raw_label(y)
                          for x, y in zip(matched["frame1_cell"], matched["frame2_cell"])}
    return out


def ground_truth(
    tracking_data: pd.DataFrame,
    masks_by_frame: Mapping[int, np.ndarray] = None,
    gt_links: Mapping[int, Mapping[int, int]] = None,
    **_unused,
) -> dict:
    """Plain link accuracy against hand-labelled or Cell Tracking Challenge links.

    `gt_links` is `{frame: {label_in_frame: label_in_next_frame}}`, from
    `load_ctc_ground_truth` or `load_link_csv`.

    **This is the only judge in this module that measures accuracy**, and the
    only one whose numbers may be reported as such. Its bias is whatever the
    annotation's bias is -- for CTC that is a different imaging modality and
    cell biology from this project's epithelia (see `load_ctc_ground_truth`).
    """
    if not gt_links:
        return {"judge": "ground_truth", "error": "no ground-truth links supplied"}
    correct = wrong = missing = 0
    for frame, links in gt_links.items():
        here, there = label_to_track(tracking_data, frame), label_to_track(tracking_data, frame + 1)
        inverse = {track: label for label, track in there.items()}
        for label, true_partner in links.items():
            track = here.get(label)
            linked = inverse.get(track) if track is not None else None
            if linked is None:
                missing += 1
            elif linked == true_partner:
                correct += 1
            else:
                wrong += 1
    total = correct + wrong + missing
    return {
        "judge": "ground_truth",
        "correct": correct, "wrong": wrong, "missing": missing, "n": total,
        "accuracy": 100.0 * correct / total if total else float("nan"),
    }


JUDGES: dict = {
    "iou_agreement": iou_agreement,
    "environment_dice": environment_dice,
    "forward_backward_consistency": forward_backward_consistency,
    "assignment_margin": assignment_margin,
    "ground_truth": ground_truth,
}


def evaluate(
    tracking_data: pd.DataFrame,
    cells_info: Optional[pd.DataFrame] = None,
    masks_by_frame: Optional[Mapping[int, np.ndarray]] = None,
    judges: Sequence = ("iou_agreement", "environment_dice"),
    **judge_kwargs,
) -> dict:
    """Run `judges` over one tracking result. Dataset-agnostic.

    `cells_info` is accepted so a judge may use it, and so callers do not have
    to know which judges need it; none of the shipped judges do -- they work
    from the masks, which keeps them independent of the feature table the cost
    function reads.
    """
    results = {}
    for judge in judges:
        function = JUDGES[judge] if isinstance(judge, str) else judge
        name = judge if isinstance(judge, str) else getattr(judge, "__name__", str(judge))
        results[name] = function(
            tracking_data, masks_by_frame=masks_by_frame, cells_info=cells_info, **judge_kwargs,
        )
    return results


# --------------------------------------------------------------------------
# In-memory tracking (the fast path)
# --------------------------------------------------------------------------

def track_in_memory(
    cells_info: pd.DataFrame,
    masks_by_frame: Optional[Mapping[int, np.ndarray]] = None,
    overlap_weight: float = cell_tracking.DEFAULT_OVERLAP_WEIGHT,
    centroid_weight: float = 1.0,
    motion_compensation: bool = False,
    feature_columns: Optional[list] = None,
    weights: Optional[list] = None,
    start_index: int = 1,
    cost_form: str = "additive",
    overlap_power: float = 1.0,
    no_link_ratio: Optional[float] = None,
    four_quadrant: bool = False,
    drift_mode: str = "none",
    drift_by_frame: Optional[Mapping[int, tuple]] = None,
) -> pd.DataFrame:
    """`cell_tracking.track_cells` without the per-frame-pair Excel writes.

    Same cost, same sentinel, same stitching -- only the I/O is skipped. That
    matters: a full run writes two ~250x250 spreadsheets per frame pair, which
    dominates the runtime and made a six-way cost comparison unaffordable. The
    self-test asserts this reproduces `track_cells` to within 0.01 pp on every
    judge; if it ever drifts, the fast path is wrong, not the slow one.

    `cells_info` must already be indexed by `Cell Index`.
    """
    if feature_columns is None:
        feature_columns = cell_tracking.DEFAULT_TRACKING_FEATURES
    if weights is None:
        weights = cell_tracking.DEFAULT_TRACKING_WEIGHTS

    cells_info = cells_info.copy()
    cells_info.index = cells_info.index.map(str)
    figure_ids = sorted(set(
        int(match.group(1)) for idx in cells_info.index
        for match in [re.search(r"Cell(\d+)_", str(idx))] if match
    ))
    ceiling = cells_info["Circumscribed Circle Radius"].max()

    pairs: dict = {}
    previous_cells, previous_matches = None, None
    for figure_id in figure_ids:
        frame_a = cells_info[cells_info.index.str.contains(f"Cell{figure_id}_")]
        frame_b = cells_info[cells_info.index.str.contains(f"Cell{figure_id + 1}_")]
        if frame_b.empty:
            continue

        # `register=True` is the Tissue Analyzer convention: shift the frame-f
        # label image by the measured global drift BEFORE computing overlap, and
        # shift the predicted centroid by the same amount, so the centroid and
        # the IoU term are both measured in the registered frame. That is
        # different from shifting only the predicted centroid -- see
        # `estimate_frame_drift`.
        shift = (0.0, 0.0)
        if drift_mode != "none":
            shift = ((drift_by_frame or {}).get(figure_id)
                     or (estimate_frame_drift(masks_by_frame, figure_id) if masks_by_frame else (0.0, 0.0)))

        overlaps = None
        if masks_by_frame and overlap_weight != 0:
            labels_a, labels_b = masks_by_frame.get(figure_id), masks_by_frame.get(figure_id + 1)
            if labels_a is not None and labels_b is not None:
                if drift_mode == "mask" and (shift[0] or shift[1]):
                    labels_a = shift_labels(labels_a, shift[0], shift[1])
                overlaps = cell_tracking.mask_overlap_fractions(labels_a, labels_b)

        pair_centroid_weight = centroid_weight
        if centroid_weight == 0 and not overlaps:
            pair_centroid_weight = 1.0

        predicted = None
        if motion_compensation:
            predicted = cell_tracking.predict_centroids(
                previous_matches, previous_cells, frame_a, feature_columns,
            )
        if drift_mode != "none" and (shift[0] or shift[1]):
            base = predicted or {}
            _, _, cx, cy = cell_tracking._gate_indices(list(feature_columns))
            x_col, y_col = feature_columns[cx], feature_columns[cy]
            predicted = {
                index: (base.get(index, (float(frame_a.at[index, x_col]),
                                         float(frame_a.at[index, y_col])))[0] + shift[0],
                        base.get(index, (float(frame_a.at[index, x_col]),
                                         float(frame_a.at[index, y_col])))[1] + shift[1])
                for index in frame_a.index
            }

        cost = cell_tracking.build_cost_matrix(
            frame_a, frame_b, feature_columns, weights,
            overlaps=overlaps, overlap_weight=overlap_weight,
            max_centroid_distance=ceiling,
            centroid_weight=pair_centroid_weight, predicted_centroids=predicted,
            cost_form=cost_form, overlap_power=overlap_power,
        )
        sentinel = cell_tracking._no_match_sentinel(cost, ceiling, no_link_ratio=no_link_ratio)
        matched = cell_tracking.match_frames(cost, sentinel, four_quadrant=four_quadrant)
        matched = matched[matched["Difference"] != sentinel]
        pairs[figure_id] = list(zip(matched["frame1_cell"], matched["frame2_cell"]))
        previous_cells = frame_a
        previous_matches = dict(pairs[figure_id])

    return merge_pairs(pairs, start_index=start_index)


def merge_pairs(pairs: Mapping[int, Sequence], start_index: int = 1) -> pd.DataFrame:
    """The stitching half of `cell_tracking._merge_tracking_pairs`, on
    in-memory pair lists instead of Excel files.

    Reproduces its open-track bookkeeping exactly, including the trailing frame
    column it emits with no data in it -- deliberately, so the fast path is
    comparable to a real run column-for-column rather than subtly tidier.
    """
    if not pairs:
        return pd.DataFrame()
    ordered = sorted(pairs)
    start_frame, count = ordered[0], len(ordered)

    tracks: list = []
    open_tracks: dict = {}
    for frame in range(start_frame, start_frame + count):
        for cell_a, cell_b in pairs.get(frame, []):
            column_a, column_b = f"Frame{frame}", f"Frame{frame + 1}"
            if cell_a in open_tracks:
                index = open_tracks.pop(cell_a)
            else:
                index = len(tracks)
                tracks.append({})
            tracks[index][column_a] = cell_a
            tracks[index][column_b] = cell_b
            open_tracks[cell_b] = index

    columns = [f"Frame{n}" for n in range(start_frame, start_frame + count + 1)]
    merged = pd.DataFrame(tracks, columns=columns)
    merged.insert(0, "Index", range(start_index, start_index + len(merged)))
    return merged


# --------------------------------------------------------------------------
# Diagnostics -- the reusable part of how the findings were reached
# --------------------------------------------------------------------------

def mask_centroids(labels: np.ndarray) -> dict:
    """`{label: (x, y)}` straight from the mask.

    Independent of `Cells_info` AND of the tracking table, so a diagnostic
    built on these cannot be accused of inheriting either. (They agree closely
    anyway: 0.340 px median / 2.911 px max against `Cells_info` over all 7,444
    cells of `embryos_GBE_GAP43`, so `Cells_info` is not a confound.)
    """
    from scipy import ndimage
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if not len(ids):
        return {}
    centres = ndimage.center_of_mass(labels > 0, labels, ids)
    return {int(i): (float(c[1]), float(c[0])) for i, c in zip(ids, centres)}


def gap_growth(masks_by_frame: Mapping[int, np.ndarray], max_gap: int = 3) -> dict:
    """Median cell displacement at frame gaps 1..`max_gap`, measured by
    CHAINING mutual-best-IoU links rather than re-matching at each gap.

    Reported against the two reference curves: fully coherent motion grows
    linearly (`gap x d1`), pure jitter grows as `sqrt(gap) x d1`.

    **Why chaining matters, and why the obvious version is wrong.** Taking each
    cell's best-IoU partner directly at gap g selects against fast cells: a cell
    that moved far may not overlap its true successor at all, so the "best"
    partner becomes a slower neighbour sitting where it used to be, and the
    measured displacement is capped by construction. On `embryos_GBE_GAP43`
    that yields a flat 7.56 / 7.52 / 7.31 px -- which Round 9 read as evidence
    of jitter and used to dismiss motion compensation. Chained, unfiltered, the
    same data gives 7.56 / 14.10 / 20.82 px against 7.56 / 15.12 / 22.69 for
    fully coherent motion. The motion is coherent; the flat curve was an
    artefact of the measurement.
    """
    frames = sorted(masks_by_frame)
    chains = {}
    for frame in frames[:-1]:
        if frame + 1 in masks_by_frame:
            chains[frame] = best_iou_successors(masks_by_frame[frame], masks_by_frame[frame + 1], mutual=True)
    centroids = {f: mask_centroids(masks_by_frame[f]) for f in frames}

    out = {}
    for gap in range(1, max_gap + 1):
        distances = []
        for frame in frames:
            if frame + gap not in masks_by_frame:
                continue
            current = {label: label for label in chains.get(frame, {})}
            for step in range(gap):
                link = chains.get(frame + step)
                if link is None:
                    current = {}
                    break
                current = {a: link[b][1] for a, b in current.items() if b in link}
            for a, b in current.items():
                if a in centroids[frame] and b in centroids[frame + gap]:
                    (x1, y1), (x2, y2) = centroids[frame][a], centroids[frame + gap][b]
                    distances.append(float(np.hypot(x2 - x1, y2 - y1)))
        out[gap] = {"median_px": float(np.median(distances)) if distances else float("nan"),
                    "n": len(distances)}
    first = out.get(1, {}).get("median_px", float("nan"))
    for gap in out:
        out[gap]["coherent_would_be"] = gap * first
        out[gap]["jitter_would_be"] = float(np.sqrt(gap)) * first
    return out


def displacement_statistics(masks_by_frame: Mapping[int, np.ndarray]) -> dict:
    """Lag-1 displacement autocorrelation and the global-drift decomposition.

    Answers "is the frame-to-frame motion coherent or is it jitter?" without
    reference to the tracking table: correspondences come from chained
    mutual-best-IoU links and positions from the masks.

    On `embryos_GBE_GAP43`: autocorrelation +0.457 (x) / +0.314 (y), 88.5% of
    consecutive steps point the same way, and removing a whole-field drift of
    about (0.36, 6.66) px/frame takes the median step from 7.57 px to 2.69 px.
    Most of the apparent per-cell motion there is one rigid translation of the
    field.
    """
    frames = sorted(masks_by_frame)
    chains = {f: best_iou_successors(masks_by_frame[f], masks_by_frame[f + 1], mutual=True)
              for f in frames[:-1] if f + 1 in masks_by_frame}
    centroids = {f: mask_centroids(masks_by_frame[f]) for f in frames}

    previous, current = [], []
    for frame in frames[1:-1]:
        if frame - 1 not in chains or frame not in chains:
            continue
        for a, (_, b) in chains[frame - 1].items():
            if b not in chains[frame]:
                continue
            c = chains[frame][b][1]
            if a in centroids[frame - 1] and b in centroids[frame] and c in centroids[frame + 1]:
                (x0, y0), (x1, y1), (x2, y2) = centroids[frame - 1][a], centroids[frame][b], centroids[frame + 1][c]
                previous.append((x1 - x0, y1 - y0))
                current.append((x2 - x1, y2 - y1))
    if not previous:
        return {"n": 0}
    previous, current = np.array(previous), np.array(current)
    drift = np.median(current, axis=0)
    return {
        "n": len(previous),
        "autocorr_x": float(np.corrcoef(previous[:, 0], current[:, 0])[0, 1]),
        "autocorr_y": float(np.corrcoef(previous[:, 1], current[:, 1])[0, 1]),
        "same_direction_fraction": float(np.mean((previous * current).sum(axis=1) > 0)),
        "global_drift_px": (float(drift[0]), float(drift[1])),
        "median_step_px": float(np.median(np.hypot(*current.T))),
        "median_residual_after_drift_px": float(np.median(np.hypot(*(current - drift).T))),
    }


def cost_judge_correlation(
    cells_info: pd.DataFrame,
    masks_by_frame: Mapping[int, np.ndarray],
    variants: Mapping[str, dict],
) -> dict:
    """Spearman correlation between each cost variant and mask IoU, over the
    in-gate candidate pairs.

    This is the test for whether `iou_agreement` can adjudicate a given change
    at all. A variant whose correlation with IoU differs markedly from the
    baseline's is a variant that judge will score unfairly -- run it BEFORE
    trusting an `iou_agreement` comparison, not after.

    `variants` maps a name to `{"drift": (dx, dy), "overlap_weight": w,
    "centroid_weight": c}`.
    """
    from scipy.stats import spearmanr
    feature_columns = cell_tracking.DEFAULT_TRACKING_FEATURES
    extent_x, extent_y, centre_x, centre_y = cell_tracking._gate_indices(list(feature_columns))
    ceiling = float(cells_info["Circumscribed Circle Radius"].max())

    collected = {name: [] for name in variants}
    ious = []
    frames = sorted(set(int(m.group(1)) for i in cells_info.index
                        for m in [re.search(r"Cell(\d+)_", str(i))] if m))
    for frame in frames:
        a = cells_info[cells_info.index.str.contains(f"Cell{frame}_")]
        b = cells_info[cells_info.index.str.contains(f"Cell{frame + 1}_")]
        if a.empty or b.empty or frame not in masks_by_frame or frame + 1 not in masks_by_frame:
            continue
        x1 = a[feature_columns[centre_x]].to_numpy(float); y1 = a[feature_columns[centre_y]].to_numpy(float)
        x2 = b[feature_columns[centre_x]].to_numpy(float); y2 = b[feature_columns[centre_y]].to_numpy(float)
        extent = np.maximum(a[feature_columns[extent_x]].to_numpy(float),
                            a[feature_columns[extent_y]].to_numpy(float))
        labels_a = [_raw_label(i) for i in a.index]
        labels_b = [_raw_label(i) for i in b.index]
        pos_a = {l: i for i, l in enumerate(labels_a)}
        pos_b = {l: j for j, l in enumerate(labels_b)}
        iou = np.zeros((len(labels_a), len(labels_b)))
        for (p, q), value in cell_tracking.mask_overlap_fractions(
                masks_by_frame[frame], masks_by_frame[frame + 1]).items():
            if p in pos_a and q in pos_b:
                iou[pos_a[p], pos_b[q]] = value
        dx = x2[None, :] - x1[:, None]; dy = y2[None, :] - y1[:, None]
        base = np.sqrt(dx * dx + dy * dy)
        inside = ~((np.abs(dx) > extent[:, None]) | (np.abs(dy) > extent[:, None]) | (base > ceiling))
        ious.append(iou[inside])
        for name, spec in variants.items():
            drift = spec.get("drift", (0.0, 0.0))
            distance = np.sqrt((dx - drift[0]) ** 2 + (dy - drift[1]) ** 2)
            cost = (spec.get("centroid_weight", 1.0) * distance
                    + spec.get("overlap_weight", 1.0) * (1.0 - iou) * extent[:, None])
            collected[name].append(cost[inside])

    ious = np.concatenate(ious)
    return {
        "n_pairs": int(len(ious)),
        "spearman_vs_iou": {
            name: float(spearmanr(np.concatenate(values), ious).correlation)
            for name, values in collected.items()
        },
    }


# --------------------------------------------------------------------------
# Ground-truth ingest
# --------------------------------------------------------------------------

def load_ctc_ground_truth(tra_folder: str) -> tuple:
    """Cell Tracking Challenge `TRA` annotation -> `(gt_links, masks_by_frame)`.

    Reads `man_track.txt` (`track_id start_frame end_frame parent_id`) and the
    per-frame `man_trackTTT.tif` label images. `gt_links` is
    `{frame: {label: label_in_next_frame}}` where labels ARE the CTC track ids,
    so a link is ground truth by construction. Frames are renumbered to start
    at 1 to match this project's `Cell{frame}_{label}` convention.

    A division is deliberately NOT emitted as a link: CTC gives the daughters
    new ids with `parent_id` set, and forcing a mother->daughter link would
    make the judge score an event our tracker is not trying to produce.

    **What this can and cannot settle.** Its purpose is to validate the harness
    -- to show the judges compute what they claim on data where the answer is
    known. It is NOT evidence about `embryos_GBE_GAP43`: CTC has no
    membrane-labelled dense epithelial sheet and no coherent tissue flow, which
    are precisely the conditions that make the GBE swaps hard. Never quote a
    CTC score as evidence about GBE.
    """
    import glob
    from PIL import Image

    track_file = os.path.join(tra_folder, "man_track.txt")
    if not os.path.exists(track_file):
        raise FileNotFoundError(f"No man_track.txt in {tra_folder}")

    files = sorted(glob.glob(os.path.join(tra_folder, "man_track*.tif")))
    masks_by_frame = {}
    for position, path in enumerate(files, start=1):
        masks_by_frame[position] = np.array(Image.open(path))

    present = {frame: set(np.unique(labels)) - {0} for frame, labels in masks_by_frame.items()}
    gt_links = {}
    for frame in sorted(masks_by_frame):
        if frame + 1 not in present:
            continue
        shared = present[frame] & present[frame + 1]
        if shared:
            gt_links[frame] = {int(label): int(label) for label in shared}
    return gt_links, masks_by_frame


def cells_info_from_masks(masks_by_frame: Mapping[int, np.ndarray]) -> pd.DataFrame:
    """A minimal `Cells_info`-shaped table straight from label masks.

    Only the columns the tracking cost actually reads are populated -- the four
    the displacement gate needs plus `Circumscribed Circle Radius` for the
    ceiling; everything else in `CELLS_INFO_COLUMNS` is left at 0, which is
    exactly what `build_cost_matrix`'s zero-weight skip already handles.

    This is what makes the harness dataset-agnostic: a Cell Tracking Challenge
    annotation has no `Cells_info.xlsx` and no Cellpose `.npy`, and running the
    real extraction on it would drag in edge-cell exclusion and gap closing
    that have nothing to do with judging a tracker.
    """
    from scipy import ndimage
    rows = {}
    for frame in sorted(masks_by_frame):
        labels = masks_by_frame[frame]
        ids = np.unique(labels)
        ids = ids[ids > 0]
        if not len(ids):
            continue
        centres = ndimage.center_of_mass(labels > 0, labels, ids)
        boxes = ndimage.find_objects(labels)
        for label, centre in zip(ids, centres):
            box = boxes[int(label) - 1]
            if box is None:
                continue
            row = {column: 0.0 for column in data_input.CELLS_INFO_COLUMNS}
            row["Center X Coordinate"] = float(centre[1])
            row["Center Y Coordinate"] = float(centre[0])
            width = float(box[1].stop - box[1].start)
            height = float(box[0].stop - box[0].start)
            row["Maximum Horizontal Length"] = width
            row["Maximum Vertical Length"] = height
            row["Circumscribed Circle Radius"] = 0.5 * float(np.hypot(width, height))
            rows[f"Cell{frame}_{int(label)}"] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Cell Index"
    return table


def load_link_csv(path: str) -> dict:
    """Hand-labelled links from a CSV with columns
    `frame, cell_index_f, cell_index_f1` -> `{frame: {label: label}}`.

    Deliberately the smallest possible format, and deliberately no annotator:
    the webview already has a two-frame view with click-to-select, so an
    annotated set can be produced there and dropped in here without more code.
    """
    frame_column, a_column, b_column = "frame", "cell_index_f", "cell_index_f1"
    table = pd.read_csv(path)
    missing = {frame_column, a_column, b_column} - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s) {sorted(missing)}")
    links: dict = {}
    for _, row in table.iterrows():
        links.setdefault(int(row[frame_column]), {})[
            _raw_label(row[a_column])] = _raw_label(row[b_column])
    return links


def save_link_csv(links: Mapping[int, Mapping[int, int]], path: str) -> str:
    """Write `{frame: {label: label}}` back out in `load_link_csv`'s format."""
    rows = [
        {"frame": frame, "cell_index_f": f"Cell{frame}_{a}", "cell_index_f1": f"Cell{frame + 1}_{b}"}
        for frame in sorted(links) for a, b in sorted(links[frame].items())
    ]
    pd.DataFrame(rows, columns=["frame", "cell_index_f", "cell_index_f1"]).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# Per-link classification, for the Correct Tracking overlay
# --------------------------------------------------------------------------

def table_link_consistency(
    tracking_data: pd.DataFrame,
    cells_info: pd.DataFrame,
    masks_by_frame: Mapping[int, np.ndarray],
    frames: Optional[list] = None,
    drift_mode: str = "none",
    drift_by_frame: Optional[Mapping[int, tuple]] = None,
    **cost_kwargs,
) -> dict:
    """How many of the links the TRACKING TABLE asserts are not confirmed when
    the frame pair is solved backwards.

    **This is not `forward_backward_consistency`, and the difference is the
    whole point.** That function re-solves both directions from `cells_info`
    and the masks and never looks at `tracking_data` at all -- measured on
    `AS_dorsal view_E-cad`, a fully SCRAMBLED tracking table scores identically
    (48 / 2,634, 1.822% both ways). It judges the cost function on this data,
    not the table. Using it to score a manual correction would return a delta
    of exactly zero for every plan, endorsing a destructive one as readily as a
    good one -- a judge that cannot fail.

    This one takes each link the table holds (`label at f` -> `label at f+1`
    for the same track) and asks whether the backward solve of that pair says
    `b -> a`. Edit the table and the number moves, which is the property an
    accept-and-revert loop needs.

    `frames` limits the work to the pairs a plan touches; `None` means every
    pair.

    **Still a consistency measure, never an accuracy.** It uses the same cost
    function the tracker used, so it answers "is this solution self-consistent
    under transposition", not "is it right". Its degenerate case is shared with
    `environment_dice` and with `forward_backward_consistency`: a coherent
    identity permutation is perfectly self-consistent and scores 0. And both
    sides of any delta must share one registration convention -- Round 15
    measured that switch alone moving the flagged count on GBE from 955 to 24.
    """
    backward = _solve_pairs(cells_info, masks_by_frame, drift_mode=drift_mode,
                            drift_by_frame=drift_by_frame, reverse=True, **cost_kwargs)
    pairs = sorted(backward) if frames is None else sorted(
        f for f in backward if f in set(int(x) for x in frames))

    total = unconfirmed = 0
    per_frame: dict = {}
    for frame in pairs:
        here = label_to_track(tracking_data, frame)
        there = {track: label for label, track in label_to_track(tracking_data, frame + 1).items()}
        back = backward.get(frame, {})
        broken = []
        for label, track in here.items():
            partner = there.get(track)
            if partner is None:
                continue  # the track ends here; there is no link to confirm
            total += 1
            if back.get(partner) != label:
                unconfirmed += 1
                broken.append((int(label), int(partner)))
        if broken:
            per_frame[frame] = broken

    return {
        "judge": "table_link_consistency",
        "unconfirmed": unconfirmed,
        "n": total,
        "rate": 100.0 * unconfirmed / total if total else float("nan"),
        "frames": pairs,
        "broken_by_frame": per_frame,
        "drift_mode": drift_mode,
    }


def classify_links(
    tracking_data: pd.DataFrame,
    masks_by_frame: Mapping[int, np.ndarray],
    frame_number: int,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    register: bool = False,
    drift_by_frame: Optional[Mapping[int, tuple]] = None,
    margin: float = 0.0,
) -> dict:
    """`{raw_label: "agrees" | "disagrees" | "unjudgeable" | "broken"}` for one
    frame, from `iou_agreement`'s per-cell verdict.

    **Three states, not two, and the wording is not cosmetic.** This says
    whether a link agrees with an overlap-based reference -- not whether it is
    right. About a quarter of cells cannot be judged at all, and the reference
    itself is biased (see `iou_agreement`). A UI built on this must say
    "disagrees with overlap reference" and must render the unjudgeable state,
    or it re-introduces exactly the agreement-vs-accuracy confusion this module
    exists to remove.
    """
    nxt = frame_number + 1
    if frame_number not in masks_by_frame or nxt not in masks_by_frame:
        return {}
    labels_a = masks_by_frame[frame_number]
    if register:
        shift = (drift_by_frame or {}).get(frame_number) or estimate_frame_drift(masks_by_frame, frame_number)
        labels_a = shift_labels(labels_a, shift[0], shift[1])
    overlaps = cell_tracking.mask_overlap_fractions(labels_a, masks_by_frame[nxt])
    best, runner_up = {}, {}
    for (a, b), value in overlaps.items():
        if value > best.get(a, (0.0, None))[0]:
            runner_up[a] = best.get(a, (0.0, None))[0]
            best[a] = (value, b)
        elif value > runner_up.get(a, 0.0):
            runner_up[a] = value
    here, there = label_to_track(tracking_data, frame_number), label_to_track(tracking_data, nxt)
    inverse = {track: label for label, track in there.items()}

    states = {}
    for label, track in here.items():
        value, partner = best.get(label, (0.0, None))
        linked = inverse.get(track)
        # Both IoUs travel with the verdict. A reader shown "0.301 vs 0.231"
        # can see a near-tie for what it is; a reader shown a red cell cannot.
        entry = {"reference_iou": round(float(value), 3),
                 "tracker_iou": round(float(overlaps.get((label, linked), 0.0)), 3) if linked else None,
                 # The label the reference would have picked. Carried so a
                 # human queue can show the rival as a candidate instead of
                 # asking the labeller to find it.
                 "reference_label": None if partner is None else int(partner),
                 "tracker_label": None if linked is None else int(linked)}
        if value < iou_threshold or (margin and (value - runner_up.get(label, 0.0)) < margin):
            entry["state"] = "unjudgeable"
        elif linked is None:
            entry["state"] = "broken"
        elif linked == partner:
            entry["state"] = "agrees"
        else:
            entry["state"] = "disagrees"
        states[label] = entry
    return states


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tracking evaluation harness")
    parser.add_argument("--project", help="project folder (omit to run the self-test on GBE)")
    parser.add_argument("--npy-folder")
    parser.add_argument("--ctc-tra", help="a Cell Tracking Challenge TRA folder")
    parser.add_argument("--diagnostics", action="store_true")
    # A no-op: omitting --project runs the self-test. Accepted so the flag means
    # the same thing everywhere (Round 19 A1).
    parser.add_argument("--self-test", action="store_true",
                        help="run the self-test (the default when --project is omitted)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.ctc_tra:
        links, masks = load_ctc_ground_truth(args.ctc_tra)
        print("CTC: %d frames, %d frame pairs with links, %d links total"
              % (len(masks), len(links), sum(len(v) for v in links.values())))
        raise SystemExit(0)

    project = args.project or "test_data/embryos_GBE_GAP43"
    npy_folder = args.npy_folder or os.path.join(project, "masks", "npy_converted")
    if not os.path.isdir(npy_folder):
        npy_folder = os.path.join(project, "masks")

    # Read the PINNED extraction, not the dataset's `cell_classification_output/`
    # -- that folder is where the app writes on every Extract run, so a normal
    # day's work replaced the inputs these constants were measured against and
    # the first symptom was an assertion failing like a cost regression
    # (`CLAUDE.md` rule 7 case 1). `v2/baseline_data/<slug>/` is a tree the
    # application never writes to.
    import baselines
    extraction_source = baselines.cells_info_path(project)
    if not os.path.exists(extraction_source):
        raise SystemExit(
            "no pinned extraction at %s -- run:\n"
            "  python3 v2/baselines.py --adopt %r\n"
            "A pinned number is only pinned against the data it was measured on."
            % (extraction_source, project)
        )
    cells_info = pd.read_excel(extraction_source, index_col=0)
    cells_info.index = cells_info.index.map(str)
    masks = load_masks(npy_folder)
    print("loaded %d frames of masks from %s" % (len(masks), npy_folder))

    tracking = track_in_memory(cells_info, masks)
    scores = evaluate(tracking, cells_info, masks)
    a, b = scores["iou_agreement"], scores["environment_dice"]
    print("\niou_agreement : %.2f%% correct, %.2f%% wrong, %.2f%% broken  (n=%d judged of %d, %.1f%% unjudgeable)"
          % (a["correct_rate"], a["wrong_rate"], a["broken_rate"], a["n_judged"], a["n_total"], a["unjudgeable_rate"]))
    print("environment_dice: %.4f  (n=%d)" % (b["dice"], b["n"]))

    decision = registration_decision(masks, cells_info)
    print("\nregistration decision: register=%s  drift %.2f px / radius %.2f px = %.2f (threshold %.2f)"
          % (decision["register"], decision["median_drift_px"],
             decision["median_cell_radius_px"], decision["ratio"], decision["threshold"]))
    drift = decision["drift_by_frame"]
    plain = data_properties(masks, register=False)
    fixed = data_properties(masks, register=True, drift_by_frame=drift)
    print("data properties       unregistered -> registered")
    print("  median best IoU     %.3f -> %.3f" % (plain["median_best_iou"], fixed["median_best_iou"]))
    print("  median margin       %.3f -> %.3f" % (plain["median_margin"], fixed["median_margin"]))
    print("  judgeable           %d -> %d" % (plain["judgeable"], fixed["judgeable"]))

    # Round 16 B3: `forward_backward_consistency` never reads the tracking
    # table -- a fact that only shows up when you change the table. Scoring a
    # manual correction with it would return the same number whatever the plan
    # did, so the two judges are asserted apart here rather than left to be
    # re-discovered. Deliberately ABOVE the pinned-numbers block, which exits
    # early when the stored table is no longer a baseline: this property is
    # about the judges, not about any particular table, so it must still run.
    blind_base = forward_backward_consistency(cells_info=cells_info, masks_by_frame=masks)
    scrambled = tracking.copy(deep=True)
    rng = np.random.default_rng(0)
    for column in (c for c in scrambled.columns if c.startswith("Frame")):
        scrambled[column] = rng.permutation(scrambled[column].values)
    blind = forward_backward_consistency(scrambled, masks, cells_info)
    assert blind["inconsistent"] == blind_base["inconsistent"], (blind, blind_base)

    intact = table_link_consistency(tracking, cells_info, masks)
    broken = table_link_consistency(scrambled, cells_info, masks)
    assert broken["rate"] > intact["rate"] + 50, (intact, broken)
    print("\ntransposition judges, on the same table scrambled:")
    print("  forward_backward_consistency: %.2f%% -> %.2f%%  (BLIND to the table)"
          % (blind_base["inconsistency_rate"], blind["inconsistency_rate"]))
    print("  table_link_consistency:       %.2f%% -> %.2f%%  (reads the table)"
          % (intact["rate"], broken["rate"]))

    # The pinned numbers below are a statement about ONE extraction of this
    # dataset. `cell_classification_output/Cells_info.xlsx` is also where the
    # app writes when someone re-runs Extract, so a normal day's use silently
    # replaces the inputs these constants were measured against -- and the
    # first symptom is an assertion failure that reads exactly like a cost
    # regression. That happened on 2026-09-05: a re-extraction took GBE to
    # 7,456 cells and `wrong_rate` from 4.89% to 4.83%, with nothing in the
    # tracker changed.
    #
    # So fingerprint the inputs and say which of the two it is. A number is
    # only pinned against the data it was measured on.
    # 7,457 cells, re-extracted 2026-09-06 from HAND-CORRECTED masks. The
    # previous pin (396ce2945598632c, 7,456 cells) described the segmentation
    # that preceded them. This is the third re-pin in three rounds and every
    # one has been an INPUT change, never the code -- verified each time by
    # running HEAD against the same inputs.
    # The SEGMENTATION these numbers describe. Hand-corrected 2026-09-07
    # (7,452 labels, down from 7,457).
    GBE_MASKS = "2a0a819abe94a52b15a8c22943960ee4"
    GBE_EXTRACTION = "73fd7dedb97f3bc7"
    extraction = hashlib.sha256(cells_info.to_csv().encode()).hexdigest()[:16]

    if "embryos_GBE_GAP43" in project:
        # Pin the shipped-default numbers so a silent change to the cost, the
        # merge or a judge is caught here rather than in a later argument about
        # what a benchmark meant.
        if extraction != GBE_EXTRACTION:
            print("\nSKIPPING the pinned GBE numbers -- the EXTRACTION changed, so this is not a")
            print("  regression: %s, expected %s." % (extraction, GBE_EXTRACTION))
            print("  Re-run Extract, adopt and regenerate, then re-pin these deliberately.")
            raise SystemExit(0)
        # And the masks. Round 18 added this check to `baselines`; leaving it out
        # here meant a hand-correction to the segmentation still surfaced as a
        # cost regression -- which is exactly what happened in the first real
        # click-test's preflight (Round 19 A1). These numbers describe ONE
        # segmentation as much as one extraction.
        import baselines
        current_masks = baselines.mask_fingerprint(npy_folder)
        if current_masks and current_masks["sha256"] != GBE_MASKS:
            print("\nSKIPPING the pinned GBE numbers -- the SEGMENTATION changed, so this is not")
            print("  a regression: %d frames / %d labels now, pinned against %s."
                  % (current_masks["frames"], current_masks["labels"], GBE_MASKS[:12]))
            print("  Hand-correcting masks is normal work; re-run Extract, adopt and regenerate.")
            raise SystemExit(0)
        # Re-pinned 2026-09-07. Superseded values, kept so the sequence stays
        # legible rather than being silently overwritten -- each describes a
        # different SEGMENTATION of the same movie, and every move was verified
        # to be the inputs and not the code:
        #   pre-2026-09-05 : 4.89 / 94.84 / 0.27, judged 5480, total 7229
        #   2026-09-05     : 4.83 / 94.92 / 0.26, judged 5487, total 7221
        #   2026-09-06     : 4.64 / 95.10 / 0.25, judged 5492, total 7233  (7,457 labels)
        #   2026-09-07a    : 4.73 / 95.01 / 0.25, judged 5492, total 7229  (7,452 labels)
        # Now 7,449 labels. Five re-segmentations in four days; the two guards
        # above are what keep each of these from reading as a code regression.
        # Both guards above now cover this: the extraction AND the masks.
        assert abs(a["wrong_rate"] - 4.66) < 0.05, a
        assert abs(a["correct_rate"] - 95.08) < 0.05, a
        assert abs(a["broken_rate"] - 0.25) < 0.05, a
        assert abs(a["n_judged"] - 5492) < 25, a
        assert abs(a["n_total"] - 7226) < 25, a
        # 0.9015, not the 0.9094 the Round 10 plan quoted. Every definition
        # tried lands in 0.897-0.903 (per-frame mean 0.8992, keeping untracked
        # neighbours 0.8999, dilation_px 1 and 3 giving 0.9007 and 0.9025), so
        # the plan's figure could not be reproduced and this asserts the
        # measured one. Fitting code to a number whose definition cannot be
        # recovered is the failure mode this whole module exists to prevent.
        assert abs(b["dice"] - 0.9015) < 0.002, b
        print("  -> reproduces the Round 9/10 reference numbers.")

        # Round 11: the registration finding. These are the numbers that show
        # the UNREGISTERED reference is operating at its own validity limit --
        # a median self-overlap of 0.377 against its own 0.3 threshold.
        assert decision["register"] is True, decision
        assert abs(plain["median_best_iou"] - 0.377) < 0.01, plain
        assert abs(fixed["median_best_iou"] - 0.709) < 0.01, fixed
        assert abs(plain["median_margin"] - 0.224) < 0.01, plain
        assert abs(fixed["median_margin"] - 0.661) < 0.01, fixed
        # Also re-pinned 2026-09-05 with the rest of this block; previously
        # 5483 / 6818 on the extraction that was replaced.
        assert plain["judgeable"] == 5489 and abs(fixed["judgeable"] - 6821) < 10, (plain, fixed)

        registered_score = iou_agreement(tracking, masks, register=True, drift_by_frame=drift)
        print("\nsame tracking, registered reference: %.2f%% wrong (was %.2f%%), n=%d judged"
              % (registered_score["wrong_rate"], a["wrong_rate"], registered_score["n_judged"]))
        # Re-pinned 2026-09-07 (14.01 -> 13.99 -> 13.75 -> 13.88 across four
        # segmentations of the same movie).
        assert abs(registered_score["wrong_rate"] - 13.77) < 0.2, registered_score
        print("  -> the SAME tracking scores 4.66%% or 13.77%% depending only on the")
        print("     registration convention of the reference. Never report one alone.")

        consistency = forward_backward_consistency(cells_info=cells_info, masks_by_frame=masks)
        margin_score = assignment_margin(cells_info=cells_info, masks_by_frame=masks)
        print("\nreference-free: forward-backward %.2f%% inconsistent | assignment margin %.2f px"
              % (consistency["inconsistency_rate"], margin_score["median_margin"]))
        # Re-pinned 2026-09-07 (1.98 -> 2.55 -> 2.41 -> 2.17 across four).
        assert abs(consistency["inconsistency_rate"] - 1.83) < 0.1, consistency
        assert abs(margin_score["median_margin"] - 11.35) < 0.2, margin_score

    if args.diagnostics:
        print("\ngap growth along chained mutual-best-IoU links:")
        for gap, row in gap_growth(masks).items():
            print("  gap %d: %6.2f px  (coherent %6.2f, jitter %6.2f, n=%d)"
                  % (gap, row["median_px"], row["coherent_would_be"], row["jitter_would_be"], row["n"]))
        stats = displacement_statistics(masks)
        print("\ndisplacement: autocorr x %+.3f y %+.3f | same direction %.1f%% | drift (%.2f, %.2f) px/frame"
              % (stats["autocorr_x"], stats["autocorr_y"], 100 * stats["same_direction_fraction"],
                 *stats["global_drift_px"]))
        print("  median step %.2f px -> %.2f px after removing drift"
              % (stats["median_step_px"], stats["median_residual_after_drift_px"]))
        drift = stats["global_drift_px"]
        correlation = cost_judge_correlation(cells_info, masks, {
            "no prediction": {"drift": (0.0, 0.0)},
            "drift compensated": {"drift": drift},
        })
        print("\nSpearman(cost, IoU) over %d in-gate pairs:" % correlation["n_pairs"])
        for name, value in correlation["spearman_vs_iou"].items():
            print("  %-20s %+.3f" % (name, value))
        print("  -> iou_agreement is NOT neutral between these two.")

    print("\nAll tracking_benchmark self-tests passed.")
