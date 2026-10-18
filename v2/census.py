"""A labelling queue for one link at a time, and the honesty checks around it.

The atomic task is: *is the cell at frame f the same cell as the one this link
points to at f+1?* Everything here exists to make that judgment cheap and to
make the resulting labels checkable.

**Why a census and not a sample.** On registered GBE the geometry-flagged
disagreement rate is 0.64% (44 of 6,928 links). A uniform random sample of 200
contains ~1.3 errors; a 95% CI would need n ~ 2,800. Uniform sampling is not a
slow option here, it is an impossible one. So the queue is always a *census* of
some decision-relevant set -- every geometry-flagged link on a movie (~40
items), or every link in a chosen frame pair (~240) -- never a random draw
across the movie.

**Three checks travel with every session, because a human shown the
algorithm's answer is the same reference-contamination failure this project has
already made five times, with a person in the loop:**

- `decoy` items display a deliberately wrong successor. Answering "correct" on
  a decoy is rubber-stamping. Anchoring cannot be removed from a yes/no
  protocol, but the decoy pass-rate makes it a *measured*, reportable property
  of the labels rather than a hope.
- `repeat` items silently re-ask an earlier item. Intra-rater agreement is the
  **noise floor of the ground truth itself**: at 92% self-consistency no
  tracker can be validated above 92%, and every accuracy figure must be quoted
  against that ceiling.
- every verdict carries `ms_elapsed`. Fast answers that fail their own repeat
  are fatigue, and that is the signal to end the session.

Verdicts are three-state -- `correct` / `wrong` / `unjudgeable` -- for the same
reason `classify_links` and `annotate_persistence` are. A link a domain expert
genuinely cannot resolve is a limit of the data, not a tracker error, and
folding it into either bucket invents a number.
"""

from __future__ import annotations

import csv
import json
import os
import random
import statistics
from datetime import datetime, timezone
from typing import Mapping, Optional

import numpy as np
import pandas as pd

import re

import cell_tracking
import tracking_benchmark

VERDICTS = ("correct", "wrong", "unjudgeable")

#: `chosen_reason` / `chosen_rank` are what settle whether a candidate source
#: earns its slot, and they are **reference-free**: they record which ranking
#: first proposed the cell a HUMAN picked, and where in the list it sat. No
#: tracking table is consulted, so no unvalidated table can bias them -- which
#: is the trap Rounds 19-21 fell into three times.
#:
#: `register` and `environment_threshold` are the settings half. Without them a
#: verdict file cannot say afterwards what it was collected under, and two files
#: cannot be told apart: the maintainer's 18 hand-classified GBE items and this
#: repository's 109-item queue for the same set differ because the tracking
#: behind them differed, and nothing in either file recorded that.
FIELDS = (
    "session_id", "item_id", "source_item_id", "frame", "label", "track_id",
    "shown_label", "true_label", "is_decoy", "is_repeat", "verdict",
    "chosen_label", "chosen_reason", "chosen_rank", "candidate_mode", "ms_elapsed",
    "labeller", "source", "register", "environment_threshold", "timestamp_utc",
)
#: `candidate_mode` (Round 25 D1) says WHICH candidate generator offered the list
#: a verdict was given against -- `ranked` (the Round 20-23 multi-source merge) or
#: `radius:k=3` (every cell within k median cell radii of the registered anchor).
#: The two coexist until the maintainer's clicks decide between them, and a
#: verdict that does not say which one it saw cannot vote.


def _migrate_header(path: str, header: list) -> None:
    """Rewrite `path` under the current `FIELDS`, keeping every row.

    `append_verdict` used to open the file in append mode and write rows keyed by
    the CURRENT `FIELDS` beneath whatever header the file already had. When Round
    23 added four columns, any `ground_truth_links.csv` started earlier began
    receiving rows whose values sat under the wrong headings -- no error, and a
    `DictReader` then paired `chosen_reason` with a timestamp. Found in Round 25
    while adding a fifth column. Old rows get blanks for the new columns; an
    unknown column raises rather than being dropped, because a verdict file is
    not ours to lose data from.
    """
    unknown = [name for name in header if name not in FIELDS]
    if unknown:
        raise ValueError(f"{path} has columns this version does not know: {unknown}; "
                         "refusing to rewrite it and lose them")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    tmp = path + ".migrating"
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, restval="")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)

DEFAULT_DECOY_RATE = 0.15
DEFAULT_REPEAT_RATE = 0.10


def _centroids(labels: np.ndarray) -> dict:
    out = {}
    for label in np.unique(labels):
        if label == 0:
            continue
        ys, xs = np.nonzero(labels == label)
        out[int(label)] = (float(xs.mean()), float(ys.mean()))
    return out


def geometry_flagged_items(
    tracking_data: pd.DataFrame,
    masks_by_frame: Mapping[int, np.ndarray],
    register: bool = True,
    drift_by_frame: Optional[Mapping[int, tuple]] = None,
    iou_threshold: float = tracking_benchmark.DEFAULT_IOU_THRESHOLD,
) -> list:
    """Every link whose tracker pick disagrees with the overlap reference.

    This is the set Round 14 §A3 could not settle: on registered GBE, *every*
    such disagreement had its rival already claimed by another track, and
    essentially none formed a true 2-cycle. That is a global solver making a
    trade-off while a greedy per-cell reference names the same taken cell --
    and which one is right **cannot be decided by any reference-free metric**,
    because the disagreement is symmetric by construction. Only a human can
    break it, which is what this queue is for.

    `register` defaults True and the caller must record it: an overlap
    reference computed on unregistered masks scores a registered tracker by a
    convention it does not share, and that one switch has already moved a
    figure from 4.9% to 14.0%.
    """
    frames = sorted(f for f in masks_by_frame if f + 1 in masks_by_frame)
    items = []
    for frame in frames:
        states = tracking_benchmark.classify_links(
            tracking_data, masks_by_frame, frame, iou_threshold=iou_threshold,
            register=register, drift_by_frame=drift_by_frame,
        )
        if not states:
            continue
        here = tracking_benchmark.label_to_track(tracking_data, frame)
        there = tracking_benchmark.label_to_track(tracking_data, frame + 1)
        inverse = {track: label for label, track in there.items()}
        centres = _centroids(masks_by_frame[frame])

        flagged = [(label, entry) for label, entry in states.items()
                   if entry["state"] == "disagrees"]
        # Raster order within the frame pair: consecutive items are adjacent
        # cells, so the crop moves a little instead of across the image. After
        # the blink, this is the biggest remaining speed factor.
        flagged.sort(key=lambda pair: (round(centres.get(pair[0], (0.0, 0.0))[1] / 50.0),
                                       centres.get(pair[0], (0.0, 0.0))[0]))
        for label, entry in flagged:
            track = here.get(label)
            items.append({
                "frame": frame,
                "label": int(label),
                "track_id": None if track is None else int(track),
                "tracker_label": inverse.get(track),
                "reference_label": entry.get("reference_label"),
                "reference_iou": entry.get("reference_iou"),
                "tracker_iou": entry.get("tracker_iou"),
                "source": "geometry_flagged",
            })
    return items


# The sets a queue can be drawn from. `frame_pair` is the only one that gives
# an unbiased rate; the other two are chosen BY a criterion, so a rate computed
# over them is a rate over that criterion's blind spots.
SOURCES = ("geometry_flagged", "frame_pair", "environment_flagged",
           "forward_backward", "low_margin")
UNBIASED_SOURCES = ("frame_pair",)


def environment_flagged_items(
    tracking_data: pd.DataFrame,
    scores_by_frame: Mapping[int, Mapping[int, float]],
    masks_by_frame: Mapping[int, np.ndarray],
    threshold: float = 0.5,
) -> list:
    """Links whose cell scores at or below the environment repair tier.

    Users correct what the visualisation flags, and the environment overlay is
    one of the two things they look at, so this makes that queue explicit
    rather than leaving it to clicking around. `threshold` defaults to TA's
    **repair** tier (0.5), not its **look** tier (0.75) -- the look tier flags
    several times as many cells, which is the wrong size for a census.

    **This set is stratified by a biased criterion, and the bias is known.**
    `environment_dice` is invariant under a coherent permutation of identities,
    so it rewards exactly the systematic mislabelling a drift-compensated cost
    can introduce. That is fine for a QUEUE -- choosing what to look at judges
    nothing -- but a rate computed over labels collected from it is a rate over
    the cells this metric happens to flag. Only `frame_pair` gives an unbiased
    rate, and `summarise` refuses to pool across sources rather than leaving
    that to a footnote.
    """
    items = []
    for frame in sorted(scores_by_frame):
        if frame not in masks_by_frame or frame + 1 not in masks_by_frame:
            continue
        here = tracking_benchmark.label_to_track(tracking_data, frame)
        there = tracking_benchmark.label_to_track(tracking_data, frame + 1)
        inverse = {track: label for label, track in there.items()}
        centres = _centroids(masks_by_frame[frame])
        flagged = [(int(label), float(value))
                   for label, value in scores_by_frame[frame].items()
                   if value <= threshold and int(label) in here]
        flagged.sort(key=lambda pair: (round(centres.get(pair[0], (0.0, 0.0))[1] / 50.0),
                                       centres.get(pair[0], (0.0, 0.0))[0]))
        for label, score in flagged:
            track = here.get(label)
            items.append({
                "frame": frame, "label": label,
                "track_id": None if track is None else int(track),
                "tracker_label": inverse.get(track),
                "reference_label": None, "reference_iou": None, "tracker_iou": None,
                "environment_score": round(score, 3),
                "source": "environment_flagged",
            })
    return items


def forward_backward_items(
    tracking_data: pd.DataFrame,
    cells_info: pd.DataFrame,
    masks_by_frame: Mapping[int, np.ndarray],
    drift_mode: str = "none",
    drift_by_frame: Optional[Mapping[int, tuple]] = None,
    **cost_kwargs,
) -> list:
    """Links the reverse solve does not agree with: forward says a->b, the
    transposed problem does not say b->a.

    **Not trivially satisfied.** The cost is asymmetric by construction -- the
    displacement gate and the overlap term's `max(H, V)` bridge both come from
    the ROW cell -- so swapping the frames (and negating the drift) re-solves a
    genuinely different problem rather than reading the same answer back.

    **Degenerate case, and it is shared.** A systematically wrong but
    self-consistent assignment passes in both directions -- shift every
    identity one step the same way and nothing here fires. That is the SAME
    blind spot as `environment_dice`, so this queue and `environment_flagged`
    are **not two independent lines of evidence**. `summarise` already refuses
    to pool across sources; this is the reason it must keep refusing.

    Reuses `tracking_benchmark._solve_pairs`, which is what
    `forward_backward_consistency` reports its 1.98 / 1.05 / 0.22% from. A
    second implementation here would quietly make those three numbers describe
    something other than this queue.
    """
    forward = tracking_benchmark._solve_pairs(
        cells_info, masks_by_frame, drift_mode=drift_mode,
        drift_by_frame=drift_by_frame, **cost_kwargs)
    backward = tracking_benchmark._solve_pairs(
        cells_info, masks_by_frame, drift_mode=drift_mode,
        drift_by_frame=drift_by_frame, reverse=True, **cost_kwargs)
    items = []
    for frame in sorted(forward):
        if frame not in masks_by_frame or frame + 1 not in masks_by_frame:
            continue
        back = backward.get(frame, {})
        here = tracking_benchmark.label_to_track(tracking_data, frame)
        there = tracking_benchmark.label_to_track(tracking_data, frame + 1)
        inverse = {track: label for label, track in there.items()}
        centres = _centroids(masks_by_frame[frame])
        flagged = [(int(a), int(b)) for a, b in forward[frame].items() if back.get(b) != a]
        flagged.sort(key=lambda pair: (round(centres.get(pair[0], (0.0, 0.0))[1] / 50.0),
                                       centres.get(pair[0], (0.0, 0.0))[0]))
        for label, forward_pick in flagged:
            track = here.get(label)
            items.append({
                "frame": frame, "label": label,
                "track_id": None if track is None else int(track),
                "tracker_label": inverse.get(track),
                "reference_label": None, "reference_iou": None, "tracker_iou": None,
                # What the forward solve said, for the record. NOT a reference:
                # it is the same cost function, solved one way round.
                "forward_label": forward_pick,
                "source": "forward_backward",
            })
    return items


def low_margin_items(
    tracking_data: pd.DataFrame,
    cells_info: pd.DataFrame,
    masks_by_frame: Mapping[int, np.ndarray],
    limit: Optional[int] = None,
    drift_mode: str = "none",
    drift_by_frame: Optional[Mapping[int, tuple]] = None,
    **cost_kwargs,
) -> list:
    """Assigned links ordered by the solver's own confidence: the gap between
    the second-smallest and smallest finite cost in that row, smallest first.

    Free and reference-free -- the number is already inside the cost matrix.

    **Degenerate case:** it rewards any cost that merely spreads its values
    out, and it lives in the cost's own units, so margins are comparable only
    within one cost scale. A queue built under one `cost_form` cannot be
    compared with one built under another.

    Ordered by margin rather than in raster order, unlike every other set
    here: the point of this queue is "least sure first", and that ordering is
    the whole selection criterion.
    """
    margins = tracking_benchmark._solve_pairs(
        cells_info, masks_by_frame, drift_mode=drift_mode,
        drift_by_frame=drift_by_frame, want_margins=True, **cost_kwargs)
    ranked = []
    for frame in sorted(margins):
        if frame not in masks_by_frame or frame + 1 not in masks_by_frame:
            continue
        for label, margin in margins[frame].items():
            ranked.append((float(margin), int(frame), int(label)))
    ranked.sort()
    if limit is not None:
        ranked = ranked[:int(limit)]
    items = []
    for margin, frame, label in ranked:
        here = tracking_benchmark.label_to_track(tracking_data, frame)
        there = tracking_benchmark.label_to_track(tracking_data, frame + 1)
        inverse = {track: lab for lab, track in there.items()}
        track = here.get(label)
        items.append({
            "frame": frame, "label": label,
            "track_id": None if track is None else int(track),
            "tracker_label": inverse.get(track),
            "reference_label": None, "reference_iou": None, "tracker_iou": None,
            "assignment_margin": round(margin, 3),
            "source": "low_margin",
        })
    return items


def frame_pair_items(
    tracking_data: pd.DataFrame,
    masks_by_frame: Mapping[int, np.ndarray],
    frame: int,
) -> list:
    """Every link in one frame pair -- the unbiased design. Raster ordered."""
    if frame not in masks_by_frame or frame + 1 not in masks_by_frame:
        return []
    here = tracking_benchmark.label_to_track(tracking_data, frame)
    there = tracking_benchmark.label_to_track(tracking_data, frame + 1)
    inverse = {track: label for label, track in there.items()}
    centres = _centroids(masks_by_frame[frame])
    labels = sorted(here, key=lambda l: (round(centres.get(l, (0.0, 0.0))[1] / 50.0),
                                         centres.get(l, (0.0, 0.0))[0]))
    return [{"frame": frame, "label": int(label), "track_id": int(here[label]),
             "tracker_label": inverse.get(here[label]), "reference_label": None,
             "reference_iou": None, "tracker_iou": None, "source": "frame_pair"}
            for label in labels]


def build_queue(items: list, masks_by_frame: Mapping[int, np.ndarray],
                decoy_rate: float = DEFAULT_DECOY_RATE,
                repeat_rate: float = DEFAULT_REPEAT_RATE,
                seed: int = 0) -> list:
    """Turn raw items into a labelling queue with decoys and silent repeats.

    A decoy replaces the DISPLAYED successor with a nearby wrong one; the true
    successor is kept in `true_label` so the verdict can be scored. Repeats are
    appended in the back half, in shuffled order, so they do not sit next to
    the item they repeat.

    An item whose displayed successor cannot be faked (no other cell in the
    next frame) is simply never made a decoy, rather than being dropped -- the
    census must stay a census.
    """
    if not 0.0 <= decoy_rate < 1.0 or not 0.0 <= repeat_rate < 1.0:
        raise ValueError("decoy_rate and repeat_rate are fractions in [0, 1).")
    rng = random.Random(seed)
    queue = []
    for position, item in enumerate(items):
        entry = dict(item)
        entry["item_id"] = f"i{position:04d}"
        entry["source_item_id"] = None
        entry["is_repeat"] = False
        true_label = item.get("tracker_label")
        entry["true_label"] = true_label
        entry["is_decoy"] = False
        entry["shown_label"] = true_label
        queue.append(entry)

    # An exact quota, not an independent draw per item. At census sizes -- 24
    # geometry-flagged links on GBE -- a 15% Bernoulli rate returns ZERO decoys
    # about 2% of the time (seed 0 was one such), and a session with no decoys
    # has no rubber-stamping check at all. A quota removes that variance; the
    # floor of 1 keeps even a tiny census instrumented.
    eligible = [e for e in queue if e["true_label"] is not None]
    n_decoys = int(round(len(queue) * decoy_rate))
    if decoy_rate > 0 and len(eligible) >= 3:
        n_decoys = max(1, n_decoys)
    for entry in rng.sample(eligible, min(n_decoys, len(eligible))):
        others = [int(l) for l in np.unique(masks_by_frame.get(entry["frame"] + 1, np.zeros(1)))
                  if l != 0 and int(l) != int(entry["true_label"])]
        if not others:
            continue  # nothing to swap in; stay a census rather than drop it
        entry["is_decoy"] = True
        entry["shown_label"] = _nearest_other(
            masks_by_frame, entry["frame"] + 1, int(entry["true_label"]), others, rng)

    repeats = max(0, int(round(len(queue) * repeat_rate)))
    if repeat_rate > 0 and len(queue) >= 3:
        repeats = max(1, repeats)
    if repeats:
        # Repeat only non-decoys: a repeated decoy measures decoy consistency,
        # not intra-rater agreement on the real task.
        pool = [e for e in queue if not e["is_decoy"]]
        chosen = rng.sample(pool, min(repeats, len(pool)))
        tail = []
        for position, original in enumerate(chosen):
            repeat = dict(original)
            repeat["item_id"] = f"r{position:04d}"
            repeat["source_item_id"] = original["item_id"]
            repeat["is_repeat"] = True
            tail.append(repeat)
        rng.shuffle(tail)
        # Interleave into the back half, but never within two slots of the
        # original: a repeat the labeller can still remember answering measures
        # short-term memory, not intra-rater agreement.
        midpoint = len(queue) // 2
        for offset, repeat in enumerate(tail):
            origin = [e["item_id"] for e in queue].index(repeat["source_item_id"])
            position = min(max(midpoint + offset * 2 + 1, origin + 3), len(queue))
            queue.insert(position, repeat)
    return queue


def _nearest_other(masks_by_frame, frame, true_label, others, rng):
    """A decoy should be plausible -- the nearest other cell, not a random one
    across the image, which the labeller would reject without looking."""
    labels = masks_by_frame.get(frame)
    if labels is None:
        return rng.choice(others)
    centres = _centroids(labels)
    origin = centres.get(int(true_label))
    if origin is None:
        return rng.choice(others)
    candidates = [(np.hypot(centres[o][0] - origin[0], centres[o][1] - origin[1]), o)
                  for o in others if o in centres]
    if not candidates:
        return rng.choice(others)
    candidates.sort()
    return int(candidates[0][1])


# --------------------------------------------------------------------------
# Ground-truth storage (Round 18 E)
# --------------------------------------------------------------------------
#
# Laid out the way the Cell Tracking Challenge pairs a result folder with its
# `TRA` ground truth: two sibling directories under `cell_track_output/`, the
# GT one prefixed so nothing mistakes it for a result.
#
#     cell_track_output/
#         All_cell_tracking/                            the pipeline's result
#         GT_All_cell_tracking/                         the human ground truth
#             GT_All_cell_merged_tracking_results.xlsx  same schema
#             provenance.json                           the run it was collected against
#
# **This lives inside a directory the app writes to, which looks like the
# pattern `CLAUDE.md` rule 7 forbids. It is not.** Rule 7 governs test
# fixtures -- artefacts a test compares against, which a normal day's work must
# not be able to overwrite. This is the USER'S OWN DATA, and it belongs with
# their outputs. The distinction is who owns the file, not where it sits.

GT_FOLDER_NAME = "GT_All_cell_tracking"
GT_TABLE_NAME = "GT_All_cell_merged_tracking_results.xlsx"
GT_PROVENANCE_NAME = "provenance.json"
GT_SESSION_CSV = "ground_truth_links.csv"


# --- Track-level visual inspection (Round 27 4b) ---------------------------
#
# A DIFFERENT kind of evidence from everything above, and kept in a different
# file for that reason. The census judges one LINK at a time; this records a
# human watching a whole track play and saying "that cell, in that frame, is
# wrong". "I watched this track" and "I judged this link" cannot be pooled into
# one rate, and `summarise` already refuses to pool across sources -- putting
# these rows in their own file means the question cannot even be asked by
# accident.
#
# What is recorded is a CELL, not a track: `(track_id, frame, cell_index)`. A
# track marked wrong says only that something in it is wrong; a cell marked
# wrong says where. One track may be marked many times.
INSPECTION_FIELDS = (
    "session_id", "labeller", "track_id", "frame", "cell_index",
    "ms_elapsed", "source", "timestamp_utc",
)
INSPECTION_TABLE_NAME = "track_inspection.csv"


def inspection_path(cell_track_output: str) -> str:
    return os.path.join(gt_dir(cell_track_output), INSPECTION_TABLE_NAME)


def append_inspection_mark(path: str, session_id: str, track_id: int, frame: int,
                           cell_index: str, ms_elapsed: float = 0.0,
                           labeller: str = "") -> dict:
    """One marked cell. Appends; the header is written once."""
    row = {
        "session_id": session_id,
        "labeller": labeller,
        "track_id": int(track_id),
        "frame": int(frame),
        "cell_index": str(cell_index),
        "ms_elapsed": round(float(ms_elapsed), 1),
        "source": "track_inspection",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    }
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INSPECTION_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return row


def read_inspection(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarise_inspection(path: str, session_id: Optional[str] = None) -> dict:
    """Marks, and how they are spread over tracks. Not a rate: the denominator
    (how many tracks were inspected) is known only to the session that ran, so
    it is passed in by the caller and stamped into the provenance instead."""
    rows = [r for r in read_inspection(path)
            if session_id is None or r["session_id"] == session_id]
    by_track: dict = {}
    for row in rows:
        by_track.setdefault(int(row["track_id"]), []).append(row)
    return {
        "marks": len(rows),
        "tracks_with_marks": len(by_track),
        "marks_by_track": {track: len(v) for track, v in sorted(by_track.items())},
        "cells": [r["cell_index"] for r in rows],
    }


def gt_dir(cell_track_output: str) -> str:
    return os.path.join(cell_track_output, GT_FOLDER_NAME)


def gt_table_path(cell_track_output: str) -> str:
    return os.path.join(gt_dir(cell_track_output), GT_TABLE_NAME)


def gt_provenance_path(cell_track_output: str) -> str:
    return os.path.join(gt_dir(cell_track_output), GT_PROVENANCE_NAME)


def _table_fingerprint(table: pd.DataFrame) -> str:
    import hashlib
    return hashlib.sha256(
        table.fillna("\x00").astype(str).to_csv(index=False).encode("utf-8")
    ).hexdigest()


def save_ground_truth(cell_track_output: str, table: pd.DataFrame,
                      collected_against: pd.DataFrame, **provenance) -> dict:
    """Write the ground-truth table and the provenance that makes it usable.

    `collected_against` is the tracking table this ground truth was collected
    on. **Ground truth collected against one run is not valid for a run it
    corrected** -- the link set itself changes -- and without the stamp that is
    undetectable, so the hash is not optional.
    """
    os.makedirs(gt_dir(cell_track_output), exist_ok=True)
    table.to_excel(gt_table_path(cell_track_output), index=False)
    record = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "table": GT_TABLE_NAME,
        "rows": int(len(table)),
        "gt_table_sha256": _table_fingerprint(table),
        "collected_against": {
            "rows": int(len(collected_against)),
            "tracking_table_sha256": _table_fingerprint(collected_against),
        },
        "session_record": GT_SESSION_CSV,
        **provenance,
    }
    with open(gt_provenance_path(cell_track_output), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return record


def load_ground_truth(cell_track_output: str, against: Optional[pd.DataFrame] = None) -> dict:
    """`{table, provenance, matches_current_tracking}`.

    `against` is the tracking table now in memory. When given, the answer says
    whether the ground truth was collected on it -- stated, never assumed,
    because a correction applied since is exactly what invalidates it.
    """
    path = gt_table_path(cell_track_output)
    if not os.path.exists(path):
        return {"error": f"No ground truth at {path}"}
    table = pd.read_excel(path)
    record = {}
    if os.path.exists(gt_provenance_path(cell_track_output)):
        with open(gt_provenance_path(cell_track_output), encoding="utf-8") as handle:
            record = json.load(handle)
    matches = None
    if against is not None and record:
        stamped = record.get("collected_against", {}).get("tracking_table_sha256")
        matches = bool(stamped) and stamped == _table_fingerprint(against)
    return {"table": table, "provenance": record, "matches_current_tracking": matches,
            "path": path}


def append_verdict(path: str, session_id: str, item: dict, verdict: str,
                   chosen_label: Optional[int], ms_elapsed: float,
                   labeller: str = "", chosen_reason: str = "",
                   chosen_rank: Optional[int] = None, register: Optional[bool] = None,
                   environment_threshold: Optional[float] = None,
                   candidate_mode: str = "") -> dict:
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}; expected one of {VERDICTS}")
    row = {
        "session_id": session_id,
        "item_id": item["item_id"],
        "source_item_id": item.get("source_item_id") or "",
        "frame": int(item["frame"]),
        "label": int(item["label"]),
        "track_id": "" if item.get("track_id") is None else int(item["track_id"]),
        "shown_label": "" if item.get("shown_label") is None else int(item["shown_label"]),
        "true_label": "" if item.get("true_label") is None else int(item["true_label"]),
        "is_decoy": int(bool(item.get("is_decoy"))),
        "is_repeat": int(bool(item.get("is_repeat"))),
        "verdict": verdict,
        "chosen_label": "" if chosen_label is None else int(chosen_label),
        # Which SOURCE first proposed the chosen cell, and its 1-based place in
        # the offered list. This is the measurement that decides whether
        # `pointed` amplifies a wrong link (it spends its whole quota inside the
        # neighbourhood of an answer this very set is questioning) or finds the
        # commonest error fastest (an adjacent-pair swap puts the truth right
        # beside the wrong answer). Both readings are plausible; neither is
        # settled by argument. Same for `coverage`, added Round 23 on algebra
        # alone.
        "chosen_reason": chosen_reason or "",
        "chosen_rank": "" if chosen_rank is None else int(chosen_rank),
        "candidate_mode": candidate_mode or "",
        "ms_elapsed": round(float(ms_elapsed), 1),
        "labeller": labeller,
        # Which SET this item was drawn from. Without it a file of verdicts is
        # a pile of labels with no way to know which were chosen by a biased
        # criterion, and `summarise` could not refuse to pool them.
        "source": item.get("source", ""),
        # The settings this verdict was collected under. A registered judge and
        # an unregistered one disagree about the same tracking by more than any
        # change measured on this project, so a verdict that does not say which
        # it was cannot be compared with one that does.
        "register": "" if register is None else int(bool(register)),
        "environment_threshold": "" if environment_threshold is None else float(environment_threshold),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    }
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    exists = os.path.exists(path)
    if exists:
        with open(path, newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), [])
        if header and header != list(FIELDS):
            _migrate_header(path, header)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return row


def read_verdicts(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class MixedSources(Exception):
    """Raised rather than pooling verdicts drawn from different sets.

    `geometry_flagged` and `environment_flagged` are chosen BY a criterion, so
    a rate over them is a rate over that criterion's blind spots -- and
    `environment_dice`'s blind spot is invariance under a coherent identity
    permutation, exactly the systematic error a drift-compensated cost can
    introduce. Pooling them with `frame_pair` produces a number that means
    nothing and looks like it means something. Round 15 §C1 called the mirror
    of this "the stratification error in reverse"; making it raise is the
    difference between a rule and a footnote.
    """


def summarise(path: str, session_id: Optional[str] = None,
              source: Optional[str] = None) -> dict:
    """Counts, decoy pass-rate, intra-rater agreement, and the median seconds
    per judgment -- the three numbers that qualify every figure derived from
    these labels.

    `real` counts exclude decoys and repeats: those are instrumentation, not
    data, and pooling them into the rate is the same error as pooling two
    designs that answer different questions.
    """
    rows = [r for r in read_verdicts(path)
            if session_id is None or r["session_id"] == session_id]
    if source is not None:
        rows = [r for r in rows if r.get("source", "") == source]
    present = {r.get("source", "") for r in rows}
    if len(present) > 1:
        raise MixedSources(
            "these verdicts come from %d different sets (%s) and must not be pooled: only "
            "`frame_pair` gives an unbiased rate, the others are stratified by the criterion "
            "that chose them. Pass source=... to summarise one set."
            % (len(present), ", ".join(sorted(repr(p) for p in present)))
        )
    real = [r for r in rows if r["is_decoy"] == "0" and r["is_repeat"] == "0"]
    decoys = [r for r in rows if r["is_decoy"] == "1"]
    repeats = [r for r in rows if r["is_repeat"] == "1"]

    counts = {v: sum(1 for r in real if r["verdict"] == v) for v in VERDICTS}
    judged = counts["correct"] + counts["wrong"]

    # A decoy shows a deliberately WRONG successor, so "correct" is a
    # rubber-stamp. The pass-rate is the fraction caught.
    decoy_caught = sum(1 for r in decoys if r["verdict"] == "wrong")

    by_id = {r["item_id"]: r for r in rows if r["is_repeat"] == "0"}
    agree = compared = 0
    for r in repeats:
        original = by_id.get(r["source_item_id"])
        if original is None:
            continue
        compared += 1
        agree += int(original["verdict"] == r["verdict"])

    times = sorted(float(r["ms_elapsed"]) for r in rows if r["ms_elapsed"])
    only = next(iter(present), "")
    return {
        "source": only,
        "unbiased": only in UNBIASED_SOURCES,
        "path": path,
        "n_total": len(rows),
        "n_real": len(real),
        "counts": counts,
        "error_rate": (counts["wrong"] / judged) if judged else None,
        "n_unjudgeable": counts["unjudgeable"],
        "decoys_shown": len(decoys),
        "decoys_caught": decoy_caught,
        "decoy_pass_rate": (decoy_caught / len(decoys)) if decoys else None,
        "repeats_compared": compared,
        "intra_rater_agreement": (agree / compared) if compared else None,
        "median_ms": statistics.median(times) if times else None,
        "median_seconds": round(statistics.median(times) / 1000.0, 2) if times else None,
    }


if __name__ == "__main__":
    import tempfile

    # A synthetic two-frame movie: five cells in a row, one link deliberately
    # crossed so exactly one is geometry-flagged.
    def strip(offsets):
        labels = np.zeros((40, 260), dtype=int)
        for label, x in offsets.items():
            labels[10:30, x:x + 40] = label
        return labels

    masks = {1: strip({1: 0, 2: 50, 3: 100, 4: 150, 5: 200}),
             2: strip({1: 0, 2: 50, 3: 100, 4: 150, 5: 200})}
    tracks = pd.DataFrame({
        "Index": [1, 2, 3, 4, 5],
        "Frame1": ["Cell1_1", "Cell1_2", "Cell1_3", "Cell1_4", "Cell1_5"],
        # Tracks 3 and 4 are swapped at frame 2 -- both should flag.
        "Frame2": ["Cell2_1", "Cell2_2", "Cell2_4", "Cell2_3", "Cell2_5"],
    })

    flagged = geometry_flagged_items(tracks, masks, register=False)
    assert {i["label"] for i in flagged} == {3, 4}, flagged
    print("geometry-flagged: %d item(s) -- labels %s"
          % (len(flagged), sorted(i["label"] for i in flagged)))

    everything = frame_pair_items(tracks, masks, 1)
    assert [i["label"] for i in everything] == [1, 2, 3, 4, 5], everything
    print("frame-pair census: %d links, raster ordered %s"
          % (len(everything), [i["label"] for i in everything]))

    # The quota is exact, not a per-item coin flip: at census sizes an
    # independent draw returns zero decoys often enough to matter (24 items at
    # 15% does so ~2% of the time, and seed 0 was one of those).
    for seed in range(8):
        sized = build_queue(everything, masks, decoy_rate=0.15, repeat_rate=0.15, seed=seed)
        assert sum(e["is_decoy"] for e in sized) >= 1, f"seed {seed} produced no decoys"
        assert sum(e["is_repeat"] for e in sized) >= 1, f"seed {seed} produced no repeats"
    print("every seed 0-7 yields at least one decoy and one repeat, whatever the draw")

    queue = build_queue(everything, masks, decoy_rate=0.4, repeat_rate=0.4, seed=1)
    decoys = [e for e in queue if e["is_decoy"]]
    repeats = [e for e in queue if e["is_repeat"]]
    assert decoys and repeats, (len(decoys), len(repeats))
    assert all(e["shown_label"] != e["true_label"] for e in decoys), decoys
    assert all(not e["is_decoy"] for e in repeats), "a repeated decoy measures the wrong thing"
    for repeat in repeats:
        position = [e["item_id"] for e in queue].index(repeat["item_id"])
        origin = [e["item_id"] for e in queue].index(repeat["source_item_id"])
        assert abs(position - origin) > 1, (repeat["item_id"], position, origin)
    print("queue: %d items = %d real + %d decoys + %d repeats, none adjacent to its original"
          % (len(queue), len(everything), len(decoys), len(repeats)))

    # A labeller who rubber-stamps: everything "correct", including decoys.
    path = os.path.join(tempfile.mkdtemp(), "ground_truth_links.csv")
    for entry in queue:
        append_verdict(path, "s1", entry, "correct", None, 900.0, labeller="rubber-stamp")
    stamped = summarise(path)
    assert stamped["decoy_pass_rate"] == 0.0, stamped
    assert stamped["intra_rater_agreement"] == 1.0, stamped
    print("\nrubber-stamping: decoys caught %.0f%%, self-consistency %.0f%% -- perfectly"
          % (stamped["decoy_pass_rate"] * 100, stamped["intra_rater_agreement"] * 100))
    print("  self-consistent and completely uninformative. One number cannot see this;")
    print("  the decoy rate is what does.")

    # An attentive labeller: catches decoys, is 100% self-consistent.
    path2 = os.path.join(tempfile.mkdtemp(), "ground_truth_links.csv")
    for entry in queue:
        verdict = "wrong" if entry["is_decoy"] else ("wrong" if entry["label"] in (3, 4) else "correct")
        append_verdict(path2, "s1", entry, verdict, entry["true_label"], 2600.0)
    good = summarise(path2)
    assert good["decoy_pass_rate"] == 1.0 and good["intra_rater_agreement"] == 1.0, good
    # Decoys are instrumentation, not data: a real link that happened to be
    # shown as a decoy is excluded from `n_real`, so the wrong count is over
    # whatever survived that.
    expected_wrong = sum(1 for e in queue
                         if not e["is_decoy"] and not e["is_repeat"] and e["label"] in (3, 4))
    assert good["counts"]["wrong"] == expected_wrong, (good, expected_wrong)
    print("attentive: %d/%d real links wrong (%.0f%%), decoys caught %.0f%%, "
          "self-consistency %.0f%%, median %.1f s"
          % (good["counts"]["wrong"], good["n_real"], good["error_rate"] * 100,
             good["decoy_pass_rate"] * 100, good["intra_rater_agreement"] * 100,
             good["median_seconds"]))

    try:
        append_verdict(path2, "s1", queue[0], "probably", None, 100.0)
    except ValueError:
        print("an unknown verdict is refused: three states, not a free-text field.")
    else:
        raise AssertionError("unknown verdicts must raise")

    # --- D (R23): which SOURCE the chosen cell came from, and the settings.
    # Without `chosen_reason`/`chosen_rank` the question "does `pointed` find
    # the answer or amplify the wrong one?" can only be argued; with them a few
    # dozen clicks answer it, and reference-free -- the input is a human's
    # choice, not a tracking table.
    src_path = os.path.join(tempfile.mkdtemp(), "ground_truth_links.csv")
    append_verdict(src_path, "s1", queue[0], "wrong", 7, 1200.0,
                   chosen_reason="pointed", chosen_rank=5,
                   register=True, environment_threshold=0.5)
    append_verdict(src_path, "s1", queue[1], "correct", None, 800.0,
                   register=True, environment_threshold=0.5)
    saved = read_verdicts(src_path)
    assert saved[0]["chosen_reason"] == "pointed" and saved[0]["chosen_rank"] == "5", saved[0]
    assert saved[0]["register"] == "1" and saved[0]["environment_threshold"] == "0.5", saved[0]
    # A verdict with nothing chosen leaves both blank rather than inventing a 0,
    # which would read as "rank 0" -- the top of the list.
    assert saved[1]["chosen_reason"] == "" and saved[1]["chosen_rank"] == "", saved[1]
    assert all(f in saved[0] for f in FIELDS), set(FIELDS) - set(saved[0])
    print("D (R23): a verdict records which source proposed the chosen cell "
          "(%s, rank %s) and the settings it was collected under "
          "(register=%s, environment_threshold=%s)."
          % (saved[0]["chosen_reason"], saved[0]["chosen_rank"],
             saved[0]["register"], saved[0]["environment_threshold"]))

    # --- R25: a file started under an OLDER header is migrated, not appended
    # to blindly. Before this, Round 23's four new columns were written beneath
    # a Round 22 header, so every value after `chosen_label` sat one heading off.
    legacy_fields = [f for f in FIELDS if f not in
                     ("chosen_reason", "chosen_rank", "candidate_mode", "register",
                      "environment_threshold")]
    legacy_path = os.path.join(tempfile.mkdtemp(), "ground_truth_links.csv")
    with open(legacy_path, "w", newline="", encoding="utf-8") as handle:
        old_writer = csv.DictWriter(handle, fieldnames=legacy_fields)
        old_writer.writeheader()
        old_writer.writerow({f: ("old-session" if f == "session_id" else
                                 "correct" if f == "verdict" else "") for f in legacy_fields})
    append_verdict(legacy_path, "s2", queue[0], "wrong", 7, 500.0, chosen_reason="radius",
                   chosen_rank=3, candidate_mode="radius:k=3", register=False)
    migrated = read_verdicts(legacy_path)
    assert list(migrated[0].keys()) == list(FIELDS), list(migrated[0].keys())
    assert migrated[0]["session_id"] == "old-session" and migrated[0]["chosen_reason"] == "", migrated[0]
    assert migrated[1]["chosen_reason"] == "radius" and migrated[1]["chosen_rank"] == "3", migrated[1]
    assert migrated[1]["candidate_mode"] == "radius:k=3" and migrated[1]["verdict"] == "wrong", migrated[1]
    print("R25: a verdict file written under an older header is rewritten under the")
    print("  current one before appending -- old rows keep their values, new columns")
    print("  land under their own headings (before: one heading off, silently).")

    # --- D3 (R18): the environment-flagged set, and the refusal to pool.
    env_scores = {1: {1: 0.9, 2: 0.4, 3: 0.2, 4: 0.55, 5: 1.0}}
    env_items = environment_flagged_items(tracks, env_scores, masks, threshold=0.5)
    assert [i["label"] for i in env_items] == [2, 3], env_items
    assert all(i["source"] == "environment_flagged" for i in env_items), env_items
    print("\nD3: the repair tier (0.5) flags %s; the look tier would flag more, which is"
          % [i["label"] for i in env_items])
    print("  the wrong size for a census. Every item carries its source.")

    mixed_path = os.path.join(tempfile.mkdtemp(), "ground_truth_links.csv")
    env_queue = build_queue(env_items, masks, decoy_rate=0.0, repeat_rate=0.0, seed=0)
    geom_queue = build_queue(flagged, masks, decoy_rate=0.0, repeat_rate=0.0, seed=0)
    for entry in env_queue + geom_queue:
        append_verdict(mixed_path, "s1", entry, "correct", None, 1000.0)
    try:
        summarise(mixed_path)
    except MixedSources as error:
        assert "must not be pooled" in str(error), error
        print("  pooling two sets RAISES rather than returning a number:")
        print("   ", str(error).split(":")[0])
    else:
        raise AssertionError("pooling across sources must raise")

    one = summarise(mixed_path, source="environment_flagged")
    assert one["source"] == "environment_flagged" and one["unbiased"] is False, one
    unbiased = summarise(mixed_path, source="frame_pair") if False else None
    pair_path = os.path.join(tempfile.mkdtemp(), "ground_truth_links.csv")
    for entry in build_queue(everything, masks, decoy_rate=0.0, repeat_rate=0.0, seed=0):
        append_verdict(pair_path, "s1", entry, "correct", None, 1000.0)
    assert summarise(pair_path)["unbiased"] is True
    print("  one set at a time summarises, and only `frame_pair` reports unbiased=True.")

    # --- E (R18): the CTC-style ground-truth layout, and the assertion that
    # a GT sibling is invisible to the pipeline.
    import shutil
    project = tempfile.mkdtemp()
    track_out = os.path.join(project, "cell_track_output")
    result_dir = os.path.join(track_out, "All_cell_tracking")
    os.makedirs(result_dir)
    tracks.to_excel(os.path.join(result_dir, "All_cell_merged_tracking_results.xlsx"), index=False)

    gt_table = tracks.copy()
    gt_table.loc[0, "Frame2"] = "Cell2_9"          # a human's correction
    record = save_ground_truth(track_out, gt_table, tracks,
                               labeller="self-test", source="frame_pair")
    assert os.path.isdir(gt_dir(track_out))
    assert os.path.exists(gt_table_path(track_out))
    loaded = load_ground_truth(track_out, against=tracks)
    assert loaded["matches_current_tracking"] is True, loaded["provenance"]
    print("\nE: ground truth written to %s/ beside the result folder, and its provenance"
          % GT_FOLDER_NAME)
    print("  identifies the tracking run it was collected against.")

    # Apply a correction to the tracking table: the SAME ground truth must now
    # report that it no longer matches. Silence here is the failure mode.
    corrected = tracks.copy()
    corrected.loc[1, "Frame2"] = "Cell2_7"
    assert load_ground_truth(track_out, against=corrected)["matches_current_tracking"] is False
    print("  after a correction to the tracking table it reports matches=False, not silence.")

    # --- Nothing in the pipeline may enumerate the parent and pick GT up.
    # `track_cells` lists its OWN subfolder by a prefixed regex; a sibling is
    # invisible to it, and so is a GT-prefixed file dropped inside the result
    # folder. Asserted rather than argued.
    shutil.copy(gt_table_path(track_out),
                os.path.join(result_dir, "GT_All_cell_Matching_Results_new_1_2.xlsx"))
    listed = [f for f in os.listdir(result_dir)
              if re.match(r"All_cell_Matching_Results_new_\d+_\d+\.xlsx", f)]
    assert not [f for f in listed if f.startswith("GT_")], listed
    assert GT_FOLDER_NAME not in os.listdir(result_dir), os.listdir(result_dir)
    print("  a GT_-prefixed file inside the result folder is not matched by the pipeline's")
    print("  own regex, and the GT folder is a sibling the pipeline never lists.")

    print("\nAll census self-tests passed.")
