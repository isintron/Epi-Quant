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
import os
import random
import statistics
from datetime import datetime, timezone
from typing import Mapping, Optional

import numpy as np
import pandas as pd

import cell_tracking
import tracking_benchmark

VERDICTS = ("correct", "wrong", "unjudgeable")

FIELDS = (
    "session_id", "item_id", "source_item_id", "frame", "label", "track_id",
    "shown_label", "true_label", "is_decoy", "is_repeat", "verdict",
    "chosen_label", "ms_elapsed", "labeller", "timestamp_utc",
)

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


def append_verdict(path: str, session_id: str, item: dict, verdict: str,
                   chosen_label: Optional[int], ms_elapsed: float,
                   labeller: str = "") -> dict:
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
        "ms_elapsed": round(float(ms_elapsed), 1),
        "labeller": labeller,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    }
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    exists = os.path.exists(path)
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


def summarise(path: str, session_id: Optional[str] = None) -> dict:
    """Counts, decoy pass-rate, intra-rater agreement, and the median seconds
    per judgment -- the three numbers that qualify every figure derived from
    these labels.

    `real` counts exclude decoys and repeats: those are instrumentation, not
    data, and pooling them into the rate is the same error as pooling two
    designs that answer different questions.
    """
    rows = [r for r in read_verdicts(path)
            if session_id is None or r["session_id"] == session_id]
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
    return {
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

    print("\nAll census self-tests passed.")
