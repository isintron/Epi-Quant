"""What the human actually looked at in the Correct Tracking view.

Without this, a cell that was displayed, inspected and deliberately left alone
is indistinguishable from a cell nobody ever saw -- which is exactly why no
ground truth has ever come out of the correction tool. The corrections it
produces are labelled errors; the *absence* of a correction is only evidence if
you know the cell was on screen and examined.

Three event kinds, appended to one CSV:

- `frame_shown`   -- a frame was displayed, with the zoom and overlay mode it
                     was displayed at, and how long it stayed up.
- `cell_inspected`-- the user clicked a cell, so its identity was examined.
- `cell_changed`  -- a correction was applied to a cell.

`summarise()` turns that into the distinction the ground truth needs:
`inspected_unchanged` (a weak positive) versus `never_displayed` (no evidence
at all). A frame that was shown for under `min_dwell_ms` is reported as
`glimpsed` and its cells are NOT counted as weak positives -- paging through a
movie is not inspection.

This log is evidence about a *person*, not a measurement of the tracker. It
records what was on screen and what was clicked; it does not infer that
anything was correct. The weak positives it yields are weak on purpose.
"""

from __future__ import annotations

import csv
import os
import uuid
from datetime import datetime, timezone
from typing import Iterable, Optional

FIELDS = (
    "session_id", "timestamp_utc", "event", "frame", "cell_index", "track_id",
    "ms_elapsed", "zoom_percent", "render_mode", "color_mode", "detail",
)

EVENTS = ("frame_shown", "cell_inspected", "cell_changed")

# Below this, a frame was paged past rather than looked at. 1.5 s is not a
# measured threshold -- it is a stated default, and `summarise` takes it as an
# argument so a session can be re-read at a different bar.
DEFAULT_MIN_DWELL_MS = 1500


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def append(path: str, session_id: str, event: str, frame: int,
           cell_index: Optional[str] = None, track_id: Optional[int] = None,
           ms_elapsed: Optional[float] = None, zoom_percent: Optional[float] = None,
           render_mode: str = "", color_mode: str = "", detail: str = "") -> dict:
    """Append one event. Creates the file with a header if it does not exist."""
    if event not in EVENTS:
        raise ValueError(f"unknown coverage event {event!r}; expected one of {EVENTS}")
    row = {
        "session_id": session_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "event": event,
        "frame": int(frame),
        "cell_index": cell_index or "",
        "track_id": "" if track_id is None else int(track_id),
        "ms_elapsed": "" if ms_elapsed is None else round(float(ms_elapsed), 1),
        "zoom_percent": "" if zoom_percent is None else round(float(zoom_percent), 1),
        "render_mode": render_mode or "",
        "color_mode": color_mode or "",
        "detail": detail or "",
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


def read(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarise(path: str, cells_by_frame: dict,
              min_dwell_ms: float = DEFAULT_MIN_DWELL_MS) -> dict:
    """Per-frame coverage against the cells that actually exist.

    `cells_by_frame` is `{frame: iterable_of_cell_index}` -- the ground set. A
    cell absent from it cannot be covered, and a frame absent from it is
    reported under `unknown_frames` rather than silently dropped.

    Returns per frame: `displayed`, `dwell_ms`, `glimpsed`, and the four
    disjoint cell sets `changed`, `inspected_unchanged`, `displayed_only`,
    `never_displayed`. The first two are evidence; the last two are not.
    """
    rows = read(path)
    dwell: dict = {}
    inspected: dict = {}
    changed: dict = {}
    zooms: dict = {}
    unknown_frames = set()

    for row in rows:
        try:
            frame = int(row["frame"])
        except (TypeError, ValueError):
            continue
        if frame not in cells_by_frame:
            unknown_frames.add(frame)
        event = row["event"]
        if event == "frame_shown":
            if row["ms_elapsed"]:
                dwell[frame] = dwell.get(frame, 0.0) + float(row["ms_elapsed"])
            else:
                dwell.setdefault(frame, 0.0)
            if row["zoom_percent"]:
                zooms.setdefault(frame, []).append(float(row["zoom_percent"]))
        elif event == "cell_inspected" and row["cell_index"]:
            inspected.setdefault(frame, set()).add(row["cell_index"])
        elif event == "cell_changed" and row["cell_index"]:
            changed.setdefault(frame, set()).add(row["cell_index"])

    frames = {}
    for frame, cells in cells_by_frame.items():
        ground = set(cells)
        was_changed = changed.get(frame, set()) & ground
        was_inspected = (inspected.get(frame, set()) & ground) - was_changed
        displayed = frame in dwell
        glimpsed = displayed and dwell.get(frame, 0.0) < min_dwell_ms
        # A glimpsed frame yields no weak positives: paging past a frame is not
        # inspection, and counting it as such is how a coverage log turns into
        # a fabricated ground truth.
        weak = set() if glimpsed else was_inspected
        frames[frame] = {
            "displayed": displayed,
            "dwell_ms": round(dwell.get(frame, 0.0), 1),
            "glimpsed": glimpsed,
            "zoom_percent": sorted(set(zooms.get(frame, []))),
            "changed": sorted(was_changed),
            "inspected_unchanged": sorted(weak),
            "displayed_only": sorted(ground - was_changed - was_inspected) if displayed else [],
            "never_displayed": [] if displayed else sorted(ground),
        }

    return {
        "path": path,
        "events": len(rows),
        "sessions": sorted({r["session_id"] for r in rows if r.get("session_id")}),
        "min_dwell_ms": min_dwell_ms,
        "frames": frames,
        "unknown_frames": sorted(unknown_frames),
        "totals": {
            key: sum(len(frame[key]) for frame in frames.values())
            for key in ("changed", "inspected_unchanged", "displayed_only", "never_displayed")
        },
    }


def weak_positives(summary: dict) -> list:
    """`[(frame, cell_index)]` for cells displayed long enough, inspected, and
    left alone. The only labels this log produces, and they are weak: the user
    looked and did not act, which is not the same as "the user verified"."""
    return [(frame, cell) for frame, info in sorted(summary["frames"].items())
            for cell in info["inspected_unchanged"]]


if __name__ == "__main__":
    import tempfile

    path = os.path.join(tempfile.mkdtemp(), "coverage_log.csv")
    session = new_session_id()

    ground = {1: ["Cell1_1", "Cell1_2", "Cell1_3"],
              2: ["Cell2_1", "Cell2_2"],
              3: ["Cell3_1"]}

    # Frame 1: looked at properly. One cell corrected, one inspected and left,
    # one never clicked. Frame 2: paged past. Frame 3: never opened.
    append(path, session, "frame_shown", 1, ms_elapsed=8200, zoom_percent=200,
           render_mode="overlay", color_mode="identity")
    append(path, session, "cell_inspected", 1, cell_index="Cell1_1", track_id=1)
    append(path, session, "cell_inspected", 1, cell_index="Cell1_2", track_id=2)
    append(path, session, "cell_changed", 1, cell_index="Cell1_1", track_id=7,
           detail="reassign_track")
    append(path, session, "frame_shown", 2, ms_elapsed=300, zoom_percent=100)
    append(path, session, "cell_inspected", 2, cell_index="Cell2_1", track_id=3)

    summary = summarise(path, ground)
    f1, f2, f3 = summary["frames"][1], summary["frames"][2], summary["frames"][3]

    assert f1["changed"] == ["Cell1_1"], f1
    assert f1["inspected_unchanged"] == ["Cell1_2"], f1
    assert f1["displayed_only"] == ["Cell1_3"], f1
    assert f1["never_displayed"] == [], f1
    print("frame 1: changed=%s  inspected-and-left=%s  displayed-but-not-clicked=%s"
          % (f1["changed"], f1["inspected_unchanged"], f1["displayed_only"]))

    assert f2["glimpsed"] and f2["inspected_unchanged"] == [], f2
    print("frame 2: shown for %.0f ms -> glimpsed, so its click yields NO weak positive"
          % f2["dwell_ms"])

    assert f3["never_displayed"] == ["Cell3_1"] and not f3["displayed"], f3
    assert f3["displayed_only"] == [], f3
    print("frame 3: never displayed -> %s, which is not the same as unchanged"
          % f3["never_displayed"])

    # The distinction the whole file exists for.
    assert set(f1["inspected_unchanged"]).isdisjoint(f3["never_displayed"])
    assert weak_positives(summary) == [(1, "Cell1_2")], weak_positives(summary)
    print("weak positives: %s -- displayed, inspected, left alone" % (weak_positives(summary),))

    try:
        append(path, session, "cell_squinted_at", 1)
    except ValueError:
        print("an unknown event kind is refused rather than silently logged.")
    else:
        raise AssertionError("unknown events must raise")

    unknown = summarise(path, {1: ground[1]})
    assert unknown["unknown_frames"] == [2], unknown["unknown_frames"]
    print("a logged frame outside the ground set is reported, not dropped: %s"
          % unknown["unknown_frames"])

    print("\nAll coverage_log self-tests passed.")
