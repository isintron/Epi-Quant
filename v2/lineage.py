"""Cell dynamics: division, delamination, and the per-track lineage sheet.

Answers the reviewer point that Epi-Quant was never demonstrated on tissues
with dividing or intercalating cells. Rearrangement (T1) already lives in
`topology.py` and is reused here rather than duplicated; this module adds the
other two event types and the lineage table that ties them together.

**Everything here produces CANDIDATES, not verdicts.** That is deliberate, for
two reasons found while building it:

1. *Division has no reference implementation to copy.* TissueMiner does not
   infer divisions -- Tissue Analyzer paints daughter cells into
   `dividing_cells.tif` and TissueMiner reads that mask
   (`DeformingTissue.cpp:228`, `pv == Pixel::DividingCellValue`). Without TA it
   reports zero divisions. Only the mother-daughter *pairing* is inference, and
   that is what is ported here. Our detection is therefore geometric and
   unvalidated.
2. *Delamination cannot be told from a tracking break by overlap alone.* In
   TissueMiner, `Apoptosis` is a residual bucket: whatever is left after
   "stayed" and "moved out of mask" are resolved
   (`DeformingTissue.cpp:516-522`). Both a delaminating cell and a broken track
   end with the cell's territory absorbed by its neighbours, so overlap does
   not separate them -- measured on `embryos_GBE_GAP43`, 308 of 414 tracks end
   in the tissue interior, far more than plausible biology. What does carry
   information is the **area trajectory** before the event, so a shrinking-area
   term is required here that TissueMiner does not have.

The intended workflow is therefore: detect candidates with their evidence, let
a human confirm or reject them in the Correct Tracking view, and build the
lineage sheet from what was confirmed.

Portions modelled on TissueMiner (BSD-3-Clause, (c) 2015 Scientific Computing
Facility @ MPI-CBG); see `third_party/tissue_miner_LICENSE` and the note in
`topology.py`.
"""

from typing import Callable, Dict, Mapping, Optional

import numpy as np
import pandas as pd

# `appears_by` / `disappears_by` vocabularies, mirroring TissueMiner's
# `gainDict`/`lossDict` (`db/CreateDbFromParser.R:332,343`) so results are
# comparable with theirs.
#
# One deliberate divergence: their loss code 5 maps to the literal
# "SegErrAppearance" -- the SAME string as gain code 4 -- so a segmentation
# error on disappearance is indistinguishable from one on appearance in their
# output. That is a bug, not a convention worth reproducing.
APPEARS_BY = ("Stayed", "Division", "MovedIntoMask", "SegErrAppearance")
DISAPPEARS_BY = ("Stays", "Division", "Delamination", "MovesOutOfMask", "SegErrDisappearance")

# A division's two daughters should together account for roughly the mother's
# area. Generous bounds: segmentation of a just-divided pair is unreliable, and
# this only gates a candidate a human will review.
_DIVISION_AREA_RATIO = (0.55, 1.6)
# ...and each daughter should sit within this multiple of the mother's own
# radius of where the mother was.
_DIVISION_RADIUS_MULTIPLE = 2.0


def track_spans(tracking_data: pd.DataFrame) -> Dict[int, dict]:
    """`{track_id: {"first": f, "last": f, "cells": {frame: cell_index}}}`.

    `last` is the last frame with a non-null claim, NOT the last frame column.
    `_merge_tracking_pairs` emits one column more than it has data for
    (`range(start, start + count + 1)`, so 30 frames of pairs yield a
    `Frame31`), and reading the column list instead would push every track's
    end one frame into a column that cannot contain anything.
    """
    id_col = tracking_data.columns[0]
    frame_cols = [(int(c[len("Frame"):]), c) for c in tracking_data.columns if c.startswith("Frame")]
    frame_cols.sort()

    spans: Dict[int, dict] = {}
    for _, row in tracking_data.iterrows():
        cells = {frame: row[col] for frame, col in frame_cols if pd.notna(row[col])}
        if not cells:
            continue
        spans[int(row[id_col])] = {
            "first": min(cells), "last": max(cells), "cells": cells,
        }
    return spans


def _raw_label(cell_index: str) -> int:
    return int(str(cell_index).rsplit("_", 1)[1])


def division_candidates(
    spans: Mapping[int, dict],
    metrics: Callable[[str], Optional[tuple]],
    neighbours: Callable[[int], Optional[Mapping]],
    last_frame: int,
) -> list:
    """Tracks that look like a mother dividing into two daughters.

    `metrics(cell_index)` -> `(x, y, area, radius)` or `None`;
    `neighbours(frame)` -> `{label: {label, ...}}` or `None`.

    A candidate needs all of: the mother's track ends at `f`; two tracks begin
    at `f + 1`; those two are neighbours of each other; both lie within
    `_DIVISION_RADIUS_MULTIPLE` of the mother's last position; and their areas
    sum to within `_DIVISION_AREA_RATIO` of the mother's.

    Each candidate carries its evidence, so the reviewer sees *why* it was
    flagged rather than being asked to trust a verdict.
    """
    births: Dict[int, list] = {}
    for track_id, span in spans.items():
        births.setdefault(span["first"], []).append(track_id)

    candidates = []
    for mother, span in spans.items():
        end = span["last"]
        if end >= last_frame:
            continue  # nothing follows; cannot be judged
        mother_metrics = metrics(span["cells"][end])
        if mother_metrics is None:
            continue
        mx, my, m_area, m_radius = mother_metrics

        graph = neighbours(end + 1)
        newborns = births.get(end + 1, [])
        if graph is None or len(newborns) < 2:
            continue

        nearby = []
        for track_id in newborns:
            info = metrics(spans[track_id]["cells"][end + 1])
            if info is None:
                continue
            distance = float(np.hypot(info[0] - mx, info[1] - my))
            if distance <= _DIVISION_RADIUS_MULTIPLE * max(m_radius, 1.0):
                nearby.append((track_id, info, distance))

        for i in range(len(nearby)):
            for j in range(i + 1, len(nearby)):
                (a_id, a_info, a_dist), (b_id, b_info, b_dist) = nearby[i], nearby[j]
                a_label = _raw_label(spans[a_id]["cells"][end + 1])
                b_label = _raw_label(spans[b_id]["cells"][end + 1])
                if b_label not in graph.get(a_label, set()):
                    continue  # daughters of one division must touch
                area_ratio = (a_info[2] + b_info[2]) / max(m_area, 1.0)
                if not (_DIVISION_AREA_RATIO[0] <= area_ratio <= _DIVISION_AREA_RATIO[1]):
                    continue
                candidates.append({
                    "type": "division",
                    "frame": end,
                    "mother": mother,
                    "daughters": (a_id, b_id),
                    "evidence": {
                        "area_ratio": round(area_ratio, 3),
                        "daughter_distances": (round(a_dist, 1), round(b_dist, 1)),
                        "mother_radius": round(m_radius, 1),
                        "daughters_adjacent": True,
                    },
                })
    return candidates


def delamination_candidates(
    spans: Mapping[int, dict],
    metrics: Callable[[str], Optional[tuple]],
    is_margin: Callable[[str], bool],
    last_frame: int,
    explained: Optional[set] = None,
    shrink_frames: int = 3,
    typical_area: Optional[float] = None,
    final_area_fraction: float = 0.5,
) -> list:
    """Tracks that end in the tissue interior with the cell shrinking away.

    Resolution order matters and follows TissueMiner: a cell reachable from the
    tissue margin is `MovesOutOfMask` and is never considered here, because
    that test runs *first* (`DeformingTissue.cpp:488-501` before `:516-522`).
    Getting the margin definition wrong therefore changes what is called a
    delamination, which is the single largest portability risk in the port.

    Where this deliberately differs: TissueMiner's bucket has no positive
    evidence at all -- anything unexplained becomes `Apoptosis`. On real data
    that is indistinguishable from a broken track, since both end with the
    cell's territory absorbed by neighbours.

    Two positive terms are required instead. The area must have been
    **shrinking** over the preceding `shrink_frames`, AND the cell must end
    **small** -- below `final_area_fraction` of `typical_area` (the population
    median). The second term is not redundant, and measurement is why it is
    here: on `embryos_GBE_GAP43` the slope test alone accepted 100 tracks whose
    median final area was 400 px against a population median of 380 -- 99% of
    them ended at more than half the typical cell size, i.e. they vanished at
    full size, which is a broken track rather than a cell leaving the
    epithelium. A cell that genuinely delaminates shrinks to nothing first.

    With `typical_area=None` the size term is skipped, reproducing the
    slope-only behaviour.
    """
    explained = explained or set()
    candidates = []
    for track_id, span in spans.items():
        end = span["last"]
        if end >= last_frame or track_id in explained:
            continue
        final_cell = span["cells"][end]
        if is_margin(final_cell):
            continue  # left the field of view, not a delamination

        areas = []
        for frame in range(max(span["first"], end - shrink_frames + 1), end + 1):
            info = metrics(span["cells"].get(frame, "")) if frame in span["cells"] else None
            if info is not None:
                areas.append(info[2])
        if len(areas) < 2:
            continue

        # Least-squares slope in px^2 per frame; negative means shrinking.
        x = np.arange(len(areas), dtype=float)
        slope = float(np.polyfit(x, np.array(areas, dtype=float), 1)[0])
        if slope >= 0:
            continue
        final_area = areas[-1]
        if typical_area and final_area >= final_area_fraction * typical_area:
            continue
        candidates.append({
            "type": "delamination",
            "frame": end,
            "track": track_id,
            "evidence": {
                "area_slope_px_per_frame": round(slope, 1),
                "final_area": round(final_area, 1),
                "final_area_vs_typical": round(final_area / typical_area, 2) if typical_area else None,
                "area_change": round(areas[-1] - areas[0], 1),
                "frames_measured": len(areas),
            },
        })
    return candidates


def _lineage_groups(spans: Mapping[int, dict], mother_of: Mapping[int, int]) -> tuple:
    """`(group_of, generation_of)` -- weakly-connected components over the
    mother->daughter edges, and depth from each component's root.

    Mirrors TissueMiner's `RevealLineage.R` (igraph `clusters()` on the
    division forest, then shortest-path depth minus 1 so a never-dividing cell
    is generation 0). Implemented with a plain union-find because the edge set
    is a forest -- no need for a graph library.
    """
    parent = {track: track for track in spans}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for daughter, mother in mother_of.items():
        if daughter in parent and mother in parent:
            parent[find(daughter)] = find(mother)

    group_of = {track: f"lg_{find(track)}" for track in spans}

    generation_of = {}
    for track in spans:
        depth, cursor, seen = 0, track, set()
        while cursor in mother_of and cursor not in seen:
            seen.add(cursor)
            cursor = mother_of[cursor]
            depth += 1
        generation_of[track] = depth
    return group_of, generation_of


def build_lineage_table(
    tracking_data: pd.DataFrame,
    confirmed: Optional[list] = None,
    margin_tracks: Optional[set] = None,
    last_frame: Optional[int] = None,
) -> pd.DataFrame:
    """The per-track lineage sheet, in TissueMiner's `cell_histories` shape
    (`db/CreateDbFromParser.R:513-532`).

    `confirmed` is the list of events a human accepted -- division candidates
    contribute the mother/daughter links and set both sides' `appears_by` /
    `disappears_by` to `"Division"`; delamination events set
    `disappears_by = "Delamination"`. Everything else falls back to the
    positional vocabulary: a track present to the end `Stays`, one starting at
    frame 1 `Stayed`, and a track that begins or ends elsewhere without an
    explanation is a segmentation/tracking error rather than biology.
    """
    spans = track_spans(tracking_data)
    if not spans:
        return pd.DataFrame(columns=[
            "track_id", "first_frame", "last_frame", "mother_track_id",
            "left_daughter_track_id", "right_daughter_track_id",
            "appears_by", "disappears_by", "lineage_group", "generation",
        ])

    first_frame = min(s["first"] for s in spans.values())
    if last_frame is None:
        last_frame = max(s["last"] for s in spans.values())
    margin_tracks = margin_tracks or set()

    mother_of: Dict[int, int] = {}
    daughters_of: Dict[int, tuple] = {}
    delaminated = set()
    for event in confirmed or []:
        if event["type"] == "division":
            a, b = event["daughters"]
            daughters_of[event["mother"]] = (a, b)
            mother_of[a] = event["mother"]
            mother_of[b] = event["mother"]
        elif event["type"] == "delamination":
            delaminated.add(event["track"])

    group_of, generation_of = _lineage_groups(spans, mother_of)

    rows = []
    for track_id, span in sorted(spans.items()):
        if track_id in mother_of:
            appears = "Division"
        elif span["first"] == first_frame:
            appears = "Stayed"
        elif track_id in margin_tracks:
            appears = "MovedIntoMask"
        else:
            appears = "SegErrAppearance"

        if track_id in daughters_of:
            disappears = "Division"
        elif track_id in delaminated:
            disappears = "Delamination"
        elif span["last"] >= last_frame:
            disappears = "Stays"
        elif track_id in margin_tracks:
            disappears = "MovesOutOfMask"
        else:
            disappears = "SegErrDisappearance"

        left, right = daughters_of.get(track_id, (None, None))
        rows.append({
            "track_id": track_id,
            "first_frame": span["first"],
            "last_frame": span["last"],
            "mother_track_id": mother_of.get(track_id),
            "left_daughter_track_id": left,
            "right_daughter_track_id": right,
            "appears_by": appears,
            "disappears_by": disappears,
            "lineage_group": group_of[track_id],
            "generation": generation_of[track_id],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # --- a division: mother track 1 ends at frame 2, daughters 2 and 3 start
    # at frame 3, adjacent to each other, areas summing to the mother's.
    tracking = pd.DataFrame({
        "Index": [1, 2, 3, 4],
        "Frame1": ["Cell1_10", None, None, "Cell1_20"],
        "Frame2": ["Cell2_10", None, None, "Cell2_20"],
        "Frame3": [None, "Cell3_11", "Cell3_12", "Cell3_20"],
        "Frame4": [None, "Cell4_11", "Cell4_12", "Cell4_20"],
    })
    geometry = {
        "Cell2_10": (10.0, 10.0, 400.0, 12.0),
        "Cell3_11": (8.0, 6.0, 190.0, 8.0),
        "Cell3_12": (12.0, 14.0, 200.0, 8.0),
        "Cell2_20": (90.0, 90.0, 380.0, 12.0),
        "Cell3_20": (90.0, 90.0, 380.0, 12.0),
    }
    graphs = {3: {11: {12}, 12: {11}, 20: set()}}
    spans = track_spans(tracking)
    print("track spans:", {k: (v["first"], v["last"]) for k, v in spans.items()})
    assert spans[1]["last"] == 2, spans[1]
    assert spans[4]["last"] == 4, "must read the last non-null claim, not the last column"

    divisions = division_candidates(
        spans, geometry.get, graphs.get, last_frame=4,
    )
    print("division candidates:", divisions)
    assert len(divisions) == 1, divisions
    assert divisions[0]["mother"] == 1 and set(divisions[0]["daughters"]) == {2, 3}, divisions
    print("  -> one candidate, correct mother and both daughters, with evidence.")

    # --- delamination: a cell shrinking away in the interior.
    shrink_tracking = pd.DataFrame({
        "Index": [7],
        "Frame1": ["Cell1_5"], "Frame2": ["Cell2_5"], "Frame3": ["Cell3_5"], "Frame4": [None],
    })
    shrink_geo = {
        "Cell1_5": (50.0, 50.0, 400.0, 12.0),
        "Cell2_5": (50.0, 50.0, 260.0, 9.0),
        "Cell3_5": (50.0, 50.0, 90.0, 5.0),
    }
    interior = delamination_candidates(
        track_spans(shrink_tracking), shrink_geo.get, lambda c: False, last_frame=4,
        typical_area=400.0,
    )
    print("\ndelamination candidates (interior):", interior)
    assert len(interior) == 1 and interior[0]["track"] == 7, interior
    assert interior[0]["evidence"]["area_slope_px_per_frame"] < 0, interior

    # The SAME cell at the margin must not be a delamination -- the margin test
    # runs first, exactly as in TissueMiner.
    at_margin = delamination_candidates(
        track_spans(shrink_tracking), shrink_geo.get, lambda c: True, last_frame=4,
        typical_area=400.0,
    )
    assert at_margin == [], at_margin
    print("  -> margin cells are excluded before the delamination test, as TissueMiner orders it.")

    # A cell that vanishes at CONSTANT area is a tracking break, not a
    # delamination -- this is the term TissueMiner's residual bucket lacks.
    steady_geo = {"Cell1_5": (50.0, 50.0, 400.0, 12.0), "Cell2_5": (50.0, 50.0, 405.0, 12.0),
                  "Cell3_5": (50.0, 50.0, 402.0, 12.0)}
    assert delamination_candidates(
        track_spans(shrink_tracking), steady_geo.get, lambda c: False, last_frame=4,
        typical_area=400.0,
    ) == [], "a cell vanishing at full size is a broken track, not a delamination"
    print("  -> a cell vanishing at constant area is rejected (broken track, not biology).")

    # Shrinking, but still large at the end -- the case that dominated the real
    # data. The slope alone accepts it; the size term must reject it.
    big_end = {"Cell1_5": (50.0, 50.0, 460.0, 13.0), "Cell2_5": (50.0, 50.0, 430.0, 12.0),
               "Cell3_5": (50.0, 50.0, 400.0, 12.0)}
    assert delamination_candidates(
        track_spans(shrink_tracking), big_end.get, lambda c: False, last_frame=4,
    ), "slope-only must accept it (this is the permissive behaviour we are fixing)"
    assert delamination_candidates(
        track_spans(shrink_tracking), big_end.get, lambda c: False, last_frame=4,
        typical_area=400.0,
    ) == [], "a cell still at full size when it vanishes is not a delamination"
    print("  -> shrinking-but-still-large is rejected once the size term is applied.")

    # --- lineage table from the confirmed division.
    table = build_lineage_table(tracking, confirmed=divisions, last_frame=4)
    print("\nlineage table:\n", table.to_string(index=False))
    row1 = table[table.track_id == 1].iloc[0]
    assert row1.disappears_by == "Division" and row1.left_daughter_track_id == 2, row1
    for daughter in (2, 3):
        r = table[table.track_id == daughter].iloc[0]
        assert r.appears_by == "Division" and r.mother_track_id == 1, r
        assert r.generation == 1, r
    assert table[table.track_id == 1].iloc[0].generation == 0
    assert len(set(table[table.track_id.isin([1, 2, 3])].lineage_group)) == 1, "the family shares one group"
    assert table[table.track_id == 4].iloc[0].lineage_group != row1.lineage_group, "unrelated track is its own group"
    print("  -> mother/daughters share one lineage group; generations 0, 1, 1.")

    print("\nAll lineage self-tests passed.")
