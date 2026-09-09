"""Why a census candidate list missed the right cell -- per source, per column.

Round 20 removed a mechanism that had been reasoned about rather than measured,
and Round 19 removed one before that. Round 21's instruction was blunt: *stop
theorising, instrument.* This is the instrument.

For every miss it records the rank of the true cell **under each source
separately**, so the answer to "which source could have found it and did not"
comes from data rather than argument:

| column | what an anomaly there would mean |
|---|---|
| `rank_reference` / `rank_iou` / `rank_nearest` / `rank_link` | that source had it, at that depth |
| `in_any_source` | separates *never generated* from *generated and dropped* |
| `distance_from_anchor` | the anchor-centroid proposal only works if this is small |
| `iou_with_anchor` | likewise for the overlap proposal |
| `claimed_at_next` | whether something is excluding an already-claimed cell |
| `anchor_area`, `centroid_inside` | a large concave cell's centroid can fall outside it |

**On the reference.** This takes whatever link table you hand it and calls the
result *agreement*, never accuracy. The maintainer's position, recorded here
because it corrects three rounds of this repo's own wording: the corrected
tracking tables kept under `baseline_data/` were **never ground truth** -- no
visual inspection was done on them -- and anything of that kind should be
regenerated from the masks by the pipeline. So a reference passed here is a
comparison target whose provenance the caller must state. `distance_gate_px`
exists because of what that costs when it is stale: Round 20 found 27 "misses"
whose reference successor sat a median 130 px from the anchor, five cell
diameters away, after the masks had been re-segmented under a table collected
on the old labels.
"""

from __future__ import annotations

import os
import statistics
from typing import Optional

import numpy as np
import pandas as pd

import cell_tracking
import census
import tracking_benchmark

#: Beyond this, a reference link is not a ranking failure -- it is a link whose
#: labels no longer mean what they meant when it was written. Three cell
#: diameters on GBE.
DEFAULT_DISTANCE_GATE_PX = 78


def links_from_table(table: pd.DataFrame) -> dict:
    """`{cell_index_at_f: cell_index_at_f+1}` for a tracking-shaped table."""
    columns = sorted((int(c[len("Frame"):]), c) for c in table.columns if c.startswith("Frame"))
    out = {}
    for _, row in table.iterrows():
        for (n1, c1), (n2, c2) in zip(columns, columns[1:]):
            if n2 == n1 + 1 and pd.notna(row[c1]) and pd.notna(row[c2]):
                out[str(row[c1])] = str(row[c2])
    return out


def probe_item(api, item, want_label: int, drift=(0.0, 0.0)) -> dict:
    """Every column above, for one item and one expected successor label."""
    frame, anchor = int(item["frame"]), int(item["label"])
    before = (api._load_frame(frame) or (None,))[0]
    after = (api._load_frame(frame + 1) or (None,))[0]
    if before is None or after is None:
        return {}

    moved = before
    if drift and (drift[0] or drift[1]):
        moved = tracking_benchmark.shift_labels(before, drift[0], drift[1])
    overlaps = cell_tracking.mask_overlap_fractions(moved, after)
    by_iou = [int(b) for _v, b in
              sorted(((v, b) for (a, b), v in overlaps.items() if a == anchor), reverse=True)]

    centres_after = census._centroids(after)
    centres_before = census._centroids(before)
    origin = centres_before.get(anchor)
    if origin is not None:
        origin = (origin[0] + drift[0], origin[1] + drift[1])
    by_anchor = ([int(l) for _d, l in sorted(
        ((np.hypot(c[0] - origin[0], c[1] - origin[1]), l) for l, c in centres_after.items()))]
        if origin is not None else [])
    shown = item.get("shown_label")
    pointed = centres_after.get(int(shown)) if shown is not None else None
    by_link = ([int(l) for _d, l in sorted(
        ((np.hypot(c[0] - pointed[0], c[1] - pointed[1]), l) for l, c in centres_after.items()))]
        if pointed is not None else [])

    def rank(sequence):
        return sequence.index(want_label) + 1 if want_label in sequence else None

    target = centres_after.get(want_label)
    ys, xs = np.nonzero(before == anchor)
    inside = False
    if len(ys):
        cy, cx = int(round(ys.mean())), int(round(xs.mean()))
        if 0 <= cy < before.shape[0] and 0 <= cx < before.shape[1]:
            inside = bool(before[cy, cx] == anchor)
    claimed = tracking_benchmark.label_to_track(api.tracking_data, frame + 1)

    ranks = (1 if item.get("reference_label") == want_label else None,
             rank(by_iou), rank(by_anchor), rank(by_link))
    return {
        "frame": frame, "anchor": anchor, "expected": want_label,
        "rank_reference": ranks[0], "rank_iou": ranks[1],
        "rank_nearest": ranks[2], "rank_link": ranks[3],
        "in_any_source": any(r is not None for r in ranks),
        "distance_from_anchor": (None if (origin is None or target is None)
                                 else round(float(np.hypot(target[0] - origin[0],
                                                           target[1] - origin[1])), 1)),
        "iou_with_anchor": round(float(overlaps.get((anchor, want_label), 0.0)), 3),
        "claimed_at_next": want_label in claimed,
        "anchor_area": int(len(ys)),
        "centroid_inside": inside,
    }


def probe_queue(api, started: dict, reference: dict,
                distance_gate_px: Optional[float] = DEFAULT_DISTANCE_GATE_PX) -> dict:
    """Probe every miss in a started census queue against `reference`."""
    misses, stale, agreed, unreferenced = [], 0, 0, 0
    for position in range(int(started.get("n", 0))):
        item = api.get_census_item(position)
        if item.get("error"):
            continue
        want = reference.get(f"Cell{item['frame']}_{item['label']}")
        if want is None:
            unreferenced += 1
            continue
        shown = (f"Cell{item['frame'] + 1}_{item['shown_label']}"
                 if item.get("shown_label") is not None else None)
        if shown == want:
            agreed += 1
            continue
        offered = {f"Cell{item['frame'] + 1}_{c['label']}" for c in item.get("candidates", [])}
        if want in offered:
            continue
        row = probe_item(api, item, int(str(want).rsplit("_", 1)[1]))
        if not row:
            continue
        if (distance_gate_px is not None and row["distance_from_anchor"] is not None
                and row["distance_from_anchor"] > distance_gate_px):
            stale += 1
            continue
        misses.append(row)
    return {"misses": misses, "excluded_stale": stale,
            "reference_agreed": agreed, "no_reference": unreferenced,
            "distance_gate_px": distance_gate_px}


def summarise(report: dict) -> str:
    """A short table, plus the column that is anomalous -- or that none is."""
    rows = report["misses"]
    lines = [f"{len(rows)} miss(es); {report['excluded_stale']} excluded as stale reference links"]
    if not rows:
        return lines[0]

    def median_of(key):
        values = [r[key] for r in rows if r[key] is not None]
        return statistics.median(values) if values else None

    for key in ("rank_reference", "rank_iou", "rank_nearest", "rank_link"):
        found = [r[key] for r in rows if r[key] is not None]
        lines.append(f"  {key:16s} found in {len(found):3d}/{len(rows)}"
                     + (f", median rank {statistics.median(found):.0f}" if found else ""))
    lines.append(f"  in_any_source    {sum(r['in_any_source'] for r in rows)}/{len(rows)}"
                 "   (generated then dropped, vs never generated)")
    for key in ("distance_from_anchor", "iou_with_anchor", "anchor_area"):
        lines.append(f"  {key:16s} median {median_of(key)}")
    lines.append(f"  claimed_at_next  {sum(r['claimed_at_next'] for r in rows)}/{len(rows)}")
    lines.append(f"  centroid_inside  {sum(r['centroid_inside'] for r in rows)}/{len(rows)}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", help="dataset directory under test_data/")
    parser.add_argument("--reference", help="a tracking-shaped .xlsx to compare against")
    parser.add_argument("--kind", default="environment_flagged")
    parser.add_argument("--output-dir", help="a throwaway directory for this run's outputs")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test or not args.project:
        # No project needed: a synthetic frame pair where the expected cell is
        # reachable by centroid but NOT by overlap, so the two ranks differ and
        # the probe has to say so.
        class _Stub:
            tracking_data = pd.DataFrame({"Index": [1], "Frame5": ["Cell5_1"], "Frame6": ["Cell6_9"]})

            def __init__(self, before, after):
                self._frames = {5: before, 6: after}

            def _load_frame(self, frame):
                return (self._frames.get(frame), None, None)

        before = np.zeros((60, 200), dtype=int)
        before[20:30, 20:30] = 1
        after = np.zeros((60, 200), dtype=int)
        after[20:24, 20:150] = 11      # clips the anchor: high overlap, far centroid
        after[20:30, 32:42] = 7        # adjacent, no overlap, nearest centroid
        item = {"frame": 5, "label": 1, "shown_label": 11, "reference_label": None}
        row = probe_item(_Stub(before, after), item, 7)
        assert row["rank_iou"] is None, row
        assert row["rank_nearest"] == 1, row
        assert row["in_any_source"] is True, row
        assert row["iou_with_anchor"] == 0.0, row
        assert row["centroid_inside"] is True, row
        print("probe: the expected cell is rank 1 by centroid and absent from the overlap")
        print("  ranking -- so the two columns disagree, which is the whole point of")
        print("  measuring them separately. in_any_source=True says it was generated")
        print("  and dropped, not never generated.")
        print("\nAll candidate_probe self-tests passed.")
        raise SystemExit(0)

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "webview"))
    from api import EpiQuantAPI

    api = EpiQuantAPI()
    api.set_project(os.path.join(args.project, "masks"),
                    os.path.join(args.project, "images"), args.output_dir)
    api.load_existing_extraction()
    api._run_tracking("single")
    started = api.start_census(kind=args.kind, mode="census", seed=0)
    if "error" in started:
        raise SystemExit(started["error"])
    reference = links_from_table(pd.read_excel(args.reference))
    print(summarise(probe_queue(api, started, reference)))
