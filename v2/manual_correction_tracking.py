"""Module 4b: manual correction — cell tracking.

Ported from `gui_cell_track.ComparisonMainWindow` (src/gui_cell_track.py),
the "Correct Cell Tracking" window.

`update_tracking_data` ports an existing v1 method almost verbatim.
`add_tracking_id` is a **new extraction**: in v1 this logic lives inline
inside the Qt `on_click` handler's "blue box" (unmatched cell) branch
(src/gui_cell_track.py:598-620), not as its own function.

Both are split from persistence: v1 calls `.to_excel(...)` directly inside
each method; here that's a separate `save_tracking_data` call, so the pure
DataFrame mutation is testable without touching the filesystem. Debug
`print(...)` calls throughout v1 (informational only, no behavior effect)
are dropped. `QMessageBox`/`QInputDialog` calls become raised exceptions —
the caller (future UI layer) is responsible for parsing/validating raw
dialog input (e.g. converting a typed string to an int) before calling
these functions; a non-integer `old_id`/`new_id`/`new_index` will raise
`ValueError` naturally via the `int(...)` conversion here, same as v1's
`except ValueError` guards did.
"""

import os
import re
from typing import Callable, Optional

import numpy as np
import pandas as pd

import adjacency


def get_original_image_path(npy_folder: str, img_folder: str, npy_filename: str) -> str:
    """Return the path to the original (non-segmentation) image for the
    frame `npy_filename` belongs to: prefers the `'img_path'` key stored in
    the `.npy` file itself, falling back to a same-frame-numbered
    `.jpeg`/`.jpg`/`.png` in `img_folder`. Raises `FileNotFoundError` if
    neither is available. Ported from
    `ComparisonMainWindow.get_original_image_path`,
    src/gui_cell_track.py:308-327.
    """
    npy_path = os.path.join(npy_folder, npy_filename)
    dat = np.load(npy_path, allow_pickle=True).item()

    if "img_path" in dat and dat["img_path"]:
        return dat["img_path"]

    frame_number = int(re.findall(r"\d+", npy_filename)[0])
    for file_name in os.listdir(img_folder):
        img_frame_match = re.findall(r"\d+", file_name)
        if img_frame_match:
            img_frame_number = int(img_frame_match[0])
            if img_frame_number == frame_number and (
                file_name.endswith(".jpeg") or file_name.endswith(".jpg") or file_name.endswith(".png")
            ):
                return os.path.join(img_folder, file_name)
    raise FileNotFoundError(f"Original image for frame {frame_number} not found.")


def update_tracking_data(
    tracking_data: pd.DataFrame,
    original_tracking_data: pd.DataFrame,
    frame_number: int,
    old_id: int,
    new_id: int,
) -> pd.DataFrame:
    """Reassign the cell that track `old_id` had in `frame_number` onto track
    `new_id` instead: looks up `old_id`'s original (pre-correction) value for
    that frame in `original_tracking_data`, writes it into `new_id`'s row,
    and nulls out any other row that already held that same cell string in
    that frame column (dedup). Returns `tracking_data` (mutated in place and
    returned for convenience) — call `save_tracking_data` separately to
    persist.

    Raises `ValueError` if `old_id` isn't found in `original_tracking_data`,
    or `new_id` isn't found in `tracking_data`'s `Index` column (v1 showed a
    `QMessageBox.critical` for each of these instead).

    Ported from `ComparisonMainWindow.update_tracking_data`,
    src/gui_cell_track.py:347-390 (excluding the final `.to_excel` call and
    the `correction_ids` in-memory UI bookkeeping, both left to the caller).
    """
    column_name = f"Frame{frame_number}"

    original_row_indices = original_tracking_data[original_tracking_data["Index"] == int(old_id)].index
    if original_row_indices.empty:
        raise ValueError("The old ID does not exist in the original data.")
    original_row_index = original_row_indices[0]
    original_cell_id_str = original_tracking_data.at[original_row_index, column_name]

    new_id_exists = tracking_data["Index"].eq(int(new_id)).any()
    if not new_id_exists:
        raise ValueError("The new ID does not exist in the index column. Invalid modification.")

    new_row_indices = tracking_data[tracking_data["Index"] == int(new_id)].index
    if new_row_indices.empty:
        raise ValueError("The new ID does not exist in the index column. Invalid modification.")
    new_row_index = new_row_indices[0]

    tracking_data.at[new_row_index, column_name] = original_cell_id_str

    for i, value in tracking_data[column_name].items():
        if value == original_cell_id_str and i != new_row_index:
            tracking_data.at[i, column_name] = None

    return tracking_data


def merge_tracks_from_frame(
    tracking_data: pd.DataFrame, frame_number: int, old_id: int, new_id: int,
    end_frame: Optional[int] = None,
) -> pd.DataFrame:
    """From `frame_number` onward (inclusive), move every cell currently
    claimed by track `old_id` onto track `new_id` instead, clearing
    `old_id`'s claim for those frames. `frame_number` itself is included,
    so this alone both corrects that frame and propagates the correction
    forward -- no separate `update_tracking_data` call needed first.

    Use when a single-frame mismatch cascades: track `old_id` picks up a
    cell at `frame_number` that's actually the continuation of track
    `new_id` (e.g. `new_id` lost its own match there), and every frame
    after that stays wrongly attached to `old_id` too, since the Hungarian
    matching just keeps extending whichever track a cell already belongs
    to. Correcting only `frame_number` (`update_tracking_data`) fixes one
    frame but leaves the tracks split for the rest of the sequence;
    merging from `frame_number` onward reunites them.

    `end_frame`, if given, bounds the merge to `frame_number..end_frame`
    inclusive instead of running unconditionally to the end of the table --
    used by the "smart-stop propagate" workflow (`find_propagation_stop_frame`)
    once it has computed where the correction's local neighborhood settles.
    `end_frame=None` (the default) reproduces the original unconditional
    behavior exactly.

    Frames before `frame_number` are untouched on both tracks -- if
    `old_id` was a genuinely different track before the mismatch, its
    earlier history is preserved under `old_id`, only its future is
    reassigned. Returns `tracking_data` (mutated in place and returned for
    convenience).

    Raises `ValueError` if `old_id`/`new_id` aren't found in `tracking_data`'s
    `Index` column.
    """
    id_col = tracking_data.columns[0]
    frame_cols = [c for c in tracking_data.columns if c.startswith("Frame")]
    later_cols = [
        c for c in frame_cols
        if int(c[len("Frame"):]) >= frame_number and (end_frame is None or int(c[len("Frame"):]) <= end_frame)
    ]

    old_rows = tracking_data.index[tracking_data[id_col] == int(old_id)]
    new_rows = tracking_data.index[tracking_data[id_col] == int(new_id)]
    if old_rows.empty:
        raise ValueError(f"Track {old_id} not found in the tracking data.")
    if new_rows.empty:
        raise ValueError(f"Track {new_id} not found in the tracking data.")
    old_pos, new_pos = old_rows[0], new_rows[0]

    for col in later_cols:
        value = tracking_data.at[old_pos, col]
        if pd.notna(value):
            tracking_data.at[new_pos, col] = value
            tracking_data.at[old_pos, col] = None

    return tracking_data


def _claimed_cells_at_frame(tracking_data: pd.DataFrame, frame_number: int) -> set:
    """Cell strings claimed by some track at `Frame{frame_number}` -- the
    same matched/unmatched notion `rendering.render_tracking_overlay` and
    `webview/api.py`'s `get_tracking_cell_at_pixel` use. Empty set if the
    column doesn't exist yet."""
    column_name = f"Frame{frame_number}"
    if column_name not in tracking_data.columns:
        return set()
    return set(tracking_data[column_name].dropna())


def _raw_cell_id(cell_index: str) -> int:
    match = re.search(r"Cell\d+_(\d+)", cell_index)
    if not match:
        raise ValueError(f"Cannot parse a raw cell id out of {cell_index!r}.")
    return int(match.group(1))


def find_propagation_stop_frame(
    tracking_data: pd.DataFrame,
    cells_info: pd.DataFrame,
    fill_labels_by_frame: Callable[[int], Optional[np.ndarray]],
    frame_number: int, old_id: int, new_id: int,
    max_frames_ahead: int = 20, dilation_px: int = 2, drift_gate_multiplier: float = 1.5,
) -> dict:
    """Suggest how far a `merge_tracks_from_frame(tracking_data, frame_number,
    old_id, new_id, end_frame=...)` correction should propagate, using the
    corrected cell's local neighbor topology instead of the current
    checkbox's unconditional to-the-end behavior.

    `fill_labels_by_frame(f)` must return frame `f`'s *filled* per-cell
    label array (or `None` if unavailable) -- a caller-supplied accessor
    (e.g. a lambda over `webview/api.py`'s `_load_frame`) so this stays
    testable with synthetic in-memory arrays, not tied to disk I/O.

    **Core idea**: the corrected cell's identity at every future frame is
    already known with certainty -- it's exactly `old_id`'s own existing
    row value, the same value `merge_tracks_from_frame` would move onto
    `new_id` if propagated that far. So this never re-derives or guesses
    that identity; it only asks whether the corrected cell's *local
    neighborhood* is still unsettled, and if so, at which later frame it
    settles or the unsettled neighbor disappears.

    **Algorithm**:
    1. At `frame_number`, find the corrected cell (via `old_id`'s row) in
       `fill_labels_by_frame(frame_number)` and compute its neighbor set
       (`adjacency.build_neighbor_graph`). `unsettled` = neighbors NOT
       already claimed at `frame_number` (matched neighbors are never
       tracked further -- this is deliberately scoped to THIS correction's
       own problem, not the whole frame's unrelated churn). If `unsettled`
       is empty, returns immediately with `reason="settled"` -- nothing to
       propagate.
    2. Walk `f = frame_number+1 .. frame_number+max_frames_ahead`. If
       `old_id`'s row has no value at `Frame{f}`, stop with
       `reason="track_ended"`. Otherwise recompute the corrected cell's
       neighbor set at `f`, and for each still-unsettled flagged neighbor,
       re-localize it by **last-known centroid position** (not raw cell ID
       -- Cellpose relabels raw IDs arbitrarily per frame, so identity must
       be tracked by position, gated by `drift_gate_multiplier * radius`,
       mirroring `build_cost_matrix`'s existing plausible-displacement
       gate):
       - No cell within the gate at all -> **disappeared** (no plausible
         continuation nearby -- the extrusion/death signal).
       - Nearest cell already claimed at `f` -> **matched**.
       - Nearest cell is within the gate but has moved more than half the
         gate radius AND is currently one of the corrected cell's own
         neighbors -> **disappeared** (absorbed: a real ~jitter-only
         continuation of the SAME cell would stay within a small fraction
         of its own radius; a larger jump landing on territory that's
         already a neighbor of the corrected cell is a better fit for "an
         existing surrounding cell expanded into the gap" than for "this
         cell kept existing and also happens to still be adjacent"). This
         is a heuristic with no labeled extrusion ground truth to check it
         against -- see the caller-facing docs for why the result is
         always a suggestion the human reviews, never auto-applied.
       - Otherwise -> still unsettled; its tracked position/radius updates
         to the relocated cell's own, for the next iteration.
       New unmatched neighbors first appearing later are deliberately never
       added to `unsettled` -- keeps the walk bounded to this specific
       correction's own problem.
       Stops with `reason="settled"` (all resolved via matching) or
       `"neighbor_disappeared"` (all resolved, at least one via
       disappearance) the moment `unsettled` empties.
    3. If the loop exhausts, stops with `reason="reached_max_frames_ahead"`.

    Returns `{"stop_frame": int, "reason": str, "unsettled_neighbors_by_frame": dict}`
    -- the trail is always meant to be surfaced to the user alongside the
    suggested stop frame, never trusted silently.

    Raises `ValueError` if `old_id`/`new_id` aren't in `tracking_data`, or if
    there's no segmentation/track data for `old_id` at `frame_number` itself.
    """
    id_col = tracking_data.columns[0]
    old_rows = tracking_data.index[tracking_data[id_col] == int(old_id)]
    if old_rows.empty:
        raise ValueError(f"Track {old_id} not found in the tracking data.")
    if tracking_data.index[tracking_data[id_col] == int(new_id)].empty:
        raise ValueError(f"Track {new_id} not found in the tracking data.")
    old_row = old_rows[0]

    def _corrected_cell_at(frame: int):
        col = f"Frame{frame}"
        if col not in tracking_data.columns:
            return None
        value = tracking_data.at[old_row, col]
        return None if pd.isna(value) else value

    cells_info_indexed = cells_info.set_index("Cell Index")
    fill_labels = fill_labels_by_frame(frame_number)
    corrected_cell = _corrected_cell_at(frame_number)
    if fill_labels is None or corrected_cell is None:
        raise ValueError(f"No segmentation/track data for track {old_id} at frame {frame_number}.")

    graph = adjacency.build_neighbor_graph(fill_labels, dilation_px=dilation_px)
    neighbor_labels = graph.get(_raw_cell_id(corrected_cell), set())
    claimed = _claimed_cells_at_frame(tracking_data, frame_number)

    unsettled = {}  # cell_index -> (x, y, radius)
    for label in neighbor_labels:
        neighbor_index = f"Cell{frame_number}_{label}"
        if neighbor_index in claimed or neighbor_index not in cells_info_indexed.index:
            continue
        row = cells_info_indexed.loc[neighbor_index]
        unsettled[neighbor_index] = (
            float(row["Center X Coordinate"]), float(row["Center Y Coordinate"]), float(row["Circumscribed Circle Radius"]),
        )

    trail: dict = {}
    if not unsettled:
        return {"stop_frame": frame_number, "reason": "settled", "unsettled_neighbors_by_frame": trail}

    for f in range(frame_number + 1, frame_number + max_frames_ahead + 1):
        corrected_cell = _corrected_cell_at(f)
        frame_fill_labels = fill_labels_by_frame(f)
        if corrected_cell is None or frame_fill_labels is None:
            return {"stop_frame": f - 1, "reason": "track_ended", "unsettled_neighbors_by_frame": trail}

        graph = adjacency.build_neighbor_graph(frame_fill_labels, dilation_px=dilation_px)
        current_neighbor_indices = {
            f"Cell{f}_{label}" for label in graph.get(_raw_cell_id(corrected_cell), set())
        }
        claimed = _claimed_cells_at_frame(tracking_data, f)
        frame_cells = cells_info_indexed[cells_info_indexed.index.str.startswith(f"Cell{f}_")]
        frame_cells = frame_cells.drop(index=corrected_cell, errors="ignore")

        disappeared_this_frame = False
        still_unsettled = {}
        for neighbor_index, (last_x, last_y, last_radius) in unsettled.items():
            if frame_cells.empty:
                disappeared_this_frame = True
                continue

            # `.astype(float)`: `Cells_info`'s coordinate columns can come through as
            # object dtype (per-row DataFrame concatenation in `data_input.extract_cell_info`
            # doesn't always infer float64), which breaks vectorized `np.hypot` on a Series
            # (each value is still a real float, just boxed in an object-dtype column).
            distances = np.hypot(
                frame_cells["Center X Coordinate"].astype(float) - last_x,
                frame_cells["Center Y Coordinate"].astype(float) - last_y,
            )
            nearest_index = distances.idxmin()
            nearest_distance = float(distances.loc[nearest_index])
            gate = drift_gate_multiplier * last_radius

            if nearest_distance > gate:
                disappeared_this_frame = True
            elif nearest_index in claimed:
                pass  # matched -- drop from unsettled
            elif nearest_distance > 0.5 * last_radius and nearest_index in current_neighbor_indices:
                disappeared_this_frame = True  # absorbed into an existing neighbor's territory
            else:
                still_unsettled[nearest_index] = (
                    float(frame_cells.at[nearest_index, "Center X Coordinate"]),
                    float(frame_cells.at[nearest_index, "Center Y Coordinate"]),
                    float(frame_cells.at[nearest_index, "Circumscribed Circle Radius"]),
                )

        unsettled = still_unsettled
        if unsettled:
            trail[f] = sorted(unsettled)
        else:
            return {
                "stop_frame": f,
                "reason": "neighbor_disappeared" if disappeared_this_frame else "settled",
                "unsettled_neighbors_by_frame": trail,
            }

    return {
        "stop_frame": frame_number + max_frames_ahead,
        "reason": "reached_max_frames_ahead",
        "unsettled_neighbors_by_frame": trail,
    }


def swap_tracks_at_frame(
    tracking_data: pd.DataFrame, frame_number: int, id_a: int, id_b: int,
) -> pd.DataFrame:
    """Atomically exchange the `Frame{frame_number}` cell strings currently
    held by tracks `id_a` and `id_b`. Returns `tracking_data` (mutated in
    place and returned for convenience).

    Reads from the LIVE `tracking_data`, not `original_tracking_data` --
    unlike `update_tracking_data`, this makes it safe to use after prior
    corrections in the same session, since it never depends on a frozen
    pre-correction snapshot.

    This is the fix for a real data-loss bug in `update_tracking_data`: for
    a genuine two-cell ID swap (track A's cell is really track B's
    continuation and vice versa), calling `update_tracking_data` twice
    silently destroys one side -- its first call overwrites `new_id`'s
    current value with `old_id`'s value, with no read/preservation of what
    `new_id` held before, so that cell vanishes from the table entirely
    rather than landing on `old_id` as a true swap would require.

    Deliberately narrow and explicit rather than folded into
    `update_tracking_data`: raises `ValueError` (directing the caller to the
    normal reassign/`add_tracking_id` flow instead) if `id_a == id_b`,
    either ID is missing from `tracking_data`, or either ID's
    `Frame{frame_number}` value is null -- a swap only makes sense when BOTH
    cells already have a claim in this frame; one-sided cases are a plain
    relabel, not a swap, and forcing them through here would risk the exact
    kind of implicit data loss this function exists to avoid.
    """
    column_name = f"Frame{frame_number}"
    id_col = tracking_data.columns[0]

    if int(id_a) == int(id_b):
        raise ValueError("id_a and id_b must be different tracks to swap.")

    row_a = tracking_data.index[tracking_data[id_col] == int(id_a)]
    row_b = tracking_data.index[tracking_data[id_col] == int(id_b)]
    if row_a.empty:
        raise ValueError(f"Track {id_a} not found in the tracking data.")
    if row_b.empty:
        raise ValueError(f"Track {id_b} not found in the tracking data.")
    row_a, row_b = row_a[0], row_b[0]

    value_a = tracking_data.at[row_a, column_name]
    value_b = tracking_data.at[row_b, column_name]
    if pd.isna(value_a) or pd.isna(value_b):
        raise ValueError(
            f"Both track {id_a} and track {id_b} must already have a cell at frame "
            f"{frame_number} to swap -- use the normal reassign/add flow for a one-sided fix."
        )

    tracking_data.at[row_a, column_name] = value_b
    tracking_data.at[row_b, column_name] = value_a

    return tracking_data


def add_tracking_id(
    tracking_data: pd.DataFrame, frame_number: int, new_index: int, cell_id_str: str,
) -> pd.DataFrame:
    """Assign a previously-unmatched segmentation cell (`cell_id_str`, e.g.
    `'Cell5_12'`) to an existing track `new_index`'s slot for `frame_number`.
    Returns `tracking_data` (mutated in place and returned for convenience).

    Raises `KeyError` if `new_index` isn't found in `tracking_data`'s `Index`
    column (v1 showed a `QMessageBox.warning` instead).

    **New extraction**: in v1 this logic is inlined directly inside the Qt
    `on_click` "blue box" branch (src/gui_cell_track.py:598-620), not its own
    function.
    """
    update_row_index = tracking_data[tracking_data["Index"] == int(new_index)].index
    if update_row_index.empty:
        raise KeyError(f"ID {new_index} was not found in the tracking data's Index column.")

    tracking_data.at[update_row_index[0], f"Frame{frame_number}"] = cell_id_str
    return tracking_data


def suggest_match_candidates(
    cells_info: pd.DataFrame,
    tracking_data: pd.DataFrame,
    frame_number: int,
    selected_cell_index: str,
    top_k: int = 3,
) -> list:
    """For `selected_cell_index` (a `Cell{frame_number}_{raw_id}` string),
    rank frame `frame_number + 1`'s currently-**unmatched** cells by
    ascending centroid distance and return up to `top_k` as
    `[{"cell_index": str, "cell_id": int, "distance": float}, ...]`.

    Pure computation, no mutation -- a human still confirms and applies any
    suggestion via `add_tracking_id`/`update_tracking_data`/
    `swap_tracks_at_frame`. Deliberately restricted to the pool of cells
    frame `frame_number + 1`'s `tracking_data` column doesn't already claim
    (the existing matched/unmatched notion, see `rendering.render_tracking_overlay`)
    -- this is a plain nearest-neighbor filter over already-known unmatched
    candidates, NOT a local Hungarian re-match: a "local re-track" feature
    doing the latter was already built, tested, and removed (see
    `v2/README.md`'s changelog) because re-deriving a match from spatial
    proximity alone just reproduces whatever wrong answer Hungarian already
    gave; this function never decides anything on its own, it only narrows
    down candidates for a human to pick from.

    Raises `ValueError` if `selected_cell_index` doesn't belong to
    `frame_number`, `KeyError` if it isn't found in `cells_info`. Returns
    `[]` (not an error) if frame `frame_number + 1` has no unmatched cells.
    """
    if not selected_cell_index.startswith(f"Cell{frame_number}_"):
        raise ValueError(
            f"selected_cell_index {selected_cell_index!r} does not belong to frame {frame_number}."
        )

    indexed = cells_info.set_index("Cell Index")
    if selected_cell_index not in indexed.index:
        raise KeyError(f"{selected_cell_index} not found in cells_info.")
    origin_x = indexed.at[selected_cell_index, "Center X Coordinate"]
    origin_y = indexed.at[selected_cell_index, "Center Y Coordinate"]

    next_frame = frame_number + 1
    next_frame_cells = indexed[indexed.index.str.startswith(f"Cell{next_frame}_")]

    next_frame_col = f"Frame{next_frame}"
    claimed = set(tracking_data[next_frame_col].dropna()) if next_frame_col in tracking_data.columns else set()

    candidates = []
    for cell_index, row in next_frame_cells.iterrows():
        if cell_index in claimed:
            continue
        distance = float(np.hypot(
            row["Center X Coordinate"] - origin_x, row["Center Y Coordinate"] - origin_y,
        ))
        raw_id_match = re.search(r"Cell\d+_(\d+)", cell_index)
        cell_id = int(raw_id_match.group(1)) if raw_id_match else None
        candidates.append({"cell_index": cell_index, "cell_id": cell_id, "distance": distance})

    candidates.sort(key=lambda c: c["distance"])
    return candidates[:top_k]


def save_tracking_data(tracking_data: pd.DataFrame, file_path: str) -> None:
    """Persist `tracking_data` to `file_path` (.xlsx). Split out from
    `update_tracking_data`/`add_tracking_id` so the pure mutation and the
    disk write are separately testable."""
    tracking_data.to_excel(file_path, index=False)


if __name__ == "__main__":
    # Synthetic 2-track / 2-frame fixture -- no real project data needed to
    # exercise the correction logic itself.
    tracking = pd.DataFrame({
        "Index": [1, 2],
        "Frame1": ["Cell1_10", "Cell1_11"],
        "Frame2": ["Cell2_20", None],
    })
    original = tracking.copy()

    print("Before:")
    print(tracking)

    update_tracking_data(tracking, original, frame_number=2, old_id=1, new_id=2)
    print("\nAfter reassigning track 1's Frame2 cell onto track 2:")
    print(tracking)

    add_tracking_id(tracking, frame_number=2, new_index=1, cell_id_str="Cell2_99")
    print("\nAfter assigning a new (previously unmatched) cell to track 1's Frame2:")
    print(tracking)

    # swap_tracks_at_frame: a genuine two-cell swap, using the LIVE table
    # this time (unlike update_tracking_data, which reads old_id from the
    # frozen `original` snapshot).
    swap_fixture = pd.DataFrame({
        "Index": [1, 2],
        "Frame1": ["Cell1_10", "Cell1_11"],
        "Frame2": ["Cell2_20", "Cell2_21"],
    })
    print("\nSwap fixture before:")
    print(swap_fixture)
    swap_tracks_at_frame(swap_fixture, frame_number=2, id_a=1, id_b=2)
    print("\nAfter swapping tracks 1 and 2's Frame2 cells:")
    print(swap_fixture)
    assert swap_fixture.at[0, "Frame2"] == "Cell2_21"
    assert swap_fixture.at[1, "Frame2"] == "Cell2_20"

    try:
        swap_tracks_at_frame(swap_fixture, frame_number=2, id_a=1, id_b=1)
        raise AssertionError("expected ValueError for id_a == id_b")
    except ValueError as e:
        print(f"\nCorrectly rejected same-id swap: {e}")

    one_sided = pd.DataFrame({
        "Index": [1, 2],
        "Frame1": ["Cell1_10", "Cell1_11"],
        "Frame2": ["Cell2_20", None],
    })
    try:
        swap_tracks_at_frame(one_sided, frame_number=2, id_a=1, id_b=2)
        raise AssertionError("expected ValueError for a one-sided (unmatched) swap target")
    except ValueError as e:
        print(f"Correctly rejected one-sided swap: {e}")

    # suggest_match_candidates: three frame-2 cells, one already claimed by
    # another track -- must be excluded, and the remaining two ranked by
    # distance from the selected frame-1 cell.
    suggest_cells_info = pd.DataFrame({
        "Cell Index": ["Cell1_5", "Cell2_10", "Cell2_11", "Cell2_12"],
        "Center X Coordinate": [0.0, 1.0, 50.0, 3.0],
        "Center Y Coordinate": [0.0, 1.0, 50.0, 4.0],
    })
    suggest_tracking = pd.DataFrame({
        "Index": [1, 2],
        "Frame1": ["Cell1_5", "Cell1_6"],
        "Frame2": ["Cell2_10", None],  # Cell2_10 already claimed -> excluded
    })
    suggestions = suggest_match_candidates(suggest_cells_info, suggest_tracking, frame_number=1, selected_cell_index="Cell1_5")
    print("\nSuggested match candidates for Cell1_5 at frame 2:", suggestions)
    assert [s["cell_index"] for s in suggestions] == ["Cell2_12", "Cell2_11"], suggestions
    assert suggestions[0]["distance"] < suggestions[1]["distance"]

    empty_suggestions = suggest_match_candidates(
        pd.DataFrame({"Cell Index": ["Cell1_5"], "Center X Coordinate": [0.0], "Center Y Coordinate": [0.0]}),
        pd.DataFrame({"Index": [1], "Frame1": ["Cell1_5"]}),
        frame_number=1, selected_cell_index="Cell1_5",
    )
    assert empty_suggestions == [], empty_suggestions

    try:
        suggest_match_candidates(suggest_cells_info, suggest_tracking, frame_number=2, selected_cell_index="Cell1_5")
        raise AssertionError("expected ValueError for a cell not belonging to frame_number")
    except ValueError as e:
        print(f"Correctly rejected mismatched frame_number: {e}")

    # merge_tracks_from_frame(end_frame=...): bounded merge leaves later
    # frames untouched; end_frame=None regression-matches the pre-change
    # (unconditional-to-the-end) behavior exactly.
    bounded = pd.DataFrame({
        "Index": [1, 2],
        "Frame1": ["Cell1_10", "Cell1_11"],
        "Frame2": ["Cell2_20", None],
        "Frame3": ["Cell3_30", None],
    })
    merge_tracks_from_frame(bounded, frame_number=2, old_id=1, new_id=2, end_frame=2)
    assert bounded.at[1, "Frame2"] == "Cell2_20" and pd.isna(bounded.at[0, "Frame2"])
    assert bounded.at[0, "Frame3"] == "Cell3_30", "frame 3 should be untouched by an end_frame=2 bound"
    print("\nmerge_tracks_from_frame(end_frame=2) correctly left Frame3 on old_id:")
    print(bounded)

    unbounded = pd.DataFrame({
        "Index": [1, 2],
        "Frame1": ["Cell1_10", "Cell1_11"],
        "Frame2": ["Cell2_20", None],
        "Frame3": ["Cell3_30", None],
    })
    merge_tracks_from_frame(unbounded, frame_number=2, old_id=1, new_id=2)
    assert unbounded.at[1, "Frame2"] == "Cell2_20" and unbounded.at[1, "Frame3"] == "Cell3_30"
    assert pd.isna(unbounded.at[0, "Frame2"]) and pd.isna(unbounded.at[0, "Frame3"])
    print("merge_tracks_from_frame(end_frame=None) correctly reproduces the unconditional-to-the-end behavior.")

    # find_propagation_stop_frame: a synthetic corrected cell (label 1) with
    # one flagged, initially-unmatched neighbor (label 2), across a few
    # tiny hand-built frames -- one scenario per termination reason.
    def _grid_with_two_cells(both_present=True):
        grid = np.zeros((10, 10), dtype=int)
        grid[2:5, 2:5] = 1
        if both_present:
            grid[2:5, 6:9] = 2
        return grid

    def _cells_info_row(cell_index, x, y, radius=3.0):
        return {"Cell Index": cell_index, "Center X Coordinate": x, "Center Y Coordinate": y, "Circumscribed Circle Radius": radius}

    # -- settled (via match) --
    settled_frames = {1: _grid_with_two_cells(), 2: _grid_with_two_cells()}
    settled_cells_info = pd.DataFrame([
        _cells_info_row("Cell1_1", 3, 3), _cells_info_row("Cell1_2", 7, 3),
        _cells_info_row("Cell2_1", 3, 3), _cells_info_row("Cell2_2", 7, 3),
    ])
    settled_tracking = pd.DataFrame({
        "Index": [1, 2, 99],
        "Frame1": ["Cell1_1", "Cell1_9", None],
        "Frame2": ["Cell2_1", None, "Cell2_2"],  # Cell2_2 claimed by an unrelated track -> matched
    })
    result = find_propagation_stop_frame(
        settled_tracking, settled_cells_info, lambda f: settled_frames.get(f),
        frame_number=1, old_id=1, new_id=2,
    )
    print("\nfind_propagation_stop_frame (settled via match):", result)
    assert result == {"stop_frame": 2, "reason": "settled", "unsettled_neighbors_by_frame": {}}, result

    # -- neighbor_disappeared (vanishes entirely) --
    disappear_frames = {1: _grid_with_two_cells(), 2: _grid_with_two_cells(both_present=False)}
    disappear_cells_info = pd.DataFrame([
        _cells_info_row("Cell1_1", 3, 3), _cells_info_row("Cell1_2", 7, 3),
        _cells_info_row("Cell2_1", 3, 3),
    ])
    disappear_tracking = pd.DataFrame({
        "Index": [1, 2],
        "Frame1": ["Cell1_1", "Cell1_9"],
        "Frame2": ["Cell2_1", None],
    })
    result = find_propagation_stop_frame(
        disappear_tracking, disappear_cells_info, lambda f: disappear_frames.get(f),
        frame_number=1, old_id=1, new_id=2,
    )
    print("find_propagation_stop_frame (neighbor disappeared):", result)
    assert result["stop_frame"] == 2 and result["reason"] == "neighbor_disappeared", result

    # -- track_ended (old_id has no Frame3 entry) --
    ended_frames = {1: _grid_with_two_cells(), 2: _grid_with_two_cells()}
    ended_cells_info = pd.DataFrame([
        _cells_info_row("Cell1_1", 3, 3), _cells_info_row("Cell1_2", 7, 3),
        _cells_info_row("Cell2_1", 3, 3), _cells_info_row("Cell2_2", 7, 3),
    ])
    ended_tracking = pd.DataFrame({
        "Index": [1, 2],
        "Frame1": ["Cell1_1", "Cell1_9"],
        "Frame2": ["Cell2_1", None],  # Cell2_2 stays unmatched -- still unsettled after frame 2
    })
    result = find_propagation_stop_frame(
        ended_tracking, ended_cells_info, lambda f: ended_frames.get(f),
        frame_number=1, old_id=1, new_id=2,
    )
    print("find_propagation_stop_frame (track ended):", result)
    assert result["stop_frame"] == 2 and result["reason"] == "track_ended", result

    # -- reached_max_frames_ahead (neighbor never resolves) --
    never_frames = {1: _grid_with_two_cells(), 2: _grid_with_two_cells(), 3: _grid_with_two_cells()}
    never_cells_info = pd.DataFrame([
        _cells_info_row("Cell1_1", 3, 3), _cells_info_row("Cell1_2", 7, 3),
        _cells_info_row("Cell2_1", 3, 3), _cells_info_row("Cell2_2", 7, 3),
        _cells_info_row("Cell3_1", 3, 3), _cells_info_row("Cell3_2", 7, 3),
    ])
    never_tracking = pd.DataFrame({
        "Index": [1, 2],
        "Frame1": ["Cell1_1", "Cell1_9"],
        "Frame2": ["Cell2_1", None],
        "Frame3": ["Cell3_1", None],
    })
    result = find_propagation_stop_frame(
        never_tracking, never_cells_info, lambda f: never_frames.get(f),
        frame_number=1, old_id=1, new_id=2, max_frames_ahead=2,
    )
    print("find_propagation_stop_frame (max frames ahead reached):", result)
    assert result["stop_frame"] == 3 and result["reason"] == "reached_max_frames_ahead", result

    print("\nAll manual_correction_tracking smoke tests passed.")
