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
) -> tuple:
    """Reassign the cell that track `old_id` had in `frame_number` onto track
    `new_id` instead: looks up `old_id`'s original (pre-correction) value for
    that frame in `original_tracking_data`, writes it into `new_id`'s row,
    and nulls out any other row that already held that same cell string in
    that frame column (dedup). Returns `(tracking_data, displaced)` — call
    `save_tracking_data` separately to persist.

    `displaced` is `None` unless `new_id` already held a *different*, live
    claim at `frame_number` — in that case, that claim's entire future (from
    `frame_number` onward) is first relocated onto a freshly-minted track ID
    via `_evict_future_if_colliding`, rather than being silently destroyed,
    and `displaced` is `{"evicted_track_id", "new_track_id", "from_frame"}`
    describing what moved. `tracking_data` may be a NEW DataFrame object
    when a relocation happens (appending a row changes object identity) --
    always use the returned value.

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

    displaced = None
    current_value = tracking_data.at[new_row_index, column_name]
    if pd.notna(current_value) and current_value != original_cell_id_str:
        tracking_data, displaced = _evict_future_if_colliding(tracking_data, "Index", new_row_index, frame_number)

    tracking_data.at[new_row_index, column_name] = original_cell_id_str

    for i, value in tracking_data[column_name].items():
        if value == original_cell_id_str and i != new_row_index:
            tracking_data.at[i, column_name] = None

    return tracking_data, displaced


def merge_tracks_from_frame(
    tracking_data: pd.DataFrame, frame_number: int, old_id: int, new_id: int,
    end_frame: Optional[int] = None,
) -> tuple:
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
    reassigned. Returns `(tracking_data, displaced)`.

    `displaced` is `None` unless `new_id` already held live data at some
    frame where `old_id` also has live data being moved onto it -- i.e. a
    genuine collision (the exact scenario from a 3+ track cascading
    mismatch, where fixing one track's identity would otherwise silently
    overwrite a third, unrelated track). When that happens, `new_id`'s
    entire future from the *first* overlapping frame onward is first
    relocated to a freshly-minted track ID via `_evict_future_if_colliding`
    (never silently discarded), and `displaced` describes what moved.
    `tracking_data` may be a NEW DataFrame object when a relocation
    happens -- always use the returned value.

    Raises `ValueError` if `old_id == new_id` (merging a track into itself
    would, per the loop below, write each frame's own value back onto
    itself and then immediately null it out -- a real self-destructive
    latent bug were this ever called that way; guarded here rather than
    relying on every caller to avoid it), or if `old_id`/`new_id` aren't
    found in `tracking_data`'s `Index` column.
    """
    if int(old_id) == int(new_id):
        raise ValueError("old_id and new_id must be different tracks to merge.")

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

    overlap_frames = [
        int(c[len("Frame"):]) for c in later_cols
        if pd.notna(tracking_data.at[old_pos, c]) and pd.notna(tracking_data.at[new_pos, c])
    ]
    displaced = None
    if overlap_frames:
        tracking_data, displaced = _evict_future_if_colliding(tracking_data, id_col, new_pos, min(overlap_frames))

    for col in later_cols:
        value = tracking_data.at[old_pos, col]
        if pd.notna(value):
            tracking_data.at[new_pos, col] = value
            tracking_data.at[old_pos, col] = None

    return tracking_data, displaced


def _centroid_and_radius_from_mask(fill_labels: np.ndarray, raw_id: int) -> Optional[tuple]:
    """Fallback geometry for a cell missing from `cells_info` -- most often
    one dropped by the Extract step's `exclude_edge_cells` option, which
    removes edge-touching cells from `Cells_info.xlsx` (a deliberate,
    scientifically-meaningful exclusion for morphometric statistics) while
    leaving them fully intact in the segmentation mask used here for
    correction-time lookups. Returns `(center_x, center_y, radius)` computed
    directly from the mask's own pixel positions (radius = max distance from
    centroid to any of the cell's own pixels, matching the "Circumscribed
    Circle Radius" semantics used elsewhere in this module), or `None` if
    `raw_id` isn't present in `fill_labels` at all.
    """
    ys, xs = np.where(fill_labels == raw_id)
    if len(xs) == 0:
        return None
    center_x, center_y = float(xs.mean()), float(ys.mean())
    radius = float(np.hypot(xs - center_x, ys - center_y).max())
    return center_x, center_y, radius


def _evict_future_if_colliding(
    tracking_data: pd.DataFrame, id_col: str, target_row_pos, from_frame: int,
) -> tuple:
    """If `target_row_pos`'s row already holds non-null values at
    `Frame{n}` for any `n >= from_frame`, relocate its entire future (from
    `from_frame` onward) onto a freshly-minted track ID *before* the caller
    overwrites that slot -- otherwise a reassign/merge into an
    already-claimed target would silently destroy a third, unrelated
    track's legitimate data. This is the collision guard used by
    `update_tracking_data`, `add_tracking_id`, and `merge_tracks_from_frame`.

    Returns `(tracking_data, None)` if nothing needed relocating (the
    common, non-colliding case -- every caller's existing behavior is
    unchanged then). Returns `(possibly-new tracking_data,
    {"evicted_track_id": <old id>, "new_track_id": <fresh id>, "from_frame": from_frame})`
    if a relocation happened -- appending a row changes the DataFrame's
    identity, so callers must use the returned `tracking_data`, never
    assume the original object was mutated in place, once eviction occurs.

    The fresh ID is `max(existing Index) + 1` -- robust to non-contiguous
    or gapped IDs (deleted/never-reused numbers), unlike a raw
    row-count-based scheme.
    """
    frame_cols = [c for c in tracking_data.columns if c.startswith("Frame")]
    later_cols = [c for c in frame_cols if int(c[len("Frame"):]) >= from_frame]

    if not any(pd.notna(tracking_data.at[target_row_pos, c]) for c in later_cols):
        return tracking_data, None

    old_track_id = tracking_data.at[target_row_pos, id_col]
    new_track_id = int(tracking_data[id_col].max()) + 1

    new_row = {c: None for c in tracking_data.columns}
    new_row[id_col] = new_track_id
    for c in later_cols:
        value = tracking_data.at[target_row_pos, c]
        if pd.notna(value):
            new_row[c] = value
            tracking_data.at[target_row_pos, c] = None

    new_row_label = (tracking_data.index.max() + 1) if len(tracking_data) else 0
    tracking_data = pd.concat([tracking_data, pd.DataFrame([new_row], index=[new_row_label])])

    return tracking_data, {
        "evicted_track_id": int(old_track_id), "new_track_id": new_track_id, "from_frame": from_frame,
    }


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
        if neighbor_index in claimed:
            continue
        if neighbor_index in cells_info_indexed.index:
            row = cells_info_indexed.loc[neighbor_index]
            position = (
                float(row["Center X Coordinate"]), float(row["Center Y Coordinate"]), float(row["Circumscribed Circle Radius"]),
            )
        else:
            # Missing from cells_info -- most often an edge-excluded cell (see
            # _centroid_and_radius_from_mask); fall back to the mask itself
            # rather than silently dropping it from consideration.
            position = _centroid_and_radius_from_mask(fill_labels, label)
        if position is None:
            continue
        unsettled[neighbor_index] = position

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


def swap_tracks_from_frame(
    tracking_data: pd.DataFrame, frame_number: int, id_a: int, id_b: int,
    end_frame: Optional[int] = None,
) -> dict:
    """From `frame_number` onward (inclusive), exchange whatever `id_a` and
    `id_b` each hold at EVERY frame in the range -- unlike a single
    `swap_tracks_at_frame` call (one frame only), this handles a mutual
    mislabeling that spans several frames, mirroring `merge_tracks_from_frame`'s
    "propagate to later frames" checkbox but for a swap instead of a
    one-directional relabel.

    Reuses `swap_tracks_at_frame` per frame: since a swap only ever
    exchanges values between these same two rows, it can never collide
    with a third track, so none of `merge_tracks_from_frame`'s eviction
    logic is needed here.

    A frame where only one of `id_a`/`id_b` has a claim isn't swappable
    (see `swap_tracks_at_frame`'s own one-sided guard) -- rather than
    aborting the whole range at the first such frame, it's simply skipped
    and reported. `end_frame=None` (the default) runs to the table's last
    `Frame{n}` column.

    Returns `{"swapped_frames": [...], "skipped_frames": [...]}`.

    Raises `ValueError` if `id_a == id_b` or either ID doesn't exist in
    `tracking_data` at all (checked once upfront, not per frame).
    """
    id_col = tracking_data.columns[0]
    if int(id_a) == int(id_b):
        raise ValueError("id_a and id_b must be different tracks to swap.")
    if tracking_data.index[tracking_data[id_col] == int(id_a)].empty:
        raise ValueError(f"Track {id_a} not found in the tracking data.")
    if tracking_data.index[tracking_data[id_col] == int(id_b)].empty:
        raise ValueError(f"Track {id_b} not found in the tracking data.")

    frame_cols = [c for c in tracking_data.columns if c.startswith("Frame")]
    frame_numbers = sorted(
        int(c[len("Frame"):]) for c in frame_cols
        if int(c[len("Frame"):]) >= frame_number and (end_frame is None or int(c[len("Frame"):]) <= end_frame)
    )

    swapped_frames = []
    skipped_frames = []
    for f in frame_numbers:
        try:
            swap_tracks_at_frame(tracking_data, f, id_a, id_b)
            swapped_frames.append(f)
        except ValueError:
            skipped_frames.append(f)

    return {"swapped_frames": swapped_frames, "skipped_frames": skipped_frames}


def find_confirmed_identity_moves(
    tracking_data: pd.DataFrame,
    cells_info: pd.DataFrame,
    fill_labels_by_frame: Callable[[int], Optional[np.ndarray]],
    frame_number: int,
    track_id: int,
    max_frames_ahead: int = 20,
    drift_gate_multiplier: float = 1.5,
    restrict_to_unmatched: bool = True,
) -> dict:
    """Starting from `track_id`'s own (human-confirmed) cell at
    `frame_number` -- typically just attached via `suggest_match_candidates`
    + `add_tracking_id`/`reassign_track` -- walk forward tracking that
    SAME physical cell by centroid position, and reconcile it against
    whichever track (if any) currently claims it at each later frame.

    This is deliberately different from `find_propagation_stop_frame`: that
    function trusts `old_id`'s own row values as ground truth for every
    future frame, which is correct when there's one single, consistently-
    wrong `old_id` to merge -- but wrong for a cell whose track number
    churns through several DIFFERENT wrong tracks across consecutive frames
    (e.g. a physical cell manually attached to track 18 at frame f, then
    found under track 12 at f+1, track 14 at f+2, track 16 at f+3 -- three
    separate wrong tracks, not one). Re-deriving the physical continuation
    by position at each step (gated by `drift_gate_multiplier * radius`,
    the same plausible-displacement convention as `build_cost_matrix` and
    `find_propagation_stop_frame`'s own neighbor walk) is what lets this
    follow the cell through that kind of churn. This is NOT a revival of
    the removed "local re-track" feature (see `v2/README.md`'s changelog):
    it only extends an already human-confirmed anchor forward one small,
    gated step at a time, it never re-runs Hungarian or guesses an initial
    match on its own.

    **`restrict_to_unmatched` (default `True`) -- "restricted pairing".**
    Candidates at each later frame are limited to cells no track currently
    claims, instead of every cell in the frame. This fixes a real drift bug
    found on `test_data/AS_dorsal view_E-cad`: a plain nearest-centroid pick
    can land on a neighbouring cell that another track legitimately owns
    merely because it happens to be a few pixels closer, and since the
    anchor position then updates to that wrong cell, every subsequent frame
    follows the wrong track too. Measured instance -- at one frame three
    cells sat inside the gate (49.5px): 25.0px (owned by track 5), 28.4px
    (owned by track 11), and 30.7px, the genuine continuation and the only
    unclaimed one. Restricting to unclaimed candidates picks it correctly.

    The restriction is principled, not just a tie-break: a cell some track
    already claims is *accounted for* -- that track asserts continuity
    through it -- whereas an unclaimed cell is by definition one the
    Hungarian matching failed to account for, which is exactly what an
    anchor needing this repair is looking for. Set `restrict_to_unmatched=False`
    to allow taking cells away from other tracks, which is what the
    multiple-different-wrong-tracks churn case needs (there the genuine
    continuation IS claimed, by a different wrong track each frame).

    Per frame, in order:
    - the frame has no cells at all, or nothing at all within the gate ->
      stop, `reason="disappeared"`.
    - the nearest cell is already claimed by `track_id` itself -> stop,
      `reason="settled"` (Hungarian recovered on its own; nothing left to
      fix).
    - `restrict_to_unmatched` and no *unclaimed* cell within the gate ->
      stop, `reason="no_unmatched_continuation"` (something is there, but
      taking it would mean stealing another track's cell -- deliberately
      left to the human, who can re-run with the restriction off).
    - otherwise record a move for the nearest eligible cell -- with
      `from_track_id=None` when unclaimed (a plain attach), or naming the
      track it is taken from (only reachable with the restriction off) --
      and keep walking.
    - `max_frames_ahead` exhausted, or a `Frame{n}` column doesn't exist in
      `tracking_data` -> stop, `reason="reached_max_frames_ahead"` or
      `"track_ended"` respectively.

    Returns `{"stop_frame": int, "reason": str, "moves": [{"frame": int,
    "from_track_id": Optional[int], "cell_index": str, "distance": float},
    ...]}` -- a plan only; nothing is written here. `distance` is the
    centroid displacement that justified each step, surfaced so a reviewer
    can judge a borderline pick. Falls back to
    `_centroid_and_radius_from_mask` for any cell missing from `cells_info`,
    matching this module's other functions.

    Raises `ValueError` if `track_id` doesn't exist in `tracking_data`, or
    if it has no cell at `frame_number` to anchor from, or if that anchor
    cell's position can't be resolved at all.
    """
    id_col = tracking_data.columns[0]
    track_rows = tracking_data.index[tracking_data[id_col] == int(track_id)]
    if track_rows.empty:
        raise ValueError(f"Track {track_id} not found in the tracking data.")
    track_row = track_rows[0]

    anchor_col = f"Frame{frame_number}"
    if anchor_col not in tracking_data.columns:
        raise ValueError(f"Frame {frame_number} not found in the tracking data.")
    anchor_cell = tracking_data.at[track_row, anchor_col]
    if pd.isna(anchor_cell):
        raise ValueError(f"Track {track_id} has no cell at frame {frame_number} to anchor from.")

    cells_info_indexed = cells_info.set_index("Cell Index")

    def _position(cell_index: str, frame: int):
        if cell_index in cells_info_indexed.index:
            row = cells_info_indexed.loc[cell_index]
            return (
                float(row["Center X Coordinate"]), float(row["Center Y Coordinate"]),
                float(row["Circumscribed Circle Radius"]),
            )
        fill_labels = fill_labels_by_frame(frame)
        if fill_labels is not None:
            return _centroid_and_radius_from_mask(fill_labels, _raw_cell_id(cell_index))
        return None

    anchor_position = _position(anchor_cell, frame_number)
    if anchor_position is None:
        raise ValueError(
            f"Could not resolve a position for {anchor_cell} (missing from cells_info and the segmentation mask)."
        )
    last_x, last_y, last_radius = anchor_position

    moves: list = []
    for f in range(frame_number + 1, frame_number + max_frames_ahead + 1):
        frame_col = f"Frame{f}"
        if frame_col not in tracking_data.columns:
            return {"stop_frame": f - 1, "reason": "track_ended", "moves": moves}

        frame_cells = cells_info_indexed[cells_info_indexed.index.str.startswith(f"Cell{f}_")]
        if frame_cells.empty:
            return {"stop_frame": f - 1, "reason": "disappeared", "moves": moves}

        distances = np.hypot(
            frame_cells["Center X Coordinate"].astype(float) - last_x,
            frame_cells["Center Y Coordinate"].astype(float) - last_y,
        )
        gate = drift_gate_multiplier * last_radius

        overall_nearest = distances.idxmin()
        if float(distances.loc[overall_nearest]) > gate:
            return {"stop_frame": f - 1, "reason": "disappeared", "moves": moves}

        claimed_by = {
            row[frame_col]: int(row[id_col])
            for _, row in tracking_data[[id_col, frame_col]].dropna(subset=[frame_col]).iterrows()
        }

        # Whoever is physically nearest already being us means the tracking
        # recovered on its own from here -- checked before any restriction,
        # since it ends the walk either way.
        if claimed_by.get(overall_nearest) == int(track_id):
            return {"stop_frame": f, "reason": "settled", "moves": moves}

        if restrict_to_unmatched:
            eligible = distances[[i for i in distances.index if i not in claimed_by]]
            if eligible.empty or float(eligible.min()) > gate:
                return {"stop_frame": f - 1, "reason": "no_unmatched_continuation", "moves": moves}
            nearest_index = eligible.idxmin()
        else:
            nearest_index = overall_nearest

        moves.append({
            "frame": f,
            "from_track_id": claimed_by.get(nearest_index),
            "cell_index": nearest_index,
            "distance": round(float(distances.loc[nearest_index]), 1),
        })
        last_x = float(frame_cells.at[nearest_index, "Center X Coordinate"])
        last_y = float(frame_cells.at[nearest_index, "Center Y Coordinate"])
        last_radius = float(frame_cells.at[nearest_index, "Circumscribed Circle Radius"])

    return {"stop_frame": frame_number + max_frames_ahead, "reason": "reached_max_frames_ahead", "moves": moves}


def apply_confirmed_identity_moves(tracking_data: pd.DataFrame, track_id: int, moves: list) -> tuple:
    """Apply a `moves` plan from `find_confirmed_identity_moves`: for each
    move, writes its `cell_index` into `track_id`'s row at `frame`, clearing
    `from_track_id`'s claim there when one is named. Routed through the same
    `_evict_future_if_colliding` guard as every other write path in this
    module -- defensive here, since `track_id`'s slot at each move's frame
    should already be empty by construction of the plan, but this keeps the
    same never-silently-overwrite guarantee.

    Returns `(tracking_data, displaced_list)` -- `tracking_data` may be a
    NEW DataFrame object if any eviction happened; always use the returned
    value. `displaced_list` collects every non-`None` displacement record,
    in move order (usually empty).

    Raises `ValueError` if `track_id` doesn't exist in `tracking_data`.
    """
    id_col = tracking_data.columns[0]
    track_rows = tracking_data.index[tracking_data[id_col] == int(track_id)]
    if track_rows.empty:
        raise ValueError(f"Track {track_id} not found in the tracking data.")
    track_row = track_rows[0]

    displaced_list = []
    for move in moves:
        frame_col = f"Frame{move['frame']}"
        cell_index = move["cell_index"]
        from_track_id = move.get("from_track_id")

        current_value = tracking_data.at[track_row, frame_col]
        if pd.notna(current_value) and current_value != cell_index:
            tracking_data, displaced = _evict_future_if_colliding(tracking_data, id_col, track_row, move["frame"])
            if displaced is not None:
                displaced_list.append(displaced)

        tracking_data.at[track_row, frame_col] = cell_index

        if from_track_id is not None:
            from_rows = tracking_data.index[tracking_data[id_col] == int(from_track_id)]
            if not from_rows.empty:
                from_row = from_rows[0]
                if tracking_data.at[from_row, frame_col] == cell_index:
                    tracking_data.at[from_row, frame_col] = None

    return tracking_data, displaced_list


def add_tracking_id(
    tracking_data: pd.DataFrame, frame_number: int, new_index: int, cell_id_str: str,
) -> tuple:
    """Assign a previously-unmatched segmentation cell (`cell_id_str`, e.g.
    `'Cell5_12'`) to an existing track `new_index`'s slot for `frame_number`.
    Returns `(tracking_data, displaced)`.

    `displaced` is `None` unless `new_index` already held a *different*,
    live claim at `frame_number` -- in that case, that claim's entire
    future (from `frame_number` onward) is first relocated onto a
    freshly-minted track ID via `_evict_future_if_colliding`, rather than
    being silently overwritten, and `displaced` describes what moved.
    `tracking_data` may be a NEW DataFrame object when a relocation
    happens -- always use the returned value.

    Raises `KeyError` if `new_index` isn't found in `tracking_data`'s `Index`
    column (v1 showed a `QMessageBox.warning` instead).

    **New extraction**: in v1 this logic is inlined directly inside the Qt
    `on_click` "blue box" branch (src/gui_cell_track.py:598-620), not its own
    function.
    """
    update_row_index = tracking_data[tracking_data["Index"] == int(new_index)].index
    if update_row_index.empty:
        raise KeyError(f"ID {new_index} was not found in the tracking data's Index column.")
    row_pos = update_row_index[0]
    column_name = f"Frame{frame_number}"

    displaced = None
    current_value = tracking_data.at[row_pos, column_name]
    if pd.notna(current_value) and current_value != cell_id_str:
        tracking_data, displaced = _evict_future_if_colliding(tracking_data, "Index", row_pos, frame_number)

    tracking_data.at[row_pos, column_name] = cell_id_str
    return tracking_data, displaced


def suggest_match_candidates(
    cells_info: pd.DataFrame,
    tracking_data: pd.DataFrame,
    frame_number: int,
    selected_cell_index: str,
    fill_labels_by_frame: Callable[[int], Optional[np.ndarray]],
    top_k: int = 3,
) -> list:
    """For `selected_cell_index` (a `Cell{frame_number}_{raw_id}` string --
    typically an unmatched cell the user just clicked, but works just as
    well for one that's currently attached to the wrong track), rank
    "orphaned" tracks -- ones that had a claim at `frame_number - 1` but
    lost it at `frame_number` -- by ascending centroid distance from the
    selected cell, and return up to `top_k` as
    `[{"track_id": int, "cell_index": str, "distance": float}, ...]`
    (`cell_index` is the candidate's own `Frame{frame_number-1}` cell
    string, for display).

    Searches BACKWARD (candidates at `frame_number - 1`) rather than
    forward, matching how a user actually notices a problem: they see an
    unmatched (or wrongly-matched) cell at the LATER frame and want to know
    which existing track it continues -- not the reverse. `frame_number` is
    the frame of the cell the user clicked; the caller decides how to apply
    a chosen candidate (`add_tracking_id` if `selected_cell_index` is
    currently unmatched, `reassign_track` if it's attached to the wrong
    track -- this function only ranks candidates, it never decides or
    writes anything itself).

    Pure computation, no mutation. Deliberately just a nearest-neighbor
    filter over already-known orphaned tracks, NOT a local Hungarian
    re-match: a "local re-track" feature doing the latter was already
    built, tested, and removed (see `v2/README.md`'s changelog) because
    re-deriving a match from spatial proximity alone just reproduces
    whatever wrong answer Hungarian already gave.

    Falls back to computing a cell's centroid directly from
    `fill_labels_by_frame`'s mask (via `_centroid_and_radius_from_mask`)
    when it's missing from `cells_info` -- e.g. an edge-touching cell the
    Extract step's `exclude_edge_cells` option deliberately dropped from
    `Cells_info.xlsx` while leaving it in the segmentation. This is what
    makes the function usable on a newly-arrived or edge-excluded cell
    instead of raising `KeyError`.

    Raises `ValueError` if `selected_cell_index` doesn't belong to
    `frame_number`, or if its position can't be resolved at all (missing
    from both `cells_info` and the mask). Returns `[]` (not an error) if
    `frame_number <= 1` (nothing precedes it) or no track is orphaned there.
    """
    if not selected_cell_index.startswith(f"Cell{frame_number}_"):
        raise ValueError(
            f"selected_cell_index {selected_cell_index!r} does not belong to frame {frame_number}."
        )
    if frame_number <= 1:
        return []

    cells_info_indexed = cells_info.set_index("Cell Index")

    def _position(cell_index: str, frame: int):
        if cell_index in cells_info_indexed.index:
            row = cells_info_indexed.loc[cell_index]
            return float(row["Center X Coordinate"]), float(row["Center Y Coordinate"])
        fill_labels = fill_labels_by_frame(frame)
        if fill_labels is not None:
            fallback = _centroid_and_radius_from_mask(fill_labels, _raw_cell_id(cell_index))
            if fallback is not None:
                return fallback[0], fallback[1]
        return None

    origin = _position(selected_cell_index, frame_number)
    if origin is None:
        raise ValueError(
            f"Could not resolve a position for {selected_cell_index} "
            "(missing from cells_info and the segmentation mask)."
        )
    origin_x, origin_y = origin

    prev_frame = frame_number - 1
    prev_col, cur_col = f"Frame{prev_frame}", f"Frame{frame_number}"
    if prev_col not in tracking_data.columns:
        return []

    id_col = tracking_data.columns[0]
    has_prev = tracking_data[prev_col].notna()
    lost_here = tracking_data[cur_col].isna() if cur_col in tracking_data.columns else True
    orphaned = tracking_data[has_prev & lost_here]

    candidates = []
    for _, row in orphaned.iterrows():
        candidate_cell_index = row[prev_col]
        position = _position(candidate_cell_index, prev_frame)
        if position is None:
            continue
        cx, cy = position
        distance = float(np.hypot(cx - origin_x, cy - origin_y))
        candidates.append({"track_id": int(row[id_col]), "cell_index": candidate_cell_index, "distance": distance})

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

    # swap_tracks_from_frame: a 3-frame range where the middle frame has
    # only one side present -- that frame should be skipped and reported,
    # not abort the whole range.
    swap_range = pd.DataFrame({
        "Index": [1, 2],
        "Frame1": ["Cell1_10", "Cell1_11"],
        "Frame2": ["Cell2_20", None],
        "Frame3": ["Cell3_30", "Cell3_31"],
    })
    swap_result = swap_tracks_from_frame(swap_range, frame_number=1, id_a=1, id_b=2)
    print("\nswap_tracks_from_frame result:", swap_result)
    assert swap_result == {"swapped_frames": [1, 3], "skipped_frames": [2]}, swap_result
    assert swap_range.at[0, "Frame1"] == "Cell1_11" and swap_range.at[1, "Frame1"] == "Cell1_10"
    assert swap_range.at[0, "Frame2"] == "Cell2_20", "frame 2 was skipped -- should be untouched"
    assert swap_range.at[0, "Frame3"] == "Cell3_31" and swap_range.at[1, "Frame3"] == "Cell3_30"

    try:
        swap_tracks_from_frame(swap_range, frame_number=1, id_a=1, id_b=1)
        raise AssertionError("expected ValueError for id_a == id_b")
    except ValueError as e:
        print(f"Correctly rejected same-id swap_tracks_from_frame: {e}")

    # suggest_match_candidates (reversed direction): a selected cell at
    # frame 2 (e.g. unmatched, or wrongly matched) should rank frame-1
    # tracks that lost their claim at frame 2 ("orphaned"), nearest first.
    # Track 2 already has a Frame2 claim -> not orphaned, excluded.
    suggest_cells_info = pd.DataFrame({
        "Cell Index": ["Cell1_10", "Cell1_11", "Cell1_12", "Cell2_50", "Cell2_99"],
        "Center X Coordinate": [1.0, 50.0, 3.0, 51.0, 2.0],
        "Center Y Coordinate": [1.0, 50.0, 4.0, 51.0, 2.0],
    })
    suggest_tracking = pd.DataFrame({
        "Index": [1, 2, 3],
        "Frame1": ["Cell1_10", "Cell1_11", "Cell1_12"],
        "Frame2": [None, "Cell2_50", None],
    })
    suggestions = suggest_match_candidates(
        suggest_cells_info, suggest_tracking, frame_number=2, selected_cell_index="Cell2_99",
        fill_labels_by_frame=lambda f: None,
    )
    print("\nSuggested orphaned tracks for Cell2_99 at frame 2:", suggestions)
    assert [s["track_id"] for s in suggestions] == [1, 3], suggestions
    assert suggestions[0]["distance"] < suggestions[1]["distance"]

    empty_suggestions = suggest_match_candidates(
        pd.DataFrame({"Cell Index": ["Cell1_5"], "Center X Coordinate": [0.0], "Center Y Coordinate": [0.0]}),
        pd.DataFrame({"Index": [1], "Frame1": ["Cell1_5"]}),
        frame_number=1, selected_cell_index="Cell1_5", fill_labels_by_frame=lambda f: None,
    )
    assert empty_suggestions == [], empty_suggestions  # frame 1 has nothing preceding it

    try:
        suggest_match_candidates(
            suggest_cells_info, suggest_tracking, frame_number=1, selected_cell_index="Cell2_99",
            fill_labels_by_frame=lambda f: None,
        )
        raise AssertionError("expected ValueError for a cell not belonging to frame_number")
    except ValueError as e:
        print(f"Correctly rejected mismatched frame_number: {e}")

    # Mask-fallback path: a selected cell missing from cells_info entirely
    # (simulating an edge-excluded cell) is still resolvable via its raw
    # mask footprint.
    edge_grid = np.zeros((10, 10), dtype=int)
    edge_grid[0:2, 0:2] = 77  # "Cell2_77", deliberately absent from cells_info
    edge_cells_info = pd.DataFrame({
        "Cell Index": ["Cell1_10"],
        "Center X Coordinate": [0.5],
        "Center Y Coordinate": [0.5],
    })
    edge_tracking = pd.DataFrame({"Index": [1], "Frame1": ["Cell1_10"], "Frame2": [None]})
    edge_suggestions = suggest_match_candidates(
        edge_cells_info, edge_tracking, frame_number=2, selected_cell_index="Cell2_77",
        fill_labels_by_frame=lambda f: edge_grid if f == 2 else None,
    )
    print("Mask-fallback suggestions for edge-excluded Cell2_77:", edge_suggestions)
    assert [s["track_id"] for s in edge_suggestions] == [1], edge_suggestions

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

    # Collision guard: the exact 3-cell cascading-mismatch scenario from the
    # user's report. Cell_i's real track is 11, Cell_j's real track is 15,
    # and Cell_k is a genuinely new cell that squatted on track 15 from
    # frame 2 onward. Fixing Cell_j (merge 11 -> 15) must not destroy
    # Cell_k's data; it should auto-evict Cell_k to a fresh track first.
    cascade = pd.DataFrame({
        "Index": [11, 13, 15],
        "Frame1": ["Cell1_i", None, "Cell1_j"],
        "Frame2": ["Cell2_j", "Cell2_i", "Cell2_k"],
        "Frame3": ["Cell3_j", "Cell3_i", "Cell3_k"],
    })
    cascade, displaced = merge_tracks_from_frame(cascade, frame_number=2, old_id=11, new_id=15)
    print("\nAfter merging 11->15 (fixing Cell_j), Cell_k auto-evicted:")
    print(cascade)
    assert displaced == {"evicted_track_id": 15, "new_track_id": 16, "from_frame": 2}, displaced
    row15 = cascade.index[cascade["Index"] == 15][0]
    row16 = cascade.index[cascade["Index"] == 16][0]
    row11 = cascade.index[cascade["Index"] == 11][0]
    assert cascade.at[row15, "Frame1"] == "Cell1_j"
    assert cascade.at[row15, "Frame2"] == "Cell2_j" and cascade.at[row15, "Frame3"] == "Cell3_j"
    assert cascade.at[row16, "Frame2"] == "Cell2_k" and cascade.at[row16, "Frame3"] == "Cell3_k"
    assert pd.isna(cascade.at[row16, "Frame1"])
    assert pd.isna(cascade.at[row11, "Frame2"]) and pd.isna(cascade.at[row11, "Frame3"])

    cascade, displaced2 = merge_tracks_from_frame(cascade, frame_number=2, old_id=13, new_id=11)
    print("\nAfter merging 13->11 (fixing Cell_i), no collision this time:")
    print(cascade)
    assert displaced2 is None, "track 11 was already vacated by the first merge -- no eviction should occur"
    row11 = cascade.index[cascade["Index"] == 11][0]
    assert cascade.at[row11, "Frame1"] == "Cell1_i"
    assert cascade.at[row11, "Frame2"] == "Cell2_i" and cascade.at[row11, "Frame3"] == "Cell3_i"
    print("All three physical identities (Cell_i=11, Cell_j=15, Cell_k=16) are now correctly separated.")

    # add_tracking_id / update_tracking_data collision guard (single-frame).
    add_collision = pd.DataFrame({
        "Index": [1, 2],
        "Frame1": ["Cell1_10", "Cell1_11"],
        "Frame2": [None, "Cell2_99"],
    })
    add_collision, add_displaced = add_tracking_id(add_collision, frame_number=2, new_index=2, cell_id_str="Cell2_50")
    assert add_displaced == {"evicted_track_id": 2, "new_track_id": 3, "from_frame": 2}, add_displaced
    row2 = add_collision.index[add_collision["Index"] == 2][0]
    row3 = add_collision.index[add_collision["Index"] == 3][0]
    assert add_collision.at[row2, "Frame2"] == "Cell2_50"
    assert add_collision.at[row3, "Frame2"] == "Cell2_99"
    print("\nadd_tracking_id correctly evicted a colliding claim to a fresh track:")
    print(add_collision)

    update_collision = pd.DataFrame({
        "Index": [1, 2],
        "Frame1": ["Cell1_10", "Cell1_11"],
        "Frame2": ["Cell2_20", "Cell2_99"],
    })
    update_original = update_collision.copy()
    update_collision, update_displaced = update_tracking_data(
        update_collision, update_original, frame_number=2, old_id=1, new_id=2,
    )
    assert update_displaced == {"evicted_track_id": 2, "new_track_id": 3, "from_frame": 2}, update_displaced
    row2 = update_collision.index[update_collision["Index"] == 2][0]
    row3 = update_collision.index[update_collision["Index"] == 3][0]
    assert update_collision.at[row2, "Frame2"] == "Cell2_20"
    assert update_collision.at[row3, "Frame2"] == "Cell2_99"
    print("update_tracking_data correctly evicted a colliding claim to a fresh track:")
    print(update_collision)

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

    # find_confirmed_identity_moves / apply_confirmed_identity_moves: a
    # physical cell manually attached to track 18 at frame 1, then found
    # under three DIFFERENT wrong tracks (12, 14, 16) across frames 2-4,
    # settling back onto track 18 itself at frame 5.
    churn_cells_info = pd.DataFrame([
        {"Cell Index": "Cell1_A", "Center X Coordinate": 5.0, "Center Y Coordinate": 5.0, "Circumscribed Circle Radius": 3.0},
        {"Cell Index": "Cell2_A", "Center X Coordinate": 5.0, "Center Y Coordinate": 5.0, "Circumscribed Circle Radius": 3.0},
        {"Cell Index": "Cell3_A", "Center X Coordinate": 5.0, "Center Y Coordinate": 5.0, "Circumscribed Circle Radius": 3.0},
        {"Cell Index": "Cell4_A", "Center X Coordinate": 5.0, "Center Y Coordinate": 5.0, "Circumscribed Circle Radius": 3.0},
        {"Cell Index": "Cell5_A", "Center X Coordinate": 5.0, "Center Y Coordinate": 5.0, "Circumscribed Circle Radius": 3.0},
    ])
    churn_tracking = pd.DataFrame({
        "Index": [18, 12, 14, 16],
        "Frame1": ["Cell1_A", None, None, None],
        "Frame2": [None, "Cell2_A", None, None],
        "Frame3": [None, None, "Cell3_A", None],
        "Frame4": [None, None, None, "Cell4_A"],
        "Frame5": ["Cell5_A", None, None, None],
    })
    # Every continuation here is claimed by a wrong track, so this needs the
    # restriction OFF -- with it on (the default) the walk must refuse to
    # steal them and stop immediately instead.
    restricted_plan = find_confirmed_identity_moves(
        churn_tracking, churn_cells_info, lambda f: None, frame_number=1, track_id=18,
    )
    assert restricted_plan["moves"] == [] and restricted_plan["reason"] == "no_unmatched_continuation", restricted_plan
    print("\nfind_confirmed_identity_moves (churn, restricted): correctly refused to steal claimed cells.")

    plan = find_confirmed_identity_moves(
        churn_tracking, churn_cells_info, lambda f: None, frame_number=1, track_id=18,
        restrict_to_unmatched=False,
    )
    print("find_confirmed_identity_moves (churn through 3 wrong tracks):", plan)
    assert plan["reason"] == "settled" and plan["stop_frame"] == 5, plan
    assert [(m["frame"], m["from_track_id"], m["cell_index"]) for m in plan["moves"]] == [
        (2, 12, "Cell2_A"), (3, 14, "Cell3_A"), (4, 16, "Cell4_A"),
    ], plan["moves"]

    churn_tracking, churn_displaced = apply_confirmed_identity_moves(churn_tracking, 18, plan["moves"])
    print("After applying the moves:")
    print(churn_tracking)
    assert churn_displaced == []
    row18 = churn_tracking.index[churn_tracking["Index"] == 18][0]
    assert churn_tracking.at[row18, "Frame2"] == "Cell2_A"
    assert churn_tracking.at[row18, "Frame3"] == "Cell3_A"
    assert churn_tracking.at[row18, "Frame4"] == "Cell4_A"
    for track_id, frame_col in [(12, "Frame2"), (14, "Frame3"), (16, "Frame4")]:
        row = churn_tracking.index[churn_tracking["Index"] == track_id][0]
        assert pd.isna(churn_tracking.at[row, frame_col]), f"track {track_id}'s claim at {frame_col} should be cleared"
    print("All three churned tracks correctly reunited onto track 18.")

    # Unclaimed-continuation case: the walk should still record a move
    # (from_track_id=None) rather than mistaking "unclaimed" for "settled".
    unclaimed_cells_info = pd.DataFrame([
        {"Cell Index": "Cell1_B", "Center X Coordinate": 0.0, "Center Y Coordinate": 0.0, "Circumscribed Circle Radius": 2.0},
        {"Cell Index": "Cell2_B", "Center X Coordinate": 0.0, "Center Y Coordinate": 0.0, "Circumscribed Circle Radius": 2.0},
    ])
    unclaimed_tracking = pd.DataFrame({"Index": [9], "Frame1": ["Cell1_B"], "Frame2": [None]})
    unclaimed_plan = find_confirmed_identity_moves(
        unclaimed_tracking, unclaimed_cells_info, lambda f: None, frame_number=1, track_id=9, max_frames_ahead=1,
    )
    print("\nfind_confirmed_identity_moves (unclaimed continuation):", unclaimed_plan)
    assert [(m["frame"], m["from_track_id"], m["cell_index"]) for m in unclaimed_plan["moves"]] == [
        (2, None, "Cell2_B"),
    ], unclaimed_plan
    assert unclaimed_plan["reason"] == "reached_max_frames_ahead"

    # Disappearance case: no cell at all in the next frame.
    gone_cells_info = pd.DataFrame([
        {"Cell Index": "Cell1_C", "Center X Coordinate": 0.0, "Center Y Coordinate": 0.0, "Circumscribed Circle Radius": 2.0},
    ])
    gone_tracking = pd.DataFrame({"Index": [7], "Frame1": ["Cell1_C"], "Frame2": [None]})
    gone_plan = find_confirmed_identity_moves(
        gone_tracking, gone_cells_info, lambda f: None, frame_number=1, track_id=7,
    )
    print("find_confirmed_identity_moves (disappeared):", gone_plan)
    assert gone_plan == {"stop_frame": 1, "reason": "disappeared", "moves": []}, gone_plan

    # Restricted pairing vs. drift onto a neighbour another track owns --
    # the shape of a real failure measured on `test_data/AS_dorsal view_E-cad`:
    # at frame 2 a cell owned by track 5 sits slightly CLOSER (2.0) to the
    # anchor than the genuine unclaimed continuation (3.0), both inside the
    # gate. Unrestricted, the walk takes the owned one and then follows that
    # wrong cell onward; restricted, it correctly takes the unclaimed one.
    drift_cells_info = pd.DataFrame([
        {"Cell Index": "Cell1_anchor", "Center X Coordinate": 0.0, "Center Y Coordinate": 0.0, "Circumscribed Circle Radius": 4.0},
        {"Cell Index": "Cell2_neighbor", "Center X Coordinate": 2.0, "Center Y Coordinate": 0.0, "Circumscribed Circle Radius": 4.0},
        {"Cell Index": "Cell2_true", "Center X Coordinate": 3.0, "Center Y Coordinate": 0.0, "Circumscribed Circle Radius": 4.0},
    ])
    drift_tracking = pd.DataFrame({
        "Index": [18, 5],
        "Frame1": ["Cell1_anchor", None],
        "Frame2": [None, "Cell2_neighbor"],
    })
    drift_restricted = find_confirmed_identity_moves(
        drift_tracking, drift_cells_info, lambda f: None, frame_number=1, track_id=18, max_frames_ahead=1,
    )
    print("\nfind_confirmed_identity_moves (restricted, neighbour owned by track 5):", drift_restricted)
    assert [m["cell_index"] for m in drift_restricted["moves"]] == ["Cell2_true"], drift_restricted
    assert drift_restricted["moves"][0]["from_track_id"] is None
    assert drift_restricted["moves"][0]["distance"] == 3.0

    drift_unrestricted = find_confirmed_identity_moves(
        drift_tracking, drift_cells_info, lambda f: None, frame_number=1, track_id=18, max_frames_ahead=1,
        restrict_to_unmatched=False,
    )
    print("find_confirmed_identity_moves (unrestricted, same fixture):", drift_unrestricted)
    assert [m["cell_index"] for m in drift_unrestricted["moves"]] == ["Cell2_neighbor"], drift_unrestricted
    assert drift_unrestricted["moves"][0]["from_track_id"] == 5
    print("Restricted pairing correctly avoids stealing the neighbour; unrestricted still can.")

    print("\nAll manual_correction_tracking smoke tests passed.")
