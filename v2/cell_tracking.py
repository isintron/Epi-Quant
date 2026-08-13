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

logger = logging.getLogger(__name__)

# The 18 Cells_info.xlsx columns used to build the tracking cost matrix.
# NOTE: `build_cost_matrix`'s gating check and weight application are
# hardcoded to specific POSITIONS in this list (index 1/2 = Max
# Horizontal/Vertical Length, used for the plausible-displacement gate;
# index 12/13 = Center X/Y, the only weighted features) — this is inherited
# directly from v1's positional (not name-based) indexing. Reordering this
# list without updating those positions, or passing a differently-ordered
# `feature_columns` to `build_cost_matrix`, will silently break the gate and
# weighting. In practice, only `DEFAULT_TRACKING_WEIGHTS` is meant to be
# varied; the column order should be left as-is.
DEFAULT_TRACKING_FEATURES = [
    "Cell Area", "Maximum Horizontal Length", "Maximum Vertical Length", "Horizontal/Vertical Length Ratio",
    "Cell Perimeter", "Approximate Polygon Vertex Count", "Fitted Ellipse Minor Axis",
    "Fitted Ellipse Major Axis", "Ellipse Major/Minor Axis Ratio", "Fitted Ellipse Angle",
    "Circumscribed Circle Radius", "Inscribed Circle Radius", "Center X Coordinate",
    "Center Y Coordinate", "Cell Left Boundary", "Cell Right Boundary", "Circularity", "P-Value",
]

# Only Center X/Y (positions 12, 13) actually contribute -- this already
# reduces to pure centroid Euclidean distance, matching the manuscript's
# stated tracking method. Kept as-is; exposed as a real parameter rather
# than a hardcoded literal.
DEFAULT_TRACKING_WEIGHTS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]


def build_cost_matrix(
    cells_frame1: pd.DataFrame,
    cells_frame2: pd.DataFrame,
    feature_columns: Optional[list[str]] = None,
    weights: Optional[list[float]] = None,
) -> pd.DataFrame:
    """Build the dissimilarity/cost matrix between every cell in
    `cells_frame1` (rows) and `cells_frame2` (columns), both indexed by
    `Cell Index` string with `feature_columns` present. A pair is marked
    implausible (`NaN`) if the frame-2 cell's centroid moved further than
    `max(cell's own Max Horizontal Length, Max Vertical Length)` in either
    axis; otherwise the weighted Euclidean distance over `feature_columns` is
    computed. Ported from src/gui_main.py:1694-1727.
    """
    if feature_columns is None:
        feature_columns = DEFAULT_TRACKING_FEATURES
    if weights is None:
        weights = DEFAULT_TRACKING_WEIGHTS

    idx1 = list(cells_frame1.index)
    idx2 = list(cells_frame2.index)
    cells_relation = np.zeros((len(idx1), len(idx2)))

    for i, id1 in enumerate(idx1):
        info1 = list(cells_frame1.loc[id1, feature_columns])
        for j, id2 in enumerate(idx2):
            info2 = list(cells_frame2.loc[id2, feature_columns])

            max_distance = max(info1[1], info1[2])
            if abs(info2[12] - info1[12]) > max_distance or abs(info2[13] - info1[13]) > max_distance:
                cells_relation[i, j] = np.nan
                continue

            distance_sq = 0.0
            for k in range(len(feature_columns)):
                distance_sq += weights[k] * (info2[k] - info1[k]) ** 2
            cells_relation[i, j] = np.sqrt(distance_sq)

    return pd.DataFrame(cells_relation, index=idx1, columns=idx2)


def match_frames(cost_matrix: pd.DataFrame, sentinel: float) -> pd.DataFrame:
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
    """
    df_filled = cost_matrix.fillna(sentinel)
    row_ind, col_ind = linear_sum_assignment(df_filled.values)
    return pd.DataFrame({
        "frame1_cell": [cost_matrix.index[r] for r in row_ind],
        "frame2_cell": [cost_matrix.columns[c] for c in col_ind],
        "Difference": [df_filled.iat[r, c] for r, c in zip(row_ind, col_ind)],
    })


def load_and_process_data(file_path: str) -> pd.DataFrame:
    """Read a `*_Matching_Results_new_*.xlsx` file and drop its `Difference`
    column. Verbatim port of src/gui_main.py:1805-1810."""
    data = pd.read_excel(file_path)
    data.drop(columns=["Difference"], inplace=True)
    return data


def _merge_tracking_pairs(frame_output_dir: str, folder_prefix: str, start_index: int) -> pd.DataFrame:
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
    if not matching_frames:
        data_final = pd.DataFrame()
        data_final.to_excel(os.path.join(frame_output_dir, f"{folder_prefix}merged_tracking_results.xlsx"), index=False)
        return data_final

    start_frame = min(matching_frames)
    files_count = len(matching_frames)

    tracks: list[dict[str, str]] = []
    open_tracks: dict[str, int] = {}

    for i in range(start_frame, start_frame + files_count):
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

    frame_columns = [f"Frame{n}" for n in range(start_frame, start_frame + files_count + 1)]
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
    """
    if feature_columns is None:
        feature_columns = DEFAULT_TRACKING_FEATURES
    if weights is None:
        weights = DEFAULT_TRACKING_WEIGHTS

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

    for idx, fig_id in enumerate(fig_ids):
        if progress_cb is not None:
            progress_cb(f"Tracking frame {fig_id} for {name}...", int((idx + 1) / total_figs * 100))

        cells_info_fig1 = cells_info[cells_info.index.str.contains(f"Cell{fig_id}_")]
        cells_info_fig2 = cells_info[cells_info.index.str.contains(f"Cell{fig_id + 1}_")]

        cost_df = build_cost_matrix(cells_info_fig1, cells_info_fig2, feature_columns, weights)
        cost_df.to_excel(os.path.join(frame_output_dir, f"{folder_prefix}Cell_Relation_{fig_id}_{fig_id + 1}.xlsx"))

        sentinel = cells_info["Circumscribed Circle Radius"].max()
        matching_result = match_frames(cost_df, sentinel)
        matching_result = matching_result.rename(
            columns={"frame1_cell": f"Frame{fig_id}", "frame2_cell": f"Frame{fig_id + 1}"}
        )
        matching_result.insert(0, "Index", range(start_index, start_index + len(matching_result)))
        matching_result = matching_result[matching_result["Difference"] != sentinel]

        matching_result.to_excel(
            os.path.join(frame_output_dir, f"{folder_prefix}Matching_Results_new_{fig_id}_{fig_id + 1}.xlsx"),
            index=False,
        )
        start_index += len(matching_result)

    return _merge_tracking_pairs(frame_output_dir, folder_prefix, start_index=merge_start_index)


def retrack_frame_range(
    cells_info: pd.DataFrame,
    tracking_data: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    feature_columns: Optional[list[str]] = None,
    weights: Optional[list[float]] = None,
) -> dict:
    """Re-solve tracking for frames `start_frame..end_frame` only, leaving
    every other frame's identities untouched. For use after hand-correcting
    segmentation: the masks changed, so the stored links for those frames are
    stale, but re-tracking the whole movie would renumber tracks and discard
    manual tracking corrections elsewhere.

    Returns `{"reconciled": [...], "new_tracks": [...], "frames": [...],
    "relinked": int}`; `tracking_data` is mutated in place (and may be a NEW
    object if rows were appended -- use the returned dict's caller-side
    value, i.e. re-read from the caller's own reference).

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
    sentinel = cells_info["Circumscribed Circle Radius"].max()

    links: dict[int, dict[str, str]] = {}
    for frame in range(pair_start, pair_end + 1):
        cells_a = cells_info[cells_info.index.str.contains(f"Cell{frame}_")]
        cells_b = cells_info[cells_info.index.str.contains(f"Cell{frame + 1}_")]
        if cells_a.empty or cells_b.empty:
            continue
        cost = build_cost_matrix(cells_a, cells_b, feature_columns, weights)
        matched = match_frames(cost, sentinel)
        matched = matched[matched["Difference"] != sentinel]
        links[frame] = dict(zip(matched["frame1_cell"], matched["frame2_cell"]))

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
    """
    has_leading_edge = "leading_edge" in cells_info.columns
    all_tracking_data = pd.DataFrame()

    if has_leading_edge:
        mes_cells_info, epi_cells_info = split_by_leading_edge(cells_info)
        mes_cells_info.to_excel(os.path.join(output_dir, "mes_cells_info.xlsx"), sheet_name="mesCells", index=False)
        epi_cells_info.to_excel(os.path.join(output_dir, "epi_cells_info.xlsx"), sheet_name="epiCells", index=False)
        mes_cells_info = mes_cells_info.set_index("Cell Index")
        epi_cells_info = epi_cells_info.set_index("Cell Index")

        if not mes_cells_info.empty:
            mes_tracking = track_cells(mes_cells_info, output_dir, progress_cb=progress_cb)
            all_tracking_data = pd.concat([all_tracking_data, mes_tracking], axis=0)
        if not epi_cells_info.empty:
            epi_tracking = track_cells(epi_cells_info, output_dir, progress_cb=progress_cb)
            all_tracking_data = pd.concat([all_tracking_data, epi_tracking], axis=0)
    else:
        cells_info = cells_info.set_index("Cell Index")
        all_tracking_data = track_cells(cells_info, output_dir, progress_cb=progress_cb)

    if all_tracking_data is not None and not all_tracking_data.empty:
        all_cell_dir = os.path.join(output_dir, "all_cell_tracking")
        os.makedirs(all_cell_dir, exist_ok=True)
        all_tracking_data.to_excel(
            os.path.join(all_cell_dir, "all_cell_merged_tracking_results.xlsx"), index=False
        )

    return all_tracking_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standalone debug entry for v2.cell_tracking")
    parser.add_argument("--cells-info", help="Cells_info.xlsx path (omit to run the self-test)")
    parser.add_argument("--output-dir")
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

        print("All cell_tracking self-tests passed.")
        raise SystemExit(0)

    os.makedirs(args.output_dir, exist_ok=True)

    cells_info_df = pd.read_excel(args.cells_info)

    def _print_progress(message: str, percent: int) -> None:
        print(f"[{percent:3d}%] {message}")

    merged = track_all_populations(cells_info_df, args.output_dir, progress_cb=_print_progress)
    print(f"\nMerged tracking shape: {merged.shape}")
    print(f"Columns: {list(merged.columns)}")
