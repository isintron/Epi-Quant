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


def _merge_tracking_pairs(frame_output_dir: str, folder_prefix: str) -> pd.DataFrame:
    """Stitch all `{folder_prefix}Matching_Results_new_{i}_{i+1}.xlsx` files
    in `frame_output_dir` into one wide `Index, Frame{start}..Frame{last}`
    DataFrame, writes `{folder_prefix}merged_tracking_results.xlsx`, and
    returns it. Ported from src/gui_main.py:1762-1803."""
    matching_files = [
        f for f in os.listdir(frame_output_dir)
        if re.match(rf"{folder_prefix}Matching_Results_new_\d+_\d+\.xlsx", f)
    ]
    matching_frames = sorted(
        set(int(re.search(r"\d+", f).group(0)) for f in matching_files if re.search(r"\d+", f))
    )
    start_frame = min(matching_frames) if matching_frames else 1
    files_count = len(matching_frames)

    if matching_files:
        first_file_path = os.path.join(
            frame_output_dir, f"{folder_prefix}Matching_Results_new_{start_frame}_{start_frame + 1}.xlsx"
        )
        data_final = load_and_process_data(first_file_path)
        data_final.columns = ["Index", f"Frame{start_frame}", f"Frame{start_frame + 1}"]

        all_new_cols = [f"Frame{i + 1}" for i in range(start_frame + 1, start_frame + files_count - 1)]
        extra_columns_df = pd.DataFrame({col: [None] * len(data_final) for col in all_new_cols})
        data_final = pd.concat([data_final, extra_columns_df], axis=1)

        for i in range(start_frame + 1, start_frame + files_count - 1):
            next_file_path = os.path.join(frame_output_dir, f"{folder_prefix}Matching_Results_new_{i}_{i + 1}.xlsx")
            data_next = load_and_process_data(next_file_path)
            data_next.columns = ["Index", f"Frame{i}", f"Frame{i + 1}"]

            for _, row in data_next.iterrows():
                match_index = data_final[data_final[f"Frame{i}"] == row[f"Frame{i}"]].index
                if not match_index.empty:
                    data_final.at[match_index[0], f"Frame{i + 1}"] = row[f"Frame{i + 1}"]
    else:
        data_final = pd.DataFrame()

    data_final.to_excel(os.path.join(frame_output_dir, f"{folder_prefix}merged_tracking_results.xlsx"), index=False)
    return data_final


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

    return _merge_tracking_pairs(frame_output_dir, folder_prefix)


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
    parser.add_argument("--cells-info", required=True, help="Cells_info.xlsx path")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    cells_info_df = pd.read_excel(args.cells_info)

    def _print_progress(message: str, percent: int) -> None:
        print(f"[{percent:3d}%] {message}")

    merged = track_all_populations(cells_info_df, args.output_dir, progress_cb=_print_progress)
    print(f"\nMerged tracking shape: {merged.shape}")
    print(f"Columns: {list(merged.columns)}")
