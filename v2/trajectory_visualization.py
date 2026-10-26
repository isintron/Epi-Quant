"""Module 5: cell trajectory + lifetime visualization.

Ported from `gui_plot_track.MainWindow` ("Cell Trajectory Visualization",
src/gui_plot_track.py) and `gui_cell_direction.CellDirection` ("Display Cell
Displacement Directions", src/gui_cell_direction.py).

Confirmed during exploration: v1's trajectory visualization does NOT compute
trajectories at all — it reads already-tracked cell IDs per frame and just
re-highlights that cell's outline in each frame's image (no centroid-to-
centroid path is ever drawn). The one file that does real computation is
`gui_cell_direction.py`: per-cell displacement vectors between two chosen
frames, then a `scipy.interpolate.griddata` interpolation for streamline
rendering.

`compute_lifetime_map` is a SCAFFOLD ONLY (per your decision) — cell-lifetime
computation/visualization doesn't exist anywhere in v1 (neither
`gui_plot_track.py` nor `gui_cell_direction.py` computes it), so there is no
v1 logic to port here; a real implementation needs its own design pass.

Note (not one of the 4 agreed bug fixes, preserved as-is): v1's frame-column
resolution in `plot_displacement_arrows` (`[c for c in column_names if
str(start_frame) in c]`) is a substring match, not an exact one -- e.g.
`start_frame=1` would also match a `'Frame10'` column if it happened to sort
before `'Frame1'`. In the normal ascending Frame1..FrameN column order this
never bites, but it's fragile. Reproduced verbatim in
`compute_displacement_vectors` below; flagged for your future consideration.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def get_track_cell_ids(tracking_data: pd.DataFrame, track_number: int) -> list[str]:
    """Return the list of `Cell{frame}_{id}` strings for `track_number`
    across all frames (its row in `tracking_data`, first column = track
    number, remaining columns = per-frame cell IDs). Returns `[]` if the
    track isn't found. Ported from `MainWindow.plot_track`,
    src/gui_plot_track.py:273-284.
    """
    row = tracking_data[tracking_data.iloc[:, 0] == track_number]
    if row.empty:
        return []
    row = row.iloc[0]
    return list(row[1:].dropna().values)


def get_cell_outline_coords(outlines: np.ndarray, cell_id: int) -> np.ndarray:
    """Pixel coordinates (rows of `[y, x]`) belonging to `cell_id` in a
    labeled segmentation mask `outlines`. Ported from the repeated
    `np.argwhere(outlines == cell_number)` pattern in
    `MainWindow.plot_all_frames`, src/gui_plot_track.py:437.
    """
    return np.argwhere(outlines == cell_id)


def get_cell_center(cells_info: pd.DataFrame, cell_id: str) -> Optional[tuple[float, float]]:
    """Look up `cell_id`'s centroid in `cells_info` (columns `Cell Index`,
    `Center X/Y Coordinate`); returns `(y, x)` or `None` if not found.
    Verbatim port of `CellDirection.get_cell_center`,
    src/gui_cell_direction.py:231-236.
    """
    row = cells_info[cells_info["Cell Index"] == cell_id]
    if not row.empty:
        x = row["Center X Coordinate"].values[0]
        y = row["Center Y Coordinate"].values[0]
        return (y, x)
    return None


def compute_displacement_vectors(
    tracking_data: pd.DataFrame,
    cells_info: pd.DataFrame,
    start_frame: int,
    end_frame: int,
) -> pd.DataFrame:
    """For every tracked cell present in both `start_frame` and `end_frame`,
    compute its centroid displacement `(dX, dY)`. Returns a DataFrame with
    columns `Cell ID, Frame Start, Frame End, Current Y, Current X, Next Y,
    Next X, dY, dX` -- the same schema v1 exported to
    `cell_centers_frame_{start}_{end}.xlsx`.

    Extracted from the pure vector-field core of
    `CellDirection.plot_displacement_arrows`, src/gui_cell_direction.py:85-222
    (specifically ~136-177), excluding all `ax.annotate`/canvas/save-dialog
    calls. Raises `ValueError` if no column matches `start_frame`/`end_frame`
    (v1 showed a `QMessageBox.warning` instead).
    """
    column_names = tracking_data.columns[1:]
    start_col = [c for c in column_names if str(start_frame) in c]
    end_col = [c for c in column_names if str(end_frame) in c]
    if not start_col or not end_col:
        raise ValueError("Invalid frame selection: no matching Frame column found for start_frame/end_frame.")

    current_column = start_col[0]
    next_column = end_col[0]

    rows = []
    for _, row in tracking_data.iterrows():
        current_cell_id = row[current_column]
        next_cell_id = row[next_column]
        if pd.isna(current_cell_id) or pd.isna(next_cell_id):
            continue

        current_center = get_cell_center(cells_info, current_cell_id)
        next_center = get_cell_center(cells_info, next_cell_id)
        if current_center is None or next_center is None:
            continue

        dx = next_center[1] - current_center[1]
        dy = next_center[0] - current_center[0]

        rows.append({
            "Cell ID": current_cell_id,
            "Frame Start": start_frame,
            "Frame End": end_frame,
            "Current Y": current_center[0],
            "Current X": current_center[1],
            "Next Y": next_center[0],
            "Next X": next_center[1],
            "dY": dy,
            "dX": dx,
        })

    return pd.DataFrame(rows)


def interpolate_vector_field(
    displacement_df: pd.DataFrame, grid_size: int = 100, fill_value: float = np.nan,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate the sparse per-cell displacement vectors in
    `displacement_df` (as returned by `compute_displacement_vectors`) onto a
    `grid_size x grid_size` regular grid via cubic `scipy.interpolate.griddata`,
    for streamline rendering. Returns `(X, Y, U, V)` meshgrids.

    Extracted from `CellDirection.plot_displacement_arrows`,
    src/gui_cell_direction.py:187-201, decoupled from `ax.streamplot`.

    **`fill_value` was `0` until Round 34.** Cubic `griddata` returns nothing
    outside the convex hull of the input points, and filling that with zero
    says "measured, and stationary" for every grid node outside the tissue --
    so `streamplot` drew flow slowing to a halt at the tissue edge, which looks
    like biology and is an artefact of the interpolation. `np.nan` is what
    `streamplot` reads as "no data here". Pass `0.0` to restore the old
    behaviour; nothing in the product does.
    """
    x_arr = displacement_df["Current X"].to_numpy(dtype=float)
    y_arr = displacement_df["Current Y"].to_numpy(dtype=float)
    u_arr = displacement_df["dX"].to_numpy(dtype=float)
    v_arr = displacement_df["dY"].to_numpy(dtype=float)

    xi = np.linspace(x_arr.min(), x_arr.max(), grid_size)
    yi = np.linspace(y_arr.min(), y_arr.max(), grid_size)
    X, Y = np.meshgrid(xi, yi)
    U = griddata((x_arr, y_arr), u_arr, (X, Y), method="cubic", fill_value=fill_value)
    V = griddata((x_arr, y_arr), v_arr, (X, Y), method="cubic", fill_value=fill_value)
    return X, Y, U, V


def compute_lifetime_map(tracking_data: pd.DataFrame, cells_info: pd.DataFrame, npy_folder: str) -> pd.DataFrame:
    """Scaffold only -- NOT YET IMPLEMENTED.

    Intended to compute, per tracked cell, the frame span it persists before
    division/extrusion/exiting the frame, for a spatial lifetime-map
    visualization (manuscript Fig. 4F: peripheral AS cells near the leading
    edge extrude earlier, central AS cells extrude later).

    No v1 logic exists to port here (confirmed: neither `gui_plot_track.py`
    nor `gui_cell_direction.py` computes cell lifetime). Before implementing,
    confirm the exact expected output shape/columns (e.g. per-track
    first/last frame + centroid, vs. a per-pixel spatial map) and how
    extrusion is distinguished from a cell simply exiting the field of view.
    """
    raise NotImplementedError(
        "compute_lifetime_map is a scaffold only -- see docstring for what needs to be confirmed before implementing."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standalone debug entry for v2.trajectory_visualization")
    parser.add_argument("--tracking-data", required=True, help="all_cell_merged_tracking_results.xlsx path")
    parser.add_argument("--cells-info", required=True, help="Cells_info.xlsx path")
    parser.add_argument("--track-number", type=int, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    args = parser.parse_args()

    tracking_df = pd.read_excel(args.tracking_data, sheet_name="Sheet1")
    cells_info_df = pd.read_excel(args.cells_info, sheet_name="Cell")

    cell_ids = get_track_cell_ids(tracking_df, args.track_number)
    print(f"Track {args.track_number} cell IDs: {cell_ids}")

    displacement_df = compute_displacement_vectors(tracking_df, cells_info_df, args.start_frame, args.end_frame)
    print(f"\nDisplacement vectors ({len(displacement_df)} cells):")
    print(displacement_df.head())

    if not displacement_df.empty:
        X, Y, U, V = interpolate_vector_field(displacement_df)
        print(f"\nInterpolated grid shape: X={X.shape}, U={U.shape}")

    try:
        compute_lifetime_map(tracking_df, cells_info_df, npy_folder="")
    except NotImplementedError as e:
        print(f"\ncompute_lifetime_map scaffold wiring OK (raised as expected): {e}")
