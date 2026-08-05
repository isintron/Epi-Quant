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

import numpy as np
import pandas as pd


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
