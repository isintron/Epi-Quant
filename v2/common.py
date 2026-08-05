"""Shared, PyQt5-free helpers used by more than one v2 module.

These are intentionally minimal: v2 modules take their working directories as
explicit function parameters rather than reading shared global state, so
nothing here is imported "implicitly" by other v2 modules. `setup_output_dirs`
exists purely as a CLI/demo convenience for wiring up a fresh output tree.
"""

import os

from openpyxl import load_workbook
import pandas as pd


def setup_output_dirs(base_output_dir: str) -> dict[str, str]:
    """Create (if needed) and return the standard v1-compatible output subfolders
    under `base_output_dir`: cell_classification_output, cell_track_output,
    quantitative_analysis_output. The data-export subfolder is intentionally
    omitted (that module is deferred).

    This is the single canonical replacement for two divergent v1
    implementations: `config.set_paths` (src/config.py:10-28, never actually
    called anywhere) and `gui_main.MainWindow.update_paths_and_create_folders`
    (src/gui_main.py:583-679), which duplicated the same logic inline and also
    set an undeclared `config.output_path` attribute. See v2/README.md's Bug
    Fix Changelog, entry #1.
    """
    paths = {
        "cell_classification_output": os.path.join(base_output_dir, "cell_classification_output"),
        "cell_track_output": os.path.join(base_output_dir, "cell_track_output"),
        "quantitative_analysis_output": os.path.join(base_output_dir, "quantitative_analysis_output"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def save_modified_rows(data: pd.DataFrame, file_path: str) -> pd.DataFrame:
    """Write rows flagged `modified == True` back into an existing .xlsx file in
    place (matching by 0-based DataFrame row position -> Excel row = position + 2),
    then clear the `modified` flag on the rows just saved. Returns `data` (mutated
    in place and returned for convenience).

    Ported verbatim from gui_comparison.py:239-265 (already pure/generic aside
    from its openpyxl I/O). Shared by the classification-correction and
    tracking-correction modules.
    """
    modified_rows = data[data["modified"] == True]  # noqa: E712 (matches v1's explicit bool comparison)

    if not modified_rows.empty:
        workbook = load_workbook(file_path)
        sheet = workbook.active

        col_index = {col_name: idx + 1 for idx, col_name in enumerate(data.columns)}

        for _, row in modified_rows.iterrows():
            excel_row = row.name + 2  # DataFrame row position + 2 (Excel is 1-based, row 1 is the header)
            for col_name, value in row.items():
                if col_name == "modified":
                    continue
                excel_col = col_index[col_name]
                sheet.cell(row=excel_row, column=excel_col, value=value)

        workbook.save(file_path)
        workbook.close()

        data.loc[modified_rows.index, "modified"] = False

    return data


if __name__ == "__main__":
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description="Standalone debug entry for v2.common")
    parser.add_argument("--base-output-dir", default=None, help="Defaults to a fresh temp dir if omitted")
    args = parser.parse_args()

    base_dir = args.base_output_dir or tempfile.mkdtemp(prefix="epiquant_v2_")
    created = setup_output_dirs(base_dir)
    print(f"Base output dir: {base_dir}")
    for name, path in created.items():
        print(f"  {name}: {path} (exists={os.path.isdir(path)})")
