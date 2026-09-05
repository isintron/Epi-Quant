"""Stored regression baselines, and the provenance that makes them baselines.

The same GBE tracking table caused three separate wrong conclusions -- a false
regression (Round 11), a wrong dead-band figure (Round 12), and a failed
regression run (Round 14) -- always for one reason: it had been produced with
drift compensation on (473 tracks) while the pipeline at defaults produces 414.
Nothing recorded that, so nothing could catch it. `CLAUDE.md` pattern 6 already
says a baseline must be pipeline-produced; this module is what enforces it.

Two rules, both mechanical:

1. **A baseline is written only by running the pipeline.** `regenerate()` calls
   `track_all_populations` and writes the table and its
   `<baseline>.provenance.json` in the same breath. There is no path here that
   stamps provenance onto a file someone produced by hand -- that is exactly
   the failure being prevented.
2. **Every comparison checks provenance first.** `load()` refuses a baseline
   whose recorded parameters differ from the settings being compared, with
   *"this is not a baseline for these settings"* and the differing names --
   NOT with a regression failure, which is what sent three rounds chasing the
   wrong thing.

The provenance also carries a hash of the table itself, so a file edited or
regenerated out of band stops being a baseline the moment it is touched, and
says so.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

import cell_tracking

# The knobs that change which cells link. A baseline is only a baseline for one
# assignment of these; anything outside this set (progress callbacks, output
# paths) cannot change the table.
TRACKING_PARAMETERS = (
    "overlap_weight", "centroid_weight", "motion_compensation", "max_frame_gap",
    "cost_form", "overlap_power", "no_link_ratio", "four_quadrant", "drift_mode",
    "feature_columns", "weights",
)

BASELINE_RELATIVE_PATH = os.path.join(
    "cell_track_output", "All_cell_tracking", "All_cell_merged_tracking_results.xlsx",
)


class StaleBaseline(Exception):
    """Raised instead of a regression failure when the stored table was not
    produced under the settings it is being compared against."""


def pipeline_defaults() -> dict:
    """The current default for every tracking parameter, read from the
    signature rather than restated here -- so changing a default cannot leave
    this list quietly describing the old one."""
    signature = inspect.signature(cell_tracking.track_all_populations)
    return {name: signature.parameters[name].default for name in TRACKING_PARAMETERS}


def _code_state() -> dict:
    def git(*args):
        try:
            return subprocess.run(("git",) + args, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None

    module_path = os.path.abspath(cell_tracking.__file__)
    with open(module_path, "rb") as handle:
        module_hash = hashlib.sha256(handle.read()).hexdigest()
    return {
        "commit": git("rev-parse", "HEAD"),
        # A commit alone is not the code that ran when the tree is dirty, which
        # it usually is mid-round -- so hash the module too.
        "tree_dirty": bool(git("status", "--porcelain")),
        "cell_tracking_sha256": module_hash,
    }


def _table_hash(table: pd.DataFrame) -> str:
    return hashlib.sha256(
        table.fillna("\x00").astype(str).to_csv(index=False).encode("utf-8")
    ).hexdigest()


def baseline_path(dataset_dir: str) -> str:
    return os.path.join(dataset_dir, BASELINE_RELATIVE_PATH)


def provenance_path(dataset_dir: str) -> str:
    return baseline_path(dataset_dir) + ".provenance.json"


def regenerate(dataset_dir: str, cells_info_path: Optional[str] = None,
               npy_folder: Optional[str] = None, **overrides) -> dict:
    """Run the pipeline on `dataset_dir` and write BOTH the baseline table and
    its provenance. The only supported way to produce a baseline."""
    if cells_info_path is None:
        cells_info_path = os.path.join(dataset_dir, "cell_classification_output", "Cells_info.xlsx")
    if npy_folder is None:
        candidate = os.path.join(dataset_dir, "masks")
        npy_folder = candidate if os.path.isdir(candidate) else None

    unknown = set(overrides) - set(TRACKING_PARAMETERS)
    if unknown:
        raise ValueError(
            "these are not tracking parameters, so a baseline cannot be scoped by them: "
            + ", ".join(sorted(unknown))
        )

    parameters = pipeline_defaults()
    parameters.update(overrides)

    cells_info = pd.read_excel(cells_info_path)
    output_dir = os.path.join(dataset_dir, "cell_track_output")
    os.makedirs(output_dir, exist_ok=True)
    table = cell_tracking.track_all_populations(
        cells_info, output_dir, npy_folder=npy_folder, **overrides
    )

    record = {
        "dataset": os.path.basename(os.path.normpath(dataset_dir)),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "produced_by": "cell_tracking.track_all_populations via v2/baselines.regenerate",
        "npy_folder_supplied": npy_folder is not None,
        "parameters": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                       for k, v in parameters.items()},
        "shape": list(table.shape),
        "table_sha256": _table_hash(table),
        "code": _code_state(),
    }
    with open(provenance_path(dataset_dir), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return record


def check(dataset_dir: str, **settings) -> dict:
    """Verify the stored baseline is a baseline for `settings` (defaults where
    unspecified). Raises `StaleBaseline` naming what differs."""
    path = provenance_path(dataset_dir)
    if not os.path.exists(path):
        raise StaleBaseline(
            f"{baseline_path(dataset_dir)} has no provenance, so it is not a baseline for any "
            "settings -- regenerate it with baselines.regenerate()."
        )
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)

    wanted = pipeline_defaults()
    wanted.update(settings)
    stored = record.get("parameters", {})
    differing = sorted(
        name for name in TRACKING_PARAMETERS
        if _normalise(stored.get(name)) != _normalise(wanted.get(name))
    )
    if differing:
        detail = "; ".join(
            f"{name}: baseline={stored.get(name)!r} requested={wanted.get(name)!r}"
            for name in differing
        )
        raise StaleBaseline(
            f"this is not a baseline for these settings ({record.get('dataset')}): {detail}. "
            "This is NOT a regression -- regenerate the baseline, or compare under the "
            "settings it was produced with."
        )

    table = pd.read_excel(baseline_path(dataset_dir))
    if _table_hash(table) != record.get("table_sha256"):
        raise StaleBaseline(
            f"{baseline_path(dataset_dir)} no longer matches its provenance hash -- it was edited "
            "or regenerated out of band, which is precisely how the GBE table caused three wrong "
            "conclusions. Regenerate it with baselines.regenerate()."
        )
    return record


def _normalise(value):
    return tuple(value) if isinstance(value, (list, tuple)) else value


def load(dataset_dir: str, **settings) -> tuple:
    """`(table, provenance)` for a baseline that has passed `check`."""
    record = check(dataset_dir, **settings)
    return pd.read_excel(baseline_path(dataset_dir)), record


if __name__ == "__main__":
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regenerate", metavar="DATASET_DIR", nargs="+",
                        help="run the pipeline and rewrite these baselines and their provenance")
    parser.add_argument("--check", metavar="DATASET_DIR", nargs="+",
                        help="verify stored baselines against the current defaults")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.regenerate:
        for directory in args.regenerate:
            record = regenerate(directory)
            print(f"{record['dataset']}: {record['shape'][0]} tracks, "
                  f"table_sha256={record['table_sha256'][:12]}...")
    if args.check:
        for directory in args.check:
            try:
                record = check(directory)
                print(f"{os.path.basename(os.path.normpath(directory))}: OK "
                      f"({record['shape'][0]} tracks, written {record['written_utc']})")
            except StaleBaseline as error:
                print(f"{os.path.basename(os.path.normpath(directory))}: {error}")
    if args.self_test:
        # A baseline written under one setting must be REFUSED under another,
        # with "this is not a baseline for these settings" -- not with a
        # regression failure.
        sandbox = tempfile.mkdtemp()
        os.makedirs(os.path.join(sandbox, "cell_classification_output"))
        rows = []
        for frame in (1, 2, 3):
            for i, x in enumerate((10.0, 200.0, 400.0), start=1):
                row = {column: 1.0 for column in cell_tracking.DEFAULT_TRACKING_FEATURES}
                row.update({"Cell Index": f"Cell{frame}_{i}",
                            "Center X Coordinate": x + 2 * frame, "Center Y Coordinate": 10.0,
                            "Circumscribed Circle Radius": 10.0,
                            "Maximum Horizontal Length": 10.0, "Maximum Vertical Length": 10.0})
                rows.append(row)
        pd.DataFrame(rows).to_excel(
            os.path.join(sandbox, "cell_classification_output", "Cells_info.xlsx"), index=False)

        written = regenerate(sandbox, max_frame_gap=1)
        assert written["parameters"]["max_frame_gap"] == 1
        check(sandbox, max_frame_gap=1)
        print("a baseline verifies under the settings it was produced with.")

        try:
            check(sandbox)  # the current defaults, which use max_frame_gap=0
        except StaleBaseline as error:
            assert "this is not a baseline for these settings" in str(error), error
            assert "max_frame_gap" in str(error), error
            print("...and is REFUSED under different settings, naming max_frame_gap:")
            print("   ", str(error).split(". ")[0])
        else:
            raise AssertionError("a baseline for other settings must be refused")

        # An out-of-band edit -- the actual GBE failure -- invalidates it too.
        table = pd.read_excel(baseline_path(sandbox))
        table.iloc[0, 1] = "Cell1_999"
        table.to_excel(baseline_path(sandbox), index=False)
        try:
            check(sandbox, max_frame_gap=1)
        except StaleBaseline as error:
            assert "out of band" in str(error), error
            print("an out-of-band edit stops it being a baseline, and says so.")
        else:
            raise AssertionError("an edited table must not pass provenance")

        try:
            regenerate(sandbox, dilation_px=3)
        except ValueError as error:
            assert "not tracking parameters" in str(error), error
            print("a baseline cannot be scoped by a parameter that does not affect linking.")
        else:
            raise AssertionError("unknown parameters must be refused")

        print("\nAll baselines self-tests passed.")
