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

**Round 17 adds the third rule, and it is the structural one:**

3. **Baselines live where the application never writes.** They used to sit
   inside `test_data/<dataset>/cell_track_output/` -- which is exactly where the
   app writes on every Extract / Classify / Track run, including the user's
   routine smoke tests. That single fact caused four of the six occurrences in
   `v2/docs/shared-directory-cost.md`, two of them wrong conclusions rather than
   just wasted rounds. They now live under `v2/baseline_data/<slug>/`, a tree no
   product code path writes to. `test_data/` is read-only to this module: masks
   are read from it and nothing is ever written back.

`Cells_info.xlsx` is pinned alongside the tracking table, because it is
*tracking's input*. The trigger for this round proves the need: a re-extraction
by the user moved a pinned constant from 4.89% to 4.83% with nothing in the
tracker changed, and it surfaced as a tracking regression. Extraction now has
its own pinned artefact, so that change is caught where it belongs.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

import cell_tracking
import data_input

# The knobs that change which cells link. A baseline is only a baseline for one
# assignment of these; anything outside this set (progress callbacks, output
# paths) cannot change the table.
TRACKING_PARAMETERS = (
    "overlap_weight", "centroid_weight", "motion_compensation", "max_frame_gap",
    "cost_form", "overlap_power", "no_link_ratio", "four_quadrant", "drift_mode",
    "feature_columns", "weights",
)

# Where the app writes its own outputs. Nothing in this module may read a
# baseline from, or write one to, a path containing one of these -- that is the
# defect `CLAUDE.md` rule 7 exists to prevent, and `_assert_outside_app_output`
# enforces it rather than trusting the paths to stay right.
APP_OUTPUT_DIRS = ("cell_classification_output", "cell_track_output",
                   "quantitative_analysis_output")

# A tree the application never writes to.
BASELINE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_data")

TRACKING_TABLE_NAME = "tracking.xlsx"
CELLS_INFO_NAME = "Cells_info.xlsx"
PROVENANCE_NAME = "provenance.json"


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


def mask_fingerprint(npy_folder: Optional[str]) -> Optional[dict]:
    """A content hash of the segmentation arrays a run will actually read.

    Recording the resolved PATH was not enough, and Round 18 found out how:
    GBE's mask TIFFs were replaced and `npy_converted/` rebuilt while the
    pinned extraction stayed put, so the tracker linked cells from the old
    segmentation while every judge measured overlap on the new masks. The
    extraction fingerprint passed, every parameter was correct, the path was
    correct -- and `iou_agreement` moved from 4.83% to 22.61% with byte-
    identical code. `CLAUDE.md` rule 7 case 3, exactly: record a fingerprint of
    what was actually read.

    Hashes the label arrays, not the files: a re-conversion that produces the
    same arrays is not a change, and should not invalidate a baseline.
    """
    if not npy_folder or not os.path.isdir(npy_folder):
        return None
    import numpy as np
    import data_input
    digest = hashlib.sha256()
    frames = 0
    labels = 0
    for name, number in sorted(
            data_input.resolve_frame_numbers(data_input.list_segmentation_files(npy_folder)).items(),
            key=lambda item: item[1]):
        data = np.load(os.path.join(npy_folder, name), allow_pickle=True).item()
        array = data.get("masks")
        if array is None:
            continue
        digest.update(str(number).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
        frames += 1
        labels += int(len(np.unique(array)) - 1)
    if not frames:
        return None
    return {"sha256": digest.hexdigest()[:32], "frames": frames, "labels": labels}


def _resolve_npy_folder(dataset_dir: str) -> Optional[str]:
    """Where the label ARRAYS live, which for a TIFF dataset is not `masks/`.

    `masks/` on a TIFF dataset holds only `.tif` files; the extractor converts
    them into `masks/npy_converted/` and that is what the product's
    `npy_folder` points at. Pointing a baseline run at `masks/` instead passes
    `_FrameLabelLoader.__bool__` (the folder exists) and then finds no arrays,
    so the mask-overlap term -- weight 1.0 by default -- silently does nothing
    and the run is centroid-only.

    **This is not hypothetical: it is what happened.** The GBE baseline written
    in Round 15 came out at 414 tracks; the same defaults with the arrays
    actually reachable give 418. A baseline produced without the overlap term
    is not a baseline for the shipped defaults, which is precisely what
    provenance is supposed to prevent -- and the mechanism could not catch it,
    because the parameters it records were all correct. Only the inputs were
    wrong.
    """
    masks = os.path.join(dataset_dir, "masks")
    if not os.path.isdir(masks):
        return None
    if any(name.endswith(".npy") for name in os.listdir(masks)):
        return masks
    converted = os.path.join(masks, "npy_converted")
    if os.path.isdir(converted) and any(n.endswith(".npy") for n in os.listdir(converted)):
        return converted
    return None


def slug(dataset_dir: str) -> str:
    """A filesystem-safe name for the dataset, e.g. `AS_dorsal_view_E-cad`."""
    name = os.path.basename(os.path.normpath(dataset_dir))
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in name)


def _assert_outside_app_output(path: str) -> str:
    """Refuse any baseline path that lands where the app writes.

    An assertion rather than a convention: the paths were right when they were
    written last time too, and then a directory layout changed underneath them.
    """
    parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
    clash = [d for d in APP_OUTPUT_DIRS if d in parts]
    if clash:
        raise ValueError(
            f"{path!r} is inside {clash[0]!r}, which the application rewrites on every run. "
            "A fixture a normal day's work can overwrite is not a fixture (CLAUDE.md rule 7)."
        )
    return path


def baseline_dir(dataset_dir: str) -> str:
    return _assert_outside_app_output(os.path.join(BASELINE_ROOT, slug(dataset_dir)))


def baseline_path(dataset_dir: str) -> str:
    return os.path.join(baseline_dir(dataset_dir), TRACKING_TABLE_NAME)


def cells_info_path(dataset_dir: str) -> str:
    """The PINNED extraction the baseline tracking was produced from."""
    return os.path.join(baseline_dir(dataset_dir), CELLS_INFO_NAME)


def provenance_path(dataset_dir: str) -> str:
    return os.path.join(baseline_dir(dataset_dir), PROVENANCE_NAME)


def dataset_extraction_path(dataset_dir: str) -> str:
    """Where the APP writes its extraction -- read only when adopting."""
    return os.path.join(dataset_dir, "cell_classification_output", "Cells_info.xlsx")


def adopt_extraction(dataset_dir: str, source: Optional[str] = None) -> dict:
    """Copy the dataset's current extraction into the baseline tree, pinning it.

    Deliberately a separate, explicit act: after this, `regenerate` reads the
    pinned copy and never touches `cell_classification_output/` again, so a
    later re-extraction by the user cannot silently change what a baseline was
    measured on.
    """
    source = source or dataset_extraction_path(dataset_dir)
    if not os.path.exists(source):
        raise FileNotFoundError(f"no extraction to adopt at {source}")
    os.makedirs(baseline_dir(dataset_dir), exist_ok=True)
    shutil.copy2(source, cells_info_path(dataset_dir))
    table = pd.read_excel(cells_info_path(dataset_dir))
    return {"dataset": slug(dataset_dir), "adopted_from": source,
            "cells": len(table), "sha256": _table_hash(table)}


def run_pipeline(dataset_dir: str, npy_folder: Optional[str] = None,
                 cells_info: Optional[pd.DataFrame] = None, **overrides):
    """Track `dataset_dir` into a THROWAWAY directory and return the table.

    Every output goes to a temp dir that is deleted on the way out, so a
    regression run leaves `test_data/` byte-identical. Masks are read from the
    dataset; nothing is written back to it.
    """
    if npy_folder is None:
        npy_folder = _resolve_npy_folder(dataset_dir)
    if cells_info is None:
        cells_info = pd.read_excel(cells_info_path(dataset_dir))
    with tempfile.TemporaryDirectory() as scratch:
        return cell_tracking.track_all_populations(
            cells_info.copy(), scratch, npy_folder=npy_folder, **overrides
        ), npy_folder


def regenerate(dataset_dir: str, npy_folder: Optional[str] = None, **overrides) -> dict:
    """Run the pipeline and write the baseline table, its pinned input and its
    provenance, all under `v2/baseline_data/<slug>/`.

    The only supported way to produce a baseline. Reads the PINNED
    `Cells_info.xlsx`; call `adopt_extraction` first if there is not one yet.
    """
    unknown = set(overrides) - set(TRACKING_PARAMETERS)
    if unknown:
        raise ValueError(
            "these are not tracking parameters, so a baseline cannot be scoped by them: "
            + ", ".join(sorted(unknown))
        )
    if not os.path.exists(cells_info_path(dataset_dir)):
        raise FileNotFoundError(
            f"no pinned extraction at {cells_info_path(dataset_dir)} -- run "
            f"adopt_extraction({dataset_dir!r}) first. A baseline is only a baseline for the "
            "inputs it was measured on."
        )

    parameters = pipeline_defaults()
    parameters.update(overrides)

    cells_info = pd.read_excel(cells_info_path(dataset_dir))
    table, resolved_npy = run_pipeline(dataset_dir, npy_folder=npy_folder,
                                       cells_info=cells_info, **overrides)

    os.makedirs(baseline_dir(dataset_dir), exist_ok=True)
    table.to_excel(_assert_outside_app_output(baseline_path(dataset_dir)), index=False)

    record = {
        "dataset": os.path.basename(os.path.normpath(dataset_dir)),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "produced_by": "cell_tracking.track_all_populations via v2/baselines.regenerate",
        # The RESOLVED path, not just whether one was given: a baseline run that
        # pointed at a folder with no arrays in it is indistinguishable from one
        # that pointed at the right folder unless the path itself is recorded --
        # and the arrays being absent is exactly what happened (rule 7 case 3).
        "npy_folder": resolved_npy,
        "npy_arrays_found": bool(cell_tracking._FrameLabelLoader(resolved_npy)),
        # The arrays themselves, not just where they were. See mask_fingerprint.
        "masks": mask_fingerprint(resolved_npy),
        # The label set a run saw is not only which files it read: a
        # minimum-area threshold silently changes it, and a table produced with
        # the filter on is indistinguishable from one produced with it off
        # unless the threshold is recorded (Round 21 A2).
        "min_cell_area_px": data_input.get_min_cell_area(),
        # Round 26: the same movie yields a different row set -- and therefore
        # different tracks -- depending on whether border cells were dropped at
        # extraction (before Round 26) or only filtered in analysis (after it).
        # A table that does not say which cannot be compared with one that does.
        "exclude_edge_cells": data_input.get_edge_exclusion(),
        "mask_source": os.path.join(dataset_dir, "masks"),
        "parameters": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                       for k, v in parameters.items()},
        "shape": list(table.shape),
        "table_sha256": _table_hash(table),
        "cells_info_sha256": _table_hash(cells_info),
        "cells_info_rows": int(len(cells_info)),
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

    # The INPUT is checked too. A baseline is a statement about one extraction,
    # and re-extraction moved a pinned constant once already while nothing in
    # the tracker changed -- which read as a tracking regression.
    pinned = cells_info_path(dataset_dir)
    if not os.path.exists(pinned):
        raise StaleBaseline(
            f"{pinned} is missing -- the baseline's own input is gone, so nothing can be "
            "reproduced from it. Re-run adopt_extraction then regenerate."
        )
    if _table_hash(pd.read_excel(pinned)) != record.get("cells_info_sha256"):
        raise StaleBaseline(
            f"{pinned} no longer matches its provenance hash: the pinned EXTRACTION changed, so "
            "this is not a baseline for the current inputs. That is an extraction change, not a "
            "tracking regression."
        )

    # And the MASKS themselves. A baseline is a statement about one
    # segmentation as much as one extraction; swapping the masks under it moved
    # GBE's agreement from 4.83% to 22.61% with nothing else changed, and every
    # other check passed while it happened.
    stamped_masks = record.get("masks")
    if stamped_masks:
        current = mask_fingerprint(record.get("npy_folder"))
        if current is None:
            raise StaleBaseline(
                f"the segmentation arrays this baseline was measured on are no longer readable at "
                f"{record.get('npy_folder')!r}."
            )
        if current["sha256"] != stamped_masks["sha256"]:
            raise StaleBaseline(
                f"the SEGMENTATION changed under this baseline "
                f"({stamped_masks['frames']} frames / {stamped_masks['labels']} labels when it was "
                f"written, {current['frames']} / {current['labels']} now). That is a segmentation "
                "change, not a tracking regression -- re-run Extract so the pinned extraction "
                "matches these masks, then adopt and regenerate."
            )

    # And the minimum-area threshold. Same class as the mask check above: the
    # inputs are the same files, but the label set derived from them is not.
    stamped_area = record.get("min_cell_area_px")
    if stamped_area is not None and int(stamped_area) != data_input.get_min_cell_area():
        raise StaleBaseline(
            f"this baseline was produced with a minimum cell area of {stamped_area} px and the "
            f"current setting is {data_input.get_min_cell_area()} px. That changes which labels "
            "exist, so it is not a baseline for these settings -- not a regression."
        )

    # And the arrays: a run that found none is centroid-only whatever its
    # recorded parameters say (rule 7 case 3).
    if record.get("npy_arrays_found") is False:
        raise StaleBaseline(
            f"the baseline for {record.get('dataset')} was produced with NO segmentation arrays "
            f"reachable at {record.get('npy_folder')!r}, so its mask-overlap term contributed "
            "nothing regardless of the weight recorded. Regenerate it."
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

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adopt", metavar="DATASET_DIR", nargs="+",
                        help="pin the dataset's current extraction into the baseline tree")
    parser.add_argument("--regenerate", metavar="DATASET_DIR", nargs="+",
                        help="run the pipeline and rewrite these baselines and their provenance")
    parser.add_argument("--check", metavar="DATASET_DIR", nargs="+",
                        help="verify stored baselines against the current defaults")
    parser.add_argument("--drift-mode", default=None,
                        help="regenerate under a non-default drift_mode")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.adopt:
        for directory in args.adopt:
            info = adopt_extraction(directory)
            print(f"{info['dataset']}: pinned {info['cells']} cells from {info['adopted_from']}")
    if args.regenerate:
        overrides = {} if args.drift_mode is None else {"drift_mode": args.drift_mode}
        for directory in args.regenerate:
            record = regenerate(directory, **overrides)
            print(f"{record['dataset']}: {record['shape'][0]} tracks, "
                  f"arrays_found={record['npy_arrays_found']}, "
                  f"npy={record['npy_folder']}, table_sha256={record['table_sha256'][:12]}...")
    if args.check:
        for directory in args.check:
            try:
                record = check(directory)
                print(f"{slug(directory)}: OK ({record['shape'][0]} tracks, "
                      f"{record['cells_info_rows']} cells, written {record['written_utc']})")
            except (StaleBaseline, FileNotFoundError) as error:
                print(f"{slug(directory)}: {error}")

    if args.self_test:
        import filecmp

        def _fingerprint(root):
            """Every file under `root`, by path and content hash."""
            out = {}
            for base, _dirs, files in os.walk(root):
                for name in sorted(files):
                    full = os.path.join(base, name)
                    with open(full, "rb") as handle:
                        out[os.path.relpath(full, root)] = hashlib.sha256(handle.read()).hexdigest()
            return out

        # A synthetic dataset shaped like the real ones, INCLUDING the app
        # output directory that must never be touched.
        sandbox = tempfile.mkdtemp()
        dataset = os.path.join(sandbox, "test_data", "Fake_dataset")
        os.makedirs(os.path.join(dataset, "cell_classification_output"))
        os.makedirs(os.path.join(dataset, "masks"))
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
            os.path.join(dataset, "cell_classification_output", "Cells_info.xlsx"), index=False)
        # Real arrays, so the ordinary checks below exercise the ordinary path.
        import numpy as _np
        for frame in (1, 2, 3):
            labels = _np.zeros((40, 460), dtype=int)
            for i, x in enumerate((10, 200, 400), start=1):
                labels[10:30, x + 2 * frame:x + 2 * frame + 20] = i
            _np.save(os.path.join(dataset, "masks", "f%03d_seg.npy" % frame),
                     {"masks": labels}, allow_pickle=True)

        # Point BASELINE_ROOT at the sandbox so the real tree is untouched.
        BASELINE_ROOT = os.path.join(sandbox, "baseline_data")

        # --- B1 (R17): no baseline path may land where the app writes.
        for path in (baseline_dir(dataset), baseline_path(dataset),
                     cells_info_path(dataset), provenance_path(dataset)):
            parts = os.path.normpath(path).split(os.sep)
            assert not any(d in parts for d in APP_OUTPUT_DIRS), path
        for bad in (os.path.join(dataset, "cell_track_output", "x.xlsx"),
                    os.path.join(dataset, "cell_classification_output")):
            try:
                _assert_outside_app_output(bad)
            except ValueError:
                continue
            raise AssertionError("a path inside an app output directory must be refused: %s" % bad)
        print("no baseline path resolves inside an application output directory,")
        print("  and one that would is refused rather than merely discouraged.")

        # --- A regression run must leave the dataset byte-identical.
        before = _fingerprint(dataset)
        adopt_extraction(dataset)
        written = regenerate(dataset, max_frame_gap=1)
        run_pipeline(dataset)
        after = _fingerprint(dataset)
        assert before == after, {
            "added": sorted(set(after) - set(before)),
            "changed": sorted(k for k in before if k in after and before[k] != after[k]),
        }
        print("adopt + regenerate + a regression run leave the dataset directory byte-identical")
        print("  (%d files, all unchanged)." % len(before))

        assert written["parameters"]["max_frame_gap"] == 1
        check(dataset, max_frame_gap=1)
        print("a baseline verifies under the settings it was produced with.")

        try:
            check(dataset)  # the current defaults, which use max_frame_gap=0
        except StaleBaseline as error:
            assert "this is not a baseline for these settings" in str(error), error
            assert "max_frame_gap" in str(error), error
            print("...and is REFUSED under different settings, naming max_frame_gap:")
            print("   ", str(error).split(". ")[0])
        else:
            raise AssertionError("a baseline for other settings must be refused")

        # --- The pinned EXTRACTION is checked too: changing it is an input
        # change, not a tracking regression, and must say so.
        pinned = pd.read_excel(cells_info_path(dataset))
        pinned.loc[0, "Center X Coordinate"] = 999.0
        pinned.to_excel(cells_info_path(dataset), index=False)
        try:
            check(dataset, max_frame_gap=1)
        except StaleBaseline as error:
            assert "extraction change, not a tracking regression" in str(error), error
            print("a changed pinned extraction is reported as an INPUT change, not a regression.")
        else:
            raise AssertionError("a changed extraction must be refused")
        adopt_extraction(dataset)
        regenerate(dataset, max_frame_gap=1)

        # --- An out-of-band edit to the table itself.
        table = pd.read_excel(baseline_path(dataset))
        table.iloc[0, 1] = "Cell1_999"
        table.to_excel(baseline_path(dataset), index=False)
        try:
            check(dataset, max_frame_gap=1)
        except StaleBaseline as error:
            assert "out of band" in str(error), error
            print("an out-of-band edit stops it being a baseline, and says so.")
        else:
            raise AssertionError("an edited table must not pass provenance")

        # --- A baseline produced with no arrays reachable is refused outright,
        # whatever its parameters say (rule 7 case 3). Shaped exactly like the
        # real failure: a TIFF dataset whose masks/ holds no arrays.
        tiff_dataset = os.path.join(sandbox, "test_data", "Fake_tiff_dataset")
        os.makedirs(os.path.join(tiff_dataset, "cell_classification_output"))
        os.makedirs(os.path.join(tiff_dataset, "masks"))
        open(os.path.join(tiff_dataset, "masks", "f001.tif"), "wb").close()
        pd.DataFrame(rows).to_excel(
            os.path.join(tiff_dataset, "cell_classification_output", "Cells_info.xlsx"), index=False)
        adopt_extraction(tiff_dataset)
        centroid_only = regenerate(tiff_dataset)
        assert centroid_only["npy_arrays_found"] is False, centroid_only
        try:
            check(tiff_dataset)
        except StaleBaseline as error:
            assert "NO segmentation arrays reachable" in str(error), error
            print("a baseline whose overlap term had nothing to read is refused, even though")
            print("  every parameter it recorded was correct.")
        else:
            raise AssertionError("a centroid-only baseline must not pass as a defaults baseline")

        # --- A2 (R21): a differing minimum-area threshold is a differing LABEL
        # SET, and must be refused the same way differing masks are. The files
        # on disk are identical; what a run derives from them is not.
        data_input.set_min_cell_area(0)
        regenerate(dataset, max_frame_gap=1)
        check(dataset, max_frame_gap=1)
        data_input.set_min_cell_area(19)
        try:
            check(dataset, max_frame_gap=1)
        except StaleBaseline as error:
            assert "minimum cell area" in str(error), error
            print("a differing minimum-area threshold is refused as a settings mismatch:")
            print("   ", str(error).split(". ")[0])
        else:
            raise AssertionError("a differing minimum cell area must be refused")
        data_input.set_min_cell_area(0)
        check(dataset, max_frame_gap=1)

        try:
            regenerate(dataset, dilation_px=3)
        except ValueError as error:
            assert "not tracking parameters" in str(error), error
            print("a baseline cannot be scoped by a parameter that does not affect linking.")
        else:
            raise AssertionError("unknown parameters must be refused")

        print("\nAll baselines self-tests passed.")
