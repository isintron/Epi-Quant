"""EpiQuantAPI: the pywebview `js_api` bridge exposed to the frontend as
`window.pywebview.api.<method>(...)`.

Every method here is a thin wrapper delegating to `v2/*.py`'s pure
functions (`common`, `data_input`, `pca_kmeans`, `manual_correction_*`,
`cell_tracking`, `trajectory_visualization`, `rendering`) -- this file's own
job is session-state management (which paths/DataFrames are "current"),
translating between those functions' DataFrame/numpy outputs and
JSON/base64-PNG-serializable outputs, and background-thread + progress
wiring for long-running calls. See `v2/README.md` and this file's own
`if __name__ == "__main__"` smoke-test harness for how each stage composes.

Method naming convention:
- `start_*`  -- spawns a background thread, returns immediately; progress
  and completion are pushed to the frontend via `window.evaluate_js` calls
  to `window.onProgress`/`window.on<Stage>Done`/`window.on<Stage>Error`.
- `get_*`    -- cheap synchronous read (from cached state or a single
  `.npy`/pixel lookup), safe to call on every frame-navigation click.
- `set_*`    -- session-state-only mutation (e.g. chosen colors), no disk
  I/O, no pipeline computation.
- `save_*`   -- explicit persistence of in-memory corrections to disk.
- everything else (`run_classification`, `reassign_*`, `add_*`) is a
  synchronous pipeline/correction step cheap enough not to need threading.

Session state held on the instance: `mask_folder`/`image_folder`/
`npy_folder`/`output_dirs` (set once at Setup); `cells_info`/
`clustering_data` (DataFrames, cached and mutated in place by corrections --
re-reading Excel on every click would be needlessly slow); `tracking_data`/
`original_tracking_data` (the pre-correction snapshot `update_tracking_data`
needs). Per-frame `outlines`/base-image arrays are deliberately NOT cached
-- a single `.npy` load is cheap, a whole dataset's worth held in memory
isn't, and re-reading avoids ever serving a stale frame after a correction.
"""

import datetime
import json
import logging
import os
import re
import sys
import threading
from typing import Optional, Union

import numpy as np
import pandas as pd
import webview

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import adjacency
import cell_tracking
import census
import common
import coverage_log
import data_input
import manual_correction_classification
import manual_correction_tracking
import pca_kmeans
import rendering
import segmentation_qc
import lineage
import tracking_benchmark
import topology
import shape_features
import tissue_features
import trajectory_visualization

logger = logging.getLogger(__name__)


def _finite(value) -> Optional[float]:
    """`float(value)`, or `None` for NaN/inf -- `json.dumps` emits bare `NaN`
    for those, which `JSON.parse` on the frontend side rejects."""
    number = float(value)
    return number if np.isfinite(number) else None


class EpiQuantAPI:
    def __init__(self):
        self.window = None

        # One id per app run, so a coverage log accumulated over several
        # sittings can still be split by session (fatigue is per-session).
        self.coverage_session_id: str = coverage_log.new_session_id()

        self.mask_folder: Optional[str] = None
        self.image_folder: Optional[str] = None
        self.npy_folder: Optional[str] = None
        self.output_dirs: Optional[dict] = None

        self.cells_info: Optional[pd.DataFrame] = None
        # Shape/tissue descriptors live in their own tables, joined to
        # `cells_info` on `Cell Index` only when classification needs them.
        # They are NOT extra columns in `Cells_info.xlsx`: `common.save_modified_rows`
        # maps DataFrame column position onto Excel column position, so growing
        # that schema would silently write into the wrong columns of an
        # existing file.
        self.cells_shape: Optional[pd.DataFrame] = None
        self.cells_tissue: Optional[pd.DataFrame] = None
        self.clustering_data: Optional[pd.DataFrame] = None
        self.cluster_color_map: dict[int, str] = {}
        self.hidden_cluster_labels: set[int] = set()
        self.show_cluster_borders: bool = False
        # frame_number -> boundary cell indices, from `find_boundary_cells`'s
        # all-pairs KDTree adjacency scan (a few seconds per frame on a
        # ~1000-cell frame) -- cleared whenever clustering changes so a
        # correction can't leave a stale border set behind.
        self._border_cell_cache: dict[int, Optional[set]] = {}

        self.tracking_data: Optional[pd.DataFrame] = None
        self.original_tracking_data: Optional[pd.DataFrame] = None
        # frame_number -> `adjacency.build_neighbor_graph` result, for the
        # environment-score overlay. Unlike `_border_cell_cache` this needs NO
        # invalidation on tracking edits: the graph is a function of the
        # frame's `fill_labels` geometry alone, which never changes once a
        # project is loaded. The tracking-dependent half (translating labels
        # to track ids and computing the Dice score) is cheap set math and is
        # recomputed on every render, so the view is always correct right
        # after a correction. Cleared only when the project changes.
        self._neighbor_graph_cache: dict[int, Optional[dict]] = {}
        # frame_number -> unmodified intensity array for segmentation QC;
        # same file-derived, never-invalidated reasoning as above.
        self._raw_image_cache: dict[int, Optional[np.ndarray]] = {}
        # metric -> LUT window fitted once from the project's own first
        # scored frame (see segmentation_qc.fit_display_window). Cached so
        # every frame shares one window and stays mutually comparable.
        self._qc_display_cache: dict[str, dict] = {}
        # Minimum junction interface size for `junction_contrast`, in band
        # pixels. `None` = derive it from the project (see
        # `get_junction_min_px`); an int is a user override.
        self._junction_min_px: Optional[int] = None
        self._junction_min_px_auto: Optional[int] = None
        # Weight of the mask-overlap term in the tracking cost, in units of the
        # cell's own diameter. 0 reproduces the centroid-only cost the
        # manuscript's published baseline was measured with.
        self.overlap_weight: float = cell_tracking.DEFAULT_OVERLAP_WEIGHT
        # Weight of the centroid term. 0 with a non-zero overlap weight is the
        # mask-overlap-only cost. See `set_tracking_cost` for what measurement
        # says about picking between the three modes.
        self.centroid_weight: float = 1.0
        # Predict each cell forward from its own previous displacement instead
        # of assuming it stays put. **Default OFF, because measurement says it
        # makes tracking worse on `embryos_GBE_GAP43`**: the swap rate goes
        # 4.89% -> 9.52% and tracks 414 -> 482. See `set_tracking_cost` for why
        # the prediction backfires here. Kept because it is the right model for
        # a movie with genuine coherent drift (stage drift, a translating
        # tissue) -- which this one does not have.
        self.motion_compensation: bool = False
        # Colours for the link-agreement overlay. Four states, because
        # "unjudgeable" must be VISIBLY distinct from "agrees" -- see
        # `get_link_agreement`.
        #
        # Unjudgeable is a low-chroma violet, not grey. Grey was the obvious
        # choice for "no verdict" and it was measured to fail: against this
        # project's greyscale membrane images it renders at RGB (142,142,147)
        # on a local median of (89,95,95), the same hue as the background, and
        # all 15 unjudgeable cells in frame 4 of `embryos_GBE_GAP43` were
        # impossible to pick out. Any saturated hue is legible over greyscale;
        # violet also stays off the red/green good-bad axis, so it still reads
        # as a category rather than a grade.
        self.agreement_colors: dict = {
            "agrees": "#4a90d9", "disagrees": "#ff3b30",
            "unjudgeable": "#a06bd4", "broken": "#ffcc00",
        }
        # Frames a track may skip and still re-attach (second-stage LAP, see
        # `cell_tracking.close_track_gaps`). 0 = off, which keeps tracking
        # byte-identical to the published baseline. Turning it on changes track
        # counts and therefore cell-loss %.
        self.max_frame_gap: int = 0
        # Environment-score dead band. At or above this the cell is drawn
        # untinted rather than "slightly dark". Tissue Analyzer's own
        # sensitivity (`take_decision`, 0.75); below it a second, lower tier
        # (`optimize_score`, 0.5) is what TA considers for automatic repair.
        # Without a dead band, 64.8% of cell-frames on `embryos_GBE_GAP43`
        # render as somewhat dark against TA's 22.5%, because 42.2% sit in
        # [0.75, 1.0) -- which TA classes as clean.
        # Whether the link-agreement reference is computed on registered masks.
        # On by default because the alternative is not neutral either -- an
        # unregistered reference favours a tracker that does not register --
        # and on this project's germband data the unregistered reference sits
        # at a median self-overlap of 0.377 against its own 0.3 threshold.
        # `registration_decision` still gates whether it applies at all.
        self.register_reference: bool = True
        # Registration applied to TRACKING (not just to the agreement
        # reference). Default "none" so the published baseline stays
        # byte-reproducible; "mask" is Tissue Analyzer's convention.
        self.drift_mode: str = "none"
        self._registration_cache: Optional[dict] = None
        self.environment_dead_band: float = 0.75
        self.environment_repair_threshold: float = 0.5
        # Schmitt-trigger hysteresis on the adjacency underlying T1 detection.
        # None = off, reproducing 1,505 gains / 1,545 losses on
        # `embryos_GBE_GAP43`. At 25/15 that becomes 1,229 / 978 with the
        # one-frame reversal rate falling 21.0% -> 14.8%; a HARD threshold of
        # similar magnitude raises it to 31.7% instead, which is the control
        # showing the hysteresis rather than the threshold is doing the work.
        self.t1_enter_px: Optional[float] = None
        self.t1_exit_px: Optional[float] = None
        self._hysteresis_cache: dict = {}
        # Which T1 axes to draw: "all" | "persistent" | "persistent_unjudgeable".
        # Three-way rather than a checkbox because `annotate_persistence`
        # produces three states, and a toggle would silently fold "unjudgeable"
        # into one of the other two. Defaults to ALL: the standing position is
        # "annotate, never drop", and a filter whose removals cannot be seen
        # cannot be judged too aggressive.
        self.t1_axis_filter: str = "all"
        # Frames a neighbour exchange must hold before it counts as settled.
        # 21% of raw events on `embryos_GBE_GAP43` reverse within one frame.
        self.t1_persist_frames: int = 1
        # Whether the tissue-rim ("free edge") junctions are drawn at all.
        # Off by default: a rim edge only means the mask stops there, which may
        # be a real tissue margin or just where segmentation was cropped, and
        # ~18% of them score <= 0 on `embryos_GBE_GAP43` -- rendering as a
        # solid dark ring that reads as a row of errors it isn't.
        self.show_free_edges: bool = False
        # Cell-dynamics curation: candidates are detected, but only events a
        # human confirms reach the lineage sheet (see `lineage.py`).
        self._confirmed_event_keys: set = set()
        self._rejected_event_keys: set = set()
        self._geometry_lookup: Optional[dict] = None
        self._margin_cache: dict = {}

        self.matched_color = "#ffd633"
        self.unmatched_color = "#3388ff"
        self.highlight_color = "#ff3333"
        self.border_color = "#ff00ff"
        self.lifecycle_colors = {"lost": "#ff3b30", "new": "#34c759", "both": "#ffffff"}
        self.dynamics_colors = {"division": "#34c759", "delamination": "#ff3b30", "t1": "#ffd633"}

    def set_window(self, window) -> None:
        self.window = window

    # ---- internal helpers -------------------------------------------------

    def _notify(self, js_function: str, payload: dict) -> None:
        if self.window is None:
            logger.info("[%s] %s", js_function, payload)
            return
        try:
            self.window.evaluate_js(f"window.{js_function} && window.{js_function}({json.dumps(payload)})")
        except Exception:
            logger.exception("Failed to notify frontend via %s", js_function)

    def _progress_cb(self, stage: str):
        def cb(message: str, percent: int) -> None:
            self._notify("onProgress", {"stage": stage, "message": message, "percent": percent})
        return cb

    def _npy_path_for_frame(self, frame_number: int) -> Optional[str]:
        if not self.npy_folder:
            return None
        files = data_input.list_segmentation_files(self.npy_folder)
        for filename, number in data_input.resolve_frame_numbers(files).items():
            if number == frame_number:
                return os.path.join(self.npy_folder, filename)
        return None

    def _load_frame(self, frame_number: int) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Returns `(fill_labels, boundary_labels, base_image)` for
        `frame_number`, or `None` if the frame's `.npy` can't be found.

        `fill_labels` must be a *filled* per-cell mask for rendering.py's
        hit-testing/tinting to work (each cell's whole interior sharing one
        ID, not just its border). Native Cellpose `.npy` output's own
        `'outlines'` key is deceptively named: it's the ~2-5%-of-pixels
        boundary line, not a filled mask -- `'masks'` is the filled one
        (same cell IDs, verified against test_data).
        `convert_tiff_masks_to_npy` writes both keys with the same meaning, so
        either input path yields a filled `fill_labels` and a thin
        `boundary_labels`. (The `'masks'`-absent fallback below is kept only
        for `.npy` files written by older versions of that converter, which
        stored a single filled array under `'outlines'`; on those the two
        render modes look identical.)

        `boundary_labels` is always the raw `'outlines'` key as-is: the thin
        boundary-line array. Two distinct uses need it kept separate from
        `fill_labels` rather than unified:
        - `manual_correction_classification.compute_leading_edge`'s contour
          distance algorithm was ported from v1, which always ran against
          this boundary array -- swapping in the filled mask there would
          silently change the leading-edge numbers reported in the
          manuscript, so `compute_leading_edge_all_frames` below always
          passes `boundary_labels`, never `fill_labels`.
        - `rendering.py`'s overlay functions can render either as the fully
          filled region (`render_mode="mask"`, the default -- easy at-a-glance
          cluster membership) or as just the boundary line tinted, leaving
          the grayscale interior visible (`render_mode="outline"`, closer to
          v1's own look, so touching same-cluster cells stay visually
          distinguishable instead of fusing into one blob).
        """
        npy_path = self._npy_path_for_frame(frame_number)
        if npy_path is None:
            return None
        dat = np.load(npy_path, allow_pickle=True).item()
        base_image = rendering.load_base_image(dat, self.image_folder, frame_number)
        fill_labels = dat["masks"] if "masks" in dat else dat["outlines"]
        boundary_labels = dat["outlines"]
        return fill_labels, boundary_labels, base_image

    # ---- Setup --------------------------------------------------------

    def select_folder(self, title: str = "Select folder") -> Optional[str]:
        if self.window is None:
            return None
        result = self.window.create_file_dialog(webview.FOLDER_DIALOG)
        if not result:
            return None
        return result[0] if isinstance(result, (list, tuple)) else result

    def set_project(self, mask_folder: str, image_folder: str, output_base_dir: str) -> dict:
        self.mask_folder = mask_folder
        self.image_folder = image_folder
        self._neighbor_graph_cache.clear()  # graphs belong to the old project's masks
        self._raw_image_cache.clear()
        self._qc_display_cache.clear()
        self._margin_cache.clear()
        self._geometry_lookup = None
        self._junction_min_px = None
        self._junction_min_px_auto = None
        self.output_dirs = common.setup_output_dirs(output_base_dir)
        fmt = data_input.detect_mask_format(mask_folder)
        return {"format": fmt, "output_dirs": self.output_dirs}

    def get_existing_results(self) -> dict:
        """Which pipeline stages already have on-disk output under the
        current `self.output_dirs`, so the frontend can offer to load them
        instead of recomputing. Each stage's file is written as a side
        effect of its own compute step (not only on an explicit "Save"), so
        this reflects the last *run*, not just the last saved correction."""
        cells_info_path = os.path.join(self.output_dirs["cell_classification_output"], "Cells_info.xlsx")
        clustering_path = os.path.join(self.output_dirs["cell_classification_output"], "cells_clustering_results.xlsx")
        tracking_path = os.path.join(
            self.output_dirs["cell_track_output"], "all_cell_tracking", "all_cell_merged_tracking_results.xlsx",
        )
        classification_dir = self.output_dirs["cell_classification_output"]
        return {
            "lineage": os.path.isfile(os.path.join(self.output_dirs["cell_track_output"], "cell_lineage.xlsx")),
            "extraction": os.path.isfile(cells_info_path),
            "classification": os.path.isfile(clustering_path),
            "tracking": os.path.isfile(tracking_path),
            "shape": os.path.isfile(os.path.join(classification_dir, "Cells_shape.xlsx")),
            "tissue": os.path.isfile(os.path.join(classification_dir, "Cells_tissue.xlsx")),
        }

    def load_existing_shape_features(self) -> dict:
        """Load previously written `Cells_shape.xlsx` / `Cells_tissue.xlsx`
        instead of recomputing them."""
        classification_dir = self.output_dirs["cell_classification_output"]
        for attribute, filename, sheet in (
            ("cells_shape", "Cells_shape.xlsx", "Shape"),
            ("cells_tissue", "Cells_tissue.xlsx", "Tissue"),
        ):
            path = os.path.join(classification_dir, filename)
            if os.path.isfile(path):
                setattr(self, attribute, pd.read_excel(path, sheet_name=sheet, index_col="Cell Index"))
        return self.get_shape_feature_summary()

    def load_existing_extraction(self) -> dict:
        """Load a previously written `Cells_info.xlsx` instead of re-running
        `extract_cell_info`. Still resolves `self.npy_folder` -- cheap for
        already-`.npy` input (a pass-through format check), and per-frame
        rendering/hit-testing needs it regardless of whether features were
        recomputed."""
        self.npy_folder = data_input.prepare_segmentation_folder(self.mask_folder, image_folder=self.image_folder)
        path = os.path.join(self.output_dirs["cell_classification_output"], "Cells_info.xlsx")
        self.cells_info = pd.read_excel(path, sheet_name="Cell")
        self._geometry_lookup = None
        return {"frame_count": self.get_frame_count(), "cell_count": len(self.cells_info)}

    def load_existing_classification(self) -> dict:
        """Load a previously written `cells_clustering_results.xlsx` instead
        of re-running PCA + K-means."""
        path = os.path.join(self.output_dirs["cell_classification_output"], "cells_clustering_results.xlsx")
        clustering_df = pd.read_excel(path)
        if "modified" not in clustering_df.columns:
            clustering_df["modified"] = False
        self.clustering_data = clustering_df
        self._border_cell_cache.clear()
        return {
            "cluster_sizes": clustering_df["Cell_Type"].value_counts().to_dict(),
            "k_used": int(clustering_df["Cluster_Label"].nunique()),
        }

    def load_existing_tracking(self) -> dict:
        """Load a previously written `all_cell_merged_tracking_results.xlsx`
        instead of re-running the Hungarian-tracking pipeline. Files written
        before the `_merge_tracking_pairs` fix (see `cell_tracking.py`) can
        carry duplicate/ambiguous claims on a single cell -- clean those up
        on load (in memory only; not written back to disk unless the user
        later hits Save Corrections) and report the count so the frontend
        can surface a notice."""
        path = os.path.join(
            self.output_dirs["cell_track_output"], "all_cell_tracking", "all_cell_merged_tracking_results.xlsx",
        )
        tracking_df = pd.read_excel(path)
        tracking_df, duplicates_cleaned = cell_tracking.dedupe_tracking_data(tracking_df)
        self.tracking_data = tracking_df
        self.original_tracking_data = tracking_df.copy()
        return {"track_count": int(len(tracking_df)), "duplicates_cleaned": duplicates_cleaned}

    # ---- Extract --------------------------------------------------------

    def start_extraction(
        self, mask_type: str = "auto", background_area_ratio: float = 10.0, exclude_edge_cells: bool = False,
    ) -> bool:
        thread = threading.Thread(
            target=self._run_extraction, args=(mask_type, background_area_ratio, exclude_edge_cells), daemon=True,
        )
        thread.start()
        return True

    def _run_extraction(self, mask_type: str, background_area_ratio: float, exclude_edge_cells: bool) -> None:
        try:
            self.npy_folder = data_input.prepare_segmentation_folder(
                self.mask_folder,
                image_folder=self.image_folder,
                mask_type=mask_type,
                background_area_ratio=background_area_ratio,
                progress_cb=self._progress_cb("extract"),
            )
            cells_info, frame_count = data_input.extract_cell_info(
                self.npy_folder,
                output_path=self.output_dirs["cell_classification_output"],
                exclude_edge_cells=exclude_edge_cells,
                progress_cb=self._progress_cb("extract"),
            )
            self.cells_info = cells_info.reset_index()
            self._geometry_lookup = None  # rebuilt lazily from the new table
            self._notify("onExtractDone", {"frame_count": frame_count, "cell_count": len(self.cells_info)})
        except Exception as e:
            logger.exception("Extraction failed")
            self._notify("onExtractError", {"message": str(e)})

    def start_shape_extraction(
        self,
        include_tissue: bool = True,
        efd_order: int = 10,
        n_contour_points: int = 256,
        smoothing_sigma_fraction: float = 0.02,
        with_coefficients: bool = False,
    ) -> bool:
        """Compute the convoluted-shape and tissue-texture descriptors.

        Separate from `start_extraction` and optional, because a user who only
        wants to track never needs them. They are cheap (0.3-0.8 ms/cell, ~10%
        of extraction) but they are also meaningless without a decision about
        what to do with them, so they are opt-in rather than automatic.
        """
        thread = threading.Thread(
            target=self._run_shape_extraction,
            args=(include_tissue, efd_order, n_contour_points, smoothing_sigma_fraction, with_coefficients),
            daemon=True,
        )
        thread.start()
        return True

    def _run_shape_extraction(
        self,
        include_tissue: bool,
        efd_order: int,
        n_contour_points: int,
        smoothing_sigma_fraction: float,
        with_coefficients: bool,
    ) -> None:
        try:
            if not self.npy_folder:
                self.npy_folder = data_input.prepare_segmentation_folder(
                    self.mask_folder, image_folder=self.image_folder
                )
            self.cells_shape = shape_features.extract_shape_features(
                self.npy_folder,
                output_path=self.output_dirs["cell_classification_output"],
                efd_order=efd_order,
                n_contour_points=n_contour_points,
                smoothing_sigma_fraction=smoothing_sigma_fraction,
                with_coefficients=with_coefficients,
                progress_cb=self._progress_cb("shape"),
            )
            if include_tissue:
                self.cells_tissue = tissue_features.extract_tissue_features(
                    self.npy_folder,
                    shape_info=self.cells_shape,
                    output_path=self.output_dirs["cell_classification_output"],
                    progress_cb=self._progress_cb("shape"),
                )
            self._notify("onShapeExtractDone", self.get_shape_feature_summary())
        except Exception as e:
            logger.exception("Shape descriptor extraction failed")
            self._notify("onShapeExtractError", {"message": str(e)})

    def get_shape_feature_summary(self) -> dict:
        """Median and IQR per descriptor, so the frontend can show what was
        computed without shipping the whole table across the bridge.

        The listed columns come from `pca_kmeans.FEATURE_FAMILIES` -- the same
        source the Classify step selects from -- so this panel can only ever
        show things the user can actually pick. An earlier version listed the
        `SCALE_INVARIANT_*` columns instead, which conflated "is dimensionless"
        (a property of the maths) with "is offered as a feature" (a design
        decision) and so kept advertising `Elongation`, `Sin2Theta`,
        `Cos2Theta` and `Frame Neighbor Entropy` after they had been retired
        from every family.

        Retired and internal columns are still computed and still written to
        the xlsx -- `Sin2Theta`/`Cos2Theta` feed `Local Nematic Order`, and
        `Frame Neighbor Entropy` is a legitimate per-frame statistic -- so they
        are reported as a `diagnostics` count rather than silently dropped.
        """
        offered = {c for spec in pca_kmeans.FEATURE_FAMILIES.values() for c in spec["columns"]}
        summary = {}
        for label, frame in (("shape", self.cells_shape), ("tissue", self.cells_tissue)):
            if frame is None or frame.empty:
                continue
            present = [c for c in frame.columns if c in offered]
            diagnostics = [
                c for c in frame.columns
                if c not in offered and not str(c).startswith("efd_")
            ]
            described = frame[present].describe()
            summary[label] = {
                "cell_count": int(len(frame)),
                "diagnostics": diagnostics,
                "columns": {
                    column: {
                        "median": _finite(described.loc["50%", column]),
                        "q1": _finite(described.loc["25%", column]),
                        "q3": _finite(described.loc["75%", column]),
                        "missing": int(frame[column].isna().sum()),
                    }
                    for column in present
                },
            }
        return summary

    def get_frame_count(self) -> int:
        if self.cells_info is not None:
            frames = self.cells_info["Cell Index"].str.extract(r"Cell(\d+)_")[0]
            return int(frames.astype(int).nunique())
        if self.npy_folder:
            return len(data_input.list_segmentation_files(self.npy_folder))
        return 0

    def get_frame_numbers(self) -> list[int]:
        """The actual sorted frame numbers with data -- NOT necessarily
        `1..get_frame_count()` contiguous. Datasets like
        `cp_dataset_puzzle-like_cells` are a folder of unrelated samples
        (`Root_YX_019`, `N_294_..._001`, `N_435_..._009`), each keeping
        whatever number is embedded in its own filename
        (`_extract_frame_number`'s last-digit-run rule) rather than a
        sequential frame counter -- e.g. `[1, 9, 19]`. The frontend must
        step through this list, not assume `frame + 1` is ever a valid next
        frame."""
        if self.cells_info is not None:
            frames = self.cells_info["Cell Index"].str.extract(r"Cell(\d+)_")[0].astype(int)
            return sorted(frames.unique().tolist())
        if self.npy_folder:
            files = data_input.list_segmentation_files(self.npy_folder)
            return sorted(set(data_input.resolve_frame_numbers(files).values()))
        return []

    def get_raw_frame_image(self, frame_number: int) -> Optional[str]:
        loaded = self._load_frame(frame_number)
        if loaded is None:
            return None
        _fill_labels, _boundary_labels, base_image = loaded
        from PIL import Image
        return rendering.encode_png_base64(Image.fromarray(base_image))

    # ---- Segmentation QC (Extract step) ----------------------------------

    def _raw_frame_array(self, frame_number: int) -> Optional[np.ndarray]:
        """The frame's **unmodified** intensity array, cached per frame.

        Deliberately not `_load_frame`'s `base_image`: `rendering.load_base_image`
        max-normalizes each frame to uint8, which makes intensities
        incomparable between frames and throws away 8 bits of a 16-bit
        acquisition (`test_data/pupae_E-cad` is genuinely uint16). Like the
        neighbor-graph cache this depends only on files on disk, so it never
        needs invalidating on a correction -- only when the project changes.
        """
        if frame_number in self._raw_image_cache:
            return self._raw_image_cache[frame_number]
        raw = None
        npy_path = self._npy_path_for_frame(frame_number)
        if npy_path is not None:
            dat = np.load(npy_path, allow_pickle=True).item()
            raw = dat.get("img")
        if raw is None and self.image_folder:
            raw = data_input._find_matching_image(self.image_folder, frame_number)
        self._raw_image_cache[frame_number] = raw
        return raw

    def _qc_scores(self, frame_number: int, metric: str) -> dict:
        """`{raw_label: score}` for one metric on one frame. Empty dict if the
        metric's inputs aren't available yet."""
        loaded = self._load_frame(frame_number)
        if loaded is None:
            return {}
        fill_labels = loaded[0]

        if metric == "boundary_contrast":
            raw = self._raw_frame_array(frame_number)
            return segmentation_qc.boundary_contrast(raw, fill_labels) if raw is not None else {}
        if metric == "junction_contrast":
            # Keyed by junction id, not cell label -- `_qc_render_labels` pairs
            # this with a matching label array so the bad *edge* is painted
            # rather than the two cells sharing it.
            raw = self._raw_frame_array(frame_number)
            if raw is None:
                return {}
            min_px = self.get_junction_min_px()["value"]
            _painted, mapping = segmentation_qc.junction_label_array(fill_labels, min_interface_px=min_px)
            scored = segmentation_qc.junction_contrast(raw, fill_labels, min_interface_px=min_px)
            return {
                jid: scored[pair] for jid, pair in mapping.items()
                if pair in scored and (self.show_free_edges or pair[0] != 0)
            }
        if metric == "neighbor_count":
            return segmentation_qc.neighbor_count_deviation(
                fill_labels, neighbor_graph=self._get_neighbor_graph(frame_number),
            )
        if self.cells_info is None:
            return {}
        if metric == "morphology":
            return segmentation_qc.morphology_outlier(self.cells_info, frame_number)
        if metric == "measurement_failures":
            return segmentation_qc.measurement_failures(self.cells_info, frame_number)
        return {}

    def _free_edge_ids(self, frame_number: int) -> set:
        """Junction ids at `frame_number` that are free edges (tissue rim).

        Recomputed per call rather than cached on the instance. It used to be
        stashed as `self._free_edge_ids` by `_qc_scores`, but `_qc_display`
        scores OTHER sample frames to fit its colour window and overwrote it --
        and junction ids are per-frame sequential, so the render then
        classified this frame's junctions with a different frame's id set. That
        misclassified 28 of 427 junctions on `AS_dorsal view_E-cad`, where the
        two colour windows differ 1.6x. Only bit on a cold cache, which is why
        it looked intermittent.
        """
        loaded = self._load_frame(frame_number)
        if loaded is None:
            return set()
        _painted, mapping = segmentation_qc.junction_label_array(
            loaded[0], min_interface_px=self.get_junction_min_px()["value"],
        )
        return {jid for jid, pair in mapping.items() if pair[0] == 0}

    def _qc_display_scores(self, metric: str, frame_number: int, scores: dict) -> dict:
        """Scores mapped into a common 0-1 display space.

        `junction_contrast` carries two incomparable classes (see
        `segmentation_qc.split_junction_scores`), so each is normalised through
        its OWN fitted window before rendering. Sharing one window would let
        the free edges -- which on an ROI-cropped dataset sit near 0 by
        construction -- drag the cell-cell colours down with them. Every other
        metric keeps its single window unchanged.
        """
        if metric != "junction_contrast":
            window = self._qc_display(metric)
            span = (window["vmax"] - window["vmin"]) or 1.0
            return {k: (v - window["vmin"]) / span for k, v in scores.items()}

        free_ids = self._free_edge_ids(frame_number)
        pair_window = self._qc_display(metric)
        free_window = self._qc_display("junction_contrast_free")
        out = {}
        for key, value in scores.items():
            window = free_window if key in free_ids else pair_window
            span = (window["vmax"] - window["vmin"]) or 1.0
            out[key] = (value - window["vmin"]) / span
        return out

    def _qc_render_labels(
        self, metric: str, fill_labels: np.ndarray, boundary_labels: np.ndarray, render_mode: str,
    ) -> np.ndarray:
        """Which label array a QC metric's scores should be painted onto.

        `junction_contrast` is scored per cell-cell interface, so it needs the
        junction map whose ids its scores are keyed by; the render-mode toggle
        doesn't apply to it (a junction is already a line). Every other metric
        is per cell and honours the toggle as usual.
        """
        if metric == "junction_contrast":
            painted, _mapping = segmentation_qc.junction_label_array(
                fill_labels, min_interface_px=self.get_junction_min_px()["value"],
            )
            return painted
        return boundary_labels if render_mode == "outline" else fill_labels

    # How many frames the LUT window is fitted from. Pooled rather than taken
    # from frame 1 alone because a project's frames are not always comparable:
    # `test_data/Demo_Seg_QC` holds unrelated samples whose contrast levels
    # differ ~2x, so fitting on whichever happened to be first would clip every
    # other frame to the top of the colormap. Three is enough to be
    # representative without paying for a whole 86-frame movie.
    _QC_FIT_FRAMES = 3

    def get_show_free_edges(self) -> bool:
        return bool(self.show_free_edges)

    def set_show_free_edges(self, enabled: bool = False) -> bool:
        """Show/hide the tissue-rim junctions. Clears the fitted colour
        windows, which are fitted over whichever classes are displayed."""
        self.show_free_edges = bool(enabled)
        self._qc_display_cache.pop("junction_contrast", None)
        self._qc_display_cache.pop("junction_contrast_free", None)
        return self.show_free_edges

    def _cell_geometry(self, cell_index: str) -> Optional[tuple]:
        """`(x, y, area, radius)` for one `Cell{frame}_{label}` string, from
        `cells_info`. The shape `lineage.py`'s detectors expect."""
        if self.cells_info is None or not cell_index:
            return None
        if self._geometry_lookup is None:
            info = self.cells_info.set_index("Cell Index")
            self._geometry_lookup = {
                idx: (
                    float(row["Center X Coordinate"]), float(row["Center Y Coordinate"]),
                    float(row["Cell Area"]), float(row["Circumscribed Circle Radius"]),
                )
                for idx, row in info.iterrows()
            }
        return self._geometry_lookup.get(cell_index)

    def _margin_labels(self, frame_number: int) -> set:
        """Labels touching the image border in `frame_number`, cached.

        This is the margin test TissueMiner runs BEFORE its delamination
        fallback, so what lands in it decides what gets called a delamination.
        Evaluated on the FILLED mask, matching `data_input.is_edge_cell`'s
        corrected convention -- the thin outline array cannot reach row 0 and
        would report an empty margin.
        """
        if frame_number in self._margin_cache:
            return self._margin_cache[frame_number]
        loaded = self._load_frame(frame_number)
        margin: set = set()
        if loaded is not None:
            fill = loaded[0]
            for edge in (fill[0, :], fill[-1, :], fill[:, 0], fill[:, -1]):
                margin.update(int(v) for v in np.unique(edge) if v)
        self._margin_cache[frame_number] = margin
        return margin

    def _is_margin_cell(self, cell_index: str) -> bool:
        match = re.match(r"Cell(\d+)_(\d+)$", str(cell_index))
        if not match:
            return False
        return int(match.group(2)) in self._margin_labels(int(match.group(1)))

    def get_dynamics_candidates(self, frame_number: Optional[int] = None) -> dict:
        """Division / delamination / T1 candidates, with their evidence.

        Candidates, not verdicts -- see `lineage.py`'s module docstring for why
        neither detector can be trusted unattended. The frontend presents these
        for confirmation; only confirmed events reach the lineage sheet.
        """
        if self.tracking_data is None or self.cells_info is None:
            return {"error": "Extract and track the movie first."}

        spans = lineage.track_spans(self.tracking_data)
        if not spans:
            return {"divisions": [], "delaminations": [], "t1": {}, "frame": frame_number}
        last = max(s["last"] for s in spans.values())

        divisions = lineage.division_candidates(
            spans, self._cell_geometry, self._get_neighbor_graph, last_frame=last,
        )
        explained = {d["mother"] for d in divisions}
        # The population median is what "shrunk away" is judged against -- a
        # cell that vanishes at full size is a broken track, not a delamination.
        typical_area = float(self.cells_info["Cell Area"].median())
        delaminations = lineage.delamination_candidates(
            spans, self._cell_geometry, self._is_margin_cell, last_frame=last, explained=explained,
            typical_area=typical_area,
        )
        result = {
            "divisions": divisions, "delaminations": delaminations,
            "confirmed": sorted(self._confirmed_event_keys),
            "rejected": sorted(self._rejected_event_keys),
        }
        if frame_number is not None:
            result["divisions"] = [d for d in divisions if d["frame"] == frame_number]
            result["delaminations"] = [d for d in delaminations if d["frame"] == frame_number]
            result["t1"] = self.get_t1_events(frame_number)
            result["frame"] = frame_number
        return result

    def set_event_decision(self, event_key: str, decision: str = "confirm") -> dict:
        """Record a human decision on one candidate. `event_key` is
        `"division:<mother>:<frame>"` or `"delamination:<track>:<frame>"`."""
        self._confirmed_event_keys.discard(event_key)
        self._rejected_event_keys.discard(event_key)
        if decision == "confirm":
            self._confirmed_event_keys.add(event_key)
        elif decision == "reject":
            self._rejected_event_keys.add(event_key)
        return {
            "confirmed": sorted(self._confirmed_event_keys),
            "rejected": sorted(self._rejected_event_keys),
        }

    def _confirmed_events(self) -> list:
        """The candidate objects the user has confirmed."""
        found = self.get_dynamics_candidates()
        if "error" in found:
            return []
        events = []
        for d in found["divisions"]:
            if f"division:{d['mother']}:{d['frame']}" in self._confirmed_event_keys:
                events.append(d)
        for d in found["delaminations"]:
            if f"delamination:{d['track']}:{d['frame']}" in self._confirmed_event_keys:
                events.append(d)
        return events

    def save_lineage(self) -> dict:
        """Build the lineage sheet from CONFIRMED events and write it to
        `cell_track_output/cell_lineage.xlsx`."""
        if self.tracking_data is None:
            return {"error": "No tracking data loaded."}
        spans = lineage.track_spans(self.tracking_data)
        margin_tracks = {
            track for track, span in spans.items()
            if self._is_margin_cell(span["cells"][span["first"]])
            or self._is_margin_cell(span["cells"][span["last"]])
        }
        table = lineage.build_lineage_table(
            self.tracking_data, confirmed=self._confirmed_events(), margin_tracks=margin_tracks,
        )
        path = os.path.join(self.output_dirs["cell_track_output"], "cell_lineage.xlsx")
        table.to_excel(path, sheet_name="Lineage", index=False)
        return {"path": path, "tracks": int(len(table)),
                "divisions": int((table.disappears_by == "Division").sum()),
                "delaminations": int((table.disappears_by == "Delamination").sum())}

    def _t1_centroids(self, frame_number: int, label_to_track: dict) -> dict:
        """`{track_id: (x, y)}` for one frame, for the T1 orientation nematic."""
        if self.cells_info is None:
            return {}
        info = self.cells_info.set_index("Cell Index")
        out = {}
        for label, track in label_to_track.items():
            if track is None:
                continue
            index = f"Cell{frame_number}_{label}"
            if index in info.index:
                out[track] = (
                    float(info.at[index, "Center X Coordinate"]),
                    float(info.at[index, "Center Y Coordinate"]),
                )
        return out

    def _contacts_at(self, frame_number: int) -> Optional[set]:
        """`topology.contacts_in_track_space` for one frame, or `None` if that
        frame has no neighbour graph. Reuses the cached graph."""
        graph = self._get_neighbor_graph(frame_number)
        if not graph:
            return None
        if self.t1_enter_px is None:
            return topology.contacts_in_track_space(graph, self._label_to_track(frame_number))
        # Hysteresis needs the PREVIOUS frame's verdict, so the chain is walked
        # from the first frame rather than evaluated pointwise. Without that the
        # band has no state to hold and degenerates to a hard threshold at
        # `enter_px` -- the exact thing it exists to avoid.
        return self._hysteresis_contacts(frame_number)

    def _hysteresis_contacts(self, frame_number: int) -> Optional[set]:
        """Contacts at `frame_number` under the Schmitt trigger, resolved by
        walking forward from the first frame. Cached per frame."""
        if frame_number in self._hysteresis_cache:
            return self._hysteresis_cache[frame_number]
        frames = [f for f in self.get_frame_numbers() if f <= frame_number]
        previous = None
        for frame in frames:
            if frame in self._hysteresis_cache:
                previous = self._hysteresis_cache[frame]
                continue
            loaded = self._load_frame(frame)
            if loaded is None:
                self._hysteresis_cache[frame] = None
                continue
            weighted = adjacency.build_neighbor_graph(loaded[0], return_weights=True)
            previous = topology.contacts_in_track_space(
                weighted, self._label_to_track(frame), previous_contacts=previous,
                enter_px=self.t1_enter_px, exit_px=self.t1_exit_px,
            )
            self._hysteresis_cache[frame] = previous
        return self._hysteresis_cache.get(frame_number)

    def get_t1_events(self, frame_number: int, direction: str = "next") -> dict:
        """Neighbour exchanges (T1 transitions) between `frame_number` and the
        adjacent frame. See `topology.detect_t1_transitions`.

        Returns per-cell gained/lost counts keyed by RAW label (so the overlay
        can paint them) plus the nematic axes for drawing. Every event is
        additionally scored for **persistence** over `self.t1_persist_frames`
        further frames (`topology.annotate_persistence`), giving a second set
        of counts and a per-axis flag. Nothing is filtered out here -- the
        caller renders settled events solid and transient ones faded, and both
        totals are reported so the filter's effect stays visible.
        """
        if self.tracking_data is None:
            return {"error": "Track the movie before looking for T1 transitions."}
        other = frame_number + 1 if direction == "next" else frame_number - 1
        graph_a, graph_b = self._get_neighbor_graph(frame_number), self._get_neighbor_graph(other)
        if not graph_a or not graph_b:
            return {"counts": {}, "counts_persistent": {}, "counts_unjudgeable": {}, "axes": [],
                    "gained": 0, "lost": 0, "gained_persistent": 0, "lost_persistent": 0,
                    "gained_unjudgeable": 0, "lost_unjudgeable": 0,
                    "persist_frames": int(self.t1_persist_frames)}

        l2t_a, l2t_b = self._label_to_track(frame_number), self._label_to_track(other)
        # Always evaluate forward in time so a "gain" means a bond formed.
        if direction == "next":
            events = topology.detect_t1_transitions(
                graph_a, graph_b, l2t_a, l2t_b, self._t1_centroids(frame_number, l2t_a),
            )
            earlier = frame_number
        else:
            events = topology.detect_t1_transitions(
                graph_b, graph_a, l2t_b, l2t_a, self._t1_centroids(other, l2t_b),
            )
            earlier = other

        # The events describe `earlier` -> `earlier + 1`, so persistence is
        # judged on what happens from `earlier + 2` onward. A frame that does
        # not exist simply shortens the horizon rather than failing the test:
        # the end of the movie is not evidence that an exchange reversed.
        # A missing frame ends the lookahead and the SHORT list is the signal:
        # `annotate_persistence` reports those events as unjudgeable rather
        # than silently passing them. Do not pad this list.
        contacts_after = []
        for offset in range(2, 2 + int(self.t1_persist_frames)):
            contacts = self._contacts_at(earlier + offset)
            if contacts is None:
                break
            contacts_after.append(contacts)
        topology.annotate_persistence(events, contacts_after, self.t1_persist_frames)

        track_to_label = {t: l for l, t in l2t_a.items() if t is not None}

        def _counts(subset: dict) -> dict:
            return {
                int(track_to_label[track]): value
                for track, value in topology.t1_counts_per_cell(subset).items()
                if track in track_to_label
            }

        def _subset(state: str) -> dict:
            return {key: [e for e in events[key] if e["state"] == state] for key in ("gained", "lost")}

        settled, unjudgeable = _subset("persistent"), _subset("unjudgeable")
        axes = [
            {"center": e["center"], "phi": e["phi"], "type": e["type"],
             "persistent": e["persistent"], "state": e["state"]}
            for e in events["gained"] + events["lost"] if "phi" in e
        ]
        return {
            "counts": _counts(events), "counts_persistent": _counts(settled),
            "counts_unjudgeable": _counts(unjudgeable), "axes": axes,
            "gained": len(events["gained"]), "lost": len(events["lost"]),
            "gained_persistent": len(settled["gained"]), "lost_persistent": len(settled["lost"]),
            "gained_unjudgeable": len(unjudgeable["gained"]), "lost_unjudgeable": len(unjudgeable["lost"]),
            "persist_frames": int(self.t1_persist_frames),
            # Provenance travels with every count. `25 / 15` is a statement
            # about one objective and one binning: on a dataset at half the
            # pixel size the same physical junction has roughly twice the
            # contact strength, so the same numbers are a looser filter. The
            # whole repo has no `pixel_size` / `um_per_px` anywhere, so without
            # this stamp no T1 count in this project is reproducible outside
            # the session that produced it.
            "parameters": self.t1_parameters(),
        }

    def t1_parameters(self) -> dict:
        """The settings any T1 count depends on, for stamping onto exports and
        figure captions. Reported even when a filter is off, because "off" is
        itself the setting that was used."""
        return {
            "enter_px": self.t1_enter_px,
            "exit_px": self.t1_exit_px,
            "dilation_px": adjacency.DEFAULT_DILATION_PX,
            "persist_frames": int(self.t1_persist_frames),
            "axis_filter": self.t1_axis_filter,
        }

    # The three named modes the UI offers. `centroid_weight` and
    # `overlap_weight` are what the cost actually reads; the mode is only a
    # preset over the pair, so a custom weight stays expressible.
    _COST_MODES = {
        "hybrid": (1.0, None),      # None = keep the user's overlap weight
        "centroid": (1.0, 0.0),
        "overlap": (0.0, None),
    }

    def get_tracking_cost(self) -> dict:
        """The tracking cost function's settings, plus their defaults.

        `mode` is derived from the weights rather than stored, so it can never
        disagree with them.
        """
        if self.centroid_weight == 0:
            mode = "overlap"
        elif self.overlap_weight == 0:
            mode = "centroid"
        else:
            mode = "hybrid"
        return {
            "mode": mode,
            "overlap_weight": float(self.overlap_weight),
            "centroid_weight": float(self.centroid_weight),
            "motion_compensation": bool(self.motion_compensation),
            "max_frame_gap": int(self.max_frame_gap),
            "drift_mode": str(self.drift_mode),
            "default_overlap_weight": float(cell_tracking.DEFAULT_OVERLAP_WEIGHT),
        }

    def set_tracking_cost(
        self, mode: str = None, overlap_weight: float = None, motion_compensation: bool = None,
        max_frame_gap: int = None, drift_mode: str = None,
    ) -> dict:
        """Set the cost function. Any argument left `None` is unchanged.

        **Measured on `embryos_GBE_GAP43`, against agreement with each cell's
        best-IoU successor: nothing here reduces the 4.89% swap rate.**

            centroid + overlap (shipped default)   4.90%
            hybrid, overlap weight 10              4.92%
            mask overlap (IoU) only                5.03%
            centroid only                          5.54%
            IoU-only + motion compensation         6.01%
            hybrid + motion compensation           9.57%

        The shipped default already wins, and the metric is biased in
        IoU-only's favour (it scores agreement with the best-IoU successor)
        yet IoU-only still loses -- so that ordering is not an artefact. The
        mask does carry real information: centroid-only is the worst of the
        three non-predictive costs, so Round 6's overlap term earns its place.
        It is simply already being used.

        Sweeping the overlap weight 0 -> 50 (effectively mask-only) barely
        moves held-out area continuity, and over a 2-frame gap mask-only
        scores 30.9% against centroid-only's 32.1%.

        **Why motion compensation backfires**, which is worth recording
        because the residual analysis predicted the opposite: per-frame
        displacement here is mostly jitter, not coherent motion. The tissue
        drifts 2.63 px/frame while individual cells move 7.9 px, and the
        median displacement does NOT grow with frame gap (7.9 / 8.2 / 8.0 px
        at gaps 1, 2, 3). A constant median across gaps is the signature of
        non-persistent motion -- exactly the regime where extrapolating the
        previous step adds noise (variance 6*sigma^2 against 2*sigma^2)
        instead of removing it. The earlier "residual 7.9 -> 2.3 px" figure
        was measured only over cells that were already tracked correctly,
        which is the subset where the prediction was not needed.

        So all four settings are offered because a dataset with different
        statistics may behave differently -- not because any of them is known
        to fix swaps here. The swaps come from ~8 px of centroid jitter
        against a 26 px neighbour spacing, which no reweighting of these terms
        resolves.
        """
        if overlap_weight is not None:
            self.overlap_weight = max(0.0, float(overlap_weight))
        if mode is not None:
            if mode not in self._COST_MODES:
                raise ValueError(f"Unknown cost mode {mode!r}; expected one of {sorted(self._COST_MODES)}.")
            centroid, overlap = self._COST_MODES[mode]
            self.centroid_weight = centroid
            if overlap is not None:
                self.overlap_weight = overlap
            elif self.overlap_weight == 0:
                # "hybrid"/"overlap" with the term switched off is the mode's
                # own contradiction; restore the default rather than hand
                # `build_cost_matrix` a cost with nothing in it.
                self.overlap_weight = cell_tracking.DEFAULT_OVERLAP_WEIGHT
        if motion_compensation is not None:
            self.motion_compensation = bool(motion_compensation)
        if max_frame_gap is not None:
            self.max_frame_gap = max(0, int(max_frame_gap))
        if drift_mode is not None:
            if drift_mode not in ("none", "centroid", "mask"):
                raise ValueError(f"drift_mode must be none/centroid/mask, got {drift_mode!r}")
            self.drift_mode = drift_mode
        return self.get_tracking_cost()

    def get_t1_persistence(self) -> dict:
        """How many frames a neighbour exchange must hold to count as settled,
        plus the hysteresis band and the axis filter."""
        return {
            "value": int(self.t1_persist_frames),
            "enter_px": self.t1_enter_px, "exit_px": self.t1_exit_px,
            "axis_filter": self.t1_axis_filter,
        }

    def set_t1_hysteresis(self, enter_px: float = None, exit_px: float = None) -> dict:
        """Set the Schmitt-trigger band on the adjacency, or clear it.

        Both `None` turns it off. `enter_px` must exceed `exit_px` -- equal
        values would BE a hard threshold, which Round 9 measured as making
        flicker worse rather than better.
        """
        if enter_px is None or exit_px is None:
            self.t1_enter_px = self.t1_exit_px = None
        else:
            enter_px, exit_px = float(enter_px), float(exit_px)
            if enter_px <= exit_px:
                raise ValueError(
                    f"enter_px ({enter_px}) must exceed exit_px ({exit_px}); equal values are a hard "
                    "threshold, which was measured to increase flicker."
                )
            self.t1_enter_px, self.t1_exit_px = enter_px, exit_px
        self._neighbor_graph_cache.clear()
        self._hysteresis_cache.clear()
        return self.get_t1_persistence()

    def set_t1_axis_filter(self, mode: str = "all") -> dict:
        """Which T1 orientation axes to draw: all / persistent /
        persistent_unjudgeable."""
        if mode not in ("all", "persistent", "persistent_unjudgeable"):
            raise ValueError(f"Unknown axis filter {mode!r}")
        self.t1_axis_filter = mode
        return self.get_t1_persistence()

    def set_t1_persistence(self, value: int = None) -> dict:
        """Set the T1 persistence horizon. 0 is TissueMiner's own unfiltered
        behaviour -- every raw event counts."""
        self.t1_persist_frames = max(0, int(value)) if value is not None else 0
        return self.get_t1_persistence()

    def get_junction_min_px(self) -> dict:
        """`{"value": int, "auto": int, "is_override": bool}` -- the minimum
        junction interface size used by `junction_contrast`.

        Derived once per project from the same sampled frames `_qc_display`
        uses, then cached: junction length scales with image resolution (the
        25th percentile runs 14-40 band px across this project's datasets), so
        a fixed constant is either too strict on a small frame or too
        permissive on a large one. Pooling a few frames rather than reading
        one keeps the threshold from drifting frame to frame; the measured
        per-frame spread is only 17-18 px on `embryos_GBE_GAP43` and 20-22 px
        on `Demo_Seg_QC`, so pooling costs nothing in accuracy.
        """
        if self._junction_min_px_auto is None:
            suggestions = []
            for frame in self._qc_fit_frames():
                loaded = self._load_frame(frame)
                if loaded is not None:
                    suggestions.append(segmentation_qc.suggest_min_interface_px(loaded[0]))
            self._junction_min_px_auto = (
                int(np.median(suggestions)) if suggestions else segmentation_qc.MIN_INTERFACE_PX
            )
        return {
            "value": int(self._junction_min_px if self._junction_min_px is not None else self._junction_min_px_auto),
            "auto": int(self._junction_min_px_auto),
            "is_override": self._junction_min_px is not None,
        }

    def set_junction_min_px(self, value: Optional[int] = None) -> dict:
        """Override the minimum junction size, or pass `None`/0 to go back to
        the derived value. Clears the LUT window, which was fitted from scores
        computed under the previous threshold."""
        self._junction_min_px = int(value) if value else None
        self._qc_display_cache.pop("junction_contrast", None)
        return self.get_junction_min_px()

    def _qc_fit_frames(self) -> list:
        """The frames used to derive project-wide QC settings."""
        frames = self.get_frame_numbers()
        if not frames:
            return []
        step = max(1, len(frames) // self._QC_FIT_FRAMES)
        return frames[::step][: self._QC_FIT_FRAMES]

    def _qc_display(self, metric: str) -> dict:
        """`metric`'s LUT window for this project, fitted once and cached.

        `"junction_contrast_free"` is a pseudo-metric: the free-edge half of
        `junction_contrast`, fitted separately because the two classes are not
        comparable (see `segmentation_qc.split_junction_scores`).
        """
        if metric not in self._qc_display_cache:
            free_only = metric == "junction_contrast_free"
            source = "junction_contrast" if free_only else metric
            sampled = self._qc_fit_frames()
            pooled = {}
            for offset, frame in enumerate(sampled):
                scores = self._qc_scores(frame, source)
                if source == "junction_contrast":
                    free_ids = self._free_edge_ids(frame)
                    scores = {
                        k: v for k, v in scores.items()
                        if (k in free_ids) == free_only
                    }
                # Keys collide across frames (a label id is per-frame), so
                # namespace them -- only the value distribution matters here.
                for key, value in scores.items():
                    pooled[(offset, key)] = value
            self._qc_display_cache[metric] = segmentation_qc.fit_display_window(source, pooled)
        return self._qc_display_cache[metric]

    def copy_image_to_clipboard(self, data_uri: str) -> dict:
        """Put a `data:image/png;base64,...` string on the system clipboard.

        A fallback only. The page is served over `http://127.0.0.1:<port>/`
        (pywebview auto-starts its own server for a filesystem path), which is
        a secure context, so the frontend's `navigator.clipboard` should
        normally handle this without a round-trip. This exists because that
        path has never been exercised in the real native window.

        Uses `AppKit.NSPasteboard`, which needs no new dependency -- pyobjc is
        already installed transitively by pywebview's Cocoa backend. Other
        platforms fall through to a clear error rather than a silent no-op.
        """
        try:
            import base64

            if "," not in data_uri:
                return {"success": False, "message": "Not a data URI."}
            png_bytes = base64.b64decode(data_uri.split(",", 1)[1])

            if sys.platform != "darwin":
                return {"success": False, "message": f"No clipboard fallback for {sys.platform}."}

            from AppKit import NSPasteboard, NSPasteboardTypePNG
            from Foundation import NSData

            pasteboard = NSPasteboard.generalPasteboard()
            pasteboard.clearContents()
            ok = pasteboard.setData_forType_(
                NSData.dataWithBytes_length_(png_bytes, len(png_bytes)), NSPasteboardTypePNG,
            )
            return {"success": bool(ok), "message": "" if ok else "NSPasteboard rejected the data."}
        except Exception as e:
            logger.exception("Clipboard copy failed")
            return {"success": False, "message": str(e)}

    def get_qc_metric_names(self) -> list:
        return segmentation_qc.QC_METRIC_NAMES

    def get_qc_metric_display(self) -> dict:
        return segmentation_qc.QC_METRIC_DISPLAY

    def get_qc_overlay(
        self, frame_number: int, metric: str = "boundary_contrast",
        lut: str = "viridis", render_mode: str = "mask",
    ) -> Optional[str]:
        loaded = self._load_frame(frame_number)
        if loaded is None:
            return None
        fill_labels, boundary_labels, base_image = loaded
        label_array = self._qc_render_labels(metric, fill_labels, boundary_labels, render_mode)
        scores = self._qc_display_scores(metric, frame_number, self._qc_scores(frame_number, metric))
        overlay = rendering.render_score_overlay(
            label_array, base_image, scores, lut, vmin=0.0, vmax=1.0,
        )
        return rendering.encode_png_base64(overlay)

    def get_extract_preview(
        self, frame_number: int, metric: Optional[str] = None,
        lut: str = "viridis", render_mode: str = "mask",
        show_outline: bool = False, outline_color: str = "#00ff00",
        show_labels: bool = False, label_font_size: int = 10,
    ) -> Optional[str]:
        """The Extract step's preview image, composited in one call: raw
        frame, optional QC tint (`metric=None` for none), optional
        segmentation outline, optional per-cell ID labels.

        One method rather than the frontend layering several, because each
        layer needs the same `_load_frame` result and the overlays must be
        drawn in a fixed order (tint, then outline, then labels on top).
        """
        loaded = self._load_frame(frame_number)
        if loaded is None:
            return None
        fill_labels, boundary_labels, base_image = loaded

        if metric:
            label_array = self._qc_render_labels(metric, fill_labels, boundary_labels, render_mode)
            scores = self._qc_display_scores(metric, frame_number, self._qc_scores(frame_number, metric))
            image = rendering.render_score_overlay(
                label_array, base_image, scores, lut, vmin=0.0, vmax=1.0,
            )
            array = np.array(image)
        else:
            array = base_image.copy()

        if show_outline:
            array = rendering.draw_cell_outlines(array, fill_labels, outline_color)

        from PIL import Image
        pil_img = Image.fromarray(array)
        if show_labels:
            pil_img = rendering.draw_cell_labels(pil_img, fill_labels, label_font_size)
        return rendering.encode_png_base64(pil_img)

    def get_qc_summary(self, metric: str = "boundary_contrast") -> list:
        """Per-frame `{frame, scored, flagged, min, median}` so the worst
        frames are findable without stepping through the whole movie. The
        flag threshold is the metric's own (see `QC_METRIC_DISPLAY`), since
        the metrics don't share a scale.

        For `junction_contrast`, free edges are counted but never FLAGGED: a
        dark free edge only means the mask stops there, which may be a real
        tissue margin or simply where the segmentation was cropped, and that
        is not decidable from the mask. Flagging them would bury the genuine
        cell-cell errors this metric exists to surface -- on
        `AS_dorsal view_E-cad` 16 of 44 free edges sit at or below 0 while not
        one of its 284 cell-cell junctions does.
        """
        flag_below = self._qc_display(metric)["flag_below"]
        rows = []
        for f in self.get_frame_numbers():
            scores = self._qc_scores(f, metric)
            free_edges = 0
            if metric == "junction_contrast":
                free_ids = self._free_edge_ids(f)
                free_edges = sum(1 for k in scores if k in free_ids)
                scores = {k: v for k, v in scores.items() if k not in free_ids}
            row = {"frame": f, **segmentation_qc.summarize(scores, flag_below)}
            if metric == "junction_contrast":
                row["free_edges"] = free_edges
            rows.append(row)
        return rows

    def get_qc_cell_at_pixel(self, frame_number: int, x: int, y: int) -> Optional[dict]:
        """Every metric's score for the clicked cell, so a flagged cell can be
        interrogated rather than just seen."""
        loaded = self._load_frame(frame_number)
        if loaded is None:
            return None
        fill_labels = loaded[0]
        cell_id = rendering.cell_id_at_pixel(fill_labels, int(x), int(y))
        if cell_id is None:
            return None

        scores = {}
        for metric in segmentation_qc.QC_METRIC_NAMES:
            if metric == "junction_contrast":
                continue  # per-interface, reported separately below
            value = self._qc_scores(frame_number, metric).get(int(cell_id))
            scores[metric] = None if value is None else round(float(value), 3)

        # For the clicked cell, list its own junctions worst-first: the point
        # of the metric is to say WHICH edge is bad, not just that one is.
        raw = self._raw_frame_array(frame_number)
        junctions = []
        if raw is not None:
            junction_scores = segmentation_qc.junction_contrast(
                raw, fill_labels, min_interface_px=self.get_junction_min_px()["value"],
            )
            for (a, b), value in junction_scores.items():
                if cell_id in (a, b):
                    junctions.append({"with_cell": int(b if a == cell_id else a), "contrast": round(float(value), 4)})
            junctions.sort(key=lambda j: j["contrast"])

        return {
            "cell_index": f"Cell{frame_number}_{cell_id}",
            "cell_id": int(cell_id),
            "scores": scores,
            "junctions": junctions,
        }

    # ---- Classify (skippable) --------------------------------------------

    def get_palette_presets(self) -> dict[str, list[str]]:
        return rendering.get_palette_presets()

    def _joined_features(self) -> pd.DataFrame:
        """`cells_info` joined with the shape and tissue tables on `Cell Index`."""
        frame = self.cells_info.set_index("Cell Index")
        for extra in (self.cells_shape, self.cells_tissue):
            if extra is not None and not extra.empty:
                frame = frame.join(extra, how="left")
        return frame

    def has_efd_coefficients(self) -> bool:
        return bool(
            self.cells_shape is not None
            and any(str(c).startswith("efd_") for c in self.cells_shape.columns)
        )

    def get_feature_options(self) -> dict:
        """The additive feature families the Classify step may offer.

        `basic` is the always-on base; every other family is added on top. `efd`
        is exclusive and unstandardized. A family is `ready` only when **every**
        column it names is present -- testing "any column present" would mark a
        family ready on a project with no shape table just because one of its
        columns happens to come from `Cells_info.xlsx`, and the run would then
        fail after the user had already selected it.
        """
        available = set(self._joined_features().columns) if self.cells_info is not None else set()
        families = {}
        for name, spec in pca_kmeans.FEATURE_FAMILIES.items():
            if name == "efd":
                ready = self.has_efd_coefficients()
                columns, missing = [], [] if ready else ["efd_* coefficients"]
            else:
                columns = list(spec["columns"])
                missing = [c for c in columns if c not in available]
                ready = not missing
            families[name] = {
                "label": spec["label"],
                "columns": columns,
                # Short labels for display only -- the values above stay the
                # real column names, which is what every lookup uses.
                "display": {c: pca_kmeans.display_name(c) for c in columns},
                "default_off": list(spec.get("default_off", ())),
                "base": bool(spec.get("base")),
                "exclusive": bool(spec.get("exclusive")),
                "note": spec.get("note", ""),
                "ready": ready,
                "missing": missing,
            }
        return {
            "families": families,
            "has_shape": self.cells_shape is not None,
            "has_tissue": self.cells_tissue is not None,
            "dataset_modes": ["time_series", "independent_samples"],
        }

    def _feature_frame(
        self, feature_families: Optional[list], selected_features: Optional[list]
    ) -> tuple[pd.DataFrame, str]:
        """The matrix to hand PCA, plus the standardize mode the selection
        mandates. Raises with a readable message rather than letting a missing
        column surface as a bare `KeyError` from pandas."""
        families = list(feature_families) if feature_families else ["basic"]
        unknown = [f for f in families if f not in pca_kmeans.FEATURE_FAMILIES]
        if unknown:
            raise ValueError(
                f"unknown feature family {unknown[0]!r}; expected one of {list(pca_kmeans.FEATURE_FAMILIES)}"
            )

        joined = self._joined_features()

        if "efd" in families:
            # Exclusive by construction: an orthogonal basis is taken whole, and
            # mixing it with hand-crafted scalars would double-count shape.
            if len(families) > 1:
                raise ValueError(
                    "The elliptic-Fourier family is used on its own - deselect the other families."
                )
            if not self.has_efd_coefficients():
                raise ValueError(
                    "EFD needs the raw Fourier coefficients. Re-run shape extraction with "
                    "'Raw elliptic-Fourier coefficients' enabled."
                )
            columns = [c for c in joined.columns if str(c).startswith("efd_")]
            standardize_mode = pca_kmeans.FEATURE_FAMILIES["efd"]["standardize"]
        else:
            columns = list(selected_features) if selected_features else pca_kmeans.family_columns(families)
            missing = [c for c in columns if c not in joined.columns]
            if missing:
                raise ValueError(
                    f"{len(missing)} feature(s) are not available in this project: {', '.join(missing[:6])}"
                    + (" ..." if len(missing) > 6 else "")
                    + ". Run the shape-descriptor step in Extract first."
                )
            standardize_mode = "global"

        frame = joined.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        before = len(frame)
        frame = frame.dropna()
        if frame.empty:
            raise ValueError("No cells have a complete set of the selected features.")
        if len(frame) < before:
            logger.info("Dropped %d cell(s) with missing values from classification", before - len(frame))
        return frame, standardize_mode

    def run_classification(
        self,
        n_clusters: Union[int, str] = "auto",
        k_range: tuple = (2, 9),
        random_state: Optional[int] = pca_kmeans.DEFAULT_RANDOM_STATE,
        feature_families: Optional[list] = None,
        selected_features: Optional[list] = None,
        prune_correlation: Optional[float] = None,
        dataset_mode: str = "time_series",
    ) -> dict:
        """PCA + K-means over the chosen feature set.

        `dataset_mode` decides whether frames share one feature space:

        - `"time_series"` (default, and v1's behaviour) pools every frame. This
          is correct for a movie of one specimen, and required if the result
          will be tracked -- tracking needs one consistent space across frames.
        - `"independent_samples"` runs PCA and K-means **per frame**. Use it
          when the frames are unrelated images, where pooling makes the leading
          component partly "which image is this": measured on
          `cp_dataset_puzzle-like_cells` (12 unrelated images), image identity
          explains 0.601 of PC1's variance under pooling. Cluster labels are
          then per-frame, so `cluster_1` in one image is unrelated to
          `cluster_1` in another.
        """
        features, standardize_mode = self._feature_frame(feature_families, selected_features)

        if isinstance(n_clusters, str) and n_clusters != "auto":
            n_clusters = int(n_clusters)
        k_range = tuple(k_range)
        output_dir = self.output_dirs["cell_classification_output"]

        if dataset_mode == "independent_samples":
            frames = pd.Series([str(i).split("_")[0] for i in features.index], index=features.index)
            parts = []
            for frame_id, chunk in features.groupby(frames, sort=False):
                if len(chunk) < 3:
                    logger.info("Skipping %s: only %d cell(s), too few to cluster", frame_id, len(chunk))
                    continue
                chunk_pca = pca_kmeans.compute_pca(
                    chunk, standardize_mode=standardize_mode, prune_correlation=prune_correlation
                )
                parts.append(pca_kmeans.cluster_cells_kmeans(
                    chunk_pca, n_clusters=n_clusters, k_range=k_range, random_state=random_state
                ))
            if not parts:
                raise ValueError("No frame had enough cells to cluster independently.")
            clustering_df = pd.concat(parts, ignore_index=True)
        elif dataset_mode == "time_series":
            pca_df = pca_kmeans.compute_pca(
                features,
                output_dir=output_dir,
                standardize_mode=standardize_mode,
                prune_correlation=prune_correlation,
            )
            clustering_df = pca_kmeans.cluster_cells_kmeans(
                pca_df, output_dir=output_dir, n_clusters=n_clusters, k_range=k_range, random_state=random_state,
            )
        else:
            raise ValueError(f"unknown dataset mode {dataset_mode!r}; expected 'time_series' or 'independent_samples'")

        clustering_df["modified"] = False
        self.clustering_data = clustering_df
        self._border_cell_cache.clear()

        return {
            "cluster_sizes": clustering_df["Cell_Type"].value_counts().to_dict(),
            "k_used": int(clustering_df["Cluster_Label"].nunique()),
            "features_used": list(features.columns),
            "feature_families": list(feature_families) if feature_families else ["basic"],
            "standardize_mode": standardize_mode,
            "dataset_mode": dataset_mode,
            "cells_used": int(len(features)),
        }

    def set_cluster_colors(self, color_map: dict) -> None:
        """Updates just the given cluster(s)' colors -- must merge, not
        replace, since the per-swatch color picker in the legend calls this
        with a single `{cluster_label: color}` entry at a time; a full
        replace would silently drop every other cluster's color back to
        `unclassified_color` on each individual edit."""
        self.cluster_color_map.update({int(k): v for k, v in color_map.items()})

    def get_cluster_legend(self) -> list[dict]:
        if self.clustering_data is None:
            return []
        counts = self.clustering_data.groupby(["Cluster_Label", "Cell_Type"]).size()
        legend = [
            {
                "cluster_label": int(label),
                "cell_type": cell_type,
                "color": self.cluster_color_map.get(int(label), "#808080"),
                "count": int(count),
                "hidden": int(label) in self.hidden_cluster_labels,
            }
            for (label, cell_type), count in counts.items()
        ]
        return sorted(legend, key=lambda entry: entry["cluster_label"])

    def set_hidden_clusters(self, hidden_labels: list) -> None:
        self.hidden_cluster_labels = {int(v) for v in hidden_labels}

    def set_show_cluster_borders(self, enabled: bool) -> None:
        self.show_cluster_borders = bool(enabled)

    def set_border_color(self, color: str) -> None:
        self.border_color = color

    def _get_border_cell_indices(self, frame_number: int, boundary_labels: np.ndarray) -> Optional[set]:
        """`Cell_Index` strings to highlight as "cluster borders" for
        `frame_number`: the cells that directly touch a cell of the other
        cluster (the physical interface, matching the manuscript's Fig. 2E
        leading-edge visualization), via
        `manual_correction_classification.find_boundary_cells`. `None` if
        there's nothing to highlight (not exactly 2 clusters, or no
        `clustering_data` yet).

        Deliberately NOT derived from `cells_info['leading_edge']` -- that
        column is a per-cell distance used for the mes/epi tracking split
        (`cell_tracking.split_by_leading_edge`), not an "is this cell on the
        boundary" flag, and a first cut at this function tried a
        distance-percentile cutoff on it, which picked a broad, arbitrary
        blob rather than the actual touching cells. `find_boundary_cells`
        doesn't need `cells_info`/a prior "Compute Cluster Borders" run at
        all -- it recomputes the same adjacency check live from
        `clustering_data` + this frame's `boundary_labels`.
        """
        if self.clustering_data is None:
            return None
        if frame_number in self._border_cell_cache:
            return self._border_cell_cache[frame_number]
        try:
            cells = manual_correction_classification.find_boundary_cells(
                boundary_labels, frame_number, self.clustering_data,
            )
        except ValueError:
            cells = None
        result = cells or None
        self._border_cell_cache[frame_number] = result
        return result

    def get_classification_overlay(self, frame_number: int, render_mode: str = "mask") -> Optional[str]:
        if self.clustering_data is None:
            return None
        loaded = self._load_frame(frame_number)
        if loaded is None:
            return None
        fill_labels, boundary_labels, base_image = loaded
        label_array = boundary_labels if render_mode == "outline" else fill_labels

        border_cell_indices = (
            self._get_border_cell_indices(frame_number, boundary_labels) if self.show_cluster_borders else None
        )

        overlay = rendering.render_classification_overlay(
            label_array, base_image, frame_number, self.clustering_data, self.cluster_color_map,
            hidden_labels=self.hidden_cluster_labels or None,
            border_cell_indices=border_cell_indices,
            border_color=self.border_color,
        )
        return rendering.encode_png_base64(overlay)

    def get_cell_at_pixel(self, frame_number: int, x: int, y: int) -> Optional[dict]:
        loaded = self._load_frame(frame_number)
        if loaded is None:
            return None
        fill_labels, _boundary_labels, _base_image = loaded
        cell_id = rendering.cell_id_at_pixel(fill_labels, int(x), int(y))
        if cell_id is None:
            return None
        cell_index = f"Cell{frame_number}_{cell_id}"
        if self.clustering_data is not None:
            row = self.clustering_data[self.clustering_data["Cell_Index"] == cell_index]
            if not row.empty:
                return {
                    "cell_index": cell_index,
                    "cluster_label": int(row["Cluster_Label"].iloc[0]),
                    "cell_type": row["Cell_Type"].iloc[0],
                }
        return {"cell_index": cell_index, "cluster_label": None, "cell_type": None}

    def reassign_cluster(self, cell_index: str, new_cluster_label: int) -> list[dict]:
        manual_correction_classification.update_clustering_data(
            self.clustering_data, cell_index, int(new_cluster_label),
        )
        self._border_cell_cache.clear()
        return self.get_cluster_legend()

    def save_classification(self) -> None:
        path = os.path.join(self.output_dirs["cell_classification_output"], "cells_clustering_results.xlsx")
        common.save_modified_rows(self.clustering_data, path)

    def can_compute_leading_edge(self) -> bool:
        if self.clustering_data is None:
            return False
        return int(self.clustering_data["Cluster_Label"].nunique()) == 2

    def compute_leading_edge_all_frames(self, threshold: float = 3) -> dict:
        if not self.can_compute_leading_edge():
            raise ValueError("compute_leading_edge requires exactly 2 clusters in the current classification.")

        frame_numbers = sorted({
            int(match.group(1))
            for match in (re.search(r"Cell(\d+)_", idx) for idx in self.cells_info["Cell Index"])
            if match
        })
        total = len(frame_numbers)
        processed = 0

        for i, frame_number in enumerate(frame_numbers):
            loaded = self._load_frame(frame_number)
            if loaded is None:
                continue
            _fill_labels, boundary_labels, _base_image = loaded
            self.cells_info = manual_correction_classification.compute_leading_edge(
                boundary_labels, frame_number, self.cells_info, self.clustering_data, threshold=threshold,
            )
            processed += 1
            self._notify("onProgress", {
                "stage": "leading_edge", "message": f"Frame {frame_number}", "percent": int((i + 1) / total * 100),
            })

        self.cells_info["modified"] = True
        return {"frames_processed": processed}

    def save_cells_info_leading_edge(self) -> None:
        path = os.path.join(self.output_dirs["cell_classification_output"], "Cells_info.xlsx")
        manual_correction_classification.save_leading_edge_column(self.cells_info, path)

    # ---- Track --------------------------------------------------------

    def get_tracking_mode_options(self) -> dict:
        has_leading_edge = self.cells_info is not None and "leading_edge" in self.cells_info.columns
        return {
            "has_leading_edge": bool(has_leading_edge),
            "split_available": bool(has_leading_edge) and self.can_compute_leading_edge(),
        }

    def start_tracking(self, mode: str = "single") -> bool:
        thread = threading.Thread(target=self._run_tracking, args=(mode,), daemon=True)
        thread.start()
        return True

    def _run_tracking(self, mode: str) -> None:
        try:
            cells_info_for_tracking = self.cells_info
            if mode == "single":
                cells_info_for_tracking = self.cells_info.drop(columns=["leading_edge"], errors="ignore")
            result = cell_tracking.track_all_populations(
                cells_info_for_tracking, self.output_dirs["cell_track_output"],
                progress_cb=self._progress_cb("track"),
                # Gives the cost function the masks back, so it can add the
                # mask-overlap term (see `cell_tracking.build_cost_matrix`).
                npy_folder=self.npy_folder,
                overlap_weight=self.overlap_weight,
                centroid_weight=self.centroid_weight,
                motion_compensation=self.motion_compensation,
                max_frame_gap=self.max_frame_gap,
                drift_mode=self.drift_mode,
                drift_by_frame=(self._registration().get("drift_by_frame")
                                if self.drift_mode != "none" else None),
            )
            self.tracking_data = result
            self.original_tracking_data = result.copy()
            self._notify("onTrackDone", {"track_count": int(len(result))})
        except Exception as e:
            logger.exception("Tracking failed")
            self._notify("onTrackError", {"message": str(e)})

    def get_stale_frames(self) -> dict:
        """Which frames' segmentation changed after the last tracking run.

        Compares each mask `.npy`'s mtime against the tracking output's, and
        reports each frame's `ismanual` count (Cellpose records which cells a
        human drew or edited; this project's real data carries 8-43 per
        frame). Purely advisory -- it pre-fills the re-track range and warns
        that tracking is out of date. EPySeg has no equivalent check at all,
        which is how a hand-corrected mask silently keeps stale tracks there.

        `ismanual` alone can't date a correction (a frame edited long ago
        still reports it), so mtime is what decides staleness and `ismanual`
        is context.
        """
        tracking_path = os.path.join(
            self.output_dirs["cell_track_output"], "all_cell_tracking", "all_cell_merged_tracking_results.xlsx",
        ) if self.output_dirs else None
        tracked_at = os.path.getmtime(tracking_path) if tracking_path and os.path.isfile(tracking_path) else None
        extract_path = os.path.join(
            self.output_dirs["cell_classification_output"], "Cells_info.xlsx",
        ) if self.output_dirs else None
        extracted_at = os.path.getmtime(extract_path) if extract_path and os.path.isfile(extract_path) else None

        # Staleness must be judged from the masks a HUMAN edits, not from
        # `npy_converted`. That folder is a derived artefact this code
        # regenerates on demand, so its mtimes always postdate the Extract
        # output and comparing against it flags every frame as stale --
        # measured: converted npy at 22:46, Cells_info at 18:29, source TIFFs
        # at 14:22. A check that always fires is worse than no check.
        # `list_segmentation_files` lists only `.npy`, so on a TIFF project it
        # returns nothing and every frame would silently fall back to the
        # derived mtime -- which is the false-positive this whole block exists
        # to avoid. Match both source formats explicitly.
        source_folder = self.mask_folder or self.npy_folder or ""
        source_times = {}
        try:
            names = sorted(
                f for f in os.listdir(source_folder)
                if f.lower().endswith((".npy", ".tif", ".tiff")) and not f.startswith(".")
            )
            for filename, frame in data_input.resolve_frame_numbers(names).items():
                source_times[frame] = os.path.getmtime(os.path.join(source_folder, filename))
        except Exception:
            logger.debug("Could not read source mask mtimes from %s", source_folder)

        frames = []
        seg_files = data_input.list_segmentation_files(self.npy_folder or "")
        for filename, frame in data_input.resolve_frame_numbers(seg_files).items():
            path = os.path.join(self.npy_folder, filename)
            mtime = os.path.getmtime(path)
            # No source mtime for this frame means the comparison is not
            # possible; treat it as NOT stale rather than blocking on a guess.
            source_mtime = source_times.get(frame)
            manual = 0
            try:
                flags = np.load(path, allow_pickle=True).item().get("ismanual")
                manual = int(np.asarray(flags).sum()) if flags is not None else 0
            except Exception:
                logger.debug("Could not read ismanual from %s", path)
            frames.append({
                "frame": frame,
                "stale": tracked_at is not None and mtime > tracked_at,
                # Stale relative to EXTRACT, which is the one that matters for
                # re-tracking: the cost reads centroids and radii from
                # `Cells_info` while the IoU term reads masks off disk, so a
                # mask edited after Extract makes the two disagree about what
                # the cells even are. `overlaps` is keyed by labels parsed from
                # the OLD `Cell{f}_{label}` strings, which after a
                # hand-correction may name different cells or cells that no
                # longer exist. Comparing only against the tracking output --
                # which is what this did -- cannot see that at all.
                "cells_info_stale": (extracted_at is not None and source_mtime is not None
                                    and source_mtime > extracted_at),
                "manual_cells": manual,
            })
        frames.sort(key=lambda f: f["frame"])
        stale = [f["frame"] for f in frames if f["stale"]]
        needs_extract = [f["frame"] for f in frames if f["cells_info_stale"]]
        return {
            "tracked": tracked_at is not None,
            "extracted": extracted_at is not None,
            "frames": frames,
            "stale_frames": stale,
            "cells_info_stale_frames": needs_extract,
            "suggested_range": [min(stale), max(stale)] if stale else None,
        }

    def start_local_retrack(self, start_frame: int, end_frame: int) -> bool:
        thread = threading.Thread(
            target=self._run_local_retrack, args=(int(start_frame), int(end_frame)), daemon=True,
        )
        thread.start()
        return True

    def _run_local_retrack(self, start_frame: int, end_frame: int) -> None:
        try:
            if self.cells_info is None or self.tracking_data is None:
                raise ValueError("Extract and track the movie before re-tracking a range.")
            cb = self._progress_cb("retrack")

            # A5: refuse rather than silently mixing old features with new
            # masks. The whole point of this feature is "you hand-corrected the
            # segmentation", and that is exactly the state in which
            # `Cells_info` no longer describes those masks.
            stale = self.get_stale_frames().get("cells_info_stale_frames") or []
            affected = [f for f in stale if start_frame - 1 <= f <= end_frame + 1]
            if affected:
                raise ValueError(
                    "Frame(s) %s changed after the last Extract, so Cells_info no longer describes "
                    "these masks. Re-run Extract first -- re-tracking now would combine old centroids "
                    "and radii with new masks, and the overlap term would be keyed by labels that may "
                    "no longer exist." % ", ".join(str(f) for f in affected)
                )

            cb(f"Re-tracking frames {start_frame}-{end_frame}...", 10)
            registration = self._registration() if self.drift_mode != "none" else {}
            shared = dict(
                # Must be the same cost the full run used, or a "repair" can
                # land on a link the full run would never have chosen. Every
                # argument `_run_tracking` passes is passed here too; the four
                # LAP parameters are deliberately excluded (see
                # `cell_tracking.RETRACK_EXCLUDED_PARAMETERS`).
                npy_folder=self.npy_folder,
                overlap_weight=self.overlap_weight,
                centroid_weight=self.centroid_weight,
                motion_compensation=self.motion_compensation,
                drift_mode=self.drift_mode,
                drift_by_frame=registration.get("drift_by_frame"),
                max_frame_gap=self.max_frame_gap,
            )

            # A3: the full run splits mes/epi into two INDEPENDENT assignment
            # problems. Re-tracking the unsplit table solves one problem across
            # both, which lets a mes cell link to an epi cell -- something the
            # full run cannot do -- and raises the displacement ceiling to the
            # max over both populations. Reuse `split_by_leading_edge` so the
            # split logic has exactly one implementation.
            info = self.cells_info
            if "leading_edge" in info.columns:
                mes, epi = cell_tracking.split_by_leading_edge(info)
                result = {"reconciled": [], "new_tracks": [], "frames": [], "relinked": 0}
                for part in (mes, epi):
                    if part.empty:
                        continue
                    piece = cell_tracking.retrack_frame_range(
                        part.set_index("Cell Index"), self.tracking_data,
                        start_frame, end_frame, **shared,
                    )
                    self.tracking_data = piece.pop("tracking_data")
                    for key in ("reconciled", "new_tracks"):
                        result[key].extend(piece.get(key, []))
                    result["relinked"] += piece.get("relinked", 0)
                    result["frames"] = piece.get("frames", result["frames"])
            else:
                result = cell_tracking.retrack_frame_range(
                    info.set_index("Cell Index"), self.tracking_data,
                    start_frame, end_frame, **shared,
                )
                self.tracking_data = result.pop("tracking_data")

            cb("Re-track complete.", 100)
            self._notify("onRetrackDone", {**result, "track_count": int(len(self.tracking_data))})
        except Exception as e:
            logger.exception("Local re-track failed")
            self._notify("onRetrackError", {"message": str(e)})

    def set_track_colors(self, matched_color: str, unmatched_color: str) -> None:
        self.matched_color = matched_color
        self.unmatched_color = unmatched_color

    def set_dynamics_colors(self, division: str, delamination: str, t1: str) -> None:
        self.dynamics_colors = {"division": division, "delamination": delamination, "t1": t1}

    def _dynamics_states(self, frame_number: int) -> dict:
        """`{raw_label: {"event": ..., "confirmed": bool}}` for one frame.

        Precedence division > delamination > T1: a dividing cell is also
        rearranging its neighbours, and the rarer, more consequential event is
        the one worth seeing.
        """
        found = self.get_dynamics_candidates(frame_number)
        if "error" in found:
            return {}
        label_of = {t: l for l, t in self._label_to_track(frame_number).items() if t is not None}
        states: dict = {}

        # A T1 needs no human confirmation, but it does need to have settled:
        # a cell whose exchange reverses next frame renders faded, the same
        # treatment an unconfirmed division candidate gets.
        t1 = found.get("t1", {})
        settled_labels = {int(label) for label in t1.get("counts_persistent", {})}
        for entry in t1.get("counts", {}):
            states[int(entry)] = {"event": "t1", "confirmed": int(entry) in settled_labels}
        for cand in found["delaminations"]:
            label = label_of.get(cand["track"])
            if label is not None:
                states[int(label)] = {
                    "event": "delamination",
                    "confirmed": f"delamination:{cand['track']}:{cand['frame']}" in self._confirmed_event_keys,
                }
        for cand in found["divisions"]:
            label = label_of.get(cand["mother"])
            if label is not None:
                states[int(label)] = {
                    "event": "division",
                    "confirmed": f"division:{cand['mother']}:{cand['frame']}" in self._confirmed_event_keys,
                }
        return states

    def get_environment_thresholds(self) -> dict:
        """The two environment-score tiers, and what each means."""
        return {
            "dead_band": float(self.environment_dead_band),
            "repair_threshold": float(self.environment_repair_threshold),
        }

    def set_environment_thresholds(self, dead_band: float = None, repair_threshold: float = None) -> dict:
        """Set the dead band (above it a cell is drawn clean) and the repair
        tier (at or below it a cell is offered to the correction tools).

        Two tiers rather than one because Tissue Analyzer uses two: 0.75 to
        decide whether to FLAG (`take_decision`) and 0.5 to decide whether to
        attempt a REPAIR (`optimize_score`). Collapsing them would either flag
        too much or repair too eagerly.
        """
        if dead_band is not None:
            self.environment_dead_band = min(1.0, max(0.0, float(dead_band)))
        if repair_threshold is not None:
            self.environment_repair_threshold = min(1.0, max(0.0, float(repair_threshold)))
        return self.get_environment_thresholds()

    def get_environment_repair_candidates(self, frame_number: int, direction: str = "prev") -> dict:
        """Cells at or below the repair tier, worst first -- the second of
        TA's two thresholds, wired to the existing correction tools.

        Deliberately not every flagged cell: TA only attempts repair on cells
        scoring <= 0.5 and skips perfect ones outright
        (`optimize_score`: `if current_score >= 1.0: continue`).
        """
        scores = self._environment_scores(frame_number, direction)
        if not scores:
            return {"candidates": [], "threshold": float(self.environment_repair_threshold)}
        worst = sorted(
            ((int(label), float(value)) for label, value in scores.items()
             if value <= self.environment_repair_threshold),
            key=lambda pair: pair[1],
        )
        return {
            "candidates": [{"label": label, "score": round(value, 3)} for label, value in worst],
            "threshold": float(self.environment_repair_threshold),
            "flagged_total": int(sum(1 for v in scores.values() if v < self.environment_dead_band)),
            "n": len(scores),
        }

    # --- Ground-truth census: one link at a time -----------------------------
    #
    # Round 14 §A3 left a question no reference-free metric can answer: every
    # geometry-flagged disagreement had its rival already claimed by another
    # track, which is a global solver trading off against a greedy per-cell
    # reference. Only a human breaks that tie. See `census` for why this is a
    # census and never a sample, and for the decoy / repeat / timing checks.

    def _census_masks(self) -> dict:
        masks = {}
        for frame in tracking_benchmark.frame_numbers(self.tracking_data):
            loaded = self._load_frame(frame)
            if loaded and loaded[0] is not None:
                masks[frame] = loaded[0]
        return masks

    def start_census(self, kind: str = "geometry_flagged", frame: int = None,
                     decoy_rate: float = census.DEFAULT_DECOY_RATE,
                     repeat_rate: float = census.DEFAULT_REPEAT_RATE,
                     seed: int = 0, labeller: str = "") -> dict:
        """Build the labelling queue. `kind` is `geometry_flagged` (the whole
        movie's disagreements) or `frame_pair` (every link in one pair).

        The judge is registered whenever the tracking was, and the answer says
        so. An overlap reference computed on unregistered masks scoring a
        registered tracker is the preprocessing mismatch that has already moved
        one figure from 4.9% to 14.0% on nothing but that switch.
        """
        if self.tracking_data is None:
            return {"error": "Track the movie before starting a census."}
        masks = self._census_masks()
        if len(masks) < 2:
            return {"error": "Need at least two frames with masks."}
        register = self.drift_mode != "none"
        drift = self._registration().get("drift_by_frame") if register else None
        if kind == "frame_pair":
            if frame is None:
                return {"error": "A frame_pair census needs a frame number."}
            items = census.frame_pair_items(self.tracking_data, masks, int(frame))
        elif kind == "geometry_flagged":
            items = census.geometry_flagged_items(
                self.tracking_data, masks, register=register, drift_by_frame=drift)
        else:
            return {"error": f"Unknown census kind {kind!r}."}
        if not items:
            return {"error": "Nothing to label: that set is empty."}

        queue = census.build_queue(items, masks, decoy_rate=float(decoy_rate),
                                   repeat_rate=float(repeat_rate), seed=int(seed))
        self._census = {
            "queue": queue, "masks": masks, "kind": kind, "frame": frame,
            "register": register, "labeller": labeller,
            "session_id": coverage_log.new_session_id(),
            "path": os.path.join(self.output_dirs["cell_track_output"], "ground_truth_links.csv"),
        }
        self._write_census_provenance()
        return {"n": len(queue), "n_real": len(items),
                "n_decoys": sum(1 for e in queue if e["is_decoy"]),
                "n_repeats": sum(1 for e in queue if e["is_repeat"]),
                "kind": kind, "register": register,
                "session_id": self._census["session_id"], "path": self._census["path"]}

    def _write_census_provenance(self) -> None:
        """Ground truth collected against one tracking run is not automatically
        valid for another -- the link set itself changes. Same treatment the
        tracking baselines get."""
        state = self._census
        record = {
            "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "session_id": state["session_id"],
            "kind": state["kind"],
            "frame": state["frame"],
            "labeller": state["labeller"],
            "judge_registered": state["register"],
            "mask_folder": self.mask_folder,
            "frames": sorted(state["masks"]),
            "queue_size": len(state["queue"]),
            "tracking": {
                "tracks": int(len(self.tracking_data)),
                "overlap_weight": self.overlap_weight,
                "centroid_weight": self.centroid_weight,
                "motion_compensation": self.motion_compensation,
                "max_frame_gap": self.max_frame_gap,
                "drift_mode": self.drift_mode,
            },
        }
        try:
            with open(state["path"] + ".provenance.json", "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, sort_keys=True)
                handle.write("\n")
        except OSError as error:
            logger.warning("could not write census provenance: %s", error)

    def get_census_item(self, position: int, pad: int = 28) -> dict:
        """One item, with the two crops the blink toggles between.

        `before` is the cell at frame f; `after` is the SHOWN successor at
        f+1 -- which for a decoy is deliberately the wrong cell. Both crops
        share one window so the blink is a true toggle: two differently-framed
        crops would make every link look like it jumped.
        """
        state = getattr(self, "_census", None)
        if not state:
            return {"error": "Start a census first."}
        queue = state["queue"]
        if not 0 <= int(position) < len(queue):
            return {"error": f"Position {position} is outside the queue (0-{len(queue) - 1})."}
        item = queue[int(position)]
        masks = state["masks"]
        frame = item["frame"]
        before_labels, after_labels = masks.get(frame), masks.get(frame + 1)
        if before_labels is None or after_labels is None:
            return {"error": f"No masks for frames {frame}/{frame + 1}."}

        window = self._census_window(before_labels, after_labels, item, pad)
        if window is None:
            return {"error": f"Cell {item['label']} has no pixels at frame {frame}."}

        candidates = self._census_candidates(item, after_labels, window)
        return {
            "position": int(position), "total": len(queue),
            "item_id": item["item_id"], "frame": frame, "label": item["label"],
            "track_id": item["track_id"], "shown_label": item["shown_label"],
            "is_repeat": bool(item["is_repeat"]),
            # `is_decoy` is deliberately NOT returned: the labeller must not be
            # able to see which items are checks.
            "reference_iou": item.get("reference_iou"),
            "tracker_iou": item.get("tracker_iou"),
            "before": self._census_crop(frame, before_labels, item["label"], window, "#00ff88"),
            "after": (self._census_crop(frame + 1, after_labels, item["shown_label"], window, "#ffcc00")
                      if item["shown_label"] is not None else None),
            "candidates": candidates,
        }

    def _census_window(self, before_labels, after_labels, item, pad):
        ys, xs = np.nonzero(before_labels == int(item["label"]))
        if not len(ys):
            return None
        if item["shown_label"] is not None:
            ys2, xs2 = np.nonzero(after_labels == int(item["shown_label"]))
            if len(ys2):
                ys = np.concatenate([ys, ys2]); xs = np.concatenate([xs, xs2])
        height, width = before_labels.shape
        return (max(0, int(ys.min()) - pad), min(height, int(ys.max()) + pad + 1),
                max(0, int(xs.min()) - pad), min(width, int(xs.max()) + pad + 1))

    def _census_crop(self, frame_number, labels, label, window, color):
        from PIL import Image
        loaded = self._load_frame(frame_number)
        base = loaded[2] if loaded else None
        if base is None:
            base = np.zeros(labels.shape + (3,), dtype=np.uint8)
        top, bottom, left, right = window
        image = rendering.render_cell_highlight(labels, base, int(label), color) \
            if label is not None else Image.fromarray(base)
        return rendering.encode_png_base64(image.crop((left, top, right, bottom)))

    def _census_candidates(self, item, after_labels, window):
        """The alternatives offered when the verdict is "wrong": the overlap
        reference's own pick, then the nearest other cells. Numbered, so the
        labeller presses a digit instead of clicking a small target."""
        shown = item.get("shown_label")
        options, seen = [], {shown}
        for label in (item.get("reference_label"), item.get("true_label")):
            if label is not None and label not in seen:
                options.append(int(label)); seen.add(int(label))
        centres = census._centroids(after_labels)
        origin = centres.get(int(shown)) if shown is not None else None
        if origin is not None:
            nearby = sorted(((np.hypot(c[0] - origin[0], c[1] - origin[1]), l)
                             for l, c in centres.items() if l not in seen),
                            key=lambda pair: pair[0])
            options.extend(int(l) for _, l in nearby[:3])
        return [{"label": label,
                 "image": self._census_crop(item["frame"] + 1, after_labels, label, window, "#ff6b6b")}
                for label in options[:3]]

    def submit_census_verdict(self, position: int, verdict: str,
                              chosen_label: int = None, ms_elapsed: float = 0.0) -> dict:
        state = getattr(self, "_census", None)
        if not state:
            return {"error": "Start a census first."}
        queue = state["queue"]
        if not 0 <= int(position) < len(queue):
            return {"error": f"Position {position} is outside the queue."}
        try:
            census.append_verdict(state["path"], state["session_id"], queue[int(position)],
                                  verdict, chosen_label, ms_elapsed, labeller=state["labeller"])
        except ValueError as error:
            return {"error": str(error)}
        return {"saved": True, "next": int(position) + 1, "total": len(queue)}

    def get_census_summary(self) -> dict:
        """Counts, decoy pass-rate, intra-rater agreement, median seconds.

        The last three are not decoration: an accuracy from these labels is
        only meaningful against the self-consistency ceiling measured in the
        same session, and only if the decoys were caught.
        """
        state = getattr(self, "_census", None)
        path = state["path"] if state else (
            os.path.join(self.output_dirs["cell_track_output"], "ground_truth_links.csv")
            if self.output_dirs else None)
        if path is None:
            return {"error": "Set up the project paths first."}
        summary = census.summarise(path, state["session_id"] if state else None)
        if state:
            summary["queue_size"] = len(state["queue"])
            summary["judge_registered"] = state["register"]
        return summary

    # --- Coverage log: what the human actually looked at ---------------------
    #
    # The Correct Tracking view produces corrections, which are labelled errors.
    # It has never produced the other half: a cell that was on screen, clicked,
    # and deliberately left alone. Without that, "unchanged" and "never looked
    # at" are the same value, which is why no ground truth has come out of this
    # tool. These three methods record the difference; `coverage_log` turns it
    # into weak positives, and is deliberately conservative about what counts.

    def _coverage_log_path(self) -> Optional[str]:
        if not self.output_dirs:
            return None
        return os.path.join(self.output_dirs["cell_track_output"], "coverage_log.csv")

    def log_coverage_event(self, event: str, frame_number: int, cell_index: str = None,
                           track_id: int = None, ms_elapsed: float = None,
                           zoom_percent: float = None, render_mode: str = "",
                           color_mode: str = "", detail: str = "") -> dict:
        """Append one `frame_shown` / `cell_inspected` / `cell_changed` event.

        Never raises into the UI: a failed log line must not interrupt a
        correction the user is in the middle of making. It returns
        `{"logged": False, ...}` instead, so a silently-empty log is still
        visible to anyone who looks at the return value.
        """
        path = self._coverage_log_path()
        if path is None:
            return {"logged": False, "reason": "no output directory yet"}
        try:
            coverage_log.append(
                path, self.coverage_session_id, event, int(frame_number),
                cell_index=cell_index, track_id=track_id, ms_elapsed=ms_elapsed,
                zoom_percent=zoom_percent, render_mode=render_mode,
                color_mode=color_mode, detail=detail,
            )
        except Exception as error:  # noqa: BLE001 - never break a correction
            logger.warning("coverage log write failed: %s", error)
            return {"logged": False, "reason": str(error)}
        return {"logged": True, "session_id": self.coverage_session_id}

    def get_coverage_summary(self, min_dwell_ms: float = None) -> dict:
        """Per-frame coverage against the cells that exist in `cells_info`.

        Distinguishes `inspected_unchanged` (a weak positive) from
        `never_displayed` (no evidence). Cells are the ground set from
        `cells_info`, not from the tracking table, so a cell the tracker never
        claimed still counts as uncovered rather than vanishing.
        """
        path = self._coverage_log_path()
        if path is None:
            return {"error": "Set up the project paths first."}
        if self.cells_info is None:
            return {"error": "Extract cell info before reading coverage."}
        index = (self.cells_info["Cell Index"] if "Cell Index" in self.cells_info.columns
                 else self.cells_info.index)
        cells_by_frame: dict = {}
        for value in index:
            match = re.match(r"Cell(\d+)_", str(value))
            if match:
                cells_by_frame.setdefault(int(match.group(1)), []).append(str(value))
        kwargs = {} if min_dwell_ms is None else {"min_dwell_ms": float(min_dwell_ms)}
        summary = coverage_log.summarise(path, cells_by_frame, **kwargs)
        summary["weak_positives"] = len(coverage_log.weak_positives(summary))
        return summary

    def save_ground_truth_links(self, links: dict = None, path: str = None) -> dict:
        """Write hand-labelled links to `cell_track_output/ground_truth_links.csv`.

        `links` is `{frame: {label_in_frame: label_in_next_frame}}`; omitted, it
        exports the CURRENT tracking as a starting point to correct, which is
        far less work than typing a CSV from nothing.

        This is the half of Round 10 that did not land: `load_link_csv` and
        `save_link_csv` existed in the harness but were reachable from no UI, so
        GBE ground truth could only be produced by hand-writing a file.
        Ground truth remains the only judge that measures an accuracy.
        """
        if self.tracking_data is None:
            return {"error": "Track the movie before exporting links."}
        if path is None:
            path = os.path.join(self.output_dirs["cell_track_output"], "ground_truth_links.csv")
        if links is None:
            links = {}
            for frame in tracking_benchmark.frame_numbers(self.tracking_data)[:-1]:
                here = self._label_to_track(frame)
                there = {t: l for l, t in self._label_to_track(frame + 1).items()}
                pairs = {label: there[track] for label, track in here.items() if track in there}
                if pairs:
                    links[frame] = pairs
        tracking_benchmark.save_link_csv(links, path)
        return {"path": path, "frames": len(links), "links": sum(len(v) for v in links.values())}

    def load_ground_truth_links(self, path: str = None) -> dict:
        """Load hand-labelled links and score the current tracking against them.

        **This is the only judge in the app that reports an accuracy.** Every
        other number is agreement with a reference that has its own bias.
        """
        if path is None:
            path = os.path.join(self.output_dirs["cell_track_output"], "ground_truth_links.csv")
        if not os.path.exists(path):
            return {"error": f"No ground-truth file at {path}"}
        links = tracking_benchmark.load_link_csv(path)
        if self.tracking_data is None:
            return {"error": "Track the movie before scoring against ground truth."}
        result = tracking_benchmark.ground_truth(self.tracking_data, gt_links=links)
        result["path"] = path
        return result

    def set_agreement_colors(self, agrees: str, disagrees: str, unjudgeable: str, broken: str) -> None:
        """Colours for the link-agreement overlay's four states."""
        self.agreement_colors = {
            "agrees": agrees, "disagrees": disagrees,
            "unjudgeable": unjudgeable, "broken": broken,
        }

    def set_lifecycle_colors(self, lost_color: str, new_color: str, both_color: str) -> None:
        """Colors for the track-lifecycle overlay. Kept separate from
        `set_track_colors` rather than overloading it: `matched_color`/
        `unmatched_color` mean something different and are shared with the
        uniform mode."""
        self.lifecycle_colors = {"lost": lost_color, "new": new_color, "both": both_color}

    def _get_neighbor_graph(self, frame_number: int) -> Optional[dict]:
        """`adjacency.build_neighbor_graph` for `frame_number`, cached.
        Always built from `fill_labels` -- never the boundary array, which
        would silently yield near-empty neighbor sets. Membership-tested (not
        truthiness-tested) so a legitimately-`None` result caches too, matching
        `_get_border_cell_indices`."""
        if frame_number in self._neighbor_graph_cache:
            return self._neighbor_graph_cache[frame_number]
        loaded = self._load_frame(frame_number)
        graph = adjacency.build_neighbor_graph(loaded[0]) if loaded is not None else None
        self._neighbor_graph_cache[frame_number] = graph
        return graph

    def _label_to_track(self, frame_number: int) -> dict:
        """`{raw_label: track_id}` for `frame_number`, from the live
        `tracking_data` -- recomputed per call (cheap) so it is never stale
        after a correction."""
        frame_col = f"Frame{frame_number}"
        if self.tracking_data is None or frame_col not in self.tracking_data.columns:
            return {}
        id_col = self.tracking_data.columns[0]
        mapping = {}
        for _, row in self.tracking_data[[id_col, frame_col]].dropna(subset=[frame_col]).iterrows():
            match = re.search(r"Cell\d+_(\d+)", str(row[frame_col]))
            if match:
                mapping.setdefault(int(match.group(1)), int(row[id_col]))
        return mapping

    def _environment_scores(self, frame_number: int, direction: str) -> dict:
        """Per-cell environment score for `frame_number` compared against the
        previous or next frame (Tissue Analyzer's Fig. 3g/h respectively), or
        the worse of the two with `direction="both"`.

        TA prints g and h as two panels side by side and tells the reader to
        scrutinise any cell dark in EITHER (Aigouy & Prud'homme: "Any
        dark-colored cell in g, h has undergone important environmental
        changes"). A single-image UI can't show two panels at once, and making
        the user toggle back and forth loses exactly the comparison the figure
        is making -- so `"both"` takes the per-cell minimum, which is the same
        reading rule in one view. Use the individual directions to tell WHICH
        side a change happened on, which is what distinguishes a cell that
        just appeared from one that is about to vanish.

        Empty dict if the compared frame doesn't exist."""
        if direction == "both":
            prev_scores = self._environment_scores(frame_number, "prev")
            next_scores = self._environment_scores(frame_number, "next")
            if not prev_scores:
                return next_scores
            if not next_scores:
                return prev_scores
            return {
                label: min(value, next_scores.get(label, value))
                for label, value in prev_scores.items()
            }

        other_frame = frame_number - 1 if direction == "prev" else frame_number + 1
        graph_current = self._get_neighbor_graph(frame_number)
        graph_other = self._get_neighbor_graph(other_frame)
        if not graph_current or not graph_other:
            return {}
        return adjacency.environment_scores(
            graph_current, graph_other,
            self._label_to_track(frame_number), self._label_to_track(other_frame),
        )

    def get_tracking_overlay(
        self, frame_number: int, render_mode: str = "mask", show_labels: bool = True,
        track_color_mode: str = "uniform", identity_palette: str = "golden_angle",
        label_font_size: int = 10, environment_lut: str = "viridis",
        environment_direction: str = "prev", tint_other_cells: bool = False,
        show_nematic_axes: bool = True,
    ) -> Optional[str]:
        if self.tracking_data is None:
            return None
        loaded = self._load_frame(frame_number)
        if loaded is None:
            return None
        fill_labels, boundary_labels, base_image = loaded
        label_array = boundary_labels if render_mode == "outline" else fill_labels

        environment_scores = (
            self._environment_scores(frame_number, environment_direction)
            if track_color_mode == "environment" else None
        )
        lifecycle_states = (
            manual_correction_tracking.classify_track_lifecycle(self.tracking_data, frame_number)
            if track_color_mode == "lifecycle" else None
        )
        dynamics_states = self._dynamics_states(frame_number) if track_color_mode == "dynamics" else None

        agreement_states = None
        if track_color_mode == "agreement":
            agreement_states = self.get_link_agreement(frame_number).get("states")

        nematic_axes = None
        if track_color_mode == "dynamics" and show_nematic_axes:
            events = self.get_t1_events(frame_number)
            nematic_axes = [
                {**axis, "length": 14,
                 "color": "#66ccff" if axis["type"] == "gain" else "#ff9933"}
                for axis in events.get("axes", [])
            ]

        overlay = rendering.render_tracking_overlay(
            label_array, base_image, frame_number, self.tracking_data, self.matched_color, self.unmatched_color,
            label_cells=show_labels, track_color_mode=track_color_mode, identity_palette=identity_palette,
            label_font_size=label_font_size,
            environment_scores=environment_scores, environment_lut=environment_lut,
            environment_dead_band=self.environment_dead_band,
            lifecycle_states=lifecycle_states, lifecycle_colors=self.lifecycle_colors,
            tint_other_cells=tint_other_cells,
            dynamics_states=dynamics_states, dynamics_colors=self.dynamics_colors,
            nematic_axes=nematic_axes,
            agreement_states=agreement_states, agreement_colors=self.agreement_colors,
        )
        return rendering.encode_png_base64(overlay)

    def get_tracking_cell_at_pixel(self, frame_number: int, x: int, y: int) -> Optional[dict]:
        loaded = self._load_frame(frame_number)
        if loaded is None:
            return None
        fill_labels, _boundary_labels, _base_image = loaded
        cell_id = rendering.cell_id_at_pixel(fill_labels, int(x), int(y))
        if cell_id is None:
            return None
        cell_index = f"Cell{frame_number}_{cell_id}"
        track_id = None
        frame_col = f"Frame{frame_number}"
        if self.tracking_data is not None and frame_col in self.tracking_data.columns:
            match = self.tracking_data[self.tracking_data[frame_col] == cell_index]
            if not match.empty:
                track_id = int(match.iloc[0, 0])
        return {"cell_index": cell_index, "cell_id": int(cell_id), "track_id": track_id}

    def reassign_track(self, frame_number: int, old_id: int, new_id: int, propagate: bool = False) -> dict:
        """`propagate=True` merges `old_id`'s cells from `frame_number`
        onward into `new_id` (`manual_correction_tracking
        .merge_tracks_from_frame`) instead of only fixing that one frame --
        for a mismatch that cascades forward (a wrong track number that
        then keeps extending itself frame after frame), a single-frame fix
        leaves every later frame still wrongly attached to `old_id`."""
        try:
            if propagate:
                self.tracking_data, displaced = manual_correction_tracking.merge_tracks_from_frame(
                    self.tracking_data, int(frame_number), int(old_id), int(new_id),
                )
            else:
                self.tracking_data, displaced = manual_correction_tracking.update_tracking_data(
                    self.tracking_data, self.original_tracking_data, int(frame_number), int(old_id), int(new_id),
                )
            return {"success": True, "displaced": displaced}
        except ValueError as e:
            return {"error": str(e)}

    def compute_propagation_stop_frame(
        self, frame_number: int, old_id: int, new_id: int, max_frames_ahead: int = 20,
    ) -> dict:
        if self.tracking_data is None or self.cells_info is None:
            return {"error": "No tracking data loaded."}
        try:
            return manual_correction_tracking.find_propagation_stop_frame(
                self.tracking_data, self.cells_info,
                lambda f: (self._load_frame(f) or (None, None, None))[0],
                int(frame_number), int(old_id), int(new_id), max_frames_ahead=int(max_frames_ahead),
            )
        except ValueError as e:
            return {"error": str(e)}

    def score_propagation_stop_frames(
        self, frame_number: int, old_id: int, new_id: int, end_frame: int, span: int = 3,
    ) -> dict:
        """Try candidate stop frames around `end_frame`, score each, keep the best.

        Tissue Analyzer's `optimize_score` applies a candidate correction,
        rescores, and reverts when the score did not improve
        (`tracking_error_detector_and_fixer.py:1096`). Our smart-stop only ever
        suggested. Because its output is a single scalar, acceptance testing
        degenerates to a **line search**, which is the simplest place to add it.

        Two constraints, both places this project has already been burned:

        1. **The score must not be the quantity that produced the candidate.**
           `find_propagation_stop_frame` decides by neighbour settlement, so
           scoring by neighbour settlement would be self-confirming -- the same
           failure as the IoU circularity and the residual selection bias. This
           scores with `iou_agreement` on REGISTERED masks over the affected
           frame range, which shares no information with the settlement rule.
        2. **Reverting cannot be an inverse operation.**
           `merge_tracks_from_frame` may trigger `_evict_future_if_colliding`,
           relocating a third track's future onto a fresh id, and that has no
           clean inverse. Every trial therefore runs against a deep copy and the
           live table is never touched until the caller applies a choice.
        """
        if self.tracking_data is None:
            return {"error": "No tracking data loaded."}
        masks = {}
        low = max(int(frame_number), 1)
        high = int(end_frame) + int(span) + 1
        for frame in self.get_frame_numbers():
            if low <= frame <= high:
                loaded = self._load_frame(frame)
                if loaded is not None:
                    masks[frame] = loaded[0]
        if len(masks) < 2:
            return {"error": "Not enough frames with masks to score this range."}
        drift = self._registration().get("drift_by_frame") if self.register_reference else None

        snapshot = self.tracking_data.copy(deep=True)
        candidates, best = [], None
        for candidate in range(max(int(frame_number) + 1, int(end_frame) - int(span)),
                               int(end_frame) + int(span) + 1):
            trial = snapshot.copy(deep=True)
            try:
                trial, _ = manual_correction_tracking.merge_tracks_from_frame(
                    trial, int(frame_number), int(old_id), int(new_id), end_frame=candidate,
                )
            except ValueError:
                continue
            score = tracking_benchmark.iou_agreement(
                trial, masks, register=bool(drift), drift_by_frame=drift,
            )
            entry = {"end_frame": candidate,
                     "wrong_rate": round(float(score["wrong_rate"]), 3),
                     "n_judged": int(score["n_judged"]),
                     "suggested": candidate == int(end_frame)}
            candidates.append(entry)
            if best is None or entry["wrong_rate"] < best["wrong_rate"]:
                best = entry

        # The live table must be untouched by scoring. Asserted, not assumed:
        # this is the whole safety property of an accept-and-revert loop.
        assert self.tracking_data.astype(str).equals(snapshot.astype(str)), \
            "scoring must not mutate the live tracking table"

        return {
            "candidates": candidates,
            "best": best,
            "suggested": int(end_frame),
            "registered": bool(drift),
            "note": ("Scored by disagreement with the overlap reference over the affected frames -- "
                     "deliberately NOT by neighbour settlement, which is what chose the stop frame."),
        }

    def apply_smart_stop_merge(self, frame_number: int, old_id: int, new_id: int, end_frame: int) -> dict:
        try:
            self.tracking_data, displaced = manual_correction_tracking.merge_tracks_from_frame(
                self.tracking_data, int(frame_number), int(old_id), int(new_id), end_frame=int(end_frame),
            )
            return {"success": True, "displaced": displaced}
        except ValueError as e:
            return {"error": str(e)}

    def compute_confirmed_identity_moves(
        self, frame_number: int, track_id: int, max_frames_ahead: int = 20,
        restrict_to_unmatched: bool = False,
    ) -> dict:
        """Diagnostic-only: plans (but does not apply) the position-based
        propagation used after a Suggest-match attach, in case the same
        physical cell was subsequently claimed by a sequence of DIFFERENT
        wrong tracks rather than one consistently-wrong one (see
        `manual_correction_tracking.find_confirmed_identity_moves`).

        `restrict_to_unmatched` (default `True`) keeps the walk to cells no
        track already claims -- see that function's docstring for why this
        matters (without it the walk can drift onto a neighbour another
        track legitimately owns and then follow the wrong cell for every
        later frame)."""
        if self.tracking_data is None or self.cells_info is None:
            return {"error": "No tracking data loaded."}
        try:
            return manual_correction_tracking.find_confirmed_identity_moves(
                self.tracking_data, self.cells_info,
                lambda f: (self._load_frame(f) or (None, None, None))[0],
                int(frame_number), int(track_id), max_frames_ahead=int(max_frames_ahead),
                restrict_to_unmatched=bool(restrict_to_unmatched),
            )
        except ValueError as e:
            return {"error": str(e)}

    def apply_confirmed_identity_moves(self, track_id: int, moves: list) -> dict:
        try:
            self.tracking_data, displaced_list = manual_correction_tracking.apply_confirmed_identity_moves(
                self.tracking_data, int(track_id), moves,
            )
            return {"success": True, "displaced_list": displaced_list}
        except ValueError as e:
            return {"error": str(e)}

    def swap_tracks(self, frame_number: int, id_a: int, id_b: int, propagate: bool = False) -> dict:
        """`propagate=True` exchanges `id_a`/`id_b` at every frame from
        `frame_number` onward (`manual_correction_tracking
        .swap_tracks_from_frame`) instead of only that one frame -- mirrors
        `reassign_track`'s own propagate option."""
        try:
            if propagate:
                result = manual_correction_tracking.swap_tracks_from_frame(
                    self.tracking_data, int(frame_number), int(id_a), int(id_b),
                )
                return {"success": True, **result}
            manual_correction_tracking.swap_tracks_at_frame(
                self.tracking_data, int(frame_number), int(id_a), int(id_b),
            )
            return {"success": True}
        except ValueError as e:
            return {"error": str(e)}

    def suggest_match_candidates(self, frame_number: int, cell_index: str, top_k: int = 3) -> dict:
        if self.cells_info is None or self.tracking_data is None:
            return {"candidates": []}
        try:
            # Topology confirmation, when both frames' graphs are available.
            # A confirmer only: it promotes a candidate geometry already
            # proposed and never demotes one -- see `suggest_match_candidates`.
            graphs = None
            previous = self._get_neighbor_graph(int(frame_number) - 1)
            current = self._get_neighbor_graph(int(frame_number))
            if previous and current:
                graphs = (previous, current,
                          self._label_to_track(int(frame_number) - 1),
                          self._label_to_track(int(frame_number)))
            candidates = manual_correction_tracking.suggest_match_candidates(
                self.cells_info, self.tracking_data, int(frame_number), cell_index,
                lambda f: (self._load_frame(f) or (None, None, None))[0],
                top_k=int(top_k), topology_graphs=graphs,
            )
            return {"candidates": candidates, "topology_available": graphs is not None}
        except (KeyError, ValueError) as e:
            return {"error": str(e)}

    def add_tracking_id(self, frame_number: int, new_index: int, cell_id_str: str) -> dict:
        try:
            self.tracking_data, displaced = manual_correction_tracking.add_tracking_id(
                self.tracking_data, int(frame_number), int(new_index), cell_id_str,
            )
            return {"success": True, "displaced": displaced}
        except KeyError as e:
            return {"error": str(e)}

    def save_tracking(self) -> None:
        path = os.path.join(
            self.output_dirs["cell_track_output"], "all_cell_tracking", "all_cell_merged_tracking_results.xlsx",
        )
        manual_correction_tracking.save_tracking_data(self.tracking_data, path)

    # ---- Trajectory --------------------------------------------------------

    def get_track_numbers(self) -> list[int]:
        if self.tracking_data is None:
            return []
        return sorted(int(v) for v in self.tracking_data.iloc[:, 0].tolist())

    def get_track_frame_range(self, track_number: int) -> Optional[dict]:
        if self.tracking_data is None:
            return None
        cell_ids = trajectory_visualization.get_track_cell_ids(self.tracking_data, int(track_number))
        frames = [
            int(match.group(1))
            for cid in cell_ids if isinstance(cid, str)
            for match in [re.search(r"Cell(\d+)_", cid)] if match
        ]
        if not frames:
            return None
        return {"min_frame": min(frames), "max_frame": max(frames)}

    def get_trajectory_overlay(
        self, track_number: int, frame_number: int, render_mode: str = "mask",
        color_mode: str = "highlight", show_nematic_axes: bool = True,
    ) -> Optional[str]:
        """Trajectory view. `color_mode="highlight"` is the original
        single-track highlight; `"dynamics"` draws the cell-dynamics overlay.

        **Cell dynamics lives here, not in Correct Tracking.** It is a result
        to look at, not a correction tool -- and this step already has frame
        playback and zoom, which is what reading division / delamination / T1
        actually needs.
        """
        if self.tracking_data is None:
            return None
        loaded = self._load_frame(frame_number)
        if loaded is None:
            return None
        fill_labels, boundary_labels, base_image = loaded
        label_array = boundary_labels if render_mode == "outline" else fill_labels

        if color_mode == "dynamics":
            nematic_axes = None
            if show_nematic_axes:
                events = self.get_t1_events(frame_number)
                keep = {"all": ("persistent", "transient", "unjudgeable"),
                        "persistent": ("persistent",),
                        "persistent_unjudgeable": ("persistent", "unjudgeable")}[self.t1_axis_filter]
                nematic_axes = [
                    {**axis, "length": 14,
                     "color": "#66ccff" if axis["type"] == "gain" else "#ff9933"}
                    for axis in events.get("axes", [])
                    if axis.get("state", "persistent") in keep
                ]
            overlay = rendering.render_tracking_overlay(
                label_array, base_image, frame_number, self.tracking_data,
                self.matched_color, self.unmatched_color,
                label_cells=False, track_color_mode="dynamics",
                dynamics_states=self._dynamics_states(frame_number),
                dynamics_colors=self.dynamics_colors, nematic_axes=nematic_axes,
            )
            return rendering.encode_png_base64(overlay)

        cell_ids = trajectory_visualization.get_track_cell_ids(self.tracking_data, int(track_number))
        target_prefix = f"Cell{frame_number}_"
        match = next((cid for cid in cell_ids if isinstance(cid, str) and cid.startswith(target_prefix)), None)
        if match is None:
            return None
        cell_id_num = int(match.split("_")[-1])
        overlay = rendering.render_cell_highlight(label_array, base_image, cell_id_num, self.highlight_color)
        return rendering.encode_png_base64(overlay)

    def get_link_agreement(self, frame_number: int) -> dict:
        """Per-cell agreement between this frame's links and an overlap-based
        reference: `{"states": {label: state}, "counts": {...}}` with states
        `agrees` / `disagrees` / `unjudgeable` / `broken`.

        **This is not an error map, and the wording is load-bearing.** It says
        a link disagrees with each cell's best-IoU successor. That reference is
        itself biased -- it favours costs correlated with overlap and penalises
        motion prediction -- and about a quarter of cells cannot be judged by
        it at all. See `tracking_benchmark.iou_agreement`.
        """
        # Every empty return names its REASON. This mode previously failed
        # silently in three ways -- sitting on the last frame, no npy folder, an
        # unreadable mask -- and each produced `{}`, which the render branch
        # draws as nothing at all. The view then looked identical to the mode
        # being switched off, so "is it broken?" was a hunt instead of a glance.
        if self.tracking_data is None:
            return {"states": {}, "detail": {}, "counts": {}, "reason": "not_tracked",
                    "message": "Track the movie before checking link agreement."}
        frames = self.get_frame_numbers()
        if frames and frame_number >= max(frames):
            return {"states": {}, "detail": {}, "counts": {}, "reason": "last_frame",
                    "message": f"Frame {frame_number} is the last frame - there is no next frame to "
                               f"compare against, so no link can be judged here."}
        masks = {}
        for frame in (frame_number, frame_number + 1):
            loaded = self._load_frame(frame)
            if loaded is None:
                return {"states": {}, "detail": {}, "counts": {}, "reason": "mask_unavailable",
                        "message": f"Could not load the segmentation for frame {frame}. "
                                   f"Check the masks folder is still readable."}
            masks[frame] = loaded[0]
        registration = self._registration()
        entries = tracking_benchmark.classify_links(
            self.tracking_data, masks, frame_number,
            register=bool(registration.get("register")) and self.register_reference,
            drift_by_frame=registration.get("drift_by_frame"),
        )
        counts: dict = {}
        for entry in entries.values():
            counts[entry["state"]] = counts.get(entry["state"], 0) + 1
        return {
            "states": {int(k): v["state"] for k, v in entries.items()},
            "detail": {int(k): v for k, v in entries.items()},
            "counts": counts,
            "registered": bool(registration.get("register")) and self.register_reference,
            "reason": None if entries else "no_cells",
            "message": "" if entries else f"No tracked cells in frame {frame_number}.",
        }

    def _registration(self) -> dict:
        """Cached `tracking_benchmark.registration_decision` for this project.

        Measured once per project rather than assumed per dataset: the rule is
        drift magnitude > 20% of the median cell radius.

        Validated on the three REAL MOVIES in `test_data/` -- `embryos_GBE_GAP43`
        (ratio 0.50, register), `AS_dorsal view_E-cad` (0.099, no) and
        `AS_LE_lateral view_E-cad` (0.092, no). An earlier version of this
        docstring named `pupae` and `cp_dataset` instead; both are separate-image
        collections, so a cross-frame drift over them is meaningless.

        The threshold has never been exercised near 0.2 (observed: 0.50 / 0.099
        / 0.092), so `radius_fraction` is an untested free parameter.
        """
        if self._registration_cache is None:
            masks = {}
            for frame in self.get_frame_numbers():
                loaded = self._load_frame(frame)
                if loaded is not None:
                    masks[frame] = loaded[0]
            if not masks:
                return {"register": False, "drift_by_frame": {}}
            info = self.cells_info.set_index("Cell Index") if self.cells_info is not None else None
            self._registration_cache = tracking_benchmark.registration_decision(masks, info)
        return self._registration_cache

    def get_registration_status(self) -> dict:
        """The measured drift and the resulting decision, for the Track tab.

        Surfaced rather than applied silently: a user seeing "6.97 px drift
        against a 13.0 px cell radius = 0.54, so the reference is registered"
        can judge the basis. A user seeing tracking behave differently cannot.
        """
        decision = self._registration()
        return {
            "register": bool(decision.get("register")),
            "applied": bool(decision.get("register")) and self.register_reference,
            "median_drift_px": round(float(decision.get("median_drift_px", 0.0)), 2),
            "median_cell_radius_px": round(float(decision.get("median_cell_radius_px", 0.0)), 2),
            "ratio": round(float(decision.get("ratio", 0.0)), 3),
            "threshold": float(decision.get("threshold", 0.2)),
            "enabled": bool(self.register_reference),
        }

    def set_register_reference(self, enabled: bool = True) -> dict:
        """Turn the registered reference on or off. The measured decision still
        governs whether registration would apply at all."""
        self.register_reference = bool(enabled)
        return self.get_registration_status()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Standalone smoke-test harness for EpiQuantAPI (no real window)")
    parser.add_argument("--mask-folder", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--classify", action="store_true", help="Run classify + leading-edge + boundary-split tracking")
    parser.add_argument("--shape", action="store_true", help="Also compute shape/tissue descriptors and exercise every feature mode")
    parser.add_argument("--n-clusters", default="2")
    args = parser.parse_args()

    api = EpiQuantAPI()

    print("== set_project ==")
    print(api.set_project(args.mask_folder, args.image_folder, args.output_dir))

    print("\n== extraction ==")
    api._run_extraction(mask_type="auto", background_area_ratio=10.0, exclude_edge_cells=False)
    assert api.cells_info is not None, "extraction failed, see log above"
    print(f"Extracted {len(api.cells_info)} cells, {api.get_frame_count()} frames")

    first_frame = int(re.search(r"Cell(\d+)_", api.cells_info["Cell Index"].iloc[0]).group(1))
    raw_img = api.get_raw_frame_image(first_frame)
    print(f"Raw frame {first_frame} image: {len(raw_img) if raw_img else 0} base64 chars")

    if args.shape:
        print("\n== shape + tissue descriptors ==")
        api._run_shape_extraction(
            include_tissue=True, efd_order=10, n_contour_points=256,
            smoothing_sigma_fraction=0.02, with_coefficients=True,
        )
        assert api.cells_shape is not None, "shape extraction failed, see log above"
        summary = api.get_shape_feature_summary()
        for table, block in summary.items():
            medians = {k: round(v["median"], 3) for k, v in list(block["columns"].items())[:5] if v["median"] is not None}
            print(f"  {table}: {block['cell_count']} cells, medians {medians} ...")

        options = api.get_feature_options()
        ready = {name: spec["ready"] for name, spec in options["families"].items()}
        print(f"  families ready: {ready}")
        not_ready = {n: s["missing"][:3] for n, s in options["families"].items() if not s["ready"]}
        print(f"  missing for not-ready families: {not_ready or 'none'}")

    if args.classify:
        print("\n== classification ==")
        n_clusters_arg = args.n_clusters if args.n_clusters == "auto" else int(args.n_clusters)
        print(api.run_classification(n_clusters=n_clusters_arg, random_state=42))

        if args.shape:
            options = api.get_feature_options()["families"]
            # Families are ADDITIVE: each is stacked on top of basic.
            for combo in ([], ["hull"], ["lobe"], ["tissue"], ["hull", "lobe"], ["hull", "lobe", "tissue"]):
                if any(not options[f]["ready"] for f in combo):
                    print(f"  families {['basic'] + combo}: not ready, skipped")
                    continue
                result = api.run_classification(n_clusters=2, random_state=42, feature_families=["basic"] + combo)
                label = "+".join(["basic"] + combo)
                print(f"  {label:28} scaler={result['standardize_mode']:7} "
                      f"{len(result['features_used']):2d} features, {result['cells_used']:5d} cells -> {result['cluster_sizes']}")
            if options["efd"]["ready"]:
                efd_result = api.run_classification(n_clusters=2, random_state=42, feature_families=["efd"])
                print(f"  {'efd (exclusive)':28} scaler={efd_result['standardize_mode']:7} "
                      f"{len(efd_result['features_used']):2d} features, {efd_result['cells_used']:5d} cells -> {efd_result['cluster_sizes']}")
                assert efd_result["standardize_mode"] == "none", "EFD must never be standardized"
                try:
                    api.run_classification(feature_families=["efd", "hull"], n_clusters=2)
                    raise AssertionError("EFD must refuse to combine with another family")
                except ValueError as err:
                    print(f"  efd exclusivity enforced: {err}")
            per_image = api.run_classification(
                n_clusters=2, random_state=42, feature_families=["basic", "hull"],
                dataset_mode="independent_samples",
            )
            print(f"  dataset_mode=independent_samples -> {per_image['cluster_sizes']}")
            try:
                api.run_classification(feature_families=["nonsense"])
            except ValueError as err:
                print(f"  unknown family rejected: {err}")
            # Re-run the default so later steps see the expected clustering.
            api.run_classification(n_clusters=n_clusters_arg, random_state=42)

        presets = api.get_palette_presets()
        print(f"Palette presets available: {list(presets.keys())}")
        tab10 = presets["tab10"]
        labels = sorted(int(l) for l in api.clustering_data["Cluster_Label"].unique())
        api.set_cluster_colors({lbl: tab10[i % len(tab10)] for i, lbl in enumerate(labels)})
        print("Legend after coloring:", api.get_cluster_legend())

        overlay_b64 = api.get_classification_overlay(first_frame)
        print(f"Classification overlay for frame {first_frame}: {len(overlay_b64) if overlay_b64 else 0} base64 chars")

        first_cell_index = api.clustering_data["Cell_Index"].iloc[0]
        old_label = int(api.clustering_data["Cluster_Label"].iloc[0])
        new_label = labels[(labels.index(old_label) + 1) % len(labels)]
        print(f"\n== manual cluster correction: {first_cell_index} {old_label} -> {new_label} ==")
        print("Legend after reassignment:", api.reassign_cluster(first_cell_index, new_label))
        api.save_classification()
        print("Saved classification.")

        if api.can_compute_leading_edge():
            print("\n== leading edge ==")
            print(api.compute_leading_edge_all_frames())
            api.save_cells_info_leading_edge()
            print("Saved leading_edge column.")
        else:
            print(f"\ncan_compute_leading_edge() is False (k={api.clustering_data['Cluster_Label'].nunique()}) -- skipping.")

    print("\n== tracking ==")
    mode_options = api.get_tracking_mode_options()
    print("Tracking mode options:", mode_options)
    mode = "split" if mode_options["split_available"] else "single"
    api._run_tracking(mode=mode)
    assert api.tracking_data is not None, "tracking failed, see log above"
    print(f"Tracked {len(api.tracking_data)} tracks (mode={mode})")

    track_overlay = api.get_tracking_overlay(first_frame)
    print(f"Tracking overlay for frame {first_frame}: {len(track_overlay) if track_overlay else 0} base64 chars")

    track_numbers = api.get_track_numbers()
    if track_numbers:
        tn = track_numbers[0]
        frame_range = api.get_track_frame_range(tn)
        print(f"Track {tn} frame range: {frame_range}")
        if frame_range:
            traj_overlay = api.get_trajectory_overlay(tn, frame_range["min_frame"])
            print(f"Trajectory overlay for track {tn}, frame {frame_range['min_frame']}: "
                  f"{len(traj_overlay) if traj_overlay else 0} base64 chars")

            print(f"\n== manual tracking correction on track {tn} ==")
            old_id = tn
            other_track = track_numbers[1] if len(track_numbers) > 1 else tn
            print("reassign_track (expect a graceful dict, error or success):",
                  api.reassign_track(frame_range["min_frame"], old_id, other_track))

            print("swap_tracks (expect a graceful dict, error or success):",
                  api.swap_tracks(frame_range["min_frame"], old_id, other_track))

            first_cell_at_min_frame = api.cells_info[
                api.cells_info["Cell Index"].str.startswith(f"Cell{frame_range['min_frame']}_")
            ]["Cell Index"].iloc[0]
            print("suggest_match_candidates (expect a graceful dict):",
                  api.suggest_match_candidates(frame_range["min_frame"], first_cell_at_min_frame))

            print("compute_propagation_stop_frame (expect a graceful dict):",
                  api.compute_propagation_stop_frame(frame_range["min_frame"], old_id, other_track))

            print("compute_confirmed_identity_moves (expect a graceful dict):",
                  api.compute_confirmed_identity_moves(frame_range["min_frame"], old_id))

            # --- coverage log (Round 15 C4): the assertion that matters is
            # that "displayed and left alone" and "never displayed" are
            # different values on a real movie.
            print("\n== coverage log ==")
            frames_present = sorted({int(m.group(1)) for m in
                                     (re.match(r"Cell(\d+)_", str(c))
                                      for c in api.cells_info["Cell Index"]) if m})
            looked_at, never_opened = frames_present[0], frames_present[-1]
            cells_here = [c for c in api.cells_info["Cell Index"]
                          if str(c).startswith(f"Cell{looked_at}_")]
            api.log_coverage_event("frame_shown", looked_at, ms_elapsed=9000,
                                   zoom_percent=200, render_mode="mask", color_mode="identity")
            api.log_coverage_event("cell_inspected", looked_at, cell_index=cells_here[0])
            if len(cells_here) > 1:
                api.log_coverage_event("cell_inspected", looked_at, cell_index=cells_here[1])
                api.log_coverage_event("cell_changed", looked_at, cell_index=cells_here[1],
                                       detail="harness")
            coverage = api.get_coverage_summary()
            seen = coverage["frames"][looked_at]
            unseen = coverage["frames"][never_opened]
            assert seen["displayed"] and not unseen["displayed"], (seen, unseen)
            assert cells_here[0] in seen["inspected_unchanged"], seen
            assert seen["never_displayed"] == [], seen
            assert unseen["never_displayed"] and unseen["inspected_unchanged"] == [], unseen
            print(f"  frame {looked_at}: displayed {seen['dwell_ms']:.0f} ms, "
                  f"{len(seen['changed'])} corrected, "
                  f"{len(seen['inspected_unchanged'])} inspected-and-left, "
                  f"{len(seen['displayed_only'])} not clicked")
            print(f"  frame {never_opened}: never displayed "
                  f"({len(unseen['never_displayed'])} cells) -- distinct from unchanged")
            print(f"  weak positives: {coverage['weak_positives']}  log: {coverage['path']}")

            api.save_tracking()
            print("Saved tracking.")

    print("\nAll smoke-test steps completed without raising.")
