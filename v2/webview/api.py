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

import base64
import csv
import datetime
import json
import logging
import os
import re
import statistics
import sys
import threading
from typing import Optional, Union

import numpy as np
import pandas as pd
import webview

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import adjacency
import baselines
import cell_tracking
import census
import common
import coverage_log
import data_input
import efd
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
        # Round 25 D1: which candidate generator the census offers. Both stay
        # until the maintainer's click-test decides; switchable from the census
        # row, and every verdict records which one it was given against.
        self.candidate_mode: str = self.CANDIDATE_MODE
        self.candidate_radius_k: float = self.CANDIDATE_RADIUS_K
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
        # Round 26 B2. The same checkbox, a different meaning: it no longer
        # drops rows at Extract, it filters the consumers whose quantities a
        # truncated cell corrupts (morphometrics, PCA/K-means). Tracking never
        # filters -- a border cell's centroid and mask are intact, and letting
        # a per-frame bounding-box test decide membership is what minted 136
        # orphans over 85 frame pairs on `AS_LE_lateral view_E-cad`.
        # Persisted with the project and stamped into tracking provenance,
        # because which semantics produced a given table has to be recoverable.
        # The value itself lives in `data_input` (module state, like the
        # minimum cell area) so the consumers and the provenance writers read
        # one answer rather than this object's copy of it; the property below
        # is a view onto it.
        data_input.set_edge_exclusion(False)
        # How many of the least-confident links the `low_margin` set offers.
        # Every assigned link has a margin, so without a cap this set is the
        # whole movie ordered differently -- which is a report, not a queue.
        self.low_margin_queue_limit: int = 200
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
        # The drift_mode that actually produced `self.tracking_data`, recorded
        # when tracking completes. `self.drift_mode` is a live UI setting the
        # user can change afterwards, so reading it to describe an existing
        # tracking run is a claim, not a fact -- and Round 18 §D1 caught it
        # making a false one: a session reported "523 links, reference computed
        # on unregistered masks, matching this tracking", but 523 is the count
        # for REGISTERED tracking judged unregistered (measured 518); the
        # unregistered/unregistered cell is 255. The tracking had drift on and
        # the setting had since been changed back.
        self.tracking_drift_mode: Optional[str] = None
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
        # Which T1 events to DRAW AT ALL: "all" | "persistent" |
        # "persistent_unjudgeable". Gates the cell fill and the nematic axis
        # together -- Round 17 C3 found it gated only the axis, so `settled
        # only` hid 10 axes and left 10 faded olive cells painted underneath
        # them. It was named `t1_axis_filter` then, accurately; the name moved
        # with the behaviour.
        # Three-way rather than a checkbox because `annotate_persistence`
        # produces three states, and a toggle would silently fold "unjudgeable"
        # into one of the other two.
        #
        # Round 16 D: default changed from "all" to "persistent", on the user's
        # report from real use that the faded transient axes interfere with
        # reading the frame -- 21% of raw events on `embryos_GBE_GAP43` reverse
        # within one frame, and at ~1,500 gains per movie that is a lot of
        # flicker drawn over the tissue. This is a RENDERING default only; no
        # count changes and nothing is dropped from the data. `all` stays one
        # click away, and `get_t1_events` now reports how many axes the current
        # filter removes, because a filter whose removals cannot be seen cannot
        # be judged too aggressive (`v2/docs/schmitt-trigger.md` section 6).
        self.t1_event_filter: str = "persistent"
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
        self._extracted_label_cache: Optional[dict] = None
        self._margin_cache: dict = {}

        self.matched_color = "#ffd633"
        self.unmatched_color = "#3388ff"
        # Round 26 C. A cell with no `Cells_info` row is not an unmatched cell:
        # no correction can attach it, because there is nothing to attach. It
        # used to render in `unmatched_color` like a genuine LAP miss, which is
        # what sent a maintainer hunting for a LAP bug whose cause was the edge
        # filter. Magenta: it has to separate from the yellow/blue pair AND from
        # the greyscale membrane image, and it should read as "not a grade" --
        # this is a category, not a worse kind of unmatched.
        self.unextracted_color = "#ff00ff"
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
        fill_labels = data_input.load_label_array(dat)
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
        # Everything cached belongs to the old project's masks.
        self._invalidate_frame_caches(reason="new project")
        self._junction_min_px = None
        self._junction_min_px_auto = None
        self.output_dirs = common.setup_output_dirs(output_base_dir)
        fmt = data_input.detect_mask_format(mask_folder)
        # Restore the Extract settings this project was last used with. The
        # minimum-area threshold is process state -- one source of truth for a
        # running process, which is right -- but a process ends: set it,
        # Extract, close the app, reopen, Track, and every loader silently read
        # the unfiltered masks (Round 21 A2).
        restored = self._load_project_settings()
        return {"format": fmt, "output_dirs": self.output_dirs, "settings": restored}

    @property
    def exclude_edge_cells(self) -> bool:
        return data_input.get_edge_exclusion()

    @exclude_edge_cells.setter
    def exclude_edge_cells(self, enabled: bool) -> None:
        data_input.set_edge_exclusion(enabled)

    def _invalidate_frame_caches(self, frames=None, reason: str = "") -> list:
        """**The one place a mask change is announced.** Every cache derived
        from the segmentation is dropped here, and every path that can change
        what a mask loader returns calls this rather than clearing whatever it
        happens to remember.

        Round 28 is why. `_neighbor_graph_cache` was cleared in exactly two
        places -- changing project, and changing the T1 thresholds -- and was
        documented as needing no invalidation because "the frame's
        `fill_labels` geometry never changes once a project is loaded". A
        segmentation correction breaks that sentence, and nothing noticed: the
        graph stayed keyed to labels the masks no longer used, and the
        environment score built on it disagreed with the same score after a
        restart on all three movies. A per-call-site `clear()` would have fixed
        the one path that was looked at and left the next one to be found the
        same way.

        `frames=None` means "everything". Per-frame caches can be dropped
        frame-by-frame; the ones that POOL across frames (the hysteresis chain,
        the QC display windows fitted from one frame for all, the registration
        decision measured over the whole movie, and the two `cells_info`
        lookups) cannot, so they go entirely whatever `frames` says.
        """
        per_frame = {
            "neighbor_graph": self._neighbor_graph_cache,
            "raw_image": self._raw_image_cache,
            "margin": self._margin_cache,
            "border_cells": self._border_cell_cache,
        }
        dropped = []
        for name, cache in per_frame.items():
            if frames is None:
                if cache:
                    dropped.append(f"{name} (all {len(cache)})")
                cache.clear()
            else:
                hit = [f for f in (int(x) for x in frames) if f in cache]
                for frame in hit:
                    cache.pop(frame, None)
                if hit:
                    dropped.append(f"{name} {hit}")
        if self._hysteresis_cache:
            dropped.append("hysteresis (pools across frames)")
        self._hysteresis_cache.clear()
        self._qc_display_cache.clear()
        self._registration_cache = None
        self._geometry_lookup = None
        self._extracted_label_cache = None
        if dropped:
            logger.info("invalidated mask-derived caches%s: %s",
                        f" ({reason})" if reason else "", "; ".join(dropped))
        return dropped

    def _settings_path(self) -> Optional[str]:
        if not self.output_dirs:
            return None
        return os.path.join(self.output_dirs["cell_classification_output"],
                            "extract_settings.json")

    def _load_project_settings(self) -> dict:
        """Restore what a previous session set for this project, and apply it."""
        path = self._settings_path()
        if not path or not os.path.exists(path):
            data_input.set_min_cell_area(0)
            return {"min_cell_area_px": 0, "restored": False,
                    "exclude_edge_cells": self.exclude_edge_cells}
        try:
            with open(path, encoding="utf-8") as handle:
                stored = json.load(handle)
        except (OSError, ValueError) as error:
            logger.warning("could not read %s: %s", path, error)
            return {"min_cell_area_px": data_input.get_min_cell_area(), "restored": False,
                    "exclude_edge_cells": self.exclude_edge_cells}
        area = int(stored.get("min_cell_area_px", 0) or 0)
        data_input.set_min_cell_area(area)
        self.exclude_edge_cells = bool(stored.get("exclude_edge_cells", False))
        return {"min_cell_area_px": area, "restored": True,
                "exclude_edge_cells": self.exclude_edge_cells}

    def _save_project_settings(self) -> None:
        path = self._settings_path()
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"min_cell_area_px": data_input.get_min_cell_area(),
                           "exclude_edge_cells": bool(self.exclude_edge_cells)},
                          handle, indent=2, sort_keys=True)
                handle.write("\n")
        except OSError as error:
            logger.warning("could not write %s: %s", path, error)

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
        # On a TIFF project this call just reconverted `npy_converted/` from
        # the TIFFs, so the masks may differ from the ones anything was cached
        # from.
        self._invalidate_frame_caches(reason="loaded existing extraction")
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

    def set_min_cell_area(self, area_px: int = 0) -> dict:
        """Minimum label area, in pixels, applied by every mask loader.

        **Default 0 (off)**, so existing baselines stay byte-reproducible. It
        exists because Tissue Analyzer's segmentation regularly leaves a 1 px
        gap at four-cell vertices and Epi-Quant reads each one as a cell -- and
        such a label is not merely a spurious row: it becomes a fifth neighbour
        in the adjacency graph and can stop the four real cells around that
        vertex being recorded as neighbours of one another, which corrupts T1
        detection precisely where T1s occur.

        Set on `data_input` rather than carried here, because the property that
        matters is that *every* consumer sees the same labels.
        """
        data_input.set_min_cell_area(area_px)
        # Not a mask edit, but it changes the label set every loader returns,
        # which is the same thing as far as anything cached from one is
        # concerned.
        self._invalidate_frame_caches(reason="minimum cell area changed")
        self._save_project_settings()
        return {"min_cell_area_px": data_input.get_min_cell_area(),
                "suggested": data_input.SUGGESTED_MIN_CELL_AREA_PX}

    def get_min_cell_area(self) -> dict:
        return {"min_cell_area_px": data_input.get_min_cell_area(),
                "suggested": data_input.SUGGESTED_MIN_CELL_AREA_PX}

    def get_min_area_removals(self) -> dict:
        """Which labels the current minimum-area setting removes, per frame.

        A filter whose removals cannot be seen cannot be judged too aggressive
        -- the same rule that governs the T1 event filter and the junction
        length minimum. This is its third application.

        `unreadable_frames` names the frames whose `.npy` holds no filled label
        array, so the count below them is understood as "not measured" rather
        than "nothing removed". Until Round 22 this read `raw.get("masks")` and
        `continue`d past such a frame in silence, so the UI's removal count came
        out low with nothing saying which frames it had skipped.
        """
        threshold = data_input.get_min_cell_area()
        if not threshold or not self.npy_folder:
            return {"min_cell_area_px": threshold, "total": 0, "by_frame": {},
                    "unreadable_frames": []}
        by_frame, total, unreadable = {}, 0, []
        files = data_input.list_segmentation_files(self.npy_folder)
        for name, number in data_input.resolve_frame_numbers(files).items():
            raw = np.load(os.path.join(self.npy_folder, name), allow_pickle=True).item()
            labels, removed = data_input.load_label_array(
                raw, threshold, with_removed=True)
            if labels is None:
                unreadable.append(int(number))
                continue
            if removed:
                by_frame[int(number)] = len(removed)
                total += len(removed)
        return {"min_cell_area_px": threshold, "total": total, "by_frame": by_frame,
                "unreadable_frames": sorted(unreadable)}

    def _run_extraction(self, mask_type: str, background_area_ratio: float, exclude_edge_cells: bool) -> None:
        try:
            # Kept as a project setting rather than consumed here: since Round
            # 26 extraction writes every row and an `Edge Distance`, and this
            # flag decides whether the ANALYSIS steps filter on it.
            self.exclude_edge_cells = bool(exclude_edge_cells)
            self._save_project_settings()
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
            # Extraction has just read the masks off disk; anything cached from
            # a previous read of them may predate a correction made since.
            self._invalidate_frame_caches(reason="extraction")
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

    def _extracted_labels(self, frame_number: int) -> Optional[set]:
        """Raw labels that have a `Cells_info` row in `frame_number`.

        `None` when there is no table at all, which `render_tracking_overlay`
        reads as "cannot tell the two kinds of unmatched apart" and renders the
        way it always did -- rather than claiming every cell is unextracted.
        Cached alongside the geometry lookup and cleared with it.
        """
        if self.cells_info is None:
            return None
        if self._extracted_label_cache is None:
            self._extracted_label_cache = {}
            for index in self.cells_info["Cell Index"].astype(str):
                match = re.match(r"Cell(\d+)_(\d+)$", index)
                if match:
                    self._extracted_label_cache.setdefault(
                        int(match.group(1)), set()).add(int(match.group(2)))
        return self._extracted_label_cache.get(int(frame_number), set())

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
            # How much the filter removes from what is drawn -- fill and axis
            # alike. Reported alongside the RAW counts so turning it on cannot
            # hide its own effect (Round 16 D, corrected in Round 17 C3).
            "event_counts": self._event_filter_counts(axes),
            # Provenance travels with every count. `25 / 15` is a statement
            # about one objective and one binning: on a dataset at half the
            # pixel size the same physical junction has roughly twice the
            # contact strength, so the same numbers are a looser filter. The
            # whole repo has no `pixel_size` / `um_per_px` anywhere, so without
            # this stamp no T1 count in this project is reproducible outside
            # the session that produced it.
            "parameters": self.t1_parameters(),
        }

    # The three filter modes, as the set of states each keeps. ONE definition,
    # read by the axis path, the cell-fill path and the counts alike, so the
    # readout can never describe a different filter than the one that drew the
    # frame -- which is exactly what C3 was: two code paths, one filter.
    _EVENT_FILTER_STATES = {
        "all": ("persistent", "transient", "unjudgeable"),
        "persistent": ("persistent",),
        "persistent_unjudgeable": ("persistent", "unjudgeable"),
    }

    def _event_filter_counts(self, axes: list) -> dict:
        """How many T1 events the current filter draws, removes, and cannot
        judge. Counts EVENTS, not axes: the filter now hides the cell fill as
        well, so a removed event leaves nothing on the frame at all."""
        keep = self._EVENT_FILTER_STATES[self.t1_event_filter]
        states = [axis.get("state", "persistent") for axis in axes]
        return {
            "filter": self.t1_event_filter,
            "shown": sum(1 for state in states if state in keep),
            "filtered": sum(1 for state in states if state not in keep),
            "unjudgeable": sum(1 for state in states if state == "unjudgeable"),
            "total": len(states),
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
            "event_filter": self.t1_event_filter,
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
            "event_filter": self.t1_event_filter,
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

    def set_t1_event_filter(self, mode: str = "all") -> dict:
        """Which T1 orientation axes to draw: all / persistent /
        persistent_unjudgeable."""
        if mode not in self._EVENT_FILTER_STATES:
            raise ValueError(f"Unknown axis filter {mode!r}")
        self.t1_event_filter = mode
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
                unavailable = []
            else:
                columns = list(spec["columns"])
                default_off = set(spec.get("default_off", ()))
                # A default-off column absent from the table does not block the
                # family: the default run never reads it. This is a project
                # extracted before Round 24 restored `Fitted Ellipse Angle` and
                # `Circularity` -- its basic family still runs exactly as it
                # did, and the two boxes are shown disabled rather than hidden.
                missing = [c for c in columns if c not in available and c not in default_off]
                unavailable = [c for c in columns if c not in available and c in default_off]
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
                "unavailable": unavailable,
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
        if self.exclude_edge_cells:
            # PCA/K-means run on geometry, and a cell the field of view cut in
            # half has the wrong geometry -- so this is exactly the consumer the
            # exclusion exists for. Applied here rather than at extraction so
            # the same table can be tracked unfiltered (Round 26 B2).
            before = len(joined)
            joined = data_input.drop_edge_cells(joined)
            logger.info("exclude_edge_cells: %d of %d cell(s) are within %d px of the border "
                        "and are left out of the feature matrix",
                        before - len(joined), before, data_input.DEFAULT_EDGE_MARGIN_PX)

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
            efd_order = sum(str(c).startswith("efd_") for c in joined.columns) // 4
            columns = pca_kmeans.efd_preset(efd_order)
            missing = [c for c in columns if c not in joined.columns]
            if missing or efd_order == 0:
                raise ValueError(
                    f"The elliptic-Fourier table is incomplete ({len(missing)} of {len(columns)} "
                    "coefficient columns missing). Re-run shape extraction."
                )
            standardize_mode = pca_kmeans.FEATURE_FAMILIES["efd"]["standardize"]
        else:
            columns = list(selected_features) if selected_features else pca_kmeans.family_columns(families)
            missing = [c for c in columns if c not in joined.columns]
            if missing:
                raise ValueError(
                    f"{len(missing)} feature(s) are not available in this project: {', '.join(missing[:6])}"
                    + (" ..." if len(missing) > 6 else "")
                    + (". Re-run Extract - this project's Cells_info.xlsx predates them."
                       if all(c in data_input.CELLS_INFO_COLUMNS for c in missing)
                       else ". Run the shape-descriptor step in Extract first.")
                )
            standardize_mode = "global"

        frame = joined.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        # `extract_cell_info` records a failed measurement as the sentinel -1
        # (a collapsed ellipse fit, a zero perimeter, a one-row cell's H/V
        # ratio), not as NaN, so `dropna` alone let those rows straight into
        # StandardScaler and PCA -- where a -1 in a column whose values run
        # 20-200 px is a many-SD outlier that owns the standardised range.
        # `segmentation_qc.measurement_failures` already excluded them on its
        # side; Classify did not. Scoped to the `Cells_info` columns because -1
        # is THAT table's sentinel convention: the shape, tissue and EFD tables
        # record failure as NaN, and a raw EFD coefficient is a signed float that
        # can legitimately be negative. Every `Cells_info` feature is a length,
        # area, radius, count, positive ratio or an angle folded into 0-90, so
        # -1 there is never a value.
        sentinel_columns = [c for c in columns if c in data_input.CELLS_INFO_COLUMNS]
        if sentinel_columns:
            frame[sentinel_columns] = frame[sentinel_columns].mask(frame[sentinel_columns] == -1)
        if "efd" in families:
            # The table holds the RAW coefficients, which change with the cell's
            # size, its orientation in the image, the traversal direction and
            # wherever the tracer began the contour. PCA must see the
            # normalized shape (`efd.efd_features`), never the raw columns:
            # on the WT/tmc/pyd pickles, raw-coefficient PC1-3 were mixtures of
            # size (|r| = 0.91 with PC2) and contour start phase, not shape.
            complete = frame.dropna()
            normalized = efd.efd_features(complete.to_numpy(dtype=float).reshape(-1, efd_order, 4))
            frame = pd.DataFrame(
                np.atleast_2d(normalized), index=complete.index,
                columns=efd.efd_feature_names(efd_order),
            ).reindex(frame.index)
        before = len(frame)
        frame = frame.dropna()
        if frame.empty:
            raise ValueError("No cells have a complete set of the selected features.")
        if len(frame) < before:
            logger.info(
                "Dropped %d cell(s) with missing or sentinel (-1) values from classification",
                before - len(frame),
            )
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
            self.tracking_drift_mode = self.drift_mode
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

    # --- Re-extract the range a re-track is about to re-track (Round 19 C3) --
    #
    # `re-track 的前提应该是 re-extract edited cell features`. Round 13 made
    # re-track REFUSE when `Cells_info` was stale, which stops the corruption
    # but leaves the user to fix it by hand and leaves a window in which the two
    # can diverge again. Closing it: re-tracking a range re-extracts that same
    # range first, so the geometry the tracker reads is by construction the
    # geometry of the masks on disk.

    #: Sidecar file -> (attribute, extractor). Which families to recompute is
    #: decided by WHICH FILES EXIST, not by inspecting columns -- the
    #: descriptors are separate files, not extra columns in `Cells_info.xlsx`.
    FEATURE_SIDECARS = ("Cells_shape.xlsx", "Cells_tissue.xlsx")

    def _computed_feature_families(self) -> list:
        """Which optional feature families this project has actually computed."""
        folder = self.output_dirs["cell_classification_output"] if self.output_dirs else None
        if not folder:
            return []
        return [name for name in self.FEATURE_SIDECARS
                if os.path.exists(os.path.join(folder, name))]

    def _reextract_frames(self, frames) -> dict:
        """Recompute every per-cell family for `frames`, splicing the new rows
        into the existing tables. Returns a report naming what was recomputed
        and what is now out of date.

        All three families are **per-cell**: each value depends only on that
        cell's mask and its own frame's neighbours, so recomputing a frame range
        is exact rather than an approximation.
        """
        frames = sorted({int(f) for f in frames})
        folder = self.output_dirs["cell_classification_output"]
        report = {"frames": frames, "recomputed": [], "stale_after": []}

        def with_index_column(table):
            """These tables come back with `Cell Index` sometimes as a column and
            sometimes as the index, depending on which loader produced them."""
            if table is None or "Cell Index" in table.columns:
                return table
            return table.reset_index().rename(columns={table.index.name or "index": "Cell Index"})

        def splice(existing, fresh):
            """Replace the rows for `frames`, keep every other row untouched."""
            fresh = with_index_column(fresh)
            existing = with_index_column(existing)
            if existing is None:
                return fresh
            frame_of = existing["Cell Index"].astype(str).str.extract(r"Cell(\d+)_", expand=False)
            keep = ~frame_of.astype(float).isin(frames)
            return pd.concat([existing[keep], fresh], ignore_index=True)

        fresh_info, _ = data_input.extract_cell_info(
            self.npy_folder, output_path=None, exclude_edge_cells=False, frames=set(frames))
        self.cells_info = splice(self.cells_info, fresh_info)
        self.cells_info.to_excel(os.path.join(folder, "Cells_info.xlsx"),
                                 sheet_name="Cell", index=False)
        # The masks for these frames have just been re-read, which is the whole
        # reason this function exists -- so everything cached from the old ones
        # goes. This is the path a re-track takes (Round 28 C1).
        report["caches_dropped"] = self._invalidate_frame_caches(
            frames=frames, reason=f"re-extracted frames {frames}")
        report["recomputed"].append("geometry")

        present = self._computed_feature_families()
        if "Cells_shape.xlsx" in present:
            fresh_shape = shape_features.extract_shape_features(
                self.npy_folder, output_path=None, frames=set(frames))
            self.cells_shape = splice(self.cells_shape, fresh_shape)
            self.cells_shape.to_excel(os.path.join(folder, "Cells_shape.xlsx"), index=False)
            report["recomputed"].append("shape")
        if "Cells_tissue.xlsx" in present:
            fresh_tissue = tissue_features.extract_tissue_features(
                self.npy_folder, output_path=None, frames=set(frames))
            self.cells_tissue = splice(self.cells_tissue, fresh_tissue)
            self.cells_tissue.to_excel(os.path.join(folder, "Cells_tissue.xlsx"), index=False)
            report["recomputed"].append("tissue")

        # C3a: the dataset-fitted tier CANNOT follow. The PCA basis is fit over
        # every cell in the movie and k-means over that basis, so recomputing a
        # few rows would score them against a model fit on the OLD feature
        # distribution -- and refitting moves labels EVERYWHERE, not only in the
        # range the user asked about. There is no correct local recomputation,
        # so these are reported out of date and left alone. Refitting is the
        # user's decision because its blast radius is the whole movie.
        for name in ("Cells_info_PCA.xlsx", "cells_clustering_results.xlsx"):
            if os.path.exists(os.path.join(folder, name)):
                report["stale_after"].append(name)
        if report["stale_after"]:
            report["note"] = (
                "Frames %s re-extracted (%s). PCA and cluster labels were fit on the previous "
                "features and are now out of date for those frames -- re-run Classify to refit. "
                "Refitting changes labels outside this range too, which is why it is not automatic."
                % (", ".join(str(f) for f in frames), " + ".join(report["recomputed"]))
            )
        else:
            report["note"] = ("Frames %s re-extracted (%s)."
                              % (", ".join(str(f) for f in frames), " + ".join(report["recomputed"])))
        return report

    def _run_local_retrack(self, start_frame: int, end_frame: int) -> None:
        try:
            if self.cells_info is None or self.tracking_data is None:
                raise ValueError("Extract and track the movie before re-tracking a range.")
            cb = self._progress_cb("retrack")

            # Round 13 §A5 REFUSED here when `Cells_info` was stale. That
            # stopped the corruption but left the user to fix it by hand, and
            # left a window in which the two could diverge again. Round 19 C3
            # closes it: re-track re-extracts the range it is about to
            # re-track, including the seam frames the cost actually reads, so
            # the geometry the tracker sees is by construction the geometry of
            # the masks on disk.
            span = set(range(start_frame - 1, end_frame + 2)) & set(self.get_frame_numbers())
            cb(f"Re-extracting frames {min(span)}-{max(span)}...", 5)
            extraction = self._reextract_frames(span)

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
            self._notify("onRetrackDone", {**result, "track_count": int(len(self.tracking_data)),
                                           "extraction": extraction})
        except Exception as e:
            logger.exception("Local re-track failed")
            self._notify("onRetrackError", {"message": str(e)})

    def set_track_colors(self, matched_color: str, unmatched_color: str,
                         unextracted_color: str = None) -> None:
        self.matched_color = matched_color
        self.unmatched_color = unmatched_color
        if unextracted_color:
            self.unextracted_color = unextracted_color

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

        # A T1 needs no human confirmation, but it does need to have settled --
        # and `t1_event_filter` decides whether an unsettled one is DRAWN AT
        # ALL, not merely dimmed (Round 17 C3).
        #
        # It used to be applied only to the nematic axes, in
        # `get_trajectory_overlay`, so `settled only` hid ten axes and left ten
        # faded olive cells painted under them: the filter and the fill were
        # two code paths and only one of them had heard of it. Applying it here
        # means every consumer of `_dynamics_states` gets the same answer.
        #
        # **Deliberately T1-only.** Divisions and delaminations run through the
        # same `confirmed -> dim` rendering path, but the decision is not the
        # same one: an unconfirmed division CANDIDATE has to stay visible or it
        # can never be confirmed, whereas an unsettled T1 is noise the user
        # asked to stop seeing. Same mechanism, different answers, on purpose.
        t1 = found.get("t1", {})
        settled_labels = {int(label) for label in t1.get("counts_persistent", {})}
        unjudgeable_labels = {int(label) for label in t1.get("counts_unjudgeable", {})}
        keep = self._EVENT_FILTER_STATES[self.t1_event_filter]
        for entry in t1.get("counts", {}):
            label = int(entry)
            if label in settled_labels:
                state = "persistent"
            elif label in unjudgeable_labels:
                state = "unjudgeable"
            else:
                state = "transient"
            if state not in keep:
                continue
            states[label] = {"event": "t1", "confirmed": state == "persistent",
                             "t1_state": state}
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
        scores, unscoreable = self._environment_scores(frame_number, direction, True)
        if not scores:
            return {"candidates": [], "threshold": float(self.environment_repair_threshold),
                    "unscoreable": len(unscoreable)}
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
            # Cells that could not be scored at all. Omitted from `scores`
            # rather than scored 0, so without this they are invisible and
            # indistinguishable from cells that simply scored well -- the same
            # "two different kinds of nothing, drawn the same" as Round 26 C.
            "unscoreable": len(unscoreable),
            "perfect": int(sum(1 for v in scores.values() if v >= 1.0)),
        }

    # --- Docked frame stepper (Round 16 E) -----------------------------------
    #
    # After clicking a cell to edit it, the main view could only render that
    # frame: stepping forward or back lost the selection. This panel steps
    # independently, so the selection is never touched.

    def get_track_filmstrip(self, track_id: int) -> dict:
        """Which frames the track actually holds a cell in, so gaps are visible
        at a glance rather than discovered by stepping into them."""
        if self.tracking_data is None:
            return {"error": "No tracking data loaded."}
        id_col = self.tracking_data.columns[0]
        rows = self.tracking_data.index[self.tracking_data[id_col] == int(track_id)]
        if not len(rows):
            return {"error": f"Track {track_id} not found."}
        row = rows[0]
        frames = self.get_frame_numbers()
        held = [f for f in frames
                if f"Frame{f}" in self.tracking_data.columns
                and pd.notna(self.tracking_data.at[row, f"Frame{f}"])]
        return {"track_id": int(track_id), "frames": frames, "held": held}

    def _anchor_window(self, track_id: int, anchor_frame: int, pad: int):
        """The one window shared by `anchor_frame` and its two neighbours."""
        id_col = self.tracking_data.columns[0]
        rows = self.tracking_data.index[self.tracking_data[id_col] == int(track_id)]
        if not len(rows):
            return None
        row = rows[0]
        boxes = []
        for number in (anchor_frame - 1, anchor_frame, anchor_frame + 1):
            column = f"Frame{number}"
            if column not in self.tracking_data.columns:
                continue
            value = self.tracking_data.at[row, column]
            if pd.isna(value):
                continue
            loaded = self._load_frame(int(number))
            if loaded is None or loaded[0] is None:
                continue
            box = self._cell_window(loaded[0], int(str(value).rsplit("_", 1)[1]), pad=pad)
            if box is not None:
                boxes.append(box)
        if not boxes:
            return None
        return (min(b[0] for b in boxes), max(b[1] for b in boxes),
                min(b[2] for b in boxes), max(b[3] for b in boxes))

    def get_track_step_frame(self, track_id: int, anchor_frame: int, frame: int,
                             show_labels: bool = False, show_outline: bool = False,
                             label_size: int = 10, pad: int = 28,
                             outline_color: str = "#ff0000") -> dict:
        """One crop of `track_id` at `frame`, anchored at `anchor_frame`.

        **The window rule is explicit, and the answer says which one is in
        force** (Round 16 E2), because the two are answering different
        questions and silently picking one would make the panel lie:

        - **within one frame of the anchor**: the SHARED window, unioned over
          the anchor cell and this frame's cell, so the immediate before/after
          is a true blink -- the cell moves and the background does not;
        - **beyond that**: re-centred on this frame's own cell, because under
          `embryos_GBE_GAP43`'s ~6.5 px/frame drift a fixed window loses the
          cell entirely within a few steps. `recentred` is True and the caller
          must show it.

        Returns `{"image", "recentred", "cell_index", "held", ...}`. `held` is
        False when the track has no cell at that frame -- the panel still
        renders the neighbourhood so the gap can be seen, rather than going
        blank.
        """
        if self.tracking_data is None:
            return {"error": "No tracking data loaded."}
        id_col = self.tracking_data.columns[0]
        rows = self.tracking_data.index[self.tracking_data[id_col] == int(track_id)]
        if not len(rows):
            return {"error": f"Track {track_id} not found."}
        row = rows[0]

        def cell_at(number):
            column = f"Frame{number}"
            if column not in self.tracking_data.columns:
                return None
            value = self.tracking_data.at[row, column]
            return None if pd.isna(value) else str(value)

        loaded = self._load_frame(int(frame))
        if loaded is None or loaded[0] is None:
            return {"error": f"No segmentation for frame {frame}."}
        labels = loaded[0]

        cell_index = cell_at(int(frame))
        label = None if cell_index is None else int(str(cell_index).rsplit("_", 1)[1])
        anchor_cell = cell_at(int(anchor_frame))
        anchor_label = None if anchor_cell is None else int(str(anchor_cell).rsplit("_", 1)[1])

        recentred = abs(int(frame) - int(anchor_frame)) > 1
        window = None
        if not recentred and anchor_label is not None:
            # ONE window for the whole +/-1 neighbourhood, unioned over the
            # anchor and BOTH its neighbours -- not over the anchor and
            # whichever frame is being asked for. Per-frame unions differ from
            # each other (measured on `AS_dorsal view_E-cad` track 1: 147x85,
            # 130x77, 164x100 for three supposedly shared frames), so stepping
            # -1 -> 0 -> +1 moved the background and the blink stopped being a
            # blink. All three must be byte-identical in size and origin.
            window = self._anchor_window(int(track_id), int(anchor_frame), int(pad))
        if window is None and label is not None:
            window = self._cell_window(labels, label, pad=int(pad))
            recentred = recentred or int(frame) != int(anchor_frame)
        if window is None:
            # No cell here and no anchor window: fall back to the anchor's own
            # box so the panel shows WHERE the track was, not a blank.
            anchor_loaded = self._load_frame(int(anchor_frame))
            if anchor_loaded is None or anchor_label is None:
                return {"error": f"Track {track_id} has no cell at frame {anchor_frame} to anchor on."}
            window = self._cell_window(anchor_loaded[0], anchor_label, pad=int(pad))
            recentred = False
        if window is None:
            return {"error": f"Could not place a crop window for track {track_id}."}

        return {
            "track_id": int(track_id), "frame": int(frame), "anchor_frame": int(anchor_frame),
            "cell_index": cell_index, "held": cell_index is not None,
            "recentred": bool(recentred),
            "window_note": ("re-centred on this frame -- background moves, so this shows where the "
                            "track goes, not whether it swapped"
                            if recentred else
                            "shared window with the anchor frame -- background is fixed, so this is "
                            "a true blink"),
            "image": self._cell_crop(int(frame), labels, label, window, "#00ff88",
                                     show_labels=bool(show_labels),
                                     show_outline=bool(show_outline),
                                     label_size=int(label_size),
                                     outline_color=outline_color or "#ff0000"),
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
                     seed: int = 0, labeller: str = "", mode: str = "census") -> dict:
        """Build the queue. `kind` is `geometry_flagged` (the whole movie's
        disagreements) or `frame_pair` (every link in one pair).

        `mode` picks what the session is FOR, and the two are mutually
        exclusive by construction (Round 16 C2):

        | | `census` | `correct` |
        |---|---|---|
        | writes | verdicts to `ground_truth_links.csv` | pending edits, one batch |
        | mutates `tracking_data` | **never** | on Apply all |
        | decoys / silent repeats | on | **off** |

        **Ground truth collected against a tracking run is not valid for a run
        it corrected**, so a session writes verdicts or edits, never both. The
        mode is stamped into the provenance, and the two write paths refuse
        each other rather than relying on the UI to keep them apart.

        Decoys are forced off in correction mode: a decoy displays a
        deliberately wrong link, and accepting one would inject a deliberate
        error into the tracking table.

        The judge is registered whenever the tracking was, and the answer says
        so. An overlap reference computed on unregistered masks scoring a
        registered tracker is the preprocessing mismatch that has already moved
        one figure from 4.9% to 14.0% on nothing but that switch.
        """
        if self.tracking_data is None:
            return {"error": "Track the movie before starting a census."}
        if mode not in ("census", "correct"):
            return {"error": f"Unknown census mode {mode!r}; expected 'census' or 'correct'."}
        if mode == "correct":
            decoy_rate = repeat_rate = 0.0
        masks = self._census_masks()
        if len(masks) < 2:
            return {"error": "Need at least two frames with masks."}
        # The convention of the RUN, not of the current setting. Falling back to
        # the setting only when the run's is unknown (a table loaded from disk),
        # and the answer says which it was.
        run_mode = self.tracking_drift_mode if self.tracking_drift_mode is not None else self.drift_mode
        register = run_mode != "none"
        drift = self._registration().get("drift_by_frame") if register else None
        measured = self._registration()
        if kind == "frame_pair":
            if frame is None:
                return {"error": "A frame_pair census needs a frame number."}
            items = census.frame_pair_items(self.tracking_data, masks, int(frame))
        elif kind == "geometry_flagged":
            items = census.geometry_flagged_items(
                self.tracking_data, masks, register=register, drift_by_frame=drift)
        elif kind == "environment_flagged":
            # The repair tier (0.5), not the look tier (0.75): the look tier
            # flags several times as many cells, which is the wrong size for a
            # census. Users correct what the visualisation flags, and this is
            # one of the two overlays they actually read.
            scores = {}
            for frame in sorted(masks):
                if frame + 1 not in masks:
                    continue
                per_frame = self._environment_scores(frame, "next")
                if per_frame:
                    scores[frame] = per_frame
            items = census.environment_flagged_items(
                self.tracking_data, scores, masks,
                threshold=float(self.environment_repair_threshold))
        elif kind in ("forward_backward", "low_margin"):
            # Both re-solve the frame pairs, so both need the feature table and
            # the SAME cost the tracking ran under -- a queue built with a
            # different cost would be a queue for a different tracker. The
            # convention follows the run, like every other set here.
            if self.cells_info is None:
                return {"error": "Extract the movie before building this set."}
            info = self.cells_info.set_index("Cell Index")
            cost = {"overlap_weight": self.overlap_weight,
                    "centroid_weight": self.centroid_weight,
                    "drift_mode": run_mode,
                    "drift_by_frame": drift}
            if kind == "forward_backward":
                items = census.forward_backward_items(self.tracking_data, info, masks, **cost)
            else:
                items = census.low_margin_items(self.tracking_data, info, masks,
                                                limit=self.low_margin_queue_limit, **cost)
        else:
            return {"error": f"Unknown census kind {kind!r}."}
        if not items:
            return {"error": "Nothing to label: that set is empty."}

        queue = census.build_queue(items, masks, decoy_rate=float(decoy_rate),
                                   repeat_rate=float(repeat_rate), seed=int(seed))
        self._census = {
            "queue": queue, "masks": masks, "kind": kind, "frame": frame,
            "register": register, "labeller": labeller, "mode": mode,
            # Kept so `get_census_item` can drift-correct its candidate search
            # without re-deriving the registration for every item.
            "drift_by_frame": drift,
            "session_id": coverage_log.new_session_id(),
            "path": os.path.join(self.output_dirs["cell_track_output"], "ground_truth_links.csv"),
            "edits": {},      # position -> {"chosen_label": int|None}
            "undo": None,     # the whole-table snapshot Apply-all can restore
            # position -> [{"label", "reason", "rank"}], filled in by
            # `get_census_item`, so a verdict can say WHICH ranking first
            # proposed the cell the human picked without the frontend having to
            # send it back (Round 23 D).
            "candidates": {},
            # The MEASURED per-pair drift and median cell radius, kept whatever the
            # judge convention. The radius search's centre is a question about
            # where the cell physically went, not about how the judge was run,
            # so it is registered even when `register` is False (Round 25 D2).
            "search_drift_by_frame": measured.get("drift_by_frame") or {},
            "cell_radius_px": float(measured.get("median_cell_radius_px") or 0.0),
            "windows": {},
            "offered_mode": {},
            "environment_threshold": (float(self.environment_repair_threshold)
                                      if kind == "environment_flagged" else None),
        }
        self._write_census_provenance()
        return {"n": len(queue), "n_real": len(items),
                "n_decoys": sum(1 for e in queue if e["is_decoy"]),
                "n_repeats": sum(1 for e in queue if e["is_repeat"]),
                "kind": kind, "mode": mode, "register": register,
                # What the queue size MEANS. On GBE, measured on matching
                # inputs: tracking and judge both unregistered gives 255 items
                # (3.53%), both registered gives 25 (0.35%) -- the same movie,
                # 10x apart, because without registration the centroid term and
                # the greedy overlap reference respond to 6.98 px/frame drift
                # differently. The queue size is a diagnostic, not a UI problem:
                # never capped, never filtered.
                "tracking_drift_mode": run_mode,
                "convention_from_run": self.tracking_drift_mode is not None,
                "current_drift_setting": self.drift_mode,
                "setting_changed_since_tracking": (self.tracking_drift_mode is not None
                                                   and self.tracking_drift_mode != self.drift_mode),
                "session_id": self._census["session_id"],
                "path": self._census["path"] if mode == "census" else None}

    def _write_census_provenance(self) -> None:
        """Ground truth collected against one tracking run is not automatically
        valid for another -- the link set itself changes. Same treatment the
        tracking baselines get."""
        state = self._census
        record = {
            "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "session_id": state["session_id"],
            "kind": state["kind"],
            "mode": state["mode"],
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
                # Which row set the tracking saw (Round 26 B2): since the
                # border exclusion moved off the extraction path, a table
                # produced under the old semantics has fewer cells per frame
                # than one produced under the new, and nothing else records it.
                "exclude_edge_cells": self.exclude_edge_cells,
                "edge_margin_px": data_input.DEFAULT_EDGE_MARGIN_PX,
            },
        }
        try:
            with open(state["path"] + ".provenance.json", "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, sort_keys=True)
                handle.write("\n")
        except OSError as error:
            logger.warning("could not write census provenance: %s", error)

    def get_census_item(self, position: int, pad: int = 28, anchor_view: bool = True) -> dict:
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

        drift = (0.0, 0.0)
        if state["register"]:
            drift = ((state.get("drift_by_frame") or {}).get(frame)
                     or tracking_benchmark.estimate_frame_drift(masks, frame) or (0.0, 0.0))
        search_drift = self._census_search_drift(state, frame)
        radius_px = self._census_radius_px(state)
        if self.candidate_mode == "radius":
            candidates = self._census_radius_candidates(
                item, before_labels, after_labels, search_drift, radius_px)
            depth = {"radius": len(candidates)}
            mode_tag = f"radius:k={self.candidate_radius_k:g}"
        else:
            # Capped at 12 (four pages of three). Beyond that the list is not a
            # shortlist any more (Round 20 C2).
            candidates = self._census_candidates(item, before_labels, after_labels, window, drift,
                                                 limit=12, render=False)
            depth = getattr(self, "_last_candidate_depth", {})
            mode_tag = "ranked"
        # ONE window for the blink and every candidate tile, centred on the
        # registered anchor and at least 2R + 2*pad across (Round 25 B2).
        window = self._census_anchor_window(before_labels, after_labels, item, candidates,
                                            search_drift, radius_px, pad)
        for candidate, image in zip(candidates, self._census_tile_images(
                frame, before_labels, after_labels, item, candidates, window,
                search_drift, anchor_view)):
            candidate["image"] = image
        state["candidates"][int(position)] = [
            {"label": c["label"], "reason": c["reason"], "rank": c["rank"]}
            for c in candidates]
        state.setdefault("windows", {})[int(position)] = window
        state.setdefault("offered_mode", {})[int(position)] = mode_tag
        top, bottom, left, right = window
        return {
            # How far down each source's own ranking the merge actually got.
            # Not the same thing as `len(candidates)` -- Round 22 reported the
            # latter and believed it was the former (Round 23 A). In radius mode
            # there is one source and its depth is the whole list.
            "candidate_depth": depth,
            "candidate_mode": mode_tag,
            "anchor_view": bool(anchor_view),
            "candidate_search": {
                "mode": self.candidate_mode, "k": float(self.candidate_radius_k),
                "radius_px": round(float(radius_px), 1),
                "cell_radius_px": round(float(state.get("cell_radius_px") or 0.0), 1),
                "drift_px": [round(float(search_drift[0]), 2), round(float(search_drift[1]), 2)],
                "window": [int(top), int(bottom), int(left), int(right)],
                "window_px": [int(bottom - top), int(right - left)],
            },
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

    # --- One cropper, used by the census queue and the frame stepper ----------
    #
    # The shared-window property is the subtle part -- it is what makes the
    # census blink a true toggle rather than two independently re-cropped
    # images, where the cell sits still and the BACKGROUND jumps. Two croppers
    # would diverge on exactly that, so there is one (Round 16 E3).

    def _cell_window(self, labels_a, label_a, labels_b=None, label_b=None, pad: int = 28):
        """The tight box around one cell, optionally unioned with a second
        cell in another frame so both fit one window."""
        ys, xs = np.nonzero(labels_a == int(label_a))
        if not len(ys):
            return None
        if labels_b is not None and label_b is not None:
            ys2, xs2 = np.nonzero(labels_b == int(label_b))
            if len(ys2):
                ys = np.concatenate([ys, ys2]); xs = np.concatenate([xs, xs2])
        height, width = labels_a.shape
        return (max(0, int(ys.min()) - pad), min(height, int(ys.max()) + pad + 1),
                max(0, int(xs.min()) - pad), min(width, int(xs.max()) + pad + 1))

    def _cell_crop(self, frame_number, labels, label, window, color,
                   show_labels: bool = False, show_outline: bool = False,
                   label_size: int = 10, outline_color: str = "#ff0000"):
        from PIL import Image
        loaded = self._load_frame(frame_number)
        base = loaded[2] if loaded else None
        if base is None:
            base = np.zeros(labels.shape + (3,), dtype=np.uint8)
        if label is not None:
            array = rendering.render_cell_highlight(labels, base, int(label), color)
            array = np.asarray(array)
        else:
            array = base.copy()
        if show_outline:
            # Red by default (Round 18 B2): the membrane signal is grey-white
            # and the highlight fills are green, so a green outline has the
            # least separation of any available choice.
            array = rendering.draw_cell_outlines(np.array(array), labels, outline_color)
        image = Image.fromarray(np.asarray(array))
        if show_labels:
            # TRACK ids, from the same resolver the main overlay uses. The
            # inset used to draw raw labels, which Cellpose renumbers every
            # frame -- so stepping it made every number change even where
            # tracking was perfect, the exact opposite of what the panel is
            # for (Round 17 C2b).
            image = rendering.draw_cell_labels(
                image, labels, font_size=int(label_size),
                matched_lookup=rendering.build_matched_lookup(self.tracking_data, int(frame_number)),
                frame_number=int(frame_number),
            )
        top, bottom, left, right = window
        return rendering.encode_png_base64(image.crop((left, top, right, bottom)))

    def _census_window(self, before_labels, after_labels, item, pad):
        return self._cell_window(before_labels, item["label"],
                                 after_labels, item.get("shown_label"), pad=pad)

    def _census_crop(self, frame_number, labels, label, window, color):
        return self._cell_crop(frame_number, labels, label, window, color)

    # --- Round 25: the radius generator, the shared window, the anchor view -----

    #: `"radius"` = Round 25's brute-force version: every f+1 cell whose centroid
    #: lies within `k` median cell radii of the registered anchor, by distance,
    #: excluding nothing. `"ranked"` = the Round 20-23 multi-source merge
    #: (`_census_candidates`). Both coexist until the maintainer's click-test
    #: decides; do not delete either -- the ranked one carries four rounds of
    #: conclusions. Switchable per session with `set_candidate_settings`.
    CANDIDATE_MODE = "radius"
    #: R = k x median cell radius, not an absolute pixel count, so the setting
    #: carries between magnifications: WT GBE (13.0 px) -> 39 px, OE GBE (23.0 px)
    #: -> 69 px.
    CANDIDATE_RADIUS_K = 3.0
    #: The anchor's outline on each candidate tile. Blue, a thin line, so it
    #: separates from the red candidate fill.
    ANCHOR_OUTLINE_COLOR = "#29b6ff"

    def get_candidate_settings(self) -> dict:
        return {"mode": self.candidate_mode, "k": float(self.candidate_radius_k),
                "modes": ["radius", "ranked"]}

    def set_candidate_settings(self, mode: str = None, k: float = None) -> dict:
        """Switch the candidate generator or its radius multiple. Only changes
        what the NEXT `get_census_item` offers; nothing already recorded moves."""
        if mode is not None:
            if mode not in ("radius", "ranked"):
                return {"error": f"Unknown candidate mode {mode!r}; expected 'radius' or 'ranked'."}
            self.candidate_mode = mode
        if k is not None:
            if not float(k) > 0:
                return {"error": "The radius multiple k must be positive."}
            self.candidate_radius_k = float(k)
        return self.get_candidate_settings()

    @staticmethod
    def _census_search_drift(state, frame):
        drift = (state.get("search_drift_by_frame") or {}).get(frame)
        return (float(drift[0]), float(drift[1])) if drift else (0.0, 0.0)

    def _census_radius_px(self, state):
        radius = float(state.get("cell_radius_px") or 0.0)
        if radius <= 0:
            radius = float(self._registration().get("median_cell_radius_px") or 0.0)
        return float(self.candidate_radius_k) * radius

    @staticmethod
    def _label_centroids(labels):
        """`{label: (x, y)}` -- pixel means, the same convention as
        `census._centroids`, in one vectorised pass."""
        from scipy import ndimage
        ids = [int(v) for v in np.unique(labels) if v]
        if not ids:
            return {}
        centres = ndimage.center_of_mass(np.ones_like(labels), labels, ids)
        return {label: (float(c[1]), float(c[0])) for label, c in zip(ids, centres)}

    @classmethod
    def _census_radius_candidates(cls, item, before_labels, after_labels, drift, radius_px):
        """Every f+1 cell whose centroid is within `radius_px` of the anchor's
        f centroid plus the pair's measured drift, nearest first (Round 25 D2).

        Nothing is excluded -- not even `shown`, the tracker's current answer.
        The ranked version drops it ("offering what the user already sees is
        offering nothing"); here it is a reference point: the user sees which
        rank the tracker's own answer holds. It is flagged, not hidden.

        No cap and no quota: the maintainer's specification is that the list
        makes no choices, at the price of more pages.
        """
        anchor = int(item["label"])
        shown = None if item.get("shown_label") is None else int(item["shown_label"])
        ys, xs = np.nonzero(before_labels == anchor)
        if not len(ys):
            return []
        cx, cy = float(xs.mean()) + drift[0], float(ys.mean()) + drift[1]
        ranked = sorted((float(np.hypot(x - cx, y - cy)), label)
                        for label, (x, y) in cls._label_centroids(after_labels).items())
        out = []
        for distance, label in ranked:
            if distance > radius_px:
                break
            out.append({
                "label": int(label), "reason": "radius", "detail": round(distance, 1),
                "rank": len(out) + 1, "is_tracker_answer": label == shown,
                "why": (f"{round(distance, 1)} px from the anchor at frame {int(item['frame'])}"
                        + (" (the tracker's answer)" if label == shown else "")),
            })
        return out

    @staticmethod
    def _census_anchor_window(before_labels, after_labels, item, candidates, drift,
                              radius_px, pad):
        """The ONE window the blink and every candidate tile share (Round 25 B).

        Centred on the anchor's centroid carried into f+1 by the measured drift,
        and square with side `max(2R + 2*pad, whatever contains the anchor, the
        tracker's answer and every offered candidate)`. The floor is what keeps
        the radius version honest: a cell offered because it lies within R must
        not be cropped out of its own tile. Clamped at the image edge, which can
        move the centre but never shrinks the window below the image size.

        Round 23 C2 gave each tile its own window centred on the candidate; the
        maintainer's click-test showed the cost -- the green anchor and the red
        candidate never sat at the same place in the view, so every comparison
        needed a mental re-alignment. Same size is still the point (crops are
        encoded 1:1 and scaled by CSS), and now same centre too.
        """
        from scipy import ndimage
        anchor = int(item["label"])
        ys, xs = np.nonzero(before_labels == anchor)
        cx, cy = float(xs.mean()) + drift[0], float(ys.mean()) + drift[1]
        reach = 0.0
        boxes = [(ys.min(), ys.max(), xs.min(), xs.max()),
                 (ys.min() + drift[1], ys.max() + drift[1], xs.min() + drift[0], xs.max() + drift[0])]
        wanted = [int(c["label"]) for c in candidates]
        if item.get("shown_label") is not None:
            wanted.append(int(item["shown_label"]))
        slices = ndimage.find_objects(after_labels)
        for label in wanted:
            if 0 < label <= len(slices) and slices[label - 1] is not None:
                sy, sx = slices[label - 1]
                boxes.append((sy.start, sy.stop - 1, sx.start, sx.stop - 1))
        for y0, y1, x0, x1 in boxes:
            reach = max(reach, abs(y0 - cy), abs(y1 - cy), abs(x0 - cx), abs(x1 - cx))
        half = int(np.ceil(max(radius_px + pad, reach + 8)))
        height, width = after_labels.shape
        side_h, side_w = min(2 * half + 1, height), min(2 * half + 1, width)
        top = min(max(0, int(round(cy)) - half), height - side_h)
        left = min(max(0, int(round(cx)) - half), width - side_w)
        return (top, top + side_h, left, left + side_w)

    def _census_tile_images(self, frame, before_labels, after_labels, item, candidates,
                            window, drift, anchor_view):
        """Candidate tiles: frame f+1 cropped to the shared window, the candidate
        filled red, and -- when `anchor_view` -- the anchor's outline (carried by
        the same drift as the window's centre) drawn over it.

        The outline is a view preference and can be switched off, because it is
        most in the way exactly when it is closest: a high-IoU candidate's
        boundary runs right beside the anchor's, and that is when the
        candidate's own shape most needs to be readable (Round 25 B3).
        """
        from PIL import Image
        top, bottom, left, right = window
        loaded = self._load_frame(frame + 1)
        base = loaded[2] if loaded and loaded[2] is not None else None
        if base is None:
            base = np.zeros(after_labels.shape + (3,), dtype=np.uint8)
        base_crop = np.asarray(base)[top:bottom, left:right]
        labels_crop = after_labels[top:bottom, left:right]
        outline = None
        if anchor_view:
            anchor_mask = (before_labels == int(item["label"])).astype(np.int32)
            if drift[0] or drift[1]:
                anchor_mask = tracking_benchmark.shift_labels(anchor_mask, drift[0], drift[1])
            outline = rendering.cell_boundary_mask(anchor_mask[top:bottom, left:right])
        red = rendering._hex_to_rgb("#ff6b6b")
        blue = rendering._hex_to_rgb(self.ANCHOR_OUTLINE_COLOR)
        images = []
        for candidate in candidates:
            tile = base_crop.copy()
            tile[labels_crop == int(candidate["label"])] = red
            if outline is not None:
                tile[outline] = blue
            images.append(rendering.encode_png_base64(Image.fromarray(tile)))
        return images

    def get_census_candidate_images(self, position: int, anchor_view: bool = True) -> dict:
        """Re-render the tiles of an item ALREADY offered, with the anchor view
        on or off. Reads the cached label list and window; never re-runs a
        generator, so switching the view cannot change what is offered -- and
        the returned `labels` let the caller assert exactly that (Round 25 B3).
        """
        state = getattr(self, "_census", None)
        if not state:
            return {"error": "Start a census first."}
        offered = state.get("candidates", {}).get(int(position))
        window = state.get("windows", {}).get(int(position))
        if offered is None or window is None:
            return {"error": f"Item {position} has not been shown yet."}
        item = state["queue"][int(position)]
        frame = item["frame"]
        masks = state["masks"]
        images = self._census_tile_images(
            frame, masks[frame], masks[frame + 1], item, offered, window,
            self._census_search_drift(state, frame), anchor_view)
        return {"position": int(position), "anchor_view": bool(anchor_view),
                "labels": [entry["label"] for entry in offered], "images": images}

    def _census_candidates(self, item, before_labels, after_labels, window,
                           drift=(0.0, 0.0), limit: int = 12, render: bool = True):
        """The alternatives offered when the verdict is "wrong", seeded from the
        **anchor** rather than from the tracker's answer.

        **This is a MERGE of several rankings, not a concatenation truncated to
        three.** It used to accumulate four sources and then take `[:3]`, which
        meant sources 3 and 4 could never appear unless source 2 ran short --
        and for an `environment_flagged` item source 1 is empty by construction
        (there is no overlap reference for that set), so the offered list was
        *always exactly the IoU ranking's top three*. When that ranking was
        wrong there was no second opinion, though two more had been computed
        (Round 20 C1).

        The sources, round-robin so a failure in any one costs at most one slot:

        1. the overlap reference's pick, where the set provides one;
        2. highest IoU after shifting the anchor's mask by the global median
           drift;
        3. highest **coverage** of the shifted anchor -- `|A n B| / |A|`;
        4. nearest centroid, drift-corrected;
        5. nearest to where the tracker pointed, as a last resort.

        Each source is ranked 12 deep, and `CANDIDATE_QUOTAS` decides how much
        of that depth each may spend before the others get a turn; whatever a
        source cannot fill is handed back so the list still reaches `limit`.
        The quotas exist because Round 22 raised both cuts from 4 to 12 and
        changed nothing: the merge is depth-first, so with three non-empty
        sources and `limit=12` it stopped after rank 3 and every source
        contributed the same first four as before. The list grew from 4-5 to 12
        only because those four stopped duplicating each other. See
        `_merge_candidate_sources`.

        **Sources 2 and 3 fail in opposite directions, which is why both are
        here.** IoU penalises a size difference in both directions, so when the
        true successor is much larger than the anchor it barely separates it
        from a same-size neighbour; coverage normalises by the anchor only, so
        it does not -- but when the anchor divides into two smaller daughters
        it is IoU that separates the daughter better. `docs/census.md` §8.8
        works both pairs out. That is **algebra, not a measurement**: which
        source a human actually picks is recorded per verdict (`chosen_reason`,
        `chosen_rank`), and that is what decides whether coverage stays.

        **A per-cell constant-velocity source was built and then removed** --
        the reasoning was that a single global translation is a poor model for
        germband extension, which converges along one axis and extends along the
        other. It measured worse at rank 3 on every dataset and queue kind tried
        (GBE geometry 77.9% -> 75.7%, GBE environment 89.5% -> 88.2%,
        `AS_dorsal` geometry 100% -> 95.2%) and gained nothing at full depth: it
        displaced good candidates off the first page. `v2/docs/census.md` §8.4
        has the table, so it does not need re-measuring to stay removed.

        **IoU and the motion prediction PROPOSE here; they do not score.**
        Nothing is measured and no figure comes out of this ordering -- it
        arranges a shortlist and the human decides. That is not the circularity
        the standing constraint forbids. Do not "fix" it back.
        """
        anchor = int(item["label"])
        shown = None if item.get("shown_label") is None else int(item["shown_label"])
        frame = int(item["frame"])

        moved_before = before_labels
        if drift and (drift[0] or drift[1]):
            moved_before = tracking_benchmark.shift_labels(before_labels, drift[0], drift[1])

        def _ranked(pairs):
            ranked = sorted(((value, b) for (a, b), value in pairs.items() if a == anchor),
                            reverse=True)
            return [(int(b), round(float(v), 3)) for v, b in ranked[:12]]

        def iou_ranking():
            return _ranked(cell_tracking.mask_overlap_fractions(moved_before, after_labels))

        def coverage_ranking():
            # `|A n B| / |A|`, NOT the raw intersection: the raw intersection
            # favours whichever candidate is biggest, which is the failure this
            # source exists to avoid. See `mask_coverage_fractions`.
            return _ranked(cell_tracking.mask_coverage_fractions(moved_before, after_labels))

        centres_after = census._centroids(after_labels)
        centres_before = census._centroids(before_labels)

        def nearest_to(point):
            if point is None:
                return []
            ranked = sorted(((np.hypot(c[0] - point[0], c[1] - point[1]), label)
                             for label, c in centres_after.items()), key=lambda pair: pair[0])
            return [(int(label), round(float(distance), 1)) for distance, label in ranked[:12]]

        origin = centres_before.get(anchor)
        drifted_origin = None if origin is None else (origin[0] + drift[0], origin[1] + drift[1])

        sources = [
            ("reference", [(int(l), None) for l in
                           (item.get("reference_label"), item.get("true_label")) if l is not None]),
            ("overlap", iou_ranking()),
            ("coverage", coverage_ranking()),
            ("nearest", nearest_to(drifted_origin)),
            ("pointed", nearest_to(centres_after.get(shown)) if shown is not None else []),
        ]

        # `shown` is excluded: it is what the user is already looking at, and
        # offering it as an alternative is offering nothing.
        merged, depth_reached = self._merge_candidate_sources(
            sources, limit, {shown} if shown is not None else set())
        # Reachable from the product, not only from a measurement script
        # (`CLAUDE.md` rule 6 case 2): `get_census_item` returns it and the
        # harness prints it. A depth that can only be measured out of band is
        # how Round 22 came to report a list length as if it were a depth.
        self._last_candidate_depth = dict(depth_reached)

        # Every tile gets its OWN window, the same pixel size as the anchor's,
        # centred on that candidate (Round 23 C2). It used to be one window
        # grown to contain all of them at once (`_window_containing`), which
        # was tolerable at 4-5 nearby candidates and became unusable at 12: the
        # candidates span ~50 px, so the shared window opened out to most of the
        # field and every candidate tile rendered as a whole-tissue thumbnail
        # with the cell a few pixels across, beside a tightly-cropped anchor.
        # Two images at different scales cannot be compared, which is the one
        # thing this panel exists for.
        for rank, candidate in enumerate(merged, start=1):
            candidate["rank"] = rank
            # `render=False` from `get_census_item` since Round 25 B: the tiles
            # are cropped there, all in the blink's own anchor-centred window,
            # so these per-candidate `_window_like` crops would be thrown away.
            if render:
                candidate["image"] = self._census_crop(
                    frame + 1, after_labels, candidate["label"],
                    self._window_like(window, after_labels, candidate["label"]), "#ff6b6b")
            # Every tile states its BASIS in the same words. They used to read
            # "13.4 px", "16.5 px from the link", "26.1 px" -- three tiles, two
            # different origins, and only one of them saying which. A user
            # cannot compare numbers measured from different points
            # (Round 21 C3).
            #
            # And the words name a PLACE (Round 23 C3): "the cell" and "the
            # link" are not locations, and the maintainer asked what the
            # difference was -- which is the answer. "the anchor at frame N" is
            # the cell being judged, at f; "the tracker's answer" is the cell it
            # linked to, at f+1.
            candidate["why"] = {
                "reference": "the overlap reference's pick",
                "overlap": (f"IoU {candidate['detail']} with the anchor at frame {frame}"
                            if candidate["detail"] is not None
                            else f"overlaps the anchor at frame {frame}"),
                "coverage": (f"covers {round(100 * candidate['detail'])}% of the anchor at frame {frame}"
                             if candidate["detail"] is not None
                             else f"covers the anchor at frame {frame}"),
                "nearest": (f"{candidate['detail']} px from the anchor at frame {frame}"
                            if candidate["detail"] is not None
                            else f"near the anchor at frame {frame}"),
                "pointed": (f"{candidate['detail']} px from the tracker's answer"
                            if candidate["detail"] is not None
                            else "near the tracker's answer"),
            }[candidate["reason"]]
        return merged

    #: How deep each source may be searched before the others get the rest of
    #: the budget. They sum to more than `limit`, deliberately: whatever a
    #: source cannot fill is handed back, so the list still reaches `limit`.
    #:
    #: The split is not even, and the measurement says why (Round 23 A3, GBE
    #: `environment_flagged`, 64 items, this run's own extraction + tracking):
    #: the 12th-nearest cell already sits a median **49.8 px** from the anchor,
    #: about 3.8 GBE cell radii, and a normally-moving cell's successor is not
    #: out there -- so ranks 4-12 of `nearest` are budget spent where it cannot
    #: help. `overlap` is the shallow one: only 1-5 f+1 cells overlap the anchor
    #: at all (9 of 64 items have exactly one), so it usually cannot spend even
    #: its four.
    CANDIDATE_QUOTAS = {"reference": 2, "overlap": 4, "coverage": 4,
                        "nearest": 3, "pointed": 2}

    @classmethod
    def _merge_candidate_sources(cls, sources, limit, exclude, quotas=None):
        """Round-robin `[(reason, [(label, detail), ...]), ...]` into one list,
        each source held to its quota until the others have had their turn.

        Returns `(merged, depth_reached)` -- `depth_reached[reason]` is **how
        far down that source's own ranking the merge actually got**, 1-based,
        and 0 for a source that contributed nothing.

        That second return value is the whole point (Round 23 A2). Round 22
        raised each source's own cut from `[:4]` to `[:12]` and the walk from
        `range(4)` to `range(12)`, and reported that the offered list grew from
        4-5 to 12 -- which it did, **without any source being searched one rank
        deeper**. The merge is depth-first: rank 0 from every source, then rank
        1 from every source, and so on. With three non-empty sources (source 1
        is empty by construction for `environment_flagged`) and `limit=12` it
        stops after rank 3, so every source contributed exactly its first four,
        the same four as before. The list got longer only because those first
        four stopped duplicating each other.

        **List length is not search depth**, and reporting the first is what hid
        that the second had not moved.

        Two passes: quota first, then the leftovers. The second pass is what
        keeps the list at `limit` when a source runs dry -- on GBE `overlap`
        typically offers 1-3 entries against a quota of 4, and that slack goes
        to whichever source still has entries rather than shortening the list.
        """
        quotas = cls.CANDIDATE_QUOTAS if quotas is None else quotas
        merged, seen = [], set(exclude)
        depth_reached = {reason: 0 for reason, _ in sources}
        taken = {reason: 0 for reason, _ in sources}

        def offer(reason, entries, cap):
            for depth in range(taken[reason], len(entries)):
                if taken[reason] >= cap or len(merged) >= limit:
                    return
                label, detail = entries[depth]
                taken[reason] = depth + 1
                if label in seen:
                    continue
                seen.add(label)
                merged.append({"label": label, "reason": reason, "detail": detail})
                depth_reached[reason] = depth + 1
                return

        longest = max((len(entries) for _, entries in sources), default=0)
        for _ in range(longest):                      # quota pass
            if len(merged) >= limit:
                break
            for reason, entries in sources:
                offer(reason, entries, quotas.get(reason, 0))
        for _ in range(longest):                      # leftover pass
            if len(merged) >= limit:
                break
            before = len(merged)
            for reason, entries in sources:
                offer(reason, entries, longest)
            if len(merged) == before:
                break
        return merged, depth_reached

    @staticmethod
    def _window_like(window, labels, label):
        """A window the SAME pixel size as `window`, centred on `label`.

        Same size is the point: the crops are encoded at 1:1 and sized by CSS,
        so two crops render at the same scale only if they have the same pixel
        dimensions. Centred rather than grown, so the candidate is not a speck
        at the edge of a wide view (Round 23 C2). Clamped at the image bounds,
        which can shift the centre but never changes the size unless the window
        is larger than the frame.
        """
        top, bottom, left, right = window
        height, width = labels.shape
        box_h, box_w = bottom - top, right - left
        ys, xs = np.nonzero(labels == int(label))
        if not len(ys):
            return window
        cy, cx = (int(ys.min()) + int(ys.max())) // 2, (int(xs.min()) + int(xs.max())) // 2
        new_top = min(max(0, cy - box_h // 2), max(0, height - box_h))
        new_left = min(max(0, cx - box_w // 2), max(0, width - box_w))
        return (new_top, min(height, new_top + box_h),
                new_left, min(width, new_left + box_w))

    @staticmethod
    def _window_containing(window, labels, extra_labels, pad: int = 8):
        """Grow `window` until it contains every label in `extra_labels`.

        No longer used for census candidates -- each of those now gets its own
        equally-sized window via `_window_like` (Round 23 C2). Kept because the
        frame stepper's shared-window property is built on the same idea."""
        top, bottom, left, right = window
        height, width = labels.shape
        for label in extra_labels:
            ys, xs = np.nonzero(labels == int(label))
            if not len(ys):
                continue
            top = min(top, max(0, int(ys.min()) - pad))
            bottom = max(bottom, min(height, int(ys.max()) + pad + 1))
            left = min(left, max(0, int(xs.min()) - pad))
            right = max(right, min(width, int(xs.max()) + pad + 1))
        return (top, bottom, left, right)

    def submit_census_verdict(self, position: int, verdict: str,
                              chosen_label: int = None, ms_elapsed: float = 0.0) -> dict:
        state = getattr(self, "_census", None)
        if not state:
            return {"error": "Start a census first."}
        if state["mode"] != "census":
            return {"error": "This is a correction session; it cannot write ground truth. "
                            "Ground truth collected against a tracking run is not valid for a "
                            "run it corrected -- start a census session instead."}
        queue = state["queue"]
        if not 0 <= int(position) < len(queue):
            return {"error": f"Position {position} is outside the queue."}
        # Which source first proposed the chosen cell, and where it sat. Looked
        # up from the list this API itself offered rather than sent back by the
        # frontend: the frontend would have to be trusted to report the reason
        # of the tile it drew, and a mismatch there is exactly the kind of thing
        # that would never surface.
        reason, rank = "", None
        offered_mode = state.get("offered_mode", {}).get(int(position), "")
        if chosen_label is not None:
            for entry in state.get("candidates", {}).get(int(position), []):
                if entry["label"] == int(chosen_label):
                    reason, rank = entry["reason"], entry["rank"]
                    break
        try:
            census.append_verdict(state["path"], state["session_id"], queue[int(position)],
                                  verdict, chosen_label, ms_elapsed, labeller=state["labeller"],
                                  chosen_reason=reason, chosen_rank=rank,
                                  register=state["register"],
                                  environment_threshold=state.get("environment_threshold"),
                                  candidate_mode=offered_mode)
        except ValueError as error:
            return {"error": str(error)}
        return {"saved": True, "next": int(position) + 1, "total": len(queue),
                "chosen_reason": reason, "chosen_rank": rank, "candidate_mode": offered_mode}

    # --- Correction mode ------------------------------------------------------
    #
    # Same queue, same blink, same keystrokes; a different destination. Edits
    # accumulate as pending and land in ONE batch with one snapshot and one
    # undo -- not N sequential writes, each of which could trip
    # `_evict_future_if_colliding` separately with no way back.

    def queue_census_edit(self, position: int, chosen_label: int = None) -> dict:
        """Record a pending correction for the item at `position`.

        `chosen_label` is the label at frame f+1 the link should point to;
        `None` means "none of these", which clears the track's claim at that
        frame rather than pointing it anywhere. Nothing is written yet.
        """
        state = getattr(self, "_census", None)
        if not state:
            return {"error": "Start a session first."}
        if state["mode"] != "correct":
            return {"error": "This is a census session; it cannot edit the tracking data. "
                            "Its verdicts judge the run as it stands -- start a correction "
                            "session instead."}
        queue = state["queue"]
        if not 0 <= int(position) < len(queue):
            return {"error": f"Position {position} is outside the queue."}
        item = queue[int(position)]
        if (chosen_label is not None and item.get("shown_label") is not None
                and int(chosen_label) == int(item["shown_label"])):
            # Radius mode offers the tracker's own answer (Round 25 D2). Picking
            # it means "the link was right", which is not an edit -- queueing it
            # would report a pending correction that changes nothing.
            return {"error": f"Cell {chosen_label} is already the tracker's answer at frame "
                             f"{item['frame'] + 1}; there is nothing to correct."}
        if item.get("track_id") is None:
            return {"error": f"The cell at frame {item['frame']} belongs to no track, so there is "
                             "no link to correct here."}
        state["edits"][int(position)] = {
            "frame": int(item["frame"]),
            "track_id": int(item["track_id"]),
            "from_label": item.get("shown_label"),
            "chosen_label": None if chosen_label is None else int(chosen_label),
        }
        return {"pending": len(state["edits"]), "position": int(position)}

    def clear_census_edit(self, position: int) -> dict:
        state = getattr(self, "_census", None)
        if not state:
            return {"error": "Start a session first."}
        state["edits"].pop(int(position), None)
        return {"pending": len(state["edits"])}

    def _apply_census_edits_to(self, table):
        """Apply every pending edit to `table`, in queue order. Returns
        `(table, displaced_list, taken_from)`. `add_tracking_id` carries the
        dedup and the eviction guard, so this does not re-implement either.

        `taken_from` records every claim the batch removed from a track that did
        not ask to lose it. `add_tracking_id`'s dedup clears the losing track's
        cell so one cell is never owned twice (Round 15 §A3) -- necessary, but
        it happened silently: `displaced_list` only carries
        `_evict_future_if_colliding` relocations, so a track could lose a frame
        with nothing in the return value saying so. Round 18 §C's invariant is
        that no track loses a claim the return value did not report, and this is
        what makes that assertable rather than merely true-looking.
        """
        state = self._census
        id_col = table.columns[0]
        displaced_list, taken_from = [], []
        for position in sorted(state["edits"]):
            edit = state["edits"][position]
            target_frame = edit["frame"] + 1
            column = f"Frame{target_frame}"
            if column not in table.columns:
                continue
            if edit["chosen_label"] is None:
                rows = table.index[table[table.columns[0]] == edit["track_id"]]
                if len(rows):
                    table.at[rows[0], column] = None
                continue
            cell_index = f"Cell{target_frame}_{edit['chosen_label']}"
            # Who holds this cell right now, before the write takes it away.
            for row in table.index:
                if (str(table.at[row, column]) == cell_index
                        and int(table.at[row, id_col]) != int(edit["track_id"])):
                    taken_from.append({"frame": target_frame, "cell_index": cell_index,
                                       "from_track_id": int(table.at[row, id_col]),
                                       "to_track_id": int(edit["track_id"])})
            table, displaced = manual_correction_tracking.add_tracking_id(
                table, target_frame, edit["track_id"], cell_index,
            )
            if displaced is not None:
                displaced_list.append(displaced)
        return table, displaced_list, taken_from

    def preview_census_edits(self) -> dict:
        """Run the whole pending batch through §B1's structural checks, without
        touching the live table."""
        state = getattr(self, "_census", None)
        if not state or state["mode"] != "correct":
            return {"error": "Not a correction session."}
        if not state["edits"]:
            return {"error": "No pending corrections."}
        snapshot = self.tracking_data.copy(deep=True)
        trial, displaced, taken_from = self._apply_census_edits_to(snapshot.copy(deep=True))
        report = manual_correction_tracking.diff_report(snapshot, trial)
        assert self.tracking_data.astype(str).equals(snapshot.astype(str)), \
            "previewing corrections must not mutate the live tracking table"
        return {
            **report,
            "pending": len(state["edits"]),
            "evicts": displaced,
            "claims_removed": taken_from,
            "takes_from": [
                {"frame": e["frame"] + 1, "track_id": e["track_id"],
                 "chosen_label": e["chosen_label"]}
                for e in state["edits"].values() if e["chosen_label"] is not None
            ],
            "blocking": bool(report["double_claimed"] or report["orphaned"]
                             or report["vanished_track_ids"]),
        }

    def apply_census_edits(self, force: bool = False) -> dict:
        """Apply the whole pending batch: one snapshot, one atomic write, one
        undo. Refuses a batch that fails a structural check unless `force`."""
        state = getattr(self, "_census", None)
        if not state:
            return {"error": "Start a session first."}
        if state["mode"] != "correct":
            return {"error": "A census session cannot apply corrections."}
        if not state["edits"]:
            return {"error": "No pending corrections."}
        preview = self.preview_census_edits()
        if preview.get("blocking") and not force:
            return {"error": "This batch would double-claim a cell or leave a track holding "
                             "nothing. Nothing was applied.", "preview": preview}

        snapshot = self.tracking_data.copy(deep=True)
        asked_frames = sorted({e["frame"] + 1 for e in state["edits"].values()})
        applied, displaced, taken_from = self._apply_census_edits_to(self.tracking_data)
        self.tracking_data = applied
        state["undo"] = snapshot
        count = len(state["edits"])
        state["edits"] = {}

        # `frames_written` is MEASURED from the before/after tables, not
        # asserted from the edit list. Reporting the edited frames was a false
        # claim: `_evict_future_if_colliding` relocates a colliding track's
        # ENTIRE future onto a fresh id, so a single edit at frame 3 was
        # observed rewriting frames 3-11. The eviction is the intended
        # behaviour -- it beats silently overwriting -- but a return value that
        # under-reports its own reach is how a caller ends up trusting a range
        # that was never true.
        written = sorted(self._frames_that_differ(snapshot, applied))
        return {"applied": count, "displaced_list": displaced,
                "claims_removed": taken_from,
                "frames_asked": asked_frames, "frames_written": written,
                "beyond_asked": sorted(set(written) - set(asked_frames)),
                "undo_available": True, "preview": preview}

    @staticmethod
    def _frames_that_differ(before, after) -> set:
        """Which `Frame{n}` columns actually changed, compared BY TRACK ID --
        an eviction appends a row, so the two tables need not be the same
        length and a positional diff would be meaningless."""
        def claims(table):
            id_col = table.columns[0]
            out = {}
            for position in table.index:
                holder = int(table.at[position, id_col])
                for column in (c for c in table.columns if c.startswith("Frame")):
                    value = table.at[position, column]
                    if pd.notna(value):
                        out[(int(column[len("Frame"):]), holder)] = str(value)
            return out

        was, now = claims(before), claims(after)
        return {frame for (frame, _holder) in set(was) | set(now)
                if was.get((frame, _holder)) != now.get((frame, _holder))}

    def undo_census_edits(self) -> dict:
        """Restore the table to the snapshot taken before the last Apply-all.

        A restore, never an inverse operation: `add_tracking_id` may trigger
        `_evict_future_if_colliding`, which relocates a third track's whole
        future onto a fresh id and has no clean inverse.
        """
        state = getattr(self, "_census", None)
        if not state or state.get("undo") is None:
            return {"error": "Nothing to undo."}
        self.tracking_data = state["undo"]
        state["undo"] = None
        return {"restored": True}

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
        try:
            summary = census.summarise(path, state["session_id"] if state else None,
                                       source=(state or {}).get("kind"))
        except census.MixedSources as error:
            return {"error": str(error)}
        if state:
            summary["queue_size"] = len(state["queue"])
            summary["judge_registered"] = state["register"]
            summary["mode"] = state["mode"]
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

    # --- Ground truth as a CTC-style sibling folder (Round 18 E) -------------

    def save_ground_truth_tracking(self, labeller: str = "", note: str = "") -> dict:
        """Write the current tracking table as ground truth under
        `cell_track_output/GT_All_cell_tracking/`, stamped with the run it was
        collected against.

        The normal pipeline never reads `GT_*`; it is loaded only when asked.
        """
        if self.tracking_data is None:
            return {"error": "Track the movie before saving ground truth."}
        if not self.output_dirs:
            return {"error": "Set up the project paths first."}
        state = getattr(self, "_census", None)
        record = census.save_ground_truth(
            self.output_dirs["cell_track_output"], self.tracking_data, self.tracking_data,
            labeller=labeller or (state or {}).get("labeller", ""),
            note=note,
            session_id=(state or {}).get("session_id"),
            source=(state or {}).get("kind"),
            mode=(state or {}).get("mode"),
            tracking_parameters={
                "overlap_weight": self.overlap_weight,
                "centroid_weight": self.centroid_weight,
                "motion_compensation": self.motion_compensation,
                "max_frame_gap": self.max_frame_gap,
                "drift_mode": self.drift_mode,
                # Which row set the tracking saw (Round 26 B2): since the
                # border exclusion moved off the extraction path, a table
                # produced under the old semantics has fewer cells per frame
                # than one produced under the new, and nothing else records it.
                "exclude_edge_cells": self.exclude_edge_cells,
                "edge_margin_px": data_input.DEFAULT_EDGE_MARGIN_PX,
            },
            resolved_inputs={"mask_folder": self.mask_folder, "npy_folder": self.npy_folder,
                             "image_folder": self.image_folder},
        )
        return {"saved": census.gt_table_path(self.output_dirs["cell_track_output"]),
                "provenance": record}

    def load_ground_truth_tracking(self) -> dict:
        """Read the GT table back, and say whether it was collected against the
        tracking currently in memory."""
        if not self.output_dirs:
            return {"error": "Set up the project paths first."}
        found = census.load_ground_truth(self.output_dirs["cell_track_output"],
                                         against=self.tracking_data)
        if "error" in found:
            return found
        table = found.pop("table")
        found["rows"] = int(len(table))
        if found.get("matches_current_tracking") is False:
            found["warning"] = (
                "This ground truth was collected against a DIFFERENT tracking table than the one "
                "loaded. Ground truth collected against one run is not valid for a run it "
                "corrected -- the link set itself changes."
            )
        return found

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

    def _environment_scores(self, frame_number: int, direction: str,
                            with_unscoreable: bool = False):
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
            prev_scores, prev_out = self._environment_scores(frame_number, "prev", True)
            next_scores, next_out = self._environment_scores(frame_number, "next", True)
            # A cell unscoreable on EITHER side is unscoreable for the "worse
            # of the two" reading: the min over a value and a non-value is not
            # that value.
            merged_out = {**prev_out, **next_out}
            if not prev_scores:
                merged = next_scores
            elif not next_scores:
                merged = prev_scores
            else:
                merged = {label: min(value, next_scores.get(label, value))
                          for label, value in prev_scores.items()}
            return (merged, merged_out) if with_unscoreable else merged

        other_frame = frame_number - 1 if direction == "prev" else frame_number + 1
        graph_current = self._get_neighbor_graph(frame_number)
        graph_other = self._get_neighbor_graph(other_frame)
        if not graph_current or not graph_other:
            return ({}, {}) if with_unscoreable else {}
        scores, unscoreable = adjacency.environment_scores(
            graph_current, graph_other,
            self._label_to_track(frame_number), self._label_to_track(other_frame),
            with_unscoreable=True,
        )
        return (scores, unscoreable) if with_unscoreable else scores

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
            extracted_labels=self._extracted_labels(frame_number),
            unextracted_color=self.unextracted_color,
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

    def verify_confirmed_identity_moves(self, track_id: int, moves: list,
                                        drift_gate_multiplier: float = 1.5) -> dict:
        """Run a "Check how far this holds" plan through the three structural
        checks and the transposition score, BEFORE Apply becomes clickable.

        See `v2/docs/check-how-far.md`. Three things travel back:

        - the structural report (`manual_correction_tracking.verify_moves_plan`)
          -- double claims, truncated or orphaned tracks, and any step that
          triggers `_evict_future_if_colliding`, which still has no clean
          inverse;
        - each step's `distance` against the gate that admitted it
          (`drift_gate_multiplier x last_radius`), so a step that only just
          squeezed through is visible;
        - a **consistency delta** from `tracking_benchmark.table_link_consistency`
          over the frame pairs the plan touches.

        **The delta is a consistency measure, never an accuracy.** It re-solves
        the affected pairs backwards with the same cost function the tracker
        used, so it answers "does the transposed problem confirm the links this
        table now asserts", not "is the table right". Its degenerate case --
        a coherent identity permutation scores 0 -- is shared with
        `environment_dice` and `forward_backward_consistency`, so it is not a
        second independent line on top of either. Both sides are measured under
        one registration convention, reported in `registered`.

        `forward_backward_consistency` is deliberately NOT used here even
        though §B3 named it: it re-solves from `cells_info` and the masks and
        never reads `tracking_data`, so its delta is identically zero for every
        plan (measured: a fully scrambled table scores the same 48 / 2,634 on
        `AS_dorsal view_E-cad`). A score that cannot move is worse than none.

        Scoring runs on a deep copy and the live table is asserted unchanged --
        that assertion is the safety property of an accept-and-revert loop, and
        it is not optional.
        """
        if self.tracking_data is None or self.cells_info is None:
            return {"error": "No tracking data loaded."}
        if not moves:
            return {"error": "This plan has no moves to verify."}

        structural = manual_correction_tracking.verify_moves_plan(
            self.tracking_data, int(track_id), moves)
        if not structural.get("ok"):
            return {"error": structural.get("error", "The plan could not be applied.")}

        # Per-move gate context. `distance` is already in the plan and nothing
        # displayed it; the gate is what admitted the step, so the ratio is the
        # readable number.
        radii = (self.cells_info.set_index("Cell Index")["Circumscribed Circle Radius"]
                 if "Cell Index" in self.cells_info.columns
                 else self.cells_info["Circumscribed Circle Radius"])
        annotated = []
        for move in moves:
            radius = float(radii.get(move["cell_index"], float("nan"))) if len(radii) else float("nan")
            gate = float(drift_gate_multiplier) * radius
            distance = move.get("distance")
            annotated.append({
                **move,
                "gate_px": None if gate != gate else round(gate, 2),
                "gate_fraction": (None if (gate != gate or not gate or distance is None)
                                  else round(float(distance) / gate, 3)),
            })

        frames = sorted({int(m["frame"]) for m in moves})
        pairs = sorted({f - 1 for f in frames} | set(frames))
        masks = {}
        for frame in self.get_frame_numbers():
            if min(pairs) <= frame <= max(pairs) + 1:
                loaded = self._load_frame(frame)
                if loaded is not None:
                    masks[frame] = loaded[0]

        delta = None
        snapshot = self.tracking_data.copy(deep=True)
        if len(masks) >= 2:
            drift = self._registration().get("drift_by_frame") if self.register_reference else None
            drift_mode = "mask" if drift else "none"
            cells = (self.cells_info.set_index("Cell Index")
                     if "Cell Index" in self.cells_info.columns else self.cells_info)
            score = lambda table: tracking_benchmark.table_link_consistency(  # noqa: E731
                table, cells, masks, frames=pairs,
                drift_mode=drift_mode, drift_by_frame=drift)
            before = score(snapshot)
            trial, _ = manual_correction_tracking.apply_confirmed_identity_moves(
                snapshot.copy(deep=True), int(track_id), moves)
            after = score(trial)
            delta = {
                "before_unconfirmed": before["unconfirmed"],
                "after_unconfirmed": after["unconfirmed"],
                "delta": after["unconfirmed"] - before["unconfirmed"],
                "n": after["n"],
                "frames": pairs,
                "registered": bool(drift),
                "judge": "table_link_consistency",
                "note": ("Consistency under transposition, using the same cost function as the "
                         "tracker -- NOT an accuracy. Negative is fewer links the backward solve "
                         "fails to confirm."),
            }

        # The live table must be untouched by verification. Asserted, not
        # assumed: this is the whole safety property of an accept-and-revert.
        assert self.tracking_data.astype(str).equals(snapshot.astype(str)), \
            "verifying a plan must not mutate the live tracking table"

        return {
            **structural,
            "moves": annotated,
            "consistency": delta,
            "blocking": bool(structural["double_claimed"] or structural["orphaned"]
                             or structural["vanished_track_ids"]),
        }

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

    # --- Track-level visual inspection (Round 27 4b) -----------------------

    def get_track_cell_at_frame(self, track_number: int, frame_number: int) -> Optional[str]:
        """The `Cell{frame}_{label}` this track holds at `frame_number`, or
        `None` for a gap. Resolved here rather than taken from the frontend:
        the mark is evidence, and evidence about which cell was on screen must
        come from the table that put it there."""
        if self.tracking_data is None:
            return None
        column = f"Frame{int(frame_number)}"
        if column not in self.tracking_data.columns:
            return None
        id_col = self.tracking_data.columns[0]
        rows = self.tracking_data[self.tracking_data[id_col] == int(track_number)]
        if rows.empty:
            return None
        value = rows.iloc[0][column]
        return None if pd.isna(value) else str(value)

    def start_track_inspection(self, labeller: str = "") -> dict:
        """Begin a pass over every track. Writes nothing yet -- the file is
        created by the first mark, and the provenance by `finish`."""
        if self.tracking_data is None:
            return {"error": "Track the movie before inspecting it."}
        tracks = self.get_track_numbers()
        if not tracks:
            return {"error": "This tracking table holds no tracks."}
        self._inspection = {
            "session_id": coverage_log.new_session_id(),
            "labeller": labeller,
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "tracks": tracks,
            "path": census.inspection_path(self.output_dirs["cell_track_output"]),
            "marks": 0,
        }
        return {"session_id": self._inspection["session_id"], "tracks": tracks,
                "total": len(tracks), "path": self._inspection["path"]}

    def mark_inspection_cell(self, track_number: int, frame_number: int,
                             ms_elapsed: float = 0.0) -> dict:
        """Record that the cell this track holds at this frame is wrong."""
        state = getattr(self, "_inspection", None)
        if not state:
            return {"error": "Start a visual inspection first."}
        cell_index = self.get_track_cell_at_frame(track_number, frame_number)
        if cell_index is None:
            return {"error": f"Track #{track_number} holds no cell at frame {frame_number}."}
        census.append_inspection_mark(
            state["path"], state["session_id"], int(track_number), int(frame_number),
            cell_index, ms_elapsed, labeller=state["labeller"])
        state["marks"] += 1
        return {"marked": cell_index, "track_id": int(track_number),
                "frame": int(frame_number), "marks": state["marks"]}

    def finish_track_inspection(self, inspected: list = None,
                                per_track_ms: dict = None) -> dict:
        """Close the pass: write the provenance beside the marks, return the
        summary.

        The provenance is the point. Rounds 18-20 treated tables that had never
        been visually inspected as ground truth; this file records that someone
        DID look, under which tracking run, which registration convention and
        which masks -- and, by naming the tracks left unseen, where the looking
        stopped.
        """
        state = getattr(self, "_inspection", None)
        if not state:
            return {"error": "Start a visual inspection first."}
        tracks = state["tracks"]
        seen = sorted({int(t) for t in (inspected or [])})
        unseen = [int(t) for t in tracks if int(t) not in set(seen)]
        summary = census.summarise_inspection(state["path"], state["session_id"])
        dwell = {int(k): float(v) for k, v in (per_track_ms or {}).items()}
        median_dwell = (round(statistics.median(dwell.values()) / 1000.0, 1)
                        if dwell else None)
        record = {
            "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "started_utc": state["started_utc"],
            "session_id": state["session_id"],
            "labeller": state["labeller"],
            "kind": "track_inspection",
            "tracks_total": len(tracks),
            "tracks_inspected": len(seen),
            "tracks_not_inspected": unseen,
            "marks": summary["marks"],
            "tracks_with_marks": summary["tracks_with_marks"],
            "median_seconds_per_track": median_dwell,
            # Which run was looked at, and under which conventions.
            "judge_registered": bool(self.register_reference),
            "mask_folder": self.mask_folder,
            "masks": baselines.mask_fingerprint(self.npy_folder),
            "min_cell_area_px": data_input.get_min_cell_area(),
            "exclude_edge_cells": self.exclude_edge_cells,
            "tracking": {
                "tracks": int(len(self.tracking_data)),
                "overlap_weight": self.overlap_weight,
                "centroid_weight": self.centroid_weight,
                "motion_compensation": self.motion_compensation,
                "max_frame_gap": self.max_frame_gap,
                "drift_mode": self.tracking_drift_mode or self.drift_mode,
            },
        }
        try:
            os.makedirs(os.path.dirname(state["path"]), exist_ok=True)
            with open(state["path"] + ".provenance.json", "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, sort_keys=True)
                handle.write("\n")
        except OSError as error:
            logger.warning("could not write inspection provenance: %s", error)
        self._inspection = None
        return {**record, "marks_by_track": summary["marks_by_track"], "path": state["path"]}

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
                keep = self._EVENT_FILTER_STATES[self.t1_event_filter]
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
                extracted_labels=self._extracted_labels(frame_number),
                unextracted_color=self.unextracted_color,
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

        Measured on the FOUR real movies in `test_data/` -- `embryos_GBE_GAP43`
        (ratio 0.50, register; 0.529 re-measured 2026-09-13 after its latest
        correction), `embryos_GBE_OE_GAP43` (**0.121**, no: 2.79 px drift against
        a 23.0 px radius, measured Round 25 -- this docstring said "three movies"
        until then, and the fourth had never been measured), `AS_dorsal
        view_E-cad` (0.099, no) and `AS_LE_lateral view_E-cad` (0.092, no). An
        earlier version named `pupae` and `cp_dataset` instead; both are
        separate-image collections, so a cross-frame drift over them is
        meaningless.

        The threshold has still not been exercised close to 0.2 (observed: 0.53 /
        0.12 / 0.099 / 0.092; OE's 0.12 is the nearest), so `radius_fraction` is
        a lightly tested free parameter.
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
    parser.add_argument("--mask-folder")
    parser.add_argument("--image-folder")
    parser.add_argument("--output-dir")
    parser.add_argument("--classify", action="store_true", help="Run classify + leading-edge + boundary-split tracking")
    parser.add_argument("--shape", action="store_true", help="Also compute shape/tissue descriptors and exercise every feature mode")
    parser.add_argument("--n-clusters", default="2")
    parser.add_argument("--self-test", action="store_true",
                        help="run the checks that need no project, and exit")
    args = parser.parse_args()

    if args.self_test:
        # --- C1 (R20): the candidate list is a MERGE, not a truncation.
        # It used to accumulate four sources and take `[:3]`, so sources 3 and 4
        # were unreachable whenever source 2 yielded three -- and for an
        # `environment_flagged` item source 1 is empty by construction, making
        # the offered list *always exactly* the IoU ranking's top three.
        probe = EpiQuantAPI()
        probe._census = {"register": False, "drift_by_frame": None}
        probe.cells_info = None
        probe._census_crop = lambda *a, **k: ""

        height, width = 80, 400
        before = np.zeros((height, width), dtype=int)
        before[30:40, 30:40] = 1
        after = np.zeros((height, width), dtype=int)
        # Strips that CLIP the anchor's footprint (IoU > 0) but whose centroids
        # are far away -- a neighbour winning on overlap, which is the GBE shape.
        after[30:34, 30:200] = 11
        after[34:37, 30:250] = 12
        after[37:40, 30:300] = 13
        # The true successor: adjacent, NOT overlapping, nearest centroid.
        after[30:40, 42:52] = 7

        item = {"label": 1, "shown_label": 99, "frame": 5,
                "reference_label": None, "true_label": None}
        merged = probe._census_candidates(item, before, after, (0, height, 0, width))
        labels = [c["label"] for c in merged]
        reasons = {c["label"]: c["reason"] for c in merged}
        assert set(l for l in labels if reasons[l] == "overlap") <= {11, 12, 13}, reasons
        assert 7 in labels[:3], (labels, reasons)
        assert reasons[7] == "nearest", reasons
        print("C1: the IoU ranking returns three wrong labels and the centroid-nearest")
        print("  candidate is still offered on the first page -- a merge, not a truncation.")
        print("  offered:", [(c["label"], c["reason"]) for c in merged])

        # `shown_label` is never offered as an alternative to itself.
        assert 99 not in labels, labels

        # --- F (R22): `limit` is the only bound. Both sources ranked 4 deep
        # and the merge walked 4 levels, in two independent places, so on an
        # `environment_flagged` item (source 1 empty, sources 2-4 overlapping)
        # the list came out 3-6 long and `limit=12` never bound it.
        wide_after = after.copy()
        for n in range(20, 40):                     # 20 more cells to rank
            wide_after[70:74, (n - 20) * 8:(n - 20) * 8 + 6] = n
        wide = probe._census_candidates(item, before, wide_after, (0, height, 0, width))
        assert len(wide) == 12, len(wide)
        narrow = probe._census_candidates(item, before, wide_after, (0, height, 0, width), limit=5)
        assert len(narrow) == 5, len(narrow)
        print("F: with 24 distinct cells at f+1 the merge returns `limit` of them")
        print("  (12, and 5 when limit=5). Before Round 22 the two hardcoded 4s")
        print("  capped it below `limit`, so `limit` was a dead parameter.")

        # --- A1 (R23): LIST LENGTH IS NOT SEARCH DEPTH.
        # Three sources, twelve distinct labels each, nothing overlapping. The
        # pre-quota merge was depth-first, so with `limit=12` it stopped after
        # rank 3 and every source contributed exactly its first four -- the same
        # four it contributed before Round 22 widened the cuts from 4 to 12.
        # Round 22 reported the list growing from 4-5 to 12 and called that
        # "widened depth"; it was not. Run against that version this asserted
        # `{'overlap': 4, 'nearest': 4, 'pointed': 4}` and failed.
        #
        # The plan's own fixture asked for "source 3's fifth element". That is
        # not what shipped, and the reason is a measurement, not a compromise:
        # `nearest` is the source whose rank 4+ is provably useless here (its
        # 12th entry sits a median 49.8 px from the anchor on GBE, ~3.8 cell
        # radii), so `CANDIDATE_QUOTAS` holds it at 3 and spends the depth on
        # `overlap`/`coverage`, which run out at 1-5 entries. The assertion is
        # therefore "the sources no longer move in lockstep at 4", which is the
        # defect, rather than a rank on a particular source.
        three = [("overlap", [(100 + i, None) for i in range(12)]),
                 ("nearest", [(200 + i, None) for i in range(12)]),
                 ("pointed", [(300 + i, None) for i in range(12)])]
        merged3, depth3 = EpiQuantAPI._merge_candidate_sources(three, 12, set())
        labels3 = [c["label"] for c in merged3]
        assert len(merged3) == 12, len(merged3)
        assert depth3 != {"overlap": 4, "nearest": 4, "pointed": 4}, (
            "A1: every source still stops at rank 4 in lockstep -- %r" % (depth3,))
        assert max(depth3.values()) >= 5, depth3
        assert 104 in labels3, (labels3, depth3)   # overlap DID reach rank 5
        print("A1 (R23): 3 sources x 12 unique labels -> depth reached %s." % depth3)
        print("  Before the quotas this was {overlap: 4, nearest: 4, pointed: 4} --")
        print("  Round 22's [:4]->[:12] moved the list length, not the depth.")

        # A source that runs dry hands its quota back, so the list still fills.
        dry = [("overlap", [(100, None)]),
               ("coverage", [(400 + i, None) for i in range(12)]),
               ("nearest", [(200 + i, None) for i in range(12)]),
               ("pointed", [(300 + i, None) for i in range(12)])]
        merged4, depth4 = EpiQuantAPI._merge_candidate_sources(dry, 12, set())
        assert len(merged4) == 12, len(merged4)
        assert depth4["overlap"] == 1 and depth4["coverage"] >= 5, depth4
        print("  a source with one entry gives its quota back: %s" % depth4)

        # --- D2 / B2 / B3 (R25): the radius generator, the shared window, the
        # anchor view.
        import base64 as _b64, io as _io
        from PIL import Image as _Image

        def _png_size(data_url):
            return _Image.open(_io.BytesIO(_b64.b64decode(data_url.split(",")[-1]))).size

        r_before = np.zeros((260, 320), dtype=int)
        r_before[100:120, 140:160] = 1                     # anchor, centroid (149.5, 109.5)
        r_after = np.zeros((260, 320), dtype=int)
        r_after[100:120, 144:164] = 5                      # moved by the drift (4, 0): distance 0
        r_after[100:120, 174:194] = 6                      # 30 px
        r_after[100:120, 108:128] = 9                      # 36 px, the other side
        r_after[100:120, 200:220] = 7                      # 56 px -- beyond R = 3 x 13 = 39
        r_item = {"label": 1, "shown_label": 6, "frame": 5, "item_id": "r1", "track_id": 3,
                  "is_repeat": False, "reference_label": None, "true_label": None}
        r_drift, r_radius = (4.0, 0.0), 39.0
        radius_list = EpiQuantAPI._census_radius_candidates(r_item, r_before, r_after, r_drift, r_radius)
        assert [c["label"] for c in radius_list] == [5, 6, 9], radius_list
        assert [c["detail"] for c in radius_list] == sorted(c["detail"] for c in radius_list)
        assert [c["is_tracker_answer"] for c in radius_list] == [False, True, False], radius_list
        print("D2 (R25): radius mode offers every centroid within R of the registered")
        print("  anchor, nearest first, and keeps the tracker's answer (flagged):",
              [(c["label"], c["detail"], c["is_tracker_answer"]) for c in radius_list])

        r_window = EpiQuantAPI._census_anchor_window(r_before, r_after, r_item, radius_list,
                                                      r_drift, r_radius, 28)
        r_side = (r_window[1] - r_window[0], r_window[3] - r_window[2])
        assert min(r_side) >= 2 * r_radius + 2 * 28, (r_side, r_window)
        assert abs((r_window[0] + r_window[1] - 1) / 2 - 109.5) <= 1 and \
            abs((r_window[2] + r_window[3] - 1) / 2 - 153.5) <= 1, r_window
        print("B2 (R25): the shared window is %dx%d px, centred on the registered anchor"
              % r_side, "(floor 2R + 2*pad = %d)." % (2 * r_radius + 2 * 28))

        rprobe = EpiQuantAPI()
        rprobe._load_frame = lambda frame: None
        rprobe.tracking_data = None
        rprobe._census = {"queue": [r_item], "masks": {5: r_before, 6: r_after},
                          "register": False, "drift_by_frame": None,
                          "search_drift_by_frame": {5: r_drift}, "cell_radius_px": 13.0,
                          "candidates": {}, "windows": {}, "offered_mode": {}}
        shown_on = rprobe.get_census_item(0, 28, True)
        shown_off = rprobe.get_census_item(0, 28, False)
        labels_on = [c["label"] for c in shown_on["candidates"]]
        labels_off = [c["label"] for c in shown_off["candidates"]]
        assert labels_on == labels_off == [5, 6, 9], (labels_on, labels_off)
        assert shown_on["candidate_search"]["window"] == shown_off["candidate_search"]["window"]
        sizes = {_png_size(shown_on["before"]), _png_size(shown_on["after"])} | \
            {_png_size(c["image"]) for c in shown_on["candidates"]}
        assert len(sizes) == 1, sizes
        assert any(a["image"] != b["image"] for a, b in
                   zip(shown_on["candidates"], shown_off["candidates"])), "outline must show"
        rerender = rprobe.get_census_candidate_images(0, False)
        assert rerender["labels"] == labels_on, rerender["labels"]
        assert rerender["images"] == [c["image"] for c in shown_off["candidates"]]
        print("B3 (R25): anchor view on and off offer the SAME labels %s; only the" % labels_on)
        print("  tiles differ, the blink and every tile share one %s px size, and" % (sizes.pop(),))
        print("  the render-only call re-draws without re-running the generator.")

        # The ranked version is untouched and still selectable.
        assert rprobe.set_candidate_settings("ranked")["mode"] == "ranked"
        ranked_item = rprobe.get_census_item(0, 28, True)
        assert ranked_item["candidate_mode"] == "ranked", ranked_item["candidate_mode"]
        assert 6 not in [c["label"] for c in ranked_item["candidates"]], "ranked still excludes shown"
        assert rprobe.set_candidate_settings("nonsense").get("error")
        print("D1 (R25): 'ranked' is still there and still excludes the tracker's answer.")

        # --- C2: pages are ceil(n / 3) on the ACTUAL list, never padded.
        for count, expected in ((1, 1), (3, 1), (4, 2), (5, 2), (6, 2), (7, 3), (12, 4)):
            assert -(-count // 3) == expected, (count, expected)
        print("C2: %d candidates -> %d page(s); 5 gives 2 pages, not a padded 3."
              % (5, -(-5 // 3)))

        # --- S (R24): a -1 in a `Cells_info` column is a failed measurement
        # and leaves the PCA matrix; a -1 in any other table is a value and
        # stays. Before Round 24 `dropna` was the only filter, so the first row
        # below entered StandardScaler with a -1 perimeter.
        fp = EpiQuantAPI()
        fp.cells_info = pd.DataFrame({
            "Cell Index": ["Cell1_1", "Cell1_2", "Cell1_3", "Cell1_4"],
            "Cell Area": [100.0, 120.0, 140.0, 160.0],
            "Cell Perimeter": [-1.0, 44.0, 48.0, 52.0],
        })
        fp.cells_shape = pd.DataFrame(
            {"signed_shape": [0.5, -1.0, 0.2, 0.1]},
            index=pd.Index(["Cell1_1", "Cell1_2", "Cell1_3", "Cell1_4"], name="Cell Index"),
        )
        fp.cells_tissue = None
        kept, _ = fp._feature_frame(["basic"], ["Cell Area", "Cell Perimeter", "signed_shape"])
        assert list(kept.index) == ["Cell1_2", "Cell1_3", "Cell1_4"], list(kept.index)
        assert kept.loc["Cell1_2", "signed_shape"] == -1.0, kept
        print("S (R24): Cells_info -1 sentinel dropped (Cell1_1); a -1 in the shape")
        print("  table is a value and kept (Cell1_2).")

        # --- O (R24): a project extracted before the two basic columns came
        # back. Its basic family must stay ready and run the same default
        # columns; the restored ones are reported `unavailable`, not `missing`
        # (which is what disables a whole family). Before this, basic went
        # not-ready and its boxes disappeared under "run shape descriptors first".
        restored = ["Fitted Ellipse Angle", "Circularity"]
        old = EpiQuantAPI()
        rng_o = np.random.default_rng(0)
        old.cells_info = pd.DataFrame(
            {c: rng_o.uniform(5, 50, 6) for c in data_input.CELLS_INFO_COLUMNS if c not in restored})
        old.cells_info.insert(0, "Cell Index", [f"Cell1_{i}" for i in range(1, 7)])
        old.cells_shape = old.cells_tissue = None
        basic = old.get_feature_options()["families"]["basic"]
        assert basic["ready"] and basic["missing"] == [], basic
        assert sorted(basic["unavailable"]) == sorted(restored), basic["unavailable"]
        default_old, _ = old._feature_frame(["basic"], None)
        assert list(default_old.columns) == pca_kmeans.family_columns(["basic"]), list(default_old.columns)
        try:
            old._feature_frame(["basic"], ["Cell Area", "Circularity"])
            raise AssertionError("an absent column must be refused")
        except ValueError as exc:
            assert "Re-run Extract" in str(exc), str(exc)
        fresh = EpiQuantAPI()
        fresh.cells_info = old.cells_info.assign(**{c: 1.0 for c in restored})
        fresh.cells_shape = fresh.cells_tissue = None
        assert fresh.get_feature_options()["families"]["basic"]["unavailable"] == []
        print("O (R24): a 15-column pre-R24 table keeps basic ready with the default")
        print("  columns; the restored two are `unavailable` (disabled), and asking for one")
        print("  says to re-run Extract rather than to run shape descriptors.")

        # --- EFD (2026-09-19): Classify must PCA the normalized shape, not the
        # raw coefficients. Fixture: the same 12 shapes twice, the second copy
        # rotated, rescaled, started elsewhere and traced the other way. If the
        # raw columns reached PCA the two copies would differ by O(1).
        import pyefd
        rng_e = np.random.default_rng(3)
        tt = np.linspace(0, 2 * np.pi, 300, endpoint=False)
        shapes = []
        for _ in range(12):
            r = 1 + 0.15 * rng_e.standard_normal(4) @ np.cos(np.outer([2, 3, 4, 5], tt) + rng_e.uniform(0, 6, (4, 1)))
            shapes.append(np.column_stack([rng_e.uniform(1.5, 3) * r * np.cos(tt), r * np.sin(tt)]) * 20)

        def efd_table(transform):
            rows = []
            for i, sh in enumerate(shapes):
                sh = transform(i, sh)
                rows.append(pyefd.elliptic_fourier_descriptors(np.vstack([sh, sh[:1]]), order=10).ravel())
            return pd.DataFrame(rows, columns=pca_kmeans.efd_preset(10), index=[f"Cell1_{i}" for i in range(12)])

        def moved(i, sh):
            a = rng_e.uniform(0, 2 * np.pi)
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            return (np.roll(sh, rng_e.integers(300), axis=0) @ R.T * rng_e.uniform(0.5, 2))[::-1]

        feats = []
        for transform in (lambda i, sh: sh, moved):
            e_api = EpiQuantAPI()
            e_api.cells_info = pd.DataFrame(
                {"Cell Index": [f"Cell1_{i}" for i in range(12)],
                 **{c: rng_e.uniform(5, 50, 12) for c in data_input.CELLS_INFO_COLUMNS}})
            e_api.cells_shape = efd_table(transform)
            e_api.cells_tissue = None
            frame_e, mode_e = e_api._feature_frame(["efd"], None)
            assert mode_e == "none" and frame_e.shape == (12, 40), (mode_e, frame_e.shape)
            feats.append(frame_e.to_numpy())
        drift = np.abs(feats[0] - feats[1]).max()
        raw_drift = np.abs(efd_table(lambda i, sh: sh).to_numpy() - efd_table(moved).to_numpy()).max()
        assert drift < 1e-9, f"EFD Classify features depend on pose: {drift:.3e}"
        assert raw_drift > 1.0, "fixture did not move the raw coefficients"
        assert np.allclose(np.sqrt((feats[0][:, :4] ** 2).sum(axis=1)), 1.0), "harmonic 1 must be unit norm"
        e_api.cells_shape.iloc[0, 5] = np.nan
        assert len(e_api._feature_frame(["efd"], None)[0]) == 11, "a cell with a NaN coefficient must drop"
        print(f"EFD (2026-09-19): Classify features are pose-invariant (max drift {drift:.1e};")
        print(f"  the raw columns moved by {raw_drift:.1f}), unit harmonic-1, NaN rows dropped.")

        print("\nAll api self-tests passed.")
        raise SystemExit(0)

    if not (args.mask_folder and args.image_folder and args.output_dir):
        parser.error("--mask-folder, --image-folder and --output-dir are required "
                     "unless --self-test is given")

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

            # --- Round 16 B: verify a plan before Apply, and prove the live
            # table is untouched by verifying. That last one is the safety
            # property of an accept-and-revert loop and is not optional.
            print("\n== plan verification (R16 B) ==")
            plan, plan_track = None, None
            for candidate in track_numbers[:80]:
                trial = api.compute_confirmed_identity_moves(
                    frame_range["min_frame"], candidate, 20, False)
                if trial.get("moves"):
                    plan, plan_track = trial, candidate
                    break
            if plan is None:
                print("  no track produced a non-empty plan on this movie -- nothing to verify")
            else:
                before_table = api.tracking_data.copy(deep=True)
                report = api.verify_confirmed_identity_moves(plan_track, plan["moves"])
                assert not report.get("error"), report
                assert api.tracking_data.astype(str).equals(before_table.astype(str)), \
                    "verifying a plan must not mutate the live tracking table"
                print(f"  track #{plan_track}: {len(plan['moves'])} move(s) -- "
                      f"{len(report['double_claimed'])} double-claim, "
                      f"{len(report['orphaned'])} orphan, {len(report['truncated'])} truncated, "
                      f"{len(report['evicts'])} eviction(s), "
                      f"{len(report['takes_from'])} take-from-another-track")
                for move in report["moves"]:
                    print(f"    frame {move['frame']}: {move['distance']}px of a "
                          f"{move['gate_px']}px gate ({move['gate_fraction']})")
                delta = report["consistency"]
                if delta:
                    print(f"  consistency delta {delta['before_unconfirmed']} -> "
                          f"{delta['after_unconfirmed']} of {delta['n']} links unconfirmed "
                          f"({delta['delta']:+d}), masks "
                          f"{'registered' if delta['registered'] else 'unregistered'} both sides. "
                          "A consistency measure, never an accuracy.")
                print("  live table byte-identical after verification.")

            # --- Round 16 C2: the two session modes cannot cross. Asserted,
            # not documented: ground truth collected against a tracking run is
            # not valid for a run it corrected.
            print("\n== census / correction modes (R16 C2) ==")
            gt_path = os.path.join(api.output_dirs["cell_track_output"], "ground_truth_links.csv")
            started = api.start_census(mode="census", seed=0, labeller="harness")
            if started.get("error"):
                print("  census unavailable:", started["error"])
            else:
                table_before = api.tracking_data.copy(deep=True)
                assert api.queue_census_edit(0, 1).get("error"), "a census session must not edit"
                assert api.apply_census_edits().get("error"), "a census session must not apply"
                api.submit_census_verdict(0, "correct", None, 1200.0)
                assert api.tracking_data.astype(str).equals(table_before.astype(str)), \
                    "a census session must never mutate the tracking table"

                # Round 23 D: a "wrong" verdict with a tile chosen must record
                # WHICH ranking proposed that cell and where it sat. Exercised
                # here and not only in `census.py --self-test`, because the
                # lookup lives in this file and reads a cache `get_census_item`
                # fills -- a unit test on `append_verdict` cannot see that wire.
                shown = api.get_census_item(1)
                offered = shown.get("candidates") or []
                geometry = shown.get("candidate_search") or {}
                print(f"  candidates ({shown.get('candidate_mode')}): {len(offered)} within "
                      f"R = {geometry.get('radius_px')} px (k {geometry.get('k')} x "
                      f"{geometry.get('cell_radius_px')} px cell radius), drift "
                      f"{geometry.get('drift_px')}, shared window {geometry.get('window_px')} px.")
                print(f"  candidate depth reached per source: {shown.get('candidate_depth')} "
                      f"({len(offered)} offered) -- depth, not list length.")
                if offered:
                    pick = offered[min(1, len(offered) - 1)]
                    wrote = api.submit_census_verdict(1, "wrong", pick["label"], 2400.0)
                    assert wrote.get("chosen_reason") == pick["reason"], (wrote, pick)
                    assert wrote.get("chosen_rank") == pick["rank"], (wrote, pick)
                    with open(gt_path, newline="", encoding="utf-8") as handle:
                        last = list(csv.DictReader(handle))[-1]
                    assert last["chosen_reason"] == pick["reason"], last
                    assert last["chosen_rank"] == str(pick["rank"]), last
                    assert last["register"] in ("0", "1"), last
                    print(f"  a chosen tile is recorded with its source and rank: "
                          f"{last['chosen_reason']} at rank {last['chosen_rank']}, "
                          f"register={last['register']}.")
                    assert api.tracking_data.astype(str).equals(table_before.astype(str)), \
                        "a census session must never mutate the tracking table"
                gt_size = os.path.getsize(gt_path)
                print(f"  census mode: {started['n_real']} links, "
                      f"{started['n_decoys']} decoys, {started['n_repeats']} repeats; "
                      "edits refused; tracking table unchanged.")

                fixing = api.start_census(mode="correct", seed=0, labeller="harness")
                assert fixing["n_decoys"] == 0 and fixing["n_repeats"] == 0, fixing
                assert api.submit_census_verdict(0, "correct", None, 1.0).get("error"), \
                    "a correction session must not write ground truth"
                first = api.get_census_item(0)
                # Not `candidates[0]`: radius mode keeps the tracker's own answer,
                # often at rank 1, and "correcting" to it changes nothing (R25).
                alternative = next(c for c in first["candidates"] if not c.get("is_tracker_answer"))
                assert api.queue_census_edit(0, first["shown_label"]).get("error"), \
                    "queueing the tracker's own answer must be refused as a no-op"
                api.queue_census_edit(0, alternative["label"])
                preview = api.preview_census_edits()
                snapshot = api.tracking_data.copy(deep=True)
                applied = api.apply_census_edits()
                assert applied.get("applied") == 1, applied
                assert not api.tracking_data.astype(str).equals(snapshot.astype(str)), \
                    "applying a correction must actually change the table"
                assert os.path.getsize(gt_path) == gt_size, \
                    "a correction session must not append to ground_truth_links.csv"
                api.undo_census_edits()
                assert api.tracking_data.astype(str).equals(snapshot.astype(str)), \
                    "undo must restore the snapshot exactly"
                print(f"  correction mode: no decoys/repeats; verdicts refused; "
                      f"{preview['pending']} pending applied as one batch; "
                      "ground truth untouched; undo restores byte-identically.")

            # --- Round 16 E: within +/-1 of the anchor every crop must share
            # ONE window, or the blink compares two differently framed images.
            print("\n== frame stepper (R16 E) ==")
            strip = api.get_track_filmstrip(track_numbers[0])
            if strip.get("error") or len(strip["held"]) < 3:
                print("  no track with enough frames to step")
            else:
                anchor = strip["held"][len(strip["held"]) // 2]
                sizes, notes = [], []
                for offset in (-1, 0, 1, 3):
                    step = api.get_track_step_frame(track_numbers[0], anchor, anchor + offset)
                    if step.get("error"):
                        continue
                    payload = base64.b64decode(step["image"].split(",", 1)[1])
                    from PIL import Image as _Image
                    from io import BytesIO as _BytesIO
                    sizes.append((offset, _Image.open(_BytesIO(payload)).size, step["recentred"]))
                shared = [size for offset, size, moved in sizes if not moved]
                assert len(set(shared)) <= 1, sizes
                print(f"  anchor frame {anchor}: {sizes}")
                print("  every crop within +/-1 shares one window; beyond that it re-centres "
                      "and says so.")

            # --- Round 17 C3: `settled only` must leave NO transient T1 cell
            # fill on the frame. Asserted by counting painted pixels, not by
            # eye -- the defect was invisible in the dev mock and survived
            # every previous verification because nobody counted.
            print("\n== T1 event filter gates the fill (R17 C3) ==")
            from PIL import Image as _Image
            from io import BytesIO as _BytesIO
            t1_frame = api.get_frame_numbers()[len(api.get_frame_numbers()) // 2]
            t1_labels = (api._load_frame(t1_frame) or (None,))[0]
            events = api.get_t1_events(t1_frame)
            if t1_labels is None or "error" in events or not events.get("counts"):
                print("  no T1 events on this frame -- nothing to filter")
            else:
                settled = {int(l) for l in events["counts_persistent"]}
                unjudgeable = {int(l) for l in events.get("counts_unjudgeable", {})}
                transient = {int(l) for l in events["counts"]} - settled - unjudgeable
                base_png = np.asarray(_Image.open(_BytesIO(base64.b64decode(
                    api.get_raw_frame_image(t1_frame).split(",", 1)[1]))).convert("RGB"))
                for mode in ("all", "persistent"):
                    api.set_t1_event_filter(mode)
                    drawn = {label for label, entry in api._dynamics_states(t1_frame).items()
                             if entry["event"] == "t1"}
                    overlay = np.asarray(_Image.open(_BytesIO(base64.b64decode(
                        api.get_trajectory_overlay(
                            track_numbers[0], t1_frame, color_mode="dynamics",
                            show_nematic_axes=True).split(",", 1)[1]))).convert("RGB"))
                    # A cell is FILLED when most of its pixels changed; a few
                    # changed pixels are a neighbouring settled event's axis
                    # line crossing it, which is not a fill.
                    filled = set()
                    for label in transient:
                        mask = t1_labels == label
                        if not mask.any():
                            continue
                        if (overlay[mask] != base_png[mask]).any(axis=1).mean() > 0.5:
                            filled.add(int(label))
                    print(f"  {mode:11s}: {len(drawn):3d} T1 cells drawn, "
                          f"{len(filled)} of {len(transient)} transient cells filled")
                    if mode == "persistent":
                        assert not filled, (
                            "'settled only' still fills transient T1 cells: %s" % sorted(filled))
                        assert not (drawn & transient), sorted(drawn & transient)
                api.set_t1_event_filter("persistent")
                print(f"  ({len(settled)} settled / {len(transient)} transient / "
                      f"{len(unjudgeable)} unjudgeable; raw counts still reported)")

            # --- Round 18 C: does a correction actually REACH the record?
            # Not "the batch API returned success" -- the frame column, the
            # untouched columns, the undo, and the file on disk.
            print("\n== correction write path, end to end (R18 C) ==")
            started = api.start_census(mode="correct", seed=0, labeller="harness")
            if started.get("error"):
                print("  no correction queue on this movie:", started["error"])
            else:
                item = api.get_census_item(0)
                # The first candidate that is NOT the tracker's own answer:
                # radius mode keeps that answer in the list, and "correcting" a
                # link to the cell it already points at writes nothing (R25).
                chosen = next(c["label"] for c in item["candidates"]
                              if not c.get("is_tracker_answer"))
                target_frame = item["frame"] + 1
                target_column = f"Frame{target_frame}"
                track = item["track_id"]
                api.queue_census_edit(0, chosen)

                before_table = api.tracking_data.copy(deep=True)
                result = api.apply_census_edits()
                assert result.get("applied") == 1, result
                after = api.tracking_data

                # (2) the cell for that track at that frame is now the chosen one
                row = after.index[after[after.columns[0]] == track][0]
                assert str(after.at[row, target_column]) == f"Cell{target_frame}_{chosen}", (
                    after.at[row, target_column], chosen)

                # (3) nothing outside the declared range moved, and every track
                # that lost a claim is named in the return value.
                assert result["frames_asked"] == [target_frame], result["frames_asked"]
                reported = {(c["frame"], c["from_track_id"]) for c in result["claims_removed"]}
                for record in result["displaced_list"]:
                    reported.add((target_frame, int(record["evicted_track_id"])))

                # Compare BY TRACK ID, not positionally: an eviction appends a
                # row, so the two tables need not be the same length.
                def claims(table):
                    id_col = table.columns[0]
                    out = {}
                    for position in table.index:
                        holder = int(table.at[position, id_col])
                        for column in (c for c in table.columns if c.startswith("Frame")):
                            value = table.at[position, column]
                            if pd.notna(value):
                                out[(int(column[len("Frame"):]), holder)] = str(value)
                    return out

                was, now = claims(before_table), claims(after)
                # A change beyond the asked frame is legitimate ONLY as part of
                # an eviction the return value names -- that is the one
                # operation with a reach wider than the edit.
                evicted = {int(r["evicted_track_id"]) for r in result["displaced_list"]}
                evicted |= {int(r["new_track_id"]) for r in result["displaced_list"]}
                unreported = []
                for key in set(was) | set(now):
                    if was.get(key) == now.get(key):
                        continue
                    frame, holder = key
                    if frame != target_frame and holder not in evicted:
                        unreported.append(("beyond the asked frame", key))
                    elif frame == target_frame and holder != track \
                            and (frame, holder) not in reported and holder not in evicted:
                        unreported.append(("lost a claim", key))
                assert not unreported, f"changes nothing in the return value accounts for: {unreported}"
                assert set(result["beyond_asked"]) == set(result["frames_written"]) - {target_frame}, result
                print(f"  track #{track} at frame {target_frame} -> Cell{target_frame}_{chosen}; "
                      f"{len(result['claims_removed'])} claim(s) removed, "
                      f"{len(result['displaced_list'])} eviction(s); "
                      f"asked {result['frames_asked']}, actually wrote {result['frames_written']}.")
                if result["beyond_asked"]:
                    print(f"  frames {result['beyond_asked']} were rewritten too -- an eviction "
                          "relocates a colliding track's whole future, and the return value "
                          "now says so instead of claiming the edited frame alone.")

                # (5) the file on disk matches memory -- the step that catches
                # "the UI updated but nothing persisted".
                api.save_tracking()
                on_disk = pd.read_excel(os.path.join(
                    api.output_dirs["cell_track_output"], "all_cell_tracking",
                    "all_cell_merged_tracking_results.xlsx"))
                assert on_disk.shape == api.tracking_data.shape, (on_disk.shape, api.tracking_data.shape)
                assert on_disk.fillna("~").astype(str).equals(
                    api.tracking_data.reset_index(drop=True).fillna("~").astype(str)), \
                    "the saved .xlsx does not match the in-memory table"
                print("  saved .xlsx matches the in-memory table exactly.")

                # (4) one undo restores byte-identically
                api.undo_census_edits()
                assert api.tracking_data.astype(str).equals(before_table.astype(str)), \
                    "undo must restore the table byte-identically"
                api.save_tracking()
                print("  one undo restores the table byte-identically, and re-saves it.")

            api.save_tracking()
            print("Saved tracking.")

    print("\nAll smoke-test steps completed without raising.")
