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
import common
import data_input
import manual_correction_classification
import manual_correction_tracking
import pca_kmeans
import rendering
import segmentation_qc
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

        self.matched_color = "#ffd633"
        self.unmatched_color = "#3388ff"
        self.highlight_color = "#ff3333"
        self.border_color = "#ff00ff"
        self.lifecycle_colors = {"lost": "#ff3b30", "new": "#34c759", "both": "#ffffff"}

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
            return {jid: scored[pair] for jid, pair in mapping.items() if pair in scored}
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
        """`metric`'s LUT window for this project, fitted once and cached."""
        if metric not in self._qc_display_cache:
            sampled = self._qc_fit_frames()
            pooled = {}
            for offset, frame in enumerate(sampled):
                # Keys collide across frames (a label id is per-frame), so
                # namespace them -- only the value distribution matters here.
                for key, value in self._qc_scores(frame, metric).items():
                    pooled[(offset, key)] = value
            self._qc_display_cache[metric] = segmentation_qc.fit_display_window(metric, pooled)
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
        display = self._qc_display(metric)
        overlay = rendering.render_score_overlay(
            label_array, base_image, self._qc_scores(frame_number, metric), lut,
            vmin=display["vmin"], vmax=display["vmax"],
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
            display = self._qc_display(metric)
            image = rendering.render_score_overlay(
                label_array, base_image, self._qc_scores(frame_number, metric), lut,
                vmin=display["vmin"], vmax=display["vmax"],
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
        the metrics don't share a scale."""
        flag_below = self._qc_display(metric)["flag_below"]
        return [
            {"frame": f, **segmentation_qc.summarize(self._qc_scores(f, metric), flag_below)}
            for f in self.get_frame_numbers()
        ]

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
        random_state: Optional[int] = None,
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
                cells_info_for_tracking, self.output_dirs["cell_track_output"], progress_cb=self._progress_cb("track"),
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

        frames = []
        seg_files = data_input.list_segmentation_files(self.npy_folder or "")
        for filename, frame in data_input.resolve_frame_numbers(seg_files).items():
            path = os.path.join(self.npy_folder, filename)
            mtime = os.path.getmtime(path)
            manual = 0
            try:
                flags = np.load(path, allow_pickle=True).item().get("ismanual")
                manual = int(np.asarray(flags).sum()) if flags is not None else 0
            except Exception:
                logger.debug("Could not read ismanual from %s", path)
            frames.append({
                "frame": frame,
                "stale": tracked_at is not None and mtime > tracked_at,
                "manual_cells": manual,
            })
        frames.sort(key=lambda f: f["frame"])
        stale = [f["frame"] for f in frames if f["stale"]]
        return {
            "tracked": tracked_at is not None,
            "frames": frames,
            "stale_frames": stale,
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
            cb(f"Re-tracking frames {start_frame}-{end_frame}...", 10)
            result = cell_tracking.retrack_frame_range(
                self.cells_info.set_index("Cell Index"), self.tracking_data, start_frame, end_frame,
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
        previous or next frame (Tissue Analyzer's Fig. 3g/h respectively).
        Empty dict if the compared frame doesn't exist."""
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

        overlay = rendering.render_tracking_overlay(
            label_array, base_image, frame_number, self.tracking_data, self.matched_color, self.unmatched_color,
            label_cells=show_labels, track_color_mode=track_color_mode, identity_palette=identity_palette,
            label_font_size=label_font_size,
            environment_scores=environment_scores, environment_lut=environment_lut,
            lifecycle_states=lifecycle_states, lifecycle_colors=self.lifecycle_colors,
            tint_other_cells=tint_other_cells,
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
        restrict_to_unmatched: bool = True,
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
            candidates = manual_correction_tracking.suggest_match_candidates(
                self.cells_info, self.tracking_data, int(frame_number), cell_index,
                lambda f: (self._load_frame(f) or (None, None, None))[0],
                top_k=int(top_k),
            )
            return {"candidates": candidates}
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

    def get_trajectory_overlay(self, track_number: int, frame_number: int, render_mode: str = "mask") -> Optional[str]:
        if self.tracking_data is None:
            return None
        cell_ids = trajectory_visualization.get_track_cell_ids(self.tracking_data, int(track_number))
        target_prefix = f"Cell{frame_number}_"
        match = next((cid for cid in cell_ids if isinstance(cid, str) and cid.startswith(target_prefix)), None)
        if match is None:
            return None
        cell_id_num = int(match.split("_")[-1])

        loaded = self._load_frame(frame_number)
        if loaded is None:
            return None
        fill_labels, boundary_labels, base_image = loaded
        label_array = boundary_labels if render_mode == "outline" else fill_labels
        overlay = rendering.render_cell_highlight(label_array, base_image, cell_id_num, self.highlight_color)
        return rendering.encode_png_base64(overlay)


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

            api.save_tracking()
            print("Saved tracking.")

    print("\nAll smoke-test steps completed without raising.")
