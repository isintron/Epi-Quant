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

import cell_tracking
import common
import data_input
import manual_correction_classification
import manual_correction_tracking
import pca_kmeans
import rendering
import trajectory_visualization

logger = logging.getLogger(__name__)


class EpiQuantAPI:
    def __init__(self):
        self.window = None

        self.mask_folder: Optional[str] = None
        self.image_folder: Optional[str] = None
        self.npy_folder: Optional[str] = None
        self.output_dirs: Optional[dict] = None

        self.cells_info: Optional[pd.DataFrame] = None
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

        self.matched_color = "#ffd633"
        self.unmatched_color = "#3388ff"
        self.highlight_color = "#ff3333"
        self.border_color = "#ff00ff"

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
        for filename in data_input.list_segmentation_files(self.npy_folder):
            if data_input._extract_frame_number(filename) == frame_number:
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
        (same cell IDs, verified against test_data). `convert_tiff_masks_to_npy`'s
        synthetic dicts have no `'masks'` key because their `'outlines'` is
        already a filled labeled mask (`load_tiff_mask`'s output) -- so
        falling back to `'outlines'` when `'masks'` is absent is correct for
        that path too.

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
        return {
            "extraction": os.path.isfile(cells_info_path),
            "classification": os.path.isfile(clustering_path),
            "tracking": os.path.isfile(tracking_path),
        }

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
            return sorted({
                data_input._extract_frame_number(f) for f in data_input.list_segmentation_files(self.npy_folder)
            })
        return []

    def get_raw_frame_image(self, frame_number: int) -> Optional[str]:
        loaded = self._load_frame(frame_number)
        if loaded is None:
            return None
        _fill_labels, _boundary_labels, base_image = loaded
        from PIL import Image
        return rendering.encode_png_base64(Image.fromarray(base_image))

    # ---- Classify (skippable) --------------------------------------------

    def get_palette_presets(self) -> dict[str, list[str]]:
        return rendering.get_palette_presets()

    def run_classification(
        self, n_clusters: Union[int, str] = "auto", k_range: tuple = (2, 9), random_state: Optional[int] = None,
    ) -> dict:
        filtered = self.cells_info.set_index("Cell Index").loc[:, data_input.ML_FEATURE_COLUMNS]
        pca_df = pca_kmeans.compute_pca(filtered, output_dir=self.output_dirs["cell_classification_output"])

        if isinstance(n_clusters, str) and n_clusters != "auto":
            n_clusters = int(n_clusters)

        clustering_df = pca_kmeans.cluster_cells_kmeans(
            pca_df,
            output_dir=self.output_dirs["cell_classification_output"],
            n_clusters=n_clusters,
            k_range=tuple(k_range),
            random_state=random_state,
        )
        clustering_df["modified"] = False
        self.clustering_data = clustering_df
        self._border_cell_cache.clear()

        return {
            "cluster_sizes": clustering_df["Cell_Type"].value_counts().to_dict(),
            "k_used": int(clustering_df["Cluster_Label"].nunique()),
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

    def set_track_colors(self, matched_color: str, unmatched_color: str) -> None:
        self.matched_color = matched_color
        self.unmatched_color = unmatched_color

    def get_tracking_overlay(
        self, frame_number: int, render_mode: str = "mask", show_labels: bool = True,
        track_color_mode: str = "uniform", identity_palette: str = "golden_angle",
    ) -> Optional[str]:
        if self.tracking_data is None:
            return None
        loaded = self._load_frame(frame_number)
        if loaded is None:
            return None
        fill_labels, boundary_labels, base_image = loaded
        label_array = boundary_labels if render_mode == "outline" else fill_labels
        overlay = rendering.render_tracking_overlay(
            label_array, base_image, frame_number, self.tracking_data, self.matched_color, self.unmatched_color,
            label_cells=show_labels, track_color_mode=track_color_mode, identity_palette=identity_palette,
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
                manual_correction_tracking.merge_tracks_from_frame(
                    self.tracking_data, int(frame_number), int(old_id), int(new_id),
                )
            else:
                manual_correction_tracking.update_tracking_data(
                    self.tracking_data, self.original_tracking_data, int(frame_number), int(old_id), int(new_id),
                )
            return {"success": True}
        except ValueError as e:
            return {"error": str(e)}

    def add_tracking_id(self, frame_number: int, new_index: int, cell_id_str: str) -> dict:
        try:
            manual_correction_tracking.add_tracking_id(
                self.tracking_data, int(frame_number), int(new_index), cell_id_str,
            )
            return {"success": True}
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

    if args.classify:
        print("\n== classification ==")
        n_clusters_arg = args.n_clusters if args.n_clusters == "auto" else int(args.n_clusters)
        print(api.run_classification(n_clusters=n_clusters_arg, random_state=42))

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
            api.save_tracking()
            print("Saved tracking.")

    print("\nAll smoke-test steps completed without raising.")
