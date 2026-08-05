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

        self.tracking_data: Optional[pd.DataFrame] = None
        self.original_tracking_data: Optional[pd.DataFrame] = None

        self.matched_color = "#ffd633"
        self.unmatched_color = "#3388ff"
        self.highlight_color = "#ff3333"

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

    def _load_frame(self, frame_number: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Returns `(outlines, base_image)` for `frame_number`, or `None` if
        the frame's `.npy` can't be found."""
        npy_path = self._npy_path_for_frame(frame_number)
        if npy_path is None:
            return None
        dat = np.load(npy_path, allow_pickle=True).item()
        base_image = rendering.load_base_image(dat, self.image_folder, frame_number)
        return dat["outlines"], base_image

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

    def get_raw_frame_image(self, frame_number: int) -> Optional[str]:
        loaded = self._load_frame(frame_number)
        if loaded is None:
            return None
        _outlines, base_image = loaded
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

        return {
            "cluster_sizes": clustering_df["Cell_Type"].value_counts().to_dict(),
            "k_used": int(clustering_df["Cluster_Label"].nunique()),
        }

    def set_cluster_colors(self, color_map: dict) -> None:
        self.cluster_color_map = {int(k): v for k, v in color_map.items()}

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
            }
            for (label, cell_type), count in counts.items()
        ]
        return sorted(legend, key=lambda entry: entry["cluster_label"])

    def get_classification_overlay(self, frame_number: int) -> Optional[str]:
        if self.clustering_data is None:
            return None
        loaded = self._load_frame(frame_number)
        if loaded is None:
            return None
        outlines, base_image = loaded
        overlay = rendering.render_classification_overlay(
            outlines, base_image, frame_number, self.clustering_data, self.cluster_color_map,
        )
        return rendering.encode_png_base64(overlay)

    def get_cell_at_pixel(self, frame_number: int, x: int, y: int) -> Optional[dict]:
        loaded = self._load_frame(frame_number)
        if loaded is None:
            return None
        outlines, _base_image = loaded
        cell_id = rendering.cell_id_at_pixel(outlines, int(x), int(y))
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
            outlines, _base_image = loaded
            self.cells_info = manual_correction_classification.compute_leading_edge(
                outlines, frame_number, self.cells_info, self.clustering_data, threshold=threshold,
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

    def get_tracking_overlay(self, frame_number: int) -> Optional[str]:
        if self.tracking_data is None:
            return None
        loaded = self._load_frame(frame_number)
        if loaded is None:
            return None
        outlines, base_image = loaded
        overlay = rendering.render_tracking_overlay(
            outlines, base_image, frame_number, self.tracking_data, self.matched_color, self.unmatched_color,
        )
        return rendering.encode_png_base64(overlay)

    def get_tracking_cell_at_pixel(self, frame_number: int, x: int, y: int) -> Optional[dict]:
        loaded = self._load_frame(frame_number)
        if loaded is None:
            return None
        outlines, _base_image = loaded
        cell_id = rendering.cell_id_at_pixel(outlines, int(x), int(y))
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

    def reassign_track(self, frame_number: int, old_id: int, new_id: int) -> dict:
        try:
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

    def get_trajectory_overlay(self, track_number: int, frame_number: int) -> Optional[str]:
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
        outlines, base_image = loaded
        overlay = rendering.render_cell_highlight(outlines, base_image, cell_id_num, self.highlight_color)
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
