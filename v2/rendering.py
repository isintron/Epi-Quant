"""Overlay rendering for the pywebview UI (v2/webview/): compositing colored
cell overlays, click-to-cell hit-testing, and PNG encoding.

None of this logic exists anywhere else in `v2/` -- it's the "rendering half"
that `manual_correction_classification.py`/`manual_correction_tracking.py`
deliberately left out when those were extracted from v1's PyQt5/matplotlib
drawing code (`gui_comparison.py:391-556`, `gui_cell_track.py:436-550`).
It's written fresh, informed by (not copied from) that v1 code:

- v1 hardcoded exactly two fill colors (yellow `(255,255,0)` / blue
  `(0,0,255)`) for the two classification clusters, and a matched/unmatched
  yellow/blue pair for tracking correction. Here every function takes an
  explicit `color_map`/color argument instead, so the UI can offer arbitrary
  per-cluster HEX colors and preset palettes.
- v1's cell-click handling hit-tested rendered matplotlib text labels
  (`event.inaxes.texts`). `cell_id_at_pixel` below does a direct O(1) array
  lookup on the labeled `outlines` mask instead -- exact-pixel, and doesn't
  require the frontend to reimplement any hit-testing; it only needs to send
  pixel coordinates computed from the displayed image's *natural* dimensions
  (not its CSS-rendered size).
- v1's image-loading fallback (when a `.npy`'s `'outlines'` dict has no
  `'img'` key) only searched for `.jpeg`/`.jpg`/`.png` files -- which misses
  a real dataset in this project (`test_data/AS_LE_lateral view_E-cad`'s
  frame 1 image is `ture1.tiff`, a different extension than the rest of that
  sequence). `load_base_image` below reuses `data_input._find_matching_image`
  instead, which already searches the wider `.tif/.tiff/.jpeg/.jpg/.png` list.

Every compositing function here returns a `PIL.Image`, not raw bytes or
base64 -- keeps this module plain-data-in/plain-data-out and testable like
every other `v2` module. Only `encode_png_base64` crosses into
bytes/base64/JSON territory; the `v2/webview/api.py` layer calls it once at
the API boundary.
"""

import base64
from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import rgb2hex
from PIL import Image, ImageDraw

from data_input import _find_matching_image

# Built-in preset qualitative palettes offered by the UI's cluster-color
# picker, alongside free per-cluster HEX customization. Derived straight
# from matplotlib's own colormaps (not hand-transcribed) so they stay
# exactly consistent with any diagnostic plots elsewhere in the project.
PALETTE_PRESET_NAMES = ["tab10", "Set2", "Dark2", "Paired", "tab20"]


def get_palette_presets() -> dict[str, list[str]]:
    """`{"tab10": ["#1f77b4", ...], "Set2": [...], ...}` for
    `PALETTE_PRESET_NAMES`. Each preset's colors come straight from the
    matching matplotlib qualitative colormap.
    """
    presets = {}
    for name in PALETTE_PRESET_NAMES:
        cmap = colormaps[name]
        raw_colors = cmap.colors if hasattr(cmap, "colors") else [cmap(i) for i in range(cmap.N)]
        presets[name] = [rgb2hex(c) for c in raw_colors]
    return presets


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """`"#rrggbb"` (or `"rrggbb"`) -> `(r, g, b)` ints 0-255."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def load_base_image(npy_dict: dict, image_folder: Optional[str], frame_number: int) -> np.ndarray:
    """Load the RGB `uint8` base image for `frame_number`: prefers
    `npy_dict['img']` if present (matches Cellpose's own convention of
    sometimes embedding the source image); otherwise searches
    `image_folder` via `data_input._find_matching_image` (wide extension
    list, see module docstring).

    Converts grayscale (2D) input to RGB and normalizes to `uint8` the same
    way v1 did (`gui_comparison.py:434-439`): `(img / img.max() * 255)` when
    not already `uint8`.

    Raises `FileNotFoundError` if no image can be found (matching v1's
    behavior of failing loudly rather than showing a blank frame).
    """
    base_img = npy_dict.get("img")
    if base_img is None:
        if image_folder is None:
            raise FileNotFoundError(
                f"No embedded image for frame {frame_number} and no image_folder given to search."
            )
        base_img = _find_matching_image(image_folder, frame_number)
        if base_img is None:
            raise FileNotFoundError(f"No image found for frame {frame_number} in {image_folder}")

    if base_img.ndim == 2:
        base_img = np.stack([base_img] * 3, axis=-1)
    if base_img.dtype != np.uint8:
        base_img = (base_img / base_img.max() * 255).astype(np.uint8)
    return base_img


def render_classification_overlay(
    outlines: np.ndarray,
    base_image: np.ndarray,
    frame_number: int,
    clustering_data: pd.DataFrame,
    color_map: dict[int, str],
    unclassified_color: str = "#808080",
    hidden_labels: Optional[set[int]] = None,
    border_cell_indices: Optional[set[str]] = None,
    border_color: str = "#ff00ff",
) -> Image.Image:
    """Tint each cell in `outlines` by its cluster's color from `color_map`
    (`{Cluster_Label: "#rrggbb"}`). Cells present in the mask but missing
    from `clustering_data` (or whose label isn't in `color_map`) get
    `unclassified_color`. Generalizes v1's hardcoded 2-color
    `img[outlines==cell_id] = color` pattern (`gui_comparison.py:462-472`)
    to an arbitrary N-entry color map.

    `hidden_labels`: cluster labels to skip entirely (left as the untinted
    base image) -- lets the UI isolate one cluster at a time.

    `border_cell_indices`: `Cell_Index` strings (e.g. from a `leading_edge`
    column) to paint `border_color` instead of their cluster color -- the
    UI's "cluster borders" highlight, independent of the hide/show state.
    """
    img = base_image.copy()
    cell_type_dict = clustering_data.set_index("Cell_Index")["Cluster_Label"].to_dict()

    for cell_id in np.unique(outlines):
        if cell_id == 0:
            continue
        cell_index = f"Cell{frame_number}_{cell_id}"
        cluster_label = cell_type_dict.get(cell_index)
        if hidden_labels and cluster_label in hidden_labels:
            continue
        if border_cell_indices and cell_index in border_cell_indices:
            hex_color = border_color
        else:
            hex_color = color_map.get(cluster_label, unclassified_color)
        img[outlines == cell_id] = _hex_to_rgb(hex_color)

    return Image.fromarray(img)


def render_tracking_overlay(
    outlines: np.ndarray,
    base_image: np.ndarray,
    frame_number: int,
    tracking_data: pd.DataFrame,
    matched_color: str,
    unmatched_color: str,
    label_cells: bool = True,
) -> Image.Image:
    """Tint each cell by whether it's matched (present in `tracking_data`'s
    `Frame{frame_number}` column) or unmatched, generalizing v1's
    yellow-matched/blue-unmatched convention (`gui_cell_track.py`) to
    user-controlled colors. If `label_cells`, draws the resolved track
    `Index` (matched cells) or the raw `cell_id` (unmatched cells -- so the
    user has something to reference when assigning it to a track) at each
    cell's centroid.
    """
    img = base_image.copy()
    frame_col = f"Frame{frame_number}"
    id_col = tracking_data.columns[0]

    matched_lookup: dict[str, object] = {}
    if frame_col in tracking_data.columns:
        for _, row in tracking_data[[id_col, frame_col]].dropna(subset=[frame_col]).iterrows():
            matched_lookup[row[frame_col]] = row[id_col]

    labels: list[tuple[int, int, str]] = []
    for cell_id in np.unique(outlines):
        if cell_id == 0:
            continue
        cell_index = f"Cell{frame_number}_{cell_id}"
        track_id = matched_lookup.get(cell_index)
        is_matched = track_id is not None
        img[outlines == cell_id] = _hex_to_rgb(matched_color if is_matched else unmatched_color)

        if label_cells:
            positions = np.where(outlines == cell_id)
            cy, cx = int(np.mean(positions[0])), int(np.mean(positions[1]))
            labels.append((cx, cy, str(track_id) if is_matched else str(cell_id)))

    pil_img = Image.fromarray(img)
    if label_cells:
        draw = ImageDraw.Draw(pil_img)
        for cx, cy, text in labels:
            draw.text((cx, cy), text, fill="white")

    return pil_img


def render_cell_highlight(
    outlines: np.ndarray,
    base_image: np.ndarray,
    cell_id: int,
    color: str,
) -> Image.Image:
    """Tint a single cell (by raw `outlines` label id, not `Cell Index`
    string) -- the trajectory view's per-frame highlight."""
    img = base_image.copy()
    img[outlines == cell_id] = _hex_to_rgb(color)
    return Image.fromarray(img)


def cell_id_at_pixel(outlines: np.ndarray, x: int, y: int) -> Optional[int]:
    """Direct array lookup: the raw `outlines` label at pixel `(x, y)`, or
    `None` if out of bounds or background (0). Caller must compute `(x, y)`
    from the displayed image's natural pixel dimensions, not its
    CSS-rendered size -- a mismatch there is the one place a scaling bug
    would silently break click-to-select.
    """
    if 0 <= y < outlines.shape[0] and 0 <= x < outlines.shape[1]:
        cell_id = int(outlines[y, x])
        return cell_id if cell_id != 0 else None
    return None


def encode_png_base64(image: Image.Image) -> str:
    """`PIL.Image` -> base64-encoded PNG string, ready to hand to the
    frontend as an `<img src="data:image/png;base64,...">`."""
    buf = BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standalone debug entry for v2.rendering")
    parser.add_argument("--npy-file", required=True, help="A single frame's .npy segmentation file")
    parser.add_argument("--frame-number", type=int, required=True)
    parser.add_argument("--image-folder", default=None)
    parser.add_argument("--clustering-data", default=None, help="cells_clustering_results.xlsx path (optional)")
    parser.add_argument("--output-png", default="rendering_debug_output.png")
    args = parser.parse_args()

    dat = np.load(args.npy_file, allow_pickle=True).item()
    outlines = dat["outlines"]
    base_image = load_base_image(dat, args.image_folder, args.frame_number)
    print(f"Base image shape: {base_image.shape}, dtype: {base_image.dtype}")

    if args.clustering_data:
        clustering_df = pd.read_excel(args.clustering_data)
        labels = sorted(clustering_df["Cluster_Label"].unique())
        default_palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33"]
        color_map = {lbl: default_palette[i % len(default_palette)] for i, lbl in enumerate(labels)}
        print(f"Rendering classification overlay with color_map={color_map}")
        overlay = render_classification_overlay(outlines, base_image, args.frame_number, clustering_df, color_map)
    else:
        overlay = Image.fromarray(base_image)
        print("No --clustering-data given; rendering the raw base image only.")

    overlay.save(args.output_png)
    print(f"Saved {args.output_png}")

    b64 = encode_png_base64(overlay)
    print(f"Base64 PNG length: {len(b64)} chars")

    test_x, test_y = base_image.shape[1] // 2, base_image.shape[0] // 2
    print(f"cell_id_at_pixel({test_x}, {test_y}) = {cell_id_at_pixel(outlines, test_x, test_y)}")
