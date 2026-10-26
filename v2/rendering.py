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
import colorsys
from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import rgb2hex
from PIL import Image, ImageDraw, ImageFont

from data_input import _find_matching_image, thin_label_outlines

# Built-in preset qualitative palettes offered by the UI's cluster-color
# picker, alongside free per-cluster HEX customization. Derived straight
# from matplotlib's own colormaps (not hand-transcribed) so they stay
# exactly consistent with any diagnostic plots elsewhere in the project.
PALETTE_PRESET_NAMES = ["tab10", "Set2", "Dark2", "Paired", "tab20"]

# CONTINUOUS colormaps, for scalar-per-cell overlays (the environment score)
# -- distinct from the qualitative presets above, which map discrete
# cluster/track ids. viridis is Tissue Analyzer's own default for this score
# (EPySeg's `matplotlib_to_TA()`); PuOr/BrBG are diverging alternatives.
CONTINUOUS_LUT_NAMES = ["viridis", "Fire", "Gem", "PuOr", "BrBG"]

# ImageJ's built-in "Fire" LUT, which exists only as hardcoded arrays inside
# ImageJ's LutLoader (it is NOT one of Fiji's ~500 `.lut` files). These are
# its canonical 32 control points; ImageJ itself interpolates them to 256, so
# reproducing that here is exact rather than an approximation.
_FIRE_CONTROL_POINTS = (
    (0, 0, 1, 25, 49, 73, 98, 122, 146, 162, 173, 184, 195, 207, 217, 229,
     240, 252, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 35, 57,
     79, 101, 117, 133, 147, 161, 175, 190, 205, 219, 234, 248, 255, 255, 255, 255),
    (0, 61, 96, 130, 165, 192, 220, 227, 210, 181, 151, 122, 93, 64, 35, 5,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 98, 160, 223, 255, 255, 255),
)

# Fiji's "Gem" LUT (`Fiji.app/luts/Gem.lut`, an 800-byte NIH-Image file: a
# 32-byte `ICOL` header followed by raw R[256]G[256]B[256]). Embedded verbatim
# as base64 rather than referenced at runtime so the app doesn't depend on a
# Fiji install, and exactly rather than as control points -- subsampling was
# measured to still cost ~3/255 of error even at 65 anchors.
_GEM_LUT_B64 = (
    "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB"
    "AQEBAQEBAQEBAQECBAYICgwOEBIUFhgaHB4gIiQmKCosLjAyNDY4Ojw+QEFDRUdJS01PUVNV"
    "V1lbXV9hY2VnaWttb3FzdXd5e31/gYOFh4mLjY+Rk5WXmZudn6Gjpaepq62vsbO1t7m7vb/A"
    "wsTGyMrMztDS1NbY2tze4OLk5ujq7O7w8vT2+Pr8/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+"
    "/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/gICAgMDBAQFBQYGBwcI"
    "CAkJCgoLCwwMDQ0ODg8PEBARERISExMUFBUVFhYXFxgYGRkaGhsbHBwdHR4eHx8gICEhICAf"
    "Hx4eHR0cHBsbGhoZGRgYFxcWFhUVFBQTExISEREQEA8PDg4NDQwMCwsKCgkJCAgHBwYGBQUE"
    "BAMDAgIBAwQGBwkKDA0PEBITFRYYGRscHh8hIiQlJygqKy0uMDEzNDY3OTo8PT9AQUJERUdI"
    "SktNTlBRU1RWV1laXF1fX2JkZ2lsbnFzdnh7fYCChYeKjI+RlJaZm56go6Woqq2vsrS3uby+"
    "wMLFx8rMz9HU1tnb3uDj5ejq7e/y9Pf5/PwCAgMEBQYHCAkKCwwNDg8QERITFBUWFxgZGhsc"
    "HR4fICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj9AQEFCQ0RFRkdISUpLTE1OT1BR"
    "UlNUVVZXWFlaW1xdXl9gYWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXp7fH1+f399e3l3dXNx"
    "b21raWdlY2FfXVtZV1VTUU9NS0lHRUNBQD48Ojg2NDIwLiwqKCYkIiAeHBoYFhQSEA4MCggG"
    "BAEECAwQFBgcICQoLDA0ODxAQ0dLT1NXW19jZ2tvc3d7f4OHi4+Tl5ufo6err7O3u7/CxsrO"
    "0tba3uLm6u7y9vr6"
)


def _lut_256(name: str) -> np.ndarray:
    """A `(256, 3)` uint8 RGB ramp for `name` (one of
    `CONTINUOUS_LUT_NAMES`). `Fire`/`Gem` are ImageJ/Fiji LUTs with no
    matplotlib equivalent (see the constants above); everything else is
    sampled from matplotlib's own continuous colormap."""
    if name == "Fire":
        r, g, b = _FIRE_CONTROL_POINTS
        anchors = np.linspace(0, 255, len(r))
        grid = np.arange(256)
        return np.stack(
            [np.interp(grid, anchors, channel) for channel in (r, g, b)], axis=1
        ).round().astype(np.uint8)
    if name == "Gem":
        raw = np.frombuffer(base64.b64decode(_GEM_LUT_B64), dtype=np.uint8)
        return np.stack([raw[0:256], raw[256:512], raw[512:768]], axis=1)
    cmap = colormaps[name]
    return (np.array([cmap(i / 255.0)[:3] for i in range(256)]) * 255).round().astype(np.uint8)


def continuous_lut_colors(values: dict, lut_name: str = "viridis",
                          dead_band: Optional[float] = None) -> dict:
    """Map `{key: score}` to `{key: (r, g, b)}` through `lut_name`, with a
    **fixed 0->1 range**.

    The fixed range is deliberate. EPySeg's equivalent (`apply_lut` called
    with `min=None, max=None`) rescales each frame to its own min/max, so the
    same color means a different score on different frames and the movie
    can't be read as a time series. Pinning the range keeps frames
    comparable; scores outside `[0, 1]` are clamped rather than rescaling
    everything around them.
    """
    lut = _lut_256(lut_name)
    colors = {}
    for key, value in values.items():
        score = float(value)
        if dead_band is not None and score >= dead_band:
            # Above the dead band the cell is CLEAN and is not tinted at all,
            # so the base image shows through. Tissue Analyzer's rule
            # (`take_decision`, sensitivity 0.75) is to ignore a cell entirely
            # when both directions score >= 0.75, with the comment that it "has
            # most likely changed one or two neighbors, no reason to worry" --
            # 0.75 is exactly the value at which losing a single neighbour never
            # flags (0.750/0.800/0.833/0.857/0.875 for 4-8 neighbours).
            #
            # Without it, a continuous LUT darkens anything below 1.0: measured
            # on `embryos_GBE_GAP43`, 64.8% of cell-frames render as "somewhat
            # dark" against 22.5% under TA's rule, because 42.2% sit in
            # [0.75, 1.0) -- a band TA classes as clean. That band was the whole
            # of the reported over-sensitivity.
            continue
        index = int(round(min(1.0, max(0.0, score)) * 255))
        colors[key] = tuple(int(c) for c in lut[index])
    return colors


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


def _golden_angle_colors(n: int) -> list[tuple[int, int, int]]:
    """`n` maximally-distinct RGB colors via the golden-angle HSV hue
    rotation (hue advances ~137.508 degrees per label, wrapped mod 360) --
    MorpholibJ's default "Golden angle" LUT for label/instance images,
    reused here since it (unlike a small fixed qualitative palette) never
    visibly repeats even across hundreds of labels.
    """
    golden_angle = 137.508
    colors = []
    for i in range(n):
        hue = (i * golden_angle) % 360
        r, g, b = colorsys.hsv_to_rgb(hue / 360, 0.65, 0.95)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def assign_identity_colors(track_ids: list, palette: str = "golden_angle") -> dict[object, tuple[int, int, int]]:
    """Map each id in `track_ids` to a distinct RGB color. `palette` is
    either `"golden_angle"` (see `_golden_angle_colors`, scales to any
    count without visibly repeating) or one of `PALETTE_PRESET_NAMES` (the
    same qualitative presets used for cluster colors elsewhere; cycles once
    there are more ids than palette entries).
    """
    n = len(track_ids)
    if palette == "golden_angle":
        colors = _golden_angle_colors(n)
    else:
        cmap = colormaps[palette]
        raw_colors = cmap.colors if hasattr(cmap, "colors") else [cmap(i) for i in range(cmap.N)]
        base_rgb = [tuple(int(c * 255) for c in col[:3]) for col in raw_colors]
        colors = [base_rgb[i % len(base_rgb)] for i in range(n)]
    return dict(zip(track_ids, colors))


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
    track_color_mode: str = "uniform",
    identity_palette: str = "golden_angle",
    label_font_size: int = 10,
    environment_scores: Optional[dict] = None,
    environment_lut: str = "viridis",
    environment_dead_band: Optional[float] = None,
    lifecycle_states: Optional[dict] = None,
    lifecycle_colors: Optional[dict] = None,
    tint_other_cells: bool = False,
    dynamics_states: Optional[dict] = None,
    dynamics_colors: Optional[dict] = None,
    nematic_axes: Optional[list] = None,
    agreement_states: Optional[dict] = None,
    agreement_colors: Optional[dict] = None,
    extracted_labels: Optional[set] = None,
    unextracted_color: str = "#ff00ff",
) -> Image.Image:
    """Tint each cell by whether it's matched (present in `tracking_data`'s
    `Frame{frame_number}` column) or unmatched, generalizing v1's
    yellow-matched/blue-unmatched convention (`gui_cell_track.py`) to
    user-controlled colors. If `label_cells`, draws the resolved track
    `Index` (matched cells) or the raw `cell_id` (unmatched cells -- so the
    user has something to reference when assigning it to a track) at each
    cell's centroid, at `label_font_size` (via `ImageFont.load_default`'s
    `size` argument, supported since Pillow 10.1 -- no bundled font file
    needed).

    **Two different states used to share `unmatched_color` (Round 26 C).** A
    cell can be absent from the tracking table because the LAP gave it to the
    sentinel or never offered it -- a tracking outcome -- or because it has no
    `Cells_info` row at all, having been filtered out at extraction, which is
    not a tracking outcome and cannot be fixed by anything the correction UI
    does. Drawn in one colour, the second reads as the first: that is what sent
    a maintainer looking for a LAP bug in `Cell1_64`, whose real cause was the
    edge filter (`docs/edge-flicker-diagnosis.md`). Pass `extracted_labels`
    (the raw labels that DO have a row in this frame) and a cell outside it is
    drawn in `unextracted_color` instead. Omit it and the colouring is exactly
    as before -- two states, one colour.

    `track_color_mode`: `"uniform"` (default) uses `matched_color`/
    `unmatched_color` as above. `"identity"` instead gives each matched
    cell its own track's color from `assign_identity_colors` (`palette`
    either `"golden_angle"`, the default -- scales to any track count
    without visibly repeating -- or one of `PALETTE_PRESET_NAMES`, cycled),
    so individual cells/tracks are visually distinguishable at a glance
    rather than all matched cells sharing one color; unmatched cells still
    get `unmatched_color`.

    Two further modes exist to help *find* errors rather than inspect a cell
    already suspected, both modeled on Tissue Analyzer's error-spotting
    workflow (Aigouy & Prud'homme, Fig. 3):

    - `"environment"` colors each cell by `environment_scores`
      (`{raw_label: 0..1}` from `adjacency.environment_scores`) through the
      `environment_lut` continuous LUT. Per TA, **dark = the cell's
      neighborhood changed a lot** (inspect it); bright = preserved. Cells
      absent from the dict are unscoreable and are left untinted rather than
      being colored as if they scored 0 -- see `adjacency.environment_scores`
      for why that distinction matters.
    - `"lifecycle"` colors each cell by `lifecycle_states`
      (`{cell_index: "lost"|"new"|"both"}` from
      `manual_correction_tracking.classify_track_lifecycle`) via
      `lifecycle_colors` (`{"lost": "#...", "new": "#...", "both": "#..."}`).
      `"both"` is a single-frame track -- TA's own oversegmentation example,
      hence its own color. Cells in no lifecycle state are left untinted
      unless `tint_other_cells`, in which case they take `matched_color`/
      `unmatched_color` as in `"uniform"`.
    """
    img = base_image.copy()
    frame_col = f"Frame{frame_number}"
    id_col = tracking_data.columns[0]

    # First-claim-wins, matching `api.py`'s `get_tracking_cell_at_pixel`
    # (`match.iloc[0]`) -- a legacy tracking file predating the
    # `_merge_tracking_pairs` fix could still have two rows claiming the
    # same cell; resolving both lookups the same way keeps which track
    # "wins" consistent between clicking a cell and reading its on-image
    # label, instead of one being first-match and the other silently being
    # last-match (a dict overwrite via `.iterrows()`).
    matched_lookup: dict = build_matched_lookup(tracking_data, frame_number)

    identity_color_by_track: dict[object, tuple[int, int, int]] = {}
    if track_color_mode == "identity":
        all_track_ids = sorted(tracking_data[id_col].dropna().unique())
        identity_color_by_track = assign_identity_colors(all_track_ids, identity_palette)

    environment_rgb: dict = {}
    if track_color_mode == "environment":
        environment_rgb = continuous_lut_colors(
            environment_scores or {}, environment_lut, dead_band=environment_dead_band)

    dynamics_rgb: dict = {}
    if track_color_mode == "dynamics":
        dynamics_rgb = {
            state: _hex_to_rgb(hex_color)
            for state, hex_color in (dynamics_colors or {}).items()
        }

    agreement_rgb: dict = {}
    if track_color_mode == "agreement":
        agreement_rgb = {
            state: _hex_to_rgb(hex_color)
            for state, hex_color in (agreement_colors or {}).items()
        }

    lifecycle_rgb: dict = {}
    if track_color_mode == "lifecycle":
        lifecycle_rgb = {
            state: _hex_to_rgb(hex_color)
            for state, hex_color in (lifecycle_colors or {}).items()
        }

    labels: list[tuple[int, int, str, bool]] = []
    for cell_id in np.unique(outlines):
        if cell_id == 0:
            continue
        cell_index = f"Cell{frame_number}_{cell_id}"
        track_id = matched_lookup.get(cell_index)
        is_matched = track_id is not None
        # "Unmatched" only means something if the cell was a candidate at all.
        absent_color = (unextracted_color
                        if extracted_labels is not None and int(cell_id) not in extracted_labels
                        else unmatched_color)

        if track_color_mode == "environment":
            # Unscoreable cells stay untinted (base image shows through) --
            # deliberately NOT colored as if they scored 0.
            if cell_id in environment_rgb:
                img[outlines == cell_id] = environment_rgb[cell_id]
        elif track_color_mode == "dynamics":
            # `{raw_label: {"event": "division"|"delamination"|"t1",
            #               "confirmed": bool}}`. A confirmed event is painted
            # solid; an unconfirmed CANDIDATE is dimmed towards the base image,
            # so the two are never mistaken for each other -- the whole point
            # of the curation workflow is that the detector proposes and a
            # human decides.
            entry = (dynamics_states or {}).get(cell_id)
            if entry is not None and entry.get("event") in dynamics_rgb:
                colour = np.array(dynamics_rgb[entry["event"]], dtype=float)
                if not entry.get("confirmed"):
                    region = img[outlines == cell_id].astype(float)
                    colour = 0.45 * colour + 0.55 * region.mean(axis=0)
                img[outlines == cell_id] = colour.round().astype(np.uint8)
            elif tint_other_cells:
                img[outlines == cell_id] = _hex_to_rgb(matched_color if is_matched else absent_color)
        elif track_color_mode == "agreement":
            # `{raw_label: "agrees"|"disagrees"|"unjudgeable"|"broken"}` from
            # `tracking_benchmark.classify_links`. THREE meaningful states, and
            # the unjudgeable one must be visible: on `embryos_GBE_GAP43` about
            # a quarter of cells cannot be judged by the overlap reference at
            # all, and painting them like agreeing cells would claim a verdict
            # that was never reached.
            state = (agreement_states or {}).get(cell_id)
            if state is not None and state in agreement_rgb:
                img[outlines == cell_id] = agreement_rgb[state]
            elif tint_other_cells:
                img[outlines == cell_id] = _hex_to_rgb(matched_color if is_matched else absent_color)
        elif track_color_mode == "lifecycle":
            state = (lifecycle_states or {}).get(cell_index)
            if state is not None and state in lifecycle_rgb:
                img[outlines == cell_id] = lifecycle_rgb[state]
            elif tint_other_cells:
                img[outlines == cell_id] = _hex_to_rgb(matched_color if is_matched else absent_color)
        elif is_matched and track_color_mode == "identity":
            img[outlines == cell_id] = identity_color_by_track[track_id]
        else:
            img[outlines == cell_id] = _hex_to_rgb(matched_color if is_matched else absent_color)

        if label_cells:
            positions = np.where(outlines == cell_id)
            cy, cx = int(np.mean(positions[0])), int(np.mean(positions[1]))
            labels.append((cx, cy) + cell_label_text(cell_id, track_id if is_matched else None))

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img) if (label_cells or nematic_axes) else None

    if nematic_axes:
        # T1 orientation, drawn as a headless axis (both directions from the
        # centre) because a nematic has no arrow -- see
        # `topology.detect_t1_transitions`.
        #
        # An axis whose exchange has not SETTLED (`persistent` False, from
        # `topology.annotate_persistence`) is drawn thin and dashed rather than
        # solid, the axis-level counterpart of dimming an unconfirmed cell
        # above. On `embryos_GBE_GAP43` 21% of raw events reverse within one
        # frame, so this is the difference between a rearrangement and a
        # flicker -- and the two must not look alike.
        for axis in nematic_axes:
            cx, cy = axis["center"]
            half = axis.get("length", 12) / 2.0
            dx, dy = half * np.cos(axis["phi"]), half * np.sin(axis["phi"])
            colour = axis.get("color", "#ffffff")
            if axis.get("persistent", True):
                draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=colour, width=2)
                continue
            for start, end in ((-1.0, -0.45), (-0.15, 0.15), (0.45, 1.0)):
                draw.line(
                    [(cx + start * dx, cy + start * dy), (cx + end * dx, cy + end * dy)],
                    fill=colour, width=1,
                )

    if label_cells:
        font = ImageFont.load_default(size=label_font_size)
        for cx, cy, text, is_track in labels:
            # An unmatched cell's raw label is drawn in a different colour as
            # well as prefixed: a number that is a track id for most cells and
            # a raw label for a few, with no way to tell which, is worse than
            # either alone.
            draw.text((cx, cy), text, fill="white" if is_track else "#ff9f0a", font=font)

    return pil_img


def render_score_overlay(
    outlines: np.ndarray,
    base_image: np.ndarray,
    scores: dict,
    lut_name: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> Image.Image:
    """Tint each cell by its `scores[raw_label]` value through `lut_name`.
    Cells with no score are left untinted so the base image shows through --
    "not judgeable" reads differently from "scored 0", the same convention
    `render_tracking_overlay`'s environment mode uses.

    `vmin`/`vmax` window which part of the score range the LUT spans, for
    metrics whose realistic values don't fill 0-1 (see
    `segmentation_qc.QC_METRIC_DISPLAY`). This is a FIXED window, not a
    per-frame autoscale -- frames stay comparable, which is the property
    `continuous_lut_colors` exists to protect.

    Deliberately generic (a plain `{label: score}` dict in, no DataFrame,
    no tracking data) so it serves the Extract step's segmentation-QC view,
    which runs before any tracking exists.
    """
    span = (vmax - vmin) or 1.0
    windowed = {label: (value - vmin) / span for label, value in scores.items()}

    img = base_image.copy()
    for label, rgb in continuous_lut_colors(windowed, lut_name).items():
        img[outlines == label] = rgb
    return Image.fromarray(img)


def render_value_map(
    fill_labels: np.ndarray,
    base_image: np.ndarray,
    values: dict,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    bar_width: int = 18,
) -> Image.Image:
    """Any per-cell value painted back onto the frame, with a colour bar.

    `render_score_overlay` already does the tinting; what this adds is the bar
    and its two end labels, so an exported frame can be read without the UI
    around it. A cell with no value is left untinted -- "not measured" is not
    the same as "measured low", and on a `Cells_info_PCA` value that is most of
    the edge cells.

    `vmin`/`vmax` are the caller's, never this frame's own range: autoscaling
    each frame of a movie makes every frame look equally variable, which is a
    statement about the colour map rather than the tissue. For a signed value
    (a PC score) the caller passes a symmetric window and a diverging map, so
    zero sits at the middle of the ramp rather than wherever the data does.

    The bar is drawn at the image's own pixel resolution; nothing is upscaled,
    because enlarging a label image adds no information.

    **No text is drawn into the image** (Round 35 3-1). It used to carry the
    value's name at the top left and the range at the ends of the bar, in
    PIL's bitmap default font -- which has no glyph for `µ`, so `µm²` rendered
    as boxes, and which cannot be edited, translated or restyled once it is
    pixels. The range belongs in the panel's status line, where it can say
    whether the scale is this frame's or the whole movie's.
    """
    tinted = np.asarray(render_score_overlay(fill_labels, base_image, values,
                                             lut_name=cmap, vmin=vmin, vmax=vmax))
    height, width = tinted.shape[:2]
    bar_total = bar_width + 2
    canvas = np.zeros((height, width + bar_total, 3), dtype=np.uint8)
    canvas[:, :width] = tinted
    ramp = _lut_256(cmap)[(np.linspace(1, 0, height) * 255).astype(int)]
    canvas[:, width + 2:] = ramp[:, None, :]

    return Image.fromarray(canvas)


def _label_centroids(fill_labels: np.ndarray) -> dict[int, tuple[int, int]]:
    """`{label: (cx, cy)}` for every non-background label, in one pass.

    Uses `bincount` sums rather than a `np.where` per cell (the pattern
    `render_tracking_overlay` uses) because this runs over every cell in the
    frame at once -- on an ~800-cell frame the per-cell scan costs a full
    image traversal each time.
    """
    flat = fill_labels.ravel()
    width = fill_labels.shape[1]
    positions = np.arange(flat.size)
    counts = np.bincount(flat)
    sum_y = np.bincount(flat, weights=positions // width)
    sum_x = np.bincount(flat, weights=positions % width)

    centroids = {}
    for label in np.nonzero(counts)[0]:
        if label == 0:
            continue
        centroids[int(label)] = (int(sum_x[label] / counts[label]), int(sum_y[label] / counts[label]))
    return centroids


def cell_boundary_mask(fill_labels: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels that sit on a cell's edge -- inside a cell and
    4-adjacent to a different label (another cell or background).

    Derived from the FILLED mask rather than from a stored `'outlines'` array
    so the "Outline" overlay draws a boundary even for a mask whose stored
    outlines are missing or unreliable. `data_input.thin_label_outlines` is
    the single implementation of the ring geometry; this just reduces it to a
    boolean.
    """
    return thin_label_outlines(fill_labels) > 0


def draw_cell_outlines(img: np.ndarray, fill_labels: np.ndarray, color: str = "#00ff00") -> np.ndarray:
    """Paint each cell's boundary onto `img` (modified in place and returned)."""
    img[cell_boundary_mask(fill_labels)] = _hex_to_rgb(color)
    return img


def build_matched_lookup(tracking_data, frame_number: int) -> dict:
    """`{cell_index: track_id}` for one frame.

    First-match wins, deliberately: two rows can still claim the same cell, and
    resolving every lookup the same way keeps which track "wins" consistent
    between clicking a cell and reading its on-image label -- instead of one
    being first-match and the other silently last-match via a dict overwrite.
    """
    if tracking_data is None:
        return {}
    frame_col = f"Frame{frame_number}"
    if frame_col not in tracking_data.columns:
        return {}
    id_col = tracking_data.columns[0]
    lookup: dict = {}
    for _, row in tracking_data[[id_col, frame_col]].dropna(subset=[frame_col]).iterrows():
        if row[frame_col] not in lookup:
            lookup[row[frame_col]] = row[id_col]
    return lookup


# A raw segmentation label shown where a track id was expected. Cellpose
# renumbers raw labels every frame, so an unprefixed one looks exactly like a
# track id while jumping between frames for no reason -- the mixing this marker
# exists to prevent (Round 17 C2).
RAW_LABEL_PREFIX = "#"


def cell_label_text(cell_id: int, track_id=None) -> tuple:
    """`(text, is_track_id)` for one cell.

    **One resolver, used by the main overlay and the inset alike.** The inset
    used to label with raw ids while the main view labelled with track ids, so
    the same three cells read 284 / 281 / 282 in one panel and 65 / 72 / 76 in
    the other -- two id spaces, no way to tell which you were looking at. That
    defect *was* the duplication, which is why this is a function and not a
    line repeated twice.
    """
    if track_id is None:
        return f"{RAW_LABEL_PREFIX}{int(cell_id)}", False
    return str(track_id), True


def draw_cell_labels(
    pil_img: Image.Image, fill_labels: np.ndarray, font_size: int = 10, color: str = "#ffffff",
    matched_lookup: Optional[dict] = None, frame_number: Optional[int] = None,
    raw_color: str = "#ff9f0a",
) -> Image.Image:
    """Draw each cell's id at its centroid.

    With `matched_lookup` and `frame_number`, labels are **track** ids -- the
    same thing the main overlay draws -- and a cell no track claims falls back
    to its raw label, prefixed and in `raw_color` so the two id spaces are
    never silently mixed. Without them, raw labels, which is what the Extract
    preview wants.
    """
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default(size=font_size)
    for label, (cx, cy) in _label_centroids(fill_labels).items():
        if matched_lookup is None or frame_number is None:
            draw.text((cx, cy), str(label), fill=color, font=font)
            continue
        track_id = matched_lookup.get(f"Cell{frame_number}_{label}")
        text, is_track = cell_label_text(label, track_id)
        draw.text((cx, cy), text, fill=color if is_track else raw_color, font=font)
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
    parser.add_argument("--npy-file", help="A single frame's .npy segmentation file")
    parser.add_argument("--frame-number", type=int)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--image-folder", default=None)
    parser.add_argument("--clustering-data", default=None, help="cells_clustering_results.xlsx path (optional)")
    parser.add_argument("--output-png", default="rendering_debug_output.png")
    args = parser.parse_args()

    if args.self_test:
        import pandas as pd

        # --- R26 C: "unmatched" was two states in one colour.
        # A cell the LAP did not link (a row exists; a correction can fix it)
        # and a cell Extract never emitted (no row; nothing to correct) looked
        # identical, which is what sent a maintainer looking for a LAP bug in a
        # cell the edge filter had removed before tracking ever saw it.
        edge_tracking = pd.DataFrame({"Index": [1], "Frame1": ["Cell1_1"]})
        grid = np.zeros((30, 30), dtype=np.int32)
        grid[2:10, 2:10] = 1       # matched
        grid[2:10, 15:25] = 2      # has a row, not linked
        grid[15:25, 2:10] = 3      # no row at all
        base = np.zeros((30, 30, 3), dtype=np.uint8)
        painted = np.asarray(render_tracking_overlay(
            grid, base, 1, edge_tracking, "#ffd633", "#3388ff", label_cells=False,
            extracted_labels={1, 2}, unextracted_color="#ff00ff"))
        assert tuple(painted[5, 5]) == _hex_to_rgb("#ffd633"), painted[5, 5]
        assert tuple(painted[5, 20]) == _hex_to_rgb("#3388ff"), painted[5, 20]
        assert tuple(painted[20, 5]) == _hex_to_rgb("#ff00ff"), painted[20, 5]
        # Omitting the set must reproduce the old behaviour exactly, so a caller
        # that cannot tell the two apart does not start claiming it can.
        legacy = np.asarray(render_tracking_overlay(
            grid, base, 1, edge_tracking, "#ffd633", "#3388ff", label_cells=False))
        assert tuple(legacy[20, 5]) == tuple(legacy[5, 20]) == _hex_to_rgb("#3388ff"), legacy[20, 5]
        print("R26 C: a cell with no Cells_info row is drawn in its own colour (magenta),")
        print("  not in the unmatched blue a correction could act on; without")
        print("  `extracted_labels` the overlay is byte-identical to before.")

        # --- C2 (R17): the main overlay and the inset must resolve a label the
        # same way. They did not -- the inset drew RAW segmentation labels,
        # which Cellpose renumbers every frame, so the same three cells read
        # 284/281/282 in the main view and 65/72/76 in the inset: two id spaces
        # with no way to tell which you were looking at. One resolver now, and
        # this asserts it across three consecutive frames.
        tracking = pd.DataFrame({
            "Index": [7, 8],
            "Frame1": ["Cell1_65", "Cell1_72"],
            "Frame2": ["Cell2_11", "Cell2_40"],   # raw labels change; tracks do not
            "Frame3": ["Cell3_5", None],          # track 8 ends here
        })
        for frame, raws in ((1, (65, 72)), (2, (11, 40)), (3, (5, 99))):
            lookup = build_matched_lookup(tracking, frame)
            for raw in raws:
                track = lookup.get(f"Cell{frame}_{raw}")
                text, is_track = cell_label_text(raw, track)
                if track is None:
                    assert text == f"{RAW_LABEL_PREFIX}{raw}" and not is_track, (frame, raw, text)
                else:
                    assert text == str(track) and is_track, (frame, raw, text)
        across = [cell_label_text(raw, build_matched_lookup(tracking, frame).get(f"Cell{frame}_{raw}"))[0]
                  for frame, raw in ((1, 65), (2, 11), (3, 5))]
        assert across == ["7", "7", "7"], across
        print("C2: one cell reads '7' at frames 1, 2 and 3 though its raw id is 65 / 11 / 5,")
        print("  and an unmatched cell falls back to '%s99' -- prefixed and drawn in a"
              % RAW_LABEL_PREFIX)
        print("  different colour, so the two id spaces are never silently mixed.")

        # The inset and the overlay share this resolver by construction: both
        # go through `build_matched_lookup` + `cell_label_text`.
        canvas = Image.fromarray(np.zeros((30, 30, 3), dtype=np.uint8))
        grid = np.zeros((30, 30), dtype=int)
        grid[5:15, 5:15] = 65
        drawn = draw_cell_labels(canvas.copy(), grid, matched_lookup=build_matched_lookup(tracking, 1),
                                 frame_number=1)
        assert np.asarray(drawn).any(), "a track label must actually be drawn"
        plain = draw_cell_labels(canvas.copy(), grid)
        assert not np.array_equal(np.asarray(drawn), np.asarray(plain)), \
            "track labels and raw labels must render differently"
        print("  draw_cell_labels renders track ids when given a lookup and raw ids without one.")

        print("\nAll rendering self-tests passed.")
        raise SystemExit(0)

    if not args.npy_file or args.frame_number is None:
        parser.error("--npy-file and --frame-number are required unless --self-test is given")

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
