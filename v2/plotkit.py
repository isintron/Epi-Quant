"""Figure style and export for the GUI's plots, ported from the manuscript
figure scripts so the two cannot drift apart.

Where this comes from, and why it is here rather than imported: `figkit.py`
exists in **four** copies under `figures/revision_2026-10/`, `morph.py` in two,
and `v2/plotting_style.py` is a fifth style with genotype colours that
contradict the manuscript (it has WT blue / tmc orange / pyd green; the
manuscript is WT green / pyd orange / tmc pink). None of them can be imported
by the GUI: `figkit` needs `cmocean` and `lxml`, neither of which this project
depends on, and `make_oscillation_figures` runs `make_efd_analyses`'s
module-level pickle load on import. So the pure parts are ported here, pinned
by the self-test, and `figures/` is left alone -- those scripts are what the
manuscript is being drawn with while it is in revision.

Three things the GUI needs that a figure script does not:

- a **style** the user can change (`style_context`), on top of which our own
  non-negotiables are re-applied -- a Matplotlib style sheet may set a font
  size or an SVG text mode, and neither is its business here;
- **physical sizes**: pt for type, pt for strokes, mm for the figure. Fixed in
  those units nothing rescales anything else, and the screen preview is only a
  zoom;
- an **Excel export** that survives real datasets (`write_tables`): a sheet
  holds 1,048,576 rows and 16,384 columns, a sheet NAME holds 31 characters and
  no `[]:*?/\\`, and `Horizontal/Vertical Length Ratio` has a slash in it.

Fonts never go near pywebview: every figure is rendered to PNG/SVG by
Matplotlib in Python and the frontend only displays the image, so the font list
is `matplotlib.font_manager`'s and nothing else.
"""

import os
import re
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

MM = 1 / 25.4

# ---------------------------------------------------------------- palette
# Cell-type colours: light = fills, dark = points and lines. The pairs are
# separated in lightness, not only hue, so they survive red-green colour
# blindness (CIEDE2000 under deuteranopia 28 for AS/LE, 22 for LEC/HB, against
# 2.5 and 5.2 for the light pair alone).
AS_FILL, LE_FILL = "#DCCBB6", "#ACD6A8"
AS, LE = "#C4A07C", "#3F6B45"
LEC_FILL, HB_FILL = "#E0A8A8", "#98B090"
LEC, HB = "#D98C8C", "#4D7350"
TAN, LIGHT_GREY, MID_GREY, INK = "#D9C2A3", "#D4D4D4", "#9A9A9A", "#333333"
# Exemplar cells: neutral, so a glyph never reads as a class.
GLYPH_FILL, GLYPH_LINE = "#EDEDED", "#555555"

# Genotypes, as the manuscript prints them. `v2/plotting_style.py` disagrees
# (WT blue / tmc orange / pyd green) and is the stale one.
GENOTYPE = {"WT": "#2F6A46", "WT_G": "#2F6A46", "pyd": "#BD7338", "tmc": "#E0989E"}
GENOTYPE_MARKER = {"WT": "o", "WT_G": "o", "tmc": "s", "pyd": "^"}

GREY_MAP = LinearSegmentedColormap.from_list("greys", ["#F2F2F2", "#BDBDBD", "#7A7A7A", "#2E2E2E"])
# AlphaFold PAE-style green ramp, reversed so high = dark.
AF_GREEN = LinearSegmentedColormap.from_list(
    "af_green", ["#E3F2DD", "#BEE0B2", "#93C98B", "#66AB68", "#44894C", "#327138"])
# The named maps a PC map may use. `cmocean`'s are deliberately absent: the
# figure scripts have them, this project does not depend on cmocean.
SEQUENTIAL_MAPS = {"AFgreen": AF_GREEN, "YlGn": "YlGn", "grey": GREY_MAP,
                   "viridis": "viridis", "magma": "magma"}
# PC scores are signed and centred on zero, so they get a diverging map and a
# symmetric range -- a sequential map on a signed value puts the visual middle
# wherever the data happens to be centred.
DIVERGING_MAPS = {"RdBu_r": "RdBu_r", "coolwarm": "coolwarm", "PuOr_r": "PuOr_r",
                  "BrBG": "BrBG"}

# ---------------------------------------------------------------- presets
# Journal column widths. Height is NOT derived from the figure width (Round 35
# 0-C): it is derived from the AXES PANEL, because the panel is what the reader
# looks at. Setting a whole-figure aspect made a one-value small-multiple plot
# a third as tall as it was wide at double-column width -- a strip with its y
# label cut off.
WIDTH_MM = {"single": 85.0, "one_and_half": 114.0, "double": 180.0}
FONT_PT = {"S": 7.0, "M": 8.0, "L": 10.0}
LINE_PT = {"thin": 0.5, "normal": 1.0}

#: A single panel is `PANEL_ASPECT x` its own width, clamped. Small multiples
#: give every row a fixed height instead, so adding a value makes the figure
#: taller rather than squeezing the rows.
PANEL_ASPECT = 0.62
PANEL_MIN_MM, PANEL_MAX_MM = 45.0, 80.0
ROW_MM = 38.0
#: Allowances taken OUT of the preset width/height rather than added to it, so
#: "double column" stays 180 mm whatever the figure carries. Measured by eye at
#: 7-10 pt type; the automatic check in `_self_test` is what keeps them honest.
Y_AXIS_MM = 22.0          # y label + tick labels
X_AXIS_MM = 16.0          # x label + tick labels
LEGEND_MM = 26.0          # an outside-right legend
TITLE_MM = 5.0

# Arial first, then the two substitutes that have the same metrics. Matplotlib
# walks this list itself, so a machine without Arial silently lands on
# DejaVu Sans rather than failing.
FONT_STACK = ["Arial", "Liberation Sans", "DejaVu Sans"]
DEFAULT_STYLE = "default"


def panel_width_mm(preset: str, legend: bool = False) -> float:
    if preset not in WIDTH_MM:
        raise ValueError(f"unknown size preset {preset!r}; expected one of {list(WIDTH_MM)}")
    return WIDTH_MM[preset] - Y_AXIS_MM - (LEGEND_MM if legend else 0.0)


def figure_size(preset: str = "single", rows: int = 1, legend: bool = False) -> tuple:
    """`(width, height)` in INCHES: the preset's width, and a height set by the
    panels it has to hold."""
    width_mm = WIDTH_MM[preset] if preset in WIDTH_MM else None
    if width_mm is None:
        raise ValueError(f"unknown size preset {preset!r}; expected one of {list(WIDTH_MM)}")
    panel_mm = panel_width_mm(preset, legend)
    if rows <= 1:
        body_mm = min(max(PANEL_ASPECT * panel_mm, PANEL_MIN_MM), PANEL_MAX_MM)
    else:
        body_mm = ROW_MM * rows
    return (width_mm * MM, (body_mm + X_AXIS_MM + TITLE_MM) * MM)


def panel_figure(preset: str = "single", rows: int = 1, legend: bool = False,
                 sharex: bool = True, **subplot_kw):
    """**The** way a chart gets its figure. Constrained layout, panel-derived
    height, and `MaxNLocator(5)` on both axes of every panel.

    One entry point on purpose: `tight_layout()` was scattered across the chart
    builders, it ignores figure-level legends and anything drawn outside the
    axes, and the preview then re-cropped the result with
    `bbox_inches="tight"` -- so the picture on screen was not the picture that
    got exported."""
    figure, axes = plt.subplots(
        rows, 1, sharex=sharex, squeeze=False, layout="constrained",
        figsize=figure_size(preset, rows, legend), **subplot_kw)
    for axis in axes[:, 0]:
        tick_density(axis)
    return figure, axes[:, 0]


def lw(scale: float = 1.0) -> float:
    """A line width RELATIVE to the user's Lines setting.

    `base_rc` sets `lines.linewidth`, but every line in `plots.py` and
    `morphospace.py` passed an explicit `lw=`, and an explicit argument beats
    the rcParam -- so the Lines control changed nothing anyone could see
    (Round 35 shipped it that way). Each of those constants is now
    `lw(constant)`: the scale is chosen so that Normal (1 pt) reproduces
    exactly the old number, and Thin (0.5 pt) halves every line together.

    `lw=0` -- a fill with no edge -- stays a literal 0; it is not a width."""
    return float(scale) * float(mpl.rcParams["lines.linewidth"])


def tick_density(axis, bins: int = 5) -> None:
    """Five ticks a side. A 180 mm panel at 7 pt otherwise gets a dozen x
    labels that run into each other."""
    from matplotlib.ticker import MaxNLocator
    axis.xaxis.set_major_locator(MaxNLocator(nbins=bins))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=bins))


def outside_legend(figure, handles=None, labels=None, **kw):
    """One legend for the whole figure, outside the axes on the right.

    Figure-level and outside, so panels that share a set of series get a single
    legend and nothing is drawn on top of the data. The width it takes was
    already subtracted from the panel width (`LEGEND_MM`), so the figure does
    not grow."""
    if handles is None:
        return figure.legend(loc="outside right upper", frameon=False, **kw)
    return figure.legend(handles, labels, loc="outside right upper", frameon=False, **kw)


#: Display units for a time axis, as (upper bound in seconds, divisor, label).
#: Storage and export stay in seconds; only the axis changes.
TIME_UNITS = ((600.0, 1.0, "s"), (7200.0, 60.0, "min"), (float("inf"), 3600.0, "h"))


def time_scale(span_seconds: float) -> tuple:
    """`(divisor, unit)` for a time axis spanning `span_seconds`. A 5,000 s
    movie labelled in seconds is five digits per tick."""
    for bound, divisor, unit in TIME_UNITS:
        if span_seconds <= bound:
            return divisor, unit
    return TIME_UNITS[-1][1], TIME_UNITS[-1][2]


def panel_title(axis, text: str) -> None:
    """The value's name, small, at the panel's top left -- not a figure title
    and not a y label. In a small-multiple stack the y axis then carries only
    the unit, which is what varies between panels."""
    axis.set_title(text, loc="left", pad=2.0,
                   fontsize=mpl.rcParams["font.size"])


def wrap_label(text: str, axis, figure) -> str:
    """Break a y label on word boundaries when it is taller than its panel.

    Rotated 90 degrees, a y label's LENGTH runs along the panel height, so a
    long name on a 45 mm panel is simply cut off at both ends."""
    if not text or " " not in text:
        return text
    figure.canvas.draw()
    height_px = axis.get_window_extent().height
    estimate = len(text) * mpl.rcParams["font.size"] * figure.dpi / 72 * 0.5
    if estimate <= height_px:
        return text
    words = text.split()
    half = len(words) // 2 or 1
    return " ".join(words[:half]) + "\n" + " ".join(words[half:])


def available_styles() -> list:
    """Every Matplotlib style sheet, plus `default`.

    The underscore-prefixed ones (`_mpl-gallery`, `_mpl-gallery-nogrid`,
    `_classic_test_patch`) are included deliberately: they are on the
    Matplotlib style reference page, so a user who saw one there should find
    it here. There is no Epi-Quant style any more -- it only differed from
    `default` by the overrides `base_rc` used to force, and those hid what the
    chosen style actually looks like."""
    return [DEFAULT_STYLE] + sorted(plt.style.available)


#: Every character an Epi-Quant axis label can contain: the Latin alphabet, the
#: digits, and the units and signs the value labels use. A font that cannot
#: draw all of them has no business in the Font menu.
REQUIRED_GLYPHS = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                   "abcdefghijklmnopqrstuvwxyz"
                   "0123456789"
                   "µ²³°±−–·×()%/.")


def _covers_required(family: str) -> bool:
    """Whether `family`'s font file has a glyph for every character in
    `REQUIRED_GLYPHS`. Read from the font's charmap, not by rendering: a
    rendering test needs a figure per font and emits the very warnings it is
    trying to detect."""
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font
    try:
        path = font_manager.findfont(font_manager.FontProperties(family=family),
                                     fallback_to_default=False)
        charmap = FT2Font(path).get_charmap()
    except Exception:
        return False
    return all(ord(character) in charmap for character in REQUIRED_GLYPHS)


def available_fonts() -> list:
    """Font families that can actually draw an Epi-Quant label.

    Round 35 listed everything `font_manager` knew, which on macOS includes
    scripts that have no Latin at all. Picking `Muna` (Arabic, 272 glyphs)
    produced `Glyph 101 (e) missing from font(s) Muna`; picking `Kai`
    (Chinese regular script) produced `Glyph 178 (SUPERSCRIPT TWO) missing`,
    because Kai has Latin but no `µ ² ³ − –`.

    **And that is also the whole of the "it turned serif" report.** Kai is a
    calligraphic brush face; its Latin letters ARE serif-like. Nothing fell
    back to a serif family -- the labels were being drawn in Kai the entire
    time, and changing the type size merely forced the re-render that made it
    obvious. Measured: `font.family` resolves to `Kai.ttf` before and after a
    size change, with the same four missing-glyph warnings either way.

    The fix for both is the same -- do not offer a font that cannot draw the
    labels. The cache is on the result because `findfont` + `FT2Font` over
    several hundred families takes a moment, and the set cannot change while
    the app is running."""
    from matplotlib import font_manager
    global _USABLE_FONTS
    if _USABLE_FONTS is None:
        _USABLE_FONTS = sorted(name for name in set(font_manager.get_font_names())
                               if _covers_required(name))
    return list(_USABLE_FONTS)


_USABLE_FONTS = None


def installed_stack() -> list:
    """`FONT_STACK` minus anything this machine does not have.

    `Liberation Sans` is not installed on the maintainer's Mac, and leaving it
    in the list made Matplotlib log `Font family 'Liberation Sans' not found`
    on essentially every draw."""
    from matplotlib import font_manager
    present = set(font_manager.get_font_names())
    stack = [name for name in FONT_STACK if name in present]
    return stack or ["DejaVu Sans"]


def resolve_font(family: Optional[str]) -> tuple:
    """`(stack, note)` -- the font list to use and, when the asked-for family
    cannot be used, a line for the status bar. Nothing raises: a figure with
    the wrong font is better than no figure."""
    stack = installed_stack()
    if not family:
        return stack, None
    usable = available_fonts()
    if family in usable:
        return [family] + stack, None
    from matplotlib import font_manager
    if family in set(font_manager.get_font_names()):
        return stack, (f"{family} cannot draw all of the label characters "
                       f"(for example µ or ²); using {stack[0]} instead.")
    return stack, (f"{family} is not installed on this machine; "
                   f"using {stack[0]} instead.")


def base_rc(font_size: str = "M", line_width: str = "normal",
            font_family: Optional[str] = None) -> dict:
    """The rcParams this project insists on, whatever style is selected.

    **Typography and export only** (Round 35 1-2). It used to also force
    `axes.spines.top/right: False`, `legend.frameon: False`,
    `figure.facecolor` and `savefig.facecolor` to white -- which is most of
    what a style sheet IS, so every style looked like a slightly recoloured
    version of ours and `dark_background` exported as a white page. Those are
    gone; what is left cannot be left to a style:

    - `svg.fonttype: none` keeps text as text so Illustrator can edit it;
    - `svg.hashsalt` + `path.simplify: False` keep an SVG reproducible;
    - the font family, size and line width are the user's own controls."""
    size = FONT_PT[font_size] if font_size in FONT_PT else float(font_size)
    width = LINE_PT[line_width] if line_width in LINE_PT else float(line_width)
    stack, _ = resolve_font(font_family)
    return {
        # A LIST on `font.family`, not `sans-serif` plus `font.sans-serif`
        # (Round 36 4). Matplotlib does per-character fallback at the
        # `font.family` level only: through `font.sans-serif` it picks one
        # font for the whole string and draws a box for anything that font
        # lacks, which is what produced `Glyph 178 (SUPERSCRIPT TWO) missing`.
        # Measured on the same label: 4 warnings the old way, 0 this way.
        "font.family": stack,
        "font.size": size, "axes.labelsize": size, "axes.titlesize": size,
        "xtick.labelsize": size, "ytick.labelsize": size, "legend.fontsize": size,
        "figure.titlesize": size + 2,
        "lines.linewidth": width, "patch.linewidth": min(width, LINE_PT["thin"]),
        "svg.fonttype": "none", "svg.hashsalt": "epiquant", "path.simplify": False,
    }


@contextmanager
def style_context(style: str = DEFAULT_STYLE, font_size: str = "M",
                  line_width: str = "normal", font_family: Optional[str] = None):
    """Draw inside this. The named style goes on first, then `base_rc`."""
    overrides = base_rc(font_size, line_width, font_family)
    if style and style != DEFAULT_STYLE:
        with plt.style.context(style), mpl.rc_context(overrides):
            yield
    else:
        with plt.style.context("default"), mpl.rc_context(overrides):
            yield


# ---------------------------------------------------------------- SVG export
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"


def unclip(figure) -> None:
    """Turn off clipping on everything but images.

    Matplotlib clips each artist to its axes, which Illustrator imports as a
    clipping mask per artist -- an SVG that is unusable for hand editing. Images
    are left alone: a raster that overflows its axes is a visible bug, not a
    tidier file."""
    from matplotlib.image import AxesImage
    for axis_set in figure.axes:
        artists = list(axis_set.get_children()) + axis_set.get_xgridlines() + axis_set.get_ygridlines()
        for axis in (axis_set.xaxis, axis_set.yaxis):
            artists += list(axis.get_ticklines())
            artists += [tick.gridline for tick in axis.get_major_ticks() + axis.get_minor_ticks()]
        for artist in artists:
            if isinstance(artist, AxesImage):
                continue
            try:
                artist.set_clip_on(False)
            except Exception:
                pass


def _parent_map(root) -> dict:
    return {child: parent for parent in root.iter() for child in parent}


def flatten_svg(path: str) -> dict:
    """Rewrite an SVG in place so Illustrator opens it with as few groups and
    clipping masks as possible, and report what is left.

    Ported from `figures/revision_2026-10/fig2/scripts/figkit.py:flatten_svg`,
    which uses `lxml`; this project does not depend on lxml, so the same
    rewrite is done with the standard library's ElementTree and an explicit
    parent map (ElementTree has no `getparent`). The self-test pins it by
    running it over an SVG the lxml version already produced and asserting it
    changes nothing -- the two cannot be run side by side on this machine,
    since lxml is not installed.

    What it does: `<use>` references become explicit `<path>` elements (markers
    are otherwise a `<defs>` entry Illustrator turns into a symbol), unused
    marker definitions and empty `<defs>` go, attribute-less `<g>` wrappers are
    dissolved, the metadata block is dropped, and every font-family becomes
    Arial."""
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    tree = ET.parse(path)
    root = tree.getroot()

    for metadata in root.findall(f"{{{SVG_NS}}}metadata"):
        root.remove(metadata)

    definitions = {node.get("id"): node for node in root.iter(f"{{{SVG_NS}}}path") if node.get("id")}
    parents = _parent_map(root)
    for use in list(root.iter(f"{{{SVG_NS}}}use")):
        reference = use.get(f"{{{XLINK_NS}}}href", use.get("href", "")).lstrip("#")
        if reference not in definitions:
            continue
        replacement = ET.Element(f"{{{SVG_NS}}}path")
        replacement.set("d", definitions[reference].get("d"))
        style = ";".join(s for s in [definitions[reference].get("style"), use.get("style")] if s)
        if style:
            replacement.set("style", style)
        transform = f"translate({use.get('x', '0')} {use.get('y', '0')})"
        if use.get("transform"):
            transform = use.get("transform") + " " + transform
        replacement.set("transform", transform)
        parent = parents[use]
        parent[list(parent).index(use)] = replacement

    parents = _parent_map(root)
    for node in list(definitions.values()):
        parent = parents.get(node)
        if parent is not None and parent.tag == f"{{{SVG_NS}}}defs":
            parent.remove(node)
    parents = _parent_map(root)
    for defs in list(root.iter(f"{{{SVG_NS}}}defs")):
        if len(defs) == 0 and defs in parents:
            parents[defs].remove(defs)

    changed = True
    while changed:
        changed = False
        parents = _parent_map(root)
        for group in list(root.iter(f"{{{SVG_NS}}}g")):
            parent = parents.get(group)
            if parent is None or set(group.attrib) - {"id"}:
                continue
            position = list(parent).index(group)
            for offset, child in enumerate(list(group)):
                parent.insert(position + offset, child)
            parent.remove(group)
            changed = True

    text = ET.tostring(root, encoding="unicode")
    text = re.sub(r"font-family: [^;\"]*", "font-family: 'Arial'", text)
    text = text.replace("'Liberation Sans'", "'Arial'")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0" encoding="utf-8"?>\n')
        handle.write(text)

    sizes = [float(v) for v in re.findall(r"font-size: ([\d.]+)px", text)]
    strokes = [float(v) for v in re.findall(r"stroke-width: ([\d.]+)", text)]
    return {
        "groups": text.count("<g"),
        "clip_paths": text.count("<clipPath"),
        "font_pt": (min(sizes), max(sizes)) if sizes else (None, None),
        "stroke_pt": (min(strokes), max(strokes)) if strokes else (None, None),
    }


def save(figure, path: str, png_dpi: int = 600, close: bool = True) -> dict:
    """Write `path` (.png or .svg). SVG goes through `unclip` + `flatten_svg`;
    PNG is written at `png_dpi`. Returns the SVG report, or `{}` for a PNG.

    Never overwrites -- every caller in this project timestamps its filenames,
    and a silent overwrite of an exported figure is exactly the kind of loss
    that is noticed a week later.

    The save-time rcParams are re-applied here rather than relied on from
    `style_context`. Type size and colour are baked into the artists when they
    are created, but `svg.fonttype` is read by `savefig` -- so a figure built
    inside the context and saved outside it came out with its text converted to
    paths, which is the one thing the SVG export exists to avoid.

    **The background comes from the figure, not from here** (Round 35 1-2).
    This used to force `savefig.facecolor: white`, so a `dark_background`
    figure exported as white paper with a black plot box on it."""
    if os.path.exists(path):
        raise FileExistsError(f"{path} already exists; exports never overwrite")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save_time = {"svg.fonttype": "none", "svg.hashsalt": "epiquant",
                 "path.simplify": False,
                 "savefig.facecolor": figure.get_facecolor(),
                 "savefig.edgecolor": figure.get_edgecolor()}
    if path.lower().endswith(".svg"):
        unclip(figure)
        with mpl.rc_context(save_time):
            figure.savefig(path, format="svg")
        report = flatten_svg(path)
    else:
        with mpl.rc_context(save_time):
            figure.savefig(path, dpi=png_dpi)
        report = {}
    if close:
        plt.close(figure)
    return report


def rasterize_threshold(n_points: int, limit: int = 5000) -> bool:
    """Whether a scatter of `n_points` should be rasterized inside the SVG. A
    vector SVG with 50,000 circles in it opens, eventually, and then cannot be
    edited."""
    return n_points > limit


# ---------------------------------------------------------------- xlsx export
EXCEL_MAX_ROWS = 1_048_576
EXCEL_MAX_COLUMNS = 16_384
_SHEET_FORBIDDEN = re.compile(r"[\[\]:*?/\\]")


def safe_sheet_name(name: str, taken=()) -> str:
    """A legal, unique sheet name: forbidden characters out, 31 characters max,
    a numeric suffix on a collision. `Horizontal/Vertical Length Ratio` is the
    case this exists for -- openpyxl raises on the slash, and truncation alone
    would collide `Local Mean Shape Index` with `Local Mean Solidity` on a
    longer panel."""
    cleaned = _SHEET_FORBIDDEN.sub("-", str(name)).strip() or "Sheet"
    candidate = cleaned[:31]
    if candidate not in taken:
        return candidate
    for suffix in range(2, 1000):
        tail = f" ({suffix})"
        candidate = cleaned[:31 - len(tail)] + tail
        if candidate not in taken:
            return candidate
    raise ValueError(f"cannot make a unique sheet name from {name!r}")


def write_tables(output_path: str, tables: dict, index: bool = True) -> dict:
    """Write `{name: DataFrame}` to one workbook, and say what had to change.

    Three limits force a rewrite rather than an error:

    - a sheet name is cleaned and truncated (above), and a `Sheets` sheet maps
      every written name back to what the caller asked for -- without it, a
      truncated name is a dead end for whoever opens the file;
    - a frame wider than 16,384 columns (a Frame x Track pivot of a large movie
      is 7,000 columns and climbing) is written as the long form
      `Frame, Track_ID, value` instead;
    - a frame with more rows than a sheet holds is written as `.csv` beside the
      workbook, because there is no rewrite that fits.

    Never overwrites."""
    if os.path.exists(output_path):
        raise FileExistsError(f"{output_path} already exists; exports never overwrite")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    notes, mapping, written, csv_files = [], [], {}, []
    for name, table in tables.items():
        frame = table
        if frame.shape[1] + (1 if index else 0) > EXCEL_MAX_COLUMNS:
            frame = frame.stack(future_stack=True).rename("value").reset_index()
            notes.append(f"{name}: {table.shape[1]} columns exceed Excel's "
                         f"{EXCEL_MAX_COLUMNS}, written as a long table")
        if len(frame) + 1 > EXCEL_MAX_ROWS:
            csv_path = os.path.splitext(output_path)[0] + f"_{safe_sheet_name(name)}.csv"
            frame.to_csv(csv_path, index=index)
            csv_files.append(csv_path)
            notes.append(f"{name}: {len(frame)} rows exceed Excel's {EXCEL_MAX_ROWS}, "
                         f"written as {os.path.basename(csv_path)}")
            continue
        sheet = safe_sheet_name(name, written)
        written[sheet] = frame
        mapping.append({"Sheet": sheet, "Value": str(name)})

    with pd.ExcelWriter(output_path) as writer:
        pd.DataFrame(mapping).to_excel(writer, sheet_name="Sheets", index=False)
        for sheet, frame in written.items():
            frame.to_excel(writer, sheet_name=sheet, index=index)
    return {"path": output_path, "sheets": list(written), "csv_files": csv_files, "notes": notes}


# ---------------------------------------------------------------- units
def timestamped(directory: str, chart: str, extension: str, when=None) -> str:
    """`{chart}_{YYYYmmdd-HHMMSS}.{ext}` under `directory`, with a `-2`, `-3`
    ... suffix if that name is taken.

    The timestamp is what makes "never overwrite" workable: a user exporting
    the same chart twice gets two files, not a refusal. The suffix is what
    makes it workable at the second boundary -- the stamp has one-second
    resolution, and exporting a PNG and then an SVG of the same chart takes
    well under a second, so without it the second export was refused for
    colliding with the first."""
    import datetime
    stamp = (when or datetime.datetime.now()).strftime("%Y%m%d-%H%M%S")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", chart).strip("_") or "chart"
    suffix = extension.lstrip(".")
    candidate = os.path.join(directory, f"{safe}_{stamp}.{suffix}")
    attempt = 2
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{safe}_{stamp}-{attempt}.{suffix}")
        attempt += 1
    return candidate


def _self_test() -> None:
    import tempfile

    # 1. Sizes are physical, and the HEIGHT comes from the panel (R35 0-C).
    assert abs(figure_size("single")[0] - 85 / 25.4) < 1e-12
    assert abs(figure_size("double")[0] - 180 / 25.4) < 1e-12
    for preset in WIDTH_MM:
        width, height = figure_size(preset)
        assert height < width, preset
        panel_mm = min(max(PANEL_ASPECT * panel_width_mm(preset), PANEL_MIN_MM), PANEL_MAX_MM)
        assert PANEL_MIN_MM - 1e-9 <= panel_mm <= PANEL_MAX_MM + 1e-9, (preset, panel_mm)
    # A single panel at double-column width must not be a strip: that was
    # screenshot 1, where a whole-figure 2/3 aspect left it 60 mm tall.
    wide_w, wide_h = figure_size("double")
    assert wide_h / wide_w > 0.5, (wide_w, wide_h)
    # More values make the figure taller; they do not squeeze the rows.
    assert figure_size("single", rows=3)[1] > figure_size("single", rows=1)[1]
    assert abs((figure_size("single", rows=4)[1] - figure_size("single", rows=2)[1])
               - 2 * ROW_MM * MM) < 1e-9
    # An outside legend is taken OUT of the preset width, never added to it.
    assert figure_size("double", legend=True)[0] == figure_size("double")[0]
    assert panel_width_mm("double", legend=True) < panel_width_mm("double")

    # The time axis picks its unit from the span; storage stays in seconds.
    assert time_scale(0)[1] == "s" and time_scale(600)[1] == "s"
    assert time_scale(601)[1] == "min" and time_scale(7200)[1] == "min"
    assert time_scale(7201)[1] == "h"
    small, large = base_rc("S"), base_rc("L")
    assert small["font.size"] == 7 and large["font.size"] == 10
    assert small["lines.linewidth"] == large["lines.linewidth"], "font size must not move line width"
    assert base_rc("M", "thin")["lines.linewidth"] == 0.5

    # 2. A style sheet decides colours; it never decides our SVG text mode or
    #    our type size. `dark_background` sets both, so it is the test case.
    with style_context("dark_background", font_size="L"):
        assert mpl.rcParams["svg.fonttype"] == "none", "a style overrode svg.fonttype"
        assert mpl.rcParams["font.size"] == 10, "a style overrode the font size"
        assert mpl.rcParams["axes.facecolor"] != "white", "the style did not apply at all"
    assert available_styles()[0] == DEFAULT_STYLE
    styles = available_styles()
    for expected in ("ggplot", "dark_background", "_mpl-gallery", "_classic_test_patch"):
        assert expected in styles, f"{expected} missing from the style list"
    assert len(styles) == len(plt.style.available) + 1

    # R35 1-2: base_rc must not impose what a style sheet is FOR. Forcing
    # spines off and the background white made every style look like a
    # recolour of ours, and exported `dark_background` as a white page.
    forbidden = {"axes.spines.top", "axes.spines.right", "axes.spines.left",
                 "axes.spines.bottom", "figure.facecolor", "savefig.facecolor",
                 "legend.frameon", "axes.facecolor", "axes.grid"}
    leaked = forbidden & set(base_rc())
    assert not leaked, f"base_rc still overrides style territory: {sorted(leaked)}"

    # 3. (R36 4) The font menu offers only fonts that can draw a label, and
    #    every one of them really can -- checked by rendering, with warnings
    #    turned into the assertion.
    import warnings as _warnings

    stack, note = resolve_font("Nonexistent Face 12345")
    assert stack == installed_stack() and note and "not installed" in note, (stack, note)
    assert resolve_font(None) == (installed_stack(), None)
    usable = available_fonts()
    assert usable, "no usable font found at all"
    from matplotlib import font_manager
    every = sorted(set(font_manager.get_font_names()))
    assert len(usable) < len(every), "the filter removed nothing, so it is not running"
    for broken in ("Kai", "Muna"):
        if broken in every:
            assert broken not in usable, f"{broken} cannot draw our labels and must not be offered"
            _, why = resolve_font(broken)
            assert why and "label characters" in why, why

    label = "Cell Area (µm²) ±1 · 50% −3 × 10³ 20°"
    complaints = []
    for family in usable:
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            with style_context(font_family=family):
                figure, axes = panel_figure("single")
                axes[0].set_ylabel(label)
                axes[0].set_xlabel(label)
                figure.canvas.draw()
            plt.close(figure)
        missing = [str(w.message) for w in caught if "Glyph" in str(w.message)]
        if missing:
            complaints.append(f"{family}: {missing[0][:60]}")
    assert not complaints, "fonts in the menu that cannot draw a label:\n  " + "\n  ".join(complaints)
    print(f"  fonts: {len(usable)} offered of {len(every)} installed "
          f"({len(every) - len(usable)} filtered out); all render "
          f"{len(REQUIRED_GLYPHS)} required glyphs with no warning")

    # `font.family` must be the LIST, or Matplotlib does no per-glyph fallback.
    assert isinstance(base_rc()["font.family"], list), base_rc()["font.family"]
    assert "font.sans-serif" not in base_rc()

    with tempfile.TemporaryDirectory() as tmp:
        # 3b. (R35 1-2) An exported PNG must carry the STYLE's background.
        from PIL import Image
        for style in ("default", "ggplot", "dark_background"):
            with style_context(style):
                figure, axes = panel_figure("single")
                axes[0].plot([0, 1, 2], [0, 1, 0])
                expected = mpl.colors.to_rgb(mpl.rcParams["figure.facecolor"])
            path = os.path.join(tmp, f"bg_{style}.png")
            save(figure, path, png_dpi=72)
            corner = Image.open(path).convert("RGB").getpixel((2, 2))
            assert all(abs(a * 255 - b) < 3 for a, b in zip(expected, corner)), (
                f"{style}: exported corner {corner}, style says "
                f"{tuple(round(v * 255) for v in expected)}")
        print("  style background survives export: default / ggplot / dark_background")

        # 4. SVG export: no clip paths left, text still text.
        with style_context():
            figure, axis = plt.subplots(figsize=figure_size("single"))
            axis.plot([0, 1, 2], [0, 1, 0], marker="o", label="demo")
            axis.set_xlabel("Frame")
            axis.legend()
        path = os.path.join(tmp, "demo.svg")
        report = save(figure, path)
        body = open(path, encoding="utf-8").read()
        assert report["clip_paths"] == 0, f"{report['clip_paths']} clipPath(s) left"
        assert "<use" not in body, "a <use> survived"
        assert "Frame" in body, "svg.fonttype: none should leave the label as text"
        assert "Arial" in body
        try:
            save(plt.figure(), path)
            raise AssertionError("save must refuse to overwrite")
        except FileExistsError:
            pass
        print(f"  svg: {report['groups']} group(s), {report['clip_paths']} clipPath(s), "
              f"font {report['font_pt']} pt, stroke {report['stroke_pt']} pt")

        # 5. flatten_svg vs the lxml original. The original cannot be run here
        #    (no lxml), so it is pinned the other way round: its own output,
        #    re-flattened, must come out unchanged.
        import shutil
        reference = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "figures/revision_2026-10/fig2/svg/Fig2_variance_explained_cumulative.svg")
        if os.path.exists(reference):
            copy = os.path.join(tmp, "already_flattened.svg")
            shutil.copy(reference, copy)
            before = open(copy, encoding="utf-8").read()
            again = flatten_svg(copy)
            after = open(copy, encoding="utf-8").read()
            assert again["clip_paths"] == before.count("<clipPath"), again
            assert again["groups"] == before.count("<g"), (again["groups"], before.count("<g"))
            assert "<use" not in after
            print(f"  flatten_svg: idempotent on figkit's own output "
                  f"({again['groups']} groups, {again['clip_paths']} clipPaths, "
                  f"{len(after) - len(before):+d} bytes)")
        else:
            print("  flatten_svg: reference SVG not present, idempotence not checked")

        # 6. Excel limits.
        wide = pd.DataFrame(np.zeros((3, EXCEL_MAX_COLUMNS + 5)),
                            columns=[f"t{i}" for i in range(EXCEL_MAX_COLUMNS + 5)])
        wide.index.name = "Frame"
        wide.columns.name = "Track_ID"
        result = write_tables(os.path.join(tmp, "tables.xlsx"), {
            "Horizontal/Vertical Length Ratio": pd.DataFrame({"v": [1, 2]}),
            "Local Mean Shape Index and then some more words": pd.DataFrame({"v": [1]}),
            "Local Mean Shape Index and then some other words": pd.DataFrame({"v": [1]}),
            "too wide": wide,
        })
        assert all("/" not in s and len(s) <= 31 for s in result["sheets"]), result["sheets"]
        assert len(set(result["sheets"])) == len(result["sheets"]), "sheet names collided"
        book = pd.ExcelFile(result["path"])
        assert book.sheet_names[0] == "Sheets", book.sheet_names
        back = book.parse("Sheets").set_index("Sheet")["Value"].to_dict()
        assert "Horizontal/Vertical Length Ratio" in back.values(), back
        long_sheet = [s for s, v in back.items() if v == "too wide"][0]
        assert book.parse(long_sheet).shape[1] <= 4, "a too-wide frame must be written long"
        assert any("long table" in n for n in result["notes"]), result["notes"]
        print(f"  xlsx: {len(result['sheets'])} sheet(s) written; "
              + "; ".join(result["notes"]))

    # 7. The manuscript genotype colours, which `plotting_style.py` contradicts.
    assert GENOTYPE["WT"] == "#2F6A46" and GENOTYPE["pyd"] == "#BD7338"
    assert GENOTYPE["tmc"] == "#E0989E"
    name = timestamped("/out", "Clone cell data", "png")
    assert re.fullmatch(r"/out/Clone_cell_data_\d{8}-\d{6}\.png", name), name
    # Two exports of one chart land in the same second; the second must get a
    # name of its own rather than being refused for colliding with the first.
    with tempfile.TemporaryDirectory() as tmp:
        first = timestamped(tmp, "chart", "png")
        open(first, "w").close()
        second = timestamped(tmp, "chart", "png")
        assert second != first and second.endswith("-2.png"), (first, second)
    print("  palette: WT #2F6A46 / pyd #BD7338 / tmc #E0989E (manuscript)")
    print("\nplotkit.py self-test: OK")


if __name__ == "__main__":
    _self_test()
