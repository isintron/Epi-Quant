"""Chart builders for the Trajectory and Data-playground steps.

Data in, `(figure, note)` out -- no file writing, no base64, no `webview`.
Physical size, layout, ticks and export are `plotkit`'s; what belongs here is
the shape of each chart and the statements it is allowed to make.

Two templates, as agreed:

- **Columns** -- scatter / box / violin / estimation over a handful of groups.
  Paired data is drawn as aligned columns with NO jitter, so the eye can follow
  a pair across; jitter is only offered for unpaired groups, where it stops
  points hiding each other.
- **XY** -- one thin line per track plus a thick mean, with an error band the
  caller chooses and the legend names. "Error band" without saying which is the
  most common way a figure misleads, so the band is never drawn unlabelled.

**Nothing explanatory is drawn inside a figure** (Round 35 0-C). Kept-vs-total
counts, the error-band name, filter settings and warnings are returned as a
`note` for the panel's status line and written into the export's Provenance
sheet. They used to be `set_title(..., loc="left")`, which put a sentence of
prose where the reader expects the data -- and then had to be cropped out of
every exported figure by hand.

Every builder gets its figure from `plotkit.panel_figure`, so one layout
engine serves all of them: constrained layout, a height set by the panels
rather than by the figure, five ticks a side, and a legend outside the axes.
"""

from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

import plotkit

ERROR_BANDS = ("SD", "SEM", "95% CI", "min-max", "none")


def band(values: np.ndarray, kind: str):
    """`(lower, upper, label)` for a column-wise error band over a
    `frames x tracks` array, ignoring NaN.

    `95% CI` is the normal-approximation interval over TRACKS, which is a
    statement about the mean of these tracks in this movie -- not about the
    genotype. Between-embryo comparisons bootstrap embryos instead
    (`dynamics_compare`), because cells inside one embryo are not independent."""
    if kind not in ERROR_BANDS:
        raise ValueError(f"unknown error band {kind!r}; expected one of {ERROR_BANDS}")
    mean = np.nanmean(values, axis=1)
    if kind == "none":
        return None, None, None
    if kind == "min-max":
        return np.nanmin(values, axis=1), np.nanmax(values, axis=1), "min-max"
    count = np.sum(~np.isnan(values), axis=1)
    deviation = np.nanstd(values, axis=1, ddof=1)
    if kind == "SD":
        return mean - deviation, mean + deviation, "mean +/- SD"
    error = deviation / np.sqrt(np.maximum(count, 1))
    if kind == "SEM":
        return mean - error, mean + error, "mean +/- SEM"
    return mean - 1.96 * error, mean + 1.96 * error, "mean +/- 95% CI (over tracks)"


def time_axis(values, is_time: bool) -> tuple:
    """`(scaled values, label)` for an x axis.

    Seconds are the stored unit everywhere, but a 5,000 s movie labelled in
    seconds puts five digits under every tick, so the DISPLAY unit follows the
    span (`plotkit.time_scale`). The exported `Time` column stays in seconds
    and says so in its header."""
    values = np.asarray(values, float)
    if not is_time or len(values) == 0:
        return values, "Frame"
    divisor, unit = plotkit.time_scale(float(np.nanmax(values) - np.nanmin(values)))
    return values / divisor, f"Time ({unit})"


def _unit_only(label: str) -> str:
    """`Cell Area (um^2)` -> `um^2`, for a small-multiple y axis whose panel
    title already carries the name. A dimensionless value gets no y label at
    all rather than a repeat of its own name."""
    if label.endswith(")") and "(" in label:
        return label[label.rindex("(") + 1:-1]
    return ""


def single_track(long: pd.DataFrame, columns: Sequence[str], labels: dict = None,
                 layout: str = "small_multiples", size_preset: str = "single",
                 is_time: bool = True):
    """One track, several values over time. Returns `(figure, note)`.

    `layout`: `"small_multiples"` (one row per value, shared x) or
    `"dual_axis"` (exactly two values, left and right y). Two values on one
    pair of axes is readable; three is a spaghetti plot with no shared scale,
    which is why `dual_axis` refuses anything but two.

    `columns` is in the order the user arranged the chips, and that order is
    the drawing order."""
    columns = list(columns)
    labels = labels or {}
    x, x_label = time_axis(long["Time" if is_time else "Frame"].to_numpy(), is_time)

    if layout == "dual_axis":
        if len(columns) != 2:
            raise ValueError("The two-y-axis layout takes exactly two values; "
                             f"{len(columns)} selected. Use small multiples instead.")
        figure, axes = plotkit.panel_figure(size_preset, rows=1)
        left = axes[0]
        right = left.twinx()
        plotkit.tick_density(right)
        for axis, column, colour in ((left, columns[0], plotkit.LE),
                                     (right, columns[1], plotkit.AS)):
            axis.plot(x, long[column].to_numpy(float), color=colour)
            axis.set_ylabel(plotkit.wrap_label(labels.get(column, column), axis, figure),
                            color=colour)
            axis.tick_params(axis="y", colors=colour)
        left.set_xlabel(x_label)
        return figure, None

    if layout != "small_multiples":
        raise ValueError(f"unknown layout {layout!r}")
    figure, axes = plotkit.panel_figure(size_preset, rows=len(columns))
    for index, (axis, column) in enumerate(zip(axes, columns)):
        axis.plot(x, long[column].to_numpy(float), color=plotkit.INK)
        full = labels.get(column, column)
        if len(columns) > 1:
            # The name goes above the panel; the y axis carries only what
            # differs between panels, which is the unit.
            plotkit.panel_title(axis, column)
            axis.set_ylabel(_unit_only(full))
        else:
            axis.set_ylabel(plotkit.wrap_label(full, axis, figure))
        if index == len(columns) - 1:
            axis.set_xlabel(x_label)
    return figure, None


def all_tracks(wide: pd.DataFrame, value_label: str, error: str = "SD",
               time: np.ndarray = None, is_time: bool = True,
               kept: tuple = None, colour: str = None, size_preset: str = "single",
               axis=None, group_label: str = None, panel_name: str = None):
    """Every track as a thin line plus the mean and an error band.

    `wide`: `Frame x Track` (what `cell_table.pivot` returns). `kept`:
    `(kept, total)`, returned in the note -- a plot of "the tracks that
    survived the coverage filter" is a biased sample, and saying so is the only
    defence.

    The thin lines go in one `LineCollection` and are rasterized above 200 of
    them: a vector SVG with 7,000 polylines does not open in Illustrator."""
    own = axis is None
    if own:
        figure, axes = plotkit.panel_figure(size_preset, rows=1,
                                            legend=group_label is not None)
        axis = axes[0]
    else:
        figure = axis.figure
    raw = np.asarray(time if time is not None else wide.index, float)
    x, x_label = time_axis(raw, is_time)
    values = wide.to_numpy(float)
    colour = colour or plotkit.LE

    segments = [np.column_stack([x, values[:, i]])[~np.isnan(values[:, i])]
                for i in range(values.shape[1])]
    segments = [s for s in segments if len(s) > 1]
    axis.add_collection(LineCollection(
        segments, colors=colour, linewidths=plotkit.lw(0.3), alpha=0.18, zorder=1,
        rasterized=len(segments) > 200))

    mean = np.nanmean(values, axis=1)
    low, high, band_label = band(values, error)
    if low is not None:
        axis.fill_between(x, low, high, color=colour, alpha=0.25, lw=0, zorder=2)
    axis.plot(x, mean, color=colour, lw=plotkit.lw(1.4), zorder=3, label=group_label or "mean")
    axis.set_xlabel(x_label)
    if panel_name:
        plotkit.panel_title(axis, panel_name)
        axis.set_ylabel(_unit_only(value_label))
    else:
        axis.set_ylabel(plotkit.wrap_label(value_label, axis, figure))
    axis.autoscale_view()

    note = []
    if kept:
        note.append(f"{kept[0]} of {kept[1]} tracks")
    if band_label:
        note.append(band_label)
    return figure, ", ".join(note) or None


def cumulative_events(counts: pd.DataFrame, is_time: bool = True,
                      colours: dict = None, censored_from=None,
                      size_preset: str = "single"):
    """Cumulative normalised event counts -- T1, division, delamination.

    `counts`: indexed by the frame (or time) a transition STARTS at, with `N`
    (the tracked, non-margin cells present in both frames of that pair) and one
    count column per event type. The curve is `C(t) = sum_{f <= t} n(f)/N(f)`,
    i.e. events per tracked cell, the normalisation Nat Commun 2023
    (s41467-023-41178-w) uses -- without it the curve mostly tracks how many
    cells were being followed.

    `censored_from`: the x value beyond which an event cannot yet be judged
    settled, because `T1 must hold` requires it to persist for tau and there is
    less than tau left in the window. That stretch is SHADED -- the shading is
    data, not prose, so it stays; the sentence explaining it goes to the status
    line with everything else."""
    colours = colours or {}
    figure, axes = plotkit.panel_figure(size_preset, rows=1, legend=True)
    axis = axes[0]
    raw = counts.index.to_numpy(float)
    divisor, unit = ((plotkit.time_scale(float(np.nanmax(raw) - np.nanmin(raw))))
                     if is_time and len(raw) else (1.0, None))
    x = raw / divisor
    x_label = f"Time ({unit})" if unit else "Frame"
    population = counts["N"].to_numpy(float)
    for column in counts.columns:
        if column == "N":
            continue
        rate = np.divide(counts[column].to_numpy(float), population,
                         out=np.zeros(len(counts)), where=population > 0)
        axis.plot(x, np.cumsum(rate), label=column, color=colours.get(column))
    note = None
    if censored_from is not None:
        # `censored_from` arrives in the index's own units, so it takes the
        # same divisor the axis did.
        censored_x = censored_from / divisor
        if censored_x < x.max():
            axis.axvspan(censored_x, x.max(), color=plotkit.LIGHT_GREY, lw=0, zorder=0)
            note = ("the shaded tail is not yet judgeable: an event there cannot "
                    "have been seen to hold")
    axis.set_xlabel(x_label)
    axis.set_ylabel(plotkit.wrap_label("Cumulative events per tracked cell", axis, figure))
    plotkit.outside_legend(figure)
    return figure, note


def columns_plot(groups: dict, value_label: str, kind: str = "box",
                 paired: bool = False, colours: dict = None,
                 size_preset: str = "single", estimation: dict = None):
    """The Columns template: one column per group.

    `kind`: `scatter`, `box`, `violin` or `estimation`.

    `paired=True` draws the points at the exact column centre and joins each
    pair with a line -- the whole point of paired data is reading one subject
    across the columns, and jitter breaks that. Unpaired groups get jitter so
    points do not hide each other.

    `estimation` carries a precomputed `{group: (point, low, high)}`, from the
    caller's own bootstrap, because what counts as a resampling unit is not
    something a plotting function can know."""
    colours = colours or {}
    names = list(groups)
    figure, axes = plotkit.panel_figure(size_preset, rows=1)
    axis = axes[0]
    rng = np.random.default_rng(0)
    series = [np.asarray(groups[name], float) for name in names]

    if kind == "box":
        artists = axis.boxplot(series, positions=np.arange(len(names)), widths=0.6,
                               showfliers=False, patch_artist=True)
        for patch, name in zip(artists["boxes"], names):
            patch.set_facecolor(colours.get(name, plotkit.LIGHT_GREY))
            patch.set_edgecolor(plotkit.INK)
        for part in ("medians", "whiskers", "caps"):
            for line in artists[part]:
                line.set_color(plotkit.INK)
    elif kind == "violin":
        artists = axis.violinplot(series, positions=np.arange(len(names)),
                                  showmeans=False, showextrema=False)
        for body, name in zip(artists["bodies"], names):
            body.set_facecolor(colours.get(name, plotkit.LIGHT_GREY))
            body.set_alpha(0.8)
    elif kind not in ("scatter", "estimation"):
        raise ValueError(f"unknown kind {kind!r}")

    if kind in ("scatter", "estimation") or paired:
        for position, (name, values) in enumerate(zip(names, series)):
            offset = (np.zeros(len(values)) if paired
                      else rng.uniform(-0.07, 0.07, len(values)))
            axis.scatter(position + offset, values, s=10, facecolor="none",
                         edgecolor=colours.get(name, plotkit.INK),
                         linewidths=plotkit.lw(0.5), zorder=2)
    if paired and len(names) > 1:
        length = min(len(v) for v in series)
        for row in range(length):
            axis.plot(np.arange(len(names)), [v[row] for v in series],
                      color=plotkit.MID_GREY, lw=plotkit.lw(0.3), alpha=0.6, zorder=1)
    if estimation:
        for position, name in enumerate(names):
            point, low, high = estimation[name]
            axis.plot([position + 0.25] * 2, [low, high], color=colours.get(name, plotkit.INK),
                      lw=plotkit.lw(1.0), solid_capstyle="butt", zorder=3)
            axis.plot(position + 0.25, point, marker="o", ms=4, mew=0,
                      color=colours.get(name, plotkit.INK), zorder=4)

    axis.set_xticks(np.arange(len(names)))
    axis.set_xticklabels(names)
    axis.set_ylabel(plotkit.wrap_label(value_label, axis, figure))
    plotkit.tick_density(axis)
    axis.xaxis.set_major_locator(plt.FixedLocator(np.arange(len(names))))
    return figure, None


def superplot(cells: pd.DataFrame, value: str, value_label: str,
              group_column: str = "Group", unit_column: str = "Individual",
              colours: dict = None, intervals: dict = None, paired: bool = False,
              size_preset: str = "single"):
    """The Columns template as the manuscript's Fig. 3D draws it: a SuperPlot.

    Three layers, in increasing order of what they are evidence for:

    - small translucent points: **one per cell**, which is the sample size the
      eye tends to read and almost never the one that counts;
    - one large point per **individual** (`unit_column`) -- a folder, or a
      cluster's member, depending on the mode;
    - a bar at the mean of the individual means, with +/- SD of THOSE means.

    `intervals` is `{group: (point, low, high)}` from the caller's own
    bootstrap over individuals, drawn only where the caller supplies one: with
    a single individual in a group there is nothing to resample and the panel
    says so instead of inventing an interval.

    `paired` drops the jitter and joins each individual across the columns
    (Round 34 2-6) -- with paired data the point is reading one subject
    across, which jitter destroys."""
    colours = colours or {}
    groups = list(dict.fromkeys(cells[group_column]))
    figure, axes = plotkit.panel_figure(size_preset, rows=1)
    axis = axes[0]
    rng = np.random.default_rng(0)

    unit_means = (cells.groupby([group_column, unit_column])[value]
                  .mean().rename("mean").reset_index())
    for position, group in enumerate(groups):
        colour = colours.get(group, plotkit.MID_GREY)
        subset = cells[cells[group_column] == group]
        offset = (np.zeros(len(subset)) if paired
                  else rng.uniform(-0.18, 0.18, len(subset)))
        axis.scatter(position + offset, subset[value].to_numpy(float), s=5, lw=0,
                     color=colour, alpha=0.25, zorder=1,
                     rasterized=plotkit.rasterize_threshold(len(subset)))
        means = unit_means[unit_means[group_column] == group]["mean"].to_numpy(float)
        spread = (np.zeros(len(means)) if paired
                  else rng.uniform(-0.09, 0.09, len(means)))
        axis.scatter(position + spread, means, s=28, zorder=3, color=colour,
                     edgecolor=plotkit.INK, linewidths=plotkit.lw(0.5))
        if len(means):
            centre = float(np.mean(means))
            axis.plot([position - 0.3, position + 0.3], [centre] * 2,
                      color=plotkit.INK, lw=plotkit.lw(1.4), zorder=4,
                      solid_capstyle="butt")
            if len(means) > 1:
                deviation = float(np.std(means, ddof=1))
                axis.plot([position] * 2, [centre - deviation, centre + deviation],
                          color=plotkit.INK, lw=plotkit.lw(1.0), zorder=4,
                          solid_capstyle="butt")
    if paired:
        for unit, rows in unit_means.groupby(unit_column):
            ordered = [rows[rows[group_column] == g]["mean"] for g in groups]
            if all(len(v) for v in ordered):
                axis.plot(np.arange(len(groups)), [float(v.iloc[0]) for v in ordered],
                          color=plotkit.MID_GREY, lw=plotkit.lw(0.3), alpha=0.6, zorder=2)
    if intervals:
        for position, group in enumerate(groups):
            if group not in intervals:
                continue
            point, low, high = intervals[group]
            axis.plot([position + 0.34] * 2, [low, high], color=plotkit.INK,
                      lw=plotkit.lw(1.0), solid_capstyle="butt", zorder=5)
            axis.plot(position + 0.34, point, marker="o", ms=4, mew=0,
                      color=plotkit.INK, zorder=6)

    axis.set_xticks(np.arange(len(groups)))
    axis.set_xticklabels(groups)
    axis.set_xlim(-0.6, len(groups) - 0.4)
    axis.set_ylabel(plotkit.wrap_label(value_label, axis, figure))
    plotkit.tick_density(axis)
    axis.xaxis.set_major_locator(plt.FixedLocator(np.arange(len(groups))))
    counts = ", ".join(
        f"{g}: {int((cells[group_column] == g).sum())} cells / "
        f"{int(unit_means[unit_means[group_column] == g].shape[0])} individuals"
        for g in groups)
    return figure, counts


def morphospace(scores: pd.DataFrame, x: str, y: str, cluster: Sequence = None,
                colours: dict = None, hulls: bool = True, centers=None,
                axes_index: tuple = None, exemplars=None, align: bool = False,
                size_preset: str = "single"):
    """A PC-vs-PC scatter with optional robust hulls, k-means boundaries and
    exemplar cell outlines. Returns `(figure, note)`.

    `centers` + `axes_index` draw the boundary; with more than two clusters
    this is each pair's Voronoi edge sliced at the pair's midpoint in the
    omitted PCs, which is an approximation -- said in the note, not written on
    the figure.

    `exemplars` is `(keys, mask_lookup)`; the glyphs are real cells, since the
    geometric descriptors cannot be inverted into an outline."""
    import morphospace as ms

    colours = colours or {}
    figure, axes = plotkit.panel_figure(size_preset, rows=1,
                                        legend=cluster is not None)
    axis = axes[0]
    points = scores[[x, y]].to_numpy(float)
    labels = (np.asarray(cluster) if cluster is not None
              else np.zeros(len(points), dtype=int))
    for value in np.unique(labels):
        chosen = labels == value
        colour = colours.get(value, plotkit.MID_GREY)
        axis.scatter(points[chosen, 0], points[chosen, 1], s=4, lw=0, color=colour,
                     alpha=0.7, label=f"cluster_{int(value) + 1}",
                     rasterized=plotkit.rasterize_threshold(int(chosen.sum())))
        if hulls and chosen.sum() >= 10:
            hull = ms.hull95(points[chosen])
            axis.plot(hull[:, 0], hull[:, 1], color=colour, lw=plotkit.lw(1.0))
    axis.set_xlabel(x)
    axis.set_ylabel(y)

    note = None
    if centers is not None and axes_index is not None:
        i, j = axes_index
        centers = np.asarray(centers, float)
        if len(centers) == 2:
            bx, by = ms.boundary_xy(centers[0], centers[1], i, j,
                                    axis.get_xlim(), axis.get_ylim())
            axis.plot(bx, by, ls="--", color=plotkit.INK, lw=plotkit.lw(0.8))
        else:
            for _, _, bx, by in ms.pairwise_boundaries(centers, i, j,
                                                       axis.get_xlim(), axis.get_ylim()):
                axis.plot(bx, by, ls="--", color=plotkit.INK, lw=plotkit.lw(0.8))
            note = "boundaries approximate (projection)"

    if exemplars is not None:
        keys, mask_lookup = exemplars
        ms.exemplar_grid(axis, points, list(keys), mask_lookup, align=align)
    if cluster is not None:
        plotkit.outside_legend(figure, markerscale=3)
    return figure, note


def streamlines(field, tissue_mask=None, colour_by=None, size_preset: str = "single",
                density: float = 1.2):
    """Streamlines of an interpolated displacement field.

    `field`: `(X, Y, U, V)` from `trajectory_visualization.interpolate_vector_field`
    with `fill_value=np.nan`. NaN is the point: `griddata`'s old `fill_value=0`
    painted every grid node outside the convex hull of the cells as *stationary*,
    and `streamplot` drew lines slowing to a halt there -- a result that looks
    like biology and is an artefact of the interpolation. Outside the tissue the
    answer is "not measured", which is what NaN means to `streamplot`.

    Density and arrow size are fixed: they trade legibility against coverage in
    a way that has one good answer per figure size."""
    X, Y, U, V = field
    U, V = np.asarray(U, float).copy(), np.asarray(V, float).copy()
    if tissue_mask is not None:
        outside = ~np.asarray(tissue_mask, bool)
        U[outside] = np.nan
        V[outside] = np.nan
    figure, axes = plotkit.panel_figure(size_preset, rows=1)
    axis = axes[0]
    speed = np.hypot(U, V)
    axis.streamplot(X, Y, U, V, density=density, linewidth=plotkit.lw(0.6), arrowsize=0.7,
                    color=(speed if colour_by is None else colour_by),
                    cmap="viridis")
    axis.set_aspect("equal")
    axis.invert_yaxis()  # image rows grow downwards
    axis.set_xlabel("x (px)")
    axis.set_ylabel("y (px)")
    return figure, None


def layout_problems(figure) -> list:
    """What is wrong with a rendered figure's layout. Empty is a pass.

    Three things, each of which Round 34 shipped at least one of:

    1. text drawn outside the figure (a y label taller than its panel, an x
       label pushed off the bottom);
    2. a y axis label overlapping its own tick labels;
    3. a legend sitting on top of the data.

    Checked by measuring the rendered boxes, not by looking -- the failures
    only appear at particular combinations of width and type size, and there
    are 63 of those."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bounds = figure.bbox
    problems = []

    def box(artist):
        try:
            extent = artist.get_window_extent(renderer=renderer)
        except Exception:
            return None
        return extent if extent.width > 0 and extent.height > 0 else None

    def overlaps(a, b, tolerance=1.0):
        return (a.x0 < b.x1 - tolerance and b.x0 < a.x1 - tolerance
                and a.y0 < b.y1 - tolerance and b.y0 < a.y1 - tolerance)

    def visible_ticks(axis, which):
        """Tick labels that are actually drawn.

        `MaxNLocator` places ticks beyond the view limits as well, and
        matplotlib keeps their label artists -- positioned off the figure and
        never rendered. Counting those as "text outside the figure" reported a
        defect in 31 of 63 figures that no reader could ever see."""
        low, high = (axis.get_xlim() if which == "x" else axis.get_ylim())
        low, high = min(low, high), max(low, high)
        locations = (axis.get_xticks() if which == "x" else axis.get_yticks())
        labels = (axis.get_xticklabels() if which == "x" else axis.get_yticklabels())
        return [label for location, label in zip(locations, labels)
                if low - 1e-9 <= location <= high + 1e-9 and label.get_visible()]

    texts = []
    for axis in figure.axes:
        texts += [axis.xaxis.label, axis.yaxis.label, axis.title]
        texts += visible_ticks(axis, "x") + visible_ticks(axis, "y")
    texts += [t for t in figure.texts]
    for text in texts:
        if not text.get_text().strip() or not text.get_visible():
            continue
        extent = box(text)
        if extent is None:
            continue
        if (extent.x0 < bounds.x0 - 1 or extent.x1 > bounds.x1 + 1
                or extent.y0 < bounds.y0 - 1 or extent.y1 > bounds.y1 + 1):
            problems.append(f"text {text.get_text()[:28]!r} falls outside the figure")

    for index, axis in enumerate(figure.axes):
        label = box(axis.yaxis.label)
        if label is None or not axis.yaxis.label.get_text().strip():
            continue
        for tick in visible_ticks(axis, "y"):
            extent = box(tick)
            if extent is not None and overlaps(label, extent):
                problems.append(f"axes {index}: the y label overlaps its tick labels")
                break

    legends = [l for l in ([figure.legends] if isinstance(figure.legends, list)
                           else list(figure.legends))]
    for legend in legends:
        legend_box = box(legend)
        if legend_box is None:
            continue
        for index, axis in enumerate(figure.axes):
            if overlaps(legend_box, axis.get_window_extent(renderer=renderer), tolerance=2.0):
                problems.append(f"the legend overlaps axes {index}")
                break
    return problems


def _self_test() -> None:
    rng = np.random.default_rng(0)

    # 1. Error bands: each says what it is, and they are ordered as expected.
    values = rng.normal(10, 2, (30, 25))
    widths = {}
    for kind in ("SD", "SEM", "95% CI", "min-max"):
        low, high, label = band(values, kind)
        assert label, kind
        widths[kind] = float(np.mean(high - low))
    assert widths["SEM"] < widths["SD"] < widths["min-max"], widths
    assert widths["SEM"] < widths["95% CI"], widths
    assert band(values, "none") == (None, None, None)
    assert "95% CI" in band(values, "95% CI")[2] and "tracks" in band(values, "95% CI")[2], (
        "the band must say what it is over")
    # NaN gaps must not poison a whole frame.
    holed = values.copy()
    holed[5, :10] = np.nan
    assert np.isfinite(band(holed, "SD")[0]).all()

    # 1b. (R35 0-C) The layout engine, on every chart kind at every width and
    #     type size. `layout_problems` is the assertion; `_self_test` runs a
    #     representative slice of the grid that the sweep script covers in
    #     full, so a regression fails here without needing the script.
    for preset in ("single", "double"):
        for size in ("S", "L"):
            with plotkit.style_context(font_size=size):
                probe = pd.DataFrame({
                    "Frame": np.arange(60), "Time": np.arange(60) * 30.0,
                    "Cell Area": rng.normal(120, 8, 60),
                    "Circularity": rng.uniform(0.6, 0.8, 60)})
                figure, _ = single_track(probe, ["Cell Area"],
                                         labels={"Cell Area": "Cell Area (µm²)"},
                                         size_preset=preset)
                found = layout_problems(figure)
                assert not found, f"{preset}/{size} single: {found}"
                # The defect from screenshot 1: a one-value plot used to be a
                # third as tall as it was wide at double-column width.
                width, height = figure.get_size_inches()
                assert height / width > 0.45, (preset, size, height / width)
                plt.close(figure)
                figure, _ = single_track(probe, ["Cell Area", "Circularity"],
                                         layout="dual_axis", size_preset=preset)
                found = layout_problems(figure)
                assert not found, f"{preset}/{size} dual: {found}"
                plt.close(figure)
    # And the check must be able to fail: a label far too long for its panel.
    with plotkit.style_context(font_size="L"):
        figure, axes = plotkit.panel_figure("single", rows=1)
        axes[0].set_ylabel("x" * 400)
        assert layout_problems(figure), "layout_problems cannot detect an overflowing label"
        plt.close(figure)

    # 1c. (R36 3) The Lines control must reach every line. Round 35 set
    #     `lines.linewidth` and then overrode it with an explicit `lw=` on
    #     every single artist, so switching Thin/Normal changed nothing.
    #     Thin is 0.5 pt against Normal's 1 pt, so every width must halve --
    #     checked artist by artist, not by eye.
    def widths_of(figure):
        found = []
        for axis in figure.axes:
            for line in axis.lines:
                found.append(float(line.get_linewidth()))
            for collection in axis.collections:
                for width in np.atleast_1d(collection.get_linewidth()):
                    if float(width) > 0:
                        found.append(float(width))
        return found

    probe_long = pd.DataFrame({"Frame": np.arange(40), "Time": np.arange(40) * 30.0,
                               "Cell Area": rng.normal(120, 8, 40),
                               "Circularity": rng.uniform(0.6, 0.9, 40)})
    probe_wide = pd.DataFrame(rng.normal(120, 8, (40, 25)))
    probe_counts = pd.DataFrame({"N": [100] * 6, "T1": [1, 0, 2, 1, 0, 1]},
                                index=np.arange(6) * 30.0)
    probe_groups = {"a": rng.normal(1, 0.1, 8), "b": rng.normal(1.3, 0.1, 8)}
    probe_scores = pd.DataFrame(rng.normal(size=(120, 3)), columns=["PC1", "PC2", "PC3"])
    grid = np.linspace(0, 20, 16)
    gx, gy = np.meshgrid(grid, grid)
    builders = {
        "single_track": lambda: single_track(probe_long, ["Cell Area", "Circularity"]),
        "all_tracks": lambda: all_tracks(probe_wide, "v", error="SD"),
        "cumulative": lambda: cumulative_events(probe_counts, is_time=False),
        "columns": lambda: columns_plot(probe_groups, "v", kind="scatter", paired=True),
        "morphospace": lambda: morphospace(
            probe_scores, "PC1", "PC2", cluster=(probe_scores["PC1"] > 0).astype(int).to_numpy(),
            centers=probe_scores.groupby((probe_scores["PC1"] > 0).astype(int)).mean().to_numpy(),
            axes_index=(0, 1)),
        "streamlines": lambda: streamlines((gx, gy, np.ones_like(gx), np.zeros_like(gx))),
    }
    for name, build in builders.items():
        measured = {}
        for setting in ("normal", "thin"):
            with plotkit.style_context(line_width=setting):
                figure, _ = build()
                measured[setting] = widths_of(figure)
                plt.close(figure)
        assert measured["normal"], f"{name}: no line widths found to compare"
        assert len(measured["normal"]) == len(measured["thin"]), name
        for thick, thin in zip(measured["normal"], measured["thin"]):
            assert abs(thin / thick - 0.5) < 1e-9, (
                f"{name}: {thick} pt -> {thin} pt, expected half")
    print(f"  line width: {len(builders)} chart kinds, every Line2D and LineCollection "
          f"halves from Normal to Thin")

    # 2. Small multiples vs dual axis.
    long = pd.DataFrame({"Frame": np.arange(20), "Time": np.arange(20) * 30.0,
                         "Cell Area": rng.normal(100, 5, 20),
                         "Circularity": rng.uniform(0.6, 0.9, 20),
                         "Shape Index": rng.uniform(3.7, 4.2, 20)})
    with plotkit.style_context():
        figure, _ = single_track(long, ["Cell Area", "Circularity", "Shape Index"])
        assert len(figure.axes) == 3, len(figure.axes)
        # A 20-frame movie at 30 s spans 570 s, so the axis stays in seconds.
        assert figure.axes[-1].get_xlabel() == "Time (s)", figure.axes[-1].get_xlabel()
        assert figure.axes[0].get_title(loc="left") == "Cell Area", figure.axes[0].get_title(loc="left")
        plt.close(figure)
        figure, _ = single_track(long, ["Cell Area", "Circularity"], layout="dual_axis")
        assert len(figure.axes) == 2
        plt.close(figure)
        # Past 10 minutes the axis switches unit; the data does not.
        hours = long.assign(Time=long["Frame"] * 600.0)
        figure, _ = single_track(hours, ["Cell Area"])
        assert figure.axes[-1].get_xlabel() == "Time (h)", figure.axes[-1].get_xlabel()
        plt.close(figure)
        try:
            single_track(long, ["Cell Area", "Circularity", "Shape Index"], layout="dual_axis")
            raise AssertionError("dual axis must refuse three values")
        except ValueError as error:
            assert "exactly two" in str(error)
        # No frame interval -> the x axis is the frame number, not a fake time.
        figure, _ = single_track(long, ["Cell Area"], is_time=False)
        assert figure.axes[-1].get_xlabel() == "Frame"
        plt.close(figure)

        # 3. all_tracks rasterizes a big bundle and keeps a small one vector.
        wide = pd.DataFrame(rng.normal(100, 8, (40, 300)), index=np.arange(40))
        figure, note = all_tracks(wide, "Cell Area (px²)", error="SD", kept=(300, 420))
        collection = [c for c in figure.axes[0].collections if isinstance(c, LineCollection)][0]
        assert collection.get_rasterized(), "300 lines should be rasterized"
        # R35 0-C: the counts are a NOTE for the status line, never drawn.
        assert "300 of 420 tracks" in note and "SD" in note, note
        assert not figure.axes[0].get_title(loc="left"), "nothing explanatory may be drawn"
        plt.close(figure)
        small = pd.DataFrame(rng.normal(100, 8, (40, 12)))
        figure, _ = all_tracks(small, "v")
        assert not [c for c in figure.axes[0].collections
                    if isinstance(c, LineCollection)][0].get_rasterized()
        plt.close(figure)

        # 4. Cumulative events: normalised by N, and the censored tail shaded.
        counts = pd.DataFrame({
            "N": [100, 100, 50, 50],
            "T1": [2, 0, 2, 1],
            "division": [0, 1, 0, 0],
        }, index=[1, 2, 3, 4])
        figure, censor_note = cumulative_events(counts, censored_from=3, is_time=False)
        line = figure.axes[0].lines[0]
        expected = np.cumsum([2 / 100, 0, 2 / 50, 1 / 50])
        assert np.allclose(line.get_ydata(), expected), (line.get_ydata(), expected)
        # The shading is data and stays; the sentence explaining it is a note.
        assert figure.axes[0].patches, "the censored span should be shaded"
        assert "not yet judgeable" in censor_note, censor_note
        assert not [t.get_text() for t in figure.axes[0].texts if t.get_text().strip()], \
            "nothing explanatory may be drawn inside the axes"
        plt.close(figure)
        # Normalising matters: halving N doubles the step for the same count.
        flat, _ = cumulative_events(pd.DataFrame({"N": [100] * 4, "T1": [2, 0, 2, 1]},
                                                 index=[1, 2, 3, 4]), is_time=False)
        assert flat.axes[0].lines[0].get_ydata()[-1] < expected[-1], (
            "a constant N must give a lower curve than a shrinking one")
        plt.close(flat)

        # 5. Columns: paired data has no jitter.
        groups = {"WT": rng.normal(1, 0.1, 8), "pyd": rng.normal(1.3, 0.1, 8)}
        figure, _ = columns_plot(groups, "Shape index", kind="scatter", paired=True)
        offsets = np.vstack([c.get_offsets() for c in figure.axes[0].collections])
        assert set(np.round(offsets[:, 0], 9)) == {0.0, 1.0}, "paired points must not be jittered"
        plt.close(figure)
        figure, _ = columns_plot(groups, "Shape index", kind="scatter", paired=False)
        offsets = np.vstack([c.get_offsets() for c in figure.axes[0].collections])
        assert len(set(np.round(offsets[:, 0], 9))) > 2, "unpaired points should be jittered"
        plt.close(figure)
        figure, _ = columns_plot(groups, "v", kind="estimation",
                                 estimation={"WT": (1.0, 0.9, 1.1), "pyd": (1.3, 1.2, 1.4)})
        plt.close(figure)

        # 6. Morphospace with k=3 must say its boundaries are projections.
        scores = pd.DataFrame(rng.normal(size=(300, 3)), columns=["PC1", "PC2", "PC3"])
        centers = np.array([[-2.0, -1.0, 0.0], [2.0, -1.0, 0.0], [0.0, 2.0, 0.0]])
        figure, note = morphospace(scores, "PC1", "PC2", cluster=rng.integers(0, 3, 300),
                                   centers=centers, axes_index=(0, 1))
        assert "approximate (projection)" in note, note
        assert not figure.axes[0].get_title(loc="left"), "nothing explanatory may be drawn"
        plt.close(figure)

        # 7. Streamlines: outside the tissue is NaN, not zero.
        grid = np.linspace(0, 10, 20)
        X, Y = np.meshgrid(grid, grid)
        U, V = np.ones_like(X), np.zeros_like(X)
        inside = X < 5
        figure, _ = streamlines((X, Y, U, V), tissue_mask=inside)
        drawn = figure.axes[0].collections[0].get_segments()
        assert drawn, "no streamlines drawn"
        assert max(s[:, 0].max() for s in drawn) < 6.5, (
            "streamlines ran into the masked-out region")
        plt.close(figure)

    print(f"  bands: SEM {widths['SEM']:.2f} < SD {widths['SD']:.2f} < "
          f"min-max {widths['min-max']:.2f}, each labelled")
    print("  single track: 3 small multiples / 2 dual-axis, 3 on two axes refused;")
    print("    the time axis picks s / min / h from the span")
    print("  all_tracks: 300 lines rasterized; kept-count is a NOTE, not a title")
    print("  cumulative: normalised by N per frame pair, censored tail shaded")
    print("  columns: paired = no jitter; morphospace k=3 reports the projection in its note")
    print("  streamlines: masked region left NaN, no lines inside it")
    print("  layout: every kind clean at 2 widths x 2 type sizes; the check can fail")
    print("\nplots.py self-test: OK")


if __name__ == "__main__":
    _self_test()
