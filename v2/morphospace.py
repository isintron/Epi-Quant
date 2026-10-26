"""Morphospace overlays: robust 95% hulls, the k-means boundary in a PC plane,
and size-normalised exemplar cells.

Ported from `figures/revision_2026-10/fig2/scripts/morph.py`, which the GUI
cannot import (it pulls in `figkit`, which needs `cmocean` and `lxml`). Two
changes, both forced by the GUI being general where the figure script was not:

- `boundary_xy` was written for exactly three PCs (`k = 3 - i - j` names the one
  omitted axis). Here any number of PCs is allowed and every omitted axis is
  held at the midpoint of the two centres -- which reduces to the original
  expression when there are three.
- k-means with more than two clusters has no single boundary.
  `pairwise_boundaries` draws the line between each pair of centres and keeps
  only the stretch where those two centres are the nearest two, which is the
  true Voronoi edge in the slice. It is still a slice, so a figure drawn with
  it says "approximate (projection)".

The exemplar glyphs are real cells, not reconstructions: the basic/hull/lobe
descriptors cannot be inverted back into an outline. Only the EFD family can,
via `efd.reconstruct_contour` on `pca.mean_ + a * c_x + b * c_y`.
"""

import numpy as np
from scipy.spatial import ConvexHull
from sklearn.covariance import MinCovDet

# Neutral, so an exemplar glyph never reads as one of the classes.
# `GLYPH_LW` is a SCALE on the user's Lines setting, not an absolute width
# (Round 36 3) -- `plotkit.lw` turns it into points at draw time.
GLYPH_FILL, GLYPH_LINE, GLYPH_LW = "#EDEDED", "#555555", 0.5


def hull95(points: np.ndarray, random_state: int = 0) -> np.ndarray:
    """The convex hull of the 95% most central points, by robust Mahalanobis
    distance (`MinCovDet`).

    The plain convex hull of a cloud is a hull of its outliers: one mis-segmented
    cell moves it across the plot. `MinCovDet` estimates the covariance from the
    densest half of the data, so the ranking it produces is not itself set by the
    points being excluded.

    Returns the hull vertices, closed (first point repeated at the end)."""
    points = np.asarray(points, float)
    distance = MinCovDet(random_state=random_state).fit(points).mahalanobis(points)
    kept = points[distance <= np.quantile(distance, 0.95)]
    hull = ConvexHull(kept)
    order = np.r_[hull.vertices, hull.vertices[:1]]
    return kept[order]


def boundary_xy(center_lo, center_hi, i: int, j: int, xlim, ylim, n: int = 400):
    """The k-means decision plane between two centres, sliced in the (i, j) PC
    plane with every other PC held at the midpoint of the two centres.

    Returns `(xs, ys)` clipped to the visible box; empty arrays when the line
    misses it or the two centres coincide in this plane.

    `morph.py` always solves `y = f(x)`, which divides by `w[j]` and so returns
    nothing for a boundary that is vertical in the slice. That is not a rare
    case here: on `AS_LE_lateral` the two clusters are separated almost
    entirely along PC1, so the PC1-PC2 boundary is close to vertical and the
    original would have drawn no line at all. The axis with the larger weight
    is solved for instead, which is the same line, sampled along whichever
    direction is better conditioned."""
    low, high = np.asarray(center_lo, float), np.asarray(center_hi, float)
    weight = high - low
    offset = (high @ high - low @ low) / 2
    midpoint = (high + low) / 2
    other = [k for k in range(len(weight)) if k not in (i, j)]
    offset = offset - sum(weight[k] * midpoint[k] for k in other)
    if abs(weight[j]) >= abs(weight[i]):
        if abs(weight[j]) < 1e-12:
            return np.array([]), np.array([])
        xs = np.linspace(*xlim, n)
        ys = (offset - weight[i] * xs) / weight[j]
        inside = (ys >= ylim[0]) & (ys <= ylim[1])
    else:
        ys = np.linspace(*ylim, n)
        xs = (offset - weight[j] * ys) / weight[i]
        inside = (xs >= xlim[0]) & (xs <= xlim[1])
    return xs[inside], ys[inside]


def pairwise_boundaries(centers: np.ndarray, i: int, j: int, xlim, ylim, n: int = 400) -> list:
    """`[(a, b, xs, ys), ...]` -- the visible part of each pair's boundary for
    k > 2 clusters.

    A point on the a-b boundary only separates a from b if no OTHER centre is
    closer than those two, so each line is masked to where a and b are the
    nearest pair. Distances are measured in the full PC space with the omitted
    axes at the pair's midpoint -- the same slice the line was drawn in, so the
    mask and the line agree."""
    centers = np.asarray(centers, float)
    out = []
    for a in range(len(centers)):
        for b in range(a + 1, len(centers)):
            xs, ys = boundary_xy(centers[a], centers[b], i, j, xlim, ylim, n)
            if len(xs) == 0:
                continue
            midpoint = (centers[a] + centers[b]) / 2
            probe = np.tile(midpoint, (len(xs), 1))
            probe[:, i], probe[:, j] = xs, ys
            distance = np.linalg.norm(probe[:, None, :] - centers[None, :, :], axis=2)
            nearest = np.argsort(distance, axis=1)[:, :2]
            keep = np.array([set(pair) == {a, b} for pair in nearest])
            if keep.any():
                out.append((a, b, xs[keep], ys[keep]))
    return out


def contour(mask: np.ndarray, n: int = 120, align: bool = False) -> np.ndarray:
    """A boolean cell mask as a closed outline of `n` points, centred and
    scaled so its farthest point is at radius 1 -- every glyph then fits the
    same circle, so a grid of them has one size and no overlaps.

    y is negated because image rows grow downwards while axes grow upwards.
    `align` rotates the long axis horizontal, for a specimen that was not
    mounted axis-aligned."""
    import cv2
    found, _ = cv2.findContours(np.asarray(mask).astype(np.uint8),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not found:
        raise ValueError("mask holds no contour")
    points = max(found, key=len)[:, 0, :].astype(float)
    points[:, 1] *= -1
    closed = np.r_[points, points[:1]]
    arc = np.r_[0, np.cumsum(np.hypot(*np.diff(closed, axis=0).T))]
    even = np.linspace(0, arc[-1], n, endpoint=False)
    points = np.c_[np.interp(even, arc, np.r_[points[:, 0], points[0, 0]]),
                   np.interp(even, arc, np.r_[points[:, 1], points[0, 1]])]
    points -= points.mean(axis=0)
    if align:
        _, vectors = np.linalg.eigh(np.cov(points.T))
        angle = np.arctan2(vectors[1, 1], vectors[0, 1])
        rotation = np.array([[np.cos(-angle), -np.sin(-angle)],
                             [np.sin(-angle), np.cos(-angle)]])
        points = points @ rotation.T
    return points / np.hypot(*points.T).max()


def exemplar_grid(axis, points: np.ndarray, keys, mask_lookup, n: int = 5,
                  align: bool = False, size: float = 0.8) -> list:
    """Draw the real cell nearest each node of an n x n grid over the visible
    plane, skipping nodes with no cell within half a grid step.

    Glyphs are drawn in POINTS around their node (a `dpi_scale_trans` affine),
    not in data units, so a cell keeps its true outline whatever the axis
    aspect ratio is. Call this after the axis limits and the axes position are
    final -- it reads both.

    Returns `[(x, y, key, distance_in_grid_steps), ...]`."""
    import matplotlib.transforms as mt

    (x0, x1), (y0, y1) = axis.get_xlim(), axis.get_ylim()
    step_x, step_y = (x1 - x0) / n, (y1 - y0) / n
    grid_x = x0 + step_x * (np.arange(n) + 0.5)
    grid_y = y0 + step_y * (np.arange(n) + 0.5)
    step = np.array([step_x, step_y])
    figure = axis.figure
    displayed = axis.transData.transform([[grid_x[0], grid_y[0]], [grid_x[1], grid_y[1]]])
    radius = size * np.abs(np.diff(displayed, axis=0)).min() / 2 / figure.dpi
    scaled = np.asarray(points, float) / step

    used = []
    for a in grid_x:
        for b in grid_y:
            distance = np.hypot(*(scaled - np.array([a, b]) / step).T)
            nearest = int(np.argmin(distance))
            if distance[nearest] > 0.5:
                continue
            outline = contour(mask_lookup(keys[nearest]), align=align)
            outline = np.vstack([outline, outline[:1]])
            transform = (mt.Affine2D().scale(radius) + figure.dpi_scale_trans
                         + mt.ScaledTranslation(a, b, axis.transData))
            axis.fill(outline[:, 0], outline[:, 1], color=GLYPH_FILL, lw=0,
                      zorder=3, transform=transform)
            import plotkit
            axis.plot(outline[:, 0], outline[:, 1], c=GLYPH_LINE, lw=plotkit.lw(GLYPH_LW),
                      zorder=3.1, transform=transform)
            used.append((a, b, keys[nearest], float(distance[nearest])))
    return used


def _self_test() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)

    # 1. hull95 vs the original, copied here as the reference -- `morph.py`
    #    cannot be imported (it pulls in figkit -> cmocean, lxml).
    def reference_hull95(P):
        md = MinCovDet(random_state=0).fit(P).mahalanobis(P)
        k = P[md <= np.quantile(md, .95)]
        h = ConvexHull(k)
        v = np.r_[h.vertices, h.vertices[:1]]
        return k[v]

    cloud = rng.multivariate_normal([0, 0], [[4, 1.5], [1.5, 1]], 500)
    cloud[:5] += 40  # outliers the plain hull would chase
    assert np.array_equal(hull95(cloud), reference_hull95(cloud)), "hull95 diverged"
    plain = cloud[ConvexHull(cloud).vertices]
    assert np.abs(hull95(cloud)).max() < np.abs(plain).max() / 2, "robustness did nothing"

    # 2. boundary_xy must reduce to the original expression for 3 PCs.
    def reference_boundary(C_lo, C_hi, i, j, xl, yl):
        w = C_hi - C_lo
        b = (C_hi @ C_hi - C_lo @ C_lo) / 2
        mid = (C_hi + C_lo) / 2
        k = 3 - i - j
        xs = np.linspace(*xl, 400)
        ys = (b - w[k] * mid[k] - w[i] * xs) / w[j]
        ok = (ys >= yl[0]) & (ys <= yl[1])
        return xs[ok], ys[ok]

    low, high = rng.normal(size=3), rng.normal(size=3) + 2
    for i, j in ((0, 1), (0, 2), (1, 2)):
        ours = boundary_xy(low, high, i, j, (-5, 5), (-5, 5))
        theirs = reference_boundary(low, high, i, j, (-5, 5), (-5, 5))
        weight = high - low
        if abs(weight[j]) >= abs(weight[i]):
            # Same orientation the original solves for: byte-for-byte equal.
            assert np.array_equal(ours[0], theirs[0]) and np.array_equal(ours[1], theirs[1]), \
                f"boundary_xy diverged on PC{i + 1}-PC{j + 1}"
        else:
            # We sample the other way round, so compare the LINE: every point
            # the original returns must satisfy our equation and vice versa.
            k = 3 - i - j
            mid = (high + low) / 2
            level = (high @ high - low @ low) / 2 - weight[k] * mid[k]
            residual = np.abs(weight[i] * ours[0] + weight[j] * ours[1] - level)
            assert residual.max() < 1e-9, f"not the same line: {residual.max():.2e}"
            assert abs(min(ours[0]) - min(theirs[0])) < 0.05, "the clipped span differs"
    # Equidistance is the property, in any dimension.
    low5, high5 = rng.normal(size=5), rng.normal(size=5) + 1.5
    xs, ys = boundary_xy(low5, high5, 0, 1, (-6, 6), (-20, 20))
    mid5 = (low5 + high5) / 2
    probe = np.tile(mid5, (len(xs), 1))
    probe[:, 0], probe[:, 1] = xs, ys
    gap = np.abs(np.linalg.norm(probe - low5, axis=1) - np.linalg.norm(probe - high5, axis=1))
    assert gap.max() < 1e-9, f"5-D boundary is not equidistant, worst {gap.max():.2e}"
    assert len(boundary_xy(low5, high5, 0, 1, (-6, 6), (-20, 20))[0]) > 0

    # 3. k > 2: each kept point's two nearest centres are the pair that drew it.
    centers = np.array([[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.5, 0.0]])
    segments = pairwise_boundaries(centers, 0, 1, (-6, 6), (-6, 6))
    assert len(segments) == 3, f"expected 3 pairs, got {len(segments)}"
    for a, b, xs, ys in segments:
        probe = np.tile((centers[a] + centers[b]) / 2, (len(xs), 1))
        probe[:, 0], probe[:, 1] = xs, ys
        order = np.argsort(np.linalg.norm(probe[:, None] - centers[None], axis=2), axis=1)
        assert all(set(row[:2]) == {a, b} for row in order), f"pair {a}-{b} kept a wrong stretch"
    full = boundary_xy(centers[0], centers[1], 0, 1, (-6, 6), (-6, 6))[0]
    kept = next(len(xs) for a, b, xs, _ in segments if (a, b) == (0, 1))
    assert kept < len(full), "masking removed nothing, so it is not a Voronoi edge"

    # 4. contour: a disc comes back round, a bar comes back long, both unit radius.
    yy, xx = np.mgrid[0:61, 0:61]
    disc = ((xx - 30) ** 2 + (yy - 30) ** 2) < 400
    outline = contour(disc)
    radius = np.hypot(*outline.T)
    assert abs(radius.max() - 1) < 1e-9 and radius.min() > 0.9, (radius.min(), radius.max())
    bar = (np.abs(xx - 30) < 25) & (np.abs(yy - 30) < 6)
    rotated = (np.abs(yy - 30) < 25) & (np.abs(xx - 30) < 6)  # the same bar, upright
    span = lambda c: (np.ptp(c[:, 0]), np.ptp(c[:, 1]))
    assert span(contour(bar))[0] > span(contour(bar))[1], "a wide bar should be wide"
    aligned_x, aligned_y = span(contour(rotated, align=True))
    assert aligned_x > aligned_y, "align=True should lay the long axis horizontal"

    # 5. exemplar_grid places glyphs at nodes with a cell nearby and skips the rest.
    figure, axis = plt.subplots()
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)
    scores = np.array([[-0.6, -0.6], [0.6, 0.6]])
    used = exemplar_grid(axis, scores, ["a", "b"], lambda key: disc, n=3)
    assert len(used) == 2, f"expected the two occupied nodes, got {len(used)}"
    assert {key for _, _, key, _ in used} == {"a", "b"}
    plt.close(figure)

    print(f"  hull95 identical to morph.py; boundary_xy identical for 3 PCs and "
          f"equidistant in 5; {len(segments)} masked Voronoi edges for k=3")
    print("\nmorphospace.py self-test: OK")


if __name__ == "__main__":
    _self_test()
