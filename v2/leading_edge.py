"""Module 4c: the epidermal leading edge, and cell rows measured from it.

Why this exists, and why it is not `manual_correction_classification.
find_boundary_cells`
-----------------------------------------------------------------------
`find_boundary_cells` answers "which cells physically touch a cell of the
other cluster", by contour proximity within `threshold` px. That is the right
answer to that question, but it is NOT the same question as "which cells are
the front row", and on real data the two diverge badly. Measured on
`test_data/AS_LE_lateral view_E-cad` (frames 30 and 60):

* 26-28% of image columns carry a background gap wider than 2 px between the
  amnioserosa and the epidermis, and the gap reaches 56-97 px. A dilation- or
  proximity-based test cannot bridge that, so a front-row cell sitting under
  such a gap reports zero neighbours in the other cluster.
* 15-23% of columns hold no amnioserosa label at all -- beyond the lateral
  ends of the imaged amnioserosa patch, the front row has nothing above it to
  touch.

Seeding neighbour rings from the amnioserosa therefore mislabels the front row
wherever either holds, which on that movie was ~30% of the cells within three
rows of the edge at frame 30.

The construction here does not use the other population at all. The leading
edge is the dorsal-most contour of the front population itself; the front row
is then the set of cells that reach it, and the rows behind it are neighbour
rings. Two failure modes of the naive "dorsal-most pixel per column" envelope
are handled explicitly:

1. a gap in the front row lets the envelope fall onto a second-row cell, and
2. at the lateral ends of the imaged tissue the envelope falls onto whatever
   deep cell happens to be the top-most one there,

both of which drag the edge ventrally and pull the wrong cells into row 1.
Columns whose top lies outside the modal band are rejected, the largest
contiguous run of surviving columns defines the edge's domain, and the edge is
held flat beyond it. Polynomial extrapolation was tried and rejected: the
amnioserosa's ventral contour is unreliable at the tips of its patch, and
degree 1-3 fits extrapolate to 89-250 px against a true edge at ~166 px.

Validated against 11 hand-checked cells on frame 30; the row assignment is
unchanged for `tol` anywhere in 8-20 px.
"""

from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.ndimage import median_filter

DEFAULT_SMOOTH_PX = 41
DEFAULT_BAND_PX = 30
DEFAULT_TOL_PX = 12


def front_profile(fill_labels: np.ndarray, front_labels: Iterable[int]) -> np.ndarray:
    """Dorsal-most `front_labels` pixel per column; NaN where the population is
    absent. "Dorsal" is row 0, matching image convention -- pass a flipped
    array if the tissue of interest faces the other way.
    """
    member = np.isin(fill_labels, list(front_labels))
    width = fill_labels.shape[1]
    profile = np.full(width, np.nan)
    for x in range(width):
        rows = np.where(member[:, x])[0]
        if rows.size:
            profile[x] = rows.min()
    return profile


def leading_edge_curve(
    fill_labels: np.ndarray,
    front_labels: Iterable[int],
    smooth_px: int = DEFAULT_SMOOTH_PX,
    band_px: int = DEFAULT_BAND_PX,
    max_hole_px: int = 5,
) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray]:
    """Return `(curve, (x0, x1), valid)`.

    `curve` is defined for every column: a running median of the front profile
    over the columns that survive the band test, held flat outside `[x0, x1]`.
    `(x0, x1)` is the edge's domain -- the largest contiguous run of surviving
    columns, i.e. where the edge was actually measured rather than extended.
    Plot only that span; use the whole curve for distance tests.
    """
    profile = front_profile(fill_labels, front_labels)
    present = ~np.isnan(profile)
    if not present.any():
        raise ValueError("front_labels match no pixel in fill_labels")
    # Bridge short holes first: a segmentation skeleton leaves 1-2 px columns
    # with no cell interior, and letting those split the domain would shrink it
    # to the width of a single cell.
    profile = profile.copy()
    idx = np.where(present)[0]
    holes = np.where(~present)[0]
    short = holes[(holes > idx[0]) & (holes < idx[-1])]
    if short.size:
        runs, start = [], short[0]
        for a, b in zip(short, np.append(short[1:], -1)):
            if b != a + 1:
                runs.append((start, a)); start = b
        for a, b in runs:
            if b - a + 1 <= max_hole_px:
                profile[a:b + 1] = np.interp(np.arange(a, b + 1), idx, profile[idx])
        present = ~np.isnan(profile)
    valid = present & (np.abs(profile - np.median(profile[present])) <= band_px)
    if not valid.any():
        raise ValueError(f"no column survives band_px={band_px}; the front is not a band")

    runs, start = [], None
    for x, ok in enumerate(valid):
        if ok and start is None:
            start = x
        elif not ok and start is not None:
            runs.append((start, x - 1)); start = None
    if start is not None:
        runs.append((start, len(valid) - 1))
    x0, x1 = max(runs, key=lambda r: r[1] - r[0])

    index = np.where(valid)[0]
    index = index[(index >= x0) & (index <= x1)]
    size = max(1, min(smooth_px, index.size))
    size -= 1 - size % 2                      # median_filter wants a size it can centre
    smoothed = median_filter(profile[index], size=size, mode="nearest")
    curve = np.interp(np.arange(fill_labels.shape[1]), index, smoothed)
    return curve, (int(x0), int(x1)), valid


def front_row_cells(
    fill_labels: np.ndarray,
    front_labels: Iterable[int],
    curve: Optional[np.ndarray] = None,
    tol_px: float = DEFAULT_TOL_PX,
) -> Set[int]:
    """Labels whose dorsal contour comes within `tol_px` of the leading edge.

    The test is on the cell's own top contour against an externally-defined
    edge, so it is absolute: a cell that happens to be the top-most one in some
    column does not qualify on that account (that is how the lateral ends of
    the tissue leak into row 1), and a cell partly overhung by its neighbour
    still qualifies (that is how a genuine front-row cell drops out).
    """
    front = set(int(l) for l in front_labels)
    if curve is None:
        curve, _, _ = leading_edge_curve(fill_labels, front)
    out = set()
    for label in front:
        ys, xs = np.where(fill_labels == label)
        if not ys.size:
            continue
        gap = min(ys[xs == x].min() - curve[x] for x in np.unique(xs))
        if gap <= tol_px:
            out.add(label)
    return out


def assign_rows(
    fill_labels: np.ndarray,
    front_labels: Iterable[int],
    graph: Mapping[int, Set[int]],
    curve: Optional[np.ndarray] = None,
    tol_px: float = DEFAULT_TOL_PX,
    n_rows: int = 3,
) -> Dict[int, int]:
    """`{label: row}` for rows 1..n_rows, counting from the leading edge.

    Row 1 is `front_row_cells`; each further row is the next neighbour ring
    inside `front_labels` (pass `adjacency.build_neighbor_graph(...)`). Cells
    further back than `n_rows` are absent from the result rather than pooled,
    so the caller decides what "beyond" means.
    """
    front = set(int(l) for l in front_labels)
    rows = {l: 1 for l in front_row_cells(fill_labels, front, curve, tol_px)}
    current = set(rows)
    for row in range(2, n_rows + 1):
        nxt = {n for c in current for n in graph.get(c, set()) if n in front} - set(rows)
        if not nxt:
            break
        for n in nxt:
            rows[n] = row
        current = nxt
    return rows


def front_roughness(
    curves: Sequence[np.ndarray], smooth_px: int = DEFAULT_SMOOTH_PX
) -> Tuple[np.ndarray, np.ndarray]:
    """`(residual, width)` for a stack of RAW front profiles, one per frame.

    `residual[t, x]` is the profile minus its own running median -- the local
    undulation of the edge, with the mean front position and its large-scale
    shape removed. `width[t]` is that residual's standard deviation, the usual
    front-roughness W. Feed raw `front_profile` output, not `leading_edge_curve`
    output: the latter is already smoothed, and subtracting a running median
    from it leaves nothing.
    """
    stack = np.asarray(curves, dtype=float)
    residual = stack - median_filter(stack, size=(1, smooth_px), mode="nearest")
    return residual, residual.std(axis=1)


def _self_test() -> None:
    rng = np.random.default_rng(0)

    # A synthetic epithelium: four rows of 20x40 px cells under a flat front at
    # y = 60, with three deliberate traps.
    H, W, CW, CH, TOP = 220, 400, 20, 40, 60
    labels = np.zeros((H, W), dtype=np.int32)
    ids = {}
    n = 0
    for row in range(4):
        for col in range(W // CW):
            n += 1
            y0 = TOP + row * CH
            labels[y0:y0 + CH - 1, col * CW:(col + 1) * CW - 1] = n
            ids[(row, col)] = n
    front = set(range(1, n + 1))

    curve, (x0, x1), _ = leading_edge_curve(labels, front)
    assert abs(np.median(curve[x0:x1]) - TOP) <= 1, np.median(curve[x0:x1])
    r1 = front_row_cells(labels, front, curve)
    assert r1 == {ids[(0, c)] for c in range(W // CW)}, sorted(r1)
    print("A: flat front recovered; row 1 is exactly the first rank")

    # Trap 1: the OTHER population sits above the front, and 80 px of it is
    # missing from the segmentation. Neither the edge nor row 1 may move --
    # this is exactly where a proximity test to the other cluster fails.
    with_other = labels.copy()
    with_other[TOP - 45:TOP - 1, :] = n + 1        # one big "amnioserosa" cell
    curve_c, _, _ = leading_edge_curve(with_other, front)
    assert front_row_cells(with_other, front, curve_c) == r1
    gapped = with_other.copy()
    gapped[TOP - 45:TOP, 100:180] = 0              # 80 px of it unsegmented
    curve_g, _, _ = leading_edge_curve(gapped, front)
    r1g = front_row_cells(gapped, front, curve_g)
    assert r1g == r1, sorted(set(r1) ^ set(r1g))
    print("B: an 80 px hole in the other population changes neither edge nor row 1")

    # Trap 2: the tissue ends laterally, so the top-most cell in those columns
    # is a deep one. It must NOT be promoted to row 1.
    ended = labels.copy()
    for row in range(2):
        for col in range(W // CW - 3, W // CW):
            ended[ended == ids[(row, col)]] = 0
    curve_e, _, _ = leading_edge_curve(ended, front)
    r1e = front_row_cells(ended, front, curve_e)
    deep = {ids[(2, c)] for c in range(W // CW - 3, W // CW)}
    assert not (r1e & deep), sorted(r1e & deep)
    print("C: cells exposed only by the lateral end of the tissue stay out of row 1")

    # Trap 3: a front-row cell overhung by its neighbour still counts.
    over = labels.copy()
    over[TOP - 8:TOP, 40:59] = ids[(0, 2)]      # cell 2 pokes 8 px dorsally
    curve_o, _, _ = leading_edge_curve(over, front)
    assert ids[(0, 3)] in front_row_cells(over, front, curve_o)
    print("D: a cell overhung by its neighbour is still row 1")

    # Rows via the neighbour graph.
    graph = {}
    for (row, col), lab in ids.items():
        graph[lab] = {ids[k] for k in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
                      if k in ids}
    rows = assign_rows(labels, front, graph, curve, n_rows=3)
    for r in (1, 2, 3):
        got = {l for l, v in rows.items() if v == r}
        assert got == {ids[(r - 1, c)] for c in range(W // CW)}, (r, sorted(got))
    assert not any(l in rows for l in (ids[(3, c)] for c in range(W // CW)))
    print("E: rows 1-3 are the first three ranks; rank 4 is absent, not pooled")

    # Roughness: a flat front has none, a jittered one has the jitter's sd.
    flat = [np.full(W, 60.0) for _ in range(5)]
    _, w_flat = front_roughness(flat)
    assert np.allclose(w_flat, 0), w_flat
    jit = [60.0 + rng.normal(0, 2.0, W) for _ in range(40)]
    _, w_jit = front_roughness(jit)
    assert 1.5 < w_jit.mean() < 2.5, w_jit.mean()
    print("F: roughness is 0 on a flat front and recovers a known jitter")

    print("leading_edge self-test OK")


if __name__ == "__main__":
    _self_test()
