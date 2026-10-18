"""Elliptic Fourier descriptors (EFD) of closed 2D contours, plus the
normalizations that make a coefficient vector comparable between cells.

This is the 2D counterpart of the spherical-harmonic decomposition a peer
reviewer asked for: spherical harmonics decompose closed *3D surfaces*, while
Epi-Quant analyses 2D apical outlines, for which the elliptic Fourier series
(Kuhl & Giardina 1982, *Computer Graphics and Image Processing* 18:236-258) is
the exact analogue. Both give a complete, truncatable orthogonal basis for the
boundary, so a cell shape is described by a coefficient vector rather than by a
handful of hand-picked scalars.

**The transform itself comes from `pyefd`** (MIT, numpy-only), the same package
`figures/Epiquant_ShapeAnalysis_minimal.ipynb` and
`figures/Epiquant_ShapeAnalysis_allMutants.ipynb` use, so the collaborator's
notebooks and this module stay directly comparable. This module re-exports
`elliptic_fourier_descriptors`, `reconstruct_contour` and `contour_centroid`
(= `pyefd.calculate_dc_coefficients`) so callers have one import.

**What is deliberately NOT taken from pyefd is its `normalize=True` path.**
That flag is not start-point invariant: sweeping the contour start over the
notebooks' own test contour, `normalize=True` leaves a worst-case residual of
**2.0** on coefficients whose leading term is 1 by construction. The chain
here is `orient_traversal` -> `phase_normalize` -> `align_rotation` ->
`canonicalize` (which end of the major axis is +x, decided by the shape's
skewness) -> `drop_size`, and is asserted invariant to start point, traversal
direction, scale and rotation at every angle (~1e-15). Direction, when wanted,
is carried separately by `orientation_tensor`. See `phase_normalize` for why
the notebooks' normalization, and this module's earlier fix of it, were not.
`irregularity`, `ellipse_from_harmonic1` and `nematic_S` are ports of
`figures/Epiquant_ShapeAnalysis_allMutants.ipynb` (cells 24, 26); the corrected
notebook is `figures/EFD_Shape_Analysis_pyd_tmc.ipynb`.

An earlier revision of this module reimplemented the Kuhl & Giardina transform
from the paper; it agreed with `pyefd.elliptic_fourier_descriptors(...,
normalize=False)` to 2.2e-16, which is the evidence that swapping in pyefd here
changed no numbers.

Convention throughout: a single cell's coefficients are `(order, 4)` laid out
as `[a_n, b_n, c_n, d_n]` with `n` 1-based ascending, i.e. row 0 is harmonic 1.
A population is `(n_cells, order, 4)`. Harmonic 1 alone is the best-fit
ellipse: its 2x2 matrix `[[a1, b1], [c1, d1]]` maps the unit circle onto that
ellipse.
"""

from typing import Tuple

import numpy as np
import pyefd

EFD_ORDER_DEFAULT = 10


def _closed(contour: np.ndarray) -> np.ndarray:
    """`(N, 2)` xy with the first point repeated at the end."""
    contour = np.asarray(contour, dtype=np.float64)
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError(f"contour must be (N, 2) xy, got {contour.shape}")
    if not np.allclose(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])
    return contour


def elliptic_fourier_descriptors(contour: np.ndarray, order: int = EFD_ORDER_DEFAULT) -> np.ndarray:
    """Elliptic Fourier coefficients `(order, 4)` of one closed contour.

    Thin wrapper over `pyefd.elliptic_fourier_descriptors(..., normalize=False)`
    that closes the contour first. Normalization is deliberately left off here
    and done by `efd_features` -- see the module docstring for why pyefd's own
    `normalize=True` is not used.
    """
    return pyefd.elliptic_fourier_descriptors(_closed(contour), order=order, normalize=False)


def reconstruct_contour(
    coeffs: np.ndarray, locus: Tuple[float, float] = (0.0, 0.0), num_points: int = 300
) -> np.ndarray:
    """Inverse transform: `(order, 4)` coefficients back to an `(num_points, 2)`
    contour. Used to draw what a principal component of shape space actually
    looks like, which is the interpretability payoff of the EFD route."""
    return pyefd.reconstruct_contour(
        np.asarray(coeffs, dtype=np.float64).reshape(-1, 4), locus=locus, num_points=num_points
    )


def contour_centroid(contour: np.ndarray) -> np.ndarray:
    """Arclength-weighted centroid -- the correct `locus` for
    `reconstruct_contour`. Wrapper over `pyefd.calculate_dc_coefficients`.

    The series is parametrized by arclength, so its DC term is the *perimeter*-
    weighted centroid, not the mean of the sample points. Those differ whenever
    the points are unevenly spaced along the boundary, which is the normal case
    for a traced contour (denser where curvature is high). Passing the point
    mean instead leaves a constant offset that shows up as an error floor the
    series cannot remove: on the notebooks' test contour, mean reconstruction
    error plateaus at ~1.7% of R_eq from order 6 upward, versus
    2.3% / 1.2% / 0.47% / 0.20% at orders 4 / 6 / 10 / 20 with this locus.
    """
    return np.asarray(pyefd.calculate_dc_coefficients(_closed(contour)), dtype=np.float64)


def _as_population(C: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Accept `(order, 4)` or `(N, order, 4)`; return the 3D form plus whether
    the caller passed a single cell (so the result can be squeezed back)."""
    C = np.asarray(C, dtype=np.float64)
    if C.ndim == 2:
        return C[None], True
    if C.ndim == 3:
        return C, False
    raise ValueError(f"expected (order, 4) or (N, order, 4), got {C.shape}")


def _rotate_start_phase(C: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Advance the contour start point by `theta` radians of phase. Harmonic
    `n` rotates by `n * theta`. `C` is `(N, order, 4)`, `theta` is `(N,)`."""
    order = C.shape[1]
    ang = theta[:, None] * np.arange(1, order + 1)[None, :]  # (N, order)
    ca, sa = np.cos(ang), np.sin(ang)
    a, b, c, d = C[..., 0], C[..., 1], C[..., 2], C[..., 3]
    return np.stack(
        [a * ca + b * sa, -a * sa + b * ca, c * ca + d * sa, -c * sa + d * ca], axis=-1
    )


def orient_traversal(C: np.ndarray) -> np.ndarray:
    """Give every contour the same traversal direction (det of the harmonic-1
    matrix positive).

    Tracing a contour the other way round maps `t -> -t`, which negates every
    `b_n` and `d_n`. Left alone, a cell and the same cell traced clockwise land
    at two different points of coefficient space. The sign of
    `det([[a1, b1], [c1, d1]])` is the sign of the harmonic-1 ellipse's signed
    area, i.e. the traversal direction, so negating `b` and `d` whenever it is
    negative puts every cell on one convention. On the WT/tmc/pyd pickles
    99.99% of cells already have `det > 0` (skimage traces consistently); the
    step exists so that the invariance does not depend on the tracer.
    """
    C, single = _as_population(C)
    det = C[:, 0, 0] * C[:, 0, 3] - C[:, 0, 1] * C[:, 0, 2]
    sign = np.where(det < 0, -1.0, 1.0)[:, None]
    R = C.copy()
    R[..., 1] *= sign
    R[..., 3] *= sign
    return R[0] if single else R


def phase_normalize(C: np.ndarray) -> np.ndarray:
    """Move the contour start point to an end of the harmonic-1 major axis.
    Size and orientation are untouched.

    Uses the standard Kuhl & Giardina expression
    `theta = 0.5 * arctan2(2*(a1*b1 + c1*d1), a1^2 - b1^2 + c1^2 - d1^2)`.

    **This pins the start point only up to half a period.** `arctan2` fixes
    `2*theta` in `(-pi, pi]`, so `theta` is known modulo `pi`: the start lands on
    *one of the two* ends of the major axis, and which one depends on where the
    tracer began. The two outcomes differ by multiplying harmonic `h` by
    `(-1)**h`. Nothing here resolves that -- it cannot be resolved from the
    start point alone, and every attempt to do so with a lab-frame sign rule
    trades it for an orientation seam:

    - `allMutants` (cell 5) does not resolve it. Its self-test rolled the start
      by 45 degrees, which lands in the working half; a roll of 90-270 degrees
      leaves a residual of 1.80 (direction kept) / 0.11 (direction removed) on
      unit-normalized coefficients. On the real pickles the branch is set by
      where `find_contours` starts, so the major-axis *vector* points to +y in
      the image for almost every cell: a director represented as a vector, with
      a seam at horizontal.
    - The minimal notebook's `if R[0, 0] < 0: R = -R` negates every harmonic,
      a 180-degree rotation of the shape, not a half-period shift.
    - This module's previous fix (`R[:, 0, 0] < 0` -> multiply by `(-1)**h`)
      restored start-point invariance but moved the seam to vertical: rotating
      the notebook's test contour by 95-265 degrees left a residual of 0.110 on
      the direction-removed features. Its self-test only rotated by 30.

    The half-period is resolved by `canonicalize`, *after* the rotation, by a
    property of the shape itself.
    """
    C, single = _as_population(C)
    a1, b1, c1, d1 = C[:, 0, 0], C[:, 0, 1], C[:, 0, 2], C[:, 0, 3]
    theta = 0.5 * np.arctan2(2 * (a1 * b1 + c1 * d1), a1 ** 2 - b1 ** 2 + c1 ** 2 - d1 ** 2)
    R = _rotate_start_phase(C, theta)
    return R[0] if single else R


def align_rotation(P: np.ndarray) -> np.ndarray:
    """Rotate each cell so the harmonic-1 major axis lies along +x, removing
    orientation. Expects phase-normalized coefficients. Size is untouched.

    After this step the only freedom left is which end of the major axis ends
    up at +x -- the half-period of `phase_normalize`, now a 180-degree
    rotation of the aligned shape (odd harmonics unchanged, even harmonics
    negated). `canonicalize` fixes it."""
    P, single = _as_population(P)
    a1, c1 = P[:, 0, 0], P[:, 0, 2]
    psi = np.arctan2(c1, a1)
    ca, sa = np.cos(-psi), np.sin(-psi)
    x, y = P[..., 0::2], P[..., 1::2]  # (a, c) and (b, d) column pairs
    a = ca[:, None] * x[..., 0] - sa[:, None] * x[..., 1]
    c = sa[:, None] * x[..., 0] + ca[:, None] * x[..., 1]
    b = ca[:, None] * y[..., 0] - sa[:, None] * y[..., 1]
    d = sa[:, None] * y[..., 0] + ca[:, None] * y[..., 1]
    R = np.stack([a, b, c, d], axis=-1)
    return R[0] if single else R


def central_moments3(P: np.ndarray, n_points: int = 128, chunk: int = 20000) -> np.ndarray:
    """Area-weighted central third moments `(mu3_x, mu3_y)`, each divided by
    `A * sigma^3` along its own axis (i.e. a skewness), of the outline that `P`
    reconstructs. `(N, 2)`.

    Computed exactly on the reconstructed polygon with Green's theorem, so it is
    a property of the shape, independent of how the contour was sampled. Both
    components change sign under a 180-degree rotation, which is what makes
    them usable to pick one of the two ends in `canonicalize`.
    """
    P, _ = _as_population(P)
    order = P.shape[1]
    t = 2 * np.pi * np.arange(n_points) / n_points
    n = np.arange(1, order + 1)[:, None]
    cos, sin = np.cos(n * t), np.sin(n * t)  # (order, n_points)
    out = np.empty((len(P), 2))
    for s in range(0, len(P), chunk):
        Q = P[s:s + chunk]
        x = Q[:, :, 0] @ cos + Q[:, :, 1] @ sin
        y = Q[:, :, 2] @ cos + Q[:, :, 3] @ sin
        x1, y1 = np.roll(x, -1, axis=1), np.roll(y, -1, axis=1)
        cross = x * y1 - x1 * y
        area = cross.sum(axis=1) / 2
        with np.errstate(invalid="ignore", divide="ignore"):
            for k, (u, u1) in enumerate(((x, x1), (y, y1))):
                m1 = ((u + u1) * cross).sum(axis=1) / 6
                m2 = ((u * u + u * u1 + u1 * u1) * cross).sum(axis=1) / 12
                m3 = ((u ** 3 + u * u * u1 + u * u1 * u1 + u1 ** 3) * cross).sum(axis=1) / 20
                mean = m1 / area
                var = m2 / area - mean ** 2
                mu3 = m3 / area - 3 * mean * m2 / area + 2 * mean ** 3
                out[s:s + chunk, k] = mu3 / np.maximum(var, 1e-300) ** 1.5
    return out


def _major_axis_vector_angle(C: np.ndarray) -> np.ndarray:
    """Lab-frame angle of the major-axis end that `phase_normalize` put the
    start point on -- the end `align_rotation` then sends to +x."""
    P0 = phase_normalize(orient_traversal(C))
    P0, _ = _as_population(P0)
    return np.arctan2(P0[:, 0, 2], P0[:, 0, 0])


def front_back_by_track(C: np.ndarray, track_ids, frames) -> np.ndarray:
    """`(N,)` bool: which cells to rotate by 180 degrees after `align_rotation`,
    decided once per *track* instead of once per frame.

    Why: the per-frame skewness rule is only as good as a single outline's
    skewness, and for cells with little front/back asymmetry that is noise. On
    the WT/tmc/pyd amnioserosa pickles the per-frame rule swaps the two ends of
    the same cell between consecutive frames in **20.5%** of frame pairs
    (frame-to-frame SD of the skewness 0.112, median |skewness| 0.100), which
    makes every even-harmonic mode flicker.

    Here the two ends are first *linked* along the track -- in each frame the
    end is chosen that points the same way (positive dot product) as the end
    chosen in the previous frame, which is the continuity rule of Poenisch et
    al. 2024 -- and the whole track is then oriented once, by the sign of its
    skewness summed over all frames. Invariant to how any frame was traced and
    to rotating the whole movie; mounting-independent; the per-frame noise is
    averaged over the track. The link assumes the long axis turns by < 90
    degrees between consecutive observations of a track (on those pickles the
    median turn is 3.9 degrees per frame; 1.9% of pairs exceed 45).

    `track_ids`: one hashable per row (e.g. embryo + cell id); `frames`: sortable
    time per row. Rows need not be sorted. A one-frame track falls back to the
    per-frame rule.
    """
    C3, _ = _as_population(C)
    P = align_rotation(phase_normalize(orient_traversal(C3)))
    skew = central_moments3(P)[:, 0]
    vec = _major_axis_vector_angle(C3)
    order_ = np.lexsort((np.asarray(frames), np.asarray(pd_codes(track_ids))))
    tracks = np.asarray(pd_codes(track_ids))[order_]
    flip = np.zeros(len(C3), dtype=bool)
    starts = np.flatnonzero(np.r_[True, tracks[1:] != tracks[:-1]])
    ends = np.r_[starts[1:], len(tracks)]
    for s0, s1 in zip(starts, ends):
        idx = order_[s0:s1]
        link = np.ones(len(idx))
        for j in range(1, len(idx)):
            same = np.cos(vec[idx[j]] - vec[idx[j - 1]]) >= 0
            link[j] = link[j - 1] if same else -link[j - 1]
        total = np.nansum(link * skew[idx])
        flip[idx] = (link * (1.0 if total >= 0 else -1.0)) < 0
    return flip


def pd_codes(values) -> np.ndarray:
    """Integer codes for arbitrary hashable per-row labels (strings, tuples)."""
    _, codes = np.unique(np.asarray([str(v) for v in values]), return_inverse=True)
    return codes


def canonicalize(C: np.ndarray, symmetric_tol: float = 1e-6, flip: np.ndarray = None) -> np.ndarray:
    """Traversal, start point and orientation removed, with the two ends of
    the major axis told apart by the shape itself. Size is untouched.

    `orient_traversal` -> `phase_normalize` -> `align_rotation`, then: if the
    shape's skewness along its major axis is negative, rotate it by 180 degrees
    (negate the even harmonics), so that the heavier tail always points to +x.
    This is the convention of Poenisch et al. 2024 (bioRxiv
    10.1101/2024.10.14.618199, Methods: "the third moment <x^3> of the mesh was
    positive"), there for 3D meshes, here for 2D outlines.

    Invariant to start point, traversal direction and rotation for every angle,
    not only on one side of a seam. A seam is unavoidable -- there is no
    continuous way to pick one of two ends -- but it now lies where the shape
    has no preferred end along its long axis (`mu3_x = 0`), instead of at a
    lab-frame orientation that half the cells of an aligned tissue sit on. For
    shapes whose `|mu3_x|` is below `symmetric_tol` (exact mirror symmetry
    across the minor axis) the skewness across the minor axis breaks the tie,
    which also changes sign under the 180-degree rotation.

    **For tracked cells pass `flip=front_back_by_track(...)`**: a single
    outline's skewness is noisy for cells with little front/back asymmetry
    (see there -- 20.5% frame-to-frame swaps on the amnioserosa pickles).
    """
    P = align_rotation(phase_normalize(orient_traversal(C)))
    P, single = _as_population(P)
    if flip is None:
        m = central_moments3(P)
        key = np.where(np.abs(m[:, 0]) > symmetric_tol, m[:, 0], m[:, 1])
        flip = key < 0
    even = (np.arange(1, P.shape[1] + 1) % 2 == 0)[None, :, None]
    R = np.where(np.asarray(flip, dtype=bool).reshape(-1)[:, None, None] & even, -P, P)
    return R[0] if single else R


def drop_size(P: np.ndarray) -> np.ndarray:
    """Divide out the harmonic-1 norm `sqrt(a1^2 + b1^2 + c1^2 + d1^2)`, which
    is rotation-invariant, so this removes size without touching shape or
    orientation. After this, harmonic-1 has unit norm by construction."""
    P, single = _as_population(P)
    scale = np.sqrt((P[:, 0] ** 2).sum(axis=1))
    scale = np.where(scale > 0, scale, np.nan)
    R = P / scale[:, None, None]
    return R[0] if single else R


def orientation_tensor(C: np.ndarray) -> np.ndarray:
    """`(N, 2)` nematic orientation `q * (cos 2psi, sin 2psi)` of the harmonic-1
    ellipse, with `q = (s1^2 - s2^2) / (s1^2 + s2^2)` from its singular values
    (0 for a circle, -> 1 for a needle).

    This is how direction is kept in `efd_features(keep_direction=True)`.
    Cell orientation is a *director* (psi and psi+pi are one state); only the
    double angle is continuous over it, and weighting it by `q` makes a round
    cell -- whose axis is undefined -- sit at the origin instead of at an
    arbitrary angle. Invariant to start point, traversal, size and to which end
    is which, so it needs none of the normalizations.
    """
    C, single = _as_population(C)
    u, sv, _ = np.linalg.svd(C[:, 0].reshape(-1, 2, 2))
    psi = np.arctan2(u[:, 1, 0], u[:, 0, 0])
    with np.errstate(invalid="ignore", divide="ignore"):
        q = (sv[:, 0] ** 2 - sv[:, 1] ** 2) / (sv[:, 0] ** 2 + sv[:, 1] ** 2)
    Q = q[:, None] * np.stack([np.cos(2 * psi), np.sin(2 * psi)], axis=1)
    return Q[0] if single else Q


def efd_feature_names(order: int = EFD_ORDER_DEFAULT, keep_direction: bool = False) -> list:
    """Column names matching `efd_features`: `efd_{h}_{a|b|c|d}` (0-based
    harmonic, as in the extraction tables) plus `Q_cos2psi`, `Q_sin2psi`."""
    names = [f"efd_{h}_{c}" for h in range(order) for c in "abcd"]
    return names + (["Q_cos2psi", "Q_sin2psi"] if keep_direction else [])


def efd_features(
    C: np.ndarray, keep_direction: bool = False, track_ids=None, frames=None
) -> np.ndarray:
    """Full normalization chain, returning a flat feature matrix: the
    `canonicalize`d, size-normalized coefficients, `(N, order*4)`.

    `track_ids` + `frames` (one per row) switch the front/back decision from
    per outline to per track (`front_back_by_track`); use them whenever the
    cells are tracked.

    `keep_direction=True` appends the two columns of `orientation_tensor`,
    `(N, order*4 + 2)`. Direction is carried there, explicitly and seam-free,
    rather than by leaving the coefficients unrotated: an unrotated coefficient
    vector represents the major axis as a *vector*, so two cells 2 degrees apart
    either side of the branch cut sit at opposite ends of the leading
    component (measured on the WT/tmc/pyd pickles: cells within 5 degrees of
    horizontal split 31/69 between the two extremes of PC1).
    """
    flip = None
    if track_ids is not None:
        if frames is None:
            raise ValueError("track_ids needs frames to order each track")
        flip = front_back_by_track(C, track_ids, frames)
    X = drop_size(canonicalize(C, flip=flip))
    X, single = _as_population(X)
    X = X.reshape(len(X), -1)
    if keep_direction:
        X = np.hstack([X, np.atleast_2d(orientation_tensor(C))])
    return X[0] if single else X


def irregularity(X: np.ndarray) -> np.ndarray:
    """Norm of harmonics >= 2 of a size-normalized feature matrix: a single
    scalar for "how far this outline departs from its best-fit ellipse".

    Harmonic 1 *is* the best-fit ellipse and has unit norm after `drop_size`,
    so everything above it is deviation from that ellipse -- which for a lobed
    cell is exactly the lobe content. Ported from allMutants cell 24 (`irreg`).
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    out = np.sqrt((X[:, 4:] ** 2).sum(axis=1))
    return out[0] if X.shape[0] == 1 else out


def ellipse_from_harmonic1(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """`(elongation, psi_rad)` from the SVD of the harmonic-1 2x2 matrix.

    Harmonic 1 maps the unit circle onto an ellipse, so its singular values are
    that ellipse's semi-axes (ratio >= 1) and its first left singular vector
    gives the major-axis direction `psi`. `psi` is a director: only `2*psi` is
    meaningful, so callers should use `sin 2psi` / `cos 2psi`.

    **`elongation` is not the true axis ratio and must not be compared with
    `Cells_info.xlsx`'s `Ellipse Major/Minor Axis Ratio`.** The series is
    parametrized by arclength, not by angle, so for an eccentric ellipse some
    of the eccentricity leaks into the odd higher harmonics and harmonic 1
    reports a compressed value. Measured on exact ellipses:

        true a/b   1.0   1.5   2.0   3.0   4.0   6.0   10.0
        harmonic 1 1.00  1.36  1.70  2.39  3.08  4.47  7.28

    It is strictly monotone, so it ranks cells correctly and is fine as a PCA
    feature or for tracking a trend over time (which is all the notebooks use
    it for) -- it is simply not on the same scale as a fitted-ellipse ratio.
    `psi`, by contrast, is exact: recovered to 0.0000 degrees at every drawn
    angle for any non-circular shape.

    Use this rather than `Cells_info.xlsx`'s `Fitted Ellipse Angle` column,
    which `data_input.extract_cell_info` stores *folded* as
    `180 - angle if angle > 90 else angle`. That is a mirror reflection, not a
    modulo, so orientations 60 degrees apart collapse onto the same value and
    the true direction cannot be recovered from it.
    """
    P, _ = _as_population(P)
    A1 = P[:, 0].reshape(-1, 2, 2)
    u, sv, _ = np.linalg.svd(A1)
    elongation = sv[:, 0] / np.maximum(sv[:, 1], 1e-9)
    psi = np.arctan2(u[:, 1, 0], u[:, 0, 0]) % np.pi
    return elongation, psi


def nematic_S(psi_rad: np.ndarray) -> float:
    """Nematic order parameter `S = |<exp(2i psi)>|` over a set of cell
    orientations: 1 = perfectly aligned, 0 = isotropic. The factor 2 is what
    makes it a *director* average, so psi and psi+pi count as the same state.
    Ported from allMutants cell 26."""
    psi = np.asarray(psi_rad, dtype=np.float64)
    psi = psi[np.isfinite(psi)]
    if psi.size == 0:
        return float("nan")
    return float(np.abs(np.exp(2j * psi).mean()))


# ---------------------------------------------------------------------------


def _self_test() -> None:
    """Numerical checks that need no dataset on disk."""

    def bumpy_ellipse(n: int = 400) -> np.ndarray:
        """The test contour from allMutants cell 6, so this module's results
        are directly comparable with the notebook's."""
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([20 * np.cos(t) + 2 * np.cos(3 * t), 9 * np.sin(t) + 1.5 * np.sin(4 * t)])

    def rotated(c: np.ndarray, deg: float) -> np.ndarray:
        a = np.deg2rad(deg)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return c @ R.T

    def egg(n: int = 400) -> np.ndarray:
        """Asymmetric along its long axis: one end blunt, one pointed."""
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([30 * np.cos(t) + 4 * np.cos(2 * t), 12 * np.sin(t) * (1 + 0.25 * np.cos(t))])

    def trapezoid(n: int = 400) -> np.ndarray:
        """Mirror-symmetric across its minor axis (mu3_x = 0 exactly) but
        wider on one side of its long axis: exercises the tie-break."""
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([30 * np.cos(t), 12 * np.sin(t) + 3 * np.cos(2 * t)])

    def lobed(k: int, amp: float) -> np.ndarray:
        tt = np.linspace(0, 2 * np.pi, 800, endpoint=False)
        r = 60.0 * (1 + amp * np.cos(k * tt))
        return np.column_stack([r * np.cos(tt), r * np.sin(tt)])

    c = bumpy_ellipse()
    order = EFD_ORDER_DEFAULT

    # 1. Invariance, swept over EVERY start position x a full turn of
    #    rotations x both traversal directions -- not one lucky roll and one
    #    lucky angle, which is how both earlier versions passed while broken.
    shapes = {"notebook contour": c, "egg": egg(), "rounded trapezoid": trapezoid()}
    worst = 0.0
    for name, shape in shapes.items():
        base = efd_features(elliptic_fourier_descriptors(shape, order))
        base_q = efd_features(elliptic_fourier_descriptors(shape, order), keep_direction=True)
        for deg in range(0, 360, 5):
            rot_shape = rotated(shape, deg)
            for k in range(0, len(shape), 20):
                for trav in (1, -1):
                    s_ = np.roll(rot_shape, k, axis=0)[::trav]
                    C_ = elliptic_fourier_descriptors(s_ * 2.5, order)
                    worst = max(worst, np.abs(efd_features(C_) - base).max())
                    # direction-kept: shape part invariant, Q rotates by 2*deg
                    q_ = efd_features(C_, keep_direction=True)
                    worst = max(worst, np.abs(q_[:-2] - base).max())
                    a2 = np.deg2rad(2 * deg)
                    Rq = np.array([[np.cos(a2), -np.sin(a2)], [np.sin(a2), np.cos(a2)]])
                    worst = max(worst, np.abs(q_[-2:] - Rq @ base_q[-2:]).max())
    assert worst < 1e-9, f"invariance broken: {worst:.3e}"
    # The canonical form must have the heavier tail at +x (the rule it claims).
    for name, shape in shapes.items():
        m = central_moments3(canonicalize(elliptic_fourier_descriptors(rotated(shape, 200), order)))
        key = m[0, 0] if abs(m[0, 0]) > 1e-6 else m[0, 1]
        assert key >= 0, f"{name}: canonical form has skewness {m[0]}"
    # The two broken baselines must still be reproducibly broken, so this test
    # would have caught each of them.
    nb = lambda C_: _notebook_features(C_)
    worst_nb = max(np.abs(nb(elliptic_fourier_descriptors(np.roll(c, k, 0), order))
                          - nb(elliptic_fourier_descriptors(c, order))).max() for k in range(0, 400, 10))
    old = lambda C_: _previous_fix_features(C_)
    worst_old = max(np.abs(old(elliptic_fourier_descriptors(rotated(c, d), order))
                           - old(elliptic_fourier_descriptors(c, order))).max() for d in range(0, 360, 5))
    assert worst_nb > 0.05 and worst_old > 0.05, (worst_nb, worst_old)
    base_py = pyefd.elliptic_fourier_descriptors(_closed(c), order=order, normalize=True)
    worst_py = max(
        np.abs(
            pyefd.elliptic_fourier_descriptors(_closed(np.roll(c, k, axis=0)), order=order, normalize=True)
            - base_py
        ).max()
        for k in range(0, len(c), 10)
    )
    assert worst_py > 1.0, f"expected pyefd normalize=True to be start-point dependent, got {worst_py:.3e}"
    print(
        f"  invariance    ours={worst:.2e} over 3 shapes x 72 angles x 20 starts x 2 traversals; "
        f"broken baselines: notebook start={worst_nb:.2f}, previous fix rotation={worst_old:.2f}, "
        f"pyefd normalize=True={worst_py:.2f}"
    )

    # 1b. Track-level front/back. A weakly skewed cell drifts and turns
    #     slowly over 60 frames, with outline noise large enough that its
    #     per-frame skewness changes sign; every frame is traced from a random
    #     start and direction. The track rule must keep the SAME physical end at
    #     +x in every frame, and give the same answer when the whole movie is
    #     rotated; the per-frame rule must not (else the fixture is too easy).
    rng_t = np.random.default_rng(7)
    tt = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    movie = []
    for f in range(60):
        noise = 0.35 * rng_t.standard_normal(3)
        if f == 0:
            noise[0] = -0.9  # first outline skewed the WRONG way: the track, not frame 0, must decide
        x = 30 * np.cos(tt) + (0.25 + noise[0]) * np.cos(2 * tt) + noise[1] * np.cos(3 * tt)
        y = 12 * np.sin(tt) + noise[2] * np.sin(2 * tt)
        movie.append(rotated(np.column_stack([x, y]), 20 + 0.8 * f))
    def track_features(angle):
        Cs = np.stack([
            elliptic_fourier_descriptors(np.roll(rotated(m_, angle), rng_t.integers(400), 0)[::rng_t.choice([1, -1])], order)
            for m_ in movie
        ])
        flip = front_back_by_track(Cs, ["cell"] * len(Cs), np.arange(len(Cs)))
        # physical end at +x: lab angle of the chosen end, relative to the drift
        vec = _major_axis_vector_angle(Cs) + np.pi * flip - np.deg2rad(angle + 20 + 0.8 * np.arange(len(Cs)))
        per_frame = (central_moments3(align_rotation(phase_normalize(orient_traversal(Cs))))[:, 0] < 0)
        vec_pf = _major_axis_vector_angle(Cs) + np.pi * per_frame - np.deg2rad(angle + 20 + 0.8 * np.arange(len(Cs)))
        mean_skew = central_moments3(canonicalize(Cs, flip=flip))[:, 0].mean()
        assert mean_skew > 0, f"track not oriented by its summed skewness ({mean_skew:.3f})"
        return np.cos(vec), np.cos(vec_pf), efd_features(Cs, track_ids=["cell"] * len(Cs), frames=np.arange(len(Cs)))
    ends, ends_pf, feats0 = track_features(0.0)
    ends_r, _, feats_r = track_features(137.0)
    assert np.all(ends > 0.5) or np.all(ends < -0.5), "track rule swapped ends within the track"
    assert np.sign(ends[0]) == np.sign(ends_r[0]), "track rule depends on the movie's rotation"
    swaps_pf = int(np.sum(np.sign(ends_pf[1:]) != np.sign(ends_pf[:-1])))
    assert swaps_pf > 0, "fixture too easy: the per-frame rule never swapped ends"
    assert np.abs(feats0 - feats_r).max() < 1e-9, "track features depend on the movie's rotation"
    print(f"  track rule    same end in 60/60 frames and under rotation; the per-frame rule swapped {swaps_pf} times")

    # 2. Scale invariance.
    base_fixed = efd_features(elliptic_fourier_descriptors(c, order))
    d_scale = np.abs(efd_features(elliptic_fourier_descriptors(c * 3.0, order)) - base_fixed).max()
    assert d_scale < 1e-10, f"scale invariance broken: {d_scale:.3e}"

    # 3. Direction: absent from the default features, present (and only
    #    there) in the two Q columns.
    rot = elliptic_fourier_descriptors(rotated(c, 30.0), order)
    d_off = np.abs(efd_features(rot) - base_fixed).max()
    d_on = np.abs(
        efd_features(rot, keep_direction=True)
        - efd_features(elliptic_fourier_descriptors(c, order), keep_direction=True)
    ).max()
    assert d_off < 1e-10, f"rotation not removed on the default path: {d_off:.3e}"
    assert d_on > 0.1, f"rotation should survive keep_direction=True, got {d_on:.3e}"
    q_circle = orientation_tensor(elliptic_fourier_descriptors(lobed(6, 0.0), order))
    assert np.abs(q_circle).max() < 1e-6, f"a circle must have Q = 0, got {q_circle}"
    assert len(efd_feature_names(order, True)) == efd_features(rot, keep_direction=True).shape[-1]
    print(f"  rotation      removed={d_off:.2e}   kept in Q={d_on:.2e}   circle Q=0")

    # 4. Reconstruction fidelity. Asserted as *convergence with order* rather
    #    than against one absolute number: a fixed threshold silently passes a
    #    truncated or mis-centred series that happens to sit under it, whereas
    #    a basis that is not converging is always wrong.
    equiv_radius = np.sqrt(np.abs(_polygon_area(c)) / np.pi)
    locus = contour_centroid(c)
    errs = []
    for o in (4, 6, 10, 20):
        rec = reconstruct_contour(elliptic_fourier_descriptors(c, o), locus=locus, num_points=2000)
        errs.append(float(np.min(np.linalg.norm(c[:, None, :] - rec[None, :, :], axis=2), axis=1).mean()))
    assert all(x > y for x, y in zip(errs, errs[1:])), f"reconstruction not converging with order: {errs}"
    assert errs[2] < 0.01 * equiv_radius, f"order-10 reconstruction error {errs[2]:.4f} px too large"
    print(
        "  reconstruct   "
        + " -> ".join(f"{100*e/equiv_radius:.2f}%" for e in errs)
        + " of R_eq  (order 4 / 6 / 10 / 20)"
    )

    # 5. Harmonic-1 ellipse. `elongation` is asserted monotone, NOT equal to
    #    the true axis ratio -- see ellipse_from_harmonic1 for why it is
    #    compressed. `psi` is asserted exact. The `% pi` is the director
    #    convention: 20 and 200 degrees are the same orientation.
    t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)

    def ellipse(ab: float) -> np.ndarray:
        return np.column_stack([ab * 20 * np.cos(t), 20 * np.sin(t)])

    ratios = [
        float(ellipse_from_harmonic1(phase_normalize(elliptic_fourier_descriptors(ellipse(ab), order)))[0][0])
        for ab in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0)
    ]
    assert abs(ratios[0] - 1.0) < 1e-6, f"a circle must give elongation 1, got {ratios[0]:.4f}"
    assert all(x < y for x, y in zip(ratios, ratios[1:])), f"elongation not monotone: {ratios}"
    for deg in (0.0, 20.0, 70.0, 155.0):
        P = phase_normalize(elliptic_fourier_descriptors(rotated(ellipse(4.0), deg), order))
        _, psi = ellipse_from_harmonic1(P)
        got = np.rad2deg(psi[0]) % 180.0
        err = min(abs(got - deg % 180.0), 180.0 - abs(got - deg % 180.0))
        assert err < 1e-3, f"psi {got:.4f} != {deg % 180.0:.4f} (err {err:.4f} deg)"
    print(
        "  ellipse       psi exact at 0/20/70/155 deg; elongation monotone "
        + "/".join(f"{v:.2f}" for v in ratios)
        + "  (true 1/1.5/2/3/4/6/10)"
    )

    # 6. Irregularity separates a pure ellipse from a lobed cell, and is
    #    monotone in lobe amplitude.

    irr = [
        float(irregularity(efd_features(elliptic_fourier_descriptors(lobed(6, a), order))))
        for a in (0.0, 0.1, 0.25, 0.40)
    ]
    assert irr[0] < 1e-3, f"a circle should have ~zero irregularity, got {irr[0]:.4f}"
    assert all(x < y for x, y in zip(irr, irr[1:])), f"irregularity not monotone: {irr}"
    print("  irregularity  " + " -> ".join(f"{v:.4f}" for v in irr) + "  (amp 0 / 0.10 / 0.25 / 0.40)")

    # 7. Nematic order parameter.
    assert abs(nematic_S(np.full(200, 0.7)) - 1.0) < 1e-12
    rng = np.random.default_rng(0)
    assert nematic_S(rng.uniform(0, np.pi, 4000)) < 0.15
    assert abs(nematic_S(np.array([0.0, np.pi])) - 1.0) < 1e-12, "psi and psi+pi must count as aligned"
    print("  nematic S     aligned=1.00, random<0.15, director convention holds")

    print("efd.py self-test: OK")


def _notebook_features(C: np.ndarray) -> np.ndarray:
    """allMutants cells 5/7, direction removed, verbatim: no half-period
    handling. Kept only so the self-test can assert it is still caught."""
    C, _ = _as_population(C)
    a1, b1, c1, d1 = C[:, 0, 0], C[:, 0, 1], C[:, 0, 2], C[:, 0, 3]
    theta = 0.5 * np.arctan2(2 * (a1 * b1 + c1 * d1), a1 ** 2 - b1 ** 2 + c1 ** 2 - d1 ** 2)
    return drop_size(align_rotation(_rotate_start_phase(C, theta))).reshape(len(C), -1)


def _previous_fix_features(C: np.ndarray) -> np.ndarray:
    """This module's pre-2026-09-19 fix: `(-1)**h` when `a1 < 0` (a lab-frame
    rule), then align. Start-point invariant, rotation dependent."""
    C, _ = _as_population(C)
    a1, b1, c1, d1 = C[:, 0, 0], C[:, 0, 1], C[:, 0, 2], C[:, 0, 3]
    theta = 0.5 * np.arctan2(2 * (a1 * b1 + c1 * d1), a1 ** 2 - b1 ** 2 + c1 ** 2 - d1 ** 2)
    R = _rotate_start_phase(C, theta)
    parity = ((-1.0) ** np.arange(1, C.shape[1] + 1))[None, :, None]
    R = np.where((R[:, 0, 0] < 0)[:, None, None], R * parity, R)
    return drop_size(align_rotation(R)).reshape(len(C), -1)


def _polygon_area(c: np.ndarray) -> float:
    """Shoelace area of a closed polygon (sign carries the winding order)."""
    x, y = c[:, 0], c[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run the numerical self-test")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
    else:
        parser.print_help()
