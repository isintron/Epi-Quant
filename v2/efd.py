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
**2.0** on coefficients whose leading term is 1 by construction. The
normalization chain here (`phase_normalize` -> `align_rotation` -> `drop_size`)
is a port of the notebooks (allMutants cells 5, 24, 26) with the half-period
correction described in `phase_normalize`, and is invariant to ~6e-16.
`irregularity`, `ellipse_from_harmonic1` and `nematic_S` are likewise ports.

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


def phase_normalize(C: np.ndarray, resolve_half_period: bool = True) -> np.ndarray:
    """Remove the arbitrary contour starting point. Size and orientation are
    untouched.

    The starting phase comes from the standard Kuhl & Giardina expression
    `theta = 0.5 * arctan2(2*(a1*b1 + c1*d1), a1^2 - b1^2 + c1^2 - d1^2)`,
    which pins the harmonic-1 ellipse's major axis to phase zero.

    **Correction to the notebooks.** `arctan2` returns `2*theta` in `(-pi, pi]`,
    so `theta` is determined only modulo `pi` -- i.e. the start point is pinned
    only up to *half a period*. A half-period shift multiplies harmonic `h` by
    `(-1)**h`, which the rotation above does not undo. The notebooks
    (allMutants cell 5) resolve the leftover sign with `if R[0, 0] < 0: R = -R`,
    negating *every* harmonic uniformly. That is a 180-degree rotation of the
    shape, not a half-period shift of the start point: it fixes the odd
    harmonics and puts the even ones on the wrong sign.

    Measured on the notebooks' own test contour, sweeping the start point over
    all 400 positions: for roughly half of them the uniform-negation form
    leaves a residual of 2.34 on the raw coefficients (whose harmonic-1 norm is
    ~22 here), which is 0.11 after `drop_size` -- i.e. ~11% of the dominant
    coefficient. Applying `(-1)**h` instead leaves ~6e-16. The notebook's
    self-test uses `np.roll(c, 50)`, which happens to land in the working half,
    which is why this was never caught. The practical consequence is that the
    *same cell* lands at two different points in shape space depending on where
    OpenCV began tracing its contour, smearing the PCA.

    `align_rotation` does **not** absorb this -- the residual is the same 0.11
    on the direction-removed path (measured), because the leftover sign error
    lives in the even harmonics and a rotation cannot undo it. The fix is
    needed on both paths.

    Pass `resolve_half_period=False` to reproduce the notebook behaviour
    exactly (only useful for diffing against it).
    """
    C, single = _as_population(C)
    a1, b1, c1, d1 = C[:, 0, 0], C[:, 0, 1], C[:, 0, 2], C[:, 0, 3]
    theta = 0.5 * np.arctan2(2 * (a1 * b1 + c1 * d1), a1 ** 2 - b1 ** 2 + c1 ** 2 - d1 ** 2)
    R = _rotate_start_phase(C, theta)

    if resolve_half_period:
        # Harmonic h flips by (-1)**h, i.e. the half-period start shift.
        parity = ((-1.0) ** np.arange(1, C.shape[1] + 1))[None, :, None]
        flip = R[:, 0, 0] < 0
        R = np.where(flip[:, None, None], R * parity, R)
    else:
        flip = R[:, 0, 0] < 0
        R = np.where(flip[:, None, None], -R, R)

    return R[0] if single else R


def align_rotation(P: np.ndarray) -> np.ndarray:
    """Rotate each cell onto its major axis, removing orientation. Expects
    phase-normalized coefficients. Size is untouched."""
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


def drop_size(P: np.ndarray) -> np.ndarray:
    """Divide out the harmonic-1 norm `sqrt(a1^2 + b1^2 + c1^2 + d1^2)`, which
    is rotation-invariant, so this removes size without touching shape or
    orientation. After this, harmonic-1 has unit norm by construction."""
    P, single = _as_population(P)
    scale = np.sqrt((P[:, 0] ** 2).sum(axis=1))
    scale = np.where(scale > 0, scale, np.nan)
    R = P / scale[:, None, None]
    return R[0] if single else R


def efd_features(
    C: np.ndarray, keep_direction: bool = False, resolve_half_period: bool = True
) -> np.ndarray:
    """Full normalization chain, returning a flat `(N, order*4)` feature matrix.

    `keep_direction=False` (the default) removes orientation as well, giving a
    pure shape spectrum; orientation is then carried separately and explicitly
    as `(sin 2psi, cos 2psi)` -- see `ellipse_from_harmonic1`. Splitting them is
    deliberate: cell orientation is a *director* (psi and psi+pi are the same
    physical state), so it cannot be represented as a plain linear coordinate,
    and mixing it into the shape coefficients is what makes an orientation axis
    masquerade as a shape axis.
    """
    P = phase_normalize(C, resolve_half_period=resolve_half_period)
    P, single = _as_population(P)
    if not keep_direction:
        P = align_rotation(P)
    X = drop_size(P).reshape(len(P), -1)
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

    c = bumpy_ellipse()
    order = EFD_ORDER_DEFAULT

    # 1. Start-point invariance, swept over EVERY start position -- not just
    #    one lucky roll. The `resolve_half_period=False` leg asserts the
    #    notebook's bug is still reproducible, so a future edit cannot quietly
    #    regress the fixed path back onto it.
    base_fixed = efd_features(elliptic_fourier_descriptors(c, order))
    base_nb = efd_features(elliptic_fourier_descriptors(c, order), resolve_half_period=False)
    worst_fixed, worst_nb = 0.0, 0.0
    for k in range(0, len(c), 10):
        rolled = elliptic_fourier_descriptors(np.roll(c, k, axis=0), order)
        worst_fixed = max(worst_fixed, np.abs(efd_features(rolled) - base_fixed).max())
        worst_nb = max(
            worst_nb, np.abs(efd_features(rolled, resolve_half_period=False) - base_nb).max()
        )
    assert worst_fixed < 1e-10, f"start-point invariance broken: {worst_fixed:.3e}"
    # ~0.11 on size-normalized coefficients whose harmonic-1 norm is 1.
    assert worst_nb > 0.05, (
        f"expected the notebook's half-period bug to be reproducible with "
        f"resolve_half_period=False, but the worst residual was only {worst_nb:.3e}"
    )
    # pyefd's own normalize=True is not start-point invariant either. Asserted
    # so that "why not just use pyefd's normalization?" has a standing answer.
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
        f"  start point   ours={worst_fixed:.2e}   notebook-repro={worst_nb:.2e}   "
        f"pyefd normalize=True={worst_py:.2e}  (both baselines broken, as expected)"
    )

    # 2. Scale invariance.
    d_scale = np.abs(efd_features(elliptic_fourier_descriptors(c * 3.0, order)) - base_fixed).max()
    assert d_scale < 1e-10, f"scale invariance broken: {d_scale:.3e}"

    # 3. Rotation: removed on the default path, retained when asked for.
    rot = elliptic_fourier_descriptors(rotated(c, 30.0), order)
    d_off = np.abs(efd_features(rot) - base_fixed).max()
    d_on = np.abs(
        efd_features(rot, keep_direction=True)
        - efd_features(elliptic_fourier_descriptors(c, order), keep_direction=True)
    ).max()
    assert d_off < 1e-10, f"rotation not removed on the default path: {d_off:.3e}"
    assert d_on > 0.1, f"rotation should survive keep_direction=True, got {d_on:.3e}"
    print(f"  rotation      removed={d_off:.2e}   kept={d_on:.2e}")

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
    def lobed(k: int, amp: float) -> np.ndarray:
        tt = np.linspace(0, 2 * np.pi, 800, endpoint=False)
        r = 60.0 * (1 + amp * np.cos(k * tt))
        return np.column_stack([r * np.cos(tt), r * np.sin(tt)])

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
