"""Shape descriptors computed from EFD coefficients alone, for asking how much
of each principal component a hand-picked descriptor panel can reproduce.

Why compute them from the *reconstruction* rather than from the mask: the
question is whether the PCs carry shape information the descriptors miss. If
the descriptors were measured on the raw mask and the PCs on order-10
coefficients, a low R-squared could simply mean "the mask has detail the
coefficients threw away". Taking both from the same order-10 outline removes
that confound -- any residual is a genuine failure of the descriptor panel to
span the coefficient space, not a difference in input.

Everything here is scale-free and pose-free: the input is expected to be the
`canonicalize`d, `drop_size`d coefficients (v2/efd.py), so x is the major axis
and the harmonic-1 norm is 1.

Two families:

* **moments** -- normalized central moments up to fourth order. These are the
  natural partners of the even/odd structure of the coefficients: `mu30_x` is
  front/back skewness, `mu21`/`mu12` are bending ("banana"), `mu40` separates a
  waisted outline from a lens-shaped one. Unlike elongation, nematic order and
  irregularity -- which are all *amplitude* quantities, built from norms -- the
  moments are sensitive to the relative *phase* of the harmonics, i.e. to where
  along the cell a bulge sits. That is the reason they can match components no
  amplitude descriptor can.
* **geometry** -- solidity, convexity, shape index, circularity, lobe count,
  max concavity depth, bending energy, curvature sign changes. Same definitions
  as v2/shape_features.py (which measures them on masks), re-expressed for a
  Fourier outline: curvature and arc length are differentiated analytically
  from the series rather than by finite differences, so they do not depend on
  how the outline was sampled.
"""
from __future__ import annotations

import numpy as np

MOMENT_NAMES = ['mu30_x', 'mu30_y', 'mu21', 'mu12', 'mu40', 'mu04', 'mu22']
GEOMETRY_NAMES = ['circularity', 'shape_index', 'solidity', 'convexity',
                  'max_concavity_depth', 'lobe_count', 'bending_energy', 'curv_sign_changes']
DESCRIPTOR_NAMES = MOMENT_NAMES + GEOMETRY_NAMES


def reconstruct(P: np.ndarray, n_points: int = 128) -> tuple:
    """Outline `(N, n_points)` x and y plus the first two analytic derivatives,
    from `(N, order, 4)` coefficients. The DC term is zero here, so the outline
    is already centred on the Fourier centroid; the true area centroid is
    removed by the moment code."""
    P = np.asarray(P, float)
    order = P.shape[1]
    t = 2 * np.pi * np.arange(n_points) / n_points
    n = np.arange(1, order + 1)[:, None]
    cos, sin = np.cos(n * t), np.sin(n * t)
    a, b, c, d = P[:, :, 0], P[:, :, 1], P[:, :, 2], P[:, :, 3]
    x, y = a @ cos + b @ sin, c @ cos + d @ sin
    dx, dy = (-a * n.T) @ sin + (b * n.T) @ cos, (-c * n.T) @ sin + (d * n.T) @ cos
    ddx = (-a * n.T ** 2) @ cos + (-b * n.T ** 2) @ sin
    ddy = (-c * n.T ** 2) @ cos + (-d * n.T ** 2) @ sin
    return x, y, dx, dy, ddx, ddy


def _poly_moments(x: np.ndarray, y: np.ndarray) -> dict:
    """Area-weighted central moments up to fourth order, exactly, by Green's
    theorem on the closed polygon through the sampled points.

    Green's theorem gives `int x^p y^q dA` as a boundary integral; for a
    polygon each edge contributes a closed-form term. Doing it this way rather
    than by summing over boundary *points* makes the result a property of the
    shape and not of `n_points`.
    """
    x1, y1 = np.roll(x, -1, axis=1), np.roll(y, -1, axis=1)
    cross = x * y1 - x1 * y
    area = cross.sum(axis=1) / 2
    # centroid
    cx = ((x + x1) * cross).sum(axis=1) / (6 * area)
    cy = ((y + y1) * cross).sum(axis=1) / (6 * area)
    u, u1 = x - cx[:, None], x1 - cx[:, None]
    v, v1 = y - cy[:, None], y1 - cy[:, None]
    cr = u * v1 - u1 * v
    A = cr.sum(axis=1) / 2

    def m(p, q):
        """`int u^p v^q dA` over the polygon, exactly (Steger 1996, "On the
        calculation of arbitrary moments of polygons"). Reduces to the
        `/6`, `/12`, `/20` edge sums of `v2/efd.central_moments3` for the
        first three orders; checked against the exact moments of a square in
        `_self_test`."""
        tot = np.zeros(len(u))
        for i in range(p + 1):
            for j in range(q + 1):
                coef = _binom(i + j, j) * _binom(p + q - i - j, q - j)
                tot = tot + coef * (u ** i * u1 ** (p - i) * v ** j * v1 ** (q - j) * cr).sum(axis=1)
        return tot / ((p + q + 2) * (p + q + 1) * _binom(p + q, p))

    s = {}
    m20, m02 = m(2, 0) / A, m(0, 2) / A
    sx, sy = np.sqrt(np.maximum(m20, 1e-300)), np.sqrt(np.maximum(m02, 1e-300))
    s['mu30_x'] = m(3, 0) / A / sx ** 3
    s['mu30_y'] = m(0, 3) / A / sy ** 3
    s['mu21'] = m(2, 1) / A / (sx ** 2 * sy)
    s['mu12'] = m(1, 2) / A / (sx * sy ** 2)
    s['mu40'] = m(4, 0) / A / sx ** 4
    s['mu04'] = m(0, 4) / A / sy ** 4
    s['mu22'] = m(2, 2) / A / (sx ** 2 * sy ** 2)
    s['_area'] = A
    return s


_BINOM_CACHE: dict = {}


def _binom(n: int, k: int) -> float:
    if (n, k) not in _BINOM_CACHE:
        from math import comb
        _BINOM_CACHE[(n, k)] = float(comb(n, k))
    return _BINOM_CACHE[(n, k)]


def _runs_circular(mask: np.ndarray) -> list:
    """Lengths of maximal True runs in a circular 1-D boolean array."""
    if not mask.any():
        return []
    if mask.all():
        return [len(mask)]
    shift = np.argmin(mask)
    m = np.roll(mask, -shift)
    idx = np.flatnonzero(np.diff(np.r_[False, m, False]) != 0)
    return list(idx[1::2] - idx[0::2])


def geometry(x, y, dx, dy, ddx, ddy, area, min_run_fraction: float = 0.02) -> dict:
    """Curvature-, perimeter- and hull-based descriptors for one chunk."""
    n_pts = x.shape[1]
    dt = 2 * np.pi / n_pts
    speed = np.hypot(dx, dy)
    perim = speed.sum(axis=1) * dt
    kappa = (dx * ddy - dy * ddx) / np.maximum(speed ** 3, 1e-300)
    r_eq = np.sqrt(np.abs(area) / np.pi)

    out = {
        'circularity': 4 * np.pi * np.abs(area) / np.maximum(perim ** 2, 1e-300),
        'shape_index': perim / np.maximum(np.sqrt(np.abs(area)), 1e-300),
        # circle -> 1: R_eq * integral(kappa^2 ds) / (2 pi)
        'bending_energy': r_eq * (kappa ** 2 * speed).sum(axis=1) * dt / (2 * np.pi),
    }
    # curvature sign changes, ignoring runs shorter than min_run_fraction of the
    # outline -- the same guard v2/shape_features.py uses, so that a single noisy
    # point does not read as a lobe.
    min_run = max(2, int(round(min_run_fraction * n_pts)))
    concave = kappa < 0
    sign_changes = np.empty(len(x))
    lobes = np.empty(len(x))
    for i in range(len(x)):
        cc = [r for r in _runs_circular(concave[i]) if r >= min_run]
        cv = [r for r in _runs_circular(~concave[i]) if r >= min_run]
        sign_changes[i] = 2 * len(cc) if cc else 0
        lobes[i] = len(cv) if cc else 0
    out['curv_sign_changes'] = sign_changes
    out['lobe_count'] = lobes

    # convex hull: solidity, convexity, deepest concavity (fraction of R_eq)
    from scipy.spatial import ConvexHull
    sol = np.empty(len(x)); conv = np.empty(len(x)); depth = np.empty(len(x))
    pts = np.stack([x, y], axis=-1)
    for i in range(len(x)):
        try:
            h = ConvexHull(pts[i])
        except Exception:                                   # degenerate outline
            sol[i] = conv[i] = np.nan; depth[i] = np.nan; continue
        v = pts[i][h.vertices]
        sol[i] = np.abs(area[i]) / h.volume                 # 2-D: .volume is the area
        conv[i] = _closed_length(v) / max(perim[i], 1e-300)
        depth[i] = _hull_depth(pts[i], v) / max(r_eq[i], 1e-300)
    out['solidity'] = sol
    out['convexity'] = conv
    out['max_concavity_depth'] = depth
    return out


def _closed_length(v: np.ndarray) -> float:
    return float(np.hypot(*(np.roll(v, -1, axis=0) - v).T).sum())


def _hull_depth(pts: np.ndarray, hull: np.ndarray) -> float:
    """Largest perpendicular distance from a contour point to the hull boundary.

    Distance to each hull *edge segment* (not the infinite line), minimised over
    edges -- the same quantity `v2/shape_features.hull_deviation` measures.
    """
    a = hull
    b = np.roll(hull, -1, axis=0)
    ab = b - a                                              # (H, 2)
    ap = pts[:, None, :] - a[None, :, :]                    # (P, H, 2)
    denom = (ab ** 2).sum(1)
    t = np.clip((ap * ab).sum(-1) / np.maximum(denom, 1e-300), 0, 1)
    closest = a[None] + t[..., None] * ab[None]
    return float(np.hypot(*(pts[:, None, :] - closest).transpose(2, 0, 1)).min(axis=1).max())


def moments(P: np.ndarray, n_points: int = 128, chunk: int = 20000) -> dict:
    """Only `MOMENT_NAMES`, for callers that do not need the hull and curvature
    descriptors. The convex hull is what makes `descriptors` slow (a per-cell
    loop); the moments are pure array arithmetic, so this is ~35 s for the full
    amnioserosa set against ~4 min."""
    P = np.asarray(P, float)
    out = {k: np.empty(len(P)) for k in MOMENT_NAMES}
    for s in range(0, len(P), chunk):
        sl = slice(s, min(s + chunk, len(P)))
        x, y = reconstruct(P[sl], n_points)[:2]
        m = _poly_moments(x, y)
        for k in MOMENT_NAMES:
            out[k][sl] = m[k]
    return out


def descriptors(P: np.ndarray, n_points: int = 128, chunk: int = 5000, verbose: bool = False) -> dict:
    """All of `DESCRIPTOR_NAMES` for `(N, order, 4)` canonical unit-size
    coefficients, as a dict of `(N,)` arrays."""
    P = np.asarray(P, float)
    out = {k: np.empty(len(P)) for k in DESCRIPTOR_NAMES}
    for s in range(0, len(P), chunk):
        sl = slice(s, min(s + chunk, len(P)))
        x, y, dx, dy, ddx, ddy = reconstruct(P[sl], n_points)
        mom = _poly_moments(x, y)
        geo = geometry(x, y, dx, dy, ddx, ddy, mom.pop('_area'))
        for k in MOMENT_NAMES:
            out[k][sl] = mom[k]
        for k in GEOMETRY_NAMES:
            out[k][sl] = geo[k]
        if verbose:
            print(f'    descriptors {sl.stop}/{len(P)}', flush=True)
    return out


# ------------------------------------------------------------------ self-test
def _self_test() -> None:
    import efd as _efd

    def coeffs_of(xy):
        return _efd.elliptic_fourier_descriptors(xy, order=10)

    # 0. the polygon moment engine, against a square whose moments are exact
    sq = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])[None]
    mm = _poly_moments(sq[..., 0], sq[..., 1])
    assert abs(mm['_area'][0] - 4) < 1e-12, mm['_area']
    for k, want in (('mu40', (1 / 5) / (1 / 3) ** 2), ('mu22', (1 / 9) / (1 / 3) ** 2),
                    ('mu30_x', 0.0), ('mu21', 0.0)):
        assert abs(mm[k][0] - want) < 1e-12, f'square {k} = {mm[k][0]:.6f}, want {want:.6f}'
    print(f"  polygon moments exact on a square (mu40 = {mm['mu40'][0]:.4f} = 9/5)")

    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    shapes = {
        'circle': np.stack([np.cos(t), np.sin(t)], 1),
        'ellipse': np.stack([2 * np.cos(t), np.sin(t)], 1),
        'egg': np.stack([np.cos(t) + .25 * np.cos(2 * t), .9 * np.sin(t)], 1),
        'five_lobed': np.stack([(1 + .25 * np.cos(5 * t)) * np.cos(t),
                                (1 + .25 * np.cos(5 * t)) * np.sin(t)], 1),
    }
    C = np.stack([coeffs_of(v) for v in shapes.values()])
    P = _efd.drop_size(_efd.canonicalize(C))
    d = descriptors(P)
    names = list(shapes)

    # 1. a circle is the fixed point of every geometric descriptor
    i = names.index('circle')
    for k, want in (('circularity', 1.0), ('solidity', 1.0), ('convexity', 1.0),
                    ('bending_energy', 1.0), ('shape_index', 2 * np.sqrt(np.pi))):
        assert abs(d[k][i] - want) < 2e-3, f'circle {k} = {d[k][i]:.4f}, want {want:.4f}'
    assert d['lobe_count'][i] == 0 and d['curv_sign_changes'][i] == 0
    for k in ('mu30_x', 'mu30_y', 'mu21', 'mu12'):
        assert abs(d[k][i]) < 1e-6, f'circle {k} = {d[k][i]:.2e}'
    # a uniform disc has mu40 = 2 * sigma^4 / sigma^4 = 2 (E[x^4] over a disc)
    assert abs(d['mu40'][i] - 2.0) < 1e-2, d['mu40'][i]

    # 2. an ellipse is still convex and unlobed; only its aspect changed
    i = names.index('ellipse')
    assert d['solidity'][i] > .999 and d['lobe_count'][i] == 0
    assert abs(d['mu30_x'][i]) < 1e-6

    # 3. the egg is skewed along its long axis, and canonicalize puts the heavy
    #    tail at +x, so the skewness must be positive whichever way it was drawn
    i = names.index('egg')
    assert d['mu30_x'][i] > .05, d['mu30_x'][i]

    # 4. five lobes are found as five lobes, and the outline is non-convex
    i = names.index('five_lobed')
    assert d['lobe_count'][i] == 5, d['lobe_count'][i]
    assert d['curv_sign_changes'][i] == 10, d['curv_sign_changes'][i]
    assert d['solidity'][i] < .95 and d['bending_energy'][i] > 2

    # 5. pose invariance: rotating, re-starting and reversing the traced outline
    #    must not move any descriptor (they are taken after canonicalize)
    base = descriptors(P[[names.index('five_lobed')]])
    v = shapes['five_lobed']
    worst = 0.0
    for ang in (0.3, 1.1, 2.7, 4.4):
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        for roll in (0, 37, 211):
            for rev in (1, -1):
                w = np.roll(v[::rev], roll, axis=0) @ R.T
                Q = _efd.drop_size(_efd.canonicalize(coeffs_of(w)[None]))
                g = descriptors(Q)
                for k in DESCRIPTOR_NAMES:
                    scale = max(abs(base[k][0]), 1e-3)
                    worst = max(worst, abs(g[k][0] - base[k][0]) / scale)
    assert worst < 5e-2, f'pose dependence {worst:.3f}'
    print(f'  pose invariance: worst relative drift {worst:.2e} over 24 poses')

    # 6. moments must respond to the front/back flip the way the theory says:
    #    odd-in-x moments change sign, even-in-x ones do not
    Pe = P[[names.index('egg')]]
    flipped = _efd.drop_size(_efd.canonicalize(C[[names.index('egg')]], flip=np.array([True])))
    a, b = descriptors(Pe), descriptors(flipped)
    for k, odd in (('mu30_x', True), ('mu12', True), ('mu40', False), ('mu22', False),
                   ('solidity', False), ('bending_energy', False)):
        want = -a[k][0] if odd else a[k][0]
        assert abs(b[k][0] - want) < 1e-6 * max(1, abs(want)), f'{k}: {b[k][0]} vs {want}'
    print('  front/back parity of the moments is as predicted')

    # 7. the cheap moments-only path must agree with the full one
    m_only = moments(P)
    for k in MOMENT_NAMES:
        assert np.allclose(m_only[k], d[k], rtol=0, atol=1e-12), k
    print('  moments() matches descriptors() exactly')
    print('efd_descriptors self-test passed')


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'v2'))
    _self_test()
