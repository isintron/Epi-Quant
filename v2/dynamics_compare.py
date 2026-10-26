"""Comparing the dynamics of two groups of tracks: DTW, detrended oscillation
statistics, and the distance / heading-angle pair of manuscript Fig. 6H.

Calculation only -- nothing here draws. Ported from
`figures/revision_2026-10/fig6_dyn_collective/scripts/fig_dynamic_demo.py`
(`dtw`, `_heading`, `_angle`, `panel_H`) and
`figures/efd_figures/make_oscillation_figures.py` (`detrended_stats`,
`_acf_extrema`), neither of which the GUI can import: the first is a figure
script with module-level data loading, and the second runs
`make_efd_analyses`'s pickle load on import.

**Every window is given in SECONDS, not frames.** The original scripts hardcode
`DETREND_W = 11` frames and `STEP = 4` frames because their data is 30 s per
frame; `embryos_GBE_OE_GAP43` is 15 s per frame, where the same constants mean
half the time. The frame interval is a property of the movie and the GUI knows
it, so the conversion happens here, once, and a project with no frame interval
set is refused rather than silently given a 30 s/frame answer.

Two conventions worth keeping in view:

- **Amplitude** is `sqrt(2) * SD` of the detrended signal, which is the
  half peak-to-trough of a sinusoid and, unlike a peak picker, is not inflated
  by frame-to-frame noise.
- **Period** is deliberately NOT reported. `make_oscillation_figures` takes it
  from the Hilbert phase shipped with its dataset, which the GUI does not have.
  What is reported instead is the lag of the ACF's first interior minimum,
  which is immune to white noise -- a fixed-level crossing time is not.
"""

import numpy as np
import pandas as pd

# The original's own constants, in the units they are actually about.
DETREND_WINDOW_SEC = 5.5 * 60      # about one AS oscillation period
HEADING_STEP_SEC = 2 * 60          # the step every rate readout in Fig. 6 uses
DEFAULT_MIN_COVERAGE = 0.9         # Fig. 6I's track filter
DEFAULT_SPLITS = 50                # within-group half-split reference, Fig. 6H
DEFAULT_MAX_PAIRS = 1000


def frames_for(seconds: float, frame_interval_sec: float, minimum: int = 3,
               odd: bool = False) -> int:
    """A window in seconds as a whole number of frames, refusing a project that
    has not said how long a frame is."""
    if not frame_interval_sec:
        raise ValueError(
            "This analysis needs the frame interval: its windows are defined in "
            "seconds, and two movies at different intervals are not comparable "
            "frame by frame. Set it in Setup."
        )
    count = max(minimum, int(round(seconds / float(frame_interval_sec))))
    if odd and count % 2 == 0:
        count += 1
    return count


def dtw(a, b) -> float:
    """Dynamic time warping distance between two 1-D series, |difference| cost,
    full dynamic program.

    Written out rather than taken from a library on purpose. The maintainer's
    earlier notebook called `fastdtw(ts.reshape(1, -1), ...)`, which hands
    fastdtw a single 1-row sequence and gets back the plain Euclidean distance
    between the two whole series -- no warping at all. It is a quiet failure:
    the number is finite, ordered sensibly, and wrong."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n, m = len(a), len(b)
    accumulated = np.full((n + 1, m + 1), np.inf)
    accumulated[0, 0] = 0
    cost = np.abs(a[:, None] - b[None, :])
    for i in range(1, n + 1):
        row, previous, costs = accumulated[i], accumulated[i - 1], cost[i - 1]
        for j in range(1, m + 1):
            row[j] = costs[j - 1] + min(previous[j], row[j - 1], previous[j - 1])
    return float(accumulated[n, m])


def heading(mean_trajectory: pd.DataFrame, step_frames: int) -> pd.DataFrame:
    """Direction of travel of a mean trajectory: `x(t + step) - x(t)`."""
    return mean_trajectory.shift(-step_frames) - mean_trajectory


def angle_between(u, v) -> np.ndarray:
    """Angle in degrees between corresponding rows of two vector arrays."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    cosine = (u * v).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def group_distance_and_angle(values: pd.DataFrame, columns, group_column: str = "group",
                             frame_column: str = "Frame", cell_column: str = "cell",
                             frame_interval_sec: float = None, step_frames: int = None,
                             n_split: int = DEFAULT_SPLITS, seed: int = 0) -> dict:
    """Fig. 6H for two groups: per-frame Euclidean distance between the group
    mean vectors, the angle between their directions of travel, and the
    within-group reference band from random half-splits.

    **The angle is between DIRECTIONS, not between positions.** In a centred
    PCA space two group means sit on opposite sides of the origin, so the
    cosine between the positions is close to -1 whatever the groups do; the
    headings carry the actual answer.

    The reference band is what makes either number readable: it is the same
    statistic computed between two random halves of ONE group, so it says how
    far apart two samples of the same thing land."""
    if step_frames is None:
        step_frames = frames_for(HEADING_STEP_SEC, frame_interval_sec, minimum=1)
    groups = list(dict.fromkeys(values[group_column]))
    if len(groups) != 2:
        raise ValueError(f"expected exactly 2 groups, got {groups}")
    means = values.groupby([group_column, frame_column])[list(columns)].mean()
    first, second = means.loc[groups[0]], means.loc[groups[1]]
    shared = first.index.intersection(second.index)
    first, second = first.loc[shared], second.loc[shared]

    distance = np.linalg.norm(first.to_numpy() - second.to_numpy(), axis=1)
    head_a, head_b = heading(first, step_frames), heading(second, step_frames)
    usable = head_a.notna().all(1) & head_b.notna().all(1)
    angle = pd.Series(angle_between(head_a[usable].to_numpy(), head_b[usable].to_numpy()),
                      index=shared[usable.to_numpy()])

    rng = np.random.default_rng(seed)
    reference_angle, reference_distance = {}, {}
    for group in groups:
        part = values[values[group_column] == group]
        cells = part[cell_column].unique()
        angles, distances = [], []
        for _ in range(n_split):
            half = set(rng.choice(cells, len(cells) // 2, replace=False))
            a = part[part[cell_column].isin(half)].groupby(frame_column)[list(columns)].mean()
            b = part[~part[cell_column].isin(half)].groupby(frame_column)[list(columns)].mean()
            common = a.index.intersection(b.index)
            if len(common) == 0:
                continue
            distances.append(float(np.median(np.linalg.norm(
                a.loc[common].to_numpy() - b.loc[common].to_numpy(), axis=1))))
            ha, hb = heading(a, step_frames).dropna(), heading(b, step_frames).dropna()
            shared_h = ha.index.intersection(hb.index)
            if len(shared_h) == 0:
                continue
            angles.append(float(np.median(angle_between(
                ha.loc[shared_h].to_numpy(), hb.loc[shared_h].to_numpy()))))
        reference_angle[group] = np.array(angles)
        reference_distance[group] = np.array(distances)

    return {
        "groups": groups, "frames": np.asarray(shared), "distance": distance,
        "angle_frames": np.asarray(angle.index), "angle": angle.to_numpy(),
        "reference_angle": reference_angle, "reference_distance": reference_distance,
        "step_frames": step_frames,
    }


def detrended_stats(series: np.ndarray, window_frames: int, lags=None) -> dict:
    """Detrended amplitude and autocorrelation of one contiguous track.

    The track is divided by its centred running mean over `window_frames`
    (about one period), so `r(t) = value / running mean - 1`; the first and
    last `window_frames // 2` points, where the window is incomplete, are
    dropped. `None` when too little is left to say anything.

    Amplitude is reported both as a percentage of the running mean (what the
    figures plot, comparable between movies and between genotypes) and in the
    value's own units."""
    series = np.asarray(series, float)
    half = window_frames // 2
    if len(series) < window_frames + 10:
        return None
    kernel = np.ones(window_frames) / window_frames
    trend = np.convolve(series, kernel, mode="valid")
    core = series[half:len(series) - half]
    if len(core) != len(trend) or np.any(trend == 0):
        return None
    ratio = core / trend - 1
    if len(ratio) < 10:
        return None
    centred = ratio - ratio.mean()
    variance = (centred * centred).mean()
    lags = list(range(1, min(len(centred), 41))) if lags is None else list(lags)
    acf = {0: 1.0}
    for lag in lags:
        if lag >= len(centred) or variance == 0:
            break
        acf[lag] = float((centred[lag:] * centred[:-lag]).mean() / variance)
    return {
        "amplitude_pct": float(np.sqrt(2) * ratio.std() * 100),
        "amplitude_abs": float(np.sqrt(2) * (core - trend).std()),
        "n_points": int(len(ratio)),
        "acf": acf,
    }


def acf_first_minimum(acf: dict):
    """`(lag, value)` of the first interior LOCAL minimum of an ACF curve, or
    `(None, None)` if there is none inside the computed lag range.

    This is the oscillation readout to report. The time at which an ACF first
    crosses a fixed level moves with the noise floor -- add white noise and
    every curve crosses sooner, because noise scales the whole curve down --
    while the position of a minimum does not.

    `make_oscillation_figures._acf_extrema` takes the GLOBAL argmin instead
    (`j = 1 + argmin(c[1:])`). On a damped curve the two agree, which is why it
    was never a problem there; on a weakly damped one the troughs at one, two
    and three periods are within numerical noise of each other and the global
    argmin picks whichever happens to be lowest. The first local minimum is
    what the quantity is supposed to mean, so that is what is computed."""
    lags = sorted(k for k in acf if k > 0)
    if len(lags) < 3:
        return None, None
    values = [acf[k] for k in lags]
    for position in range(1, len(values) - 1):
        if values[position] <= values[position - 1] and values[position] < values[position + 1]:
            return lags[position], values[position]
    return None, None


def dtw_pairs(group_tracks: dict, max_pairs: int = DEFAULT_MAX_PAIRS, seed: int = 0) -> pd.DataFrame:
    """DTW distances for within-group and between-group track pairs.

    `group_tracks`: `{group: {track_id: 1-D array}}`, already filtered to the
    time window and the coverage threshold. Pair counts grow as n^2, so each
    class is capped at `max_pairs` by a seeded random draw and the table says
    how many pairs each class actually had and used -- a cap that is silent is
    a sample selected by nothing in particular."""
    rng = np.random.default_rng(seed)
    names = list(group_tracks)
    classes = [(a, a) for a in names] + [(a, b) for i, a in enumerate(names)
                                         for b in names[i + 1:]]
    rows = []
    for first, second in classes:
        keys_a, keys_b = list(group_tracks[first]), list(group_tracks[second])
        if first == second:
            pairs = [(a, b) for ia, a in enumerate(keys_a) for b in keys_b[ia + 1:]]
        else:
            pairs = [(a, b) for a in keys_a for b in keys_b]
        available = len(pairs)
        if available > max_pairs:
            chosen = rng.choice(available, max_pairs, replace=False)
            pairs = [pairs[int(k)] for k in sorted(chosen)]
        for a, b in pairs:
            rows.append({
                "pair": f"{first}-{second}", "group_a": first, "group_b": second,
                "track_a": a, "track_b": b,
                "dtw": dtw(group_tracks[first][a], group_tracks[second][b]),
                "pairs_available": available, "pairs_used": len(pairs),
            })
    return pd.DataFrame(rows)


def resample_to_grid(frames, values, frame_interval_sec: float, grid_sec) -> np.ndarray:
    """One track's values on a common time grid, in seconds, by linear
    interpolation with NO extrapolation (outside the track's own span the
    result is NaN).

    Movies at different frame intervals -- `embryos_GBE_GAP43` at 30 s and
    `embryos_GBE_OE_GAP43` at 15 s -- cannot be aligned frame by frame at all,
    and nearest-frame alignment silently stretches one of them."""
    times = np.asarray(frames, float) * float(frame_interval_sec)
    values = np.asarray(values, float)
    order = np.argsort(times)
    out = np.interp(np.asarray(grid_sec, float), times[order], values[order],
                    left=np.nan, right=np.nan)
    return out


def common_time_grid(intervals, spans_sec) -> np.ndarray:
    """The shared grid for movies of different frame intervals: the COARSEST
    interval (interpolating up would invent samples the slower movie never
    took) over the overlapping span."""
    step = max(float(v) for v in intervals)
    start = max(s[0] for s in spans_sec)
    stop = min(s[1] for s in spans_sec)
    if stop <= start:
        raise ValueError("the movies' time spans do not overlap")
    return np.arange(start, stop + step / 2, step)


def _self_test() -> None:
    rng = np.random.default_rng(0)

    # 1. dtw vs fig_dynamic_demo.py's implementation, copied here as the
    #    reference (that module loads data at import).
    def reference_dtw(a, b):
        n, m = len(a), len(b)
        D = np.full((n + 1, m + 1), np.inf)
        D[0, 0] = 0
        cost = np.abs(a[:, None] - b[None, :])
        for i in range(1, n + 1):
            Di, Dp, ci = D[i], D[i - 1], cost[i - 1]
            for j in range(1, m + 1):
                Di[j] = ci[j - 1] + min(Dp[j], Di[j - 1], Dp[j - 1])
        return D[n, m]

    for _ in range(5):
        a, b = rng.normal(size=rng.integers(8, 30)), rng.normal(size=rng.integers(8, 30))
        assert dtw(a, b) == reference_dtw(a, b), "dtw diverged"
    # DTW is a warp distance, not a Euclidean one: a shifted copy of a signal
    # is near-zero away under DTW and far under the subtraction the old
    # notebook's fastdtw call actually computed.
    base = np.sin(np.linspace(0, 6 * np.pi, 60))
    shifted = np.sin(np.linspace(0, 6 * np.pi, 60) + 0.8)
    euclidean = float(np.abs(base - shifted).sum())
    assert dtw(base, shifted) < euclidean / 3, (dtw(base, shifted), euclidean)
    assert dtw(base, base) == 0.0

    # 2. Windows are seconds, and the frame count follows the interval.
    assert frames_for(DETREND_WINDOW_SEC, 30) == 11, "30 s/frame must give the original's 11"
    assert frames_for(DETREND_WINDOW_SEC, 15) == 22, "15 s/frame is half the time per frame"
    assert frames_for(HEADING_STEP_SEC, 30, minimum=1) == 4, "the original's STEP"
    try:
        frames_for(DETREND_WINDOW_SEC, None)
        raise AssertionError("a project with no frame interval must be refused")
    except ValueError as error:
        assert "frame interval" in str(error)

    # 3. detrended_stats against the original's inline formula, on a track with
    #    a known amplitude riding on a linear trend.
    period, n = 12, 120
    t = np.arange(n)
    signal = (100 + 0.5 * t) * (1 + 0.1 * np.sin(2 * np.pi * t / period))
    window = 11
    stats = detrended_stats(signal, window)
    half = window // 2
    trend = np.convolve(signal, np.ones(window) / window, mode="valid")
    r = signal[half:len(signal) - half] / trend - 1
    assert abs(stats["amplitude_pct"] - np.sqrt(2) * r.std() * 100) < 1e-12
    # sqrt(2) * SD of a 10% sinusoid is its semi-amplitude, attenuated by the
    # running mean: an 11-point mean passes a period-12 sinusoid at
    # sin(11*pi/12) / (11 * sin(pi/12)) = 0.0909 of its height, so dividing it
    # out leaves 10% * (1 - 0.0909) = 9.09%. The detrending window is
    # deliberately about one period, which is exactly where this bias sits --
    # amplitudes are comparable between tracks, not absolute.
    expected = 10 * (1 - np.sin(11 * np.pi / 12) / (11 * np.sin(np.pi / 12)))
    assert abs(stats["amplitude_pct"] - expected) < 0.1, (stats["amplitude_pct"], expected)
    lag, value = acf_first_minimum(stats["acf"])
    assert lag == period // 2, f"first ACF minimum at {lag}, expected half a period"

    # The first minimum survives noise; a fixed-level crossing does not.
    noisy = signal * (1 + 0.08 * rng.normal(size=n))
    noisy_stats = detrended_stats(noisy, window)
    noisy_lag, _ = acf_first_minimum(noisy_stats["acf"])
    crossing = next(k for k in sorted(stats["acf"]) if stats["acf"][k] < 0.5)
    noisy_crossing = next(k for k in sorted(noisy_stats["acf"]) if noisy_stats["acf"][k] < 0.5)
    # Noise at 80% of the signal's own amplitude moves the first minimum by at
    # most one lag (a 120-point track is short), while it halves the crossing
    # time -- which is the asymmetry the choice of readout rests on.
    assert abs(noisy_lag - lag) <= 1, f"first minimum moved {lag} -> {noisy_lag} under noise"
    assert noisy_crossing <= crossing / 2, (
        "fixture noise was too small to move the 0.5 crossing, so it cannot show the difference")

    # 4. Distance and angle: two groups moving the same way must give a small
    #    angle even though their mean POSITIONS are on opposite sides of zero.
    frames = np.arange(40)
    rows = []
    for group, offset in (("A", +3.0), ("B", -3.0)):
        for cell in range(12):
            jitter = rng.normal(0, 0.1, len(frames))
            rows.append(pd.DataFrame({
                "group": group, "Frame": frames, "cell": f"{group}{cell}",
                "PC1": offset + 0.05 * frames + jitter,
                "PC2": 0.02 * frames + rng.normal(0, 0.1, len(frames)),
            }))
    table = pd.concat(rows, ignore_index=True)
    result = group_distance_and_angle(table, ["PC1", "PC2"], frame_interval_sec=30)
    assert np.median(result["angle"]) < 20, np.median(result["angle"])
    mean_a = table[table.group == "A"].groupby("Frame")[["PC1", "PC2"]].mean().to_numpy()
    mean_b = table[table.group == "B"].groupby("Frame")[["PC1", "PC2"]].mean().to_numpy()
    cosine = (mean_a * mean_b).sum(1) / np.linalg.norm(mean_a, axis=1) / np.linalg.norm(mean_b, axis=1)
    assert np.median(cosine) < -0.5, (
        "fixture does not reproduce the opposite-sides case the heading rule exists for")
    assert all(len(v) == DEFAULT_SPLITS for v in result["reference_angle"].values())
    assert result["step_frames"] == 4

    # 5. DTW pair classes, and the cap that must report itself.
    tracks = {"A": {i: rng.normal(size=20) for i in range(40)},
              "B": {i: rng.normal(size=20) for i in range(40)}}
    pairs = dtw_pairs(tracks, max_pairs=100)
    assert set(pairs["pair"]) == {"A-A", "B-B", "A-B"}, set(pairs["pair"])
    counts = pairs.groupby("pair").size().to_dict()
    assert all(v == 100 for v in counts.values()), counts
    assert pairs[pairs.pair == "A-B"]["pairs_available"].iloc[0] == 1600
    assert pairs[pairs.pair == "A-A"]["pairs_available"].iloc[0] == 40 * 39 // 2
    assert (dtw_pairs(tracks, max_pairs=100, seed=0)["track_a"].to_numpy()
            == dtw_pairs(tracks, max_pairs=100, seed=0)["track_a"].to_numpy()).all()

    # 6. Different frame intervals: the shared grid takes the coarser step and
    #    interpolation never extrapolates.
    grid = common_time_grid([30, 15], [(0, 300), (0, 600)])
    assert grid[1] - grid[0] == 30 and grid[-1] == 300, (grid[0], grid[1], grid[-1])
    sampled = resample_to_grid([0, 1, 2, 3], [0.0, 1.0, 2.0, 3.0], 15, [0, 15, 30, 100])
    assert np.allclose(sampled[:3], [0, 1, 2]) and np.isnan(sampled[3]), sampled

    print(f"  dtw identical to fig_dynamic_demo; shift distance {dtw(base, shifted):.2f} "
          f"vs Euclidean {euclidean:.2f}")
    print(f"  detrend window 5.5 min = {frames_for(DETREND_WINDOW_SEC, 30)} frames at 30 s/frame, "
          f"{frames_for(DETREND_WINDOW_SEC, 15)} at 15 s/frame")
    print(f"  ACF first minimum lag {lag} clean, {noisy_lag} under 80%-of-signal noise; "
          f"the 0.5 crossing moved {crossing} -> {noisy_crossing}")
    print(f"  heading angle {np.median(result['angle']):.1f} deg while the position cosine is "
          f"{np.median(cosine):.2f}")
    print("\ndynamics_compare.py self-test: OK")


if __name__ == "__main__":
    _self_test()
