"""Gap analysis: cell-area oscillation period/amplitude.

v1 only has an MSD (mean-squared-displacement) power-law exponent gamma
(duplicated in gui_cell_shadow_plot.py and gui_quantitative_analysis.py),
which is a different metric (scaling of positional fluctuation vs. time
lag). Nothing extracts oscillation PERIOD or AMPLITUDE from a track's
area-over-time series. Signature + docstring only, no implementation this
round.
"""

import pandas as pd


def compute_area_oscillation(track_area_series: pd.Series, frame_interval_minutes: float) -> dict:
    """Extracts oscillation period and amplitude from a single track's
    cell-area-over-time series. Manuscript reports ~6 min periodicity during
    the elliptical AS stage (Fig.4B) and a significant WT-vs-pyd amplitude
    difference with comparable frequency (Fig.6D,E, unpaired t-test
    **p < 0.01 for amplitude; Kolmogorov-Smirnov for period distribution)."""
    raise NotImplementedError
