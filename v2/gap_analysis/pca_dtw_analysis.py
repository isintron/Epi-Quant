"""Gap analysis: PCA-subtractive + Dynamic Time Warping (DTW) trajectory
comparison between genotypes.

No `dtw`-related code exists anywhere in src/. Implementing
`compute_dtw_divergence` will need a new dependency (e.g. `fastdtw` or
`dtaidistance`), not currently in the project's requirements. Signatures +
docstrings only, no implementation this round.
"""

import numpy as np
import pandas as pd


def compute_pc3_density_distributions(wt_scores: pd.Series, mutant_scores: pd.Series) -> dict:
    """PC3 score density-distribution comparison between genotypes at GBR
    and DC stages. Per manuscript Fig.6F,G (WT converges to a unimodal
    compact distribution at DC; pyd remains bimodal)."""
    raise NotImplementedError


def compute_trajectory_distance(wt_trajectory: np.ndarray, mutant_trajectory: np.ndarray, metric: str = "euclidean") -> float:
    """Euclidean or cosine distance between WT and mutant trajectories in
    PC1-3 space. Per manuscript Fig.6H."""
    raise NotImplementedError


def compute_dtw_divergence(wt_trajectory: np.ndarray, mutant_trajectory: np.ndarray) -> dict:
    """Dynamic Time Warping divergence between WT and mutant PC1-3
    trajectories, testing whether divergence reflects structural differences
    rather than mere phase shifts. Per manuscript Fig.6I. Needs a new
    dependency (`fastdtw`/`dtaidistance`) to implement."""
    raise NotImplementedError
