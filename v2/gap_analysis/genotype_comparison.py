"""Gap analysis: cross-genotype (WT vs. pyd-mutant) statistical comparison.

No code anywhere in src/ performs a two-dataset genotype comparison --
`gui_comparison.py` ("Correct Cell Classification Results") is single-dataset
classification-correction UI only, despite manuscript Fig.5-6 relying
heavily on WT-vs-pyd statistics. Signatures + docstrings only, no
implementation this round.
"""

import pandas as pd


def compare_genotypes_unpaired_ttest(wt_values: pd.Series, mutant_values: pd.Series) -> dict:
    """Unpaired Student's t-test comparing a scalar per-cell/per-track metric
    (e.g. ellipse aspect ratio, shape parameter p) between WT and pyd-mutant
    cohorts. Per manuscript Fig.5C,E,G and Fig.6D figure-legend statistics
    ("Unpaired t-tests, ****p < 0.0001")."""
    raise NotImplementedError


def compare_genotypes_wilcoxon_rank_sum(wt_values: pd.Series, mutant_values: pd.Series) -> dict:
    """Wilcoxon rank-sum (Mann-Whitney U) test, for non-normally-distributed
    metrics. Per manuscript Fig.5G figure legend ("Wilcoxon's rank sum
    test")."""
    raise NotImplementedError


def compare_genotypes_ks_test(wt_values: pd.Series, mutant_values: pd.Series) -> dict:
    """Two-sample Kolmogorov-Smirnov test for distributional differences
    between WT and mutant (e.g. oscillation period distributions, PC3 score
    density distributions). Per manuscript Fig.6E,G figure legends."""
    raise NotImplementedError
