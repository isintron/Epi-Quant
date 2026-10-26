"""Module 2: PCA + K-Means cell classification.

Ported from `gui_main.MainWindow.bio_compute_cell_pca` /
`bio_clustering_k_mean` (src/gui_main.py:1391-1580). Both were already
`self`-free/pure in v1 aside from their file-writing side effects, which are
now gated behind an optional `output_dir` (pass `None` to skip all writes).

Note: this module does NOT produce a `leading_edge` column — that is computed
downstream, from segmentation contours rather than PCA/K-means output, in
`manual_correction_classification.compute_leading_edge`.

Bug Fix #2 (see v2/README.md changelog) fixed a duplicate key in v1's
feature-abbreviation dict, which silently left `Cell Right Boundary` with no
abbreviation. The dict itself is gone as of Round 34: loading tables now carry
full feature names, because `S` stood for both `Cell Area` and
`Local Nematic Order` and the table did not say which.

N-cluster generalization (was previously hardcoded to exactly 2 clusters
named 'mes cell'/'epi cell'): v1's original biology (Drosophila dorsal
closure) only ever had two cell populations, so `bio_clustering_k_mean`
hardcoded `n_clusters=2` and always relabeled the majority-count cluster to
`Cluster_Label=0`/`'epi cell'`, minority to `1`/`'mes cell'`. Many other
tissues naturally split into more than two morphologically distinct cell
populations, so `cluster_cells_kmeans` now accepts any `n_clusters`,
including `"auto"` (silhouette-score-selected via `_select_k_by_silhouette`).
Clusters are still ranked by population size, descending, but named
generically: `Cluster_Label=0`/`Cell_Type='cluster_1'` is always the largest
cluster, `Cluster_Label=N-1`/`'cluster_N'` the smallest. For the classic N=2
case this reproduces the exact same `Cluster_Label` INTEGERS as before
(majority still gets 0, minority still gets 1) -- only the `Cell_Type`
STRING changes (`'cluster_1'`/`'cluster_2'` instead of `'epi cell'`/
`'mes cell'`). See `v2/README.md` for the full design rationale, including
why `manual_correction_classification.compute_leading_edge`'s sentinel-sign
convention needed correcting rather than naively reusing "population
majority" as "large cell" (they are not the same thing -- v1's "large cell"
sentinel was tied to the fixed `Cluster_Label==1` value, which for the
classic AS/LE case happens to be the population MINORITY, not majority).
"""

import logging
import os
import re
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Feature families for the Classify step. **Additive, not exclusive**: Basic
# geometrics is the always-on base and any further family is added on top of it.
# Columns may come from `Cells_info.xlsx`, `Cells_shape.xlsx` or
# `Cells_tissue.xlsx`, which all share the `Cell Index` index.
#
# `default_off` names columns that are offered but left unchecked.
#
# Columns deliberately in NO family, each with a measured reason:
#   `Frame Neighbor Entropy` -- within-frame variance is exactly 0 (22 distinct
#       values for 2824 cells), so pooled it encodes only "which frame".
#   `Local Density` -- cells per pixel^2, not comparable across magnifications.
#   `Elongation` -- Spearman 0.9913 with `Ellipse Major/Minor Axis Ratio`.
#   `Sin2Theta`/`Cos2Theta` -- absolute orientation in the image frame, i.e. how
#       the specimen was mounted. They exist to feed `Local Nematic Order`,
#       which is itself rotation-invariant.
# All of them are still written to the xlsx files; they are just not offered as
# PCA inputs. (`P-Value` went further and was deleted from the schema outright;
# `Fitted Ellipse Angle` and `Circularity`, deleted with it, came back in Round 24
# as default-off `basic` columns -- see `data_input.CELLS_INFO_COLUMNS`.)
FEATURE_FAMILIES: dict[str, dict] = {
    "basic": {
        "label": "Basic geometrics",
        "base": True,
        "columns": [
            "Cell Area", "Cell Perimeter",
            "Fitted Ellipse Major Axis", "Fitted Ellipse Minor Axis",
            "Ellipse Major/Minor Axis Ratio", "Approximate Polygon Vertex Count",
            "Circumscribed Circle Radius", "Inscribed Circle Radius",
            "Maximum Horizontal Length", "Maximum Vertical Length",
            "Horizontal/Vertical Length Ratio",
            "Fitted Ellipse Angle", "Circularity",
        ],
        # Bbox extents are measured along the IMAGE axes, so they describe
        # orientation rather than shape: rotating one real fern cell from 0 to
        # 90 degrees swings the ratio from 2.458 to 0.407 while solidity and
        # shape index do not move. Meaningful only when the specimen is
        # deliberately aligned, as in the manuscript's AP/DV convention.
        # The ellipse angle is image-frame orientation too, and folded (see
        # `data_input.CELLS_INFO_COLUMNS`). Circularity is off because it is
        # `hull`'s Shape Index again (C = 4*pi/p^2, Spearman -1.00): ticking both
        # counts one shape quantity twice. Pruning does not guard it -- it is off
        # unless `prune_correlation` is passed, and Pearson rates the pair only
        # |r| = 0.966, so a threshold of 0.97 or above keeps both
        # (`docs/shape-index-vs-circularity.md`, section 5).
        "default_off": [
            "Maximum Horizontal Length", "Maximum Vertical Length",
            "Horizontal/Vertical Length Ratio",
            "Fitted Ellipse Angle", "Circularity",
        ],
        "note": (
            "H/V lengths measure orientation - tick them only if your specimen is axis-aligned. "
            "Ellipse Angle is orientation as well, and it is stored folded (70 and 110 degrees share one value), "
            "so it cannot be read back as a direction. Circularity is the same quantity as Shape Index in the "
            "Convex hull family - tick at most one of the two, or PCA counts that shape twice."
        ),
    },
    "hull": {
        "label": "Convex hull / compactness",
        "columns": ["Solidity", "Convexity", "Shape Index"],
        "note": "Solidity and Convexity are the only clean lobing measures - they do not move with elongation.",
    },
    "lobe": {
        "label": "Convolution / lobe",
        "columns": ["Lobe Count", "Max Concavity Depth", "Mean Neck Width", "EFD Irregularity"],
        "note": "EFD Irregularity also picks up elongation (0.119 for a smooth 8:1 ellipse vs 0.268 for a 6-lobed round cell).",
    },
    "tissue": {
        "label": "Local tissue texture",
        "columns": [
            "Neighbor Count", "Local Hexagonal Fraction", "Local Nematic Order",
            "Local Mean Shape Index", "Local Mean Solidity",
        ],
        "note": "Border cells have no neighbour count by construction and drop out of the run.",
    },
    "efd": {
        "label": "Elliptic Fourier coefficients",
        # Exclusive: an orthogonal basis is taken whole, and it must not be
        # standardized (see `standardize`).
        "exclusive": True,
        "standardize": "none",
        "columns": [],  # resolved at runtime from the extraction order
        "note": (
            "Used alone and without standardization - all coefficients together form one basis. "
            "Coefficients are normalized before PCA (start point, tracing direction, rotation, "
            "which end of the long axis is front, and size), so the result describes shape only. "
            "Front/back is decided per cell outline here, which is noisy for nearly symmetric "
            "cells; for tracked data decide it per track."
        ),
    },
}

# Two named column lists that are NOT families: `legacy_13` reproduces v1's
# exact PCA input (the manuscript's methodology) and is what the CLI uses;
# `tracking_only` names the four columns `cell_tracking.build_cost_matrix`
# actually reads. Everything else is a family in `FEATURE_FAMILIES` above.
# `legacy_13` is GONE, and stays gone although the two columns it lost on
# 2026-08-18 (`Fitted Ellipse Angle`, `Circularity`) are back since Round 24:
# it would not reproduce v1's numbers once geometry moved to the mask
# (perimeter ~-5%, circularity ~+10%), and a preset of that name would claim it
# does. `src/` stays frozen and is the reference for exact v1 reproduction.
FEATURE_PRESETS: dict[str, list[str]] = {
    "tracking_only": [
        "Maximum Horizontal Length", "Maximum Vertical Length",
        "Center X Coordinate", "Center Y Coordinate",
    ],
}


# Shorter labels for the Classify checkboxes ONLY. The dict key stays the real
# column name -- these never touch the DataFrame, the xlsx, or any lookup, so a
# missing entry just falls back to the full name, and the UI carries the full
# name as a tooltip.
#
# The principle is to drop words that carry no information in a per-cell table:
# "Cell" (everything here is a cell), "Fitted"/"Approximate" (how it was
# measured, not what it is), and "Circle" (implied by "Radius").
FEATURE_DISPLAY_NAMES = {
    "Cell Area": "Area",
    "Cell Perimeter": "Perimeter",
    "Fitted Ellipse Major Axis": "Ellipse Major Axis",
    "Fitted Ellipse Minor Axis": "Ellipse Minor Axis",
    "Ellipse Major/Minor Axis Ratio": "Ellipse Axis Ratio",
    "Fitted Ellipse Angle": "Ellipse Angle",
    "Approximate Polygon Vertex Count": "Vertex Count",
    "Circumscribed Circle Radius": "Circumscribed Radius",
    "Inscribed Circle Radius": "Inscribed Radius",
    "Maximum Horizontal Length": "Max Horizontal Length",
    "Maximum Vertical Length": "Max Vertical Length",
    "Horizontal/Vertical Length Ratio": "H/V Length Ratio",
}


def display_name(column: str) -> str:
    """The short label to show in the UI for `column`."""
    return FEATURE_DISPLAY_NAMES.get(column, column)


def family_columns(families, include_default_off: bool = False) -> list[str]:
    """Union of the columns of `families`, in family order, de-duplicated.

    `basic` is always included -- it is the base every other family adds to.
    """
    names = ["basic"] + [f for f in families if f != "basic"]
    out: list[str] = []
    for name in names:
        spec = FEATURE_FAMILIES.get(name)
        if spec is None:
            raise ValueError(f"unknown feature family {name!r}; expected one of {list(FEATURE_FAMILIES)}")
        skip = set() if include_default_off else set(spec.get("default_off", ()))
        for column in spec["columns"]:
            if column not in skip and column not in out:
                out.append(column)
    return out


def efd_preset(order: int = 10) -> list[str]:
    """The raw elliptic Fourier coefficient columns, as its own feature set.

    Not in `FEATURE_PRESETS` because it depends on the order used at
    extraction. Callers must pair it with `standardize_mode="none"`; see
    `standardize`.
    """
    return [f"efd_{h}_{c}" for h in range(order) for c in "abcd"]


def standardize(
    features: pd.DataFrame,
    mode: str = "global",
    frame_index: Optional[pd.Series] = None,
) -> np.ndarray:
    """Scale a feature matrix before PCA.

    - `"global"` -- `StandardScaler` over all rows. v1's behaviour, the default.
    - `"per_frame"` -- z-score within each frame, so PCA sees within-frame shape
      variation rather than frame-to-frame drift in illumination, magnification
      or developmental stage. Needed whenever frames are not directly
      comparable.

      This removes the *between-image* confound and nothing else -- a distinct
      problem from size dominance, which it does not touch. Measured on
      `cp_dataset_puzzle-like_cells` (12 unrelated images pooled) as the share
      of PC1's variance explained by image identity: with the legacy 13
      features, 0.601 globally versus **0.000** per frame; with the
      dimensionless preset, 0.180 versus 0.000. Meanwhile PC1's *size* share
      stays ~85% either way. Use `per_frame` for the image confound and a
      dimensionless preset for size dominance; they are not substitutes.
    - `"none"` -- pass through unscaled. **Required for the EFD path**: the
      collaborator's `figures/Epiquant_ShapeAnalysis_allMutants.ipynb` documents
      that standardizing all 40 coefficients gives each equal weight, which
      promotes the near-zero high harmonics -- mostly segmentation noise -- and
      flattens the spectrum until no component exceeds ~8%. Harmonic 1 already
      carries the natural scale, so the coefficients arrive comparable.
    """
    if mode == "none":
        return features.to_numpy(dtype=float)
    if mode == "global":
        return StandardScaler().fit_transform(features)
    if mode == "per_frame":
        if frame_index is None:
            frame_index = pd.Series(
                [str(i).split("_")[0] for i in features.index], index=features.index
            )
        grouped = features.groupby(frame_index, sort=False)
        # A frame with one cell, or a feature constant within a frame, has zero
        # spread; leaving it as 0 keeps it from becoming inf/NaN.
        centered = grouped.transform(lambda column: column - column.mean())
        spread = grouped.transform(lambda column: column.std(ddof=0))
        return (centered / spread.where(spread > 0, 1.0)).to_numpy(dtype=float)
    raise ValueError(f"unknown standardize mode {mode!r}; expected 'global', 'per_frame' or 'none'")


def prune_correlated(
    features: pd.DataFrame,
    threshold: float = 0.95,
    keep_order: Optional[list] = None,
) -> tuple[pd.DataFrame, dict]:
    """Drop features that duplicate one already kept, at `|Pearson r| >
    threshold`. Returns the pruned frame and `{dropped: the column it
    duplicated}` so a UI can explain why a ticked checkbox is missing from the
    loadings.

    PCA is not invariant to redundant inputs: a latent variable entered several
    times gets that many times the weight. Measured on the 13-feature default,
    8 of them are size proxies correlated with `Cell Area` at r = 0.88-0.94, and
    they take 85% of PC1's squared loading -- so PC1 is "how big is this cell"
    by construction, no matter what else is in the matrix.

    **A threshold of 0.95 barely helps that, and callers should know it.**
    Because the size proxies correlate at 0.88-0.94, they mostly survive it.
    Measured on `cp_dataset_puzzle-like_cells`, PC1's size share against the
    threshold:

        r > 0.99   12/13 kept   83.6%
        r > 0.95   10/13 kept   77.9%
        r > 0.90    8/13 kept   66.7%
        r > 0.85    6/13 kept   36.7%

    So pruning is a blunt instrument for size dominance -- the direct fix is to
    select dimensionless features. Measured, a ratio-only set puts PC1's size
    share at 0.0%, and so does the `lobe` family on its own; every column in
    `hull` and `lobe` is dimensionless by construction. Pruning is still worth
    having for genuine duplicates,
    such as `Circularity` and `Shape Index`, which are algebraically the same
    quantity (r = -0.96).

    Greedy in `keep_order` (column order by default), so the caller controls
    which member of a correlated group survives.
    """
    order = list(keep_order) if keep_order is not None else list(features.columns)
    correlation = features.corr().abs()
    kept: list = []
    dropped: dict = {}
    for column in order:
        if column not in features.columns:
            continue
        duplicate = next(
            (k for k in kept if pd.notna(correlation.loc[column, k]) and correlation.loc[column, k] > threshold),
            None,
        )
        if duplicate is None:
            kept.append(column)
        else:
            dropped[column] = duplicate
    return features.loc[:, kept], dropped


_EFD_COEFFICIENT = re.compile(r"^efd_(\d+)_([abcd])$")
_CELL_INDEX = re.compile(r"^Cell(\d+)_(\d+)$")


def pca_feature_label(column) -> str:
    """The full name a loading table shows for a feature column.

    Only the EFD coefficients need translating -- `efd_1_a` is the cosine term
    of the x projection of harmonic 2, which nobody reads off the column name.
    Everything else is already its full name, deliberately: the abbreviation
    table this replaced gave both `Cell Area` and `Local Nematic Order` the
    label `S`, so a loading table could not be read without knowing which
    family had been selected."""
    match = _EFD_COEFFICIENT.match(str(column))
    if match:
        return f"Harmonic {int(match.group(1)) + 1} {match.group(2)}"
    return str(column)


def cell_index_order(index) -> list:
    """`index` sorted by (frame, label) NUMERICALLY, not as strings -- `Cell10_2`
    belongs after `Cell9_2`, which a lexicographic sort gets backwards. Keys
    that are not `Cell{frame}_{label}` keep their relative order, after the
    ones that are."""
    def key(position_and_value):
        position, value = position_and_value
        match = _CELL_INDEX.match(str(value))
        if match:
            return (0, int(match.group(1)), int(match.group(2)), position)
        return (1, 0, 0, position)

    return [value for _, value in sorted(enumerate(index), key=key)]


def write_pca_components(
    output_path: str,
    components: np.ndarray,
    explained_variance_ratio: np.ndarray,
    feature_names: list,
    scores: pd.DataFrame,
    raw_features: pd.DataFrame,
    descriptor_correlation: Optional[pd.DataFrame] = None,
) -> str:
    """One workbook describing the fit, replacing the per-component
    `PCA_Component_{i}.xlsx` files and `Explained_Variance_Ratio_All_PC.xlsx`.

    Sheets:

    - `Loadings` -- features x PCs, for reading the whole basis at once.
    - `Explained variance` -- every component, not only the retained ones.
    - `Feature correlation` -- PCs x features, Pearson r between each PC score
      and each feature's RAW value. This is not the loading: a loading is the
      weight on a standardized feature within an orthogonal basis, while r is
      what Supp. Fig. 1A plots. Same definition as
      `figures/revision_2026-10/fig6_dyn_collective/scripts/static_pca.py:14`.
    - `PC{i}` -- one per retained component, sorted by contribution squared.
    - `Descriptor correlation` -- only when the caller has one (EFD; see
      `efd_descriptors`), because 40 Fourier coefficients have no readable
      loading table.

    `raw_features` must be the matrix PCA was fit on BEFORE standardizing, and
    aligned to `scores` by index; correlation is scale-free so standardizing
    would not change r, but the sentinel-masking and pruning done upstream
    would."""
    labels = [pca_feature_label(name) for name in feature_names]
    retained = [f"PC{i + 1}" for i in range(components.shape[0])]
    aligned = raw_features.loc[scores.index]

    loadings = pd.DataFrame(components.T, index=labels, columns=retained)
    loadings.index.name = "Feature"

    variance = pd.DataFrame({
        "Principal Component": [f"PC{i + 1}" for i in range(len(explained_variance_ratio))],
        "Explained Variance Ratio": explained_variance_ratio,
        "Cumulative": np.cumsum(explained_variance_ratio),
        "Retained": [i < components.shape[0] for i in range(len(explained_variance_ratio))],
    })

    correlation = pd.DataFrame(
        [[float(np.corrcoef(aligned[column].to_numpy(float), scores[pc].to_numpy(float))[0, 1])
          for column in feature_names] for pc in retained],
        index=retained, columns=labels,
    )
    correlation.index.name = "Principal Component"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        loadings.to_excel(writer, sheet_name="Loadings")
        variance.to_excel(writer, sheet_name="Explained variance", index=False)
        correlation.to_excel(writer, sheet_name="Feature correlation")
        if descriptor_correlation is not None:
            descriptor_correlation.to_excel(writer, sheet_name="Descriptor correlation")
        for i, pc in enumerate(retained):
            contribution = components[i]
            pd.DataFrame({
                "Feature": labels,
                "Contribution": contribution,
                "Contribution Squared": contribution ** 2,
            }).sort_values("Contribution Squared", ascending=False).to_excel(
                writer, sheet_name=pc, index=False)
    return output_path


def efd_descriptor_correlation(
    features: pd.DataFrame, scores: pd.DataFrame, group_by: str = "frame"
) -> pd.DataFrame:
    """How well readable shape descriptors explain each EFD principal
    component -- the `Descriptor correlation` sheet.

    An EFD loading table is 40 Fourier coefficients, which nobody can read. The
    same component expressed as "r = 0.93 with skewness along the axis" can be.
    The descriptor panel is the one `make_efd_analyses.descriptor_table` uses:
    `efd_descriptors.DESCRIPTOR_NAMES` (moments + geometry) plus elongation,
    EFD irregularity and the per-harmonic amplitudes, all computed from the
    order-10 RECONSTRUCTION rather than from the mask -- see the file header of
    `efd_descriptors.py` for why that is the only comparison that isolates the
    descriptor panel from the coefficient truncation.

    Reported per PC: Pearson r, Spearman rho, and the cross-validated R-squared
    of a linear (Ridge) fit from the whole panel. `group_by` names the CV
    grouping, which goes into the column header -- inside one project that is
    the frame, so a fold cannot learn a PC from cells that share a frame with
    its training set. The boosted-tree and quadratic fits
    `make_efd_analyses.fig_descriptor_panel` also runs are deliberately left
    out: minutes of compute for a number the workbook does not need.

    **Depends on the front/back convention.** `mu30_x` (skewness along the
    axis) and the bending moments change sign with it, so this sheet reads
    differently under a per-outline fit than under a per-track one."""
    import efd
    import efd_descriptors as ed
    from scipy.stats import pearsonr, spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline

    matrix = features.loc[scores.index].to_numpy(float)
    order = matrix.shape[1] // 4
    canonical = matrix.reshape(-1, order, 4)

    panel = ed.descriptors(canonical)
    # Elongation is the harmonic-1 axis ratio, which `canonicalize`/`drop_size`
    # leave alone (they rotate and rescale uniformly), so it can be read off
    # the canonical coefficients instead of the raw ones.
    panel["elongation"] = efd.ellipse_from_harmonic1(efd.phase_normalize(canonical))[0]
    panel["irregularity"] = efd.irregularity(matrix)
    amplitude = np.sqrt((canonical ** 2).sum(axis=2))
    for harmonic in range(2, order + 1):
        panel[f"amp_h{harmonic}"] = amplitude[:, harmonic - 1]

    pretty = {
        "mu30_x": "Skewness along axis", "mu30_y": "Skewness across axis",
        "mu21": "Bending (x2y)", "mu12": "Bending (xy2)",
        "mu40": "Kurtosis along axis", "mu04": "Kurtosis across axis",
        "mu22": "Cross moment (x2y2)", "circularity": "Circularity",
        "shape_index": "Shape index", "solidity": "Solidity", "convexity": "Convexity",
        "max_concavity_depth": "Max concavity depth", "lobe_count": "Lobe count",
        "bending_energy": "Bending energy", "curv_sign_changes": "Curvature sign changes",
        "elongation": "Elongation", "irregularity": "EFD irregularity",
        **{f"amp_h{n}": f"Amplitude h{n}" for n in range(2, order + 1)},
    }
    names = list(ed.DESCRIPTOR_NAMES) + ["elongation", "irregularity"] + \
        [f"amp_h{n}" for n in range(2, order + 1)]
    panel_matrix = np.column_stack([panel[name] for name in names])
    usable = np.isfinite(panel_matrix).all(axis=1)
    panel_matrix = panel_matrix[usable]

    groups = pd.Index(scores.index[usable]).str.extract(r"Cell(\d+)_", expand=False).to_numpy()
    folds = min(5, len(np.unique(groups)))

    rows = {}
    for pc in scores.columns:
        y = scores[pc].to_numpy(float)[usable]
        row = {}
        if folds >= 2:
            prediction = np.empty(len(y))
            for train, test in GroupKFold(n_splits=folds).split(panel_matrix, y, groups):
                model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                prediction[test] = model.fit(panel_matrix[train], y[train]).predict(panel_matrix[test])
            row[f"Linear CV R2 ({folds}-fold GroupKFold by {group_by})"] = r2_score(y, prediction)
        else:
            row[f"Linear CV R2 (not computed: only {folds} {group_by} group)"] = np.nan
        for position, name in enumerate(names):
            row[f"{pretty[name]} r"] = pearsonr(panel_matrix[:, position], y).statistic
            row[f"{pretty[name]} rho"] = spearmanr(panel_matrix[:, position], y).statistic
        rows[pc] = row

    table = pd.DataFrame(rows).T
    table.index.name = f"Principal Component ({int(usable.sum())} of {len(usable)} cells scored)"
    return table


def fit_pca(
    features, n_components: float = 0.85, sign_convention: bool = True
) -> tuple[PCA, np.ndarray]:
    """Fit PCA on an already-prepared matrix and return `(model, scores)`.

    A PCA component's sign is arbitrary -- `-c` explains exactly as much
    variance as `c` -- so two runs on the same data can report opposite
    loadings and mirrored scores. `sign_convention` fixes it by flipping each
    component so its largest-magnitude loading is positive. Same rule, same
    order of operations as `fit_pca` in
    `figures/efd_figures/make_efd_figures.py:118`, which is what the
    manuscript figures are drawn with; `_self_test` pins the two together
    element by element."""
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(features)
    if sign_convention:
        flip = np.sign(pca.components_[
            np.arange(len(pca.components_)), np.abs(pca.components_).argmax(axis=1)
        ])
        pca.components_ *= flip[:, None]
        scores = scores * flip[None, :]
    return pca, scores


def compute_pca(
    cells_info: pd.DataFrame,
    output_dir: Optional[str] = None,
    variance_threshold: float = 0.85,
    standardize_mode: str = "global",
    prune_correlation: Optional[float] = None,
    sign_convention: bool = False,
    with_efd_descriptors: bool = False,
) -> pd.DataFrame:
    """Standardize `cells_info` (already filtered to the desired feature
    columns, e.g. `data_input.ML_FEATURE_COLUMNS`) and run PCA. Returns a
    DataFrame indexed like `cells_info`, columns `PC1..PCk`, in `Cell Index`
    order -- frame then label, numerically.

    `variance_threshold` is passed straight to sklearn: a fraction keeps enough
    components to explain that much cumulative variance (v1 hardcoded 0.85), an
    integer keeps exactly that many. The distinction matters because 0.85 keeps
    only PC1-PC2 on `AS_LE_lateral` (cum2 = 87.6%) while the manuscript's
    Supp. Fig. 1 panels are drawn on PC1-PC3.

    **No `PC_score` column** (Round 34, 1-4): it was a variance-weighted sum of
    scores whose signs are arbitrary, used only to sort these rows. Removing it
    changes the row order to `Cell Index` order; `cluster_cells_kmeans` used to
    pick its PC columns as "everything but the last one", which would have
    silently dropped a real PC, and now matches `PC\d+` by name.

    If `output_dir` is given, writes `Cells_info_PCA.xlsx` (unchanged format:
    first column `Cell Index`) and one `PCA_Components.xlsx` workbook. No PNGs:
    every chart this used to export is better drawn by the plotting step, and
    four of them were written on every run whether or not anyone looked.
    `with_efd_descriptors` adds the `Descriptor correlation` sheet, and is only
    meaningful when `cells_info` holds EFD features (see
    `efd_descriptor_correlation`).

    `standardize_mode`, `prune_correlation` and `sign_convention` all default to
    reproducing v1 exactly; see `standardize`, `prune_correlated` and `fit_pca`.
    """
    if prune_correlation is not None:
        cells_info, pruned = prune_correlated(cells_info, threshold=prune_correlation)
        if pruned:
            logger.info(
                "Correlation pruning at |r| > %.2f dropped %d feature(s): %s",
                prune_correlation, len(pruned),
                ", ".join(f"{k} (duplicates {v})" for k, v in pruned.items()),
            )

    cell_info_scaled = standardize(cells_info, mode=standardize_mode)

    explained_variance_ratio = PCA().fit(cell_info_scaled).explained_variance_ratio_
    pca, cell_info_pca = fit_pca(cell_info_scaled, variance_threshold, sign_convention)

    pca_columns = ["PC" + str(i + 1) for i in range(cell_info_pca.shape[1])]
    cell_scores = pd.DataFrame(cell_info_pca, index=cells_info.index, columns=pca_columns)
    cell_scores = cell_scores.loc[cell_index_order(cell_scores.index)]

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        cell_scores.to_excel(os.path.join(output_dir, "Cells_info_PCA.xlsx"), sheet_name="Cell1")
        write_pca_components(
            os.path.join(output_dir, "PCA_Components.xlsx"),
            components=pca.components_,
            explained_variance_ratio=explained_variance_ratio,
            feature_names=list(cells_info.columns),
            scores=cell_scores,
            raw_features=cells_info,
            descriptor_correlation=(
                efd_descriptor_correlation(cells_info, cell_scores)
                if with_efd_descriptors else None),
        )

    return cell_scores


def plot_shape_modes(
    coefficients: pd.DataFrame,
    output_dir: str,
    order: int = 10,
    n_components: int = 6,
    percentiles: tuple = (10, 30, 50, 70, 90),
    filename: str = "PCA_Shape_Modes.png",
) -> pd.DataFrame:
    """Draw the cell outline at several positions along each principal
    component of elliptic Fourier shape space, writing `filename` under
    `output_dir` and returning the per-component explained variance.

    This is what makes an EFD PCA interpretable: a component is a direction in
    coefficient space, so any point along it can be turned back into an actual
    contour. Instead of "PC2 = 14% of variance" the reader sees the shape
    change PC2 encodes. Ported from `plot_modes`/`contour_of` in
    `figures/Epiquant_ShapeAnalysis_allMutants.ipynb`.

    `coefficients` holds the raw `efd_{h}_{c}` columns from
    `shape_features.extract_shape_features(with_coefficients=True)`. They are
    phase-normalized, rotation-aligned and size-normalized here, then PCA'd
    **without standardizing** -- see `standardize` for why.
    """
    import efd  # local: only this function needs it

    columns = efd_preset(order)
    missing = [c for c in columns if c not in coefficients.columns]
    if missing:
        raise ValueError(f"missing {len(missing)} EFD coefficient column(s), e.g. {missing[:3]}")

    raw = coefficients.loc[:, columns].dropna()
    features = efd.efd_features(raw.to_numpy(dtype=float).reshape(-1, order, 4))

    n_components = min(n_components, features.shape[1], len(features))
    pca = PCA(n_components=n_components).fit(features)
    flip = np.sign(pca.components_[np.arange(n_components), np.abs(pca.components_).argmax(axis=1)])
    pca.components_ *= flip[:, None]
    scores = pca.transform(features) * flip[None, :]
    mean_shape = features.mean(axis=0)

    figure, axes = plt.subplots(
        n_components, len(percentiles), figsize=(2.0 * len(percentiles), 2.1 * n_components), dpi=150
    )
    axes = np.atleast_2d(axes)
    for row in range(n_components):
        for column, percentile in enumerate(percentiles):
            offset = np.percentile(scores[:, row], percentile)
            vector = mean_shape + offset * pca.components_[row]
            # Rescale to unit harmonic-1 norm so panels are comparable; the
            # components carry shape, not size.
            norm = np.sqrt((vector[:4] ** 2).sum())
            outline = efd.reconstruct_contour((vector / norm if norm > 0 else vector).reshape(order, 4))
            outline = outline - outline.mean(axis=0)
            axis = axes[row, column]
            axis.fill(outline[:, 0], outline[:, 1], alpha=0.35, color="steelblue")
            axis.plot(outline[:, 0], outline[:, 1], color="steelblue", lw=1.2)
            reference = efd.reconstruct_contour(
                (mean_shape / np.sqrt((mean_shape[:4] ** 2).sum())).reshape(order, 4)
            )
            reference = reference - reference.mean(axis=0)
            axis.plot(reference[:, 0], reference[:, 1], "--", color="grey", lw=0.8)
            axis.set_aspect("equal")
            axis.axis("off")
            if row == 0:
                axis.set_title(f"{percentile}th", fontsize=10)
            if column == 0:
                axis.text(
                    -0.25, 0.5, f"PC{row + 1}\n{pca.explained_variance_ratio_[row] * 100:.1f}%",
                    transform=axis.transAxes, ha="center", va="center", fontsize=10, fontweight="bold",
                )
    figure.suptitle("Cell shape along each principal component (grey dashed = mean shape)", fontsize=12)
    figure.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    figure.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close(figure)

    return pd.DataFrame({
        "Principal Component": [f"PC{i + 1}" for i in range(n_components)],
        "Explained Variance Ratio": pca.explained_variance_ratio_,
    })


# Seeded by default. `KMeans(n_init=10)` with ANY fixed seed converges to the
# same optimum, but with `random_state=None` it occasionally lands in a worse
# local one: measured on `pupae_E-cad` (n=1213), 20 different seeds all gave a
# (1186, 27) split while unseeded runs wandered between (1183, 30), (1184, 29)
# and (1186, 27). So seeding removes real flakiness rather than hiding it --
# and `_select_k_by_silhouette` chose k=2 in 20/20 seeds, so it is not masking
# an unstable k either. Pass `random_state=None` explicitly for the old
# behaviour.
DEFAULT_RANDOM_STATE = 0


def _select_k_by_silhouette(
    data: np.ndarray,
    k_range: tuple[int, int] = (2, 9),
    random_state: Optional[int] = DEFAULT_RANDOM_STATE,
) -> tuple[int, dict[int, float]]:
    """Try every k in `[k_range[0], min(k_range[1], n_samples - 1)]`, fit
    `KMeans(n_clusters=k, n_init=10, random_state=random_state)` on `data`
    for each, and score with `sklearn.metrics.silhouette_score`. Returns
    `(best_k, {k: score, ...})` -- the full per-k score dict is returned (not
    just the winner) so callers/CLI tooling can inspect the whole search
    rather than trusting a single opaque number.

    Raises `ValueError` if the clamped range yields fewer than 2 candidate k
    values (i.e. `data` has fewer than 3 samples -- silhouette scoring is
    undefined below that).
    """
    n_samples = data.shape[0]
    max_k = min(k_range[1], n_samples - 1)
    candidate_ks = list(range(k_range[0], max_k + 1))
    if len(candidate_ks) < 1:
        raise ValueError(
            f"Not enough samples ({n_samples}) to try any k in range {k_range} "
            f"(clamped upper bound: {max_k})."
        )

    scores: dict[int, float] = {}
    for k in candidate_ks:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = kmeans.fit_predict(data)
        scores[k] = silhouette_score(data, labels)

    best_k = max(scores, key=scores.get)
    return best_k, scores


def cluster_cells_kmeans(
    cells_info_pca: pd.DataFrame,
    output_dir: Optional[str] = None,
    n_clusters: Union[int, str] = 2,
    k_range: tuple[int, int] = (2, 9),
    random_state: Optional[int] = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """K-means cluster cells on their PC scores -- every `PC{n}` column of
    `cells_info_pca`, matched by NAME.

    It used to be `columns[:-1]`, i.e. "everything but the last column",
    because the last column was the `PC_score` sort key. With `PC_score` gone
    (Round 34, 1-4) that rule would have dropped the last real component
    without a word, so the selection is by name and the result is asserted to
    be exactly `PC1..PCk` with nothing else in the frame.

    `n_clusters`: an integer cluster count (any N >= 2, not just 2), or the
    literal string `"auto"` to pick k via `_select_k_by_silhouette` over
    `k_range` (default `(2, 9)`, clamped to the sample count).

    Clusters are ranked by population size, descending, and labeled
    generically: `Cluster_Label=0`/`Cell_Type='cluster_1'` is always the
    largest cluster, ..., `Cluster_Label=N-1`/`'cluster_N'` the smallest.
    For the classic N=2 case this produces the exact same `Cluster_Label`
    integers v1 did (majority still 0, minority still 1) -- only the
    `Cell_Type` string changes from `'epi cell'`/`'mes cell'` to
    `'cluster_1'`/`'cluster_2'`.

    Returns the results DataFrame (`Cell_Index, Cluster_Label, Cell_Type`)
    directly — v1's `bio_clustering_k_mean` additionally returned
    `(big_cells_predict, small_cells_predict)` index lists, but its only
    caller never used them (src/gui_main.py:1112), so that dead output is
    dropped here rather than ported.

    If `output_dir` is given, writes `cells_clustering_results.xlsx`. The
    `cells_PCA_2D.png` scatter it also used to write is gone with the other
    default PNG exports (Round 34, 1-1); the Morphospace view draws the same
    thing, with the decision boundary, on demand.

    `random_state` is newly exposed (v1 had none, i.e. `None`/non-deterministic)
    so callers can pass a fixed seed for reproducible runs/tests.
    """
    fac_index = cells_info_pca.index
    components = [col for col in cells_info_pca.columns if re.fullmatch(r"PC\d+", str(col))]
    expected = [f"PC{i + 1}" for i in range(len(components))]
    if components != expected or len(components) != len(cells_info_pca.columns):
        raise ValueError(
            f"expected the PCA score table to hold exactly {expected} and nothing else, "
            f"got {list(cells_info_pca.columns)}"
        )
    data = cells_info_pca[components].values

    if n_clusters == "auto":
        n_clusters, _ = _select_k_by_silhouette(data, k_range, random_state)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    label = kmeans.fit_predict(data)

    n_clusters_actual = label.max() + 1
    counts = np.bincount(label, minlength=n_clusters_actual)
    size_rank = np.argsort(-counts)  # raw label indices, largest population first
    raw_to_final = np.empty(len(size_rank), dtype=int)
    raw_to_final[size_rank] = np.arange(len(size_rank))
    final_label = raw_to_final[label]

    results = pd.DataFrame({
        "Cell_Index": fac_index,
        "Cluster_Label": final_label,
        "Cell_Type": [f"cluster_{lbl + 1}" for lbl in final_label],
    })

    if output_dir is not None:
        results.to_excel(os.path.join(output_dir, "cells_clustering_results.xlsx"), index=False)

    return results


def _self_test() -> None:
    """Checks for the feature-space options that need no dataset on disk."""
    rng = np.random.default_rng(0)
    n = 400
    index = pd.Index([f"Cell{1 + i // 100}_{i}" for i in range(n)], name="Cell Index")

    # A latent "size" entered five times, plus one independent shape feature.
    size = rng.normal(0, 1, n)
    frame = np.repeat([0.0, 6.0, 12.0, 18.0], n // 4)  # a large per-frame offset
    features = pd.DataFrame({
        "Cell Area": size + frame + rng.normal(0, 0.05, n),
        "Cell Perimeter": size + frame + rng.normal(0, 0.05, n),
        "Maximum Horizontal Length": size + frame + rng.normal(0, 0.05, n),
        "Maximum Vertical Length": size + frame + rng.normal(0, 0.05, n),
        "Circumscribed Circle Radius": size + frame + rng.normal(0, 0.05, n),
        "Circularity": rng.normal(0, 1, n),
    }, index=index)

    # 1. Pruning collapses the duplicated latent variable to one column.
    pruned, dropped = prune_correlated(features, threshold=0.95)
    assert len(pruned.columns) == 2, f"expected 2 surviving features, got {list(pruned.columns)}"
    assert set(dropped) == {
        "Cell Perimeter", "Maximum Horizontal Length",
        "Maximum Vertical Length", "Circumscribed Circle Radius",
    }, f"unexpected drops: {dropped}"
    assert all(v == "Cell Area" for v in dropped.values()), "each drop should name what it duplicated"
    print(f"  prune: 6 features -> {len(pruned.columns)}; "
          + ", ".join(f"{k} ~ {v}" for k, v in list(dropped.items())[:2]) + ", ...")

    # 2. Standardization modes.
    assert np.allclose(standardize(features, "none"), features.to_numpy()), "'none' must pass through"
    globally = standardize(features, "global")
    assert abs(globally.mean()) < 1e-12 and abs(globally.std() - 1) < 1e-6, "'global' must z-score"
    per_frame = standardize(features, "per_frame")
    frames = pd.Series([i.split("_")[0] for i in index], index=index)
    grouped = pd.DataFrame(per_frame, index=index).groupby(frames).mean().abs().to_numpy().max()
    assert grouped < 1e-10, f"'per_frame' must centre every frame, worst residual {grouped:.2e}"
    # The per-frame offset dominates PC1 globally, and must be gone afterwards.
    pc1_global = PCA(n_components=1).fit_transform(globally)[:, 0]
    pc1_frame = PCA(n_components=1).fit_transform(per_frame)[:, 0]

    def eta_squared(scores):
        table = pd.DataFrame({"s": scores, "g": frames.to_numpy()})
        grand = table.s.mean()
        between = sum(len(g) * (g.s.mean() - grand) ** 2 for _, g in table.groupby("g"))
        return between / ((table.s - grand) ** 2).sum()

    assert eta_squared(pc1_global) > 0.8, "the synthetic frame offset should dominate PC1 globally"
    assert eta_squared(pc1_frame) < 0.01, (
        f"'per_frame' should remove the frame confound, eta^2 still {eta_squared(pc1_frame):.3f}"
    )
    print(f"  standardize: frame share of PC1 {eta_squared(pc1_global):.3f} global "
          f"-> {eta_squared(pc1_frame):.3f} per-frame")

    # 3. Sign convention makes components reproducible across runs. Without it
    #    PCA signs are arbitrary, so loading tables need not match.
    first = compute_pca(features, sign_convention=True)
    assert np.isfinite(first.to_numpy()).all(), "PCA produced non-finite scores"
    again = compute_pca(features, sign_convention=True)
    pd.testing.assert_frame_equal(first, again)
    print("  sign convention: repeated runs give identical components")

    # 3b. (Round 34, 0-4) The GUI and `/figures` must be the SAME PCA, not two
    #     implementations that agree in spirit. `make_efd_figures.py` cannot be
    #     imported -- it loads pickles at module level -- so its `fit_pca`
    #     (`figures/efd_figures/make_efd_figures.py:118-122`) is copied here
    #     verbatim as the reference, and the copy is what this pins against.
    def reference_fit_pca(F, n=10):
        p = PCA(n_components=n).fit(F)
        flip = np.sign(p.components_[np.arange(n), np.abs(p.components_).argmax(axis=1)])
        p.components_ *= flip[:, None]
        return p, p.transform(F)

    reference_model, reference_scores = reference_fit_pca(features.to_numpy(float), n=6)
    ours, our_scores = fit_pca(features.to_numpy(float), n_components=6, sign_convention=True)
    assert np.allclose(ours.components_, reference_model.components_, rtol=0, atol=1e-12), (
        f"components differ by up to {np.abs(ours.components_ - reference_model.components_).max():.2e}"
    )
    assert np.allclose(our_scores, reference_scores, rtol=0, atol=1e-9), (
        f"scores differ by up to {np.abs(our_scores - reference_scores).max():.2e}"
    )
    print(f"  figures parity: components identical to make_efd_figures.fit_pca "
          f"(max |diff| {np.abs(ours.components_ - reference_model.components_).max():.1e})")

    # 3c. (Round 34, 0-2) The same for the EFD feature matrix: with `track_ids`
    #     it must be exactly `make_efd_figures.py:129-131`'s `X`. Without them
    #     the front/back decision is per outline, which is a DIFFERENT matrix --
    #     assert that too, so the distinction cannot quietly stop mattering.
    import efd as _efd

    rng_e = np.random.default_rng(3)
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    contours, tracks, frames_e = [], [], []
    for cell in range(6):
        for t in range(4):
            radius = 1 + 0.3 * np.cos(theta) + 0.12 * np.cos(2 * theta + 0.4 * t + cell)
            outline = np.c_[radius * np.cos(theta), radius * np.sin(theta)]
            contours.append(_efd.elliptic_fourier_descriptors(outline, order=10))
            tracks.append(f"cell{cell}")
            frames_e.append(t)
    C = np.asarray(contours)
    reference_X = _efd.drop_size(
        _efd.canonicalize(C, flip=_efd.front_back_by_track(C, tracks, frames_e))
    ).reshape(len(C), -1)
    ours_X = _efd.efd_features(C, track_ids=tracks, frames=frames_e)
    assert np.allclose(ours_X, reference_X, rtol=0, atol=0), "efd_features must BE the figures' chain"
    per_outline_X = _efd.efd_features(C)
    assert not np.allclose(per_outline_X, reference_X), (
        "fixture has no front/back ambiguity -- it cannot tell the two conventions apart"
    )
    flipped = (np.abs(per_outline_X - reference_X).max(axis=1) > 1e-9).sum()
    print(f"  figures parity: EFD features identical with track_ids; "
          f"{flipped}/{len(C)} outlines differ under the per-outline rule")

    # 3d. (Round 34, Part 1) What a classification run leaves on disk: one
    #     workbook, no PNGs, no `PC_score`, rows in Cell Index order.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        written = compute_pca(features, output_dir=tmp, sign_convention=True)
        assert "PC_score" not in written.columns, written.columns
        assert list(written.columns) == [f"PC{i + 1}" for i in range(written.shape[1])], written.columns
        files = sorted(os.listdir(tmp))
        assert not [f for f in files if f.endswith(".png")], f"PNG exports are back: {files}"
        assert files == ["Cells_info_PCA.xlsx", "PCA_Components.xlsx"], files
        book = pd.ExcelFile(os.path.join(tmp, "PCA_Components.xlsx"))
        expected_sheets = ["Loadings", "Explained variance", "Feature correlation"] + \
            [f"PC{i + 1}" for i in range(written.shape[1])]
        assert book.sheet_names == expected_sheets, book.sheet_names
        loadings = book.parse("Loadings", index_col=0)
        assert list(loadings.index) == list(features.columns), list(loadings.index)
        per_pc = book.parse("PC1")
        assert list(per_pc.columns) == ["Feature", "Contribution", "Contribution Squared"]
        assert per_pc["Contribution Squared"].is_monotonic_decreasing, "PC sheets sort by contribution"
        assert set(per_pc["Feature"]) == set(loadings.index), "the two sheets must name the same features"
        correlation = book.parse("Feature correlation", index_col=0)
        assert list(correlation.index) == list(written.columns), list(correlation.index)
        # r is between the PC score and the RAW feature, i.e. what Supp. Fig. 1A plots.
        manual = np.corrcoef(features.loc[written.index, "Cell Area"], written["PC1"])[0, 1]
        assert abs(correlation.loc["PC1", "Cell Area"] - manual) < 1e-12, correlation.loc["PC1", "Cell Area"]
        variance = book.parse("Explained variance")
        assert len(variance) == features.shape[1], "every component, not only the retained ones"
        assert int(variance["Retained"].sum()) == written.shape[1], variance

    # Rows come out in Cell Index order, numerically -- Cell10 after Cell9.
    shuffled = features.sample(frac=1.0, random_state=1)
    shuffled.index = pd.Index(
        [f"Cell{1 + i % 11}_{1 + i // 11}" for i in range(len(shuffled))], name="Cell Index")
    ordered = compute_pca(shuffled, sign_convention=True)
    keys = [(int(i[4:].split("_")[0]), int(i.split("_")[1])) for i in ordered.index]
    assert keys == sorted(keys), f"not in (frame, label) order: {keys[:6]}"
    assert list(ordered.index) != sorted(ordered.index), (
        "fixture has no Cell10-vs-Cell9 case, so it cannot tell a numeric sort from a string one"
    )

    # And k-means now picks its PC columns by name. The old `columns[:-1]`
    # rule, applied to a PC_score-free table, drops a real component.
    clusters = cluster_cells_kmeans(ordered, n_clusters=2, random_state=0)
    assert list(clusters.columns) == ["Cell_Index", "Cluster_Label", "Cell_Type"], clusters.columns
    assert list(clusters["Cell_Index"]) == list(ordered.index), "k-means must keep the row order"
    try:
        cluster_cells_kmeans(ordered.assign(PC_score=1.0), n_clusters=2)
        raise AssertionError("a stray column must be refused, not silently used or dropped")
    except ValueError as error:
        assert "PC_score" in str(error), error
    print(f"  Part 1: one PCA_Components.xlsx ({len(expected_sheets)} sheets), no PNGs, no PC_score; "
          f"rows in Cell Index order; k-means used {ordered.shape[1]} PC(s) by name")

    # 3e. (Round 34, 1-3) The EFD descriptor sheet. Re-uses 3c's contours.
    efd_frame = pd.DataFrame(
        ours_X, index=pd.Index([f"Cell{f + 1}_{t[4:]}" for t, f in zip(tracks, frames_e)],
                               name="Cell Index"),
        columns=_efd.efd_feature_names(10))
    with tempfile.TemporaryDirectory() as tmp:
        efd_scores = compute_pca(efd_frame, output_dir=tmp, standardize_mode="none",
                                 variance_threshold=3, sign_convention=True,
                                 with_efd_descriptors=True)
        sheets = pd.ExcelFile(os.path.join(tmp, "PCA_Components.xlsx")).sheet_names
        assert "Descriptor correlation" in sheets, sheets
        sheet = pd.read_excel(os.path.join(tmp, "PCA_Components.xlsx"),
                              sheet_name="Descriptor correlation", index_col=0)
        assert list(sheet.index) == list(efd_scores.columns), list(sheet.index)
        assert sheet.columns[0].startswith("Linear CV R2"), sheet.columns[0]
        assert "Skewness along axis r" in sheet.columns and "Amplitude h10 rho" in sheet.columns
        # 40 coefficients -> 15 descriptors + elongation + irregularity + 9 amplitudes,
        # each with an r and a rho, plus the one R2 column.
        assert sheet.shape[1] == 1 + 2 * 26, sheet.shape
        assert "Loadings" in sheets and "Harmonic 1 a" in \
            list(pd.read_excel(os.path.join(tmp, "PCA_Components.xlsx"),
                               sheet_name="Loadings", index_col=0).index)
    # Elongation is read off the canonical coefficients; it must match the
    # value the figures take from the raw ones.
    canonical = ours_X.reshape(-1, 10, 4)
    from_canonical = _efd.ellipse_from_harmonic1(_efd.phase_normalize(canonical))[0]
    from_raw = _efd.ellipse_from_harmonic1(_efd.phase_normalize(C))[0]
    assert np.allclose(from_canonical, from_raw, atol=1e-9), (
        f"elongation shortcut drifts by {np.abs(from_canonical - from_raw).max():.2e}"
    )
    print("  1-3: Descriptor correlation sheet written (26 descriptors x r/rho + CV R2); "
          "elongation matches the raw-coefficient route")

    # 4. Families are additive, well-formed, and exclude the retired columns.
    RETIRED = {
        "P-Value", "Frame Neighbor Entropy",
        "Local Density", "Elongation", "Sin2Theta", "Cos2Theta",
    }
    for name, spec in FEATURE_FAMILIES.items():
        columns = spec["columns"]
        assert len(columns) == len(set(columns)), f"family {name} repeats a column"
        leaked = RETIRED & set(columns)
        assert not leaked, f"family {name} still offers retired column(s) {sorted(leaked)}"
        assert set(spec.get("default_off", ())) <= set(columns), f"family {name} default_off names a column it lacks"
    assert FEATURE_FAMILIES["basic"].get("base"), "basic must be the always-on base"
    assert FEATURE_FAMILIES["efd"].get("exclusive"), "efd must stay exclusive"
    assert FEATURE_FAMILIES["efd"]["standardize"] == "none", "efd must never be standardized"

    # basic is included whether or not it is asked for, and default_off columns
    # are withheld unless requested.
    base_only = family_columns([])
    assert base_only == family_columns(["basic"]), "basic must be implicit"
    assert "Horizontal/Vertical Length Ratio" not in base_only, "AP/DV columns must default off"
    assert "Horizontal/Vertical Length Ratio" in family_columns([], include_default_off=True)
    # Round 24: both restored columns are offered but withheld by default, so
    # the default Classify input is exactly what it was before they came back.
    for restored in ("Fitted Ellipse Angle", "Circularity"):
        assert restored not in base_only, f"{restored} must default off"
        assert restored in family_columns([], include_default_off=True), f"{restored} must be offered"
        assert restored in data_input.CELLS_INFO_COLUMNS, f"{restored} must be in the schema"
    # The note is the only place the user can read WHY a box is unticked.
    basic_note = FEATURE_FAMILIES["basic"]["note"]
    assert "Ellipse Angle" in basic_note and "Circularity" in basic_note, basic_note
    combined = family_columns(["hull", "lobe"])
    assert combined[: len(base_only)] == base_only, "families must stack on top of basic, in order"
    assert len(combined) == len(set(combined)), "stacking must not duplicate columns"
    try:
        family_columns(["nonsense"])
        raise AssertionError("an unknown family must be rejected")
    except ValueError:
        pass

    # The retired columns must not reappear anywhere.
    assert "legacy_13" not in FEATURE_PRESETS, "legacy_13 was retired with the columns it named"
    for retired in ("P-Value",):
        assert retired not in data_input.CELLS_INFO_COLUMNS, f"{retired} is back in the schema"
        assert retired not in data_input.ML_FEATURE_COLUMNS, f"{retired} is back in ML_FEATURE_COLUMNS"
    assert set(FEATURE_FAMILIES["basic"]["columns"]) == set(data_input.ML_FEATURE_COLUMNS), (
        "the basic family and ML_FEATURE_COLUMNS must stay the same set"
    )

    # Display names are cosmetic. Every key must be a real column, and no short
    # label may collide with a real column name -- if one did, a lookup could
    # silently resolve the label instead of the column it stands for.
    real_columns = set(data_input.CELLS_INFO_COLUMNS) | {
        c for spec in FEATURE_FAMILIES.values() for c in spec["columns"]
    }
    for column, shown in FEATURE_DISPLAY_NAMES.items():
        assert column in real_columns, f"display name for {column!r}, which is not a column"
        assert shown not in real_columns, f"short label {shown!r} collides with a real column name"
    assert display_name("Solidity") == "Solidity", "an unmapped column must fall through unchanged"
    print(f"  families: {', '.join(FEATURE_FAMILIES)}; basic={len(base_only)} cols, "
          f"+hull+lobe={len(combined)} cols (+ efd_preset({len(efd_preset(10))}))")

    print("\npca_kmeans.py self-test: OK")


if __name__ == "__main__":
    import argparse

    import data_input

    parser = argparse.ArgumentParser(description="Standalone debug entry for v2.pca_kmeans")
    parser.add_argument("--self-test", action="store_true", help="run feature-space checks")
    parser.add_argument("--cells-info", help="Path to a Cells_info.xlsx (sheet 'Cell')")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--n-clusters", default="2",
        help='Integer cluster count, or the literal string "auto" to pick k via silhouette score.',
    )
    parser.add_argument(
        "--k-range", type=int, nargs=2, default=[2, 9], metavar=("MIN_K", "MAX_K"),
        help="Only used when --n-clusters auto. Default: 2 9.",
    )
    parser.add_argument("--random-state", type=int, default=None)
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        raise SystemExit(0)
    if not args.cells_info:
        parser.error("--cells-info is required unless --self-test is given")

    from data_input import ML_FEATURE_COLUMNS

    cells_info = pd.read_excel(args.cells_info, sheet_name="Cell", index_col="Cell Index")
    filtered = cells_info.loc[:, ML_FEATURE_COLUMNS]

    pca_df = compute_pca(filtered, args.output_dir)
    print("Explained variance (retained components):")
    print(pca_df.columns.tolist())

    n_clusters_arg = "auto" if args.n_clusters.strip().lower() == "auto" else int(args.n_clusters)
    k_range_arg = tuple(args.k_range)

    if n_clusters_arg == "auto":
        best_k, scores = _select_k_by_silhouette(pca_df.values, k_range_arg, args.random_state)
        print(f"\nSilhouette scores over k_range={k_range_arg}:")
        for k, score in scores.items():
            marker = "  <-- selected" if k == best_k else ""
            print(f"  k={k}: {score:.4f}{marker}")

    clustering_df = cluster_cells_kmeans(
        pca_df, args.output_dir, n_clusters=n_clusters_arg, k_range=k_range_arg, random_state=args.random_state
    )
    print("\nCluster sizes:")
    print(clustering_df["Cell_Type"].value_counts())
