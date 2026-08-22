"""Module 2: PCA + K-Means cell classification.

Ported from `gui_main.MainWindow.bio_compute_cell_pca` /
`bio_clustering_k_mean` (src/gui_main.py:1391-1580). Both were already
`self`-free/pure in v1 aside from their file-writing side effects, which are
now gated behind an optional `output_dir` (pass `None` to skip all writes).

Note: this module does NOT produce a `leading_edge` column — that is computed
downstream, from segmentation contours rather than PCA/K-means output, in
`manual_correction_classification.compute_leading_edge`.

Bug Fix #2 (see v2/README.md changelog): v1's feature-abbreviation dict had
`'Cell Left Boundary'` as a dict key twice (the second one was meant to be
`'Cell Right Boundary'`), so Python's duplicate-key overwrite silently gave
Right Boundary no abbreviation at all (its full column name leaked onto the
PCA contribution chart's x-axis instead). Fixed here.

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
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Bug Fix #2: 'Cell Right Boundary' now has its own entry instead of being
# silently shadowed by a duplicate 'Cell Left Boundary' key.
PCA_FEATURE_ABBREVIATIONS = {
    "Cell Area": "S",
    "Maximum Horizontal Length": "AP",
    "Maximum Vertical Length": "DV",
    "Horizontal/Vertical Length Ratio": "AP/DV",
    "Cell Perimeter": "P",
    "Approximate Polygon Vertex Count": "N",
    "Fitted Ellipse Minor Axis": "Sa",
    "Fitted Ellipse Major Axis": "La",
    "Ellipse Major/Minor Axis Ratio": "La/Sa",
    "Circumscribed Circle Radius": "R_c",
    "Inscribed Circle Radius": "r_i",
    "Center X Coordinate": "Cx",
    "Center Y Coordinate": "Cy",
    "Cell Left Boundary": "Lb",
    "Cell Right Boundary": "Rb",
    # shape_features.py
    "Solidity": "Sol",
    "Convexity": "Cvx",
    "Shape Index": "p",
    "Lobe Count": "N_lobe",
    "Max Concavity Depth": "d_max",
    "Mean Neck Width": "w_neck",
    "Elongation": "e",
    "Sin2Theta": "sin2θ",
    "Cos2Theta": "cos2θ",
    "EFD Irregularity": "irr",
    # tissue_features.py
    "Neighbor Count": "n_nb",
    "Local Hexagonal Fraction": "f_hex",
    "Local Nematic Order": "S",
    "Local Mean Shape Index": "<p>",
    "Local Mean Solidity": "<Sol>",
    "Local Density": "ρ",
    "Frame Neighbor Entropy": "H",
}

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
# PCA inputs. (`Fitted Ellipse Angle`, `Circularity` and `P-Value` went further
# and were deleted from the schema outright -- see `data_input.CELLS_INFO_COLUMNS`.)
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
        ],
        # Bbox extents are measured along the IMAGE axes, so they describe
        # orientation rather than shape: rotating one real fern cell from 0 to
        # 90 degrees swings the ratio from 2.458 to 0.407 while solidity and
        # shape index do not move. Meaningful only when the specimen is
        # deliberately aligned, as in the manuscript's AP/DV convention.
        "default_off": [
            "Maximum Horizontal Length", "Maximum Vertical Length",
            "Horizontal/Vertical Length Ratio",
        ],
        "note": "H/V lengths measure orientation, not shape - tick them only if your specimen is axis-aligned.",
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
        "note": "Used alone and without standardization - all coefficients together form one basis.",
    },
}

# Two named column lists that are NOT families: `legacy_13` reproduces v1's
# exact PCA input (the manuscript's methodology) and is what the CLI uses;
# `tracking_only` names the four columns `cell_tracking.build_cost_matrix`
# actually reads. Everything else is a family in `FEATURE_FAMILIES` above.
# `legacy_13` is GONE. It named `Fitted Ellipse Angle` and `Circularity`, both
# removed from the schema on 2026-08-18, so it can no longer be assembled -- and
# it would not have reproduced v1's numbers anyway once geometry moved to the
# mask (perimeter ~-5%, circularity ~+10%). `src/` stays frozen and is the
# reference for exact v1 reproduction.
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


def compute_pca(
    cells_info: pd.DataFrame,
    output_dir: Optional[str] = None,
    variance_threshold: float = 0.85,
    standardize_mode: str = "global",
    prune_correlation: Optional[float] = None,
    sign_convention: bool = False,
) -> pd.DataFrame:
    """Standardize `cells_info` (already filtered to the desired feature
    columns, e.g. `data_input.ML_FEATURE_COLUMNS`) and run PCA, keeping enough
    components to explain `variance_threshold` cumulative variance (v1
    hardcoded 0.85). Returns a DataFrame indexed like `cells_info`, columns
    `PC1..PCk, PC_score` (PC_score = variance-weighted sum of scores), sorted
    descending by `PC_score` — exactly as `bio_compute_cell_pca` did.

    If `output_dir` is given, writes the same diagnostic artifacts v1 did:
    `Explained_Variance_Ratio_All_PC.xlsx/.png`, per-component
    `PCA_Component_{i}.xlsx/.png`, and `Cells_info_PCA.xlsx`.

    `standardize_mode`, `prune_correlation` and `sign_convention` all default to
    reproducing v1 exactly; see `standardize`, `prune_correlated`, and below.
    `sign_convention` flips each component so its largest loading is positive,
    a port of `fit_pca` in
    `figures/Epiquant_ShapeAnalysis_allMutants.ipynb` -- PCA signs are otherwise
    arbitrary, so loading tables and mode plots need not reproduce across runs.
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

    pca_all = PCA()
    pca_all.fit(cell_info_scaled)
    explained_variance_ratio = pca_all.explained_variance_ratio_

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        explained_variance_df = pd.DataFrame({
            "Principal Component": ["PC" + str(i + 1) for i in range(len(explained_variance_ratio))],
            "Explained Variance Ratio": explained_variance_ratio,
        })
        explained_variance_df.to_excel(
            os.path.join(output_dir, "Explained_Variance_Ratio_All_PC.xlsx"), index=False
        )

        plt.figure(figsize=(10, 6), dpi=300)
        bars = plt.bar(
            explained_variance_df["Principal Component"],
            explained_variance_df["Explained Variance Ratio"] * 100,
            width=0.4, color="lightgreen",
        )
        plt.xlabel("Principal Component", fontsize=18, fontweight="bold")
        plt.ylabel("Explained Variance Ratio (%)", fontsize=18, fontweight="bold")
        plt.title("Explained Variance Ratio for All PC", fontsize=20, fontweight="bold")
        plt.xticks(rotation=0, fontsize=16, fontweight="bold")
        plt.tight_layout()
        for bar, variance in zip(bars, explained_variance_df["Explained Variance Ratio"] * 100):
            plt.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{variance:.2f}%", ha="center", color="black", fontsize=14, fontweight="bold",
            )
        plt.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
        plt.savefig(os.path.join(output_dir, "Explained_Variance_Ratio_All_PC.png"), dpi=300)
        plt.close()

    pca = PCA(n_components=variance_threshold)
    cell_info_pca = pca.fit_transform(cell_info_scaled)
    if sign_convention:
        flip = np.sign(pca.components_[
            np.arange(len(pca.components_)), np.abs(pca.components_).argmax(axis=1)
        ])
        pca.components_ *= flip[:, None]
        cell_info_pca = cell_info_pca * flip[None, :]
    components = pca.components_

    if output_dir is not None:
        for i in range(components.shape[0]):
            contributions = components[i]
            contributions_squared = contributions ** 2
            feature_abbreviations = [PCA_FEATURE_ABBREVIATIONS.get(col, col) for col in cells_info.columns]
            component_contributions = pd.DataFrame({
                "Feature": feature_abbreviations,
                "Contribution": contributions,
                "Contribution Squared": contributions_squared,
            })
            component_contributions.to_excel(
                os.path.join(output_dir, f"PCA_Component_{i + 1}.xlsx"), index=False
            )

            plt.figure(figsize=(10, 6), dpi=300)
            bars = plt.bar(
                component_contributions["Feature"],
                component_contributions["Contribution Squared"] * 100,
                width=0.4, color="skyblue",
            )
            plt.xlabel("Feature", fontsize=18, fontweight="bold")
            plt.ylabel("Contribution Squared (%)", fontsize=18, fontweight="bold")
            plt.title(f"Contribution Squared for PC {i + 1}", fontsize=20, fontweight="bold")
            plt.xticks(rotation=0, ha="right", fontsize=16, fontweight="bold")
            plt.tight_layout()
            for bar, contribution in zip(bars, component_contributions["Contribution Squared"] * 100):
                plt.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{contribution:.2f}%", ha="center", color="black", fontsize=14, fontweight="bold",
                )
            plt.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
            plt.savefig(os.path.join(output_dir, f"PCA_Component_{i + 1}_Contribution_Squared.png"), dpi=300)
            plt.close()

    pca_columns = ["PC" + str(i + 1) for i in range(cell_info_pca.shape[1])]
    cell_scores = pd.DataFrame(cell_info_pca, index=cells_info.index, columns=pca_columns)
    cell_scores["PC_score"] = np.sum(cell_scores * pca.explained_variance_ratio_, axis=1)

    if output_dir is not None:
        cell_scores.to_excel(os.path.join(output_dir, "Cells_info_PCA.xlsx"), sheet_name="Cell1")

    cells_info_pca = cell_scores.sort_values(axis=0, by="PC_score", ascending=False)
    return cells_info_pca


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


def _select_k_by_silhouette(
    data: np.ndarray,
    k_range: tuple[int, int] = (2, 9),
    random_state: Optional[int] = None,
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
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """K-means cluster cells on their PC scores (all `PC*` columns of
    `cells_info_pca` except the trailing `PC_score`).

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

    If `output_dir` is given, writes `cells_PCA_2D.png` and
    `cells_clustering_results.xlsx`. The plot draws a PC1-vs-PC2 scatter
    plus a logistic-regression decision-boundary contour ONLY when the
    final cluster count is exactly 2 (a boundary line is inherently a
    binary-classifier concept and doesn't generalize past 2 classes); for
    N>2 it's a plain multi-color scatter with no boundary.

    `random_state` is newly exposed (v1 had none, i.e. `None`/non-deterministic)
    so callers can pass a fixed seed for reproducible runs/tests.
    """
    fac_index = cells_info_pca.index
    components = [col for col in cells_info_pca.columns[:-1] if col.startswith("PC")]
    data = cells_info_pca[components].values

    if n_clusters == "auto":
        n_clusters, _ = _select_k_by_silhouette(data, k_range, random_state)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    label = kmeans.fit_predict(data)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 8), dpi=300)

        if len(np.unique(label)) == 2:
            plt.scatter(data[:, 0], data[:, 1], c=label, cmap="viridis", s=5)
            ax = plt.gca()
            ax.spines["right"].set_visible(True)
            ax.spines["top"].set_visible(True)
            ax.set_xlim(data[:, 0].min() - 1, data[:, 0].max() + 1)
            ax.set_ylim(data[:, 1].min() - 1, data[:, 1].max() + 1)

            log_reg = LogisticRegression(random_state=random_state)
            log_reg.fit(data[:, :2], label)

            x_values = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
            y_values = np.linspace(data[:, 1].min(), data[:, 1].max(), 100)
            xx, yy = np.meshgrid(x_values, y_values)
            grid = np.c_[xx.ravel(), yy.ravel()]
            probabilities = log_reg.predict_proba(grid)[:, 1].reshape(xx.shape)
            plt.contour(xx, yy, probabilities, levels=[0.5], linestyles=["--"], colors="r")
        else:
            plt.scatter(data[:, 0], data[:, 1], c=label, cmap="tab10", s=5)
            ax = plt.gca()
            ax.spines["right"].set_visible(True)
            ax.spines["top"].set_visible(True)
            ax.set_xlim(data[:, 0].min() - 1, data[:, 0].max() + 1)
            ax.set_ylim(data[:, 1].min() - 1, data[:, 1].max() + 1)

        plt.xlabel("PC1", fontsize=18, fontweight="bold")
        plt.ylabel("PC2", fontsize=18, fontweight="bold")
        plt.xticks(fontsize=16, fontweight="bold")
        plt.yticks(fontsize=16, fontweight="bold")
        plt.tight_layout(pad=1)
        plt.savefig(os.path.join(output_dir, "cells_PCA_2D.png"), dpi=300)
        plt.close()

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

    # 4. Families are additive, well-formed, and exclude the retired columns.
    RETIRED = {
        "Fitted Ellipse Angle", "Circularity", "P-Value", "Frame Neighbor Entropy",
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
    for retired in ("Fitted Ellipse Angle", "Circularity", "P-Value"):
        assert retired not in data_input.CELLS_INFO_COLUMNS, f"{retired} is back in the schema"
        assert retired not in data_input.ML_FEATURE_COLUMNS, f"{retired} is back in ML_FEATURE_COLUMNS"
        assert retired not in PCA_FEATURE_ABBREVIATIONS, f"{retired} still has an abbreviation"
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
    print(pca_df.drop(columns="PC_score").columns.tolist())

    n_clusters_arg = "auto" if args.n_clusters.strip().lower() == "auto" else int(args.n_clusters)
    k_range_arg = tuple(args.k_range)

    if n_clusters_arg == "auto":
        components = [col for col in pca_df.columns[:-1] if col.startswith("PC")]
        best_k, scores = _select_k_by_silhouette(pca_df[components].values, k_range_arg, args.random_state)
        print(f"\nSilhouette scores over k_range={k_range_arg}:")
        for k, score in scores.items():
            marker = "  <-- selected" if k == best_k else ""
            print(f"  k={k}: {score:.4f}{marker}")

    clustering_df = cluster_cells_kmeans(
        pca_df, args.output_dir, n_clusters=n_clusters_arg, k_range=k_range_arg, random_state=args.random_state
    )
    print("\nCluster sizes:")
    print(clustering_df["Cell_Type"].value_counts())
