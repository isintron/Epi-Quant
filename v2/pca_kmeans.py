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
    "Fitted Ellipse Angle": "θ",
    "Circumscribed Circle Radius": "R_c",
    "Inscribed Circle Radius": "r_i",
    "Center X Coordinate": "Cx",
    "Center Y Coordinate": "Cy",
    "Cell Left Boundary": "Lb",
    "Cell Right Boundary": "Rb",
    "Circularity": "C",
    "P-Value": "P-value",
}


def compute_pca(
    cells_info: pd.DataFrame,
    output_dir: Optional[str] = None,
    variance_threshold: float = 0.85,
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
    """
    scaler = StandardScaler()
    cell_info_scaled = scaler.fit_transform(cells_info)

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standalone debug entry for v2.pca_kmeans")
    parser.add_argument("--cells-info", required=True, help="Path to a Cells_info.xlsx (sheet 'Cell')")
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
