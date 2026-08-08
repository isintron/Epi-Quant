"""Gap analysis: cell packing / neighbor-count analysis.

No neighbor-counting/packing-statistics code exists anywhere in src/. The
`Approximate Polygon Vertex Count` column (from `data_input.extract_cell_info`,
via `cv2.approxPolyDP`) describes a cell's own contour complexity, not its
number of topological neighbors.

`build_neighbor_graph` below is no longer a stub -- it delegates to
`adjacency.build_neighbor_graph`, built for the tracking-correction
"smart-stop propagate" feature and reusable here as-is. The two analyses
below it remain unimplemented (signatures + docstrings only).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import adjacency


def build_neighbor_graph(fill_labels: np.ndarray) -> dict[int, list[int]]:
    """Construct a cell-adjacency graph from a *filled* per-cell labeled
    segmentation mask (`{cell_id: [neighbor_cell_id, ...]}`) -- the
    prerequisite for the neighbor-count/packing analyses below. Per
    manuscript Fig.5G, Fig.6B. Parameter renamed from `outlines` to
    `fill_labels` for clarity: this must be a filled mask, not a thin
    boundary-line array, matching the convention documented in
    `webview/api.py`'s `_load_frame`. Thin wrapper around
    `adjacency.build_neighbor_graph`; see that module for the
    dilation-overlap algorithm."""
    graph = adjacency.build_neighbor_graph(fill_labels)
    return {cell_id: sorted(neighbors) for cell_id, neighbors in graph.items()}


def compute_neighbor_count_distribution(neighbor_graph: dict[int, list[int]]) -> pd.DataFrame:
    """Fraction of 5-6-sided vs. 7-9-sided polygons (by neighbor count) over
    time, per manuscript Fig.6B."""
    raise NotImplementedError


def compute_area_vs_neighbor_count(cells_info: pd.DataFrame, neighbor_graph: dict[int, list[int]]) -> pd.DataFrame:
    """Cell area vs. neighbor-count / vertex-class relationship, per
    manuscript Fig.5G (normalized cell area across vertex classes, WT vs.
    pyd)."""
    raise NotImplementedError
