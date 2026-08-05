"""Gap analysis: cell packing / neighbor-count analysis.

No adjacency-graph or neighbor-counting code exists anywhere in src/. The
`Approximate Polygon Vertex Count` column (from `data_input.extract_cell_info`,
via `cv2.approxPolyDP`) describes a cell's own contour complexity, not its
number of topological neighbors -- packing analysis needs an actual
cell-adjacency graph, which nothing in v1 builds. Signatures + docstrings
only, no implementation this round.
"""

import numpy as np
import pandas as pd


def build_neighbor_graph(outlines: np.ndarray) -> dict[int, list[int]]:
    """Construct a cell-adjacency graph from a labeled segmentation mask
    (`{cell_id: [neighbor_cell_id, ...]}`) -- the prerequisite for the
    neighbor-count/packing analyses below. Per manuscript Fig.5G, Fig.6B.
    No adjacency-graph code exists anywhere in v1 to port from."""
    raise NotImplementedError


def compute_neighbor_count_distribution(neighbor_graph: dict[int, list[int]]) -> pd.DataFrame:
    """Fraction of 5-6-sided vs. 7-9-sided polygons (by neighbor count) over
    time, per manuscript Fig.6B."""
    raise NotImplementedError


def compute_area_vs_neighbor_count(cells_info: pd.DataFrame, neighbor_graph: dict[int, list[int]]) -> pd.DataFrame:
    """Cell area vs. neighbor-count / vertex-class relationship, per
    manuscript Fig.5G (normalized cell area across vertex classes, WT vs.
    pyd)."""
    raise NotImplementedError
