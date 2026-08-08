"""Same-frame cell-adjacency (neighbor) graph, from a filled per-cell label
array.

No cell-adjacency graph exists anywhere in `src/` or the rest of `v2/` --
`manual_correction_classification.get_adjacent_cells` is a KDTree-based
cross-cluster-only check (used for the classification "cluster borders"
visualization), not a general same-population neighbor primitive.
`gap_analysis/packing_analysis.py`'s `build_neighbor_graph` stub is the
explicit placeholder for exactly what this module provides; see that
module for how it delegates here.

This module is deliberately separate from `gap_analysis/` (which is an
explicitly deferred, stub-only package per its own docstring) since it is
real runtime logic used by the live "Correct Tracking" webview UI, not a
manuscript-analysis stub.
"""

from typing import Dict, Set

import numpy as np
from scipy import ndimage


def build_neighbor_graph(fill_labels: np.ndarray, dilation_px: int = 2) -> Dict[int, Set[int]]:
    """Return `{cell_id: {neighbor_cell_id, ...}}` for every non-zero label
    in `fill_labels`.

    `fill_labels` must be a *filled* per-cell label array (each cell's whole
    interior sharing one ID), matching `webview/api.py`'s `_load_frame`
    convention -- passing the thin boundary-line array (`boundary_labels`,
    raw Cellpose `'outlines'`) will silently produce near-empty neighbor
    sets, since dilating a 1px-wide line by `dilation_px` rarely reaches
    another cell's own boundary line.

    Two cells are "neighbors" if, after dilating one cell's mask by
    `dilation_px` pixels, its footprint overlaps the other cell's pixels --
    this tolerates the small gaps segmentation tools often leave between
    adjacent cells, not just literal pixel-touching.

    Performance: restricted to each cell's own padded bounding box (via
    `scipy.ndimage.find_objects`) rather than dilating the whole frame per
    label, so per-cell cost scales with that cell's own footprint size, not
    the frame's total cell count.
    """
    graph: Dict[int, Set[int]] = {}
    if fill_labels.size == 0:
        return graph

    height, width = fill_labels.shape
    objects = ndimage.find_objects(fill_labels)

    for index, bbox in enumerate(objects):
        if bbox is None:
            continue
        label = index + 1
        row_slice, col_slice = bbox
        r0 = max(row_slice.start - dilation_px, 0)
        r1 = min(row_slice.stop + dilation_px, height)
        c0 = max(col_slice.start - dilation_px, 0)
        c1 = min(col_slice.stop + dilation_px, width)

        window = fill_labels[r0:r1, c0:c1]
        cell_mask = window == label
        dilated_mask = ndimage.binary_dilation(cell_mask, iterations=dilation_px)
        neighbor_labels = set(np.unique(window[dilated_mask]).tolist()) - {0, label}

        graph.setdefault(label, set()).update(neighbor_labels)
        for neighbor in neighbor_labels:
            graph.setdefault(neighbor, set()).add(label)

    return graph


if __name__ == "__main__":
    # Synthetic 20x20 label grid: A and B are one background column apart
    # (neighbors under the default dilation_px=2); C is far from both
    # (not a neighbor of either); D is an isolated single-pixel cell with
    # no neighbors at all.
    grid = np.zeros((20, 20), dtype=int)
    grid[2:5, 2:5] = 1   # A
    grid[2:5, 6:9] = 2   # B -- 1-column gap from A
    grid[2:5, 15:18] = 3  # C -- far from A/B
    grid[18, 18] = 4     # D -- isolated

    graph = build_neighbor_graph(grid)
    print("Neighbor graph:", graph)

    assert graph[1] == {2}, f"expected A's only neighbor to be B, got {graph[1]}"
    assert graph[2] == {1}, f"expected B's only neighbor to be A, got {graph[2]}"
    assert graph[3] == set(), f"expected C to have no neighbors, got {graph[3]}"
    assert graph[4] == set(), f"expected D to have no neighbors, got {graph[4]}"
    print("All assertions passed.")
