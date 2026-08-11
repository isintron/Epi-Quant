"""Cell-adjacency (neighbor) graphs from filled per-cell label arrays, plus
the cross-frame "environment score" built on top of them.

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

from typing import Dict, Mapping, Set

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


def environment_scores(
    graph_current: Mapping[int, Set[int]],
    graph_other: Mapping[int, Set[int]],
    label_to_track_current: Mapping[int, int],
    label_to_track_other: Mapping[int, int],
) -> Dict[int, float]:
    """How much each cell's *neighborhood* changed between two frames, as a
    Dice/Sorensen coefficient over its neighbor sets:
    `2*|A intersect B| / (|A| + |B|)`, in `[0, 1]`. Returns
    `{raw_label_at_current_frame: score}`.

    This is Tissue Analyzer's "environment score" (Aigouy & Prud'homme,
    "Segmentation and Quantitative Analysis of Epithelial Tissues", Fig. 3g/h;
    the metric itself is EPySeg's `pairwise_score`, `tracking_error_detector_and_fixer.py`).
    A **low score means the cell's environment changed rapidly**, which often
    indicates a segmentation or tracking error worth inspecting; a high score
    means the neighborhood was preserved. Note that a low score is NOT
    necessarily an error -- a division or a delamination produces one
    legitimately (TA's own Note 16).

    `graph_*` come from `build_neighbor_graph`; `label_to_track_*` map that
    frame's raw labels to cross-frame-stable track ids.

    **Neighbor sets are compared in TRACK-ID space, never raw-label space.**
    Segmentation relabels cells arbitrarily each frame, so a raw label at
    frame f has no relationship to the same number at frame f-1 -- comparing
    raw labels would produce meaningless scores. EPySeg gets this for free by
    computing neighbors directly on tracked (identity-colored) images; here
    the translation is explicit.

    Three deliberate differences from the reference implementation, all
    because Epi-Quant has genuinely untracked cells where EPySeg does not
    (it assigns every cell an identity, minting a new one when a track is
    lost):

    - **Untracked neighbors are dropped from both sets.** They have no
      cross-frame identity, so they cannot be confirmed same-or-different;
      counting them either way would be a guess.
    - **Unscoreable cells are omitted from the result entirely** rather than
      being scored `0`. A cell is unscoreable when it is itself untracked,
      when its track has no cell in the compared frame, or when neither side
      retains a comparable neighbor. EPySeg scores all of these `0`, which
      conflates "brand-new cell" with "environment completely changed"; the
      caller can render an absent key distinctly instead.
    - No thresholding happens here. TA treats `>= 0.75` as fine and `<= 0.5`
      as worth examining, but that is a display/triage concern, so raw
      scores are returned.
    """
    track_to_label_other = {track: label for label, track in label_to_track_other.items()}
    scores: Dict[int, float] = {}

    for label, track in label_to_track_current.items():
        other_label = track_to_label_other.get(track)
        if other_label is None:
            continue  # this track has no cell in the compared frame

        current_neighbor_tracks = {
            label_to_track_current[n] for n in graph_current.get(label, set())
            if n in label_to_track_current
        }
        other_neighbor_tracks = {
            label_to_track_other[n] for n in graph_other.get(other_label, set())
            if n in label_to_track_other
        }

        total = len(current_neighbor_tracks) + len(other_neighbor_tracks)
        if total == 0:
            continue  # nothing comparable on either side

        shared = len(current_neighbor_tracks & other_neighbor_tracks)
        scores[label] = (2.0 * shared) / total

    return scores


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
    print("Neighbor-graph assertions passed.")

    # --- environment_scores -------------------------------------------------
    # Cell of interest is track 100. At the current frame it neighbors tracks
    # {200, 300, 400}; at the other frame it neighbors {200, 300, 500}, i.e.
    # one of three neighbors was exchanged. Dice = 2*2 / (3+3) = 0.666...
    # Deliberately uses DIFFERENT raw labels for the same tracks between the
    # two frames, since that is exactly what real segmentation does -- if the
    # implementation ever compared raw labels instead of tracks, this would
    # score 0 rather than 2/3.
    graph_cur = {1: {2, 3, 4}, 2: {1}, 3: {1}, 4: {1}}
    graph_oth = {9: {8, 7, 6}, 8: {9}, 7: {9}, 6: {9}}
    l2t_cur = {1: 100, 2: 200, 3: 300, 4: 400}
    l2t_oth = {9: 100, 8: 200, 7: 300, 6: 500}

    scores = environment_scores(graph_cur, graph_oth, l2t_cur, l2t_oth)
    print("\nEnvironment scores (one neighbor exchanged):", scores)
    assert abs(scores[1] - 2 / 3) < 1e-9, scores[1]

    # Identical neighborhood -> 1.0, even with all raw labels renumbered.
    identical = environment_scores(
        graph_cur, {9: {8, 7, 6}, 8: {9}, 7: {9}, 6: {9}},
        l2t_cur, {9: 100, 8: 200, 7: 300, 6: 400},
    )
    assert identical[1] == 1.0, identical[1]
    print("Identical neighborhood scores 1.0 despite renumbered raw labels.")

    # An UNTRACKED neighbor (label 4 absent from the map) is dropped from both
    # sides rather than counted as a mismatch: {200,300} vs {200,300} -> 1.0.
    untracked = environment_scores(
        graph_cur, graph_oth, {1: 100, 2: 200, 3: 300}, {9: 100, 8: 200, 7: 300},
    )
    assert untracked[1] == 1.0, untracked[1]
    print("Untracked neighbors correctly excluded from the comparison.")

    # Track absent in the other frame -> omitted entirely (NOT scored 0).
    absent = environment_scores(graph_cur, graph_oth, l2t_cur, {8: 200, 7: 300})
    assert 1 not in absent, absent
    # ...and an untracked cell is likewise absent from the result.
    assert environment_scores(graph_cur, graph_oth, {}, l2t_oth) == {}
    print("Unscoreable cells omitted rather than scored 0.")

    print("\nAll adjacency smoke tests passed.")
