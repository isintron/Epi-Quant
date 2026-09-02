"""Cell-adjacency (neighbor) graphs from filled per-cell label arrays, plus
the cross-frame "environment score" built on top of them.

`manual_correction_classification.get_adjacent_cells` is a KDTree-based
cross-cluster-only check (used for the classification "cluster borders"
visualization), not a general same-population neighbor primitive, so this
module is the only general adjacency graph in the project.

**Adjacency here is dilation-based**, which tolerates the gaps segmentation
tools leave (on `test_data/pupae_E-cad` no two of its 826 cells touch at all:
at `dilation_px=1` every cell is isolated, at the default 2 the median
neighbour count is 6). What it cannot do is tell "shares a wall" from "meets
at a corner" -- on a synthetic square lattice it reports 8 neighbours per cell
instead of 4. `tissue_features.build_interface_graph` counts shared border
pixels instead and gets both lattices right; use that for packing/topology
analyses, and this one where gap-tolerant proximity is what is wanted (the
live "Correct Tracking" environment score).
"""

from typing import Dict, Mapping, Set

import numpy as np
from scipy import ndimage


# The dilation radius every neighbour graph in this project is built with.
# Named rather than repeated as a default literal so a T1 count can be stamped
# with the value that produced it -- the count is not reproducible without it.
DEFAULT_DILATION_PX = 2


def build_neighbor_graph(
    fill_labels: np.ndarray, dilation_px: int = DEFAULT_DILATION_PX, return_weights: bool = False,
):
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

    `return_weights=True` returns `{cell_id: {neighbor_id: strength}}` instead,
    where strength is the neighbour's pixel count inside the dilated footprint
    -- a proxy for contact length.

    **Why the weight matters.** TissueMiner derives neighbours from a
    `directed_bonds` half-edge table: a polygonal mesh where two cells are
    neighbours iff they share an edge. That is combinatorial, with no marginal
    cases. Dilation admits a continuum instead -- a 2 px contact and a 47 px
    contact are both simply "neighbour" -- and the T1 flicker lives in exactly
    that continuum. Measured on `embryos_GBE_GAP43`, contacts that disappear
    are systematically weak: median 34 px of dilated contact against 47 px for
    contacts that survive, with 49% of disappearing contacts below the 25th
    percentile of surviving ones. See `topology.contacts_in_track_space`'s
    hysteresis, which is what consumes these weights.

    Default behaviour is unchanged and asserted identical in the self-test.
    """
    graph: Dict[int, Set[int]] = {}
    weights: Dict[int, Dict[int, int]] = {}
    if fill_labels.size == 0:
        return weights if return_weights else graph

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
        touched = window[dilated_mask]
        found, counts = np.unique(touched, return_counts=True)
        neighbor_labels = set(found.tolist()) - {0, label}

        graph.setdefault(label, set()).update(neighbor_labels)
        for neighbor in neighbor_labels:
            graph.setdefault(neighbor, set()).add(label)

        if return_weights:
            # Every label gets an entry, isolated ones included, so the weighted
            # form has exactly the same key set as the unweighted graph. Without
            # this, a cell with no neighbours is present in one and absent from
            # the other, and any caller that iterates both disagrees about which
            # cells exist.
            weights.setdefault(label, {})
            for neighbor, count in zip(found.tolist(), counts.tolist()):
                if neighbor in (0, label):
                    continue
                # Symmetric by construction: the pair's strength is the larger
                # of the two one-sided counts, so A->B and B->A never disagree
                # and a hysteresis test cannot depend on iteration order.
                strength = max(int(count), weights.get(label, {}).get(neighbor, 0))
                weights.setdefault(label, {})[neighbor] = strength
                weights.setdefault(neighbor, {})[label] = strength

    return weights if return_weights else graph


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

    # --- weighted form: identical topology, plus contact strength
    weighted = build_neighbor_graph(grid, return_weights=True)
    assert {k: set(v) for k, v in weighted.items()} == {k: set(v) for k, v in graph.items()}, (
        "return_weights must not change WHICH cells are neighbours, only add strengths")
    for cell, neighbours in weighted.items():
        for neighbour, strength in neighbours.items():
            assert strength > 0, (cell, neighbour, strength)
            assert weighted[neighbour][cell] == strength, "contact strength must be symmetric"
    print("Weighted form: same topology, symmetric strengths, isolated cells kept:",
          {k: dict(v) for k, v in sorted(weighted.items())})

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
