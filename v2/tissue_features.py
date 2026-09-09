"""Local tissue-texture descriptors: how a cell's neighbourhood is arranged,
rather than what the cell itself looks like.

The second half of reviewer 2's point 4 -- "Could they also implement local
tissue texture parameters? (this is admittedly not purely cell centric, but
that could be important way to sort cell type and get information about local
mechanical properties)". Per cell, over its 1-ring of neighbours:

- **Neighbor Count** -- the packing topology; 6 is the hexagonal ideal
- **Local Hexagonal Fraction** -- how much of the ring is 6-sided
- **Local Nematic Order** -- `S = |<exp(2i psi)>|` over neighbour long axes;
  1 = aligned, 0 = isotropic. Detects a shared stretch direction.
- **Local Mean Shape Index / Solidity** -- is this cell in a fluid-like or
  solid-like patch, in a lobed patch or a smooth one
- **Local Density** -- cells per unit area
- **Frame Neighbor Entropy** -- one value per frame (see below), the disorder
  of the whole frame's polygon-class distribution

**Adjacency here is edge-based, not dilation-based.** `adjacency.build_neighbor_graph`
decides adjacency by dilating each cell and seeing what it reaches, which
cannot tell "shares a wall" from "meets at a corner". On a synthetic square
lattice it therefore reports 8 neighbours per cell instead of 4, because the
four diagonal cells touch at the vertex. That is not a corner case for this
project: square packing is exactly what Tamada & Zallen (2015) describe in the
Drosophila presumptive pharynx, and it is one of the arrangements these
descriptors exist to detect. `build_interface_graph` counts *shared border
pixels* instead, via `segmentation_qc._interface_pixels`, and gets both
lattices right (hex 6, square 4).

Gaps still have to be closed first: on `test_data/pupae_E-cad` **no two of its
826 cells touch at all**, giving 0 raw interfaces, and
`segmentation_qc._bridge_gaps` recovers 2341. Bridging was verified not to
introduce corner adjacencies -- both lattices stay correct with it applied, at
every minimum-interface threshold from 1 to 8 px.

All of it needs the *filled* label array; the thin `'outlines'` array yields
near-empty neighbour sets either way.

Orientation `psi` comes from `shape_features`' `Sin2Theta`/`Cos2Theta`, which
are recomputed from the elliptic Fourier harmonic-1 SVD. It must **not** be
taken from `Cells_info.xlsx`'s `Fitted Ellipse Angle`: that column is stored
folded (`180 - angle if angle > 90`), a mirror reflection rather than a modulo,
so orientations 60 degrees apart collapse onto the same value -- measured, e.g.
cells drawn at 70 and 110 degrees both store 20.0. Averaging directions from
that column would be meaningless.
"""

import logging
import math
import os
from typing import Callable, Dict, Iterable, Optional, Set

import numpy as np
import pandas as pd
from scipy import ndimage

import data_input
import segmentation_qc
import shape_features as shape_features_module

logger = logging.getLogger(__name__)

HEXAGONAL_NEIGHBORS = 6

# Minimum shared-border pixels for two cells to count as neighbours. Only has
# to exceed incidental contact: a single-pixel touch from segmentation noise.
# Verified on synthetic hex and square lattices, which stay correct anywhere
# from 1 to 8 px.
#
# Deliberately NOT `segmentation_qc.suggest_min_interface_px`, which returns
# 14-40 px on these datasets. That threshold is tuned for *scoring individual
# junctions* in the QC overlay, where short edges are unreliable; applied to
# topology it discards real walls and drives the median neighbour count from 6
# down to 4 on all three test datasets.
MIN_INTERFACE_PX = 2

CELLS_TISSUE_COLUMNS = [
    "Neighbor Count",
    "Local Hexagonal Fraction",
    "Local Nematic Order",
    "Local Mean Shape Index",
    "Local Mean Solidity",
    "Local Density",
    "Frame Neighbor Entropy",
    "Is Border Cell",
]

# Dimensionless subset. `Local Density` is excluded: it is cells per pixel^2,
# so it changes with magnification and must not be pooled across images without
# a physical scale.
SCALE_INVARIANT_TISSUE_COLUMNS = [
    "Neighbor Count", "Local Hexagonal Fraction", "Local Nematic Order",
    "Local Mean Shape Index", "Local Mean Solidity", "Frame Neighbor Entropy",
]


def border_labels(
    fill_labels: np.ndarray,
    max_gap_px: int = segmentation_qc.MAX_GAP_PX,
    background_dilation_px: int = 2,
) -> Set[int]:
    """Labels whose neighbour count is truncated by something other than
    biology: those touching the image edge, **and** those on the edge of the
    segmented tissue.

    `segmentation_qc.neighbor_count_deviation` excludes only the first kind,
    which is right for its purpose but not enough here. In this project the
    tissue usually does not reach the image edge -- `AS_dorsal view_E-cad` is
    70.7% background and `pupae_E-cad` 67.6%, and **not one cell in either
    touches the frame**. Excluding only frame-edge cells therefore excludes
    nothing while 119 of 145 AS cells sit on the tissue boundary, dragging the
    mean neighbour count to 5.24 when Euler's theorem puts a 2D packing at ~6.

    Background adjacency has to be measured **after gap-closing**, or it
    catches everything: Cellpose leaves thin gaps between cells, so on
    `pupae_E-cad` all 826 cells are within a few pixels of "background". After
    `_bridge_gaps` only genuine outside-the-tissue space remains.

    With both rules the mean interior neighbour count lands at 5.97 / 5.92 /
    5.74 / 6.04 on `AS_dorsal` / `pupae` / `embryos_GBE_GAP43` / `Demo_Seg_QC`,
    with a median of exactly 6 on all four.
    """
    edges = (
        set(fill_labels[0, :].tolist()) | set(fill_labels[-1, :].tolist())
        | set(fill_labels[:, 0].tolist()) | set(fill_labels[:, -1].tolist())
    )
    edges.discard(0)
    border = {int(label) for label in edges}

    bridged = segmentation_qc._bridge_gaps(fill_labels, max_gap_px)
    outside = bridged == 0
    if outside.any():
        reaching = ndimage.binary_dilation(outside, iterations=background_dilation_px)
        border |= {int(label) for label in np.unique(bridged[reaching]).tolist() if label != 0}
    return border


def build_interface_graph(
    fill_labels: np.ndarray,
    min_interface_px: int = MIN_INTERFACE_PX,
    max_gap_px: int = segmentation_qc.MAX_GAP_PX,
) -> Dict[int, Set[int]]:
    """`{cell_id: {neighbor_id, ...}}` where two cells are neighbours only if
    they share at least `min_interface_px` of border.

    Gaps are closed first with `segmentation_qc._bridge_gaps`, because Cellpose
    output frequently leaves no two cells touching at all. See the module
    docstring for why this is used instead of `adjacency.build_neighbor_graph`.
    """
    bridged = segmentation_qc._bridge_gaps(fill_labels, max_gap_px)
    graph: Dict[int, Set[int]] = {}
    for (first, second), pixels in segmentation_qc._interface_pixels(bridged).items():
        if pixels.size < min_interface_px:
            continue
        graph.setdefault(int(first), set()).add(int(second))
        graph.setdefault(int(second), set()).add(int(first))
    # Cells with no qualifying interface still belong in the graph, or they
    # would silently vanish from the output table.
    for label in np.unique(fill_labels):
        if label != 0:
            graph.setdefault(int(label), set())
    return graph


def local_ring(graph: Dict[int, Set[int]], label: int, rings: int = 1, include_self: bool = True) -> Set[int]:
    """Labels within `rings` graph hops of `label`."""
    reached = {label}
    frontier = {label}
    for _ in range(rings):
        frontier = {n for f in frontier for n in graph.get(f, set())} - reached
        if not frontier:
            break
        reached |= frontier
    return reached if include_self else reached - {label}


def nematic_order(psi_values: Iterable[float]) -> float:
    """`S = |<exp(2i psi)>|` over a set of orientations: 1 = perfectly aligned,
    0 = isotropic.

    The factor 2 makes it a *director* average, so `psi` and `psi + pi` count as
    the same state -- which is what a cell long axis is. `S` is also invariant
    to any global rotation or sign convention, so it does not depend on how the
    imaging frame was oriented, unlike the individual angles.
    """
    values = np.asarray(list(psi_values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.abs(np.exp(2j * values).mean()))


def neighbor_count_entropy(graph: Dict[int, Set[int]], exclude: Set[int]) -> float:
    """Shannon entropy (bits) of a *frame's* polygon-class distribution over
    the classes {<=4, 5, 6, 7, 8, >=9}.

    Deliberately per frame, not per cell. A 1-ring holds 6-7 cells, and the
    entropy of a 6-sample categorical distribution is dominated by sampling
    noise -- as a per-cell column it would enter PCA looking like a tissue-order
    readout while carrying almost none. A perfectly hexagonal lattice gives 0;
    a disordered epithelium gives more.
    """
    counts = [len(neighbors) for label, neighbors in graph.items() if label not in exclude]
    if not counts:
        return float("nan")
    classes = np.clip(np.asarray(counts), 4, 9)
    _, frequency = np.unique(classes, return_counts=True)
    probability = frequency / frequency.sum()
    # max(0, ...) only to avoid printing "-0.000" for a single-class frame.
    return max(0.0, float(-np.sum(probability * np.log2(probability))))


def frame_tissue_features(
    fill_labels: np.ndarray,
    psi_by_label: Optional[Dict[int, float]] = None,
    shape_index_by_label: Optional[Dict[int, float]] = None,
    solidity_by_label: Optional[Dict[int, float]] = None,
    neighbor_graph: Optional[Dict[int, Set[int]]] = None,
    min_interface_px: int = MIN_INTERFACE_PX,
    rings: int = 1,
) -> Dict[int, dict]:
    """Every column in `CELLS_TISSUE_COLUMNS` for one frame, keyed by label.

    Pass `neighbor_graph` to reuse an already-built graph -- the webview caches
    one per frame in `EpiQuantAPI._get_neighbor_graph`.
    """
    graph = neighbor_graph if neighbor_graph is not None else build_interface_graph(
        fill_labels, min_interface_px=min_interface_px
    )
    border = border_labels(fill_labels)
    entropy = neighbor_count_entropy(graph, border)
    areas = np.bincount(fill_labels.ravel())

    out: Dict[int, dict] = {}
    for label, neighbors in graph.items():
        label = int(label)
        ring = local_ring(graph, label, rings=rings)
        # Border cells are reported (so the caller can see them) but excluded
        # from every ring statistic, including their neighbours' rings, since a
        # truncated neighbour count would corrupt those too.
        scored = [m for m in ring if m not in border]

        values = {name: np.nan for name in CELLS_TISSUE_COLUMNS}
        values["Is Border Cell"] = float(label in border)
        values["Frame Neighbor Entropy"] = entropy
        values["Neighbor Count"] = float(len(neighbors)) if label not in border else np.nan

        if scored:
            counts = [len(graph.get(m, ())) for m in scored]
            values["Local Hexagonal Fraction"] = float(np.mean([c == HEXAGONAL_NEIGHBORS for c in counts]))
            ring_area = float(sum(areas[m] if m < len(areas) else 0 for m in scored))
            if ring_area > 0:
                values["Local Density"] = len(scored) / ring_area
            if psi_by_label:
                values["Local Nematic Order"] = nematic_order(
                    [psi_by_label[m] for m in scored if m in psi_by_label]
                )
            if shape_index_by_label:
                present = [shape_index_by_label[m] for m in scored if m in shape_index_by_label]
                if present:
                    values["Local Mean Shape Index"] = float(np.nanmean(present))
            if solidity_by_label:
                present = [solidity_by_label[m] for m in scored if m in solidity_by_label]
                if present:
                    values["Local Mean Solidity"] = float(np.nanmean(present))
        out[label] = values
    return out


def extract_tissue_features(
    npy_folder: str,
    shape_info: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    min_interface_px: int = MIN_INTERFACE_PX,
    rings: int = 1,
    progress_cb: Optional[Callable[[str, int], None]] = None,
    frames: Optional[set] = None,
) -> pd.DataFrame:
    """Tissue-texture descriptors for every cell in every frame of `npy_folder`.

    Indexed by the same `Cell{frame}_{cell_id}` strings as
    `data_input.extract_cell_info` and `shape_features.extract_shape_features`,
    so all three tables join.

    `shape_info` supplies orientation, shape index and solidity per cell; it is
    computed here if not given. Passing an already-computed table avoids
    repeating that work.
    """
    npy_files = data_input.list_segmentation_files(npy_folder)
    frame_number_by_file = data_input.resolve_frame_numbers(npy_files)
    # `frames` limits the work to a range. Every value here is a function of
    # that cell's own mask and its own frame's neighbours, so a range is exact
    # (Round 19 C3).
    if frames is not None:
        wanted = {int(f) for f in frames}
        npy_files = [f for f in npy_files if frame_number_by_file.get(f) in wanted]
    total = len(npy_files)

    if shape_info is None:
        shape_info = shape_features_module.extract_shape_features(npy_folder)

    row_index: list[str] = []
    row_values: list[list[float]] = []

    for idx, filename in enumerate(npy_files):
        if os.path.splitext(filename)[1] != ".npy":
            continue
        loaded = np.load(os.path.join(npy_folder, filename), allow_pickle=True).item()
        labels = data_input.load_label_array(loaded)
        frame_id = str(frame_number_by_file[filename])

        if progress_cb is not None:
            progress_cb(
                f"Computing tissue descriptors for frame {frame_id}...",
                int((idx + 1) / total * 100) if total else 100,
            )

        prefix = f"Cell{frame_id}_"
        frame_shape = shape_info[shape_info.index.str.startswith(prefix)] if len(shape_info) else shape_info
        psi, shape_index, solidity = {}, {}, {}
        for cell_index, row in frame_shape.iterrows():
            label = int(cell_index[len(prefix):])
            sin2, cos2 = row.get("Sin2Theta", np.nan), row.get("Cos2Theta", np.nan)
            if np.isfinite(sin2) and np.isfinite(cos2):
                psi[label] = 0.5 * math.atan2(sin2, cos2)
            shape_index[label] = row.get("Shape Index", np.nan)
            solidity[label] = row.get("Solidity", np.nan)

        per_label = frame_tissue_features(
            labels,
            psi_by_label=psi,
            shape_index_by_label=shape_index,
            solidity_by_label=solidity,
            min_interface_px=min_interface_px,
            rings=rings,
        )
        for label, values in per_label.items():
            row_index.append(f"{prefix}{label}")
            row_values.append([values[name] for name in CELLS_TISSUE_COLUMNS])

    tissue_info = pd.DataFrame(row_values, index=row_index, columns=CELLS_TISSUE_COLUMNS)
    tissue_info.index.name = "Cell Index"

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        tissue_info.to_excel(os.path.join(output_path, "Cells_tissue.xlsx"), sheet_name="Tissue", index=True)

    return tissue_info


# ---------------------------------------------------------------------------


def _lattice(kind: str, size: int = 420, spacing: int = 42, jitter: float = 0.0, seed: int = 0) -> np.ndarray:
    """A synthetic packed epithelium: seeds on a lattice, every pixel assigned
    to its nearest seed. `hex` gives 6 neighbours per interior cell, `square`
    gives 4 -- the contrast the reviewer's square-packing example needs."""
    rng = np.random.default_rng(seed)
    seeds = []
    row_step = spacing * math.sqrt(3) / 2 if kind == "hex" else spacing
    row = 0
    y = spacing // 2
    while y < size:
        offset = (spacing // 2) if (kind == "hex" and row % 2) else 0
        x = spacing // 2 + offset
        while x < size:
            seeds.append((x + rng.normal(0, jitter), y + rng.normal(0, jitter)))
            x += spacing
        y += row_step
        row += 1
    seeds = np.array(seeds)
    yy, xx = np.mgrid[0:size, 0:size]
    points = np.column_stack([xx.ravel(), yy.ravel()])
    nearest = np.argmin(((points[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2), axis=1)
    return (nearest.reshape(size, size) + 1).astype(np.int32)


class _self_test:
    """Synthetic-lattice regression tests for the tissue descriptors."""

    @staticmethod
    def _check_neighbor_topology() -> None:
        hexagonal = _lattice("hex")
        square = _lattice("square")
        for name, labels, expected in (("hex", hexagonal, 6), ("square", square, 4)):
            graph = build_interface_graph(labels)
            border = border_labels(labels)
            interior = [len(v) for k, v in graph.items() if k not in border]
            entropy = neighbor_count_entropy(graph, border)
            counts = np.unique(interior, return_counts=True)
            assert int(np.median(interior)) == expected, (
                f"{name} lattice: median neighbour count {np.median(interior)}, expected {expected}"
            )
            print(f"  {name:6} lattice: {len(interior)} interior cells, neighbour counts "
                  f"{dict(zip(counts[0].tolist(), counts[1].tolist()))}, entropy={entropy:.3f} bits")

    @staticmethod
    def _check_ordering() -> None:
        hexagonal = _lattice("hex")
        hex_graph = build_interface_graph(hexagonal)
        hex_border = border_labels(hexagonal)
        hex_entropy = neighbor_count_entropy(hex_graph, hex_border)
        jittered = _lattice("hex", jitter=9.0, seed=3)
        jitter_entropy = neighbor_count_entropy(
            build_interface_graph(jittered), border_labels(jittered)
        )
        assert hex_entropy < jitter_entropy, (
            f"a perfect lattice must be more ordered than a jittered one "
            f"({hex_entropy:.3f} vs {jitter_entropy:.3f} bits)"
        )
        features = frame_tissue_features(hexagonal)
        interior_hex = [v["Local Hexagonal Fraction"] for k, v in features.items()
                        if not v["Is Border Cell"] and np.isfinite(v["Local Hexagonal Fraction"])]
        assert np.median(interior_hex) == 1.0, f"hex lattice local hexagonal fraction {np.median(interior_hex)}"
        print(f"  ordering: perfect hex entropy={hex_entropy:.3f} < jittered={jitter_entropy:.3f} bits; "
              f"median local hexagonal fraction={np.median(interior_hex):.2f}")

    @staticmethod
    def _check_nematic_order() -> None:
        assert abs(nematic_order([0.3] * 50) - 1.0) < 1e-12, "identical orientations must give S = 1"
        assert abs(nematic_order([0.0, math.pi]) - 1.0) < 1e-12, "psi and psi+pi are the same director"
        rng = np.random.default_rng(0)
        isotropic = nematic_order(rng.uniform(0, math.pi, 5000))
        assert isotropic < 0.15, f"random orientations must give S ~ 0, got {isotropic:.3f}"
        angles = rng.uniform(0, math.pi, 500)
        assert abs(nematic_order(angles) - nematic_order(angles + 0.7)) < 1e-12, (
            "S must be invariant to a global rotation"
        )
        hexagonal = _lattice("hex")
        graph = build_interface_graph(hexagonal)
        labels_present = sorted(graph)
        aligned = frame_tissue_features(hexagonal, psi_by_label={m: 0.4 for m in labels_present})
        scrambled = frame_tissue_features(
            hexagonal, psi_by_label={m: float(rng.uniform(0, math.pi)) for m in labels_present}
        )
        aligned_s = np.nanmedian([v["Local Nematic Order"] for v in aligned.values()])
        scrambled_s = np.nanmedian([v["Local Nematic Order"] for v in scrambled.values()])
        assert aligned_s > 0.99, f"aligned field local S = {aligned_s:.3f}"
        assert scrambled_s < 0.8, f"scrambled field local S = {scrambled_s:.3f}"
        print(f"  nematic: aligned=1.000, isotropic={isotropic:.3f}, rotation-invariant, director convention holds")
        print(f"  local nematic order: aligned field={aligned_s:.3f}, scrambled field={scrambled_s:.3f} "
              f"(a 1-ring is only ~7 cells, so scrambled cannot reach 0)")

    @staticmethod
    def _check_border_exclusion() -> None:
        inset = np.zeros((520, 520), np.int32)
        inset[50:470, 50:470] = _lattice("hex", size=420)
        frame_only = {
            int(label) for label in (
                set(inset[0, :].tolist()) | set(inset[-1, :].tolist())
                | set(inset[:, 0].tolist()) | set(inset[:, -1].tolist())
            ) if label != 0
        }
        assert not frame_only, "the inset tissue must not touch the frame, or this case proves nothing"
        inset_graph = build_interface_graph(inset)
        inset_border = border_labels(inset)
        assert inset_border, "tissue-edge cells must still be recognised as border cells"
        inset_interior = [len(v) for k, v in inset_graph.items() if k not in inset_border]
        assert 5.5 < np.mean(inset_interior) < 6.5, (
            f"interior mean neighbour count {np.mean(inset_interior):.2f} is off Euler's ~6 -- "
            f"tissue-edge cells are leaking into the statistics"
        )
        print(f"  inset tissue (never touches frame): {len(inset_border)} border cells found, "
              f"{len(inset_interior)} interior, mean neighbours={np.mean(inset_interior):.2f}")

    def run(self) -> None:
        self._check_neighbor_topology()
        self._check_ordering()
        self._check_nematic_order()
        self._check_border_exclusion()
        print("\ntissue_features.py self-test: OK")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standalone debug entry for v2.tissue_features")
    parser.add_argument("--self-test", action="store_true", help="run synthetic-lattice checks")
    parser.add_argument("--npy-folder", help="segmentation folder to summarize")
    parser.add_argument("--output-path", help="write Cells_tissue.xlsx here")
    args = parser.parse_args()

    if args.self_test:
        _self_test().run()
    elif args.npy_folder:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        folder = data_input.prepare_segmentation_folder(args.npy_folder)
        frame = extract_tissue_features(
            folder,
            output_path=args.output_path,
            progress_cb=lambda message, percent: print(f"[{percent:3d}%] {message}"),
        )
        print(f"\n{len(frame)} cells ({int(frame['Is Border Cell'].sum())} on the border)")
        print(frame[CELLS_TISSUE_COLUMNS].describe().round(4).to_string())
    else:
        parser.print_help()
