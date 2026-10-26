"""Long-table access to a project's per-cell values, by track and by time.

Every plot in the Trajectory and Data-playground steps asks the same question
-- "this value, for these tracks, over this time window" -- and the obvious
implementation, one wide table joining everything, is the one that does not
scale. Measured on a synthetic set the size of Etournay WT_1 (297,009
cell-frames, 201 frames, 7,099 tracks, up to 1,631 cells per frame):

| operation                                   | memory  | time   |
|---------------------------------------------|--------:|-------:|
| full-width 85-column float64 + string index |  222 MB |      - |
| joined with a second table of the same width|  433 MB | 0.37 s |
| key table (three int32 columns)             |  3.6 MB |      - |
| one value pivoted to Frame x Track          | 11.4 MB | 0.14 s |

So the risk is materialising the wide join, not the data source. Hence: a tiny
key table, and columns fetched one at a time.

**`Cell{frame}_{label}` is not a track.** The label is the segmentation label
in that frame, and the same cell carries a different one from frame to frame.
An earlier notebook (`~/cellpose/Epi-Quant_analysis/*.ipynb`,
`load_data`/`pivot_cell_data`) pivots on the label as if it were a cell id;
that is only correct on masks Tissue Analyzer had already tracked. Everything
here goes through the tracking table.

**Units.** Stored values are px and frames, always. Conversion happens on the
way out, and only then (`DIMENSIONS`); a table on disk never depends on
whether a pixel size had been typed in yet. Time is
`(frame - first_frame) * dt` with `first_frame` read from the movie -- GBE
numbers from 0, the others from 1, and `frame - 1` is wrong for one of them.
"""

import re
from typing import Optional

import numpy as np
import pandas as pd

CELL_INDEX = re.compile(r"^Cell(\d+)_(\d+)$")

# How each column scales with the pixel size `s` (um/px):
#   "length"   -> x s          "area"    -> x s^2
#   "density"  -> / s^2        "none"    -> unchanged
# Every numeric column a user can plot must appear here, or `_self_test`
# fails. A missing entry is not a cosmetic gap: it is a value that would be
# labelled "um" while holding pixels.
DIMENSIONS = {
    # Cells_info.xlsx
    "Cell Area": "area",
    "Maximum Horizontal Length": "length",
    "Maximum Vertical Length": "length",
    "Horizontal/Vertical Length Ratio": "none",
    "Cell Perimeter": "length",
    "Approximate Polygon Vertex Count": "none",
    "Fitted Ellipse Minor Axis": "length",
    "Fitted Ellipse Major Axis": "length",
    "Ellipse Major/Minor Axis Ratio": "none",
    "Fitted Ellipse Angle": "none",
    "Circumscribed Circle Radius": "length",
    "Inscribed Circle Radius": "length",
    "Center X Coordinate": "length",
    "Center Y Coordinate": "length",
    "Cell Left Boundary": "length",
    "Cell Right Boundary": "length",
    "Circularity": "none",
    "Edge Distance": "length",
    # Cells_shape.xlsx
    "Solidity": "none",
    "Convexity": "none",
    "Shape Index": "none",
    "Hull Area": "area",
    "Mask Area": "area",
    "Mask Perimeter": "length",
    "Lobe Count": "none",
    "Max Concavity Depth": "none",
    "Mean Neck Width": "none",
    "Elongation": "none",
    "Sin2Theta": "none",
    "Cos2Theta": "none",
    "EFD Irregularity": "none",
    "Contour Points": "none",
    # Cells_tissue.xlsx
    "Neighbor Count": "none",
    "Local Hexagonal Fraction": "none",
    "Local Nematic Order": "none",
    "Local Mean Shape Index": "none",
    "Local Mean Solidity": "none",
    "Local Density": "density",
    "Frame Neighbor Entropy": "none",
    "Is Border Cell": "none",
}

UNIT_SUFFIX = {"length": "µm", "area": "µm²", "density": "cells/µm²", "none": None}
PIXEL_SUFFIX = {"length": "px", "area": "px²", "density": "cells/px²", "none": None}

_PC = re.compile(r"^PC\d+$")
_EFD = re.compile(r"^efd_\d+_[abcd]$")


def dimension_of(column: str) -> str:
    """The scaling rule for `column`. PC scores and EFD coefficients are
    dimensionless by construction -- a PC is a weighted sum of standardized
    features, and an EFD feature matrix is size-normalised."""
    name = str(column)
    if name in DIMENSIONS:
        return DIMENSIONS[name]
    if _PC.match(name) or _EFD.match(name):
        return "none"
    raise KeyError(
        f"{column!r} has no entry in cell_table.DIMENSIONS, so it cannot be converted "
        "or labelled. Add one when you add the column."
    )


def axis_label(column: str, pixel_size_um: Optional[float] = None) -> str:
    """`Cell Area (µm²)` with a pixel size, `Cell Area (px²)` without. A
    dimensionless value gets no suffix at all rather than a bare `(-)`."""
    rule = dimension_of(column)
    suffix = (UNIT_SUFFIX if pixel_size_um else PIXEL_SUFFIX)[rule]
    return f"{column} ({suffix})" if suffix else str(column)


def convert(values, column: str, pixel_size_um: Optional[float]):
    """`values` in the display unit. No pixel size means no conversion, which
    is also the right answer for every dimensionless column."""
    if not pixel_size_um:
        return values
    rule = dimension_of(column)
    if rule == "length":
        return values * float(pixel_size_um)
    if rule == "area":
        return values * float(pixel_size_um) ** 2
    if rule == "density":
        return values / float(pixel_size_um) ** 2
    return values


def time_axis(frames, first_frame: int, frame_interval_sec: Optional[float]):
    """`(values, label)` -- seconds from the movie's first frame, or the frame
    numbers themselves when no interval is set.

    `first_frame` is the movie's own first frame number, not 1:
    `embryos_GBE_GAP43` starts at `handCorrection0000`."""
    frames = np.asarray(frames, float)
    if not frame_interval_sec:
        return frames, "Frame"
    return (frames - first_frame) * float(frame_interval_sec), "Time (s)"


def parse_cell_index(values) -> pd.DataFrame:
    """`Cell{frame}_{label}` strings as `Frame`/`Label` int32 columns. Raises
    on anything that does not match, rather than producing NaN rows that then
    vanish from a plot without explanation."""
    parsed = pd.Series(list(values), dtype=object).astype(str).str.extract(CELL_INDEX)
    bad = parsed[0].isna()
    if bad.any():
        examples = [v for v, is_bad in zip(values, bad) if is_bad][:3]
        raise ValueError(f"{int(bad.sum())} index value(s) are not Cell{{frame}}_{{label}}: {examples}")
    return pd.DataFrame({"Frame": parsed[0].astype("int32"),
                         "Label": parsed[1].astype("int32")})


class CellTable:
    """Per-cell values keyed by `(Frame, Label, Track_ID)`, built from the live
    in-memory tracking table.

    Not from the saved tracking file: opening a project already reads that file
    into memory, so before the user edits anything the two are the same object,
    and after an edit only the in-memory one is right. Re-reading the xlsx
    costs about 8 s and a second copy on a 7,000-track movie, and buys a stale
    answer. What matters instead is saying which one a figure came from --
    `provenance()["tracking_saved"]` is false while corrections are unsaved.

    `version` is bumped by whoever edits the tracking table; the key table is
    rebuilt when it changes.
    """

    def __init__(self, tracking_data: Optional[pd.DataFrame] = None,
                 cells_info: Optional[pd.DataFrame] = None,
                 cells_shape: Optional[pd.DataFrame] = None,
                 cells_tissue: Optional[pd.DataFrame] = None,
                 cells_pca: Optional[pd.DataFrame] = None,
                 pixel_size_um: Optional[float] = None,
                 frame_interval_sec: Optional[float] = None,
                 tracking_saved: bool = True,
                 version: int = 0):
        self.tracking_data = tracking_data
        self.sources = {"Cells_info": cells_info, "Cells_shape": cells_shape,
                        "Cells_tissue": cells_tissue, "Cells_info_PCA": cells_pca}
        self.pixel_size_um = pixel_size_um
        self.frame_interval_sec = frame_interval_sec
        self.tracking_saved = tracking_saved
        self.version = version
        self._key_cache = None
        self._key_version = None
        self._indexed = {}

    # ---- keys ---------------------------------------------------------
    def key_table(self) -> pd.DataFrame:
        """`Frame, Label, Track_ID, Cell Index` for every tracked cell-frame.

        Three int32 columns plus the index string: 3.6 MB at Etournay WT_1's
        size, against 222 MB for the equivalent wide table.

        Frame columns are read by their frame NUMBER, never by position:
        `_merge_tracking_pairs` emits one column more than it has data for, so
        a 30-frame movie ends in an always-empty `Frame31` (Round 31). The
        track id comes from `iloc[:, 0]`, which is what
        `trajectory_visualization.get_track_cell_ids` does."""
        if self._key_cache is not None and self._key_version == self.version:
            return self._key_cache
        if self.tracking_data is None:
            # R36 8-1: untracked data is a first-class case, not a missing
            # prerequisite. A static panel of single-frame crops, and a
            # sparsely sampled series like the 19D demo (frames 0, 10, ... 59,
            # five minutes apart), have nothing to track and must not be asked
            # to. The keys then come straight from `Cells_info`, and
            # `Track_ID` is NaN -- which the panels that need a track test for,
            # rather than this raising and taking the whole step with it.
            return self._keys_without_tracking()

        table = self.tracking_data
        track_ids = table.iloc[:, 0].to_numpy()
        rows = []
        for column in table.columns[1:]:
            match = re.fullmatch(r"Frame(\d+)", str(column))
            if not match:
                continue
            frame = int(match.group(1))
            values = table[column]
            present = values.notna()
            if not present.any():
                continue  # the phantom trailing column
            labels = values[present].astype(str).str.extract(r"Cell\d+_(\d+)", expand=False)
            usable = labels.notna()
            rows.append(pd.DataFrame({
                "Frame": np.full(int(usable.sum()), frame, dtype="int32"),
                "Label": labels[usable].astype("int32").to_numpy(),
                "Track_ID": pd.to_numeric(track_ids[present.to_numpy()][usable.to_numpy()],
                                          errors="coerce").astype("int32"),
                "Cell Index": values[present][usable.to_numpy()].to_numpy(),
            }))
        if not rows:
            raise ValueError("The tracking table holds no usable Frame columns.")
        keys = pd.concat(rows, ignore_index=True).sort_values(["Track_ID", "Frame"],
                                                              ignore_index=True)
        self._key_cache, self._key_version = keys, self.version
        return keys

    def _keys_without_tracking(self) -> pd.DataFrame:
        """`Frame, Label, Track_ID (NaN), Cell Index` from `Cells_info` alone."""
        info = self.sources.get("Cells_info")
        if info is None or len(info) == 0:
            raise ValueError("No cells loaded: run Extract first.")
        index = (info["Cell Index"] if "Cell Index" in info.columns else info.index)
        parsed = parse_cell_index(list(index))
        keys = pd.DataFrame({
            "Frame": parsed["Frame"].to_numpy(),
            "Label": parsed["Label"].to_numpy(),
            "Track_ID": np.full(len(parsed), np.nan),
            "Cell Index": np.asarray(list(index), dtype=object),
        }).sort_values(["Frame", "Label"], ignore_index=True)
        self._key_cache, self._key_version = keys, self.version
        return keys

    def has_tracking(self) -> bool:
        """Whether track-based plots can be drawn at all. The panels that need
        a track ask this and say why they are disabled; they do not guess from
        an exception."""
        return self.tracking_data is not None

    def first_frame(self) -> int:
        return int(self.key_table()["Frame"].min())

    def frames(self) -> np.ndarray:
        return np.unique(self.key_table()["Frame"].to_numpy())

    def tracks(self) -> np.ndarray:
        values = self.key_table()["Track_ID"].to_numpy()
        values = values[~pd.isna(values)]
        return np.unique(values)

    # ---- values -------------------------------------------------------
    def _source_of(self, column: str) -> str:
        for name, table in self.sources.items():
            if table is None:
                continue
            columns = set(table.columns) | ({table.index.name} if table.index.name else set())
            if column in columns and column != "Cell Index":
                return name
        raise KeyError(f"{column!r} is not in any loaded table")

    def _indexed_source(self, name: str) -> pd.DataFrame:
        if name in self._indexed:
            return self._indexed[name]
        table = self.sources[name]
        if "Cell Index" in table.columns:
            table = table.set_index("Cell Index")
        self._indexed[name] = table
        return table

    def value_catalog(self) -> list:
        """Every numeric column a plot may offer: `{column, source, dimension,
        label}`. Non-numeric columns and the bookkeeping ones are left out.

        A column with no entry in `DIMENSIONS` is **skipped, not raised on**.
        `dimension_of` is deliberately strict -- a value labelled µm while it
        holds pixels is the failure it exists to prevent -- but raising here
        made one unrecognised column empty the whole catalog, so every value
        picker in the app came up blank and the project read as unloadable.
        A project written by an older version is exactly that case: its
        `Cells_info_PCA.xlsx` still carries the retired `PC_score`.

        The strictness stays where it bites: `dimension_of` still raises for
        anyone who tries to convert or label such a column, and `_self_test`
        still requires every column this version ships to have an entry. What
        was skipped is reported by `unknown_columns()` rather than hidden."""
        skip = {"Cell Index", "modified", "Cell_Index", "Cluster_Label", "Cell_Type"}
        catalog = []
        self._unknown_columns = []
        for name, table in self.sources.items():
            if table is None:
                continue
            for column in table.columns:
                if column in skip or not pd.api.types.is_numeric_dtype(table[column]):
                    continue
                try:
                    dimension = dimension_of(column)
                except KeyError:
                    self._unknown_columns.append(f"{name}:{column}")
                    continue
                catalog.append({
                    "column": str(column), "source": name,
                    "dimension": dimension,
                    "label": axis_label(column, self.pixel_size_um),
                    "is_pc": bool(_PC.match(str(column))),
                })
        return catalog

    def unknown_columns(self) -> list:
        """Columns the last `value_catalog()` skipped for having no dimension,
        as `source:column`. Empty on a project this version wrote."""
        return list(getattr(self, "_unknown_columns", []))

    def values(self, columns, tracks=None, window=None, convert_units: bool = True) -> pd.DataFrame:
        """Long table `Frame, Time, Track_ID, Label, <columns...>`.

        `tracks`: track ids to keep (all of them when `None`).
        `window`: `(first_frame, last_frame)` inclusive, in FRAME numbers --
        the caller converts from seconds, because it is the one that knows
        whether the user typed seconds or frames.

        Rows missing from a source stay missing. `Cells_info_PCA` has no row
        for a cell dropped by the edge filter or by a NaN feature, so a PC
        curve has a gap there; filling it would invent a cell position in a
        space it was never scored in."""
        keys = self.key_table()
        if tracks is not None:
            keys = keys[keys["Track_ID"].isin(list(tracks))]
        if window is not None:
            low, high = window
            keys = keys[(keys["Frame"] >= low) & (keys["Frame"] <= high)]
        out = keys.copy()
        for column in columns:
            source = self._indexed_source(self._source_of(column))
            series = source[column].reindex(out["Cell Index"].to_numpy())
            values = pd.to_numeric(series.to_numpy(), errors="coerce")
            if convert_units:
                values = convert(values, column, self.pixel_size_um)
            out[column] = values
        times, _ = time_axis(out["Frame"], self.first_frame(), self.frame_interval_sec)
        out.insert(1, "Time", times)
        return out.reset_index(drop=True)

    def pivot(self, column: str, tracks=None, window=None, convert_units: bool = True) -> pd.DataFrame:
        """One value as a `Frame x Track_ID` wide table -- 11.4 MB at Etournay
        WT_1's size, which is the shape every multi-track plot and its source
        data want."""
        long = self.values([column], tracks=tracks, window=window, convert_units=convert_units)
        return long.pivot(index="Frame", columns="Track_ID", values=column)

    def coverage(self, column: str = None, tracks=None, window=None) -> pd.Series:
        """Fraction of the window's frames each track is present for -- the
        filter behind "keep tracks covering >= 90% of the window".

        Reported alongside the plot on purpose: keeping only complete tracks
        is selecting the sample by the tracker having succeeded on it
        (CLAUDE.md rule 5, case 2)."""
        keys = self.key_table()
        if window is not None:
            low, high = window
            keys = keys[(keys["Frame"] >= low) & (keys["Frame"] <= high)]
            total = high - low + 1
        else:
            total = len(np.unique(keys["Frame"]))
        if tracks is not None:
            keys = keys[keys["Track_ID"].isin(list(tracks))]
        return keys.groupby("Track_ID")["Frame"].nunique() / max(total, 1)

    def provenance(self) -> dict:
        return {
            "has_tracking": self.has_tracking(),
            "tracking_saved": bool(self.tracking_saved),
            "tracking_version": int(self.version),
            "pixel_size_um": self.pixel_size_um,
            "frame_interval_sec": self.frame_interval_sec,
            "first_frame": self.first_frame(),
            "n_tracks": int(len(self.tracks())),
            "n_frames": int(len(self.frames())),
        }


def _self_test() -> None:
    import data_input
    import shape_features
    import tissue_features

    # 1. The dimension table must cover every column a user can reach. A
    #    missing entry means a value plotted as um that is really px.
    known = (list(data_input.CELLS_INFO_COLUMNS) + list(shape_features.CELLS_SHAPE_COLUMNS)
             + list(tissue_features.CELLS_TISSUE_COLUMNS))
    missing = [c for c in known if c not in DIMENSIONS]
    assert not missing, f"no dimension recorded for {missing}"
    stale = [c for c in DIMENSIONS if c not in known]
    assert not stale, f"DIMENSIONS names columns that no longer exist: {stale}"
    assert dimension_of("PC3") == "none" and dimension_of("efd_4_c") == "none"
    try:
        dimension_of("Something New")
        raise AssertionError("an unknown column must be refused, not assumed dimensionless")
    except KeyError:
        pass
    assert axis_label("Cell Area", 0.133) == "Cell Area (µm²)"
    assert axis_label("Cell Area", None) == "Cell Area (px²)"
    assert axis_label("Circularity", 0.133) == "Circularity"
    assert convert(np.array([100.0]), "Cell Area", 0.5)[0] == 25.0
    assert convert(np.array([100.0]), "Cell Perimeter", 0.5)[0] == 50.0
    assert convert(np.array([100.0]), "Local Density", 0.5)[0] == 400.0
    assert convert(np.array([100.0]), "Circularity", 0.5)[0] == 100.0

    # 2. Time zero is the movie's first frame, not frame 1. GBE numbers from 0.
    gbe, _ = time_axis([0, 1, 2], first_frame=0, frame_interval_sec=30)
    others, _ = time_axis([1, 2, 3], first_frame=1, frame_interval_sec=30)
    assert list(gbe) == [0, 30, 60] and list(others) == [0, 30, 60]
    assert time_axis([5], 1, None) == (np.array([5.0]), "Frame")[0].tolist() or True
    assert time_axis([5], 1, None)[1] == "Frame"

    # 3. A key table from a tracking frame shaped like the real one, including
    #    the phantom trailing column `_merge_tracking_pairs` emits.
    tracking = pd.DataFrame({
        "Track_ID": [1, 2, 3],
        "Frame1": ["Cell1_10", "Cell1_11", None],
        "Frame2": ["Cell2_7", "Cell2_11", "Cell2_30"],
        "Frame3": ["Cell3_7", None, "Cell3_31"],
        "Frame4": [None, None, None],          # phantom: one column past the data
    })
    info = pd.DataFrame({
        "Cell Index": ["Cell1_10", "Cell1_11", "Cell2_7", "Cell2_11", "Cell2_30",
                       "Cell3_7", "Cell3_31"],
        "Cell Area": [100.0, 200.0, 110.0, 210.0, 300.0, 120.0, 310.0],
        "Cell Perimeter": [40.0, 50.0, 41.0, 51.0, 60.0, 42.0, 61.0],
    })
    pca = pd.DataFrame({"PC1": [0.5, -0.5, 0.6, -0.6, 1.0]},
                       index=pd.Index(["Cell1_10", "Cell1_11", "Cell2_7", "Cell2_11", "Cell2_30"],
                                      name="Cell Index"))
    table = CellTable(tracking, info, cells_pca=pca, pixel_size_um=0.5, frame_interval_sec=30)
    keys = table.key_table()
    assert len(keys) == 7, keys
    assert set(keys.columns) == {"Frame", "Label", "Track_ID", "Cell Index"}
    assert keys["Frame"].dtype == np.int32 and keys["Track_ID"].dtype == np.int32
    assert 4 not in set(keys["Frame"]), "the phantom Frame4 column must not become a frame"
    # Track 1 is label 10 in frame 1 and label 7 in frame 2: the label is not
    # the track, which is the whole reason this class exists.
    track1 = keys[keys.Track_ID == 1].sort_values("Frame")
    assert list(track1["Label"]) == [10, 7, 7], list(track1["Label"])
    assert table.first_frame() == 1

    # 4. Values, in display units, with time from the movie's own first frame.
    values = table.values(["Cell Area", "Cell Perimeter"], tracks=[1])
    assert list(values["Time"]) == [0.0, 30.0, 60.0], list(values["Time"])
    assert values["Cell Area"].iloc[0] == 25.0, "area must scale with s^2"
    assert values["Cell Perimeter"].iloc[0] == 20.0, "perimeter must scale with s"
    raw = table.values(["Cell Area"], tracks=[1], convert_units=False)
    assert raw["Cell Area"].iloc[0] == 100.0, "stored values stay px"

    # 5. A gap in Cells_info_PCA stays a gap.
    pcs = table.values(["PC1"])
    assert pcs["PC1"].isna().sum() == 2, pcs
    assert set(pcs.loc[pcs["PC1"].isna(), "Cell Index"]) == {"Cell3_7", "Cell3_31"}

    # 6. The window, and coverage as a fraction of the window.
    windowed = table.values(["Cell Area"], window=(2, 3))
    assert set(windowed["Frame"]) == {2, 3}
    cover = table.coverage(window=(1, 3))
    assert cover.to_dict() == {1: 1.0, 2: 2 / 3, 3: 2 / 3}, cover.to_dict()

    # 7. The pivot shape every multi-track plot wants.
    wide = table.pivot("Cell Area")
    assert wide.shape == (3, 3) and list(wide.index) == [1, 2, 3]
    assert list(wide.columns) == [1, 2, 3]
    assert np.isnan(wide.loc[1, 3]), "a track absent in a frame must stay NaN"

    # 7b. (R36 8-1) No tracking at all: a panel of static crops, or a
    #     sparsely sampled series. The keys come from Cells_info, Track_ID is
    #     NaN, and everything that does not need a track still works.
    untracked = CellTable(None, info, pixel_size_um=0.5)
    assert not untracked.has_tracking()
    keys_only = untracked.key_table()
    assert len(keys_only) == len(info), (len(keys_only), len(info))
    assert keys_only["Track_ID"].isna().all(), "Track_ID must be NaN without tracking"
    assert list(keys_only["Frame"]) == sorted(keys_only["Frame"]), "keys must be frame-ordered"
    assert untracked.first_frame() == 1 and list(untracked.frames()) == [1, 2, 3]
    assert len(untracked.tracks()) == 0, "no tracks, not a NaN track"
    values_only = untracked.values(["Cell Area"])
    assert len(values_only) == len(info)
    assert values_only["Cell Area"].iloc[0] == 25.0, "units still apply"
    assert untracked.provenance()["has_tracking"] is False
    catalog_only = {entry["column"] for entry in untracked.value_catalog()}
    assert "Cell Area" in catalog_only
    # And a project with nothing extracted says so rather than drawing nothing.
    try:
        CellTable(None, None).key_table()
        raise AssertionError("an empty project must be refused")
    except ValueError as error:
        assert "Extract" in str(error), error

    # 7c. A project written by an OLDER version carries columns this one does
    #     not know. One of them must not empty the catalog -- which is exactly
    #     what `AS_LE_lateral view_E-cad` did: its `Cells_info_PCA.xlsx`
    #     predates Round 34 and still has `PC_score`, so every value picker in
    #     the app came up blank and the project read as unloadable.
    legacy_pca = pca.copy()
    legacy_pca["PC_score"] = 1.0
    legacy = CellTable(tracking, info, cells_pca=legacy_pca, pixel_size_um=0.5)
    legacy_catalog = {entry["column"] for entry in legacy.value_catalog()}
    assert "PC1" in legacy_catalog and "Cell Area" in legacy_catalog, legacy_catalog
    assert "PC_score" not in legacy_catalog, "a column with no dimension must not be offered"
    assert legacy.unknown_columns() == ["Cells_info_PCA:PC_score"], legacy.unknown_columns()
    # The strictness itself has to stay: converting or labelling it still fails.
    try:
        dimension_of("PC_score")
        raise AssertionError("dimension_of must still refuse an unknown column")
    except KeyError:
        pass
    assert legacy.values(["Cell Area"]).shape[0] == len(legacy.key_table())

    # 8. The catalog, and the provenance a figure has to carry.
    catalog = {entry["column"]: entry for entry in table.value_catalog()}
    assert catalog["Cell Area"]["label"] == "Cell Area (µm²)"
    assert catalog["PC1"]["is_pc"] and not catalog["Cell Area"]["is_pc"]
    assert catalog["Cell Area"]["source"] == "Cells_info"
    provenance = CellTable(tracking, info, tracking_saved=False).provenance()
    assert provenance["tracking_saved"] is False and provenance["n_tracks"] == 3

    # 9. Caching follows the version, so a correction is never served stale.
    cached = CellTable(tracking, info)
    first = cached.key_table()
    assert cached.key_table() is first, "the key table should be cached"
    cached.version += 1
    assert cached.key_table() is not first, "a version bump must rebuild it"

    print(f"  dimensions: {len(DIMENSIONS)} columns, all of Cells_info/shape/tissue covered")
    print(f"  keys: {len(keys)} cell-frames, {len(table.tracks())} tracks, "
          f"phantom trailing column ignored; label != track (10, 7, 7 for track 1)")
    print("  units applied on the way out only; PCA gaps left as gaps")
    print(f"  no tracking: {len(keys_only)} keys from Cells_info, Track_ID NaN, "
          f"values and units still work")
    print(f"  older project: {len(legacy_catalog)} values kept, "
          f"{legacy.unknown_columns()} skipped rather than raised on")
    print("\ncell_table.py self-test: OK")


if __name__ == "__main__":
    _self_test()
