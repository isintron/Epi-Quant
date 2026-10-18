// api.js — thin wrapper around window.pywebview.api.*, plus a dev-mode mock
// used ONLY when this page is opened directly in a regular browser (no
// window.pywebview present) rather than inside the real pywebview app. This
// lets the frontend be visually sanity-checked (layout, step navigation,
// color pickers, click handling) without a native window or real data. It
// is not a user-facing feature and never activates inside the real app.

(function () {
  function hasPywebview() {
    return typeof window.pywebview !== "undefined" && window.pywebview.api;
  }

  // 1x1 transparent PNG, used as a stand-in "image" for every mock overlay.
  const MOCK_PNG =
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

  const mockState = {
    frameCount: 5,
    // Deliberately sparse/non-contiguous, like a real "unrelated samples"
    // dataset (e.g. cp_dataset_puzzle-like_cells' [1, 9, 19]) -- exercises
    // frame-nav's index-based stepping rather than the easy 1..N case.
    frameNumbers: [1, 3, 7, 12, 19],
    clusterLegend: [
      { cluster_label: 0, cell_type: "cluster_1", color: "#e41a1c", count: 120, hidden: false },
      { cluster_label: 1, cell_type: "cluster_2", color: "#377eb8", count: 30, hidden: false },
    ],
    trackNumbers: [1, 2, 3, 4, 5],
    // The two Round 27 census sets re-solve the frame pairs, so they need the
    // feature table; the real API refuses without it and so does the mock.
    extracted: false,
    inspection: null,
    hasLeadingEdge: false,
    hasShape: false,
    trackingCost: { mode: "hybrid", overlap_weight: 1.0, centroid_weight: 1.0,
                    motion_compensation: false, max_frame_gap: 0, drift_mode: "none" },
    t1PersistFrames: 1,
    t1Enter: null, t1Exit: null, t1EventFilter: "persistent",
    registerReference: true,
    deadBand: 0.75,
    repairThreshold: 0.5,
    // Coverage log. `outputDirsReady` starts FALSE so dev mode exercises the
    // "no output directory yet" path -- the real API returns
    // `{logged: false}` there, and a mock that always succeeded would hide it.
    outputDirsReady: false,
    coverageSessionId: "devmode00cov",
    coverage: [],
    census: null,
    censusVerdicts: [],
    trackingRunDriftMode: null,   // set when the mock "runs" tracking
    minCellArea: 0,
    // True after "load existing extraction": the mock then plays a project
    // extracted before Round 24, whose Cells_info.xlsx lacks the two restored
    // basic columns, so dev mode reaches the disabled-checkbox path too.
    preR24CellsInfo: false,
  };

  // A tiny distinguishable PNG per key, so two census crops are not the same
  // pixels. `offset` shifts the block, which is what the blink reveals.
  function mockCrop(key, color, offset) {
    const size = 48;
    const canvas = typeof document !== "undefined" && document.createElement
      ? document.createElement("canvas") : null;
    if (!canvas) return MOCK_PNG;
    canvas.width = size; canvas.height = size;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#1c1c1e"; ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = color; ctx.fillRect(8 + offset, 12, 24, 22);
    ctx.fillStyle = "#ffffff"; ctx.font = "9px sans-serif"; ctx.fillText(key, 3, 45);
    return canvas.toDataURL("image/png");
  }

  // Axis states mirror the real `annotate_persistence` output. With the
  // persistence filter off, nothing is transient.
  function t1MockAxes() {
    const settled = mockState.t1PersistFrames === 0;
    return [
      { center: [20, 20], phi: 0.4, type: "gain", persistent: true, state: "persistent" },
      { center: [40, 30], phi: 1.2, type: "loss", persistent: settled,
        state: settled ? "persistent" : "transient" },
      { center: [60, 45], phi: 0.9, type: "gain", persistent: false,
        state: mockState.t1PersistFrames > 1 ? "unjudgeable" : "persistent" },
    ];
  }

  const EVENT_FILTER_STATES = {
    all: ["persistent", "transient", "unjudgeable"],
    persistent: ["persistent"],
    persistent_unjudgeable: ["persistent", "unjudgeable"],
  };

  function t1EventCounts(axes) {
    const keep = EVENT_FILTER_STATES[mockState.t1EventFilter] || EVENT_FILTER_STATES.all;
    const states = axes.map((a) => a.state || "persistent");
    return {
      filter: mockState.t1EventFilter,
      shown: states.filter((st) => keep.includes(st)).length,
      filtered: states.filter((st) => !keep.includes(st)).length,
      unjudgeable: states.filter((st) => st === "unjudgeable").length,
      total: states.length,
    };
  }

  function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  const mockApi = {
    async select_folder(_title) {
      return "/mock/selected/folder";
    },
    async set_project(_maskFolder, _imageFolder, outputBase) {
      mockState.outputDirsReady = true;
      return {
        format: "npy",
        output_dirs: {
          cell_classification_output: outputBase + "/cell_classification_output",
          cell_track_output: outputBase + "/cell_track_output",
          quantitative_analysis_output: outputBase + "/quantitative_analysis_output",
        },
      };
    },
    // Round 19 D. The mock reports a real removal so dev-mode click-through
    // reaches the "N cells removed" branch, not only the silent one.
    async set_min_cell_area(areaPx) {
      mockState.minCellArea = Math.max(0, areaPx || 0);
      return { min_cell_area_px: mockState.minCellArea, suggested: 19 };
    },
    async get_min_cell_area() {
      return { min_cell_area_px: mockState.minCellArea, suggested: 19 };
    },
    async get_min_area_removals() {
      if (!mockState.minCellArea) {
        return { min_cell_area_px: 0, total: 0, by_frame: {}, unreadable_frames: [] };
      }
      return { min_cell_area_px: mockState.minCellArea, total: 3,
               by_frame: { 19: 1, 24: 1, 29: 1 }, unreadable_frames: [] };
    },
    async get_existing_results() {
      return { extraction: false, classification: false, tracking: false, shape: false, tissue: false };
    },
    async load_existing_extraction() {
      mockState.preR24CellsInfo = true;
      mockState.extracted = true;
      return { frame_count: mockState.frameCount, cell_count: 500 };
    },
    async load_existing_classification() {
      return { cluster_sizes: { cluster_1: 120, cluster_2: 30 }, k_used: 2 };
    },
    async load_existing_tracking() {
      return { track_count: mockState.trackNumbers.length, duplicates_cleaned: 0 };
    },
    async start_extraction(_maskType, _bgRatio, _excludeEdge) {
      mockState.preR24CellsInfo = false;
      mockState.extracted = true;
      (async () => {
        for (let p = 0; p <= 100; p += 25) {
          await delay(120);
          window.onProgress && window.onProgress({ stage: "extract", message: `mock frame ${p}`, percent: p });
        }
        window.onExtractDone && window.onExtractDone({ frame_count: mockState.frameCount, cell_count: 500 });
      })();
      return true;
    },
    async start_shape_extraction(_includeTissue, _order, _points, _sigma, _coeffs) {
      (async () => {
        for (let p = 0; p <= 100; p += 50) {
          await delay(120);
          window.onProgress && window.onProgress({ stage: "shape", message: `mock shape ${p}`, percent: p });
        }
        mockState.hasShape = true;
        window.onShapeExtractDone && window.onShapeExtractDone(await mockApi.get_shape_feature_summary());
      })();
      return true;
    },
    async get_shape_feature_summary() {
      if (!mockState.hasShape) return {};
      const column = (median) => ({ median, q1: median * 0.9, q3: median * 1.1, missing: 0 });
      return {
        shape: {
          cell_count: 500,
          columns: { Solidity: column(0.93), "Shape Index": column(4.12), "Lobe Count": column(3) },
          diagnostics: ["Mask Area", "Elongation", "Sin2Theta", "Cos2Theta"],
        },
        tissue: {
          cell_count: 500,
          columns: { "Neighbor Count": column(6), "Local Nematic Order": column(0.82) },
          diagnostics: ["Local Density", "Frame Neighbor Entropy"],
        },
      };
    },
    async load_existing_shape_features() {
      mockState.hasShape = true;
      return mockApi.get_shape_feature_summary();
    },
    async get_feature_options() {
      const ready = mockState.hasShape;
      const family = (label, columns, extra = {}) => ({
        label, columns, default_off: [], unavailable: [], base: false, exclusive: false, note: "",
        // Mirrors pca_kmeans.FEATURE_DISPLAY_NAMES: cosmetic only, the real
        // column name stays the checkbox value.
        display: Object.fromEntries(columns.map((c) => [c, ({
          "Cell Area": "Area",
          "Cell Perimeter": "Perimeter",
          "Ellipse Major/Minor Axis Ratio": "Ellipse Axis Ratio",
          "Fitted Ellipse Angle": "Ellipse Angle",
          "Maximum Horizontal Length": "Max Horizontal Length",
          "Maximum Vertical Length": "Max Vertical Length",
          "Horizontal/Vertical Length Ratio": "H/V Length Ratio",
        })[c] || c])),
        ready: extra.base ? true : ready, missing: ready || extra.base ? [] : ["Solidity"], ...extra,
      });
      return {
        families: {
          // Column names here must be the REAL ones -- `display` is the only
          // place a shortened label may appear. Getting that wrong in the mock
          // hides exactly the bug the mock exists to catch.
          basic: family("Basic geometrics",
            ["Cell Area", "Cell Perimeter", "Ellipse Major/Minor Axis Ratio",
             "Maximum Horizontal Length", "Maximum Vertical Length", "Horizontal/Vertical Length Ratio",
             "Fitted Ellipse Angle", "Circularity"],
            { base: true, ready: true, missing: [],
              default_off: ["Maximum Horizontal Length", "Maximum Vertical Length", "Horizontal/Vertical Length Ratio",
                            "Fitted Ellipse Angle", "Circularity"],
              unavailable: mockState.preR24CellsInfo ? ["Fitted Ellipse Angle", "Circularity"] : [],
              note: "H/V lengths measure orientation, not shape - tick them only if your "
                + "specimen is axis-aligned. Ellipse Angle is orientation as well, and it "
                + "is stored folded (70 and 110 degrees share one value), so it cannot be "
                + "read back as a direction: it separates cell types well on an "
                + "axis-aligned specimen and misleads on any other mounting."
                + "Circularity is the same quantity as Shape Index in the "
                + "Convex hull family - tick at most one of the two,"
                + "or PCA counts that shape twice." }),
          hull: family("Convex hull / compactness", ["Solidity", "Convexity", "Shape Index"],
            { note: "Solidity and Convexity are the only clean lobing measures." }),
          lobe: family("Convolution / lobe",
            ["Lobe Count", "Max Concavity Depth", "Mean Neck Width", "EFD Irregularity"]),
          tissue: family("Local tissue texture",
            ["Neighbor Count", "Local Hexagonal Fraction", "Local Nematic Order",
             "Local Mean Shape Index", "Local Mean Solidity"]),
          efd: family("Elliptic Fourier coefficients", [],
            { exclusive: true, note: "Used alone and without standardization." }),
        },
        has_shape: ready,
        has_tissue: ready,
        dataset_modes: ["time_series", "independent_samples"],
      };
    },
    async get_frame_count() {
      return mockState.frameCount;
    },
    async get_frame_numbers() {
      return mockState.frameNumbers;
    },
    async get_raw_frame_image(_frame) {
      return MOCK_PNG;
    },
    async get_qc_metric_names() {
      return ["junction_contrast", "boundary_contrast", "morphology", "neighbor_count", "measurement_failures"];
    },
    async get_tracking_cost() {
      return { ...mockState.trackingCost, default_overlap_weight: 1.0 };
    },
    async set_tracking_cost(mode, overlapWeight, motionCompensation, maxFrameGap, driftMode) {
      if (driftMode) mockState.trackingCost.drift_mode = driftMode;
      if (maxFrameGap !== null && maxFrameGap !== undefined) {
        mockState.trackingCost.max_frame_gap = Math.max(0, maxFrameGap);
      }
      if (overlapWeight !== null && overlapWeight !== undefined && !Number.isNaN(overlapWeight)) {
        mockState.trackingCost.overlap_weight = Math.max(0, overlapWeight);
      }
      if (mode) {
        mockState.trackingCost.mode = mode;
        mockState.trackingCost.centroid_weight = mode === "overlap" ? 0 : 1;
        if (mode === "centroid") mockState.trackingCost.overlap_weight = 0;
        else if (mockState.trackingCost.overlap_weight === 0) mockState.trackingCost.overlap_weight = 1;
      }
      if (motionCompensation !== null && motionCompensation !== undefined) {
        mockState.trackingCost.motion_compensation = !!motionCompensation;
      }
      return { ...mockState.trackingCost, default_overlap_weight: 1.0 };
    },
    async get_t1_persistence() {
      return { value: mockState.t1PersistFrames, enter_px: mockState.t1Enter,
               exit_px: mockState.t1Exit, event_filter: mockState.t1EventFilter };
    },
    async set_t1_hysteresis(enterPx, exitPx) {
      if (enterPx == null || exitPx == null) { mockState.t1Enter = mockState.t1Exit = null; }
      else {
        // The real backend REJECTS enter <= exit; a mock that accepted it would
        // hide the error path the UI has to handle.
        if (enterPx <= exitPx) throw new Error(`enter_px (${enterPx}) must exceed exit_px (${exitPx})`);
        mockState.t1Enter = enterPx; mockState.t1Exit = exitPx;
      }
      return mockApi.get_t1_persistence();
    },
    async set_t1_event_filter(mode) {
      if (!(mode in EVENT_FILTER_STATES)) throw new Error(`Unknown axis filter '${mode}'`);
      mockState.t1EventFilter = mode;
      return mockApi.get_t1_persistence();
    },
    async set_t1_persistence(value) {
      mockState.t1PersistFrames = Math.max(0, value || 0);
      return { value: mockState.t1PersistFrames };
    },
    async get_junction_min_px() {
      return { value: 21, auto: 21, is_override: false };
    },
    async set_junction_min_px(value) {
      return { value: value || 21, auto: 21, is_override: !!value };
    },
    async copy_image_to_clipboard(_dataUri) {
      return { success: true, message: "" };
    },
    async get_qc_metric_display() {
      return { boundary_contrast: { vmin: 0.0, vmax: 0.6, flag_below: 0.25 } };
    },
    async get_qc_overlay(_frame, _metric, _lut, _renderMode) {
      return MOCK_PNG;
    },
    async get_extract_preview(_frame, _metric, _lut, _renderMode, _showOutline, _outlineColor,
                              _showLabels, _labelFontSize) {
      return MOCK_PNG;
    },
    async get_qc_summary(_metric) {
      return [
        { frame: 1, scored: 176, flagged: 12, min: 0.16, median: 0.44 },
        { frame: 2, scored: 177, flagged: 31, min: 0.08, median: 0.39 },
      ];
    },
    async get_qc_cell_at_pixel(frame, _x, _y) {
      return { cell_index: `Cell${frame}_5`, cell_id: 5,
               scores: { boundary_contrast: 0.536, morphology: 0.95, neighbor_count: 0.75, measurement_failures: 1.0 },
               junctions: [{ with_cell: 40, contrast: -0.0145 }, { with_cell: 12, contrast: 0.131 }] };
    },
    async get_palette_presets() {
      return {
        tab10: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
        Set2: ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"],
        Dark2: ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"],
      };
    },
    async run_classification(_n, _k, _rs, featureFamilies, selectedFeatures, _prune, datasetMode) {
      const families = featureFamilies && featureFamilies.length ? featureFamilies : ["basic"];
      return {
        cluster_sizes: { cluster_1: 120, cluster_2: 30 },
        k_used: 2,
        features_used: selectedFeatures && selectedFeatures.length ? selectedFeatures : ["Cell Area", "Circularity"],
        feature_families: families,
        // Mirrors the backend rule that EFD is never standardized.
        standardize_mode: families.includes("efd") ? "none" : "global",
        dataset_mode: datasetMode || "time_series",
        cells_used: 500,
      };
    },
    async set_cluster_colors(colorMap) {
      mockState.clusterLegend.forEach((c) => {
        if (colorMap[c.cluster_label] !== undefined) c.color = colorMap[c.cluster_label];
      });
    },
    async get_cluster_legend() {
      return mockState.clusterLegend;
    },
    async set_hidden_clusters(hiddenLabels) {
      const hiddenSet = new Set(hiddenLabels);
      mockState.clusterLegend.forEach((c) => { c.hidden = hiddenSet.has(c.cluster_label); });
    },
    async set_show_cluster_borders(_enabled) {
      return null;
    },
    async set_border_color(_color) {
      return null;
    },
    async get_classification_overlay(_frame, _renderMode) {
      return MOCK_PNG;
    },
    async get_cell_at_pixel(frame, _x, _y) {
      return { cell_index: `Cell${frame}_1`, cluster_label: 0, cell_type: "cluster_1" };
    },
    async reassign_cluster(_cellIndex, _newLabel) {
      return mockState.clusterLegend;
    },
    async save_classification() {
      return null;
    },
    async can_compute_leading_edge() {
      return mockState.clusterLegend.length === 2;
    },
    async compute_leading_edge_all_frames(_threshold) {
      mockState.hasLeadingEdge = true;
      return { frames_processed: mockState.frameCount };
    },
    async save_cells_info_leading_edge() {
      return null;
    },
    async get_tracking_mode_options() {
      return { has_leading_edge: mockState.hasLeadingEdge, split_available: mockState.hasLeadingEdge };
    },
    async get_stale_frames() {
      return { tracked: true,
               frames: [{ frame: 1, stale: false, manual_cells: 8 }, { frame: 2, stale: true, manual_cells: 4 }],
               stale_frames: [2], suggested_range: [2, 2] };
    },
    async start_local_retrack(start, end) {
      (async () => {
        await delay(300);
        window.onRetrackDone && window.onRetrackDone({
          frames: [start, end], relinked: 12, track_count: 185,
          reconciled: [{ frame: end + 1, from_track_id: 99, into_track_id: 2, cell_index: `Cell${end + 1}_7` }],
          new_tracks: [{ track_id: 186, from_frame: start, cell_index: `Cell${start}_31` }],
        });
      })();
      return true;
    },
    async start_tracking(_mode) {
      (async () => {
        await delay(400);
        mockState.trackingRunDriftMode = mockState.trackingCost.drift_mode;
        window.onTrackDone && window.onTrackDone({ track_count: mockState.trackNumbers.length });
      })();
      return true;
    },
    async get_tracking_overlay(_frame, _renderMode, _showLabels, _colorMode, _identityPalette, _labelFontSize,
                               _environmentLut, _environmentDirection, _tintOthers) {
      return MOCK_PNG;
    },
    // Docked frame stepper (Round 16 E). The mock returns a track with a real
    // GAP and crops that actually differ between frames -- a stepper whose
    // frames all looked the same would make the shared-window/re-centre
    // distinction untestable in dev mode, which is the subtle part.
    async get_track_filmstrip(trackId) {
      if (!mockState.trackNumbers.includes(trackId)) return { error: `Track ${trackId} not found.` };
      const frames = mockState.frameNumbers;
      return { track_id: trackId, frames, held: frames.filter((_, i) => i !== 2) };
    },
    async get_track_step_frame(trackId, anchorFrame, frame, showLabels, showOutline, labelSize,
                               _pad, outlineColor) {
      if (!mockState.trackNumbers.includes(trackId)) return { error: `Track ${trackId} not found.` };
      const strip = await mockApi.get_track_filmstrip(trackId);
      const held = strip.held.includes(frame);
      const recentred = Math.abs(frame - anchorFrame) > 1;
      return {
        track_id: trackId, frame, anchor_frame: anchorFrame,
        cell_index: held ? `Cell${frame}_3` : null, held, recentred,
        window_note: recentred
          ? "re-centred on this frame -- background moves, so this shows where the track goes, "
            + "not whether it swapped"
          : "shared window with the anchor frame -- background is fixed, so this is a true blink",
        // Re-centred frames keep the cell still and shift the background;
        // shared-window frames do the opposite. The mock reproduces that.
        image: mockCrop(`f${frame}${showLabels ? "#" : ""}${showOutline ? "o" + (outlineColor || "") : ""}`,
                        held ? "#00ff88" : "#555555",
                        recentred ? 8 : Math.min(Math.max((frame - anchorFrame) * 6 + 8, 0), 20)),
      };
    },
    async get_tracking_cell_at_pixel(frame, _x, _y) {
      return { cell_index: `Cell${frame}_1`, cell_id: 1, track_id: 1 };
    },
    // Ground-truth census (Round 15 C1/C2). The mock keeps a real queue with
    // real decoys and repeats, and its crops differ between the two frames, so
    // the blink actually blinks in dev mode. A stub that returned one constant
    // image for both frames would make every link look correct -- which is the
    // failure the blink exists to expose.
    async start_census(kind, frame, decoyRate, repeatRate, seed, labeller, mode) {
      if (!mockState.trackNumbers.length) return { error: "Track the movie before starting a census." };
      if (kind === "frame_pair" && (frame === null || frame === undefined)) {
        return { error: "A frame_pair census needs a frame number." };
      }
      // The two Round 27 sets re-solve the frame pairs, so the real API refuses
      // them without an extraction; the mock refuses for the same reason rather
      // than silently succeeding where the product would not.
      if (!["geometry_flagged", "frame_pair", "environment_flagged",
            "forward_backward", "low_margin"].includes(kind)) {
        return { error: `Unknown census kind '${kind}'.` };
      }
      if (["forward_backward", "low_margin"].includes(kind) && !mockState.extracted) {
        return { error: "Extract the movie before building this set." };
      }
      const sessionMode = mode || "census";
      if (!["census", "correct"].includes(sessionMode)) {
        return { error: `Unknown census mode '${sessionMode}'; expected 'census' or 'correct'.` };
      }
      const real = kind === "frame_pair" ? 6
        : kind === "environment_flagged" ? 3
        : kind === "forward_backward" ? 5
        : kind === "low_margin" ? 7 : 4;
      const correcting = sessionMode === "correct";
      const queue = [];
      for (let i = 0; i < real; i += 1) {
        queue.push({ item_id: `i000${i}`, frame: mockState.frameNumbers[i % mockState.frameNumbers.length],
                     label: 10 + i, track_id: 100 + i, shown_label: 20 + i,
                     // Correction mode has NO decoys: accepting one would inject
                     // a deliberate error into the tracking table.
                     is_decoy: !correcting && i === 1, is_repeat: false, source_item_id: null });
      }
      if (!correcting) {
        queue.push({ ...queue[0], item_id: "r0000", source_item_id: "i0000", is_repeat: true, is_decoy: false });
      }
      mockState.census = { queue, kind, mode: sessionMode,
                           register: mockState.trackingCost.drift_mode !== "none",
                           labeller, edits: {}, undo: null };
      mockState.censusVerdicts = [];
      return { n: queue.length, n_real: real,
               n_decoys: queue.filter((e) => e.is_decoy).length,
               n_repeats: queue.filter((e) => e.is_repeat).length,
               kind, mode: sessionMode, register: mockState.census.register,
               // The mock reproduces the mismatch path: a session whose drift
               // setting changed after tracking must warn, because reading the
               // live setting to describe an existing run is how the status
               // line came to claim a convention the run had not used.
               tracking_drift_mode: mockState.trackingRunDriftMode ?? mockState.trackingCost.drift_mode,
               convention_from_run: mockState.trackingRunDriftMode != null,
               current_drift_setting: mockState.trackingCost.drift_mode,
               setting_changed_since_tracking:
                 mockState.trackingRunDriftMode != null
                 && mockState.trackingRunDriftMode !== mockState.trackingCost.drift_mode,
               session_id: "devmodecensus",
               path: correcting ? null : "(dev mode: nothing is written to disk)" };
    },
    // The two write paths refuse each other in the mock exactly as they do in
    // the backend. A mock that let either through would make dev-mode
    // click-through unable to reach the constraint this whole design rests on.
    async queue_census_edit(position, chosenLabel) {
      const census = mockState.census;
      if (!census) return { error: "Start a session first." };
      if (census.mode !== "correct") {
        return { error: "This is a census session; it cannot edit the tracking data. Its verdicts "
                        + "judge the run as it stands -- start a correction session instead." };
      }
      if (position < 0 || position >= census.queue.length) {
        return { error: `Position ${position} is outside the queue.` };
      }
      census.edits[position] = { chosen_label: chosenLabel === undefined ? null : chosenLabel };
      return { pending: Object.keys(census.edits).length, position };
    },
    async clear_census_edit(position) {
      const census = mockState.census;
      if (!census) return { error: "Start a session first." };
      delete census.edits[position];
      return { pending: Object.keys(census.edits).length };
    },
    async preview_census_edits() {
      const census = mockState.census;
      if (!census || census.mode !== "correct") return { error: "Not a correction session." };
      const pending = Object.keys(census.edits).length;
      if (!pending) return { error: "No pending corrections." };
      // A batch of 3 or more trips a check in dev mode, so the blocked path is
      // reachable without needing real data to misbehave.
      const messy = pending >= 3;
      return {
        pending,
        double_claimed: messy ? [{ frame: 3, cell_index: "Cell3_7", tracks: [100, 101] }] : [],
        truncated: [{ track_id: 101, lost_frames: [3], remaining: 5 }],
        orphaned: [],
        vanished_track_ids: [],
        evicts: messy ? [{ evicted_track_id: 101, new_track_id: 98, from_frame: 3 }] : [],
        takes_from: Object.entries(census.edits)
          .filter(([, e]) => e.chosen_label !== null)
          .map(([pos, e]) => ({ frame: Number(pos) + 1, track_id: 100, chosen_label: e.chosen_label })),
        blocking: messy,
      };
    },
    async apply_census_edits(force) {
      const census = mockState.census;
      if (!census) return { error: "Start a session first." };
      if (census.mode !== "correct") return { error: "A census session cannot apply corrections." };
      const preview = await mockApi.preview_census_edits();
      if (preview.error) return preview;
      if (preview.blocking && !force) {
        return { error: "This batch would double-claim a cell or leave a track holding nothing. "
                        + "Nothing was applied.", preview };
      }
      const applied = Object.keys(census.edits).length;
      census.edits = {};
      census.undo = true;
      return { applied, displaced_list: [], undo_available: true, preview };
    },
    async undo_census_edits() {
      const census = mockState.census;
      if (!census || !census.undo) return { error: "Nothing to undo." };
      census.undo = null;
      return { restored: true };
    },
    async get_candidate_settings() {
      return { mode: mockState.candidateMode || "radius", k: mockState.candidateK || 3,
               modes: ["radius", "ranked"] };
    },
    async set_candidate_settings(mode, k) {
      if (mode !== null && mode !== undefined && !["radius", "ranked"].includes(mode)) {
        return { error: `Unknown candidate mode '${mode}'; expected 'radius' or 'ranked'.` };
      }
      if (k !== null && k !== undefined && !(k > 0)) return { error: "The radius multiple k must be positive." };
      if (mode) mockState.candidateMode = mode;
      if (k) mockState.candidateK = k;
      return mockApi.get_candidate_settings();
    },
    // Render-only, like the real call: same labels, different tiles.
    async get_census_candidate_images(position, anchorView) {
      const item = await mockApi.get_census_item(position, 28, anchorView);
      if (item.error) return item;
      return { position, anchor_view: !!anchorView, labels: item.candidates.map((c) => c.label),
               images: item.candidates.map((c) => c.image) };
    },
    async get_census_item(position, _pad, anchorView = true) {
      const census = mockState.census;
      if (!census) return { error: "Start a census first." };
      if (position < 0 || position >= census.queue.length) {
        return { error: `Position ${position} is outside the queue (0-${census.queue.length - 1}).` };
      }
      const item = census.queue[position];
      // Two visibly different crops, so blink is testable in dev mode.
      const before = mockCrop(`f${item.frame}`, "#00ff88", 0);
      const after = mockCrop(`f${item.frame + 1}`, "#ffcc00", item.is_decoy ? 14 : 3);
      return {
        position, total: census.queue.length, item_id: item.item_id,
        frame: item.frame, label: item.label, track_id: item.track_id,
        shown_label: item.shown_label, is_repeat: item.is_repeat,
        reference_iou: 0.37, tracker_iou: 0.05,
        before, after,
        // How far each source was actually searched. The real API returns it
        // so a depth can be read from the product rather than only from a
        // script -- Round 22 reported a list length as if it were this.
        candidate_depth: (mockState.candidateMode || "radius") === "radius"
          ? { radius: 7 } : { reference: 1, overlap: 2, coverage: 1, nearest: 1, pointed: 1 },
        candidate_mode: (mockState.candidateMode || "radius") === "radius"
          ? `radius:k=${mockState.candidateK || 3}` : "ranked",
        anchor_view: !!anchorView,
        candidate_search: { mode: mockState.candidateMode || "radius", k: mockState.candidateK || 3,
                            radius_px: 39 * (mockState.candidateK || 3) / 3, cell_radius_px: 13,
                            drift_px: [6.1, 1.2], window: [0, 135, 0, 135], window_px: [135, 135] },
        // Five candidates, so dev mode exercises PAGINATION (2 pages, not a
        // padded 3) and the per-tile "why" labels. A three-candidate mock
        // would leave both unreachable.
        candidates: ((mockState.candidateMode || "radius") === "radius" ? [
          // Seven, so radius mode exercises THREE pages, and one of them is the
          // tracker's own answer -- the real radius list keeps and flags it.
          { label: 45, reason: "radius", why: `3.2 px from the anchor at frame ${item.frame}` },
          { label: item.shown_label, reason: "radius", is_tracker_answer: true,
            why: `9.8 px from the anchor at frame ${item.frame} (the tracker's answer)` },
          { label: 46, reason: "radius", why: `14.0 px from the anchor at frame ${item.frame}` },
          { label: 47, reason: "radius", why: `21.5 px from the anchor at frame ${item.frame}` },
          { label: 48, reason: "radius", why: `27.9 px from the anchor at frame ${item.frame}` },
          { label: 49, reason: "radius", why: `33.0 px from the anchor at frame ${item.frame}` },
          { label: 50, reason: "radius", why: `38.4 px from the anchor at frame ${item.frame}` },
        ] : [
          // Every basis spelled out the same way (Round 21 C3), and naming a
          // PLACE rather than "the cell" / "the link" (Round 23 C3): a tile
          // reading "12.4 px" next to one reading "18.1 px from the link"
          // cannot be compared, because the two are measured from different
          // origins and neither says which. `reason` and `rank` are what the
          // real API records with the verdict (Round 23 D), so the mock must
          // carry them or dev mode cannot exercise that path.
          { label: 40, reason: "reference", why: "the overlap reference's pick" },
          { label: 41, reason: "overlap", why: `IoU 0.42 with the anchor at frame ${item.frame}` },
          { label: 42, reason: "coverage", why: `covers 61% of the anchor at frame ${item.frame}` },
          { label: 43, reason: "nearest", why: `12.4 px from the anchor at frame ${item.frame}` },
          { label: 44, reason: "pointed", why: "18.1 px from the tracker's answer" },
        ]).map((c, i) => ({ ...c, rank: i + 1,
                            // Anchor view changes the TILE, never the label list.
                            image: mockCrop(`c${c.label}${anchorView ? "+a" : ""}`,
                                            "#ff6b6b", 4 + i * 3) })),
      };
    },
    async submit_census_verdict(position, verdict, chosenLabel, msElapsed) {
      const census = mockState.census;
      if (!census) return { error: "Start a census first." };
      if (census.mode === "correct") {
        return { error: "This is a correction session; it cannot write ground truth. Ground truth "
                        + "collected against a tracking run is not valid for a run it corrected -- "
                        + "start a census session instead." };
      }
      if (!["correct", "wrong", "unjudgeable"].includes(verdict)) {
        return { error: `unknown verdict '${verdict}'; expected one of correct, wrong, unjudgeable` };
      }
      const offered = (await mockApi.get_census_item(position)).candidates || [];
      const picked = offered.find((c) => c.label === chosenLabel);
      mockState.censusVerdicts.push({ position, verdict, chosenLabel, msElapsed,
                                      item: census.queue[position],
                                      chosen_reason: picked ? picked.reason : "",
                                      chosen_rank: picked ? picked.rank : null });
      return { saved: true, next: position + 1, total: census.queue.length,
               chosen_reason: picked ? picked.reason : "",
               chosen_rank: picked ? picked.rank : null,
               candidate_mode: (mockState.candidateMode || "radius") === "radius"
                 ? `radius:k=${mockState.candidateK || 3}` : "ranked" };
    },
    async get_census_summary() {
      const rows = mockState.censusVerdicts || [];
      if (!mockState.census) return { error: "Start a census first." };
      const real = rows.filter((r) => !r.item.is_decoy && !r.item.is_repeat);
      const decoys = rows.filter((r) => r.item.is_decoy);
      const repeats = rows.filter((r) => r.item.is_repeat);
      const counts = { correct: 0, wrong: 0, unjudgeable: 0 };
      real.forEach((r) => { counts[r.verdict] += 1; });
      const judged = counts.correct + counts.wrong;
      const caught = decoys.filter((r) => r.verdict === "wrong").length;
      let agree = 0, compared = 0;
      repeats.forEach((r) => {
        const original = rows.find((o) => o.item.item_id === r.item.source_item_id);
        if (original) { compared += 1; agree += original.verdict === r.verdict ? 1 : 0; }
      });
      const times = rows.map((r) => r.msElapsed).sort((a, b) => a - b);
      const median = times.length ? times[Math.floor(times.length / 2)] : null;
      return {
        path: "(dev mode: nothing is written to disk)",
        n_total: rows.length, n_real: real.length, counts,
        error_rate: judged ? counts.wrong / judged : null,
        n_unjudgeable: counts.unjudgeable,
        decoys_shown: decoys.length, decoys_caught: caught,
        decoy_pass_rate: decoys.length ? caught / decoys.length : null,
        repeats_compared: compared,
        intra_rater_agreement: compared ? agree / compared : null,
        median_ms: median, median_seconds: median === null ? null : Math.round(median / 100) / 10,
        queue_size: mockState.census.queue.length,
        judge_registered: mockState.census.register,
      };
    },
    // Coverage log (Round 15 C4). The mock keeps a real in-memory log rather
    // than returning a constant `{logged: true}`: a stub that always succeeds
    // has structurally hidden a real bug twice on this project, and a coverage
    // log that silently records nothing is invisible by construction. Dev mode
    // must be able to show an EMPTY summary and a wrong-frame summary too.
    async log_coverage_event(event, frame, cellIndex, trackId, msElapsed, zoomPercent,
                             renderMode, colorMode, detail) {
      if (!["frame_shown", "cell_inspected", "cell_changed"].includes(event)) {
        return { logged: false, reason: `unknown coverage event '${event}'` };
      }
      if (!mockState.outputDirsReady) {
        return { logged: false, reason: "no output directory yet" };
      }
      mockState.coverage.push({ event, frame, cellIndex, trackId, msElapsed,
                                zoomPercent, renderMode, colorMode, detail });
      return { logged: true, session_id: mockState.coverageSessionId };
    },
    async get_coverage_summary(minDwellMs) {
      const dwell = minDwellMs === null || minDwellMs === undefined ? 1500 : minDwellMs;
      const frames = {};
      const cellsPerFrame = 3;
      for (const frame of mockState.frameNumbers || []) {
        const ground = Array.from({ length: cellsPerFrame }, (_, i) => `Cell${frame}_${i + 1}`);
        const rows = mockState.coverage.filter((r) => r.frame === frame);
        const shown = rows.filter((r) => r.event === "frame_shown");
        const ms = shown.reduce((total, r) => total + (r.msElapsed || 0), 0);
        const changed = [...new Set(rows.filter((r) => r.event === "cell_changed")
                                        .map((r) => r.cellIndex))].filter((c) => ground.includes(c));
        const glimpsed = shown.length > 0 && ms < dwell;
        const inspected = glimpsed ? [] :
          [...new Set(rows.filter((r) => r.event === "cell_inspected").map((r) => r.cellIndex))]
            .filter((c) => ground.includes(c) && !changed.includes(c));
        frames[frame] = {
          displayed: shown.length > 0,
          dwell_ms: ms,
          glimpsed,
          zoom_percent: [...new Set(shown.map((r) => r.zoomPercent).filter((z) => z != null))],
          changed,
          inspected_unchanged: inspected,
          displayed_only: shown.length
            ? ground.filter((c) => !changed.includes(c) && !inspected.includes(c)) : [],
          never_displayed: shown.length ? [] : ground,
        };
      }
      const total = (key) => Object.values(frames).reduce((n, f) => n + f[key].length, 0);
      return {
        path: "(dev mode: nothing is written to disk)",
        events: mockState.coverage.length,
        sessions: mockState.coverage.length ? [mockState.coverageSessionId] : [],
        min_dwell_ms: dwell,
        frames,
        unknown_frames: [],
        totals: {
          changed: total("changed"), inspected_unchanged: total("inspected_unchanged"),
          displayed_only: total("displayed_only"), never_displayed: total("never_displayed"),
        },
        weak_positives: total("inspected_unchanged"),
      };
    },
    async reassign_track(_frame, _oldId, _newId, _propagate) {
      return { success: true, displaced: null };
    },
    async add_tracking_id(_frame, _newIndex, _cellIdStr) {
      return { success: true, displaced: null };
    },
    async swap_tracks(frame, _idA, _idB, propagate) {
      if (propagate) {
        return { success: true, swapped_frames: [frame, frame + 1], skipped_frames: [] };
      }
      return { success: true };
    },
    async suggest_match_candidates(frame, _cellIndex, _topK) {
      return { candidates: [
        { topology: "swap", s_current: 0.55, s_rival: 0.82, track_id: 2, cell_index: `Cell${frame - 1}_2`, distance: 4.2,
          status: "matched", current_cell: `Cell${frame}_7`, swap_gain: 12.5 },
        // A matched candidate where topology DISAGREES. The UI must show it
        // unchanged and undemoted -- a mock with only the confirming case
        // would hide the rule that matters most here.
        { topology: "keep", s_current: 0.88, s_rival: 0.41, track_id: 5, cell_index: `Cell${frame - 1}_5`,
          distance: 6.9, status: "matched", current_cell: `Cell${frame}_9`, swap_gain: 3.1 },
        { track_id: 3, cell_index: `Cell${frame - 1}_3`, distance: 11.7,
          status: "unmatched", current_cell: null, swap_gain: null, topology: "unknown" },
      ], topology_available: true };
    },
    async compute_propagation_stop_frame(frame, _oldId, _newId, _maxFramesAhead) {
      return { stop_frame: frame + 2, reason: "settled", unsettled_neighbors_by_frame: { [frame + 1]: [`Cell${frame + 1}_4`] } };
    },
    async apply_smart_stop_merge(_frame, _oldId, _newId, _endFrame) {
      return { success: true, displaced: null };
    },
    async compute_confirmed_identity_moves(frame, trackId, _maxFramesAhead, restrictToUnmatched) {
      const moves = [
        { frame: frame + 1, from_track_id: null, cell_index: `Cell${frame + 1}_5`, distance: 7.1 },
        { frame: frame + 2, from_track_id: null, cell_index: `Cell${frame + 2}_6`, distance: 30.7 },
      ];
      if (restrictToUnmatched === false) {
        moves.push({ frame: frame + 3, from_track_id: 12, cell_index: `Cell${frame + 3}_4`, distance: 25.0 });
      }
      return { stop_frame: frame + 3, reason: "settled", moves };
    },
    // Round 16 B. The mock returns a plan that FAILS two of the three checks
    // whenever the plan contains a take-from-another-track step, so dev-mode
    // click-through exercises the warning path and the blocked-Apply wording.
    // A mock that always came back clean would leave those branches unreached
    // -- the exact way this mock has hidden two real bugs before.
    async verify_confirmed_identity_moves(trackId, moves, gateMultiplier) {
      if (!moves || !moves.length) return { error: "This plan has no moves to verify." };
      const multiplier = gateMultiplier || 1.5;
      const radius = 22.0;
      const gate = multiplier * radius;
      const annotated = moves.map((m) => ({
        ...m,
        gate_px: Math.round(gate * 100) / 100,
        gate_fraction: m.distance == null ? null : Math.round((m.distance / gate) * 1000) / 1000,
      }));
      const takesFrom = moves.filter((m) => m.from_track_id !== null && m.from_track_id !== undefined);
      const messy = takesFrom.length > 0;
      return {
        ok: true,
        error: null,
        moves: annotated,
        double_claimed: messy
          ? [{ frame: takesFrom[0].frame, cell_index: takesFrom[0].cell_index, tracks: [trackId, 12] }]
          : [],
        takes_from: takesFrom.map((m) => ({ frame: m.frame, cell_index: m.cell_index,
                                            from_track_id: m.from_track_id, distance: m.distance })),
        truncated: messy ? [{ track_id: 12, lost_frames: [takesFrom[0].frame], remaining: 4 }] : [],
        orphaned: [],
        evicts: messy ? [{ evicted_track_id: 12, new_track_id: 99, from_frame: takesFrom[0].frame }] : [],
        vanished_track_ids: [],
        would_apply: true,
        blocking: messy,
        consistency: {
          before_unconfirmed: 6,
          after_unconfirmed: messy ? 9 : 4,
          delta: messy ? 3 : -2,
          n: 240,
          frames: moves.map((m) => m.frame - 1).concat(moves.map((m) => m.frame))
            .filter((v, i, a) => a.indexOf(v) === i).sort((a, b) => a - b),
          registered: mockState.registerReference,
          judge: "table_link_consistency",
          note: "Consistency under transposition -- NOT an accuracy.",
        },
      };
    },
    async apply_confirmed_identity_moves(_trackId, _moves) {
      return { success: true, displaced_list: [] };
    },
    async save_tracking() {
      return null;
    },
    async set_track_colors(_matchedColor, _unmatchedColor) {
      return null;
    },
    async set_dynamics_colors(_d, _x, _t) {
      return null;
    },
    async get_dynamics_candidates(frame) {
      // Mirrors the real shape, including an UNCONFIRMED candidate -- a mock
      // that only ever returns a happy path has hidden real bugs before.
      return {
        divisions: [{ type: "division", frame, mother: 4, daughters: [11, 12],
                      evidence: { area_ratio: 0.98, daughter_distances: [4.5, 5.1], daughters_adjacent: true } }],
        delaminations: [{ type: "delamination", frame, track: 7,
                          evidence: { area_slope_px_per_frame: -31.0, final_area: 90, final_area_vs_typical: 0.24 } }],
        // Cell 3's exchange settles; cell 9's reverses next frame. Both are
        // returned -- the transient one must render faded with a dashed axis,
        // and a mock that only shows the settled case would hide that path.
        t1: {
          counts: { 3: { gained: 1, lost: 0 }, 9: { gained: 1, lost: 0 } },
          counts_persistent: mockState.t1PersistFrames > 0
            ? { 3: { gained: 1, lost: 0 } }
            : { 3: { gained: 1, lost: 0 }, 9: { gained: 1, lost: 0 } },
          // Every axis carries `state`, and one of each of the three states is
          // present. The default filter is now `persistent` (Round 16 D), so a
          // mock with only settled axes would make the filter look like it
          // removes nothing -- exactly the thing the counts exist to show.
          axes: t1MockAxes(),
          gained: 2, lost: 1,
          gained_persistent: mockState.t1PersistFrames > 0 ? 1 : 2,
          lost_persistent: 1,
          gained_unjudgeable: mockState.t1PersistFrames > 1 ? 1 : 0,
          lost_unjudgeable: 0,
          persist_frames: mockState.t1PersistFrames,
          event_counts: t1EventCounts(t1MockAxes()),
        },
        confirmed: [], rejected: [], frame,
      };
    },
    async get_t1_events(frame, _direction) {
      return (await mockApi.get_dynamics_candidates(frame)).t1;
    },
    async get_registration_status() {
      return { register: true, applied: mockState.registerReference, median_drift_px: 6.97,
               median_cell_radius_px: 13.0, ratio: 0.536, threshold: 0.2,
               enabled: mockState.registerReference };
    },
    async set_register_reference(enabled) {
      mockState.registerReference = !!enabled;
      return mockApi.get_registration_status();
    },
    async get_environment_thresholds() {
      return { dead_band: mockState.deadBand, repair_threshold: mockState.repairThreshold };
    },
    async set_environment_thresholds(deadBand, repairThreshold) {
      if (deadBand !== null && deadBand !== undefined && !Number.isNaN(deadBand)) mockState.deadBand = deadBand;
      if (repairThreshold !== null && repairThreshold !== undefined && !Number.isNaN(repairThreshold)) {
        mockState.repairThreshold = repairThreshold;
      }
      return mockApi.get_environment_thresholds();
    },
    async get_environment_repair_candidates(_frame, _direction) {
      // `perfect` and `unscoreable` are part of the real answer (Round 28 C3)
      // and the mock carries them, or dev mode cannot exercise the line that
      // reports what the score is not about.
      return { candidates: [{ label: 3, score: 0.33 }, { label: 9, score: 0.5 }],
               threshold: mockState.repairThreshold, flagged_total: 4, n: 40,
               perfect: 22, unscoreable: 3 };
    },
    async get_link_agreement(frame) {
      // The mock must be able to produce the SILENT-FAILURE paths too, or it
      // hides exactly the class of bug this fix addresses. Frame 19 is the last
      // frame in `frameNumbers`, so it has no successor.
      if (frame >= 19) {
        return { states: {}, detail: {}, counts: {}, reason: "last_frame",
                 message: `Frame ${frame} is the last frame - there is no next frame to compare `
                          + `against, so no link can be judged here.` };
      }
      // Deliberately includes the unjudgeable and broken states: a mock that
      // only ever returns agrees/disagrees would hide the very case the
      // overlay exists to render honestly.
      // Both IoUs travel with each verdict; the disagreeing cell is a NEAR-TIE
      // (0.301 vs 0.231), which is what the real data mostly contains. A mock
      // showing a decisive disagreement would misrepresent the common case.
      return {
        states: { 1: "agrees", 2: "agrees", 3: "disagrees", 4: "unjudgeable", 5: "broken" },
        detail: {
          1: { state: "agrees", reference_iou: 0.72, tracker_iou: 0.72 },
          2: { state: "agrees", reference_iou: 0.66, tracker_iou: 0.66 },
          3: { state: "disagrees", reference_iou: 0.301, tracker_iou: 0.231 },
          4: { state: "unjudgeable", reference_iou: 0.12, tracker_iou: 0.09 },
          5: { state: "broken", reference_iou: 0.55, tracker_iou: null },
        },
        counts: { agrees: 2, disagrees: 1, unjudgeable: 1, broken: 1 },
        registered: mockState.registerReference,
      };
    },
    async set_agreement_colors(_a, _d, _u, _b) {
      return null;
    },
    async set_event_decision(key, decision) {
      return { confirmed: decision === "confirm" ? [key] : [], rejected: decision === "reject" ? [key] : [] };
    },
    async save_lineage() {
      return { path: "/mock/cell_lineage.xlsx", tracks: 414, divisions: 0, delaminations: 1 };
    },
    async set_lifecycle_colors(_lostColor, _newColor, _bothColor) {
      return null;
    },
    async get_track_numbers() {
      return mockState.trackNumbers;
    },
    async get_track_frame_range(_track) {
      return { min_frame: 1, max_frame: mockState.frameCount };
    },
    async get_track_cell_at_frame(track, frame) {
      // A gap on one track, so dev mode exercises the "no cell here" path the
      // real API takes when a track holds nothing at that frame.
      if (track === mockState.trackNumbers[1] && frame === 2) return null;
      return `Cell${frame}_${track}`;
    },
    async start_track_inspection(labeller) {
      if (!mockState.trackNumbers.length) return { error: "Track the movie before inspecting it." };
      mockState.inspection = { session_id: "insp0001", labeller: labeller || "", marks: 0, rows: [] };
      return { session_id: "insp0001", tracks: mockState.trackNumbers,
               total: mockState.trackNumbers.length, path: "/mock/GT_cell_track_output/track_inspection.csv" };
    },
    async mark_inspection_cell(track, frame, msElapsed) {
      if (!mockState.inspection) return { error: "Start a visual inspection first." };
      const cell = await mockApi.get_track_cell_at_frame(track, frame);
      if (cell === null) return { error: `Track #${track} holds no cell at frame ${frame}.` };
      mockState.inspection.marks += 1;
      mockState.inspection.rows.push({ track, frame, cell, msElapsed });
      return { marked: cell, track_id: track, frame, marks: mockState.inspection.marks };
    },
    async finish_track_inspection(inspected, perTrackMs) {
      if (!mockState.inspection) return { error: "Start a visual inspection first." };
      const seen = inspected || [];
      const marks = mockState.inspection.rows;
      const unseen = mockState.trackNumbers.filter((t) => !seen.includes(t));
      const dwell = Object.values(perTrackMs || {}).sort((a, b) => a - b);
      mockState.inspection = null;
      return {
        kind: "track_inspection", session_id: "insp0001",
        tracks_total: mockState.trackNumbers.length, tracks_inspected: seen.length,
        tracks_not_inspected: unseen, marks: marks.length,
        tracks_with_marks: new Set(marks.map((m) => m.track)).size,
        median_seconds_per_track: dwell.length
          ? Math.round(dwell[Math.floor(dwell.length / 2)] / 100) / 10 : null,
        marks_by_track: {}, path: "/mock/GT_cell_track_output/track_inspection.csv",
      };
    },
    async get_trajectory_overlay(_track, _frame, _renderMode) {
      return MOCK_PNG;
    },
  };

  window.EpiQuantAPI = new Proxy(
    {},
    {
      get(_target, prop) {
        return (...args) => {
          if (hasPywebview()) {
            return window.pywebview.api[prop](...args);
          }
          if (prop in mockApi) {
            console.warn(`[dev-mock] ${String(prop)}(${JSON.stringify(args)})`);
            return mockApi[prop](...args);
          }
          console.error(`[dev-mock] no mock implemented for api.${String(prop)}`);
          return Promise.resolve(null);
        };
      },
    }
  );

  window.usingMockApi = !hasPywebview();
})();
