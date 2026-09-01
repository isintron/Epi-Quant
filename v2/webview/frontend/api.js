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
    hasLeadingEdge: false,
    hasShape: false,
    trackingCost: { mode: "hybrid", overlap_weight: 1.0, centroid_weight: 1.0,
                    motion_compensation: false, max_frame_gap: 0, drift_mode: "none" },
    t1PersistFrames: 1,
    t1Enter: null, t1Exit: null, t1AxisFilter: "all",
    registerReference: true,
    deadBand: 0.75,
    repairThreshold: 0.5,
  };

  function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  const mockApi = {
    async select_folder(_title) {
      return "/mock/selected/folder";
    },
    async set_project(_maskFolder, _imageFolder, outputBase) {
      return {
        format: "npy",
        output_dirs: {
          cell_classification_output: outputBase + "/cell_classification_output",
          cell_track_output: outputBase + "/cell_track_output",
          quantitative_analysis_output: outputBase + "/quantitative_analysis_output",
        },
      };
    },
    async get_existing_results() {
      return { extraction: false, classification: false, tracking: false, shape: false, tissue: false };
    },
    async load_existing_extraction() {
      return { frame_count: mockState.frameCount, cell_count: 500 };
    },
    async load_existing_classification() {
      return { cluster_sizes: { cluster_1: 120, cluster_2: 30 }, k_used: 2 };
    },
    async load_existing_tracking() {
      return { track_count: mockState.trackNumbers.length, duplicates_cleaned: 0 };
    },
    async start_extraction(_maskType, _bgRatio, _excludeEdge) {
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
        label, columns, default_off: [], base: false, exclusive: false, note: "",
        // Mirrors pca_kmeans.FEATURE_DISPLAY_NAMES: cosmetic only, the real
        // column name stays the checkbox value.
        display: Object.fromEntries(columns.map((c) => [c, ({
          "Cell Area": "Area",
          "Cell Perimeter": "Perimeter",
          "Ellipse Major/Minor Axis Ratio": "Ellipse Axis Ratio",
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
             "Maximum Horizontal Length", "Maximum Vertical Length", "Horizontal/Vertical Length Ratio"],
            { base: true, ready: true, missing: [],
              default_off: ["Maximum Horizontal Length", "Maximum Vertical Length", "Horizontal/Vertical Length Ratio"],
              note: "H/V lengths measure orientation, not shape - tick them only if your specimen is axis-aligned." }),
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
               exit_px: mockState.t1Exit, axis_filter: mockState.t1AxisFilter };
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
    async set_t1_axis_filter(mode) {
      mockState.t1AxisFilter = mode;
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
        window.onTrackDone && window.onTrackDone({ track_count: mockState.trackNumbers.length });
      })();
      return true;
    },
    async get_tracking_overlay(_frame, _renderMode, _showLabels, _colorMode, _identityPalette, _labelFontSize,
                               _environmentLut, _environmentDirection, _tintOthers) {
      return MOCK_PNG;
    },
    async get_tracking_cell_at_pixel(frame, _x, _y) {
      return { cell_index: `Cell${frame}_1`, cell_id: 1, track_id: 1 };
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
        { track_id: 2, cell_index: `Cell${frame - 1}_2`, distance: 4.2,
          status: "matched", current_cell: `Cell${frame}_7`, swap_gain: 12.5 },
        { track_id: 3, cell_index: `Cell${frame - 1}_3`, distance: 11.7,
          status: "unmatched", current_cell: null, swap_gain: null },
      ] };
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
          axes: [
            { center: [20, 20], phi: 0.4, type: "gain", persistent: true },
            { center: [40, 30], phi: 1.2, type: "loss", persistent: mockState.t1PersistFrames === 0 },
          ],
          gained: 2, lost: 1,
          gained_persistent: mockState.t1PersistFrames > 0 ? 1 : 2,
          lost_persistent: 1,
          gained_unjudgeable: mockState.t1PersistFrames > 1 ? 1 : 0,
          lost_unjudgeable: 0,
          persist_frames: mockState.t1PersistFrames,
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
      return { candidates: [{ label: 3, score: 0.33 }, { label: 9, score: 0.5 }],
               threshold: mockState.repairThreshold, flagged_total: 4, n: 40 };
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
