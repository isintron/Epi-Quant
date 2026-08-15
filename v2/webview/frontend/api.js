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
      return { extraction: false, classification: false, tracking: false };
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
    async run_classification(_n, _k, _rs) {
      return { cluster_sizes: { cluster_1: 120, cluster_2: 30 }, k_used: 2 };
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
        { track_id: 2, cell_index: `Cell${frame - 1}_2`, distance: 4.2 },
        { track_id: 3, cell_index: `Cell${frame - 1}_3`, distance: 11.7 },
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
