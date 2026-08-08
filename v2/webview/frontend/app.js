// app.js — step navigation, state, image rendering, click handlers.
// Talks to the backend only through window.EpiQuantAPI (see api.js), never
// window.pywebview.api directly, so the dev-mode mock is transparent here.

const state = {
  maskFolder: null,
  imageFolder: null,
  outputFolder: null,
  frameCount: 0,
  // Actual sorted frame numbers with data -- NOT necessarily 1..frameCount
  // contiguous (some datasets are unrelated samples, each keeping whatever
  // number is embedded in its own filename, e.g. [1, 9, 19]). Frame-nav
  // must step through this list rather than assume frame+1 is valid.
  frameNumbers: [],

  // Per-step "currently displayed frame" -- kept separate since a user can
  // jump between steps and each should remember its own position.
  extractFrame: 1,
  classifyFrame: 1,
  correctClassFrame: 1,
  correctTrackFrame: 1,
  trajectoryFrame: 1,

  clusterLegend: [],
  selectedCellForCorrection: null, // {cell_index, cluster_label, cell_type}
  selectedTrackCell: null, // {cell_index, cell_id, track_id}
  trackSwapSelection: [], // up to 2 {track_id, cell_index} picked while swap mode is on, same frame only
  smartStopPending: null, // {oldId, newId, stopFrame} once a stop frame has been computed, awaiting Apply

  trackingModeOptions: { has_leading_edge: false, split_available: false },
  trackNumbers: [],
  currentTrackNumber: null,

  existingResults: { extraction: false, classification: false, tracking: false },

  playIntervals: {}, // prefix -> setInterval id, for frame autoplay
  zoomLevels: {}, // prefix -> current zoom percent
};

const enabledSteps = new Set(["setup"]);

// ---------------------------------------------------------------------
// Step navigation
// ---------------------------------------------------------------------

function enableStep(step) {
  enabledSteps.add(step);
  const btn = document.querySelector(`.step-btn[data-step="${step}"]`);
  if (btn) btn.disabled = false;
}

function goToStep(step) {
  if (!enabledSteps.has(step)) return;
  document.querySelectorAll(".step-section").forEach((el) => el.classList.remove("active"));
  document.querySelectorAll(".step-btn").forEach((el) => el.classList.remove("active"));
  document.getElementById(`step-${step}`).classList.add("active");
  document.querySelector(`.step-btn[data-step="${step}"]`).classList.add("active");
}

function setStatus(elId, message, kind) {
  const el = document.getElementById(elId);
  el.textContent = message;
  el.className = "status-line" + (kind ? " " + kind : "");
}

// ---------------------------------------------------------------------
// Image click -> pixel coordinates (natural image size, not CSS size --
// see v2/rendering.py's cell_id_at_pixel docstring for why this matters)
// ---------------------------------------------------------------------

function imagePixelFromClick(imgEl, event) {
  const rect = imgEl.getBoundingClientRect();
  const scaleX = imgEl.naturalWidth / rect.width;
  const scaleY = imgEl.naturalHeight / rect.height;
  return {
    x: Math.round((event.clientX - rect.left) * scaleX),
    y: Math.round((event.clientY - rect.top) * scaleY),
  };
}

// ---------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------

async function browseInto(inputId, title) {
  const path = await window.EpiQuantAPI.select_folder(title);
  if (path) document.getElementById(inputId).value = path;
}

async function handleSetProject() {
  const maskFolder = document.getElementById("mask-folder-input").value.trim();
  const imageFolder = document.getElementById("image-folder-input").value.trim();
  const outputFolder = document.getElementById("output-folder-input").value.trim();

  if (!maskFolder || !outputFolder) {
    setStatus("setup-status", "Masks folder and output folder are required.", "error");
    return;
  }

  setStatus("setup-status", "Detecting mask format…");
  try {
    const result = await window.EpiQuantAPI.set_project(maskFolder, imageFolder, outputFolder);
    state.maskFolder = maskFolder;
    state.imageFolder = imageFolder;
    state.outputFolder = outputFolder;
    setStatus("setup-status", `Detected format: ${result.format}. Output folders created.`, "success");
    enableStep("extract");

    const existing = await window.EpiQuantAPI.get_existing_results();
    state.existingResults = existing;
    if (existing.extraction || existing.classification || existing.tracking) {
      const parts = ["extraction", "classification", "tracking"].filter((k) => existing[k]);
      document.getElementById("existing-results-summary").textContent =
        `Found existing results for: ${parts.join(", ")}.`;
      document.getElementById("existing-results-prompt").style.display = "flex";
    } else {
      goToStep("extract");
    }
  } catch (err) {
    setStatus("setup-status", `Error: ${err}`, "error");
  }
}

async function handleLoadExistingResults() {
  const existing = state.existingResults;
  document.getElementById("load-existing-btn").disabled = true;
  setStatus("setup-status", "Loading existing results…");
  try {
    let furthestStep = "extract";
    if (existing.extraction) {
      const data = await window.EpiQuantAPI.load_existing_extraction();
      await onExtractDone(data);
      furthestStep = "classify";
    }
    if (existing.classification) {
      const result = await window.EpiQuantAPI.load_existing_classification();
      await afterClassificationReady(result);
      furthestStep = "correct-classification";
    }
    if (existing.tracking) {
      const data = await window.EpiQuantAPI.load_existing_tracking();
      await onTrackDone(data);
      furthestStep = "trajectory";
    }
    document.getElementById("existing-results-prompt").style.display = "none";
    setStatus("setup-status", "Loaded existing results.", "success");
    goToStep(furthestStep);
  } catch (err) {
    setStatus("setup-status", `Error loading existing results: ${err}`, "error");
  } finally {
    document.getElementById("load-existing-btn").disabled = false;
  }
}

function handleStartFresh() {
  document.getElementById("existing-results-prompt").style.display = "none";
  goToStep("extract");
}

// ---------------------------------------------------------------------
// Extract
// ---------------------------------------------------------------------

async function handleStartExtraction() {
  const excludeEdge = document.getElementById("exclude-edge-cells").checked;
  const maskType = document.getElementById("mask-type-select").value;
  const bgRatio = parseFloat(document.getElementById("background-area-ratio-input").value) || 10.0;

  document.getElementById("start-extraction-btn").disabled = true;
  document.getElementById("extract-progress-bar").style.display = "block";
  setStatus("extract-status", "Starting extraction…");

  await window.EpiQuantAPI.start_extraction(maskType, bgRatio, excludeEdge);
}

function onProgress(data) {
  const barIdByStage = { extract: "extract-progress-bar", track: "track-progress-bar" };
  const statusIdByStage = { extract: "extract-status", track: "track-status", leading_edge: "leading-edge-status" };

  const barId = barIdByStage[data.stage];
  if (barId) {
    const bar = document.getElementById(barId);
    bar.style.display = "block";
    bar.querySelector(".progress-fill").style.width = `${data.percent}%`;
  }
  const statusId = statusIdByStage[data.stage];
  if (statusId) setStatus(statusId, `${data.message} (${data.percent}%)`);
}

async function onExtractDone(data) {
  state.frameCount = data.frame_count;
  state.frameNumbers = await window.EpiQuantAPI.get_frame_numbers();
  state.extractFrame = state.frameNumbers[0] ?? 1;
  setStatus("extract-status", `Done: ${data.frame_count} frames, ${data.cell_count} cells extracted.`, "success");
  document.getElementById("start-extraction-btn").disabled = false;
  document.getElementById("extract-preview").style.display = "block";
  document.getElementById("extract-next-choices").style.display = "flex";
  enableStep("classify");
  await showExtractFrame(state.extractFrame);
}

function onExtractError(data) {
  setStatus("extract-status", `Extraction failed: ${data.message}`, "error");
  document.getElementById("start-extraction-btn").disabled = false;
}

function frameLabel(frame) {
  const idx = state.frameNumbers.indexOf(frame);
  return idx === -1 ? `Frame ${frame}` : `Frame ${frame} (${idx + 1} / ${state.frameNumbers.length})`;
}

// `${prefix}-render-mode` radio group -- "mask" (filled regions, default) or
// "outline" (boundary-line only, closer to v1's look, keeps touching
// same-cluster cells visually distinguishable).
function getRenderMode(prefix) {
  const checked = document.querySelector(`input[name="${prefix}-render-mode"]:checked`);
  return checked ? checked.value : "mask";
}

function wireRenderMode(prefix, onChange) {
  document.querySelectorAll(`input[name="${prefix}-render-mode"]`).forEach((el) => {
    el.addEventListener("change", onChange);
  });
}

// `${prefix}-color-mode` radio group -- "uniform" (matched/unmatched colors,
// default) or "identity" (each track its own viridis color, rank-normalized).
function getColorMode(prefix) {
  const checked = document.querySelector(`input[name="${prefix}-color-mode"]:checked`);
  return checked ? checked.value : "uniform";
}

function wireColorMode(prefix, onChange) {
  document.querySelectorAll(`input[name="${prefix}-color-mode"]`).forEach((el) => {
    el.addEventListener("change", onChange);
  });
}

async function showExtractFrame(frame) {
  state.extractFrame = frame;
  document.getElementById("extract-frame-label").textContent = frameLabel(frame);
  const img = await window.EpiQuantAPI.get_raw_frame_image(frame);
  if (img) document.getElementById("extract-preview-img").src = img;
}

// ---------------------------------------------------------------------
// Classify
// ---------------------------------------------------------------------

function updateNClustersModeUI() {
  const isAuto = document.getElementById("n-clusters-auto").checked;
  document.getElementById("k-range-min").disabled = !isAuto;
  document.getElementById("k-range-max").disabled = !isAuto;
  document.getElementById("n-clusters-input").disabled = isAuto;
}

async function afterClassificationReady(result) {
  setStatus(
    "classify-status",
    `Done: k=${result.k_used}. Sizes: ${JSON.stringify(result.cluster_sizes)}`,
    "success"
  );
  document.getElementById("classify-results").style.display = "block";

  await populatePalettePresets();
  await applyPaletteToAllClusters();
  await refreshClusterLegend();

  state.classifyFrame = state.frameNumbers[0] ?? 1;
  await showClassifyFrame(state.classifyFrame);

  const canBoundary = await window.EpiQuantAPI.can_compute_leading_edge();
  document.getElementById("compute-leading-edge-btn").style.display = canBoundary ? "inline-block" : "none";

  enableStep("correct-classification");
  enableStep("track");
}

async function handleRunClassification() {
  const isAuto = document.getElementById("n-clusters-auto").checked;
  const nClusters = isAuto ? "auto" : parseInt(document.getElementById("n-clusters-input").value, 10);
  const kRange = [
    parseInt(document.getElementById("k-range-min").value, 10),
    parseInt(document.getElementById("k-range-max").value, 10),
  ];

  setStatus("classify-status", "Running PCA + K-means…");
  document.getElementById("run-classification-btn").disabled = true;
  try {
    const result = await window.EpiQuantAPI.run_classification(nClusters, kRange, null);
    await afterClassificationReady(result);
  } catch (err) {
    setStatus("classify-status", `Error: ${err}`, "error");
  } finally {
    document.getElementById("run-classification-btn").disabled = false;
  }
}

async function populatePalettePresets() {
  const presets = await window.EpiQuantAPI.get_palette_presets();
  state.palettePresets = presets;
  const select = document.getElementById("palette-preset-select");
  select.innerHTML = "";
  Object.keys(presets).forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    select.appendChild(opt);
  });
}

async function applyPaletteToAllClusters() {
  const presetName = document.getElementById("palette-preset-select").value;
  const palette = (state.palettePresets && state.palettePresets[presetName]) || [];
  const legend = await window.EpiQuantAPI.get_cluster_legend();
  const colorMap = {};
  legend.forEach((entry, i) => {
    colorMap[entry.cluster_label] = palette[i % Math.max(palette.length, 1)] || "#808080";
  });
  await window.EpiQuantAPI.set_cluster_colors(colorMap);
}

async function refreshClusterLegend() {
  const legend = await window.EpiQuantAPI.get_cluster_legend();
  state.clusterLegend = legend;
  const container = document.getElementById("cluster-legend");
  container.innerHTML = "";
  legend.forEach((entry) => {
    const row = document.createElement("div");
    row.className = "cluster-legend-row";

    const visibleCheckbox = document.createElement("input");
    visibleCheckbox.type = "checkbox";
    visibleCheckbox.checked = !entry.hidden;
    visibleCheckbox.title = "Show this cluster";
    visibleCheckbox.addEventListener("change", async (e) => {
      const hiddenLabels = state.clusterLegend.filter((c) => c.hidden).map((c) => c.cluster_label);
      const hiddenSet = new Set(hiddenLabels);
      if (e.target.checked) hiddenSet.delete(entry.cluster_label);
      else hiddenSet.add(entry.cluster_label);
      await window.EpiQuantAPI.set_hidden_clusters([...hiddenSet]);
      await refreshClusterLegend();
      await showClassifyFrame(state.classifyFrame);
    });

    const colorInput = document.createElement("input");
    colorInput.type = "color";
    colorInput.value = entry.color;
    colorInput.addEventListener("change", async (e) => {
      await window.EpiQuantAPI.set_cluster_colors({ [entry.cluster_label]: e.target.value });
      await refreshClusterLegend();
      await showClassifyFrame(state.classifyFrame);
    });

    const label = document.createElement("span");
    label.textContent = `${entry.cell_type} (label ${entry.cluster_label})`;

    const count = document.createElement("span");
    count.className = "count";
    count.textContent = `${entry.count} cells`;

    row.append(visibleCheckbox, colorInput, label, count);
    container.appendChild(row);
  });
  renderClusterPicker();
}

function renderClusterPicker() {
  const picker = document.getElementById("correct-class-cluster-picker");
  picker.innerHTML = "";
  state.clusterLegend.forEach((entry) => {
    const btn = document.createElement("button");
    btn.className = "secondary-btn";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = entry.color;
    btn.appendChild(swatch);
    btn.appendChild(document.createTextNode(entry.cell_type));
    btn.addEventListener("click", () => applyClusterReassignment(entry.cluster_label));
    picker.appendChild(btn);
  });
}

async function showClassifyFrame(frame) {
  state.classifyFrame = frame;
  document.getElementById("classify-frame-label").textContent = frameLabel(frame);
  const img = await window.EpiQuantAPI.get_classification_overlay(frame, getRenderMode("classify"));
  if (img) document.getElementById("classify-overlay-img").src = img;
}

async function handleClassifyImageClick(event) {
  const img = event.target;
  const { x, y } = imagePixelFromClick(img, event);
  const result = await window.EpiQuantAPI.get_cell_at_pixel(state.classifyFrame, x, y);
  document.getElementById("classify-cell-info").textContent = result
    ? `${result.cell_index}: ${result.cell_type || "(unclassified)"}`
    : "No cell at that position.";
}

async function handleComputeLeadingEdge() {
  setStatus("leading-edge-status", "Computing leading edge for all frames…");
  document.getElementById("compute-leading-edge-btn").disabled = true;
  try {
    const result = await window.EpiQuantAPI.compute_leading_edge_all_frames(3);
    await window.EpiQuantAPI.save_cells_info_leading_edge();
    setStatus("leading-edge-status", `Done: ${result.frames_processed} frames processed and saved.`, "success");
  } catch (err) {
    setStatus("leading-edge-status", `Error: ${err}`, "error");
  } finally {
    document.getElementById("compute-leading-edge-btn").disabled = false;
  }
}

// ---------------------------------------------------------------------
// Correct Classification
// ---------------------------------------------------------------------

async function showCorrectClassFrame(frame) {
  state.correctClassFrame = frame;
  document.getElementById("correct-class-frame-label").textContent = frameLabel(frame);
  const img = await window.EpiQuantAPI.get_classification_overlay(frame, getRenderMode("correct-class"));
  if (img) document.getElementById("correct-class-overlay-img").src = img;
  document.getElementById("correct-class-selected-cell").textContent = "";
  state.selectedCellForCorrection = null;
}

async function handleCorrectClassImageClick(event) {
  const img = event.target;
  const { x, y } = imagePixelFromClick(img, event);
  const result = await window.EpiQuantAPI.get_cell_at_pixel(state.correctClassFrame, x, y);
  if (!result) {
    document.getElementById("correct-class-selected-cell").textContent = "No cell at that position.";
    state.selectedCellForCorrection = null;
    return;
  }
  state.selectedCellForCorrection = result;
  document.getElementById("correct-class-selected-cell").textContent =
    `Selected ${result.cell_index} (currently ${result.cell_type || "unclassified"}). Pick a new cluster below.`;
}

async function applyClusterReassignment(newLabel) {
  if (!state.selectedCellForCorrection) return;
  const legend = await window.EpiQuantAPI.reassign_cluster(state.selectedCellForCorrection.cell_index, newLabel);
  state.clusterLegend = legend;
  document.getElementById("correct-class-selected-cell").textContent =
    `Reassigned ${state.selectedCellForCorrection.cell_index} -> cluster ${newLabel}.`;
  await refreshClusterLegend();
  await showCorrectClassFrame(state.correctClassFrame);
}

async function handleSaveClassificationCorrections() {
  await window.EpiQuantAPI.save_classification();
  setStatus("correct-class-selected-cell", "Saved.", "success");
}

// ---------------------------------------------------------------------
// Track
// ---------------------------------------------------------------------

async function enterTrackStep() {
  const options = await window.EpiQuantAPI.get_tracking_mode_options();
  state.trackingModeOptions = options;
  const container = document.getElementById("track-mode-options");
  container.innerHTML = "";

  if (!options.has_leading_edge) {
    const note = document.createElement("p");
    note.className = "hint";
    note.textContent = options.split_available
      ? ""
      : "Tracking will treat all cells as one population" +
        (state.clusterLegend.length > 2 ? " (N>2 clusters: split-by-boundary tracking is only available for exactly 2 clusters this round)." : ".");
    container.appendChild(note);
    state.selectedTrackingMode = "single";
  } else {
    const wrap = document.createElement("div");
    wrap.innerHTML = `
      <label class="radio-label"><input type="radio" name="track-mode" value="single" checked> Track as one population</label>
      <label class="radio-label"><input type="radio" name="track-mode" value="split"> Track split by leading-edge boundary (2 populations)</label>
    `;
    container.appendChild(wrap);
    state.selectedTrackingMode = "single";
    wrap.querySelectorAll('input[name="track-mode"]').forEach((el) => {
      el.addEventListener("change", (e) => { state.selectedTrackingMode = e.target.value; });
    });
  }
}

async function handleStartTracking() {
  document.getElementById("start-tracking-btn").disabled = true;
  document.getElementById("track-progress-bar").style.display = "block";
  setStatus("track-status", "Starting tracking…");
  await window.EpiQuantAPI.start_tracking(state.selectedTrackingMode || "single");
}

async function onTrackDone(data) {
  const dupNote = data.duplicates_cleaned ? ` (cleaned ${data.duplicates_cleaned} duplicate track claims from this file)` : "";
  setStatus("track-status", `Done: ${data.track_count} tracks.${dupNote}`, "success");
  document.getElementById("start-tracking-btn").disabled = false;
  document.getElementById("track-next-choices").style.display = "flex";
  enableStep("correct-tracking");
  enableStep("trajectory");

  state.trackNumbers = await window.EpiQuantAPI.get_track_numbers();
  const list = document.getElementById("track-number-list");
  list.innerHTML = "";
  state.trackNumbers.forEach((tn) => {
    const opt = document.createElement("option");
    opt.value = tn;
    list.appendChild(opt);
  });

  state.correctTrackFrame = state.frameNumbers[0] ?? 1;
  await showCorrectTrackFrame(state.correctTrackFrame);
}

function onTrackError(data) {
  setStatus("track-status", `Tracking failed: ${data.message}`, "error");
  document.getElementById("start-tracking-btn").disabled = false;
}

// ---------------------------------------------------------------------
// Correct Tracking
// ---------------------------------------------------------------------

async function showCorrectTrackFrame(frame) {
  state.correctTrackFrame = frame;
  document.getElementById("correct-track-frame-label").textContent = frameLabel(frame);
  const showLabels = document.getElementById("correct-track-show-labels").checked;
  const identityPalette = document.getElementById("correct-track-identity-palette-select").value;
  const img = await window.EpiQuantAPI.get_tracking_overlay(
    frame, getRenderMode("correct-track"), showLabels, getColorMode("correct-track"), identityPalette
  );
  if (img) document.getElementById("correct-track-overlay-img").src = img;
  document.getElementById("correct-track-selected-cell").textContent = "";
  document.getElementById("correct-track-reassign").style.display = "none";
  document.getElementById("correct-track-suggest").style.display = "none";
  document.getElementById("correct-track-suggest-list").style.display = "none";
  resetSmartStopPanel();
  state.selectedTrackCell = null;
  resetTrackSwapSelection();
}

function resetSmartStopPanel() {
  state.smartStopPending = null;
  document.getElementById("correct-track-smart-stop").style.display = "none";
  document.getElementById("smart-stop-result").textContent = "";
  document.getElementById("apply-smart-stop-btn").style.display = "none";
}

function resetTrackSwapSelection() {
  state.trackSwapSelection = [];
  document.getElementById("swap-track-a-input").value = "";
  document.getElementById("swap-track-b-input").value = "";
  document.getElementById("correct-track-swap-status").textContent = "";
}

function handleSwapCellClick(result) {
  if (!result || result.track_id === null || result.track_id === undefined) {
    setStatus("correct-track-swap-status", "Click a matched cell to swap -- unmatched cells can't be swapped.", "error");
    return;
  }
  if (state.trackSwapSelection.some((c) => c.track_id === result.track_id)) {
    setStatus("correct-track-swap-status", `Track #${result.track_id} is already selected.`, "error");
    return;
  }
  state.trackSwapSelection.push(result);
  if (state.trackSwapSelection.length > 2) state.trackSwapSelection.shift();

  const [a, b] = state.trackSwapSelection;
  document.getElementById("swap-track-a-input").value = a ? a.track_id : "";
  document.getElementById("swap-track-b-input").value = b ? b.track_id : "";
  setStatus(
    "correct-track-swap-status",
    b ? `Ready to swap track #${a.track_id} and track #${b.track_id}.` : `Selected track #${a.track_id} -- click a second matched cell.`,
    "success"
  );
}

async function handleApplyTrackSwap() {
  const idA = parseInt(document.getElementById("swap-track-a-input").value, 10);
  const idB = parseInt(document.getElementById("swap-track-b-input").value, 10);
  if (Number.isNaN(idA) || Number.isNaN(idB)) {
    setStatus("correct-track-swap-status", "Enter two valid track numbers.", "error");
    return;
  }
  const result = await window.EpiQuantAPI.swap_tracks(state.correctTrackFrame, idA, idB);
  if (result && result.error) {
    setStatus("correct-track-swap-status", `Error: ${result.error}`, "error");
    return;
  }
  await showCorrectTrackFrame(state.correctTrackFrame);
  setStatus("correct-track-swap-status", "Swapped.", "success");
}

async function handleCorrectTrackImageClick(event) {
  const img = event.target;
  const { x, y } = imagePixelFromClick(img, event);
  const result = await window.EpiQuantAPI.get_tracking_cell_at_pixel(state.correctTrackFrame, x, y);
  if (document.getElementById("track-swap-mode-checkbox").checked) {
    handleSwapCellClick(result);
    return;
  }
  if (!result) {
    document.getElementById("correct-track-selected-cell").textContent = "No cell at that position.";
    state.selectedTrackCell = null;
    document.getElementById("correct-track-reassign").style.display = "none";
    document.getElementById("correct-track-suggest").style.display = "none";
    document.getElementById("correct-track-suggest-list").style.display = "none";
    resetSmartStopPanel();
    return;
  }
  state.selectedTrackCell = result;
  const isMatched = result.track_id !== null && result.track_id !== undefined;
  document.getElementById("correct-track-selected-cell").textContent = isMatched
    ? `${result.cell_index}: currently track #${result.track_id}.`
    : `${result.cell_index}: unmatched (raw cell id ${result.cell_id}).`;
  document.getElementById("correct-track-reassign").style.display = "flex";
  document.getElementById("new-track-id-input").value = "";
  document.getElementById("propagate-correction-checkbox").checked = false;

  resetSmartStopPanel();
  document.getElementById("correct-track-smart-stop").style.display = isMatched ? "flex" : "none";

  document.getElementById("correct-track-suggest").style.display = "flex";
  document.getElementById("correct-track-suggest-list").style.display = "none";
  document.getElementById("correct-track-suggest-list").innerHTML = "";
  document.getElementById("correct-track-suggest-hint").textContent = isMatched
    ? ""
    : "Select a matched cell to find its likely continuation.";
  document.getElementById("suggest-match-btn").disabled = !isMatched;
}

async function handleApplyTrackCorrection() {
  if (!state.selectedTrackCell) return;
  const newId = parseInt(document.getElementById("new-track-id-input").value, 10);
  if (Number.isNaN(newId)) {
    setStatus("correct-track-selected-cell", "Enter a valid track number.", "error");
    return;
  }

  const isMatched = state.selectedTrackCell.track_id !== null && state.selectedTrackCell.track_id !== undefined;
  const correctedCellIndex = state.selectedTrackCell.cell_index;
  let result;
  if (isMatched) {
    const propagate = document.getElementById("propagate-correction-checkbox").checked;
    result = await window.EpiQuantAPI.reassign_track(
      state.correctTrackFrame, state.selectedTrackCell.track_id, newId, propagate
    );
  } else {
    result = await window.EpiQuantAPI.add_tracking_id(state.correctTrackFrame, newId, correctedCellIndex);
  }

  if (result && result.error) {
    setStatus("correct-track-selected-cell", `Error: ${result.error}`, "error");
  } else {
    setStatus("correct-track-selected-cell", "Applied.", "success");
    await showCorrectTrackFrame(state.correctTrackFrame);
  }
}

async function handleComputeStopFrame() {
  if (!state.selectedTrackCell) return;
  const oldId = state.selectedTrackCell.track_id;
  const newId = parseInt(document.getElementById("new-track-id-input").value, 10);
  if (Number.isNaN(newId)) {
    setStatus("smart-stop-result", "Enter a target track # above first.", "error");
    return;
  }

  const result = await window.EpiQuantAPI.compute_propagation_stop_frame(state.correctTrackFrame, oldId, newId, 20);
  if (result && result.error) {
    setStatus("smart-stop-result", `Error: ${result.error}`, "error");
    document.getElementById("apply-smart-stop-btn").style.display = "none";
    return;
  }
  state.smartStopPending = { oldId, newId, stopFrame: result.stop_frame };
  const trailNote = Object.keys(result.unsettled_neighbors_by_frame || {}).length
    ? ` Unsettled neighbors by frame: ${JSON.stringify(result.unsettled_neighbors_by_frame)}`
    : "";
  setStatus("smart-stop-result", `Suggested stop frame: ${result.stop_frame} (${result.reason}).${trailNote}`, "success");
  const applyBtn = document.getElementById("apply-smart-stop-btn");
  applyBtn.textContent = `Merge through frame ${result.stop_frame}`;
  applyBtn.style.display = "inline-block";
}

async function handleApplySmartStop() {
  if (!state.smartStopPending) return;
  const { oldId, newId, stopFrame } = state.smartStopPending;
  const result = await window.EpiQuantAPI.apply_smart_stop_merge(state.correctTrackFrame, oldId, newId, stopFrame);
  if (result && result.error) {
    setStatus("smart-stop-result", `Error: ${result.error}`, "error");
    return;
  }
  await showCorrectTrackFrame(state.correctTrackFrame);
  setStatus("smart-stop-result", `Merged through frame ${stopFrame}.`, "success");
}

async function handleSuggestMatch() {
  if (!state.selectedTrackCell || state.selectedTrackCell.track_id === null || state.selectedTrackCell.track_id === undefined) return;
  const result = await window.EpiQuantAPI.suggest_match_candidates(state.correctTrackFrame, state.selectedTrackCell.cell_index, 3);
  const list = document.getElementById("correct-track-suggest-list");
  list.innerHTML = "";
  list.style.display = "block";

  if (result && result.error) {
    setStatus("correct-track-suggest-list", `Error: ${result.error}`, "error");
    return;
  }
  const candidates = (result && result.candidates) || [];
  if (candidates.length === 0) {
    list.textContent = "No unmatched candidates found in the next frame.";
    return;
  }
  const trackId = state.selectedTrackCell.track_id;
  candidates.forEach((candidate) => {
    const item = document.createElement("li");
    item.textContent = `${candidate.cell_index} (${candidate.distance.toFixed(1)}px) `;
    const useBtn = document.createElement("button");
    useBtn.className = "secondary-btn";
    useBtn.textContent = "Use this";
    useBtn.addEventListener("click", async () => {
      const addResult = await window.EpiQuantAPI.add_tracking_id(state.correctTrackFrame + 1, trackId, candidate.cell_index);
      if (addResult && addResult.error) {
        setStatus("correct-track-suggest-list", `Error: ${addResult.error}`, "error");
      } else {
        setStatus("correct-track-selected-cell", `Attached ${candidate.cell_index} to track #${trackId} at frame ${state.correctTrackFrame + 1}.`, "success");
        list.style.display = "none";
        list.innerHTML = "";
      }
    });
    item.appendChild(useBtn);
    list.appendChild(item);
  });
}

async function handleSaveTrackingCorrections() {
  await window.EpiQuantAPI.save_tracking();
  setStatus("correct-track-selected-cell", "Saved.", "success");
}

// ---------------------------------------------------------------------
// Trajectory
// ---------------------------------------------------------------------

async function enterTrajectoryStep() {
  goToStep("trajectory");
  if (state.trackNumbers.length > 0 && state.currentTrackNumber === null) {
    await loadTrack(state.trackNumbers[0]);
  }
}

async function loadTrack(tn) {
  if (!state.trackNumbers.includes(tn)) {
    setStatus("trajectory-status", `Track ${tn} not found.`, "error");
    return;
  }
  state.currentTrackNumber = tn;
  document.getElementById("track-number-input").value = tn;
  const range = await window.EpiQuantAPI.get_track_frame_range(tn);
  if (!range) {
    setStatus("trajectory-status", "No frame data for this track.", "error");
    return;
  }
  state.trajectoryFrame = range.min_frame;
  await showTrajectoryFrame(range.min_frame);
}

async function handleTrackNumberInputChange() {
  const tn = parseInt(document.getElementById("track-number-input").value, 10);
  if (Number.isNaN(tn)) return;
  await loadTrack(tn);
}

function handlePrevTrack() {
  const idx = state.trackNumbers.indexOf(state.currentTrackNumber);
  if (idx > 0) loadTrack(state.trackNumbers[idx - 1]);
}

function handleNextTrack() {
  const idx = state.trackNumbers.indexOf(state.currentTrackNumber);
  if (idx !== -1 && idx < state.trackNumbers.length - 1) loadTrack(state.trackNumbers[idx + 1]);
}

async function showTrajectoryFrame(frame) {
  state.trajectoryFrame = frame;
  document.getElementById("trajectory-frame-label").textContent = `Frame ${frame}`;
  const img = await window.EpiQuantAPI.get_trajectory_overlay(state.currentTrackNumber, frame, getRenderMode("trajectory"));
  if (img) {
    document.getElementById("trajectory-overlay-img").src = img;
    setStatus("trajectory-status", "");
  } else {
    setStatus("trajectory-status", "This track has no cell in this frame (gap).", "error");
  }
}

// ---------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------

function stopAutoplay(prefix) {
  const intervalId = state.playIntervals[prefix];
  if (intervalId) {
    clearInterval(intervalId);
    state.playIntervals[prefix] = null;
  }
  const playBtn = document.getElementById(`${prefix}-play-pause`);
  if (playBtn) playBtn.innerHTML = "&#9654; Play";
}

function startAutoplay(prefix, stepFn) {
  const fpsInput = document.getElementById(`${prefix}-fps-input`);
  const fps = Math.max(0.5, parseFloat(fpsInput.value) || 2);
  state.playIntervals[prefix] = setInterval(stepFn, 1000 / fps);
  document.getElementById(`${prefix}-play-pause`).innerHTML = "&#10074;&#10074; Pause";
}

// Index of `frame` within `state.frameNumbers` (the actual, possibly sparse,
// frame numbers with data) -- -1 falls back to 0 rather than propagating.
function frameIndex(frame) {
  const idx = state.frameNumbers.indexOf(frame);
  return idx === -1 ? 0 : idx;
}

// `prefix` derives every control's id (`${prefix}-first-frame`, `-prev-frame`,
// `-play-pause`, `-fps-input`, `-next-frame`, `-last-frame`) -- every frame-nav
// in index.html already follows this naming convention. Steps through
// `state.frameNumbers` by index rather than `frame +/- 1` -- some datasets'
// frame numbers are sparse (e.g. `[1, 9, 19]`, see `get_frame_numbers`
// docstring), so `frame + 1` is not reliably a valid next frame.
function wireFrameNav(prefix, getFrame, setFrame, showFn) {
  const goFirst = () => {
    stopAutoplay(prefix);
    const f = state.frameNumbers[0];
    if (f === undefined) return;
    setFrame(f);
    showFn(f);
  };
  const goPrev = () => {
    const idx = Math.max(0, frameIndex(getFrame()) - 1);
    const f = state.frameNumbers[idx];
    if (f === undefined) return;
    setFrame(f);
    showFn(f);
  };
  const goNext = () => {
    const idx = Math.min(state.frameNumbers.length - 1, frameIndex(getFrame()) + 1);
    const f = state.frameNumbers[idx];
    if (f === undefined) return;
    setFrame(f);
    showFn(f);
    if (idx >= state.frameNumbers.length - 1) stopAutoplay(prefix);
  };
  const goLast = () => {
    stopAutoplay(prefix);
    const f = state.frameNumbers[state.frameNumbers.length - 1];
    if (f === undefined) return;
    setFrame(f);
    showFn(f);
  };

  document.getElementById(`${prefix}-first-frame`).addEventListener("click", goFirst);
  document.getElementById(`${prefix}-prev-frame`).addEventListener("click", () => { stopAutoplay(prefix); goPrev(); });
  document.getElementById(`${prefix}-next-frame`).addEventListener("click", () => { stopAutoplay(prefix); goNext(); });
  document.getElementById(`${prefix}-last-frame`).addEventListener("click", goLast);

  document.getElementById(`${prefix}-play-pause`).addEventListener("click", () => {
    if (state.playIntervals[prefix]) {
      stopAutoplay(prefix);
    } else {
      startAutoplay(prefix, goNext);
    }
  });
  document.getElementById(`${prefix}-fps-input`).addEventListener("change", () => {
    if (state.playIntervals[prefix]) {
      stopAutoplay(prefix);
      startAutoplay(prefix, goNext);
    }
  });
}

// `prefix` derives `${prefix}-zoom-in/out/reset/level/slider`; `imgId` is
// the `<img>` being zoomed. 100% keeps today's default fit-to-container
// look (CSS max-width:100%); any other level sets an explicit pixel width
// off the image's natural size AND clears max-width (a large image's
// zoomed width would otherwise still get clamped back down to the
// container by the CSS max-width:100% rule -- that was the original bug),
// so `.image-click-wrap`'s overflow:auto scrolls it. Persists across frame
// navigation for free since a new frame's `src` doesn't reset the
// element's inline `style.width`. Also wires click-drag panning inside the
// wrap: a real drag (movement past a small threshold) scrolls the wrap and
// swallows the resulting `click` so it doesn't misfire the section's
// click-to-select-cell handler; a plain click (no movement) passes through
// untouched.
function wireZoom(prefix, imgId) {
  const img = document.getElementById(imgId);
  const levelEl = document.getElementById(`${prefix}-zoom-level`);
  const slider = document.getElementById(`${prefix}-zoom-slider`);
  const wrap = img.closest(".image-click-wrap");
  state.zoomLevels[prefix] = 100;

  const setZoom = (percent) => {
    percent = Math.max(25, Math.min(400, percent));
    state.zoomLevels[prefix] = percent;
    if (percent === 100) {
      img.style.width = "";
      img.style.maxWidth = "";
    } else {
      img.style.maxWidth = "none";
      if (img.naturalWidth) img.style.width = `${Math.round(img.naturalWidth * percent / 100)}px`;
    }
    levelEl.textContent = `${percent}%`;
    slider.value = percent;
  };

  document.getElementById(`${prefix}-zoom-in`).addEventListener("click", () => setZoom(state.zoomLevels[prefix] + 25));
  document.getElementById(`${prefix}-zoom-out`).addEventListener("click", () => setZoom(state.zoomLevels[prefix] - 25));
  document.getElementById(`${prefix}-zoom-reset`).addEventListener("click", () => setZoom(100));
  slider.addEventListener("input", () => setZoom(parseInt(slider.value, 10)));
  slider.addEventListener("wheel", (e) => e.stopPropagation()); // let the slider keep its own native wheel-drag; only the image below should scroll frames

  let dragging = false;
  let dragged = false;
  let startX = 0, startY = 0, startScrollLeft = 0, startScrollTop = 0;

  img.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    dragging = true;
    dragged = false;
    startX = e.clientX;
    startY = e.clientY;
    startScrollLeft = wrap.scrollLeft;
    startScrollTop = wrap.scrollTop;
    e.preventDefault(); // suppress the browser's native image-drag ghost
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) dragged = true;
    wrap.scrollLeft = startScrollLeft - dx;
    wrap.scrollTop = startScrollTop - dy;
  });
  window.addEventListener("mouseup", () => { dragging = false; });
  img.addEventListener("click", (e) => {
    if (dragged) {
      e.stopImmediatePropagation();
      e.preventDefault();
      dragged = false;
    }
  }, true);
}

// Mouse-wheel-over-the-image steps through frames (scroll down = next,
// scroll up = prev), reusing the same image `wireZoom` already wires up
// rather than adding a separate control. Debounced to one frame per ~120ms
// so a single wheel notch/trackpad flick doesn't skip several frames --
// mice fire one coarse deltaY per notch, but trackpads fire many small
// deltas per gesture. `preventDefault()` stops the browser's native
// scroll-the-overflowing-container behavior that would otherwise fire here
// once zoomed in; panning while zoomed still works via click-drag.
function wireFrameWheel(prefix, imgId) {
  const img = document.getElementById(imgId);
  let cooling = false;
  img.addEventListener("wheel", (e) => {
    e.preventDefault();
    if (cooling) return;
    cooling = true;
    setTimeout(() => { cooling = false; }, 120);
    const btnId = e.deltaY > 0 ? `${prefix}-next-frame` : `${prefix}-prev-frame`;
    const btn = document.getElementById(btnId);
    if (btn) btn.click();
  }, { passive: false });
}

function init() {
  if (window.usingMockApi) {
    document.getElementById("mock-banner").style.display = "block";
  }

  document.querySelectorAll(".step-btn").forEach((btn) => {
    btn.addEventListener("click", () => goToStep(btn.dataset.step));
  });
  goToStep("setup");

  // Setup
  document.getElementById("browse-mask-folder").addEventListener("click", () => browseInto("mask-folder-input", "Select masks folder"));
  document.getElementById("browse-image-folder").addEventListener("click", () => browseInto("image-folder-input", "Select images folder"));
  document.getElementById("browse-output-folder").addEventListener("click", () => browseInto("output-folder-input", "Select output folder"));
  document.getElementById("set-project-btn").addEventListener("click", handleSetProject);
  document.getElementById("load-existing-btn").addEventListener("click", handleLoadExistingResults);
  document.getElementById("start-fresh-btn").addEventListener("click", handleStartFresh);

  // Extract
  document.getElementById("start-extraction-btn").addEventListener("click", handleStartExtraction);
  document.getElementById("extract-preview-img").addEventListener("click", () => {}); // raw preview, no click action
  wireFrameNav("extract", () => state.extractFrame, (f) => (state.extractFrame = f), showExtractFrame);
  wireZoom("extract", "extract-preview-img");
  wireFrameWheel("extract", "extract-preview-img");
  document.getElementById("go-to-classify").addEventListener("click", () => goToStep("classify"));
  document.getElementById("skip-classification").addEventListener("click", async () => {
    enableStep("track");
    goToStep("track");
    await enterTrackStep();
  });

  // Classify
  document.getElementById("n-clusters-auto").addEventListener("change", updateNClustersModeUI);
  document.getElementById("n-clusters-manual").addEventListener("change", updateNClustersModeUI);
  updateNClustersModeUI();
  document.getElementById("run-classification-btn").addEventListener("click", handleRunClassification);
  document.getElementById("palette-preset-select").addEventListener("change", async () => {
    await applyPaletteToAllClusters();
    await refreshClusterLegend();
    await showClassifyFrame(state.classifyFrame);
  });
  document.getElementById("save-classification-btn").addEventListener("click", async () => {
    await window.EpiQuantAPI.save_classification();
    setStatus("classify-status", "Saved.", "success");
  });
  wireFrameNav("classify", () => state.classifyFrame, (f) => (state.classifyFrame = f), showClassifyFrame);
  wireZoom("classify", "classify-overlay-img");
  wireFrameWheel("classify", "classify-overlay-img");
  wireRenderMode("classify", () => showClassifyFrame(state.classifyFrame));
  document.getElementById("classify-overlay-img").addEventListener("click", handleClassifyImageClick);
  document.getElementById("go-to-correct-classification").addEventListener("click", async () => {
    goToStep("correct-classification");
    await showCorrectClassFrame(state.classifyFrame);
  });
  document.getElementById("compute-leading-edge-btn").addEventListener("click", handleComputeLeadingEdge);
  document.getElementById("show-cluster-borders-toggle").addEventListener("change", async (e) => {
    await window.EpiQuantAPI.set_show_cluster_borders(e.target.checked);
    await showClassifyFrame(state.classifyFrame);
  });
  document.getElementById("border-color-input").addEventListener("change", async (e) => {
    await window.EpiQuantAPI.set_border_color(e.target.value);
    await showClassifyFrame(state.classifyFrame);
  });
  document.getElementById("go-to-track-from-classify").addEventListener("click", async () => {
    goToStep("track");
    await enterTrackStep();
  });

  // Correct Classification
  wireFrameNav("correct-class", () => state.correctClassFrame, (f) => (state.correctClassFrame = f), showCorrectClassFrame);
  wireZoom("correct-class", "correct-class-overlay-img");
  wireFrameWheel("correct-class", "correct-class-overlay-img");
  wireRenderMode("correct-class", () => showCorrectClassFrame(state.correctClassFrame));
  document.getElementById("correct-class-overlay-img").addEventListener("click", handleCorrectClassImageClick);
  document.getElementById("save-correction-classification-btn").addEventListener("click", handleSaveClassificationCorrections);
  document.getElementById("done-correcting-classification").addEventListener("click", async () => {
    goToStep("track");
    await enterTrackStep();
  });

  // Track
  document.getElementById("start-tracking-btn").addEventListener("click", handleStartTracking);
  document.getElementById("go-to-correct-tracking").addEventListener("click", () => goToStep("correct-tracking"));
  document.getElementById("go-to-trajectory").addEventListener("click", enterTrajectoryStep);

  // Correct Tracking
  wireFrameNav("correct-track", () => state.correctTrackFrame, (f) => (state.correctTrackFrame = f), showCorrectTrackFrame);
  wireZoom("correct-track", "correct-track-overlay-img");
  wireFrameWheel("correct-track", "correct-track-overlay-img");
  wireRenderMode("correct-track", () => showCorrectTrackFrame(state.correctTrackFrame));
  document.getElementById("correct-track-show-labels").addEventListener("change", () => showCorrectTrackFrame(state.correctTrackFrame));
  wireColorMode("correct-track", () => {
    const isIdentity = getColorMode("correct-track") === "identity";
    document.getElementById("correct-track-uniform-colors").style.display = isIdentity ? "none" : "flex";
    document.getElementById("correct-track-identity-palette").style.display = isIdentity ? "flex" : "none";
    showCorrectTrackFrame(state.correctTrackFrame);
  });
  document.getElementById("correct-track-identity-palette-select").addEventListener("change", () => showCorrectTrackFrame(state.correctTrackFrame));
  document.getElementById("correct-track-overlay-img").addEventListener("click", handleCorrectTrackImageClick);
  document.getElementById("apply-track-correction-btn").addEventListener("click", handleApplyTrackCorrection);
  document.getElementById("suggest-match-btn").addEventListener("click", handleSuggestMatch);
  document.getElementById("compute-stop-frame-btn").addEventListener("click", handleComputeStopFrame);
  document.getElementById("apply-smart-stop-btn").addEventListener("click", handleApplySmartStop);
  document.getElementById("track-swap-mode-checkbox").addEventListener("change", resetTrackSwapSelection);
  document.getElementById("apply-track-swap-btn").addEventListener("click", handleApplyTrackSwap);
  document.getElementById("save-tracking-btn").addEventListener("click", handleSaveTrackingCorrections);
  document.getElementById("done-correcting-tracking").addEventListener("click", enterTrajectoryStep);
  document.getElementById("matched-color-input").addEventListener("change", async (e) => {
    await window.EpiQuantAPI.set_track_colors(e.target.value, document.getElementById("unmatched-color-input").value);
    await showCorrectTrackFrame(state.correctTrackFrame);
  });
  document.getElementById("unmatched-color-input").addEventListener("change", async (e) => {
    await window.EpiQuantAPI.set_track_colors(document.getElementById("matched-color-input").value, e.target.value);
    await showCorrectTrackFrame(state.correctTrackFrame);
  });

  // Trajectory
  document.getElementById("track-number-input").addEventListener("change", handleTrackNumberInputChange);
  document.getElementById("trajectory-prev-track").addEventListener("click", handlePrevTrack);
  document.getElementById("trajectory-next-track").addEventListener("click", handleNextTrack);
  wireFrameNav("trajectory", () => state.trajectoryFrame, (f) => (state.trajectoryFrame = f), showTrajectoryFrame);
  wireZoom("trajectory", "trajectory-overlay-img");
  wireFrameWheel("trajectory", "trajectory-overlay-img");
  wireRenderMode("trajectory", () => showTrajectoryFrame(state.trajectoryFrame));

  // Global callbacks pushed from Python via window.evaluate_js
  window.onProgress = onProgress;
  window.onExtractDone = onExtractDone;
  window.onExtractError = onExtractError;
  window.onTrackDone = onTrackDone;
  window.onTrackError = onTrackError;
}

document.addEventListener("DOMContentLoaded", init);
