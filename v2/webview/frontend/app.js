// app.js — step navigation, state, image rendering, click handlers.
// Talks to the backend only through window.EpiQuantAPI (see api.js), never
// window.pywebview.api directly, so the dev-mode mock is transparent here.

const state = {
  maskFolder: null,
  imageFolder: null,
  outputFolder: null,
  frameCount: 0,

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

  trackingModeOptions: { has_leading_edge: false, split_available: false },
  trackNumbers: [],
  currentTrackNumber: null,
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
    goToStep("extract");
  } catch (err) {
    setStatus("setup-status", `Error: ${err}`, "error");
  }
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
  state.extractFrame = 1;
  setStatus("extract-status", `Done: ${data.frame_count} frames, ${data.cell_count} cells extracted.`, "success");
  document.getElementById("start-extraction-btn").disabled = false;
  document.getElementById("extract-preview").style.display = "block";
  document.getElementById("extract-next-choices").style.display = "flex";
  enableStep("classify");
  await showExtractFrame(1);
}

function onExtractError(data) {
  setStatus("extract-status", `Extraction failed: ${data.message}`, "error");
  document.getElementById("start-extraction-btn").disabled = false;
}

async function showExtractFrame(frame) {
  state.extractFrame = frame;
  document.getElementById("extract-frame-label").textContent = `Frame ${frame} / ${state.frameCount}`;
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
    setStatus(
      "classify-status",
      `Done: k=${result.k_used}. Sizes: ${JSON.stringify(result.cluster_sizes)}`,
      "success"
    );
    document.getElementById("classify-results").style.display = "block";

    await populatePalettePresets();
    await applyPaletteToAllClusters();
    await refreshClusterLegend();

    state.classifyFrame = 1;
    await showClassifyFrame(1);

    const canBoundary = await window.EpiQuantAPI.can_compute_leading_edge();
    document.getElementById("compute-leading-edge-btn").style.display = canBoundary ? "inline-block" : "none";

    enableStep("correct-classification");
    enableStep("track");
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

    row.append(colorInput, label, count);
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
  document.getElementById("classify-frame-label").textContent = `Frame ${frame} / ${state.frameCount}`;
  const img = await window.EpiQuantAPI.get_classification_overlay(frame);
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
  document.getElementById("correct-class-frame-label").textContent = `Frame ${frame} / ${state.frameCount}`;
  const img = await window.EpiQuantAPI.get_classification_overlay(frame);
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
  setStatus("track-status", `Done: ${data.track_count} tracks.`, "success");
  document.getElementById("start-tracking-btn").disabled = false;
  document.getElementById("track-next-choices").style.display = "flex";
  enableStep("correct-tracking");
  enableStep("trajectory");

  state.trackNumbers = await window.EpiQuantAPI.get_track_numbers();
  const select = document.getElementById("track-number-select");
  select.innerHTML = "";
  state.trackNumbers.forEach((tn) => {
    const opt = document.createElement("option");
    opt.value = tn;
    opt.textContent = `Track ${tn}`;
    select.appendChild(opt);
  });

  state.correctTrackFrame = 1;
  await showCorrectTrackFrame(1);
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
  document.getElementById("correct-track-frame-label").textContent = `Frame ${frame} / ${state.frameCount}`;
  const img = await window.EpiQuantAPI.get_tracking_overlay(frame);
  if (img) document.getElementById("correct-track-overlay-img").src = img;
  document.getElementById("correct-track-selected-cell").textContent = "";
  document.getElementById("correct-track-reassign").style.display = "none";
  state.selectedTrackCell = null;
}

async function handleCorrectTrackImageClick(event) {
  const img = event.target;
  const { x, y } = imagePixelFromClick(img, event);
  const result = await window.EpiQuantAPI.get_tracking_cell_at_pixel(state.correctTrackFrame, x, y);
  if (!result) {
    document.getElementById("correct-track-selected-cell").textContent = "No cell at that position.";
    state.selectedTrackCell = null;
    document.getElementById("correct-track-reassign").style.display = "none";
    return;
  }
  state.selectedTrackCell = result;
  const isMatched = result.track_id !== null && result.track_id !== undefined;
  document.getElementById("correct-track-selected-cell").textContent = isMatched
    ? `${result.cell_index}: currently track #${result.track_id}.`
    : `${result.cell_index}: unmatched (raw cell id ${result.cell_id}).`;
  document.getElementById("correct-track-reassign").style.display = "flex";
  document.getElementById("new-track-id-input").value = "";
}

async function handleApplyTrackCorrection() {
  if (!state.selectedTrackCell) return;
  const newId = parseInt(document.getElementById("new-track-id-input").value, 10);
  if (Number.isNaN(newId)) {
    setStatus("correct-track-selected-cell", "Enter a valid track number.", "error");
    return;
  }

  const isMatched = state.selectedTrackCell.track_id !== null && state.selectedTrackCell.track_id !== undefined;
  let result;
  if (isMatched) {
    result = await window.EpiQuantAPI.reassign_track(state.correctTrackFrame, state.selectedTrackCell.track_id, newId);
  } else {
    result = await window.EpiQuantAPI.add_tracking_id(state.correctTrackFrame, newId, state.selectedTrackCell.cell_index);
  }

  if (result && result.error) {
    setStatus("correct-track-selected-cell", `Error: ${result.error}`, "error");
  } else {
    setStatus("correct-track-selected-cell", "Applied.", "success");
    await showCorrectTrackFrame(state.correctTrackFrame);
  }
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
  const select = document.getElementById("track-number-select");
  if (select.options.length > 0 && state.currentTrackNumber === null) {
    await handleTrackNumberChange();
  }
}

async function handleTrackNumberChange() {
  const tn = parseInt(document.getElementById("track-number-select").value, 10);
  state.currentTrackNumber = tn;
  const range = await window.EpiQuantAPI.get_track_frame_range(tn);
  if (!range) {
    setStatus("trajectory-status", "No frame data for this track.", "error");
    return;
  }
  state.trajectoryFrame = range.min_frame;
  await showTrajectoryFrame(range.min_frame);
}

async function showTrajectoryFrame(frame) {
  state.trajectoryFrame = frame;
  document.getElementById("trajectory-frame-label").textContent = `Frame ${frame}`;
  const img = await window.EpiQuantAPI.get_trajectory_overlay(state.currentTrackNumber, frame);
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

function wireFrameNav(prevId, nextId, getFrame, setFrame, showFn) {
  document.getElementById(prevId).addEventListener("click", () => {
    const f = Math.max(1, getFrame() - 1);
    setFrame(f);
    showFn(f);
  });
  document.getElementById(nextId).addEventListener("click", () => {
    const f = Math.min(state.frameCount || Infinity, getFrame() + 1);
    setFrame(f);
    showFn(f);
  });
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

  // Extract
  document.getElementById("start-extraction-btn").addEventListener("click", handleStartExtraction);
  document.getElementById("extract-preview-img").addEventListener("click", () => {}); // raw preview, no click action
  wireFrameNav("extract-prev-frame", "extract-next-frame", () => state.extractFrame, (f) => (state.extractFrame = f), showExtractFrame);
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
  wireFrameNav("classify-prev-frame", "classify-next-frame", () => state.classifyFrame, (f) => (state.classifyFrame = f), showClassifyFrame);
  document.getElementById("classify-overlay-img").addEventListener("click", handleClassifyImageClick);
  document.getElementById("go-to-correct-classification").addEventListener("click", async () => {
    goToStep("correct-classification");
    await showCorrectClassFrame(state.classifyFrame);
  });
  document.getElementById("compute-leading-edge-btn").addEventListener("click", handleComputeLeadingEdge);
  document.getElementById("go-to-track-from-classify").addEventListener("click", async () => {
    goToStep("track");
    await enterTrackStep();
  });

  // Correct Classification
  wireFrameNav(
    "correct-class-prev-frame", "correct-class-next-frame",
    () => state.correctClassFrame, (f) => (state.correctClassFrame = f), showCorrectClassFrame
  );
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
  wireFrameNav(
    "correct-track-prev-frame", "correct-track-next-frame",
    () => state.correctTrackFrame, (f) => (state.correctTrackFrame = f), showCorrectTrackFrame
  );
  document.getElementById("correct-track-overlay-img").addEventListener("click", handleCorrectTrackImageClick);
  document.getElementById("apply-track-correction-btn").addEventListener("click", handleApplyTrackCorrection);
  document.getElementById("save-tracking-btn").addEventListener("click", handleSaveTrackingCorrections);
  document.getElementById("done-correcting-tracking").addEventListener("click", enterTrajectoryStep);

  // Trajectory
  document.getElementById("track-number-select").addEventListener("change", handleTrackNumberChange);
  wireFrameNav(
    "trajectory-prev-frame", "trajectory-next-frame",
    () => state.trajectoryFrame, (f) => (state.trajectoryFrame = f), showTrajectoryFrame
  );

  // Global callbacks pushed from Python via window.evaluate_js
  window.onProgress = onProgress;
  window.onExtractDone = onExtractDone;
  window.onExtractError = onExtractError;
  window.onTrackDone = onTrackDone;
  window.onTrackError = onTrackError;
}

document.addEventListener("DOMContentLoaded", init);
