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
  // Coverage log (Round 15 C4): when the currently-shown Correct Tracking
  // frame went up, so the dwell can be reported when it comes down. Null
  // means no frame is being timed -- leaving the step, or never entered.
  correctTrackShownAt: null,
  correctTrackShownFrame: null,
  trajectoryFrame: 1,

  clusterLegend: [],
  // Cached get_feature_options() result, refreshed on every entry to Classify
  // and after shape extraction.
  featureOptions: null,
  selectedCellForCorrection: null, // {cell_index, cluster_label, cell_type} -- last-clicked cell, informational only
  activeReassignCluster: null, // cluster_label picked as the reassignment target; persists across clicks/frames until toggled off
  selectedTrackCell: null, // {cell_index, cell_id, track_id}
  trackSwapSelection: [], // up to 2 {track_id, cell_index} picked while swap mode is on, same frame only
  smartStopPending: null, // {oldId, newId, stopFrame} once a stop frame has been computed, awaiting Apply
  suggestCheckTrackId: null, // track id awaiting a "Check how far this holds" diagnostic, after a Suggest-match apply
  confirmedMovesPending: null, // {trackId, moves} once a moves plan has been computed, awaiting Apply

  trackingModeOptions: { has_leading_edge: false, split_available: false },
  trackNumbers: [],
  currentTrackNumber: null,

  existingResults: { extraction: false, classification: false, tracking: false },

  // The most recent image rendered in ANY step, so the clipboard button still
  // has something to copy on steps that have no preview of their own.
  lastViewDataUri: null,
  playIntervals: {}, // prefix -> setInterval id, for frame autoplay
  zoomLevels: {}, // prefix -> current zoom percent
};

const enabledSteps = new Set(["setup"]);

// ---------------------------------------------------------------------
// Step navigation
// ---------------------------------------------------------------------

// data-step -> the <img> that step displays. Steps 1 (setup) and 5 (track)
// are parameter-only and have no preview, which is why the clipboard button
// falls back to the last view rendered anywhere.
const STEP_PREVIEW_IMG = {
  extract: "extract-preview-img",
  classify: "classify-overlay-img",
  "correct-classification": "correct-class-overlay-img",
  "correct-tracking": "correct-track-overlay-img",
  trajectory: "trajectory-overlay-img",
};

/** The single writer of displayed image content. Pass `null` to clear. */
function setPreviewImage(elementId, dataUri) {
  const el = document.getElementById(elementId);
  if (!el) return;
  if (dataUri) {
    el.src = dataUri;
    state.lastViewDataUri = dataUri;
    document.getElementById("copy-view-btn").disabled = false;
  } else {
    el.removeAttribute("src");
  }
}

function currentStep() {
  // goToStep records the active step only in the DOM -- there is no
  // state.currentStep to read.
  const active = document.querySelector(".step-btn.active");
  return active ? active.dataset.step : null;
}

/** The image to copy: this step's own if it has one, else the last rendered. */
function currentViewDataUri() {
  const imgId = STEP_PREVIEW_IMG[currentStep()];
  if (imgId) {
    const el = document.getElementById(imgId);
    if (el && el.getAttribute("src")) return el.getAttribute("src");
  }
  return state.lastViewDataUri;
}

async function handleCopyView() {
  const dataUri = currentViewDataUri();
  if (!dataUri) {
    setStatus("copy-view-status", "Nothing has been rendered yet.", "error");
    return;
  }
  const blob = await (await fetch(dataUri)).blob();

  // The page is served from http://127.0.0.1:<port>/ (pywebview auto-starts
  // its own server), which is a secure context, so the async clipboard API is
  // available. It has still never been exercised in the real native window,
  // hence the Python fallback below rather than a bare failure.
  try {
    await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    flashCopyResult("Copied the current view to the clipboard.");
    return;
  } catch (err) {
    console.warn("navigator.clipboard failed, falling back to the Python bridge:", err);
  }

  const result = await window.EpiQuantAPI.copy_image_to_clipboard(dataUri);
  if (result && result.success) {
    flashCopyResult("Copied to the clipboard.");
  } else {
    flashCopyResult(`Copy failed: ${(result && result.message) || "clipboard unavailable"}`, true);
  }
}

function flashCopyResult(message, isError) {
  const btn = document.getElementById("copy-view-btn");
  const original = btn.textContent;
  btn.textContent = isError ? "⧉ Failed" : "⧉ Copied";
  btn.title = message;
  setTimeout(() => { btn.textContent = original; }, 1600);
}

function enableStep(step) {
  enabledSteps.add(step);
  const btn = document.querySelector(`.step-btn[data-step="${step}"]`);
  if (btn) btn.disabled = false;
}

function goToStep(step) {
  if (!enabledSteps.has(step)) return;
  // Close the open dwell before navigating away. A dwell that is never closed
  // reads as "never displayed" in the coverage summary -- the exact confusion
  // the log exists to remove.
  if (step !== "correct-tracking") flushCorrectTrackDwell();
  if (step === "classify") populateFeatureOptions();
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

// Collision-guard notices (see manual_correction_tracking.py's
// _evict_future_if_colliding): never silently absorbed, always appended to
// whatever success message the caller is already showing.
function describeDisplacement(displaced) {
  if (!displaced) return "";
  return ` (Track #${displaced.evicted_track_id} already had different data from frame ${displaced.from_frame} on -- moved it to new track #${displaced.new_track_id}.)`;
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
  // A typed 0 means "disable the filter", so it must reach Python as null.
  // `parseFloat(...) || 10.0` turned 0 into 10 (0 is falsy in JS), leaving no
  // way to switch the background-component filter off from the UI at all.
  const bgRaw = parseFloat(document.getElementById("background-area-ratio-input").value);
  const bgRatio = Number.isFinite(bgRaw) ? (bgRaw > 0 ? bgRaw : null) : 10.0;

  document.getElementById("start-extraction-btn").disabled = true;
  document.getElementById("extract-progress-bar").style.display = "block";
  setStatus("extract-status", "Starting extraction…");

  await window.EpiQuantAPI.start_extraction(maskType, bgRatio, excludeEdge);
}

function onProgress(data) {
  const barIdByStage = { extract: "extract-progress-bar", track: "track-progress-bar" };
  const statusIdByStage = { extract: "extract-status", track: "track-status", leading_edge: "leading-edge-status",
                            retrack: "retrack-status" };

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
  document.getElementById("shape-descriptors").style.display = "block";
  enableStep("classify");
  // Seed the minimum-junction box with the value derived from this project's
  // own junction lengths, so the number shown is the one actually in use.
  const minPx = await window.EpiQuantAPI.get_junction_min_px();
  if (minPx) {
    const input = document.getElementById("extract-junction-min-input");
    input.value = minPx.value;
    input.title = `Derived from this project's junction lengths (${minPx.auto} px). Edit to override.`;
  }
  syncExtractQcPanels();
  await showExtractFrame(state.extractFrame);
}

async function handleRunShapeExtraction() {
  const includeTissue = document.getElementById("shape-include-tissue").checked;
  const withCoefficients = document.getElementById("shape-with-coefficients").checked;
  const efdOrder = parseInt(document.getElementById("shape-efd-order").value, 10);
  const contourPoints = parseInt(document.getElementById("shape-contour-points").value, 10);
  const smoothing = parseFloat(document.getElementById("shape-smoothing").value);

  setStatus("shape-status", "Computing shape descriptors…");
  document.getElementById("run-shape-btn").disabled = true;
  try {
    await window.EpiQuantAPI.start_shape_extraction(
      includeTissue, efdOrder, contourPoints, smoothing, withCoefficients,
    );
  } catch (err) {
    setStatus("shape-status", `Error: ${err}`, "error");
    document.getElementById("run-shape-btn").disabled = false;
  }
}

function renderShapeSummary(summary) {
  const host = document.getElementById("shape-summary");
  host.innerHTML = "";
  Object.entries(summary || {}).forEach(([table, block]) => {
    const rows = Object.entries(block.columns)
      .map(([name, stats]) => {
        const median = stats.median === null ? "—" : stats.median.toFixed(3);
        const missing = stats.missing ? ` <span class="tag">${stats.missing} n/a</span>` : "";
        return `<tr><td>${name}</td><td>${median}</td><td>${missing}</td></tr>`;
      })
      .join("");
    const section = document.createElement("div");
    // Diagnostics are computed and written to the xlsx but are not offered as
    // PCA features, so they are named rather than tabulated -- listing them
    // alongside the selectable ones is what made the retired columns look
    // pickable.
    const diagnostics = (block.diagnostics || []).length
      ? `<p class="hint">Also written to the xlsx, not offered as features:
         ${block.diagnostics.join(", ")}.</p>`
      : "";
    section.innerHTML =
      `<h4>${table} — ${block.cell_count} cells</h4>` +
      `<table class="summary-table"><thead><tr><th>descriptor</th><th>median</th><th></th></tr></thead>` +
      `<tbody>${rows}</tbody></table>` +
      diagnostics;
    host.appendChild(section);
  });
}

async function onShapeExtractDone(summary) {
  setStatus("shape-status", "Shape descriptors ready.", "success");
  document.getElementById("run-shape-btn").disabled = false;
  renderShapeSummary(summary);
  // The Classify step may already have been visited, so its mode list has to
  // be rebuilt now rather than only on first entry.
  await populateFeatureOptions();
}

function onShapeExtractError(data) {
  setStatus("shape-status", `Shape extraction failed: ${data.message}`, "error");
  document.getElementById("run-shape-btn").disabled = false;
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
// default), "identity" (each track its own colour from a qualitative LUT),
// "environment" (TA-style neighbourhood-change score through a continuous
// LUT) or "lifecycle" (tracks starting/ending at this frame).
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
  await refreshExtractImage();
  document.getElementById("extract-qc-status").textContent = "";
}

async function refreshExtractImage() {
  const qcOn = document.getElementById("extract-qc-toggle").checked;
  const img = await window.EpiQuantAPI.get_extract_preview(
    state.extractFrame,
    qcOn ? document.getElementById("extract-qc-metric-select").value : null,
    document.getElementById("extract-qc-lut-select").value,
    getRenderMode("extract"),
    document.getElementById("extract-show-outline").checked,
    document.getElementById("extract-outline-color-input").value,
    document.getElementById("extract-show-labels").checked,
    parseInt(document.getElementById("extract-label-size-input").value, 10) || 10,
  );
  setPreviewImage("extract-preview-img", img);
}

function syncExtractQcPanels() {
  const on = document.getElementById("extract-qc-toggle").checked;
  document.getElementById("extract-qc-render-mode").style.display = on ? "flex" : "none";
  const junctionSelected = document.getElementById("extract-qc-metric-select").value === "junction_contrast";
  document.getElementById("extract-junction-min-wrap").style.display = on && junctionSelected ? "" : "none";
  if (!on) setQcHelpVisible(false);
}

// The metric help lives in a popover rather than inline prose: it now covers
// five metrics and was pushing the preview off screen.
// Open/close a `?` popover: toggles on its own button, closes on the X, on
// Escape, and on any click outside itself. Only one popover is open at a time.
function wirePopover(buttonId, closeId, popoverId) {
  const popover = document.getElementById(popoverId);
  document.getElementById(buttonId).addEventListener("click", (e) => {
    e.stopPropagation();
    const open = popover.style.display !== "none";
    document.querySelectorAll(".popover").forEach((p) => { p.style.display = "none"; });
    popover.style.display = open ? "none" : "block";
  });
  document.getElementById(closeId).addEventListener("click", () => { popover.style.display = "none"; });
  popover.addEventListener("click", (e) => e.stopPropagation());
  document.addEventListener("click", () => { popover.style.display = "none"; });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") popover.style.display = "none"; });
}

function setQcHelpVisible(visible) {
  document.getElementById("qc-help-popover").style.display = visible ? "block" : "none";
}

function qcHelpVisible() {
  return document.getElementById("qc-help-popover").style.display !== "none";
}

async function handleExtractImageClick(event) {
  if (!document.getElementById("extract-qc-toggle").checked) return;
  const { x, y } = imagePixelFromClick(event.target, event);
  const result = await window.EpiQuantAPI.get_qc_cell_at_pixel(state.extractFrame, x, y);
  if (!result) {
    setStatus("extract-qc-status", "No cell at that position.", "error");
    return;
  }
  const parts = Object.entries(result.scores)
    .map(([m, v]) => `${m}: ${v === null ? "n/a" : v}`)
    .join("   ");

  // Worst edges first, and name the neighbour, so a flagged cell says WHICH
  // of its junctions is the problem rather than only that one is.
  const edges = (result.junctions || []).slice(0, 3)
    .map((j) => `${j.contrast <= 0 ? "⚠ " : ""}vs cell ${j.with_cell}: ${j.contrast}`)
    .join("   ");
  const edgeText = edges ? `\n   junctions (worst first) — ${edges}` : "";
  setStatus("extract-qc-status", `${result.cell_index} — ${parts}${edgeText}`);
}

async function handleExtractQcSummary() {
  const metric = document.getElementById("extract-qc-metric-select").value;
  setStatus("extract-qc-status", "Scoring every frame…");
  const summary = await window.EpiQuantAPI.get_qc_summary(metric);
  const ranked = summary.filter((r) => r.scored > 0)
    .sort((a, b) => (b.flagged / b.scored) - (a.flagged / a.scored)).slice(0, 5);
  if (ranked.length === 0) {
    setStatus("extract-qc-status", "No frames could be scored with this metric.", "error");
    return;
  }
  const text = ranked
    .map((r) => `frame ${r.frame}: ${r.flagged}/${r.scored} flagged (median ${r.median})`)
    .join("   |   ");
  setStatus("extract-qc-status", `Worst frames by ${metric} — ${text}`);
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

  const used = document.getElementById("classify-features-used");
  if (used && result.features_used) {
    const scaler = result.standardize_mode === "none" ? "no standardization" : result.standardize_mode;
    const pooling = result.dataset_mode === "independent_samples"
      ? "each image clustered on its own — cluster_1 in one image is unrelated to cluster_1 in another"
      : "all frames pooled";
    used.textContent =
      `${result.features_used.length} features (${(result.feature_families || []).join(" + ")}, ${scaler}), ` +
      `${result.cells_used} cells, ${pooling}.`;
  }

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

// Populated from get_feature_options() on every entry to Classify -- not once
// -- because a user can run shape extraction after already visiting this step,
// and a cached list would leave the new families greyed out.
async function populateFeatureOptions() {
  const options = await window.EpiQuantAPI.get_feature_options();
  state.featureOptions = options;
  state.featureSelection = state.featureSelection || {};

  const picker = document.getElementById("feature-families");
  picker.innerHTML = "";
  Object.entries(options.families).forEach(([name, spec]) => {
    const label = document.createElement("label");
    label.className = "checkbox-label";
    // `basic` is the always-on base every other family stacks onto, so it is
    // checked and locked rather than merely checked.
    const checked = spec.base || (state.checkedFamilies || []).includes(name) ? " checked" : "";
    const disabled = spec.base || !spec.ready ? " disabled" : "";
    label.innerHTML =
      `<input type="checkbox" class="family-checkbox" value="${name}"${checked}${disabled}> ${spec.label}` +
      (spec.ready ? "" : ' <span class="tag">run shape descriptors first</span>');
    picker.appendChild(label);
  });
  picker.querySelectorAll(".family-checkbox").forEach((el) => {
    el.addEventListener("change", handleFamilyChange);
  });
  handleFamilyChange();
}

function checkedFamilies() {
  return Array.from(document.querySelectorAll(".family-checkbox:checked")).map((el) => el.value);
}

function handleFamilyChange() {
  const options = state.featureOptions;
  if (!options) return;
  let families = checkedFamilies();

  // EFD is exclusive: an orthogonal basis is taken whole and is never
  // standardized, so it cannot be mixed with the hand-crafted scalars. The
  // backend enforces this too -- this only keeps the UI honest about it.
  const efdOn = families.includes("efd");
  document.querySelectorAll(".family-checkbox").forEach((el) => {
    if (el.value === "efd") return;
    const spec = options.families[el.value];
    el.disabled = efdOn || spec.base || !spec.ready;
    // Selecting EFD clears the others; deselecting it must bring the base
    // back, or nothing is selected at all and the next run fails validation
    // with no visible cause.
    if (efdOn) el.checked = false;
    else if (spec.base) el.checked = true;
  });
  families = efdOn ? ["efd"] : checkedFamilies();
  state.checkedFamilies = families;

  const host = document.getElementById("feature-groups");
  host.innerHTML = "";
  if (efdOn) {
    host.innerHTML =
      `<p class="hint">${options.families.efd.note} All coefficients from the extraction are used;
       there is nothing to tick, because dropping arbitrary harmonics from a basis is not a
       meaningful subset.</p>`;
    return;
  }

  families.forEach((name) => {
    const spec = options.families[name];
    if (!spec.ready || !spec.columns.length) return;
    const group = document.createElement("fieldset");
    group.className = "feature-group";
    group.dataset.family = name;
    const boxes = spec.columns
      .map((column) => {
        const off = spec.default_off.includes(column);
        const remembered = state.featureSelection[column];
        const on = remembered === undefined ? !off : remembered;
        const id = `feature-cb-${column.replace(/[^A-Za-z0-9]/g, "-")}`;
        // `value` stays the real column name; only the visible text is shortened.
        const shown = (spec.display && spec.display[column]) || column;
        return `<label class="checkbox-label" title="${column}"><input type="checkbox" id="${id}" value="${column}"${on ? " checked" : ""}> ${shown}</label>`;
      })
      .join("");
    group.innerHTML =
      `<legend>${spec.label}</legend>` +
      `<div class="feature-group-boxes">${boxes}</div>` +
      (spec.note ? `<p class="hint">${spec.note}</p>` : "") +
      `<div class="feature-group-actions">
         <button type="button" class="link-btn" data-action="all">Select all</button>
         <button type="button" class="link-btn" data-action="none">Unselect all</button>
       </div>`;
    host.appendChild(group);
  });

  host.querySelectorAll(".link-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const group = btn.closest(".feature-group");
      group.querySelectorAll("input[type=checkbox]").forEach((cb) => {
        cb.checked = btn.dataset.action === "all";
        state.featureSelection[cb.value] = cb.checked;
      });
    });
  });
  host.querySelectorAll("input[type=checkbox]").forEach((cb) => {
    cb.addEventListener("change", () => { state.featureSelection[cb.value] = cb.checked; });
  });
}

function selectedFeatureColumns() {
  return Array.from(document.querySelectorAll("#feature-groups input[type=checkbox]:checked")).map((el) => el.value);
}

async function handleRunClassification() {
  const isAuto = document.getElementById("n-clusters-auto").checked;
  const nClusters = isAuto ? "auto" : parseInt(document.getElementById("n-clusters-input").value, 10);
  const kRange = [
    parseInt(document.getElementById("k-range-min").value, 10),
    parseInt(document.getElementById("k-range-max").value, 10),
  ];
  const families = state.checkedFamilies && state.checkedFamilies.length ? state.checkedFamilies : ["basic"];
  const isEfd = families.includes("efd");
  const selected = isEfd ? null : selectedFeatureColumns();
  const datasetMode = document.querySelector("input[name='dataset-mode']:checked").value;
  const pruneRaw = document.getElementById("prune-correlation").value;
  const prune = pruneRaw === "" ? null : parseFloat(pruneRaw);

  if (!isEfd && selected.length < 2) {
    setStatus("classify-status", "Select at least two features.", "error");
    return;
  }

  setStatus("classify-status", "Running PCA + K-means…");
  document.getElementById("run-classification-btn").disabled = true;
  try {
    const result = await window.EpiQuantAPI.run_classification(
      nClusters, kRange, null, families, selected, prune, datasetMode,
    );
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
    btn.className = "secondary-btn" + (state.activeReassignCluster === entry.cluster_label ? " active" : "");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = entry.color;
    btn.appendChild(swatch);
    btn.appendChild(document.createTextNode(entry.cell_type));
    btn.addEventListener("click", () => {
      const alreadyActive = state.activeReassignCluster === entry.cluster_label;
      state.activeReassignCluster = alreadyActive ? null : entry.cluster_label;
      renderClusterPicker();
      setStatus(
        "correct-class-selected-cell",
        alreadyActive ? "" : `Target cluster: ${entry.cell_type}. Click cells in the image to reassign them.`,
      );
    });
    picker.appendChild(btn);
  });
}

async function showClassifyFrame(frame) {
  state.classifyFrame = frame;
  document.getElementById("classify-frame-label").textContent = frameLabel(frame);
  const img = await window.EpiQuantAPI.get_classification_overlay(frame, getRenderMode("classify"));
  setPreviewImage("classify-overlay-img", img);
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
  setPreviewImage("correct-class-overlay-img", img);
  document.getElementById("correct-class-selected-cell").textContent = "";
  state.selectedCellForCorrection = null;
}

async function handleCorrectClassImageClick(event) {
  const img = event.target;
  const { x, y } = imagePixelFromClick(img, event);
  const result = await window.EpiQuantAPI.get_cell_at_pixel(state.correctClassFrame, x, y);
  if (!result) {
    setStatus("correct-class-selected-cell", "No cell at that position.", "error");
    state.selectedCellForCorrection = null;
    return;
  }
  state.selectedCellForCorrection = result;

  if (state.activeReassignCluster === null || state.activeReassignCluster === undefined) {
    setStatus(
      "correct-class-selected-cell",
      `${result.cell_index}: currently ${result.cell_type || "unclassified"}. Pick a target cluster below first.`,
    );
    return;
  }
  if (result.cluster_label === state.activeReassignCluster) {
    setStatus("correct-class-selected-cell", `${result.cell_index} is already in this cluster.`);
    return;
  }

  const targetLabel = state.activeReassignCluster;
  const legend = await window.EpiQuantAPI.reassign_cluster(result.cell_index, targetLabel);
  state.clusterLegend = legend;
  await refreshClusterLegend();
  await showCorrectClassFrame(state.correctClassFrame);
  setStatus("correct-class-selected-cell", `Reassigned ${result.cell_index} -> cluster ${targetLabel}.`, "success");
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

async function handleDetectStaleFrames() {
  const info = await window.EpiQuantAPI.get_stale_frames();
  const manual = (info.frames || []).filter((f) => f.manual_cells > 0).length;
  if (!info.tracked) {
    setStatus("retrack-status", "Track the movie once before re-tracking a range.", "error");
    return;
  }
  if (!info.suggested_range) {
    setStatus("retrack-status",
      `No mask has changed since the last tracking run (${manual} frame(s) contain hand-drawn cells).`, "success");
    return;
  }
  const [start, end] = info.suggested_range;
  document.getElementById("retrack-start-input").value = start;
  document.getElementById("retrack-end-input").value = end;
  setStatus("retrack-status",
    `Segmentation changed after tracking on frame(s) ${info.stale_frames.join(", ")} — range pre-filled.`, "error");
}

async function handleStartRetrack() {
  const start = parseInt(document.getElementById("retrack-start-input").value, 10);
  const end = parseInt(document.getElementById("retrack-end-input").value, 10);
  if (Number.isNaN(start) || Number.isNaN(end)) {
    setStatus("retrack-status", "Enter a valid Start and End frame.", "error");
    return;
  }
  if (start > end) {
    setStatus("retrack-status", "Start must not be after End.", "error");
    return;
  }
  document.getElementById("start-retrack-btn").disabled = true;
  document.getElementById("retrack-report-list").style.display = "none";
  setStatus("retrack-status", `Re-tracking frames ${start}-${end}…`);
  await window.EpiQuantAPI.start_local_retrack(start, end);
}

async function onRetrackDone(data) {
  document.getElementById("start-retrack-btn").disabled = false;
  const list = document.getElementById("retrack-report-list");
  list.innerHTML = "";

  (data.reconciled || []).forEach((r) => {
    const li = document.createElement("li");
    li.textContent = `Seam at frame ${r.frame}: track #${r.from_track_id} spliced into track #${r.into_track_id} (${r.cell_index}).`;
    list.appendChild(li);
  });
  (data.new_tracks || []).forEach((t) => {
    const li = document.createElement("li");
    li.textContent = `New track #${t.track_id} starts at frame ${t.from_frame} (${t.cell_index}).`;
    list.appendChild(li);
  });
  list.style.display = list.children.length ? "block" : "none";

  setStatus("retrack-status",
    `Re-tracked ${(data.frames || []).length} frame(s): ${data.relinked} link(s) updated, ` +
    `${(data.reconciled || []).length} seam reconciliation(s), ${(data.new_tracks || []).length} new track(s). ` +
    `${data.track_count} tracks total.`, "success");

  state.trackNumbers = await window.EpiQuantAPI.get_track_numbers();
}

function onRetrackError(data) {
  document.getElementById("start-retrack-btn").disabled = false;
  setStatus("retrack-status", `Re-track failed: ${data.message}`, "error");
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

// Each colour mode owns one options panel; show that one and hide the rest.
// (Was a hardcoded two-way toggle before the environment/lifecycle modes.)
const TRACK_COLOR_MODE_PANELS = {
  uniform: ["correct-track-uniform-colors"],
  identity: ["correct-track-identity-palette"],
  environment: ["correct-track-environment", "correct-track-environment-hint"],
  lifecycle: ["correct-track-lifecycle"],
  agreement: ["correct-track-agreement"],
};

// Every control writes through the one API call and then re-reads what was
// actually applied, so the three widgets can never drift apart -- the backend
// derives `mode` from the weights rather than storing it separately.
async function applyTrackingCost(mode, overlapWeight, motionCompensation, maxFrameGap) {
  const applied = await window.EpiQuantAPI.set_tracking_cost(mode, overlapWeight, motionCompensation, maxFrameGap);
  if (!applied) return;
  syncTrackingCostControls(applied);
}

function syncTrackingCostControls(cost) {
  const radio = document.querySelector(`input[name="track-cost-mode"][value="${cost.mode}"]`);
  if (radio) radio.checked = true;
  const weight = document.getElementById("overlap-weight-input");
  weight.value = cost.overlap_weight;
  // The weight is meaningless when the overlap term is switched off entirely.
  weight.disabled = cost.mode === "centroid";
  document.getElementById("motion-compensation-input").checked = !!cost.motion_compensation;
  document.getElementById("gap-closing-input").checked = !!cost.max_frame_gap;
  const drift = document.getElementById("drift-register-input");
  if (drift) drift.checked = cost.drift_mode && cost.drift_mode !== "none";
}

function syncTrackColorModePanels() {
  const mode = getColorMode("correct-track");
  Object.entries(TRACK_COLOR_MODE_PANELS).forEach(([panelMode, ids]) => {
    ids.forEach((id) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.style.display = panelMode === mode ? (id.endsWith("-hint") ? "block" : "flex") : "none";
    });
  });
}

async function refreshCorrectTrackImage() {
  const showLabels = document.getElementById("correct-track-show-labels").checked;
  const identityPalette = document.getElementById("correct-track-identity-palette-select").value;
  const labelFontSize = parseInt(document.getElementById("correct-track-label-size-input").value, 10) || 10;
  const environmentLut = document.getElementById("correct-track-environment-lut-select").value;
  const envDirEl = document.querySelector('input[name="correct-track-env-direction"]:checked');
  const environmentDirection = envDirEl ? envDirEl.value : "prev";
  const tintOthers = document.getElementById("lifecycle-tint-others-checkbox").checked;
  const img = await window.EpiQuantAPI.get_tracking_overlay(
    state.correctTrackFrame, getRenderMode("correct-track"), showLabels, getColorMode("correct-track"),
    identityPalette, labelFontSize, environmentLut, environmentDirection, tintOthers,
    document.getElementById("dynamics-axes-checkbox").checked
  );
  setPreviewImage("correct-track-overlay-img", img);
  await refreshAgreementCounts();
  await refreshRegistrationStatus();
}

// The counts are always shown alongside the overlay, including how many cells
// could NOT be judged -- an overlay that silently omits them would read as if
// every cell had been checked.
async function refreshAgreementCounts() {
  const readout = document.getElementById("agreement-counts");
  if (getColorMode("correct-track") !== "agreement") {
    readout.textContent = "";
    return;
  }
  const result = await window.EpiQuantAPI.get_link_agreement(state.correctTrackFrame);
  if (!result) {
    readout.textContent = "";
    return;
  }
  // A mode that cannot draw must say why. Previously every one of these paths
  // rendered nothing and looked exactly like the mode being off.
  if (result.reason) {
    setStatus("agreement-counts", result.message || `Nothing to show (${result.reason}).`, "error");
    state.agreementDetail = {};
    return;
  }
  const c = result.counts || {};
  const total = Object.values(c).reduce((a, b) => a + b, 0) || 1;
  const pct = (n) => `${n || 0} (${Math.round((100 * (n || 0)) / total)}%)`;
  // Naming the convention is not optional: the same tracking scores 4.9% or
  // 14.0% depending only on whether the reference masks were registered.
  const ref = result.registered ? "registered" : "unregistered";
  readout.textContent =
    `Against the highest-overlap successor (${ref} masks): agrees ${pct(c.agrees)}, ` +
    `disagrees ${pct(c.disagrees)}, can't judge ${pct(c.unjudgeable)}, broken ${pct(c.broken)}.`;
  state.agreementDetail = result.detail || {};
}

async function refreshRegistrationStatus() {
  const el = document.getElementById("registration-status");
  if (!el) return;
  const r = await window.EpiQuantAPI.get_registration_status();
  if (!r) { el.textContent = ""; return; }
  el.textContent = r.register
    ? `Tissue drift ${r.median_drift_px} px per frame against a ${r.median_cell_radius_px} px cell radius `
      + `(ratio ${r.ratio}, threshold ${r.threshold}) — the agreement reference `
      + `${r.applied ? "is" : "would be"} computed on registered masks.`
    : `Tissue drift ${r.median_drift_px} px per frame against a ${r.median_cell_radius_px} px cell radius `
      + `(ratio ${r.ratio}, below the ${r.threshold} threshold) — no registration needed.`;
}

// Both totals, always. The persistence filter never hides an event, so the
// user can see exactly how much of the raw count it accounts for.
async function refreshT1Counts() {
  const readout = document.getElementById("dynamics-t1-counts");
  const modeEl = document.querySelector('input[name="trajectory-color-mode"]:checked');
  if (!modeEl || modeEl.value !== "dynamics") {
    readout.textContent = "";
    return;
  }
  const events = await window.EpiQuantAPI.get_t1_events(state.trajectoryFrame, "next");
  if (!events || events.error) {
    readout.textContent = "";
    return;
  }
  // Three states, always all three. An unjudgeable event is one there was not
  // enough movie left to check -- folding it into either of the other two
  // would claim a verdict that was never reached.
  const hold = events.persist_frames;
  if (hold > 0) {
    readout.textContent =
      `T1 this frame: raw ${events.gained} gained / ${events.lost} lost; ` +
      `settled ${events.gained_persistent} / ${events.lost_persistent} after ${hold} frame(s); ` +
      `can't judge ${events.gained_unjudgeable || 0} / ${events.lost_unjudgeable || 0}.`;
  } else {
    readout.textContent =
      `T1 this frame: ${events.gained} gained / ${events.lost} lost (no persistence filter).`;
  }
}

// --- Coverage log (Round 15 C4) -------------------------------------------
// Corrections are labelled errors. Without a record of what was DISPLAYED and
// clicked, a cell left alone is indistinguishable from a cell nobody saw, so
// the correction tool has never produced the other half of a ground truth.
// These calls are fire-and-forget: a coverage write must never delay or break
// a correction the user is making.
function logCoverage(event, frame, extra = {}) {
  try {
    const promise = window.EpiQuantAPI.log_coverage_event(
      event, frame, extra.cellIndex ?? null, extra.trackId ?? null,
      extra.msElapsed ?? null, extra.zoomPercent ?? null,
      extra.renderMode ?? "", extra.colorMode ?? "", extra.detail ?? ""
    );
    if (promise && promise.catch) promise.catch(() => {});
  } catch (e) {
    /* never surface a logging failure into the correction flow */
  }
}

// Close out the frame currently being timed, if any. Called when the frame
// changes and when the Correct Tracking step is left -- a dwell that is never
// closed is a frame that reads as never displayed, which is the failure this
// whole log exists to avoid.
function flushCorrectTrackDwell() {
  if (state.correctTrackShownAt === null || state.correctTrackShownFrame === null) return;
  const elapsed = Date.now() - state.correctTrackShownAt;
  logCoverage("frame_shown", state.correctTrackShownFrame, {
    msElapsed: elapsed,
    zoomPercent: state.zoomLevels["correct-track"] ?? 100,
    renderMode: getRenderMode("correct-track"),
    colorMode: getColorMode("correct-track"),
  });
  state.correctTrackShownAt = null;
  state.correctTrackShownFrame = null;
}

async function showCorrectTrackFrame(frame) {
  // Every piece of panel state is cleared SYNCHRONOUSLY, before the await.
  //
  // It used to be cleared afterwards, which made the ordering a race: the
  // "Use this" handler does `await refreshCorrectTrackImage(); showSuggestCheckButton(...)`,
  // and if a frame change resolved its await in between, the reset landed on
  // top of the freshly-shown panel and the Check-how-far button vanished.
  // Reordering here is not cosmetic -- it removes the await from between the
  // reset and any later show, so no other async caller can interleave.
  // Same shape as the `_qc_scores` / `_qc_display` clobber from an earlier
  // round: two paths writing one piece of state, ordering decided by timing.
  flushCorrectTrackDwell();
  state.correctTrackFrame = frame;
  state.correctTrackShownFrame = frame;
  state.correctTrackShownAt = Date.now();
  document.getElementById("correct-track-frame-label").textContent = frameLabel(frame);
  document.getElementById("correct-track-selected-cell").textContent = "";
  document.getElementById("correct-track-reassign").style.display = "none";
  document.getElementById("correct-track-suggest").style.display = "none";
  document.getElementById("correct-track-suggest-list").style.display = "none";
  resetSmartStopPanel();
  resetSuggestCheckButton();
  state.selectedTrackCell = null;
  resetTrackSwapSelection();
  await refreshCorrectTrackImage();
}

// --- Ground-truth census (Round 15 C1/C2) ---------------------------------
//
// One link per judgment, keyboard-driven, auto-advancing. The blink is the
// point: holding space toggles the SAME crop window between frame f and f+1,
// so a correct link shifts slightly and a swap jumps. Humans see that
// pre-attentively; side-by-side comparison does not exploit it, and it is the
// single largest speed factor in this view.
const censusState = {
  active: false, position: 0, total: 0, item: null,
  shownAt: 0, blinking: false, awaitingCandidate: false,
};

function censusShow(which) {
  if (!censusState.item) return;
  const img = document.getElementById("census-image");
  const tag = document.getElementById("census-frame-tag");
  const item = censusState.item;
  if (which === "after" && item.after) {
    img.src = item.after;
    tag.textContent = `frame ${item.frame + 1} — hold space to blink back`;
  } else {
    img.src = item.before;
    tag.textContent = `frame ${item.frame} — hold space to blink`;
  }
}

async function loadCensusItem(position) {
  const item = await window.EpiQuantAPI.get_census_item(position, 28);
  if (item && item.error) {
    setStatus("census-status", item.error, "error");
    document.getElementById("census-panel").style.display = "none";
    censusState.active = false;
    return;
  }
  censusState.item = item;
  censusState.position = position;
  censusState.total = item.total;
  censusState.awaitingCandidate = false;
  censusState.shownAt = Date.now();
  document.getElementById("census-candidates").style.display = "none";
  document.getElementById("census-candidates").innerHTML = "";
  censusShow("before");
  const ious = (item.reference_iou === null || item.reference_iou === undefined)
    ? "" : ` (this link overlaps ${item.tracker_iou}; the reference's pick overlaps ${item.reference_iou})`;
  setStatus("census-progress",
    `${position + 1} of ${item.total} — frame ${item.frame} cell ${item.label}` +
    `${item.track_id === null ? "" : `, track #${item.track_id}`}${ious}`, "");
}

async function handleStartCensus() {
  const kind = document.getElementById("census-kind-select").value;
  const frame = parseInt(document.getElementById("census-frame-input").value, 10);
  const labeller = document.getElementById("census-labeller-input").value;
  setStatus("census-status", "Building the queue…", "");
  const started = await window.EpiQuantAPI.start_census(
    kind, kind === "frame_pair" ? frame : null, 0.15, 0.10, 0, labeller);
  if (started && started.error) {
    setStatus("census-status", started.error, "error");
    return;
  }
  censusState.active = true;
  document.getElementById("census-panel").style.display = "block";
  setStatus("census-status",
    `${started.n_real} links to judge (${started.n} items including the hidden checks). ` +
    `Overlap reference computed on ${started.register ? "registered" : "unregistered"} masks — ` +
    `the same convention this tracking used. Saving to ${started.path}`, "success");
  await loadCensusItem(0);
}

function beginCensusWrong() {
  const candidates = (censusState.item && censusState.item.candidates) || [];
  if (!candidates.length) {
    submitCensus("wrong", null);
    return;
  }
  censusState.awaitingCandidate = true;
  const box = document.getElementById("census-candidates");
  box.innerHTML = "<span class=\"hint\">Which cell is it? 1–3, or 0 for none of these.</span>";
  candidates.forEach((candidate, i) => {
    const wrap = document.createElement("div");
    wrap.className = "census-candidate";
    const img = document.createElement("img");
    img.src = candidate.image;
    const caption = document.createElement("span");
    caption.textContent = `${i + 1} — cell ${candidate.label}`;
    wrap.appendChild(img); wrap.appendChild(caption);
    wrap.addEventListener("click", () => submitCensus("wrong", candidate.label));
    box.appendChild(wrap);
  });
  const none = document.createElement("button");
  none.className = "secondary-btn";
  none.textContent = "0 — none of these";
  none.addEventListener("click", () => submitCensus("wrong", null));
  box.appendChild(none);
  box.style.display = "flex";
}

async function submitCensus(verdict, chosenLabel = null) {
  if (!censusState.active || !censusState.item) return;
  const elapsed = Date.now() - censusState.shownAt;
  const result = await window.EpiQuantAPI.submit_census_verdict(
    censusState.position, verdict, chosenLabel, elapsed);
  if (result && result.error) {
    setStatus("census-status", result.error, "error");
    return;
  }
  if (result.next >= result.total) {
    censusState.active = false;
    document.getElementById("census-panel").style.display = "none";
    setStatus("census-status", `Census complete — ${result.total} items.`, "success");
    await handleCensusSummary();
    return;
  }
  await loadCensusItem(result.next);
}

function handleCensusKeyDown(event) {
  if (!censusState.active) return;
  const tag = (event.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select") return;
  const key = event.key.toLowerCase();
  if (key === " " || event.code === "Space") {
    event.preventDefault();
    if (!censusState.blinking) { censusState.blinking = true; censusShow("after"); }
    return;
  }
  if (censusState.awaitingCandidate && "0123".includes(event.key)) {
    event.preventDefault();
    const candidates = censusState.item.candidates || [];
    const index = parseInt(event.key, 10);
    submitCensus("wrong", index === 0 ? null : (candidates[index - 1] || {}).label ?? null);
    return;
  }
  if (key === "j") { event.preventDefault(); submitCensus("correct"); }
  else if (key === "k") { event.preventDefault(); beginCensusWrong(); }
  else if (key === "l") { event.preventDefault(); submitCensus("unjudgeable"); }
}

function handleCensusKeyUp(event) {
  if (!censusState.active) return;
  if (event.key === " " || event.code === "Space") {
    censusState.blinking = false;
    censusShow("before");
  }
}

async function handleCensusSummary() {
  const s = await window.EpiQuantAPI.get_census_summary();
  if (s && s.error) { setStatus("census-summary", s.error, "error"); return; }
  const pct = (v) => v === null || v === undefined ? "n/a" : `${(v * 100).toFixed(0)}%`;
  // Every one of these qualifies the headline number, so all four are shown
  // together or none of them are. An error rate without the self-consistency
  // ceiling it was measured under is not a quotable figure.
  setStatus("census-summary",
    `${s.n_real} real judgments: ${s.counts.correct} correct, ${s.counts.wrong} wrong, ` +
    `${s.counts.unjudgeable} can't-tell. Error rate ${pct(s.error_rate)} of the judgeable ones. ` +
    `Decoys caught ${s.decoys_caught}/${s.decoys_shown} (${pct(s.decoy_pass_rate)}); ` +
    `self-consistency ${pct(s.intra_rater_agreement)} over ${s.repeats_compared} repeat(s) — ` +
    `no accuracy from these labels can exceed that. Median ${s.median_seconds ?? "n/a"} s per judgment.`,
    "success");
}

async function handleShowCoverageSummary() {
  // Close the open dwell first, or the frame the user is looking at right now
  // reports 0 ms and shows as glimpsed.
  flushCorrectTrackDwell();
  state.correctTrackShownFrame = state.correctTrackFrame;
  state.correctTrackShownAt = Date.now();

  const dwell = parseFloat(document.getElementById("coverage-dwell-input").value);
  const summary = await window.EpiQuantAPI.get_coverage_summary(Number.isNaN(dwell) ? null : dwell);
  const list = document.getElementById("coverage-summary-list");
  if (summary && summary.error) {
    setStatus("coverage-summary-status", `Error: ${summary.error}`, "error");
    list.style.display = "none";
    return;
  }
  const totals = summary.totals || {};
  setStatus("coverage-summary-status",
    `${summary.events} event(s) over ${(summary.sessions || []).length} session(s). ` +
    `${totals.changed} corrected, ${totals.inspected_unchanged} inspected and left alone ` +
    `(weak positives), ${totals.displayed_only} displayed but not clicked, ` +
    `${totals.never_displayed} never displayed.`,
    summary.events ? "success" : "");

  const frames = Object.entries(summary.frames || {})
    .sort((a, b) => Number(a[0]) - Number(b[0]))
    .filter(([, f]) => f.displayed || f.never_displayed.length);
  list.innerHTML = "";
  for (const [frame, f] of frames) {
    const li = document.createElement("li");
    li.textContent = f.displayed
      ? `Frame ${frame}: ${Math.round(f.dwell_ms)} ms${f.glimpsed ? " (glimpsed -- no weak positives)" : ""}, ` +
        `${f.changed.length} corrected, ${f.inspected_unchanged.length} inspected and left, ` +
        `${f.displayed_only.length} not clicked.`
      : `Frame ${frame}: never displayed (${f.never_displayed.length} cells, no evidence either way).`;
    list.appendChild(li);
  }
  list.style.display = frames.length ? "block" : "none";
}

function resetSmartStopPanel() {
  state.smartStopPending = null;
  document.getElementById("compute-stop-frame-btn").style.display = "none";
  document.getElementById("smart-stop-result").textContent = "";
  document.getElementById("apply-smart-stop-btn").style.display = "none";
}

function resetTrackSwapSelection() {
  state.trackSwapSelection = [];
  document.getElementById("swap-track-a-input").value = "";
  document.getElementById("swap-track-b-input").value = "";
  document.getElementById("swap-propagate-checkbox").checked = false;
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
  const propagate = document.getElementById("swap-propagate-checkbox").checked;
  const result = await window.EpiQuantAPI.swap_tracks(state.correctTrackFrame, idA, idB, propagate);
  if (result && result.error) {
    setStatus("correct-track-swap-status", `Error: ${result.error}`, "error");
    return;
  }
  let message = "Swapped.";
  if (propagate && result) {
    const swappedCount = (result.swapped_frames || []).length;
    const skippedCount = (result.skipped_frames || []).length;
    message = `Swapped ${swappedCount} frame(s) from frame ${state.correctTrackFrame} on.`;
    if (skippedCount) {
      message += ` (${skippedCount} frame(s) skipped -- one side was unmatched there: ${result.skipped_frames.join(", ")}.)`;
    }
  }
  await showCorrectTrackFrame(state.correctTrackFrame);
  setStatus("correct-track-swap-status", message, "success");
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
  resetSuggestCheckButton();
    return;
  }
  state.selectedTrackCell = result;
  const isMatched = result.track_id !== null && result.track_id !== undefined;
  logCoverage("cell_inspected", state.correctTrackFrame, {
    cellIndex: result.cell_index, trackId: isMatched ? result.track_id : null,
    zoomPercent: state.zoomLevels["correct-track"] ?? 100,
    colorMode: getColorMode("correct-track"),
  });
  // In agreement mode, show BOTH overlaps for the clicked cell. "0.301 vs
  // 0.231" reads as the near-tie it is; a red cell alone does not, and 65% of
  // flagged disagreements differ by less than 0.15 IoU.
  let agreementNote = "";
  const detail = (state.agreementDetail || {})[result.cell_id];
  if (getColorMode("correct-track") === "agreement" && detail) {
    const label = { agrees: "agrees with", disagrees: "disagrees with",
                    unjudgeable: "can't be judged by", broken: "no link;" }[detail.state] || detail.state;
    const tracker = detail.tracker_iou === null || detail.tracker_iou === undefined
      ? "no successor" : `this tracking's pick ${detail.tracker_iou}`;
    agreementNote = ` — ${label} the overlap reference (reference ${detail.reference_iou} vs ${tracker}).`;
  }
  document.getElementById("correct-track-selected-cell").textContent = (isMatched
    ? `${result.cell_index}: currently track #${result.track_id}.`
    : `${result.cell_index}: unmatched (raw cell id ${result.cell_id}).`) + agreementNote;
  document.getElementById("correct-track-reassign").style.display = "flex";
  document.getElementById("new-track-id-input").value = "";
  document.getElementById("propagate-correction-checkbox").checked = false;

  resetSmartStopPanel();
  resetSuggestCheckButton();
  document.getElementById("compute-stop-frame-btn").style.display = isMatched ? "inline-block" : "none";

  document.getElementById("correct-track-suggest").style.display = "flex";
  document.getElementById("correct-track-suggest-list").style.display = "none";
  document.getElementById("correct-track-suggest-list").innerHTML = "";
  document.getElementById("correct-track-suggest-hint").textContent = "";
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
    const note = describeDisplacement(result && result.displaced);
    logCoverage("cell_changed", state.correctTrackFrame, {
      cellIndex: correctedCellIndex, trackId: newId,
      detail: isMatched ? "reassign_track" : "add_tracking_id",
    });
    await showCorrectTrackFrame(state.correctTrackFrame);
    setStatus("correct-track-selected-cell", `Applied.${note}`, "success");
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
  const note = describeDisplacement(result && result.displaced);
  await showCorrectTrackFrame(state.correctTrackFrame);
  setStatus("smart-stop-result", `Merged through frame ${stopFrame}.${note}`, "success");
}

async function handleSuggestMatch() {
  if (!state.selectedTrackCell) return;
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
    list.textContent = "No candidate tracks found in the previous frame.";
    return;
  }
  const selectedCell = state.selectedTrackCell;
  const currentTrackId = selectedCell.track_id !== null && selectedCell.track_id !== undefined ? selectedCell.track_id : null;

  candidates.forEach((candidate) => {
    const item = document.createElement("li");
    // A positive swap gain means these two tracks look exchanged: each cell
    // sits nearer the OTHER track's previous position. That is the dominant
    // error in a densely tracked movie, where nothing is ever orphaned.
    const swap = candidate.swap_gain;
    const held = candidate.current_cell
      ? `now holds ${candidate.current_cell}`
      : "unmatched here";
    const flag = swap && swap > 0
      ? ` \u26a0 looks swapped \u2014 exchanging saves ${swap.toFixed(1)}px`
      : "";
    // Topology MARKS; it takes no part in the ordering. Measured with the flag
    // and the judge sharing a registration convention, geometry alone is the
    // informative signal (93.1% / 66.2% on the two movies) and topology alone
    // is near baseline (8.3% / 5.3%); where both agree, precision is
    // 100% / 76.9%. But the flag itself mostly fires on CROWDING -- 100% / 98.1%
    // of flagged disagreements have their nearest rival already claimed, and
    // only 2.6% / 0.0% are true 2-cycles -- so reordering on it would have users
    // undoing global assignment decisions one at a time.
    let topoNote = "";
    if (candidate.topology === "swap") {
      topoNote = " \u2713 two criteria agree \u2014 the neighbour set also favours this track";
    } else if (candidate.topology === "keep") {
      topoNote = " (neighbour set unchanged \u2014 geometry's call stands)";
    }
    item.textContent =
      `Track #${candidate.track_id} (last seen ${candidate.cell_index}, ${candidate.distance.toFixed(1)}px away, ${held})${flag}${topoNote} `;
    if (candidate.topology === "keep") {
      item.title = "The neighbour set did not change, so topology does not confirm this swap. "
        + "The ordering here is purely geometric \u2014 topology marks, it never reorders. "
        + "Measured on two movies, the geometry flag mostly detects crowding (the nearest cell "
        + "already taken by another track) rather than an identity swap.";
    }
    if (swap && swap > 0) item.style.fontWeight = "600";
    if (candidate.topology === "swap") item.style.color = "#1a7f37";
    const useBtn = document.createElement("button");
    useBtn.className = "secondary-btn";
    useBtn.textContent = "Use this";
    useBtn.addEventListener("click", async () => {
      // Unmatched source -> attach directly; already (wrongly) matched
      // source -> move its claim onto the candidate track instead.
      const applyResult = currentTrackId === null
        ? await window.EpiQuantAPI.add_tracking_id(state.correctTrackFrame, candidate.track_id, selectedCell.cell_index)
        : await window.EpiQuantAPI.reassign_track(state.correctTrackFrame, currentTrackId, candidate.track_id, false);

      if (applyResult && applyResult.error) {
        setStatus("correct-track-suggest-list", `Error: ${applyResult.error}`, "error");
        return;
      }
      const note = describeDisplacement(applyResult && applyResult.displaced);
      list.style.display = "none";
      list.innerHTML = "";
      await refreshCorrectTrackImage();
      showSuggestCheckButton(candidate.track_id);
      setStatus("correct-track-selected-cell", `Attached ${selectedCell.cell_index} to track #${candidate.track_id}.${note}`, "success");
    });
    item.appendChild(useBtn);
    list.appendChild(item);
  });
}

function showSuggestCheckButton(trackId) {
  state.suggestCheckTrackId = trackId;
  document.getElementById("correct-track-suggest-check").style.display = "flex";
  document.getElementById("suggest-check-result").textContent = "";
}

function resetSuggestCheckButton() {
  state.suggestCheckTrackId = null;
  state.confirmedMovesPending = null;
  document.getElementById("correct-track-suggest-check").style.display = "none";
  document.getElementById("suggest-check-result").textContent = "";
  document.getElementById("suggest-check-moves-list").style.display = "none";
  document.getElementById("suggest-check-moves-list").innerHTML = "";
  document.getElementById("apply-confirmed-moves-btn").style.display = "none";
}

async function handleCheckSuggestHolds() {
  if (state.suggestCheckTrackId === null || state.suggestCheckTrackId === undefined) return;
  const trackId = state.suggestCheckTrackId;
  const restrictToUnmatched = document.getElementById("restrict-unmatched-checkbox").checked;
  const result = await window.EpiQuantAPI.compute_confirmed_identity_moves(
    state.correctTrackFrame, trackId, 20, restrictToUnmatched
  );
  const list = document.getElementById("suggest-check-moves-list");
  const applyBtn = document.getElementById("apply-confirmed-moves-btn");

  if (result && result.error) {
    setStatus("suggest-check-result", `Error: ${result.error}`, "error");
    list.style.display = "none";
    applyBtn.style.display = "none";
    return;
  }

  const moves = result.moves || [];
  list.innerHTML = "";

  if (moves.length === 0) {
    const hint = result.reason === "no_unmatched_continuation"
      ? " The nearest cell there already belongs to another track -- untick \"Pair with unmatched cells only\" to allow taking it."
      : " No further corrections needed.";
    setStatus("suggest-check-result", `Holds through frame ${result.stop_frame} (${result.reason}).${hint}`, "success");
    list.style.display = "none";
    applyBtn.style.display = "none";
    state.confirmedMovesPending = null;
    return;
  }

  setStatus(
    "suggest-check-result",
    `Found ${moves.length} correction(s) needed through frame ${result.stop_frame} (${result.reason}):`,
    "success"
  );
  moves.forEach((move) => {
    const item = document.createElement("li");
    const fromText = move.from_track_id === null || move.from_track_id === undefined
      ? "unmatched" : `track #${move.from_track_id}`;
    const distanceText = move.distance === undefined || move.distance === null ? "" : `, ${move.distance}px`;
    item.textContent = `Frame ${move.frame}: currently ${fromText} → would become track #${trackId} (${move.cell_index}${distanceText})`;
    list.appendChild(item);
  });
  list.style.display = "block";

  state.confirmedMovesPending = { trackId, moves };
  applyBtn.textContent = `Apply ${moves.length} correction(s)`;
  applyBtn.style.display = "inline-block";
}

async function handleApplyConfirmedMoves() {
  if (!state.confirmedMovesPending) return;
  const { trackId, moves } = state.confirmedMovesPending;
  const result = await window.EpiQuantAPI.apply_confirmed_identity_moves(trackId, moves);
  if (result && result.error) {
    setStatus("suggest-check-result", `Error: ${result.error}`, "error");
    return;
  }
  const notes = (result.displaced_list || []).map(describeDisplacement).join("");
  document.getElementById("suggest-check-moves-list").style.display = "none";
  document.getElementById("suggest-check-moves-list").innerHTML = "";
  document.getElementById("apply-confirmed-moves-btn").style.display = "none";
  state.confirmedMovesPending = null;
  await refreshCorrectTrackImage();
  setStatus("suggest-check-result", `Applied ${moves.length} correction(s).${notes}`, "success");
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

// `preserveFrame` keeps the frame currently on screen when the newly selected
// track also exists there -- used by click-to-jump, where the user just
// clicked a cell they can see and yanking them to the track's first frame
// would lose their place. The number input and Prev/Next Track buttons keep
// the original jump-to-start behaviour.
async function loadTrack(tn, { preserveFrame = false } = {}) {
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
  const current = state.trajectoryFrame;
  const keep = preserveFrame && current >= range.min_frame && current <= range.max_frame;
  const frame = keep ? current : range.min_frame;
  state.trajectoryFrame = frame;
  await showTrajectoryFrame(frame);
}

async function handleTrajectoryImageClick(event) {
  const { x, y } = imagePixelFromClick(event.target, event);
  const result = await window.EpiQuantAPI.get_tracking_cell_at_pixel(state.trajectoryFrame, x, y);
  if (!result) {
    setStatus("trajectory-status", "No cell at that position.", "error");
    return;
  }
  if (result.track_id === null || result.track_id === undefined) {
    setStatus("trajectory-status", `${result.cell_index} is unmatched — it belongs to no track.`, "error");
    return;
  }
  await loadTrack(result.track_id, { preserveFrame: true });
  setStatus("trajectory-status", `Jumped to track #${result.track_id} (${result.cell_index}).`, "success");
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
  const modeEl = document.querySelector('input[name="trajectory-color-mode"]:checked');
  const colorMode = modeEl ? modeEl.value : "highlight";
  document.getElementById("correct-track-dynamics").style.display = colorMode === "dynamics" ? "flex" : "none";
  const img = await window.EpiQuantAPI.get_trajectory_overlay(
    state.currentTrackNumber, frame, getRenderMode("trajectory"), colorMode,
    document.getElementById("dynamics-axes-checkbox").checked
  );
  await refreshT1Counts();
  if (img) {
    setPreviewImage("trajectory-overlay-img", img);
    setStatus("trajectory-status", "");
  } else {
    // Clear the image rather than leaving the previous frame's on screen:
    // it would show frame N-1's pixels while `state.trajectoryFrame` says N,
    // so a click-to-jump would hit-test the wrong frame entirely.
    setPreviewImage("trajectory-overlay-img", null);
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

  document.getElementById(`${prefix}-zoom-in`).addEventListener("click", () => setZoom(state.zoomLevels[prefix] + 10));
  document.getElementById(`${prefix}-zoom-out`).addEventListener("click", () => setZoom(state.zoomLevels[prefix] - 10));
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
  document.getElementById("extract-preview-img").addEventListener("click", handleExtractImageClick);
  document.getElementById("extract-qc-toggle").addEventListener("change", async () => {
    syncExtractQcPanels();
    await refreshExtractImage();
  });
  document.getElementById("extract-qc-metric-select").addEventListener("change", async () => {
    syncExtractQcPanels();
    await refreshExtractImage();
  });
  document.getElementById("extract-junction-min-input").addEventListener("change", async (e) => {
    const applied = await window.EpiQuantAPI.set_junction_min_px(parseInt(e.target.value, 10) || null);
    if (applied) e.target.value = applied.value;
    await refreshExtractImage();
  });
  wirePopover("qc-help-btn", "qc-help-close", "qc-help-popover");
  wirePopover("overlap-help-btn", "overlap-help-close", "overlap-help-popover");
  document.getElementById("overlap-weight-input").addEventListener("change", async (e) => {
    await applyTrackingCost(null, parseFloat(e.target.value), null);
  });
  document.querySelectorAll('input[name="track-cost-mode"]').forEach((el) => {
    el.addEventListener("change", async (ev) => {
      await applyTrackingCost(ev.target.value, null, null);
    });
  });
  document.getElementById("motion-compensation-input").addEventListener("change", async (e) => {
    await applyTrackingCost(null, null, e.target.checked, null);
  });
  document.getElementById("gap-closing-input").addEventListener("change", async (e) => {
    await applyTrackingCost(null, null, null, e.target.checked ? 1 : 0);
  });
  document.getElementById("drift-register-input").addEventListener("change", async (e) => {
    const applied = await window.EpiQuantAPI.set_tracking_cost(
      null, null, null, null, e.target.checked ? "mask" : "none");
    if (applied) syncTrackingCostControls(applied);
    await refreshRegistrationStatus();
  });
  document.getElementById("copy-view-btn").addEventListener("click", handleCopyView);
  ["extract-show-outline", "extract-outline-color-input", "extract-show-labels", "extract-label-size-input"]
    .forEach((id) => document.getElementById(id).addEventListener("change", refreshExtractImage));
  document.getElementById("extract-qc-lut-select").addEventListener("change", refreshExtractImage);
  document.getElementById("extract-qc-summary-btn").addEventListener("click", handleExtractQcSummary);
  wireRenderMode("extract", refreshExtractImage);
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
  document.getElementById("run-shape-btn").addEventListener("click", handleRunShapeExtraction);
  document.querySelectorAll("input[name='dataset-mode']").forEach((el) => {
    el.addEventListener("change", () => {
      document.getElementById("dataset-mode-note").textContent = el.value === "independent_samples"
        ? "PCA and K-means run per image. Use this for folders of unrelated samples, where pooling makes the leading component partly 'which image is this'. Not suitable if you will track these cells."
        : "Pooled into one feature space. Required if you will track these cells.";
    });
  });
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
  document.getElementById("detect-stale-btn").addEventListener("click", handleDetectStaleFrames);
  document.getElementById("start-retrack-btn").addEventListener("click", handleStartRetrack);
  document.getElementById("go-to-correct-tracking").addEventListener("click", () => goToStep("correct-tracking"));
  document.getElementById("go-to-trajectory").addEventListener("click", enterTrajectoryStep);

  // Correct Tracking
  wireFrameNav("correct-track", () => state.correctTrackFrame, (f) => (state.correctTrackFrame = f), showCorrectTrackFrame);
  wireZoom("correct-track", "correct-track-overlay-img");
  wireFrameWheel("correct-track", "correct-track-overlay-img");
  wireRenderMode("correct-track", () => showCorrectTrackFrame(state.correctTrackFrame));
  document.getElementById("correct-track-show-labels").addEventListener("change", () => showCorrectTrackFrame(state.correctTrackFrame));
  document.getElementById("correct-track-label-size-input").addEventListener("change", refreshCorrectTrackImage);
  wireColorMode("correct-track", () => {
    syncTrackColorModePanels();
    showCorrectTrackFrame(state.correctTrackFrame);
  });
  syncTrackColorModePanels();
  document.getElementById("correct-track-identity-palette-select").addEventListener("change", refreshCorrectTrackImage);
  document.getElementById("correct-track-environment-lut-select").addEventListener("change", refreshCorrectTrackImage);
  document.querySelectorAll('input[name="correct-track-env-direction"]').forEach((el) => {
    el.addEventListener("change", refreshCorrectTrackImage);
  });
  document.getElementById("lifecycle-tint-others-checkbox").addEventListener("change", refreshCorrectTrackImage);
  document.getElementById("dynamics-axes-checkbox").addEventListener("change",
    () => showTrajectoryFrame(state.trajectoryFrame));
  document.getElementById("t1-axis-filter-select").addEventListener("change", async (e) => {
    await window.EpiQuantAPI.set_t1_axis_filter(e.target.value);
    await showTrajectoryFrame(state.trajectoryFrame);
  });
  ["t1-enter-input", "t1-exit-input"].forEach((id) => {
    document.getElementById(id).addEventListener("change", async () => {
      const enter = parseFloat(document.getElementById("t1-enter-input").value);
      const exit = parseFloat(document.getElementById("t1-exit-input").value);
      const both = Number.isFinite(enter) && Number.isFinite(exit);
      try {
        await window.EpiQuantAPI.set_t1_hysteresis(both ? enter : null, both ? exit : null);
        setStatus("dynamics-t1-counts", "", "");
      } catch (err) {
        // enter <= exit is rejected by the backend; say so rather than
        // silently leaving the old band in place.
        setStatus("dynamics-t1-counts", String(err && err.message ? err.message : err), "error");
        return;
      }
      await showTrajectoryFrame(state.trajectoryFrame);
    });
  });
  document.getElementById("t1-persist-input").addEventListener("change", async (e) => {
    const applied = await window.EpiQuantAPI.set_t1_persistence(parseInt(e.target.value, 10));
    if (applied) e.target.value = applied.value;
    await showTrajectoryFrame(state.trajectoryFrame);
  });
  wirePopover("dynamics-help-btn", "dynamics-help-close", "dynamics-help-popover");
  ["dynamics-division-color-input", "dynamics-delamination-color-input", "dynamics-t1-color-input"]
    .forEach((id) => document.getElementById(id).addEventListener("change", async () => {
      await window.EpiQuantAPI.set_dynamics_colors(
        document.getElementById("dynamics-division-color-input").value,
        document.getElementById("dynamics-delamination-color-input").value,
        document.getElementById("dynamics-t1-color-input").value);
      await showTrajectoryFrame(state.trajectoryFrame);
    }));
  document.getElementById("save-lineage-btn").addEventListener("click", async () => {
    const r = await window.EpiQuantAPI.save_lineage();
    setStatus("trajectory-status", r && r.path
      ? `Lineage sheet saved: ${r.tracks} tracks, ${r.divisions} divisions, ${r.delaminations} delaminations.`
      : `Could not save: ${(r && r.error) || "unknown error"}`, r && r.path ? "success" : "error");
  });
  ["lifecycle-lost-color-input", "lifecycle-new-color-input", "lifecycle-both-color-input"].forEach((id) => {
    document.getElementById(id).addEventListener("change", async () => {
      await window.EpiQuantAPI.set_lifecycle_colors(
        document.getElementById("lifecycle-lost-color-input").value,
        document.getElementById("lifecycle-new-color-input").value,
        document.getElementById("lifecycle-both-color-input").value
      );
      await refreshCorrectTrackImage();
    });
  });
  document.getElementById("correct-track-overlay-img").addEventListener("click", handleCorrectTrackImageClick);
  document.getElementById("apply-track-correction-btn").addEventListener("click", handleApplyTrackCorrection);
  document.getElementById("suggest-match-btn").addEventListener("click", handleSuggestMatch);
  document.getElementById("check-suggest-holds-btn").addEventListener("click", handleCheckSuggestHolds);
  document.getElementById("apply-confirmed-moves-btn").addEventListener("click", handleApplyConfirmedMoves);
  document.getElementById("compute-stop-frame-btn").addEventListener("click", handleComputeStopFrame);
  document.getElementById("apply-smart-stop-btn").addEventListener("click", handleApplySmartStop);
  document.getElementById("track-swap-mode-checkbox").addEventListener("change", resetTrackSwapSelection);
  document.getElementById("apply-track-swap-btn").addEventListener("click", handleApplyTrackSwap);
  document.getElementById("census-start-btn").addEventListener("click", handleStartCensus);
  document.getElementById("census-kind-select").addEventListener("change", (e) => {
    document.getElementById("census-frame-wrap").style.display =
      e.target.value === "frame_pair" ? "inline-block" : "none";
  });
  document.getElementById("census-correct-btn").addEventListener("click", () => submitCensus("correct"));
  document.getElementById("census-wrong-btn").addEventListener("click", () => beginCensusWrong());
  document.getElementById("census-unjudgeable-btn").addEventListener("click", () => submitCensus("unjudgeable"));
  document.getElementById("census-summary-btn").addEventListener("click", handleCensusSummary);
  document.addEventListener("keydown", handleCensusKeyDown);
  document.addEventListener("keyup", handleCensusKeyUp);
  document.getElementById("coverage-summary-btn").addEventListener("click", handleShowCoverageSummary);
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
  document.querySelectorAll('input[name="trajectory-color-mode"]').forEach((el) => {
    el.addEventListener("change", () => showTrajectoryFrame(state.trajectoryFrame));
  });
  wirePopover("agreement-help-btn", "agreement-help-close", "agreement-help-popover");
  document.getElementById("agreement-register-input").addEventListener("change", async (e) => {
    await window.EpiQuantAPI.set_register_reference(e.target.checked);
    await refreshRegistrationStatus();
    await refreshCorrectTrackImage();
  });
  ["environment-deadband-input", "environment-repair-input"].forEach((id) => {
    document.getElementById(id).addEventListener("change", async () => {
      const applied = await window.EpiQuantAPI.set_environment_thresholds(
        parseFloat(document.getElementById("environment-deadband-input").value),
        parseFloat(document.getElementById("environment-repair-input").value));
      if (applied) {
        document.getElementById("environment-deadband-input").value = applied.dead_band;
        document.getElementById("environment-repair-input").value = applied.repair_threshold;
      }
      await refreshCorrectTrackImage();
    });
  });
  ["agrees", "disagrees", "unjudgeable", "broken"].forEach((state_) => {
    const el = document.getElementById(`agreement-${state_}-color-input`);
    if (el) el.addEventListener("change", async () => {
      await window.EpiQuantAPI.set_agreement_colors(
        document.getElementById("agreement-agrees-color-input").value,
        document.getElementById("agreement-disagrees-color-input").value,
        document.getElementById("agreement-unjudgeable-color-input").value,
        document.getElementById("agreement-broken-color-input").value);
      await refreshCorrectTrackImage();
    });
  });
  document.getElementById("trajectory-overlay-img").addEventListener("click", handleTrajectoryImageClick);

  // Global callbacks pushed from Python via window.evaluate_js
  window.onProgress = onProgress;
  window.onExtractDone = onExtractDone;
  window.onExtractError = onExtractError;
  window.onShapeExtractDone = onShapeExtractDone;
  window.onShapeExtractError = onShapeExtractError;
  window.onTrackDone = onTrackDone;
  window.onRetrackDone = onRetrackDone;
  window.onRetrackError = onRetrackError;
  window.onTrackError = onTrackError;

  // Seed the tracking-cost and T1-persistence widgets from the backend rather
  // than trusting the markup's defaults, so the app never shows a setting it
  // is not actually using.
  window.EpiQuantAPI.get_tracking_cost().then((cost) => {
    if (cost) syncTrackingCostControls(cost);
  });
  window.EpiQuantAPI.get_t1_persistence().then((p) => {
    if (!p) return;
    document.getElementById("t1-persist-input").value = p.value;
    if (p.enter_px != null) document.getElementById("t1-enter-input").value = p.enter_px;
    if (p.exit_px != null) document.getElementById("t1-exit-input").value = p.exit_px;
    if (p.axis_filter) document.getElementById("t1-axis-filter-select").value = p.axis_filter;
  });
  window.EpiQuantAPI.get_environment_thresholds().then((t) => {
    if (!t) return;
    document.getElementById("environment-deadband-input").value = t.dead_band;
    document.getElementById("environment-repair-input").value = t.repair_threshold;
  });
}

document.addEventListener("DOMContentLoaded", init);
