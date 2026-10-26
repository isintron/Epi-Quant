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
  // Round 34: the Data playground's Cell map has its own player.
  cellmapFrame: 1,

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

// The step the user is on. One reader, because Ctrl/Cmd+S dispatches on it
// and a second opinion about "which step is this" is how a save ends up in
// the wrong file (Round 29 §8).
function currentStep() {
  const active = document.querySelector(".step-section.active");
  return active ? active.id.replace(/^step-/, "") : null;
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
    // R35 2-2: `get_existing_results` has always reported these, and nothing
    // ever loaded them -- so a project with Cells_shape.xlsx and
    // Cells_tissue.xlsx on disk came up with neither, and none of their
    // columns appeared in Add value. Loaded before classification, which may
    // want them as features.
    if (existing.shape || existing.tissue) {
      const summary = await window.EpiQuantAPI.load_existing_shape_features();
      state.featureOptions = null;
      // `onExtractDone` above already built the value catalog, from
      // `Cells_info` alone -- so without this the shape and tissue columns
      // were loaded and usable but absent from every picker.
      await loadValueCatalog();
      await refreshPlaygroundCapabilities();
      if (summary && summary.families) {
        setStatus("setup-status", `Loaded ${Object.keys(summary.families).join(", ")} features…`);
      }
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

  // Applied on `data_input` before extraction starts, so every loader in the
  // run sees the same label set (Round 19 D2).
  const minArea = parseInt(document.getElementById("min-cell-area-input").value, 10) || 0;
  await window.EpiQuantAPI.set_min_cell_area(minArea);
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
  state.cellmapFrame = state.frameNumbers[0] ?? 1;
  // R36 8-1: the Data playground opens here, not after tracking. Static crops
  // and sparsely sampled series have nothing to track and were locked out of
  // the analyses they exist for.
  await unlockPlaygroundAfterExtract();
  // Report the removals, per frame and as a total. A filter whose removals
  // cannot be seen cannot be judged too aggressive (Round 19 D4) — the same
  // rule as the T1 event filter and the junction-length minimum.
  let removalNote = "";
  const removed = await window.EpiQuantAPI.get_min_area_removals();
  if (removed && removed.total > 0) {
    const perFrame = Object.entries(removed.by_frame)
      .map(([f, n]) => `frame ${f}: ${n}`).join(", ");
    removalNote = ` ${removed.total} cell(s) below ${removed.min_cell_area_px} px removed `
      + `from the masks (${perFrame}).`;
  } else if (removed && removed.min_cell_area_px > 0) {
    removalNote = ` No cell fell below ${removed.min_cell_area_px} px.`;
  }
  // "Not measured" is not "nothing removed" (Round 22 §D).
  if (removed && (removed.unreadable_frames || []).length) {
    removalNote += ` Frame(s) ${removed.unreadable_frames.join(", ")} hold no filled`
      + ` label array, so the filter could not be applied there.`;
  }
  setStatus("extract-status",
    `Done: ${data.frame_count} frames, ${data.cell_count} cells extracted.${removalNote}`, "success");
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
  return idx === -1 ? `${frame}` : `${idx + 1} / ${state.frameNumbers.length}`;
}

// A position reads `12 / 86`, the index into the frames that have data. On a
// sparse dataset that is NOT the frame number (`get_frame_numbers` can return
// [1, 9, 19]), so the frame number goes in the title rather than being dropped.
function setFrameLabel(prefix, frame) {
  const el = document.getElementById(`${prefix}-frame-label`);
  if (!el) return;
  el.textContent = frameLabel(frame);
  el.title = `Frame ${frame}`;
}

// --- The player: ONE template, five instances ------------------------------
//
// `index.html` carried five hand-copied frame-nav + zoom-controls blocks, and
// hand-copied markup drifts: the same round that found this also found a
// legend that had been left out of one lookup table. The JS was already shared
// (`wireFrameNav`/`wireZoomControls` derive every id from the prefix), so only
// the markup needed collapsing. Change a word here and all five change.
//
// Transport is icons: these five symbols are the one set of controls that
// needs no words. Everything that is a real setting (fps) keeps its label.
// Step uses the double-triangle glyphs, not the single ones: a single right
// triangle next to Play renders as the SAME shape, and two identical-looking
// buttons side by side is worse than the words they replaced. Checked in the
// browser, which is where it showed up.
const PLAYER_BUTTONS = [
  ["first-frame", "&#9198;", "First frame (Home)"],
  ["prev-frame", "&#9194;", "Previous frame (Left arrow)"],
  ["play-pause", "&#9654;", "Play / pause (Space)"],
  ["next-frame", "&#9193;", "Next frame (Right arrow)"],
  ["last-frame", "&#9197;", "Last frame (End)"],
];

function playerMarkup(prefix) {
  const buttons = PLAYER_BUTTONS.map(([suffix, glyph, title]) =>
    `<button id="${prefix}-${suffix}" class="icon-btn" title="${title}" aria-label="${title}">${glyph}</button>`
  ).join("");
  return `
    <span class="player-transport">${buttons}</span>
    <span id="${prefix}-frame-label" class="player-position"></span>
    <label class="player-rate"><input type="number" id="${prefix}-fps-input"
      class="small-input fps-input" value="2" min="0.5" max="30" step="0.5"> fps</label>
    <span class="player-zoom">
      <button id="${prefix}-zoom-out" class="icon-btn" title="Zoom out" aria-label="Zoom out">&minus;</button>
      <input type="range" id="${prefix}-zoom-slider" class="zoom-slider" min="25" max="400" step="25" value="100">
      <span id="${prefix}-zoom-level" class="zoom-level" role="button" tabindex="0"
            title="Click to reset to 100%">100%</span>
      <button id="${prefix}-zoom-in" class="icon-btn" title="Zoom in" aria-label="Zoom in">+</button>
    </span>
    <button id="${prefix}-player-help-btn" class="help-btn" aria-label="Keyboard shortcuts">?</button>
    <div id="${prefix}-player-help-popover" class="popover" style="display:none">
      <div class="popover-head">
        <strong>Player</strong>
        <button id="${prefix}-player-help-close" class="help-btn" aria-label="Close">&times;</button>
      </div>
      <dl>
        <dt>&larr; / &rarr;</dt><dd>previous / next frame</dd>
        <dt>Space</dt><dd>play / pause</dd>
        <dt>Home / End</dt><dd>first / last frame</dd>
      </dl>
      <p>The position reads <b>index / total</b> over the frames that have data; hover it for the
        frame number. Click the zoom percentage to reset it to 100%.</p>
      <p>While the magnifier is open the arrows step <b>it</b>, not the movie &mdash; close it
        (Esc) to get them back. During a census queue the keys belong to the census.</p>
    </div>`;
}

// Runs before any wiring: every `getElementById` below depends on these nodes.
function instantiatePlayers() {
  document.querySelectorAll("[data-player]").forEach((host) => {
    host.innerHTML = playerMarkup(host.dataset.player);
  });
}

function playerPrefixes() {
  return Array.from(document.querySelectorAll("[data-player]")).map((el) => el.dataset.player);
}

// The player on the step currently being shown. `goToStep` displays one step
// section at a time, so at most one player is laid out; `offsetParent` is the
// cheapest honest test for "is this actually on screen".
function activePlayerPrefix() {
  const host = Array.from(document.querySelectorAll("[data-player]"))
    .find((el) => el.offsetParent !== null);
  return host ? host.dataset.player : null;
}

function playerVisible() {
  return activePlayerPrefix() !== null;
}

// Nothing here knows about the census or the magnifier, and nothing here may
// learn: `KEY_OWNERS` has already decided that this owner is only asked when
// neither of them wanted the key.
function playerKeyDown(event) {
  const prefix = activePlayerPrefix();
  if (!prefix) return false;
  const press = (suffix) => {
    const button = document.getElementById(`${prefix}-${suffix}`);
    if (!button) return false;
    button.click();
    return true;
  };
  switch (event.key) {
    case "ArrowLeft": return press("prev-frame");
    case "ArrowRight": return press("next-frame");
    case " ": return press("play-pause");
    case "Home": return press("first-frame");
    case "End": return press("last-frame");
    default: return false;
  }
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
  setFrameLabel("extract", frame);
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
        // Absent from this project's table (extracted before the column
        // existed): shown so the option is discoverable, never ticked.
        const unavailable = (spec.unavailable || []).includes(column);
        const remembered = state.featureSelection[column];
        const on = !unavailable && (remembered === undefined ? !off : remembered);
        const id = `feature-cb-${column.replace(/[^A-Za-z0-9]/g, "-")}`;
        // `value` stays the real column name; only the visible text is shortened.
        const shown = (spec.display && spec.display[column]) || column;
        const title = unavailable
          ? `${column} - not in this project's Cells_info.xlsx; re-run Extract to use it`
          : column;
        return `<label class="checkbox-label" title="${title}"><input type="checkbox" id="${id}" value="${column}"${on ? " checked" : ""}${unavailable ? " disabled" : ""}> ${shown}</label>`;
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
      group.querySelectorAll("input[type=checkbox]:not(:disabled)").forEach((cb) => {
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
  const varianceRaw = document.getElementById("variance-threshold").value;
  const varianceThreshold = varianceRaw === "" ? 0.85 : parseFloat(varianceRaw);

  if (!isEfd && selected.length < 2) {
    setStatus("classify-status", "Select at least two features.", "error");
    return;
  }

  setStatus("classify-status", "Running PCA + K-means…");
  document.getElementById("run-classification-btn").disabled = true;
  try {
    const result = await window.EpiQuantAPI.run_classification(
      nClusters, kRange, null, families, selected, prune, datasetMode, varianceThreshold,
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
  setFrameLabel("classify", frame);
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
  setFrameLabel("correct-class", frame);
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

// Round 29 §8. ONE handler decides what Ctrl/Cmd+S saves. Two handlers would
// be two places that evolve separately until one day they disagree -- and the
// failure mode there is writing the classification result into the tracking
// file, which destroys data rather than merely annoying someone.
//
// This is the FRONT half of the defence. The back half is in the API: the two
// save methods check the shape of what they are about to write and refuse a
// table of the wrong kind. `currentStep()` is a thing that gets refactored;
// the backend's refusal is not.
function saveKeyDown(event) {
  if (!((event.key === "s" || event.key === "S") && (event.ctrlKey || event.metaKey))) return false;
  const step = currentStep();
  if (step === "correct-classification") { handleSaveClassificationCorrections(); return true; }
  if (step === "correct-tracking") { handleSaveTrackingCorrections(); return true; }
  // Any other step: not ours. Returning false lets the next owner look, and
  // nothing is written.
  return false;
}

async function handleSaveClassificationCorrections() {
  const result = await window.EpiQuantAPI.save_classification();
  if (result && result.error) {
    setStatus("correct-class-selected-cell", result.error, "error");
    return;
  }
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
    `${describeTrackCount(data)} total.`, "success");

  await refreshTrackNumbers();
}

function onRetrackError(data) {
  document.getElementById("start-retrack-btn").disabled = false;
  setStatus("retrack-status", `Re-track failed: ${data.message}`, "error");
}

// The track set and the search dropdown are two views of one fact, and they
// were updated in different places: `onTrackDone` refreshed both, `onRetrackDone`
// only the array, and the correction handlers refreshed neither. So a track
// created by hand (e.g. #501 on `embryos_GBE_GAP43`, which really does have 501
// tracks) was missing from `state.trackNumbers`, and `loadTrack` answered
// "Track 501 not found." for a track the backend returns correctly. Two call
// sites that must agree, one of them fixed and the other not -- `CLAUDE.md`
// rule 6. There is one function now, and every mutation calls it.
async function refreshTrackNumbers() {
  state.trackNumbers = await window.EpiQuantAPI.get_track_numbers();
  const list = document.getElementById("track-number-list");
  if (!list) return state.trackNumbers;
  list.innerHTML = "";
  state.trackNumbers.forEach((tn) => {
    const opt = document.createElement("option");
    opt.value = tn;
    list.appendChild(opt);
  });
  return state.trackNumbers;
}

// A row of the tracking table is not necessarily a track: a stored table can
// carry rows that claim no cell in any frame (`AS_LE_lateral view_E-cad` has
// 37 of 145, and they are like that on disk). Every plot counts the ones that
// hold cells, so the Track step said 145 while the plots said 108. Say both.
function describeTrackCount(data) {
  const rows = data.track_count;
  const withCells = data.tracks_with_cells;
  const empty = data.empty_rows;
  if (withCells === undefined || !empty) return `${rows} tracks`;
  return `${withCells} tracks (${empty} of the ${rows} rows hold no cell and are not shown)`;
}

async function onTrackDone(data) {
  const dupNote = data.duplicates_cleaned ? ` (cleaned ${data.duplicates_cleaned} duplicate track claims from this file)` : "";
  setStatus("track-status", `Done: ${describeTrackCount(data)}.${dupNote}`, "success");
  document.getElementById("start-tracking-btn").disabled = false;
  document.getElementById("track-next-choices").style.display = "flex";
  enableStep("correct-tracking");
  enableStep("trajectory");

  await refreshTrackNumbers();
  // Round 34: the value catalog, the time window and the PC gate all depend
  // on there being a tracking table, so they are refreshed here rather than
  // on every entry to a plotting step.
  await refreshRound34Data();

  state.correctTrackFrame = state.frameNumbers[0] ?? 1;
  state.cellmapFrame = state.frameNumbers[0] ?? 1;
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
// Every panel that belongs to exactly one colour mode. A control missing from
// this map is not "always relevant" -- it is a control nothing ever hides: the
// matched/unmatched legend sat outside the map and stayed on screen under
// Environment score, explaining colours that view does not use (Round 27 §1a).
// The `?` that replaced it lives INSIDE `correct-track-uniform-colors`, so it
// is hidden with the row it explains and cannot drift out of the map again.
const TRACK_COLOR_MODE_PANELS = {
  uniform: ["correct-track-uniform-colors"],
  identity: ["correct-track-identity-palette"],
  // `correct-track-environment-hint` is NOT listed: it became a popover
  // (Round 29 §1) and the sync below sets `display`, which would force it
  // open. It is closed explicitly when the mode changes, like the colour one.
  environment: ["correct-track-environment"],
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

// The persistence horizon is stored in SECONDS and shown in both units: the
// same 150 s is 5 frames at 30 s/frame and 10 at 15 s/frame, so a frame count
// alone does not say what was required. The input's own max comes from the
// backend for the same reason -- it used to be a literal 10, which was exactly
// the requirement of a 15 s/frame movie, with no headroom for a faster one.
// Round 31 C: seconds per frame and microns per pixel are properties of the
// ACQUISITION, entered once in Extract and persisted with the project. Both
// may stay empty, and empty must behave exactly as before -- so this only
// displays what the backend reports and never substitutes a default.
function syncAcquisition(values) {
  if (!values) return;
  const interval = document.getElementById("frame-interval-input");
  const pixel = document.getElementById("pixel-size-input");
  if (interval && values.frame_interval_sec != null) interval.value = values.frame_interval_sec;
  if (pixel && values.pixel_size_um != null) pixel.value = values.pixel_size_um;
  const text = [
    values.frame_interval_sec ? `${values.frame_interval_sec} s/frame` : "frame interval unset",
    values.pixel_size_um ? `${values.pixel_size_um} µm/px` : "pixel size unset",
  ].join(" · ");
  ["acquisition-hint", "acquisition-readout"].forEach((id) => {
    const node = document.getElementById(id);
    if (node) node.textContent = text;
  });
  syncT1Persistence(values);
  window.EpiQuantAPI.t1_length_settings().then(syncLengthFilter);
}

function wireAcquisitionInputs() {
  document.getElementById("frame-interval-input").addEventListener("change", async (e) => {
    const value = parseFloat(e.target.value);
    syncAcquisition(await window.EpiQuantAPI.set_frame_interval(Number.isFinite(value) ? value : null));
  });
  document.getElementById("pixel-size-input").addEventListener("change", async (e) => {
    const value = parseFloat(e.target.value);
    syncAcquisition(await window.EpiQuantAPI.set_pixel_size(Number.isFinite(value) ? value : null));
  });
}

// The threshold is in TRUE pixels. Beside it: where it came from (this
// movie's sweep, or the user), and its length in microns when -- and only
// when -- Extract has a pixel size.
function syncLengthFilter(settings) {
  if (!settings) return;
  const span = document.getElementById("t1-length-px");
  if (!span) return;
  const parts = [];
  if (settings.min_length_px != null) {
    if (settings.source === "derived from this movie") parts.push(`auto ${settings.min_length_px} px`);
    if (settings.min_length_um != null) parts.push(`= ${settings.min_length_um} µm`);
  } else if (settings.source === "not derived yet") {
    parts.push("auto: derived from this movie when switched on");
  }
  span.textContent = parts.length ? ` (${parts.join(", ")})` : "";
}

function syncT1Persistence(bounds) {
  const input = document.getElementById("t1-persist-input");
  if (bounds.value != null) input.value = bounds.value;
  else if (bounds.frames != null) input.value = bounds.frames;
  if (bounds.max_frames) input.max = bounds.max_frames;
  const interval = document.getElementById("frame-interval-input");
  if (interval && bounds.frame_interval_sec != null) interval.value = bounds.frame_interval_sec;
  const hint = document.getElementById("t1-persist-seconds");
  if (hint) {
    hint.textContent = bounds.frame_interval_sec
      ? ` = ${Math.round(parseInt(input.value, 10) * bounds.frame_interval_sec)} s`
      : " (set the frame interval to see this in seconds)";
  }
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
  const gateScale = document.getElementById("gate-scale-input");
  if (gateScale && cost.gate_scale !== undefined) gateScale.value = cost.gate_scale;
  const drift = document.getElementById("drift-register-input");
  if (drift) drift.checked = cost.drift_mode && cost.drift_mode !== "none";
}

function syncTrackColorModePanels() {
  const mode = getColorMode("correct-track");
  // A popover left open while the mode changes would outlive the row that
  // opened it. Closed here rather than listed above, because the sync sets
  // `display` and would force this one visible.
  const colorHelp = document.getElementById("correct-track-color-help-popover");
  if (colorHelp && mode !== "uniform") colorHelp.style.display = "none";
  const envHelp = document.getElementById("correct-track-environment-hint");
  if (envHelp && mode !== "environment") envHelp.style.display = "none";
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
  await refreshEnvironmentCounts(environmentDirection);
  await refreshRegistrationStatus();
}

// What the environment overlay's colours are, and are not, about. Dice
// saturates on small neighbour sets -- two cells each keeping one comparable
// neighbour, the same one, score a perfect 1.0 -- so a healthy-looking median
// is exactly what a degraded input produces. The share of perfect scores and
// the count of cells that could not be scored at all are the two numbers that
// show it, and the unscoreable ones are omitted from the result rather than
// scored 0, so without this line they are invisible (Round 28 C3).
async function refreshEnvironmentCounts(direction) {
  const box = document.getElementById("correct-track-environment-counts");
  if (getColorMode("correct-track") !== "environment") {
    box.style.display = "none";
    return;
  }
  const info = await window.EpiQuantAPI.get_environment_repair_candidates(
    state.correctTrackFrame, direction);
  box.style.display = "block";
  if (!info || !info.n) {
    box.textContent = info && info.unscoreable
      ? `No cell in this frame could be scored (${info.unscoreable} unscoreable).`
      : "No environment score for this frame.";
    return;
  }
  const perfect = info.perfect ?? 0;
  box.textContent =
    `${info.n} cell(s) scored — ${perfect} of them a perfect 1.0 ` +
    `(${Math.round(100 * perfect / info.n)}%), ${info.flagged_total ?? 0} below the dead band, ` +
    `${info.candidates.length} at or below the repair threshold. ` +
    `${info.unscoreable ?? 0} cell(s) could not be scored at all and are uncoloured.`;
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
  // Lead with the DECISION in plain terms; the arithmetic goes in the popover.
  // `ratio` and `threshold` are internals of `registration_decision` — a user
  // reading them cannot tell whether anything is wrong or what to do about it
  // (Round 19 B3).
  const px = Math.round(r.median_drift_px);
  const share = r.median_cell_radius_px
    ? r.median_drift_px / (2 * r.median_cell_radius_px) : 0;
  const size = share >= 0.5 ? "more than half a cell width"
    : share >= 0.25 ? "about a quarter of a cell width"
    : "a small fraction of a cell width";
  el.textContent = r.register
    ? `The tissue drifts about ${px} px per frame — ${size} — so registration is on, `
      + `and the agreement reference ${r.applied ? "is" : "would be"} measured on registered masks too.`
    : `The tissue drifts about ${px} px per frame — ${size} — small enough that registration `
      + `is not needed here.`;
  const help = document.getElementById("drift-help-btn");
  if (help) {
    help.style.display = "inline-block";
    const body = document.getElementById("drift-help-numbers");
    if (body) {
      body.textContent =
        `Measured: ${r.median_drift_px} px median drift / ${r.median_cell_radius_px} px median cell `
        + `radius = ${r.ratio}, against a threshold of ${r.threshold}. Registration is `
        + `${r.register ? "on" : "off"} and ${r.applied ? "applied" : "not applied"}.`;
    }
  }
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
  // What the filter removes from the drawing, always shown next to it. The
  // default is `persistent` (Round 16 D), and since Round 17 C3 the filter
  // hides the cell FILL as well as the axis -- so this line has to say "T1s
  // shown", not "axes drawn", or it understates what disappeared.
  const shownCounts = events.event_counts;
  const axisNote = shownCounts
    ? ` T1s drawn ${shownCounts.shown} of ${shownCounts.total} ` +
      `(${shownCounts.filtered} hidden by "${shownCounts.filter}", ` +
      `${shownCounts.unjudgeable} can't judge) — cell and axis together.`
    : "";
  // Round 32: the two optional filters, each with what it removed, beside
  // the raw numbers they never change.
  const lf = events.length_filter;
  const lengthNote = lf && lf.active
    ? ` Interface ≥ ${lf.min_length_px} px` +
      (lf.min_length_um != null ? ` (${lf.min_length_um} µm)` : "") +
      ` [${lf.source}]: removed ${lf.gains_removed} gained / ${lf.losses_removed} lost.`
    : "";
  const quartetNote = events.quartet_filter
    ? ` Quartet check: ${events.quartet_rejected} gain(s) rejected` +
      ` (${events.quartet_orphaned_losses} paired loss(es) hidden with them).`
    : "";
  if (hold > 0) {
    readout.textContent =
      `T1 this frame: raw ${events.gained} gained / ${events.lost} lost; ` +
      `settled ${events.gained_persistent} / ${events.lost_persistent} after ${hold} frame(s); ` +
      `can't judge ${events.gained_unjudgeable || 0} / ${events.lost_unjudgeable || 0}.` + axisNote +
      lengthNote + quartetNote;
  } else {
    readout.textContent =
      `T1 this frame: ${events.gained} gained / ${events.lost} lost (no persistence filter).` + axisNote +
      lengthNote + quartetNote;
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
  closeTrackStepper();
  state.correctTrackFrame = frame;
  state.correctTrackShownFrame = frame;
  state.correctTrackShownAt = Date.now();
  setFrameLabel("correct-track", frame);
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

// --- Docked frame stepper (Round 16 E) ------------------------------------
//
// The reported problem: after clicking a cell to edit it, the UI could only
// render that frame -- stepping forward or back lost the selection. This panel
// walks the selected track's frames on its own. It never writes
// `state.correctTrackFrame` and never touches `state.selectedTrackCell`, which
// is the whole point and is asserted in the click-through.
const stepperState = { trackId: null, anchor: null, frame: null, held: [], frames: [] };

function stepperVisible() {
  const panel = document.getElementById("track-stepper");
  // `hidden` alone is not enough: leave the magnifier open, switch to another
  // step, and the attribute is still false while the panel is not on screen --
  // it would then claim the arrows on a page it is not even visible on. And
  // `offsetParent` is null for a `position: fixed` element whether or not it
  // is shown, so the test is "does it occupy any box at all".
  return !panel.hidden && panel.getClientRects().length > 0;
}

// Where the magnifier sits and how big it is. Session-only and deliberately
// NOT persisted to disk, for the same reason "Enable inset view" is not: it is
// a view preference, and nothing downstream reads it.
const stepperWindow = { left: null, top: null, width: null, height: null };

// A remembered position is only valid against the window it was taken in. Make
// the panel smaller, or move it to another screen, and a remembered corner can
// land outside the viewport -- the magnifier would then open invisibly, with
// no way for the user to get it back. So every open clamps, and so does every
// window resize while it is open.
function clampStepperIntoView() {
  const panel = document.getElementById("track-stepper");
  if (!panel || panel.hidden) return;
  const margin = 8;
  const rect = panel.getBoundingClientRect();
  const width = stepperWindow.width || rect.width;
  const height = stepperWindow.height || rect.height;
  let left = stepperWindow.left;
  let top = stepperWindow.top;
  if (left === null || top === null) return;        // never moved: CSS places it
  left = Math.min(Math.max(margin, left), Math.max(margin, window.innerWidth - width - margin));
  top = Math.min(Math.max(margin, top), Math.max(margin, window.innerHeight - height - margin));
  stepperWindow.left = left;
  stepperWindow.top = top;
  Object.assign(panel.style, {
    left: `${left}px`, top: `${top}px`, right: "auto", bottom: "auto",
  });
}

function applyStepperWindow() {
  const panel = document.getElementById("track-stepper");
  if (stepperWindow.width) panel.style.width = `${stepperWindow.width}px`;
  if (stepperWindow.height) panel.style.height = `${stepperWindow.height}px`;
  if (stepperWindow.left !== null && stepperWindow.top !== null) {
    Object.assign(panel.style, {
      left: `${stepperWindow.left}px`, top: `${stepperWindow.top}px`,
      right: "auto", bottom: "auto",
    });
  }
  clampStepperIntoView();
}

// Dragging by the head. Pointer events rather than mouse events so a drag that
// leaves the window still ends (`setPointerCapture`).
function wireStepperWindow() {
  const panel = document.getElementById("track-stepper");
  const head = document.querySelector(".track-stepper-head");
  if (!panel || !head) return;
  let dragging = false;
  let offsetX = 0;
  let offsetY = 0;
  head.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || e.target.closest("button")) return;
    const rect = panel.getBoundingClientRect();
    dragging = true;
    offsetX = e.clientX - rect.left;
    offsetY = e.clientY - rect.top;
    stepperWindow.width = rect.width;
    stepperWindow.height = rect.height;
    head.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  head.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    stepperWindow.left = e.clientX - offsetX;
    stepperWindow.top = e.clientY - offsetY;
    applyStepperWindow();
  });
  const end = (e) => {
    if (!dragging) return;
    dragging = false;
    if (head.hasPointerCapture(e.pointerId)) head.releasePointerCapture(e.pointerId);
  };
  head.addEventListener("pointerup", end);
  head.addEventListener("pointercancel", end);
  // The CSS resize handle changes the size without telling anyone, so read it
  // back when it settles -- otherwise the remembered size is the pre-resize one
  // and the next open would snap back.
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(() => {
      if (panel.hidden) return;
      const rect = panel.getBoundingClientRect();
      stepperWindow.width = rect.width;
      stepperWindow.height = rect.height;
    }).observe(panel);
  }
  window.addEventListener("resize", clampStepperIntoView);
}

// C2c (R17): the inset can be switched off without deselecting the cell.
// Default on; the choice survives re-selection for the rest of the session
// (deliberately not persisted to disk -- it is a view preference, not a
// setting anything downstream reads).
function insetEnabled() {
  const box = document.getElementById("enable-inset-checkbox");
  return !box || box.checked;
}

async function openTrackStepper(trackId, anchorFrame) {
  if (!insetEnabled()) {
    closeTrackStepper();
    return;
  }
  stepperState.trackId = trackId;
  stepperState.anchor = anchorFrame;
  stepperState.frame = anchorFrame;
  // Both toggles start from the main view's current setting, so the panel
  // opens showing what the user is already looking at.
  document.getElementById("track-stepper-labels").checked =
    document.getElementById("correct-track-show-labels").checked;
  // Deliberately NOT synced from the main view's render mode: the inset's
  // outlines start off regardless (Round 19 B2a).
  const strip = await window.EpiQuantAPI.get_track_filmstrip(trackId);
  if (strip && strip.error) {
    closeTrackStepper();
    return;
  }
  stepperState.frames = strip.frames || [];
  stepperState.held = strip.held || [];
  document.getElementById("track-stepper").hidden = false;
  applyStepperWindow();
  document.getElementById("track-stepper-title").textContent =
    `Track #${trackId} — ${stepperState.held.length} of ${stepperState.frames.length} frames`;
  renderFilmstrip();
  await showStepperFrame(anchorFrame);
}

function closeTrackStepper() {
  document.getElementById("track-stepper").hidden = true;
  stepperState.trackId = null;
}

function renderFilmstrip() {
  const strip = document.getElementById("track-stepper-filmstrip");
  strip.innerHTML = "";
  const held = new Set(stepperState.held);
  stepperState.frames.forEach((frame) => {
    const tick = document.createElement("span");
    if (held.has(frame)) tick.classList.add("held");
    if (frame === stepperState.frame) tick.classList.add("current");
    tick.title = `Frame ${frame}${held.has(frame) ? "" : " — no cell"}`;
    tick.addEventListener("click", () => showStepperFrame(frame));
    strip.appendChild(tick);
  });
}

async function showStepperFrame(frame) {
  if (stepperState.trackId === null) return;
  const mainFrameBefore = state.correctTrackFrame;
  const result = await window.EpiQuantAPI.get_track_step_frame(
    stepperState.trackId, stepperState.anchor, frame,
    document.getElementById("track-stepper-labels").checked,
    document.getElementById("track-stepper-outline").checked,
    parseInt(document.getElementById("correct-track-label-size-input").value, 10) || 10,
    28,
    document.getElementById("track-stepper-outline-color").value);
  if (result && result.error) {
    document.getElementById("track-stepper-window-note").textContent = result.error;
    return;
  }
  stepperState.frame = frame;
  document.getElementById("track-stepper-img").src = result.image;
  document.getElementById("track-stepper-frame").textContent =
    `${frameLabel(frame)}${result.held ? "" : " — no cell"}`;
  // Which of the two questions this crop is answering. Never silent: a
  // re-centred crop holds the cell still and moves the background, which is
  // right for "where does this track go" and wrong for "did it swap".
  document.getElementById("track-stepper-window-note").textContent =
    (result.recentred ? "⤢ " : "▣ ") + result.window_note;
  renderFilmstrip();
  // The main view must not have moved. Kept as an assertion in the code
  // because "stepping loses the selection" is the bug this panel exists for.
  if (state.correctTrackFrame !== mainFrameBefore) {
    console.error("the stepper moved the main view; that is the bug it exists to avoid");
  }
}

function stepTrackStepper(delta) {
  const frames = stepperState.frames;
  const index = frames.indexOf(stepperState.frame);
  const next = frames[Math.min(Math.max(index + delta, 0), frames.length - 1)];
  if (next !== undefined && next !== stepperState.frame) showStepperFrame(next);
}

// Returns true when it consumed the key. See `dispatchKey` for the ownership
// rule -- this function no longer decides for itself whether it is allowed to
// run, which is what used to make the arrows die (Round 19 B2b).
function stepperKeyDown(event) {
  if (event.key === "Escape") { closeTrackStepper(); return true; }
  if (event.key === "ArrowLeft") { stepTrackStepper(-1); return true; }
  if (event.key === "ArrowRight") { stepTrackStepper(1); return true; }
  return false;
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
  // "census" records verdicts and never touches the tracking data;
  // "correct" queues edits and never writes ground truth (Round 16 C2).
  mode: "census", pending: 0,
  candidatePage: 0,
};

// The three overlay colours always travel together: sending two of them would
// leave the third at whatever the backend last heard (Round 26 C).
async function pushTrackColors() {
  await window.EpiQuantAPI.set_track_colors(
    document.getElementById("matched-color-input").value,
    document.getElementById("unmatched-color-input").value,
    document.getElementById("unextracted-color-input").value);
}

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

// Round 25 B3: the anchor's outline on every candidate tile, modelled on
// "Enable inset view". Default on; the choice survives re-selection for the
// rest of the session (deliberately not persisted to disk -- it is a view
// preference, not a setting anything downstream reads). A missing box means ON,
// so the dev mock and any path that never renders the control cannot silently
// turn it off.
function anchorViewEnabled() {
  const box = document.getElementById("enable-anchor-view-checkbox");
  return !box || box.checked;
}

// Round 25 D1: push the Candidates / R controls to the backend. Returns false
// (and says so) if the backend refused them.
async function applyCandidateSettings() {
  const mode = document.getElementById("census-candidate-mode-select").value;
  const k = parseFloat(document.getElementById("census-radius-k-input").value);
  document.getElementById("census-radius-k-wrap").style.display = mode === "radius" ? "" : "none";
  const applied = await window.EpiQuantAPI.set_candidate_settings(mode, Number.isNaN(k) ? null : k);
  if (applied && applied.error) {
    setStatus("census-status", applied.error, "error");
    return false;
  }
  return true;
}

async function loadCensusItem(position) {
  const item = await window.EpiQuantAPI.get_census_item(position, 28, anchorViewEnabled());
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
  censusState.candidatePage = 0;
  censusState.shownAt = Date.now();
  document.getElementById("census-candidates").style.display = "none";
  document.getElementById("census-candidates").innerHTML = "";
  censusShow("before");
  const ious = (item.reference_iou === null || item.reference_iou === undefined)
    ? "" : ` (this link overlaps ${item.tracker_iou}; the reference's pick overlaps ${item.reference_iou})`;
  const pendingNote = censusState.mode === "correct" && censusState.pending
    ? ` — ${censusState.pending} correction(s) pending` : "";
  setStatus("census-progress",
    `${position + 1} of ${item.total} — frame ${item.frame} cell ${item.label}` +
    `${item.track_id === null ? "" : `, track #${item.track_id}`}${ious}${pendingNote}`, "");
}

async function handleStartCensus() {
  const kind = document.getElementById("census-kind-select").value;
  const mode = document.getElementById("census-mode-select").value;
  const frame = parseInt(document.getElementById("census-frame-input").value, 10);
  const labeller = document.getElementById("census-labeller-input").value;
  setStatus("census-status", "Building the queue…", "");
  if (!(await applyCandidateSettings())) return;
  const started = await window.EpiQuantAPI.start_census(
    kind, kind === "frame_pair" ? frame : null, 0.15, 0.10, 0, labeller, mode);
  if (started && started.error) {
    setStatus("census-status", started.error, "error");
    return;
  }
  censusState.active = true;
  censusState.mode = started.mode || mode;
  censusState.pending = 0;
  document.getElementById("census-panel").style.display = "block";
  document.getElementById("census-apply-report").innerHTML = "";
  // Undo `finishCensusPanel`: a new queue gets the per-item controls back.
  for (const id of ["census-correct-btn", "census-wrong-btn", "census-unjudgeable-btn"]) {
    document.getElementById(id).style.display = "inline-block";
  }
  document.getElementById("census-image").style.display = "block";
  const correcting = censusState.mode === "correct";
  document.getElementById("census-apply-btn").style.display = correcting ? "inline-block" : "none";
  document.getElementById("census-undo-btn").style.display = "none";
  document.getElementById("census-summary-btn").style.display = correcting ? "none" : "inline-block";
  document.getElementById("census-correct-btn").textContent =
    correcting ? "J — leave as is" : "J — correct";
  // The convention, and what it costs. The queue size is a property of the
  // tracking run, not of the UI: on GBE, both sides unregistered gives 255
  // items and both registered gives 25 — the same movie, 10x apart. Say why
  // the number is what it is instead of shrinking it (Round 18 D1).
  const conv = started.register ? "registered" : "unregistered";
  const comparison = started.register
    ? ""
    : ` This tracking ran without registration; with drift_mode="mask" the same set is far smaller.`;
  const drifted = started.setting_changed_since_tracking
    ? ` ⚠ The drift setting is now "${started.current_drift_setting}" but this tracking ran` +
      ` "${started.tracking_drift_mode}" — the queue follows the run, not the setting.`
    : "";
  setStatus("census-status", correcting
    ? `${started.n_real} links to review. Corrections are held as pending and applied in one ` +
      `batch — nothing is written to ground truth in this mode. Reference computed on ` +
      `${conv} masks, matching the tracking run.${comparison}${drifted}`
    : `${started.n_real} links to judge (${started.n} items including the hidden checks). ` +
      `Overlap reference computed on ${conv} masks — the convention this tracking run used.` +
      `${comparison}${drifted} Saving to ${started.path}`, "success");
  await loadCensusItem(0);
}

// Pages of three (Round 20 C2). Even with a correct ranking the answer is
// sometimes fourth, so `4` cycles. Digits mean THE TILE IN THAT POSITION, not
// an absolute index — the tiles are labelled 1–3 on every page.
const CANDIDATES_PER_PAGE = 3;

function censusPageCount() {
  const n = ((censusState.item && censusState.item.candidates) || []).length;
  // ceil(n / 3) on the REAL length: 5 candidates is two pages, not a padded three.
  return Math.max(1, Math.ceil(n / CANDIDATES_PER_PAGE));
}

function beginCensusWrong() {
  const candidates = (censusState.item && censusState.item.candidates) || [];
  if (!candidates.length) {
    submitCensus("wrong", null);
    return;
  }
  censusState.awaitingCandidate = true;
  censusState.candidatePage = 0;
  renderCensusCandidates();
}

function renderCensusCandidates() {
  const candidates = (censusState.item && censusState.item.candidates) || [];
  const pages = censusPageCount();
  const page = ((censusState.candidatePage % pages) + pages) % pages;
  censusState.candidatePage = page;
  const start = page * CANDIDATES_PER_PAGE;
  const shown = candidates.slice(start, start + CANDIDATES_PER_PAGE);

  const box = document.getElementById("census-candidates");
  box.innerHTML = "";
  const hint = document.createElement("span");
  hint.className = "hint";
  // Round 20 read the list LENGTH as a signal — short meant the rankings
  // agreed, long meant they disagreed. That reading died with Round 22 §F:
  // the two hardcoded depths of 4 were what made the length vary (3–6), and
  // with `limit=12` actually binding, GBE's environment queue is 11 or 12 long
  // on 109 of 109 items and `AS_dorsal`'s on 54 of 55. A phrase that fires on
  // every item is not a signal, so it is gone rather than always-on.
  const search = (censusState.item && censusState.item.candidate_search) || {};
  // Radius mode says what the list IS: everything within R, nothing chosen.
  const basis = search.mode === "radius"
    ? ` All ${candidates.length} cell(s) within ${search.radius_px} px ` +
      `(${search.k} × ${search.cell_radius_px} px cell radius) of the registered anchor, nearest first.`
    : "";
  hint.textContent =
    `Which cell is it? 1–${shown.length}` +
    (pages > 1 ? `, 4 for the next page` : "") + ", or 0 for none of these. " +
    `Showing ${start + 1}–${start + shown.length} of ${candidates.length}.${basis}`;
  box.appendChild(hint);

  // Tiles in their own row, so the question above them and the buttons below
  // them are not flex siblings of the tiles (Round 21 B).
  const tiles = document.createElement("div");
  tiles.className = "census-tiles";
  shown.forEach((candidate, i) => {
    const wrap = document.createElement("div");
    wrap.className = "census-candidate";
    const img = document.createElement("img");
    img.src = candidate.image;
    const caption = document.createElement("span");
    caption.textContent = `${i + 1} — cell ${candidate.label}`;
    // Radius mode keeps the tracker's own answer in the list (Round 25 D2) and
    // marks it, so the user sees which rank it holds instead of it vanishing.
    if (candidate.is_tracker_answer) wrap.classList.add("census-candidate-tracker");
    // The evidence on its OWN line (Round 23 C1): `1 — cell 40 (44.6 px from
    // the tracker's answer)` on one line puts the choice and its justification
    // in one run of text, and the tile is narrow, so it wrapped mid-phrase.
    // Why it was offered still matters — three implausible tiles should tell
    // the user whether the ranking is off or the cell genuinely left the field.
    const why = document.createElement("span");
    why.className = "census-candidate-why";
    why.textContent = candidate.why || "";
    wrap.appendChild(img); wrap.appendChild(caption);
    if (candidate.why) wrap.appendChild(why);
    wrap.addEventListener("click", () => submitCensus("wrong", candidate.label));
    tiles.appendChild(wrap);
  });
  box.appendChild(tiles);

  const actions = document.createElement("div");
  actions.className = "census-candidate-actions";
  if (pages > 1) {
    const more = document.createElement("button");
    more.className = "secondary-btn";
    more.textContent = `4 — more (page ${page + 1} of ${pages})`;
    more.addEventListener("click", () => { censusState.candidatePage += 1; renderCensusCandidates(); });
    actions.appendChild(more);
  }
  const none = document.createElement("button");
  none.className = "secondary-btn";
  none.textContent = "0 — none of these";
  none.addEventListener("click", () => submitCensus("wrong", null));
  actions.appendChild(none);
  box.appendChild(actions);
  box.style.display = "flex";
}

async function submitCensus(verdict, chosenLabel = null) {
  if (!censusState.active || !censusState.item) return;
  const elapsed = Date.now() - censusState.shownAt;
  const position = censusState.position;
  // Picking the tracker's own answer (offered in radius mode) is not a
  // correction: the link was right. Recorded as `correct` WITH the chosen label,
  // so its source and rank still land in the verdict file (Round 25 D2).
  const pickedTrackerAnswer = verdict === "wrong" && chosenLabel !== null
    && chosenLabel === censusState.item.shown_label;
  if (pickedTrackerAnswer) verdict = "correct";

  // Correction mode never calls `submit_census_verdict` — the backend refuses
  // it, and relying on that refusal rather than on this branch is the point:
  // the two write paths must not depend on the UI to stay apart.
  if (censusState.mode === "correct") {
    if (verdict === "wrong" && !pickedTrackerAnswer) {
      const queued = await window.EpiQuantAPI.queue_census_edit(position, chosenLabel);
      if (queued && queued.error) {
        setStatus("census-status", queued.error, "error");
        return;
      }
      censusState.pending = queued.pending;
    }
    await advanceCensus(position + 1);
    return;
  }

  const result = await window.EpiQuantAPI.submit_census_verdict(
    position, verdict, chosenLabel, elapsed);
  if (result && result.error) {
    setStatus("census-status", result.error, "error");
    return;
  }
  await advanceCensus(result.next);
}

async function advanceCensus(next) {
  if (next >= censusState.total) {
    // A COMPLETED state, not a hidden panel. `#census-apply-btn` lives inside
    // `#census-panel`, so hiding the panel at the end of a correction queue
    // took the Apply button with it: the work was done and there was no way to
    // commit it (Round 19 C1). A correction session must end at the control
    // that commits it.
    censusState.active = false;
    finishCensusPanel();
    if (censusState.mode === "correct") {
      setStatus("census-status",
        `Reviewed all ${censusState.total} items — ${censusState.pending} correction(s) pending. ` +
        "Nothing has been written yet — press Apply to commit them.", "success");
      if (censusState.pending) await handlePreviewCensusEdits();
    } else {
      setStatus("census-status", `Census complete — ${censusState.total} items.`, "success");
      await handleCensusSummary();
    }
    return;
  }
  await loadCensusItem(next);
}

// The queue is done: hide the per-item controls, keep the panel and the
// controls that COMMIT the work.
function finishCensusPanel() {
  document.getElementById("census-panel").style.display = "block";
  for (const id of ["census-correct-btn", "census-wrong-btn", "census-unjudgeable-btn"]) {
    document.getElementById(id).style.display = "none";
  }
  document.getElementById("census-image").style.display = "none";
  document.getElementById("census-frame-tag").textContent = "";
  document.getElementById("census-candidates").style.display = "none";
  const correcting = censusState.mode === "correct";
  const apply = document.getElementById("census-apply-btn");
  apply.style.display = correcting ? "inline-block" : "none";
  apply.disabled = correcting && censusState.pending === 0;
  document.getElementById("census-summary-btn").style.display = correcting ? "none" : "inline-block";
  setStatus("census-progress",
    `${censusState.total} of ${censusState.total} reviewed` +
    (correcting ? ` — ${censusState.pending} pending correction(s)` : ""), "success");
}

function censusKeyDown(event) {
  const key = event.key.toLowerCase();
  if (key === " " || event.code === "Space") {
    if (!censusState.blinking) { censusState.blinking = true; censusShow("after"); }
    return true;
  }
  if (censusState.awaitingCandidate && "01234".includes(event.key)) {
    const candidates = censusState.item.candidates || [];
    const index = parseInt(event.key, 10);
    if (index === 4) {
      // Next page, wrapping. `0` stays available on every page.
      if (censusPageCount() > 1) { censusState.candidatePage += 1; renderCensusCandidates(); }
      return true;
    }
    if (index === 0) { submitCensus("wrong", null); return true; }
    const offset = (censusState.candidatePage || 0) * CANDIDATES_PER_PAGE;
    submitCensus("wrong", (candidates[offset + index - 1] || {}).label ?? null);
    return true;
  }
  if (key === "j") { submitCensus("correct"); return true; }
  if (key === "k") { beginCensusWrong(); return true; }
  if (key === "l") { submitCensus("unjudgeable"); return true; }
  // Claimed and deliberately ignored. The movie behind this panel must not
  // page while a link is being judged, and the only way to say that in this
  // design is to consume the key here. The magnifier still steps on the arrows
  // when it is open, because it is a HIGHER-priority owner than the census.
  if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return true;
  return false;
}

function censusKeyUp(event) {
  if (event.key === " " || event.code === "Space") {
    censusState.blinking = false;
    censusShow("before");
    return true;
  }
  return false;
}

// --- One keyboard dispatcher (Round 19 B2b) --------------------------------
//
// There used to be two independent global `keydown` listeners with no
// ownership rule between them, and the stepper's handler defended itself with
// `if (censusState.active) return`. That flag stays true when a queue is
// abandoned rather than finished, so the arrow keys died for the rest of the
// session and nothing said why.
//
// The fix is not a better guard inside one of them. It is having ONE place
// that answers "who owns this key right now": owners are tried in priority
// order, each returns whether it consumed the key, and an unconsumed key falls
// through to the next owner. So the census owns J/K/L/space/digits while its
// queue is up, and the arrows still reach the stepper underneath it.
// Priority order IS the rule. A handler that re-states it -- `if
// (stepperVisible()) return;` inside the player, say -- is the self-defence
// this dispatcher replaced, and the second copy is the one that goes stale.
//
// stepper > census is deliberate and is the older decision: "the census owns
// J/K/L/space/digits while its queue is up, and the arrows still reach the
// stepper underneath it". The stepper only claims Escape and the arrows and
// only while it is open, so J/K/L reach the census regardless.
//
// The census then SWALLOWS the arrows (see `censusKeyDown`): with the player
// added as an owner below, an unconsumed arrow would page the movie behind the
// census panel while the user judges. Swallowing is the census's own decision
// about its own keys, stated once, in its handler -- not a question the player
// asks about the census.
const KEY_OWNERS = [
  { name: "stepper", active: () => stepperVisible(), down: stepperKeyDown, up: null },
  { name: "census", active: () => censusState.active, down: censusKeyDown, up: censusKeyUp },
  { name: "inspection", active: () => inspectionActive(), down: inspectionKeyDown, up: null },
  // `X` (swap mode) and Ctrl/Cmd+S (save) claim nothing the owners above use,
  // so they sit below them and above the player, which owns the arrows.
  { name: "swap", active: () => swapModeActive(), down: swapKeyDown, up: null },
  { name: "save", active: () => true, down: saveKeyDown, up: null },
  { name: "player", active: () => playerVisible(), down: playerKeyDown, up: null },
];

function typingInAField(target) {
  const tag = (target && target.tagName || "").toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select" || (target && target.isContentEditable);
}

function dispatchKey(event, phase) {
  if (typingInAField(event.target)) return;
  for (const owner of KEY_OWNERS) {
    if (!owner.active()) continue;
    const handler = phase === "down" ? owner.down : owner.up;
    if (handler && handler(event)) { event.preventDefault(); return; }
  }
}

async function handlePreviewCensusEdits() {
  const box = document.getElementById("census-apply-report");
  const preview = await window.EpiQuantAPI.preview_census_edits();
  box.innerHTML = "";
  if (preview && preview.error) {
    setStatus("census-summary", preview.error, "error");
    return null;
  }
  const line = (text, kind) => {
    const el = document.createElement("div");
    el.className = "status-line" + (kind ? " " + kind : "");
    el.textContent = text;
    box.appendChild(el);
  };
  line(`${preview.pending} pending correction(s), applied as one batch with one undo.`, "");
  line(preview.double_claimed.length
    ? `⚠ ${preview.double_claimed.length} cell(s) would be claimed by two tracks.`
    : "✓ No cell would be claimed by two tracks.",
    preview.double_claimed.length ? "error" : "success");
  line(preview.orphaned.length
    ? `⚠ ${preview.orphaned.length} track(s) would be left holding nothing.`
    : preview.truncated.length
      ? `${preview.truncated.length} track(s) would lose frames.`
      : "✓ No track would be truncated or orphaned.",
    preview.orphaned.length ? "error" : "");
  line(preview.evicts.length
    ? `⚠ ${preview.evicts.length} step(s) would relocate another track's future onto a new id ` +
      "(no clean inverse — the batch undo restores a snapshot instead)."
    : "✓ No step would relocate another track's future.",
    preview.evicts.length ? "error" : "success");
  if (preview.blocking) {
    line("Apply is blocked by the checks above.", "error");
  }
  return preview;
}

async function handleApplyCensusEdits() {
  const preview = await handlePreviewCensusEdits();
  if (!preview) return;
  if (preview.blocking) {
    setStatus("census-summary",
      "Not applied: the batch fails a structural check. Fix or drop the offending corrections.",
      "error");
    return;
  }
  const result = await window.EpiQuantAPI.apply_census_edits(false);
  if (result && result.error) {
    setStatus("census-summary", result.error, "error");
    return;
  }
  censusState.pending = 0;
  document.getElementById("census-undo-btn").style.display = "inline-block";
  const notes = (result.displaced_list || []).map(describeDisplacement).join("");
  await refreshTrackNumbers();
  setStatus("census-summary", `Applied ${result.applied} correction(s) in one batch.${notes}`, "success");
  await refreshCorrectTrackImage();
}

async function handleUndoCensusEdits() {
  const result = await window.EpiQuantAPI.undo_census_edits();
  if (result && result.error) {
    setStatus("census-summary", result.error, "error");
    return;
  }
  document.getElementById("census-undo-btn").style.display = "none";
  document.getElementById("census-apply-report").innerHTML = "";
  setStatus("census-summary", "Restored the tracking table to the snapshot taken before Apply.", "success");
  await refreshTrackNumbers();
  await refreshCorrectTrackImage();
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

// Round 29 §6: `X` for swap/exchange. NOT Ctrl/Cmd+W -- that is "close
// window" on macOS and "close tab" in a browser, so the native window would
// very likely be gone before JS saw the event, and even a successful
// preventDefault would be fighting a system-wide habit over unsaved work.
// `X` was grepped as unclaimed: the census owns J/K/L/digits/space, the
// magnifier the arrows and Esc, the inspection K/arrows/space/Esc.
function swapModeActive() {
  return currentStep() === "correct-tracking";
}

function setSwapMode(on) {
  const box = document.getElementById("track-swap-mode-checkbox");
  if (!box) return;
  box.checked = on;
  resetTrackSwapSelection();
  syncSwapModeIndicator();
}

// The mode has to be VISIBLE, or the user cannot tell whether the next click
// selects a cell or swaps two tracks -- which matters more than the shortcut.
function syncSwapModeIndicator() {
  const box = document.getElementById("track-swap-mode-checkbox");
  const on = !!(box && box.checked);
  const wrap = document.querySelector("#step-correct-tracking .image-click-wrap");
  if (wrap) wrap.classList.toggle("swap-mode-on", on);
  const label = box && box.closest("label");
  if (label) label.classList.toggle("swap-mode-label-on", on);
  // Round 32 UI-2: the status line follows the SAME state as the checkbox,
  // in both directions. It used to be written only when the mode came on, so
  // leaving with `X` left "Swap mode ON" on screen over an unticked box.
  // Turning off only replaces the line if it is still the swap message, so an
  // unrelated status (a selected cell, an error) is not clobbered.
  const status = document.getElementById("correct-track-selected-cell");
  if (on) {
    setStatus("correct-track-selected-cell",
      "Swap mode ON (X or Esc to leave) — click two matched cells to exchange their tracks.", "");
  } else if (status && status.textContent.startsWith("Swap mode ON")) {
    setStatus("correct-track-selected-cell", "Swap mode off.", "");
  }
}

function swapKeyDown(event) {
  const box = document.getElementById("track-swap-mode-checkbox");
  if (!box) return false;
  if (event.key === "x" || event.key === "X") {
    setSwapMode(!box.checked);
    return true;
  }
  // Esc leaves the mode, and only claims the key when there is a mode to leave.
  if (event.key === "Escape" && box.checked) {
    setSwapMode(false);
    return true;
  }
  return false;
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
    closeTrackStepper();
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
  // Smart-stop propagate is retired (Round 16 A): its line search never
  // discriminated (all three candidates scored 2.155% on the tested merge) and
  // its stop rule overshoots by one frame by construction. The button stays in
  // the DOM inside `#smart-stop-retired[hidden]` so the handlers below keep
  // working, but nothing shows it any more.

  document.getElementById("correct-track-suggest").style.display = "flex";
  document.getElementById("correct-track-suggest-list").style.display = "none";
  document.getElementById("correct-track-suggest-list").innerHTML = "";
  document.getElementById("correct-track-suggest-hint").textContent = "";

  // Round 16 E: the stepper opens on a matched cell, anchored at the frame the
  // user is on. An unmatched cell has no track to walk, so it closes instead.
  if (isMatched) {
    await openTrackStepper(result.track_id, state.correctTrackFrame);
  } else {
    closeTrackStepper();
  }
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
    await refreshTrackNumbers();
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
  await refreshTrackNumbers();
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
      await refreshTrackNumbers();
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
  document.getElementById("plan-checks").innerHTML = "";
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
    `Found ${moves.length} correction(s) needed through frame ${result.stop_frame} (${result.reason}). Verifying…`,
    "success"
  );

  // Round 16 B: nothing is offered for Apply until the plan has been run
  // through the structural checks and scored. Verification runs on a deep
  // copy; the live table is asserted unchanged on the Python side.
  const check = await window.EpiQuantAPI.verify_confirmed_identity_moves(trackId, moves, 1.5);
  if (check && check.error) {
    setStatus("suggest-check-result", `Cannot verify this plan: ${check.error}`, "error");
    list.style.display = "none";
    applyBtn.style.display = "none";
    state.confirmedMovesPending = null;
    return;
  }
  const verified = check.moves || moves;

  verified.forEach((move) => {
    const item = document.createElement("li");
    const takes = move.from_track_id !== null && move.from_track_id !== undefined;
    const fromText = takes ? `track #${move.from_track_id}` : "unmatched";
    // The gate is what admitted the step; the fraction of it used is the
    // readable number, and a step near 1.0 is the one most likely to be wrong.
    const gateText = move.gate_fraction === null || move.gate_fraction === undefined
      ? (move.distance == null ? "" : `, ${move.distance}px`)
      : `, ${move.distance}px of a ${move.gate_px}px gate (${Math.round(move.gate_fraction * 100)}%)`;
    item.textContent =
      `Frame ${move.frame}: currently ${fromText} → would become track #${trackId} (${move.cell_index}${gateText})`;
    // Taking a cell from another track is the only destructive action a plan
    // contains, and it used to look identical to a plain attach.
    if (takes) item.classList.add("move-takes-cell");
    if (move.gate_fraction !== null && move.gate_fraction !== undefined && move.gate_fraction > 0.8) {
      item.classList.add("move-near-gate");
      item.title = "This step only just fits inside the displacement gate.";
    }
    list.appendChild(item);
  });
  list.style.display = "block";
  renderPlanChecks(check);

  state.confirmedMovesPending = { trackId, moves };
  applyBtn.textContent = check.blocking
    ? `Apply anyway (${moves.length} correction(s))`
    : `Apply ${moves.length} correction(s)`;
  applyBtn.style.display = "inline-block";
}

// The three structural checks and the consistency delta, beside the moves.
// Shown, never auto-applied: TA's accept-and-revert is a batch repairer where
// a wrong revert is invisible; this is interactive, so the score assists the
// decision rather than making it.
function renderPlanChecks(check) {
  const box = document.getElementById("plan-checks");
  box.innerHTML = "";
  const line = (text, kind) => {
    const el = document.createElement("div");
    el.className = "status-line" + (kind ? " " + kind : "");
    el.textContent = text;
    box.appendChild(el);
  };

  line(check.double_claimed.length
    ? `⚠ Would leave ${check.double_claimed.length} cell(s) claimed by two tracks: ` +
      check.double_claimed.map((d) => `${d.cell_index} (tracks ${d.tracks.join(", ")})`).join("; ")
    : "✓ No cell would be claimed by two tracks.",
    check.double_claimed.length ? "error" : "success");

  const orphan = check.orphaned.length, truncate = check.truncated.length;
  line(orphan
    ? `⚠ Would leave ${orphan} track(s) holding nothing: ` +
      check.orphaned.map((o) => `#${o.track_id}`).join(", ")
    : truncate
      ? `${truncate} track(s) would lose frames: ` +
        check.truncated.map((t) => `#${t.track_id} (−${t.lost_frames.length})`).join(", ")
      : "✓ No track would be truncated or orphaned.",
    orphan ? "error" : "");

  line(check.evicts.length
    ? `⚠ ${check.evicts.length} step(s) would relocate another track's future onto a new id. ` +
      "That has no clean inverse — Save Corrections before applying."
    : "✓ No step would relocate another track's future.",
    check.evicts.length ? "error" : "success");

  if (check.takes_from.length) {
    line(`${check.takes_from.length} step(s) take a cell from another track (marked in red above).`, "");
  }

  const c = check.consistency;
  if (c) {
    const sign = c.delta > 0 ? "+" : "";
    line(
      `Consistency under transposition over frames ${c.frames[0]}–${c.frames[c.frames.length - 1]}: ` +
      `${c.before_unconfirmed} → ${c.after_unconfirmed} of ${c.n} links unconfirmed (${sign}${c.delta}). ` +
      `Masks ${c.registered ? "registered" : "unregistered"}, same convention both sides. ` +
      `This is a consistency delta, not an accuracy — it re-solves with the same cost function the ` +
      `tracker used, so it says whether the result is self-consistent, not whether it is right.`,
      c.delta > 0 ? "error" : "");
  } else {
    line("Consistency delta not computed: fewer than two of the affected frames have masks.", "");
  }
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
  document.getElementById("plan-checks").innerHTML = "";
  state.confirmedMovesPending = null;
  await refreshTrackNumbers();
  await refreshCorrectTrackImage();
  setStatus("suggest-check-result", `Applied ${moves.length} correction(s).${notes}`, "success");
}

async function handleSaveTrackingCorrections() {
  const result = await window.EpiQuantAPI.save_tracking();
  if (result && result.error) {
    setStatus("correct-track-selected-cell", result.error, "error");
    return;
  }
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
// `trusted` skips the membership check for callers whose track number came
// from the BACKEND in the first place (the inspection pass). That check exists
// to catch a typo in the number box; against a backend-supplied id it is one
// more chance for a stale frontend cache to veto something that does exist --
// and it vetoed silently, which is how the inspection pass could stop dead on
// its first track (Round 28 A4).
async function loadTrack(tn, { preserveFrame = false, trusted = false } = {}) {
  if (!trusted && !state.trackNumbers.includes(tn)) {
    setStatus("trajectory-status", `Track ${tn} not found.`, "error");
    return false;
  }
  state.currentTrackNumber = tn;
  document.getElementById("track-number-input").value = tn;
  const range = await window.EpiQuantAPI.get_track_frame_range(tn);
  if (!range) {
    setStatus("trajectory-status", "No frame data for this track.", "error");
    return false;
  }
  const current = state.trajectoryFrame;
  const keep = preserveFrame && current >= range.min_frame && current <= range.max_frame;
  const frame = keep ? current : range.min_frame;
  state.trajectoryFrame = frame;
  await showTrajectoryFrame(frame);
  // Round 34, 4-1: the chosen values survive a change of cell; only the cell
  // changes. Clicking a cell in the image goes through here too.
  await refreshTrajectoryCharts();
  return true;
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
  setFrameLabel("trajectory", frame);
  const modeEl = document.querySelector('input[name="trajectory-color-mode"]:checked');
  const colorMode = modeEl ? modeEl.value : "highlight";
  document.getElementById("correct-track-dynamics").style.display = colorMode === "dynamics" ? "flex" : "none";
  // Round 32 T1-4: the highlight view needs a track. With none selected (or
  // an empty track list) there is nothing to draw, so say so and do not
  // call -- the backend used to crash on `int(None)` here.
  if (colorMode === "highlight" && state.currentTrackNumber == null) {
    setPreviewImage("trajectory-overlay-img", null);
    setStatus("trajectory-status", "Select a track to highlight it.", "");
    return;
  }
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

// --- Visual inspection of every track (Round 27 4b) -------------------
//
// Each track loops over its own frames at the player's fps while the user
// watches; `K` marks the cell currently on screen. The mark is a CELL, not a
// track -- `(track, frame, cell)` says where the track goes wrong, which a
// track-level flag does not. The cell index itself is resolved in the backend
// from the tracking table, not sent from here.
const inspectionState = {
  active: false, tracks: [], index: 0, frames: [], frame: null,
  timer: null, paused: false, marks: 0, seen: new Set(),
  trackStartedAt: 0, perTrackMs: {}, markStartedAt: 0,
};

function inspectionActive() {
  return inspectionState.active;
}

function inspectionFps() {
  const input = document.getElementById("trajectory-fps-input");
  return Math.max(0.5, parseFloat(input && input.value) || 2);
}

function stopInspectionTimer() {
  if (inspectionState.timer) clearInterval(inspectionState.timer);
  inspectionState.timer = null;
}

function startInspectionTimer() {
  stopInspectionTimer();
  if (inspectionState.paused) return;
  inspectionState.timer = setInterval(async () => {
    const frames = inspectionState.frames;
    if (!frames.length) return;
    const idx = frames.indexOf(inspectionState.frame);
    // Loops rather than stopping at the end: the point is to watch the track
    // over and over until you are sure, not to reach its last frame.
    inspectionState.frame = frames[(idx + 1) % frames.length];
    await showTrajectoryFrame(inspectionState.frame);
  }, 1000 / inspectionFps());
}

function describeInspectionProgress(note) {
  const track = inspectionState.tracks[inspectionState.index];
  setStatus("inspection-status",
    `Track ${inspectionState.index + 1} of ${inspectionState.tracks.length} ` +
    `(#${track})${inspectionState.paused ? " — paused" : ""}. ` +
    `K marks this cell, ↑/↓ change track, ←/→ step frames (pauses), Space pauses, Esc ends. ` +
    `${inspectionState.marks} cell(s) marked.${note ? " " + note : ""}`, "");
}

async function inspectionGoToIndex(index) {
  if (!inspectionState.active) return;
  const tracks = inspectionState.tracks;
  const next = Math.min(Math.max(0, index), tracks.length - 1);
  // Dwell time belongs to the track that was on screen, so bank it before moving.
  const leaving = tracks[inspectionState.index];
  if (inspectionState.trackStartedAt && leaving !== undefined) {
    inspectionState.perTrackMs[leaving] =
      (inspectionState.perTrackMs[leaving] || 0) + (Date.now() - inspectionState.trackStartedAt);
  }
  inspectionState.index = next;
  inspectionState.trackStartedAt = Date.now();
  inspectionState.markStartedAt = Date.now();
  const track = tracks[next];
  inspectionState.seen.add(track);
  // `trusted`: this id came from the backend's own list (see
  // `startTrackInspection`), so the frontend cache gets no vote.
  const loaded = await loadTrack(track, { trusted: true });
  const range = await window.EpiQuantAPI.get_track_frame_range(track);
  inspectionState.frames = range
    ? state.frameNumbers.filter((f) => f >= range.min_frame && f <= range.max_frame)
    : [];
  // Always restarts at the track's first frame, so every track is watched from
  // the same place rather than from wherever the previous one happened to stop.
  inspectionState.frame = inspectionState.frames[0] ?? null;

  // A pass that cannot show anything must SAY so and stop. It used to keep
  // advancing its clock and its index with the picture frozen, which from the
  // outside is indistinguishable from a player that will not play (Round 28 A4).
  if (!loaded || inspectionState.frame === null) {
    stopInspectionTimer();
    const why = !loaded
      ? `track #${track} could not be loaded`
      : !range
        ? `track #${track} has no frame range`
        : state.frameNumbers.length === 0
          ? "this project has no frame list loaded - re-run Extract, or reload the project"
          : `no frame of this project falls in track #${track}'s range ` +
            `(${range.min_frame}-${range.max_frame}; frames go ` +
            `${state.frameNumbers[0]}-${state.frameNumbers[state.frameNumbers.length - 1]})`;
    setStatus("inspection-status",
      `Visual inspection stopped at track ${next + 1} of ${tracks.length}: ${why}. ` +
      `Press Finish inspection to write what has been marked so far.`, "error");
    return;
  }
  await showTrajectoryFrame(inspectionState.frame);
  describeInspectionProgress();
  startInspectionTimer();
}

async function startTrackInspection() {
  const labeller = document.getElementById("inspection-labeller-input").value;
  const started = await window.EpiQuantAPI.start_track_inspection(labeller);
  // `null` is a real answer here, not an impossible one: the bridge returns it
  // when a call does not reach the backend. Reading `.tracks` off it throws
  // inside a promise, which surfaces only in the console -- the user just sees
  // a button that did nothing.
  if (!started || started.error) {
    setStatus("inspection-status",
      (started && started.error) || "Could not start the inspection (no response from the backend).",
      "error");
    return;
  }
  Object.assign(inspectionState, {
    active: true, tracks: started.tracks, index: 0, marks: 0,
    seen: new Set(), perTrackMs: {}, paused: false, trackStartedAt: Date.now(),
  });
  // The pass walks the BACKEND's list, so bring the frontend's copy (and the
  // search dropdown) up to the same truth before starting rather than leaving
  // two lists to disagree for the rest of the session.
  await refreshTrackNumbers();
  document.getElementById("start-inspection-btn").style.display = "none";
  document.getElementById("stop-inspection-btn").style.display = "inline-block";
  await inspectionGoToIndex(0);
}

async function markInspectionCell() {
  const track = inspectionState.tracks[inspectionState.index];
  const elapsed = Date.now() - (inspectionState.markStartedAt || Date.now());
  const result = await window.EpiQuantAPI.mark_inspection_cell(
    track, inspectionState.frame, elapsed);
  inspectionState.markStartedAt = Date.now();
  if (!result || result.error) {
    describeInspectionProgress(`⚠ ${(result && result.error) || "no response from the backend"}`);
    return;
  }
  inspectionState.marks = result.marks;
  describeInspectionProgress(`Marked ${result.marked}.`);
}

async function finishTrackInspection() {
  if (!inspectionState.active) return;
  stopInspectionTimer();
  const leaving = inspectionState.tracks[inspectionState.index];
  if (inspectionState.trackStartedAt && leaving !== undefined) {
    inspectionState.perTrackMs[leaving] =
      (inspectionState.perTrackMs[leaving] || 0) + (Date.now() - inspectionState.trackStartedAt);
  }
  const summary = await window.EpiQuantAPI.finish_track_inspection(
    Array.from(inspectionState.seen), inspectionState.perTrackMs);
  inspectionState.active = false;
  document.getElementById("start-inspection-btn").style.display = "inline-block";
  document.getElementById("stop-inspection-btn").style.display = "none";
  if (!summary || summary.error) {
    setStatus("inspection-status",
      (summary && summary.error) || "The inspection ended, but the backend returned no summary.",
      "error");
    return;
  }
  const unseen = summary.tracks_not_inspected || [];
  // The tracks NOT reached are part of the result, not an omission from it:
  // a pass that stopped a third of the way through says something different
  // from one that finished, and only this line distinguishes them.
  const unseenNote = unseen.length
    ? ` Not inspected: ${unseen.length} track(s) (#${unseen[0]}–#${unseen[unseen.length - 1]}).`
    : " Every track was inspected.";
  setStatus("inspection-status",
    `Inspected ${summary.tracks_inspected} of ${summary.tracks_total} track(s). ` +
    `${summary.marks} cell(s) marked across ${summary.tracks_with_marks} track(s). ` +
    `Median ${summary.median_seconds_per_track ?? "?"} s per track.${unseenNote} ` +
    `Written to ${summary.path} with its provenance.`, "success");
}

// Step one frame inside the current track, and PAUSE. Without the pause the
// next tick overwrites the frame a few hundred milliseconds later, so the key
// looks like it did nothing.
async function inspectionStepFrame(delta) {
  const frames = inspectionState.frames;
  if (!frames.length) return;
  inspectionState.paused = true;
  stopInspectionTimer();
  const at = frames.indexOf(inspectionState.frame);
  const next = Math.min(Math.max(0, (at === -1 ? 0 : at) + delta), frames.length - 1);
  inspectionState.frame = frames[next];
  await showTrajectoryFrame(inspectionState.frame);
  describeInspectionProgress();
}

function inspectionKeyDown(event) {
  switch (event.key) {
    case "k": case "K":
      markInspectionCell();
      return true;
    // Up/Down change TRACK (the maintainer's request). `<` `>` `,` `.` stay as
    // aliases: the unshifted pair is easier to hold down, and keeping them
    // costs nothing.
    case "ArrowDown": case ">": case ".":
      inspectionGoToIndex(inspectionState.index + 1);
      return true;
    case "ArrowUp": case "<": case ",":
      inspectionGoToIndex(inspectionState.index - 1);
      return true;
    // Left/Right MUST be consumed here. Unconsumed they fell through to the
    // player -- which is visible on the Trajectory step -- so the loop and the
    // player were animating the same `<img>` from two different places. That
    // is the conflict, not the choice of keys (Round 28 B2).
    case "ArrowLeft":
      inspectionStepFrame(-1);
      return true;
    case "ArrowRight":
      inspectionStepFrame(1);
      return true;
    case " ":
      inspectionState.paused = !inspectionState.paused;
      if (inspectionState.paused) stopInspectionTimer(); else startInspectionTimer();
      describeInspectionProgress();
      return true;
    case "Escape":
      finishTrackInspection();
      return true;
    default:
      return false;
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
  if (playBtn) { playBtn.innerHTML = "&#9654;"; playBtn.title = "Play (Space)"; }
}

function startAutoplay(prefix, stepFn) {
  const fpsInput = document.getElementById(`${prefix}-fps-input`);
  const fps = Math.max(0.5, parseFloat(fpsInput.value) || 2);
  state.playIntervals[prefix] = setInterval(stepFn, 1000 / fps);
  const playing = document.getElementById(`${prefix}-play-pause`);
  playing.innerHTML = "&#9208;";
  playing.title = "Pause (Space)";
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
  // The percentage is the reset control (Round 27 §4a): one fewer button, and
  // the thing you want back to 100% is the thing you click.
  const level = document.getElementById(`${prefix}-zoom-level`);
  level.addEventListener("click", () => setZoom(100));
  level.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setZoom(100); }
  });
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
  instantiatePlayers();
  wireStepperWindow();
  playerPrefixes().forEach((prefix) => {
    wirePopover(`${prefix}-player-help-btn`, `${prefix}-player-help-close`,
                `${prefix}-player-help-popover`);
  });
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
  document.getElementById("gate-scale-input").addEventListener("change", async (e) => {
    const value = parseFloat(e.target.value);
    // Re-read what the backend applied: it rejects <= 0, and the box must never
    // show a scale the next tracking run will not use.
    let applied = null;
    if (value > 0) {
      applied = await window.EpiQuantAPI.set_tracking_cost(null, null, null, null, null, value);
    }
    if (applied) syncTrackingCostControls(applied);
    else syncTrackingCostControls(await window.EpiQuantAPI.get_tracking_cost());
  });
  document.getElementById("drift-register-input").addEventListener("change", async (e) => {
    const applied = await window.EpiQuantAPI.set_tracking_cost(
      null, null, null, null, e.target.checked ? "mask" : "none");
    if (applied) syncTrackingCostControls(applied);
    await refreshRegistrationStatus();
  });
  // Theme: session-only, deliberately not persisted -- the same reason
  // "Enable inset view" and the magnifier's geometry are not. It is a view
  // preference and nothing downstream reads it.
  document.getElementById("theme-select").addEventListener("change", (e) => {
    document.documentElement.dataset.theme = e.target.value;
  });
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
  document.getElementById("t1-event-filter-select").addEventListener("change", async (e) => {
    await window.EpiQuantAPI.set_t1_event_filter(e.target.value);
    await refreshT1Counts();
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
    if (applied) syncT1Persistence(applied);
    await showTrajectoryFrame(state.trajectoryFrame);
  });
  document.getElementById("t1-interface-input").addEventListener("change", async (e) => {
    await window.EpiQuantAPI.set_t1_min_interface_px(parseInt(e.target.value, 10));
    await showTrajectoryFrame(state.trajectoryFrame);
  });
  const applyLengthFilter = async () => {
    const box = document.getElementById("t1-length-checkbox");
    const px = parseFloat(document.getElementById("t1-length-input").value);
    // Empty means "use the value derived from this movie" (Round 33).
    const result = await window.EpiQuantAPI.set_t1_length_filter(
      box.checked, Number.isFinite(px) ? px : null, !Number.isFinite(px));
    syncLengthFilter(result);
    if (result && result.error) {
      // Refused, not quietly run in pixels: say why and leave it off. Nothing
      // changed, so do not redraw -- the redraw rewrites this status line
      // with the counts and the reason would vanish before it is read.
      box.checked = false;
      setStatus("dynamics-t1-counts", result.error, "error");
      return;
    }
    // Clear a previous refusal's error styling: `refreshT1Counts` rewrites
    // the text but not the class, which left the counts in red.
    setStatus("dynamics-t1-counts", "", "");
    await showTrajectoryFrame(state.trajectoryFrame);
  };
  document.getElementById("t1-length-checkbox").addEventListener("change", applyLengthFilter);
  document.getElementById("t1-length-input").addEventListener("change", applyLengthFilter);
  document.getElementById("t1-quartet-checkbox").addEventListener("change", async (e) => {
    await window.EpiQuantAPI.set_t1_quartet_filter(e.target.checked);
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
  document.getElementById("track-swap-mode-checkbox").addEventListener("change", () => {
    resetTrackSwapSelection();
    syncSwapModeIndicator();
  });
  document.getElementById("apply-track-swap-btn").addEventListener("click", handleApplyTrackSwap);
  document.getElementById("enable-inset-checkbox").addEventListener("change", async (e) => {
    if (!e.target.checked) {
      closeTrackStepper();
    } else if (state.selectedTrackCell && state.selectedTrackCell.track_id != null) {
      await openTrackStepper(state.selectedTrackCell.track_id, state.correctTrackFrame);
    }
  });
  document.getElementById("track-stepper-close").addEventListener("click", closeTrackStepper);
  document.getElementById("track-stepper-prev").addEventListener("click", () => stepTrackStepper(-1));
  document.getElementById("track-stepper-next").addEventListener("click", () => stepTrackStepper(1));
  ["track-stepper-labels", "track-stepper-outline", "track-stepper-outline-color"].forEach((id) => {
    document.getElementById(id).addEventListener("change", () => showStepperFrame(stepperState.frame));
  });

  document.getElementById("census-start-btn").addEventListener("click", handleStartCensus);
  // Switching generator or R re-fetches the current item: that IS a different
  // candidate set, unlike the anchor view below.
  ["census-candidate-mode-select", "census-radius-k-input"].forEach((id) => {
    document.getElementById(id).addEventListener("change", async () => {
      if (!(await applyCandidateSettings())) return;
      if (!censusState.active || !censusState.item) return;
      const reopen = censusState.awaitingCandidate;
      await loadCensusItem(censusState.position);
      if (reopen) beginCensusWrong();
    });
  });
  // The anchor view only re-draws. It must not re-run the generator, and the
  // backend's render-only call returns the label list so that is checked here
  // rather than assumed (Round 25 B3).
  document.getElementById("enable-anchor-view-checkbox").addEventListener("change", async (e) => {
    const item = censusState.item;
    if (!censusState.active || !item || !(item.candidates || []).length) return;
    const drawn = await window.EpiQuantAPI.get_census_candidate_images(censusState.position, e.target.checked);
    if (drawn && drawn.error) { setStatus("census-status", drawn.error, "error"); return; }
    const before = item.candidates.map((c) => c.label);
    if (drawn.labels.length !== before.length || drawn.labels.some((l, i) => l !== before[i])) {
      setStatus("census-status", "Anchor view changed the candidate list — refusing to redraw.", "error");
      return;
    }
    drawn.images.forEach((image, i) => { item.candidates[i].image = image; });
    if (censusState.awaitingCandidate) renderCensusCandidates();
  });
  document.getElementById("census-kind-select").addEventListener("change", (e) => {
    document.getElementById("census-frame-wrap").style.display =
      e.target.value === "frame_pair" ? "inline-block" : "none";
  });
  document.getElementById("census-correct-btn").addEventListener("click", () => submitCensus("correct"));
  document.getElementById("census-wrong-btn").addEventListener("click", () => beginCensusWrong());
  document.getElementById("census-unjudgeable-btn").addEventListener("click", () => submitCensus("unjudgeable"));
  document.getElementById("census-summary-btn").addEventListener("click", handleCensusSummary);
  document.getElementById("census-apply-btn").addEventListener("click", handleApplyCensusEdits);
  document.getElementById("census-undo-btn").addEventListener("click", handleUndoCensusEdits);
  // Exactly two global key listeners, both going through the one dispatcher.
  document.addEventListener("keydown", (event) => dispatchKey(event, "down"));
  document.addEventListener("keyup", (event) => dispatchKey(event, "up"));
  document.getElementById("coverage-summary-btn").addEventListener("click", handleShowCoverageSummary);
  document.getElementById("save-tracking-btn").addEventListener("click", handleSaveTrackingCorrections);
  document.getElementById("done-correcting-tracking").addEventListener("click", enterTrajectoryStep);
  document.getElementById("unextracted-color-input").addEventListener("change", async () => {
    await pushTrackColors();
    await showCorrectTrackFrame(state.correctTrackFrame);
  });
  document.getElementById("matched-color-input").addEventListener("change", async (e) => {
    await window.EpiQuantAPI.set_track_colors(e.target.value, document.getElementById("unmatched-color-input").value,
                                              document.getElementById("unextracted-color-input").value);
    await showCorrectTrackFrame(state.correctTrackFrame);
  });
  document.getElementById("unmatched-color-input").addEventListener("change", async (e) => {
    await window.EpiQuantAPI.set_track_colors(document.getElementById("matched-color-input").value, e.target.value,
                                              document.getElementById("unextracted-color-input").value);
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
  wirePopover("census-help-btn", "census-help-close", "census-help-popover");
  wirePopover("census-set-help-btn", "census-set-help-close", "census-set-help-popover");
  wirePopover("inspection-help-btn", "inspection-help-close", "inspection-help-popover");
  document.getElementById("start-inspection-btn").addEventListener("click", startTrackInspection);
  document.getElementById("stop-inspection-btn").addEventListener("click", finishTrackInspection);
  wirePopover("correct-track-color-help-btn", "correct-track-color-help-close",
              "correct-track-color-help-popover");
  wirePopover("environment-help-btn", "environment-help-close",
              "correct-track-environment-hint");
  wirePopover("min-cell-area-help-btn", "min-cell-area-help-close", "min-cell-area-help-popover");
  wirePopover("drift-help-btn", "drift-help-close", "drift-help-popover");
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

  wireRound34();

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
  wireAcquisitionInputs();
  window.EpiQuantAPI.get_acquisition().then(syncAcquisition);
  window.EpiQuantAPI.get_t1_persistence().then((p) => {
    if (!p) return;
    syncT1Persistence(p);
    if (p.enter_px != null) document.getElementById("t1-enter-input").value = p.enter_px;
    if (p.exit_px != null) document.getElementById("t1-exit-input").value = p.exit_px;
    if (p.event_filter) document.getElementById("t1-event-filter-select").value = p.event_filter;
  });
  window.EpiQuantAPI.get_environment_thresholds().then((t) => {
    if (!t) return;
    document.getElementById("environment-deadband-input").value = t.dead_band;
    document.getElementById("environment-repair-input").value = t.repair_threshold;
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  if (window.isNativeEpiQuant) await window.pywebviewReady;
  init();
});

// ---------------------------------------------------------------------
// Round 34: chart controls, the Trajectory charts, the Data playground,
// the right-click export menu and the movie maker.
//
// One control bar per plotting step, one export menu for every chart. The
// controls are session settings (style, type size, line width, figure width,
// error band) plus the step's shared time window -- shared so that changing
// the value or the cell does not silently reset the window a figure was
// framed with.
// ---------------------------------------------------------------------

const chartState = {
  options: null,
  values: [],          // the value catalog, for the chip pickers
  pc: { allowed: true, warning: null },
  chips: { clone: [], tracks: [] },
  menuChart: null,     // which chart the context menu was opened on
  menuStatus: null,    // and which status line belongs to its panel
  cellmapValue: null,  // the Cell map's chosen value (its picker holds no state)
  distValue: null,     // same, for Distributions
  jointFamilies: ["basic"],
  capabilities: null,  // what each panel of step 8 can do right now
  groups: [],          // group comparator: [{name, folders: []}]
  pendingFolders: [],
};

// R35 1-1: the Figure settings live in a right-click popover now, not in a
// bar on every panel. What stays on the panel is the time window, which is a
// property of the DATA being shown, not of how it is drawn.
function timeWindowMarkup(step) {
  return `
    <span class="control-group-label">Time window</span>
    <label>from <input type="number" id="${step}-window-start" class="small-input"></label>
    <label>to <input type="number" id="${step}-window-end" class="small-input"></label>
    <span id="${step}-window-unit" class="hint-inline"></span>`;
}

function fillSelect(id, entries, current) {
  const select = document.getElementById(id);
  if (!select) return;
  select.innerHTML = "";
  entries.forEach((entry) => {
    const value = typeof entry === "string" ? entry : entry.value;
    const label = typeof entry === "string" ? entry : entry.label;
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    // The number the label no longer carries (R35 0-B) lives here.
    if (entry && entry.title) option.title = entry.title;
    if (value === current) option.selected = true;
    select.appendChild(option);
  });
}

async function instantiateChartControls() {
  document.querySelectorAll("[data-time-window]").forEach((host) => {
    host.innerHTML = timeWindowMarkup(host.dataset.timeWindow);
  });
  chartState.options = await window.EpiQuantAPI.get_plot_options();
  const current = chartState.options.current;
  fillSelect("figure-style-select", chartState.options.styles, current.style);
  // "Default" is a name, not a description: the fallback chain belongs in the
  // tooltip, not in a menu entry that stretches the control (R35 0-B).
  const fontSelect = [{ value: "", label: "Default" }]
    .concat(chartState.options.fonts.map((f) => ({ value: f, label: f })));
  fillSelect("figure-font-select", fontSelect, current.font_family || "");
  document.getElementById("figure-font-select").title =
    `Default = ${chartState.options.font_stack.join(" → ")}`;
  fillSelect("figure-fontsize-select", chartState.options.font_sizes, current.font_size);
  fillSelect("figure-linewidth-select", chartState.options.line_widths, current.line_width);
  fillSelect("figure-sizepreset-select", chartState.options.size_presets, current.size_preset);
  fillSelect("figure-error-select", chartState.options.error_bands, current.error_band);
  ["style", "font", "fontsize", "linewidth", "sizepreset", "error"].forEach((name) => {
    const el = document.getElementById(`figure-${name}-select`);
    if (el) el.addEventListener("change", applyPlotSettings);
  });
  ["trajectory", "playground"].forEach((step) => {
    ["start", "end"].forEach((edge) => {
      const el = document.getElementById(`${step}-window-${edge}`);
      if (el) el.addEventListener("change", () => applyTimeWindow(step));
    });
  });
  wireDraggable("figure-settings-popover", "figure-settings-handle");
  document.getElementById("figure-settings-close").addEventListener("click", () => {
    document.getElementById("figure-settings-popover").style.display = "none";
  });
}

// The settings are session level and apply to every chart, so a change
// redraws whatever the user is looking at -- in both plotting steps, since a
// chart on the step they are not on would otherwise still be showing the old
// style when they switch back.
async function applyPlotSettings() {
  const value = (name) => document.getElementById(`figure-${name}-select`).value;
  const result = await window.EpiQuantAPI.set_plot_settings({
    style: value("style"),
    font_family: value("font") || null,
    font_size: value("fontsize"),
    line_width: value("linewidth"),
    size_preset: value("sizepreset"),
    error_band: value("error"),
  });
  setStatus("figure-settings-note", result.note || "", result.note ? "warn" : "");
  await refreshTrajectoryCharts();
  await refreshPlaygroundPanel();
}

function openFigureSettings(event) {
  const popover = document.getElementById("figure-settings-popover");
  popover.style.display = "block";
  popover.style.transform = "none";
  // R36 2: the popover is `position: fixed`, so it must be placed in VIEWPORT
  // coordinates. It used `pageX/pageY`, which are document coordinates -- on
  // the Trajectory step, where Clone cell data sits well down a scrolled
  // page, that put it below the bottom of the screen. It did open; it just
  // could not be seen. The Playground page is short, which is why it looked
  // fine there.
  const box = popover.getBoundingClientRect();
  const x = event ? event.clientX : window.innerWidth / 2 - box.width / 2;
  const y = event ? event.clientY - 40 : 80;
  const clamp = (value, lower, upper) => Math.max(lower, Math.min(upper, value));
  popover.style.left = `${clamp(x, 8, Math.max(8, window.innerWidth - box.width - 8))}px`;
  popover.style.top = `${clamp(y, 8, Math.max(8, window.innerHeight - box.height - 8))}px`;
}

// Drag by the header. Same shape as the inset-view popover already uses.
function wireDraggable(popoverId, handleId) {
  const popover = document.getElementById(popoverId);
  const handle = document.getElementById(handleId);
  if (!popover || !handle) return;
  let dx = 0;
  let dy = 0;
  const move = (event) => {
    popover.style.left = `${event.clientX - dx}px`;
    popover.style.top = `${event.clientY - dy}px`;
    popover.style.transform = "none";
  };
  const stop = () => {
    document.removeEventListener("mousemove", move);
    document.removeEventListener("mouseup", stop);
  };
  handle.addEventListener("mousedown", (event) => {
    if (event.target.closest("button")) return;
    const box = popover.getBoundingClientRect();
    dx = event.clientX - box.left;
    dy = event.clientY - box.top;
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", stop);
    event.preventDefault();
  });
}

async function syncTimeWindow(step) {
  const window_ = await window.EpiQuantAPI.get_time_window(step);
  const unit = document.getElementById(`${step}-window-unit`);
  if (!window_ || !window_.available) {
    if (unit) unit.textContent = "(track the movie first)";
    return;
  }
  const seconds = window_.unit === "seconds";
  const startEl = document.getElementById(`${step}-window-start`);
  const endEl = document.getElementById(`${step}-window-end`);
  startEl.value = seconds ? Math.round(window_.start_sec) : window_.start_frame;
  endEl.value = seconds ? Math.round(window_.end_sec) : window_.end_frame;
  // R35 0-B: the label is the UNIT. The range belongs on the inputs, where the
  // browser enforces it, not in a sentence beside them.
  const span = window_.last_frame - window_.first_frame;
  const dt = window_.frame_interval_sec || 1;
  [startEl, endEl].forEach((el) => {
    el.min = seconds ? 0 : window_.first_frame;
    el.max = seconds ? Math.round(span * dt) : window_.last_frame;
  });
  if (unit) {
    unit.textContent = seconds ? "s" : "frame";
    unit.title = seconds
      ? `Seconds from frame ${window_.first_frame}, the movie's first`
      : "Set a frame interval in Extract to work in seconds";
  }
}

async function applyTimeWindow(step) {
  const current = await window.EpiQuantAPI.get_time_window(step);
  if (!current || !current.available) return;
  await window.EpiQuantAPI.set_time_window(
    parseFloat(document.getElementById(`${step}-window-start`).value),
    parseFloat(document.getElementById(`${step}-window-end`).value),
    step, current.unit);
  await syncTimeWindow(step);
  await refreshChartsFor(step);
}

async function refreshChartsFor(step) {
  if (step === "trajectory") await refreshTrajectoryCharts();
  else await refreshPlaygroundPanel();
}

// --- value chips -------------------------------------------------------

async function loadValueCatalog() {
  const catalog = await window.EpiQuantAPI.get_value_catalog();
  chartState.values = catalog.values || [];
  chartState.pc = catalog.pc || { allowed: true, warning: null };
  const pcs = chartState.values.filter((v) => v.is_pc).map((v) => v.column);
  ["morphospace-x", "morphospace-y"].forEach((id, i) =>
    fillSelect(id, pcs, pcs[Math.min(i, pcs.length - 1)]));
}

// R35 2-1: a drawn-by-us dropdown, not a native <datalist>.
//
// A `<datalist>` whose option carries both `value` and `label` is rendered by
// WebKit on TWO lines -- "Cell Area" above "Cell Area (µm²) (Cells_info)" --
// which is what the maintainer saw. There is no way to style it. One line
// each: the name in body colour, the unit and source in grey; the search
// matches both.
function valuePickerMarkup(key) {
  return `
    <span class="value-picker-wrap">
      <input type="text" id="${key}-value-input" class="value-picker-input"
             placeholder="Search a value" autocomplete="off" role="combobox"
             aria-expanded="false" aria-controls="${key}-value-menu">
      <div id="${key}-value-menu" class="value-picker-menu" role="listbox" style="display:none"></div>
    </span>`;
}

function valueSecondary(entry) {
  const unit = entry.label && entry.label !== entry.column
    ? entry.label.slice(entry.column.length).trim() : "";
  return [unit, entry.source].filter(Boolean).join(" · ");
}

function wireValuePicker(key, onPick) {
  const input = document.getElementById(`${key}-value-input`);
  const menu = document.getElementById(`${key}-value-menu`);
  if (!input || !menu) return;
  let active = -1;

  const matches = () => {
    const needle = input.value.trim().toLowerCase();
    return chartState.values.filter((entry) =>
      !needle
      || entry.column.toLowerCase().includes(needle)
      || String(entry.source || "").toLowerCase().includes(needle));
  };

  const close = () => {
    menu.style.display = "none";
    input.setAttribute("aria-expanded", "false");
    active = -1;
  };

  const render = () => {
    const found = matches();
    menu.innerHTML = "";
    if (!found.length) {
      close();
      return;
    }
    found.forEach((entry, index) => {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "value-picker-row" + (index === active ? " active" : "");
      row.setAttribute("role", "option");
      const name = document.createElement("span");
      name.className = "value-picker-name";
      name.textContent = entry.column;
      const meta = document.createElement("span");
      meta.className = "value-picker-meta";
      meta.textContent = valueSecondary(entry);
      row.append(name, meta);
      row.addEventListener("mousedown", (event) => {
        event.preventDefault();
        input.value = "";
        close();
        onPick(entry.column);
      });
      menu.appendChild(row);
    });
    menu.style.display = "block";
    input.setAttribute("aria-expanded", "true");
  };

  input.addEventListener("input", () => { active = -1; render(); });
  input.addEventListener("focus", render);
  input.addEventListener("blur", () => setTimeout(close, 120));
  input.addEventListener("keydown", (event) => {
    const found = matches();
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      active = Math.max(0, Math.min(found.length - 1,
                                    active + (event.key === "ArrowDown" ? 1 : -1)));
      render();
      const rows = menu.querySelectorAll(".value-picker-row");
      if (rows[active]) rows[active].scrollIntoView({ block: "nearest" });
    } else if (event.key === "Enter") {
      event.preventDefault();
      const chosen = found[active >= 0 ? active : 0];
      if (chosen) {
        input.value = "";
        close();
        onPick(chosen.column);
      }
    } else if (event.key === "Escape") {
      close();
    }
  });
}

function renderChips(key, hostId, onChange) {
  const host = document.getElementById(hostId);
  host.innerHTML = "";
  chartState.chips[key].forEach((column, index) => {
    const chip = document.createElement("span");
    chip.className = "value-chip";
    chip.draggable = true;
    chip.dataset.index = index;
    chip.textContent = column;
    const remove = document.createElement("button");
    remove.className = "chip-remove";
    remove.textContent = "×";
    remove.title = "Remove";
    remove.addEventListener("click", () => {
      chartState.chips[key].splice(index, 1);
      renderChips(key, hostId, onChange);
      onChange();
    });
    chip.appendChild(remove);
    // Drag to reorder: the chip order IS the drawing order, for the small
    // multiples and for which value gets the left axis.
    chip.addEventListener("dragstart", (e) => e.dataTransfer.setData("text/plain", String(index)));
    chip.addEventListener("dragover", (e) => e.preventDefault());
    chip.addEventListener("drop", (e) => {
      e.preventDefault();
      const from = parseInt(e.dataTransfer.getData("text/plain"), 10);
      const to = parseInt(chip.dataset.index, 10);
      if (Number.isNaN(from) || from === to) return;
      const moved = chartState.chips[key].splice(from, 1)[0];
      chartState.chips[key].splice(to, 0, moved);
      renderChips(key, hostId, onChange);
      onChange();
    });
    host.appendChild(chip);
  });
}

function wireChipInput(pickerKey, key, hostId, onChange) {
  wireValuePicker(pickerKey, (column) => {
    if (!chartState.chips[key].includes(column)) chartState.chips[key].push(column);
    renderChips(key, hostId, onChange);
    onChange();
  });
}

// --- charts: render + the right-click export menu ----------------------

function showChart(imgId, result, statusId) {
  const img = document.getElementById(imgId);
  if (result && result.error) {
    img.removeAttribute("src");
    img.dataset.chart = "";
    setStatus(statusId, result.error, "error");
    return false;
  }
  img.src = result.image;
  img.dataset.chart = result.chart_id;
  img.dataset.kinds = (result.kinds || ["png", "svg", "xlsx"]).join(",");
  // R35 1-4: the preview is rendered at 600 dpi, so it must be SHOWN at its
  // physical width (mm -> CSS px at 96/in) or it arrives six times too big.
  // On a Retina screen this is what makes it sharp rather than resampled.
  if (result.width_mm) img.style.width = `${(result.width_mm / 25.4) * 96}px`;
  const parts = [result.note, result.warning].filter(Boolean);
  setStatus(statusId, parts.join(" — "), result.warning ? "warn" : "");
  return true;
}

// R35 4-2: the status line of the panel the chart lives in, found from the
// image itself. Morphospace and Streamlines reported nothing after an export
// because the old callback only knew two panels by name; this one knows none
// of them, which is why it works for all of them.
function statusLineFor(imgEl) {
  const panel = imgEl.closest(".playground-panel, #clone-cell-panel, "
                              + "#cumulative-events-panel, #group-dynamics-wrap, .step-section");
  const line = panel ? panel.querySelector(".status-line") : null;
  return line ? line.id : "tracks-status";
}

function wireChartContextMenu() {
  const menu = document.getElementById("chart-context-menu");
  document.addEventListener("contextmenu", (event) => {
    const img = event.target.closest(".chart-img");
    if (!img || !img.dataset.chart) {
      menu.style.display = "none";
      return;
    }
    event.preventDefault();
    chartState.menuChart = img.dataset.chart;
    chartState.menuStatus = statusLineFor(img);
    // An image panel has no vector form, so it does not offer SVG.
    const kinds = (img.dataset.kinds || "png,svg,xlsx").split(",");
    menu.querySelectorAll("button[data-export]").forEach((button) => {
      button.style.display = kinds.includes(button.dataset.export) ? "" : "none";
    });
    // `.nav-menu` is `position: absolute`, so PAGE coordinates are the right
    // ones here -- unlike the settings popover above, which is fixed. The
    // right edge is clamped against the document width, not the viewport, or
    // the clamp itself would misplace it on a scrolled page.
    menu.style.display = "block";
    const width = menu.getBoundingClientRect().width || 200;
    menu.style.left = `${Math.min(event.pageX, document.documentElement.scrollWidth - width - 8)}px`;
    menu.style.top = `${event.pageY}px`;
  });
  document.addEventListener("click", () => { menu.style.display = "none"; });
  menu.querySelectorAll("button[data-menu='settings']").forEach((button) => {
    button.addEventListener("click", (event) => {
      menu.style.display = "none";
      openFigureSettings(event);
    });
  });
  menu.querySelectorAll("button[data-export]").forEach((button) => {
    button.addEventListener("click", async () => {
      menu.style.display = "none";
      if (!chartState.menuChart) return;
      const statusId = chartState.menuStatus || "tracks-status";
      const result = await window.EpiQuantAPI.export_chart(chartState.menuChart, button.dataset.export);
      if (result.error) setStatus(statusId, result.error, "error");
      else {
        const extra = (result.notes && result.notes.length) ? ` — ${result.notes.join("; ")}` : "";
        setStatus(statusId, `Saved to ${result.saved}${extra}`, "success");
      }
    });
  });
}

// --- step 7 charts -----------------------------------------------------

function trajectoryColorMode() {
  const checked = document.querySelector("input[name='trajectory-color-mode']:checked");
  return checked ? checked.value : "highlight";
}

async function refreshTrajectoryCharts() {
  const dynamics = trajectoryColorMode() === "dynamics";
  document.getElementById("clone-cell-panel").style.display = dynamics ? "none" : "";
  document.getElementById("cumulative-events-panel").style.display = dynamics ? "" : "none";
  if (dynamics) {
    const result = await window.EpiQuantAPI.get_cumulative_events_plot();
    if (showChart("cumulative-events-chart", result, "cumulative-events-status")
        && result.losses_without_a_paired_gain != null) {
      setStatus("cumulative-events-status",
                `${result.losses_without_a_paired_gain} loss(es) with no paired gain are not counted `
                + "(rosettes and the like); the three curves count the loss axes on screen.");
    }
    return;
  }
  const track = parseInt(document.getElementById("track-number-input").value, 10);
  if (!Number.isFinite(track) || !chartState.chips.clone.length) {
    setStatus("clone-cell-status",
              chartState.chips.clone.length ? "Pick a track." : "Add a value to plot.");
    return;
  }
  const layout = document.getElementById("clone-layout-select").value;
  const result = await window.EpiQuantAPI.get_single_track_plot(track, chartState.chips.clone, layout);
  showChart("clone-cell-chart", result, "clone-cell-status");
}

// --- step 8 ------------------------------------------------------------

function playgroundPanel() {
  const checked = document.querySelector("input[name='playground-panel']:checked");
  return checked ? checked.value : "tracks";
}

async function refreshPlaygroundPanel() {
  const panel = playgroundPanel();
  const map = { tracks: "playground-tracks", morphospace: "playground-morphospace",
                cellmap: "playground-cellmap", streamlines: "playground-streamlines",
                distributions: "playground-distributions" };
  Object.entries(map).forEach(([name, id]) => {
    document.getElementById(id).style.display = name === panel ? "" : "none";
  });
  const comparator = document.getElementById("group-comparator-checkbox").checked;
  document.getElementById("group-comparator").style.display = comparator ? "" : "none";
  document.getElementById("group-dynamics-wrap").style.display =
    comparator && panel === "tracks" ? "" : "none";

  if (panel === "tracks") {
    if (!chartState.chips.tracks.length) {
      setStatus("tracks-status", "Add a value to plot.");
      return;
    }
    const coverage = parseFloat(document.getElementById("tracks-coverage-input").value) / 100;
    const result = await window.EpiQuantAPI.get_all_tracks_plot(
      chartState.chips.tracks, "small_multiples", coverage, null,
      document.getElementById("tracks-by-cluster-checkbox").checked);
    showChart("tracks-chart", result, "tracks-status");
  } else if (panel === "morphospace") {
    const x = document.getElementById("morphospace-x").value;
    const y = document.getElementById("morphospace-y").value;
    if (!x || !y) {
      setStatus("morphospace-status", "Run Classify first.", "error");
      return;
    }
    const result = await window.EpiQuantAPI.get_morphospace_plot(
      x, y,
      document.getElementById("morphospace-hulls").checked,
      document.getElementById("morphospace-boundary").checked,
      document.getElementById("morphospace-exemplars").checked);
    showChart("morphospace-chart", result, "morphospace-status");
  } else if (panel === "cellmap") {
    await refreshCellMap();
  } else if (panel === "distributions") {
    await refreshDistributions();
  }
}

async function showCellMapFrame(frame) {
  state.cellmapFrame = frame;
  setFrameLabel("cellmap", frame);
  await refreshCellMap();
}

async function refreshCellMap() {
  const column = chartState.cellmapValue;
  if (!column) {
    setStatus("cellmap-status", "Pick a value to map.");
    return;
  }
  const isTrack = column === "Track ID";
  // The Track ID has no continuous scale: it is coloured by the identity
  // palette, so neither control applies to it.
  document.getElementById("cellmap-colormap").disabled = isTrack;
  document.getElementById("cellmap-fixed-scale").disabled = isTrack;
  const result = await window.EpiQuantAPI.get_cell_map(
    state.cellmapFrame, column,
    document.getElementById("cellmap-colormap").value || null,
    document.getElementById("cellmap-fixed-scale").checked);
  if (showChart("cellmap-img", result, "cellmap-status")) {
    // An image panel is shown at its own pixel size, never scaled to a
    // physical width; `showChart` set one for the figure panels.
    document.getElementById("cellmap-img").style.width = "";
  }
}

async function drawStreamlines() {
  const start = parseInt(document.getElementById("streamlines-start").value, 10);
  const end = parseInt(document.getElementById("streamlines-end").value, 10);
  const result = await window.EpiQuantAPI.get_streamlines_plot(
    Number.isFinite(start) ? start : null, Number.isFinite(end) ? end : null,
    null, false, document.getElementById("streamlines-registration").value);
  showChart("streamlines-chart", result, "streamlines-status");
}

// --- Part 6: the clipboard menu and the movie maker --------------------

function wireClipboardMenu() {
  const button = document.getElementById("copy-view-btn");
  const menu = document.getElementById("copy-view-menu");
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    menu.style.display = menu.style.display === "none" ? "block" : "none";
  });
  document.addEventListener("click", () => { menu.style.display = "none"; });
  menu.querySelectorAll("button").forEach((entry) => {
    entry.addEventListener("click", async () => {
      menu.style.display = "none";
      if (entry.dataset.action === "copy") await handleCopyView();
      else openMovieMaker();
    });
  });
}

async function openMovieMaker() {
  const dialog = document.getElementById("movie-maker-dialog");
  dialog.style.display = "block";
  const window_ = await window.EpiQuantAPI.get_time_window(
    currentStep() === "playground" ? "playground" : "trajectory");
  if (window_ && window_.available) {
    document.getElementById("movie-start-input").value = window_.start_frame;
    document.getElementById("movie-end-input").value = window_.end_frame;
  }
  setStatus("movie-status", "");
}

async function startMovie() {
  const source = document.getElementById("movie-source-select").value;
  const step = source === "cellmap" ? "cellmap" : currentStep();
  const body = {
    step,
    start_frame: parseInt(document.getElementById("movie-start-input").value, 10),
    end_frame: parseInt(document.getElementById("movie-end-input").value, 10),
    value: chartState.cellmapValue || null,
  };
  document.getElementById("movie-progress").style.display = "block";
  document.getElementById("movie-cancel-btn").style.display = "";
  document.getElementById("movie-start-btn").disabled = true;
  setStatus("movie-status", "Rendering…");
  await window.EpiQuantAPI.start_movie_render(body.step, body.start_frame, body.end_frame, body.value,
                                             state.currentTrackNumber);
}

function onMovieProgress(payload) {
  const bar = document.getElementById("movie-progress");
  bar.value = payload.percent || 0;
  setStatus("movie-status", payload.message || "");
}

function onMovieDone(payload) {
  document.getElementById("movie-progress").style.display = "none";
  document.getElementById("movie-cancel-btn").style.display = "none";
  document.getElementById("movie-start-btn").disabled = false;
  setStatus("movie-status",
            payload.error ? payload.error : `${payload.frames} frame(s) written to ${payload.folder}`,
            payload.error ? "error" : "success");
}

// --- Part 7: the group comparator (folders + joint PCA) ----------------

function renderGroupList() {
  const host = document.getElementById("group-list");
  host.innerHTML = "";
  chartState.groups.forEach((group, index) => {
    const chip = document.createElement("span");
    chip.className = "value-chip";
    chip.textContent = `${group.name} (${group.folders.length} individual`
      + `${group.folders.length === 1 ? "" : "s"})`;
    chip.title = group.folders.join("\n");
    const remove = document.createElement("button");
    remove.className = "chip-remove";
    remove.textContent = "×";
    remove.addEventListener("click", () => {
      chartState.groups.splice(index, 1);
      renderGroupList();
      pushGroupState().then(refreshPlaygroundCapabilities);
    });
    chip.appendChild(remove);
    host.appendChild(chip);
  });
  chartState.pendingFolders.forEach((folder) => {
    const chip = document.createElement("span");
    chip.className = "value-chip pending";
    chip.textContent = folder;
    host.appendChild(chip);
  });
}

function wireGroupComparator() {
  // R36 5: one short line on the panel, the full explanation in the tooltip,
  // and BOTH follow the mode. The tooltip used to be a static HTML attribute
  // describing the Clusters mode, so it still said "Within one movie..."
  // while Multi-groups was selected. It is also set from JS as a single-line
  // string -- written in the markup it carried the HTML indentation into the
  // tooltip.
  const describeGroupMode = (folders) => {
    const note = document.getElementById("group-comparator-note");
    note.textContent = folders
      ? "Each group is one or more project folders."
      : "Clusters of the current project.";
    note.title = folders
      ? "Each folder counts as one individual: group means are means of individual means, "
        + "and intervals resample individuals."
      : "Descriptive only: the clusters were defined on these values, and cells in one sample "
        + "are not independent.";
  };
  document.getElementById("group-mode-select").addEventListener("change", (event) => {
    const folders = event.target.value === "folders";
    document.getElementById("group-folders").style.display = folders ? "" : "none";
    describeGroupMode(folders);
    refreshPlaygroundPanel();
  });
  describeGroupMode(document.getElementById("group-mode-select").value === "folders");
  document.getElementById("group-add-folder-btn").addEventListener("click", async () => {
    const path = await window.EpiQuantAPI.select_folder("Select an individual's project folder");
    if (path) {
      chartState.pendingFolders.push(path);
      renderGroupList();
    }
  });
  document.getElementById("group-add-btn").addEventListener("click", () => {
    const name = document.getElementById("group-name-input").value.trim();
    if (!name || !chartState.pendingFolders.length) return;
    chartState.groups.push({ name, folders: chartState.pendingFolders.slice() });
    chartState.pendingFolders = [];
    document.getElementById("group-name-input").value = "";
    renderGroupList();
    pushGroupState().then(refreshPlaygroundCapabilities);
  });
  document.getElementById("joint-pca-btn").addEventListener("click", async () => {
    const name = document.getElementById("joint-name-input").value.trim();
    setStatus("joint-pca-status", "Building the joint PCA…");
    await pushGroupState();
    const result = await window.EpiQuantAPI.build_joint_pca(
      name, chartState.groups, chartState.jointFamilies || ["basic"], null);
    setStatus("joint-pca-status",
              result.error ? result.error
                : `${result.cells} cells from ${result.folders} folder(s), ${result.components} components; `
                  + `written to ${result.written.length} folder(s).`,
              result.error ? "error" : "success");
    if (!result.error) {
      // The new comparison has to appear in the PCA source menu without a
      // restart -- that gap is what made Round 34's joint PCA unreachable.
      await refreshPlaygroundCapabilities();
      document.getElementById("pca-source-select").value = `joint:${result.comparison}`;
      await applyPcaSource();
    }
  });
}

// --- wiring ------------------------------------------------------------

async function wireRound34() {
  await instantiateChartControls();
  wireChartContextMenu();
  wireClipboardMenu();
  wireGroupComparator();

  document.querySelectorAll("[data-picker]").forEach((host) => {
    host.innerHTML = valuePickerMarkup(host.dataset.picker);
  });
  wireChipInput("clone", "clone", "clone-value-chips", refreshTrajectoryCharts);
  wireChipInput("tracks", "tracks", "tracks-value-chips", refreshPlaygroundPanel);
  wireValuePicker("cellmap", (column) => {
    chartState.cellmapValue = column;
    refreshCellMap();
  });
  wireValuePicker("dist", (column) => {
    chartState.distValue = column;
    refreshDistributions();
  });
  document.getElementById("clone-layout-select").addEventListener("change", refreshTrajectoryCharts);
  document.querySelectorAll("input[name='trajectory-color-mode']").forEach((radio) => {
    radio.addEventListener("change", refreshTrajectoryCharts);
  });
  document.querySelectorAll("input[name='playground-panel']").forEach((radio) => {
    radio.addEventListener("change", refreshPlaygroundPanel);
  });
  ["group-comparator-checkbox", "tracks-by-cluster-checkbox", "morphospace-hulls",
   "morphospace-boundary", "morphospace-exemplars", "cellmap-fixed-scale"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", refreshPlaygroundPanel);
  });
  ["morphospace-x", "morphospace-y"].forEach((id) => {
    document.getElementById(id).addEventListener("change", refreshPlaygroundPanel);
  });
  document.getElementById("tracks-coverage-input").addEventListener("change", refreshPlaygroundPanel);
  document.getElementById("cellmap-colormap").addEventListener("change", refreshCellMap);
  document.getElementById("streamlines-draw-btn").addEventListener("click", drawStreamlines);
  document.getElementById("streamlines-registration").addEventListener("change", drawStreamlines);
  document.getElementById("group-dynamics-btn").addEventListener("click", async () => {
    const kind = document.getElementById("group-dynamics-select").value;
    setStatus("group-dynamics-status", "Computing…");
    const result = await window.EpiQuantAPI.get_group_dynamics_plot(
      kind, chartState.chips.tracks[0] || "Cell Area");
    showChart("group-dynamics-chart", result, "group-dynamics-status");
  });
  document.getElementById("pca-source-select").addEventListener("change", applyPcaSource);
  document.getElementById("dist-paired-checkbox").addEventListener("change", refreshDistributions);
  document.getElementById("joint-features-btn").addEventListener("click", openJointFeatures);
  document.getElementById("joint-features-close").addEventListener("click", () => {
    document.getElementById("joint-features-popover").style.display = "none";
  });
  wireDraggable("joint-features-popover", "joint-features-handle");
  document.getElementById("group-extract-btn").addEventListener("click", extractMissingGroupFolders);
  window.onGroupExtractDone = onGroupExtractDone;
  document.getElementById("movie-start-btn").addEventListener("click", startMovie);
  document.getElementById("movie-maker-close").addEventListener("click", () => {
    document.getElementById("movie-maker-dialog").style.display = "none";
  });
  document.getElementById("movie-cancel-btn").addEventListener("click", async () => {
    await window.EpiQuantAPI.cancel_movie_render();
  });

  fillSelect("cellmap-colormap",
             ["viridis", "magma", "YlGn", "RdBu_r", "coolwarm", "PuOr_r", "BrBG"], "viridis");
  wireFrameNav("cellmap", () => state.cellmapFrame, (f) => (state.cellmapFrame = f),
               showCellMapFrame);
  wireZoom("cellmap", "cellmap-img");
  window.onMovieProgress = onMovieProgress;
  window.onMovieDone = onMovieDone;
}

// Called whenever tracking becomes available or changes: the catalog, the
// window bounds and the PC gate all depend on it.
async function refreshRound34Data() {
  await loadValueCatalog();
  await syncTimeWindow("trajectory");
  await syncTimeWindow("playground");
  if (chartState.pc && chartState.pc.warning) {
    setStatus("clone-cell-status", chartState.pc.warning, "warn");
  }
  await refreshPlaygroundCapabilities();
  enableStep("playground");
}

// ---------------------------------------------------------------------
// Round 36: step 8 after Extract, the joint PCA, and Distributions
// ---------------------------------------------------------------------

// R36 8-1: step 8 opens once there are cells. Static crops and sparsely
// sampled series have nothing to track, and requiring tracking locked them
// out of every analysis they were collected for.
async function unlockPlaygroundAfterExtract() {
  await loadValueCatalog();
  await syncTimeWindow("playground");
  await refreshPlaygroundCapabilities();
  enableStep("playground");
}

async function refreshPlaygroundCapabilities() {
  const caps = await window.EpiQuantAPI.get_playground_capabilities();
  chartState.capabilities = caps;
  // Each panel's radio carries its own reason, so a disabled control always
  // answers "why not" on hover instead of just being grey.
  Object.entries(caps.panels || {}).forEach(([panel, state_]) => {
    const radio = document.querySelector(`input[name="playground-panel"][value="${panel}"]`);
    if (!radio) return;
    radio.disabled = !state_.enabled;
    const label = radio.closest("label");
    if (label) label.title = state_.reason || "";
    if (label) label.classList.toggle("disabled-reason", !state_.enabled);
  });
  const comparator = document.getElementById("group-comparator-checkbox");
  if (comparator && caps.panels && caps.panels.comparator) {
    comparator.disabled = !caps.panels.comparator.enabled;
    const label = comparator.closest("label");
    if (label) label.title = caps.panels.comparator.reason || "";
  }
  // If the selected panel just became unavailable, fall back to one that works.
  const current = document.querySelector("input[name='playground-panel']:checked");
  if (current && current.disabled) {
    const fallback = Array.from(document.querySelectorAll("input[name='playground-panel']"))
      .find((el) => !el.disabled);
    if (fallback) fallback.checked = true;
  }
  await refreshPcaSources();
  return caps;
}

async function refreshPcaSources() {
  const options = await window.EpiQuantAPI.get_joint_pca_options();
  const select = document.getElementById("pca-source-select");
  if (!select) return;
  const previous = select.value;
  select.innerHTML = "";
  options.options.forEach((entry) => {
    const option = document.createElement("option");
    option.value = entry.value;
    option.textContent = entry.label;
    option.disabled = !entry.available;
    if (entry.folders) {
      option.title = `${entry.folders} folder(s), group(s): ${(entry.groups || []).join(", ")}`;
    }
    select.appendChild(option);
  });
  select.value = options.options.some((o) => o.value === previous) ? previous : options.current;
  const note = document.getElementById("pca-source-note");
  if (note) {
    const joint = options.options.filter((o) => o.value !== "project").length;
    note.textContent = joint
      ? `${joint} joint comparison(s) found`
      : "no joint PCA yet — build one in the Group comparator";
  }
}

async function applyPcaSource() {
  const select = document.getElementById("pca-source-select");
  const result = await window.EpiQuantAPI.set_pca_source(select.value);
  if (result && result.error) {
    setStatus("morphospace-status", result.error, "error");
    select.value = "project";
    await window.EpiQuantAPI.set_pca_source("project");
    return;
  }
  await refreshPlaygroundPanel();
}

// --- Distributions (8-2) ----------------------------------------------

async function refreshDistributions() {
  const column = chartState.distValue;
  if (!column) {
    setStatus("dist-status", "Pick a value.");
    return;
  }
  await pushGroupState();
  const result = await window.EpiQuantAPI.get_distributions_plot(
    column, document.getElementById("dist-paired-checkbox").checked);
  showChart("dist-chart", result, "dist-status");
}

// The backend needs the comparator's mode and groups to know how to split the
// cells and where to look for a joint PCA.
async function pushGroupState() {
  const using = document.getElementById("group-comparator-checkbox").checked;
  const mode = using ? document.getElementById("group-mode-select").value : "none";
  await window.EpiQuantAPI.set_group_state(mode, chartState.groups);
}

// --- the joint PCA's feature families (6-1) ---------------------------

async function openJointFeatures(event) {
  const popover = document.getElementById("joint-features-popover");
  const options = await window.EpiQuantAPI.get_feature_options();
  const picker = document.getElementById("joint-feature-families");
  picker.innerHTML = "";
  Object.entries(options.families).forEach(([name, spec]) => {
    const label = document.createElement("label");
    label.className = "checkbox-label";
    const checked = spec.base || (chartState.jointFamilies || []).includes(name) ? " checked" : "";
    const disabled = spec.base ? " disabled" : "";
    label.innerHTML =
      `<input type="checkbox" class="joint-family-checkbox" value="${name}"${checked}${disabled}>`
      + ` ${spec.label}`;
    picker.appendChild(label);
  });
  picker.querySelectorAll(".joint-family-checkbox").forEach((el) => {
    el.addEventListener("change", handleJointFamilyChange);
  });
  handleJointFamilyChange();
  popover.style.display = "block";
  popover.style.transform = "none";
  const box = popover.getBoundingClientRect();
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  popover.style.left = `${clamp(event ? event.clientX : 200, 8,
                                Math.max(8, window.innerWidth - box.width - 8))}px`;
  popover.style.top = `${clamp(event ? event.clientY - 40 : 80, 8,
                               Math.max(8, window.innerHeight - box.height - 8))}px`;
}

function handleJointFamilyChange() {
  const boxes = Array.from(document.querySelectorAll(".joint-family-checkbox"));
  const efdOn = boxes.some((el) => el.value === "efd" && el.checked);
  boxes.forEach((el) => {
    if (el.value === "efd") return;
    el.disabled = efdOn || el.value === "basic";
    if (efdOn) el.checked = false;
    else if (el.value === "basic") el.checked = true;
  });
  chartState.jointFamilies = efdOn ? ["efd"]
    : boxes.filter((el) => el.checked).map((el) => el.value);
  setStatus("joint-features-status",
            efdOn
              ? "Elliptic Fourier is used on its own; every folder needs its raw coefficients."
              : `Selected: ${chartState.jointFamilies.join(", ")}`);
}

// --- batch extraction over the group folders --------------------------

async function extractMissingGroupFolders() {
  await pushGroupState();
  const status = await window.EpiQuantAPI.get_group_extraction_status(chartState.groups);
  if (status.no_masks && status.no_masks.length) {
    setStatus("group-extract-status",
              `No masks/ directory in: ${status.no_masks.join(", ")}`, "error");
    return;
  }
  if (!status.missing.length) {
    setStatus("group-extract-status",
              `All ${status.ready.length} folder(s) already have Cells_info.xlsx.`, "success");
    return;
  }
  document.getElementById("group-extract-progress").style.display = "block";
  document.getElementById("group-extract-btn").disabled = true;
  setStatus("group-extract-status",
            `Extracting ${status.missing.length} of ${status.total} folder(s)…`);
  await window.EpiQuantAPI.start_group_extraction(chartState.groups);
}

function onGroupExtractDone(payload) {
  document.getElementById("group-extract-progress").style.display = "none";
  document.getElementById("group-extract-btn").disabled = false;
  if (payload.error) {
    setStatus("group-extract-status", payload.error, "error");
    return;
  }
  const parts = [`${payload.extracted} folder(s) extracted`,
                 `${payload.already} already had a table`];
  if (payload.cancelled) parts.push("cancelled");
  if (payload.failed && payload.failed.length) parts.push(`failed: ${payload.failed.join("; ")}`);
  if (payload.settings) {
    parts.push(`min area ${payload.settings.min_cell_area_px} px, `
               + `edge cells ${payload.settings.exclude_edge_cells ? "excluded" : "kept"}`);
  }
  setStatus("group-extract-status", parts.join("; "),
            (payload.failed && payload.failed.length) ? "warn" : "success");
}
