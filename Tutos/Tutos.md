# Epi-Quant GUI tutorial

This guide adds the operational details behind the [20 illustrated pages](<Epi-Quant%20GUI%20Tutos.pdf>). Start with the [README](../README.md) for installation and input formats. The screenshots show an example dataset; use your own microscopy images and segmentation masks for analysis.

**Workflow:** Setup → Extract → optional Classify → Track → optional Correct Tracking → Trajectory → Data Playground. Classification can be skipped. Epi-Quant analyzes two-dimensional cell outlines; inspect the image–mask pairing and segmentation before interpreting measurements.

## 1. Set up a project

![Project setup and input folders](<Epi-Quant%20GUI%20Tutos/Epi-Quant%20GUI%20Tutos-01.jpg>)

Choose three paths in **Setup**: **Masks folder** (segmentation files), **Images folder** (the matching microscopy frames), and **Output folder** (new analysis results). Click **Set Project**. The app creates `cell_classification_output/`, `cell_track_output/`, and `quantitative_analysis_output/` under the output path; it does not copy the two input folders there. When reopening an existing project, choose **Load Existing Results** to restore its tables.

Use consistent frame numbers in both input folders. Masks can be Cellpose-style `.npy` files or TIFF (`.tif`/`.tiff`) masks/boundary images, but keep only one mask format in a folder. Original images can be TIFF, JPEG, or PNG. The [README input example](../README.md#input-data) shows the layout.

> **Tip — Tissue Analyzer folders:** If each frame is a subfolder with a `handCorrection.tif`, collect those files into one `masks/` directory before selecting it in Epi-Quant. The macOS/Linux and Windows commands are [below](#collect-tissue-analyzer-handcorrection-masks). Keep the frame number at the end of each source folder name.

## 2. Inspect segmentation and extract features

![Segmentation QC and feature extraction](<Epi-Quant%20GUI%20Tutos/Epi-Quant%20GUI%20Tutos-02.jpg>)

In **Extract Cell Features**, set **Frame interval** and **Pixel size** if known. **Exclude edge cells** removes truncated border cells from feature analysis; **Minimum cell area** removes labels below the chosen area threshold from the mask. If a TIFF with only two pixel values is a filled-cell mask rather than boundary lines, open **Advanced options → Mask type** and select `binary_mask` instead of `auto`. Click **Extract Features** and inspect several frames in the player before continuing.

The **QC** overlay can highlight weak junction contrast, unusual cell shapes, unexpected neighbour counts, or failed measurements. These are prompts for visual review, not automatic proof that a segmentation is wrong. **Compute shape descriptors** is optional; use it when you need convexity, lobing, local tissue texture, or elliptic Fourier descriptors (EFDs) for classification. Skip it when your immediate goal is tracking.

> **Tip — apparent image flicker:** For grayscale images above 8-bit depth, the viewer scales each frame to an 8-bit display range. Changing frame maxima can make a movie appear to flicker even if the specimen did not. For *preview only*, make 8-bit RGB copies using one fixed intensity range across the entire time series; keep the original images for quantitative intensity checks. Check the raw images before attributing brightness changes to biology.

> **Tip — find segmentation errors quickly:** Run a preliminary track after a rough segmentation. Unmatched cells, short tracks, and abrupt identity changes often make merged or split cells easier to spot than a boundary network alone. Correct the source masks in your segmentation tool, then repeat inspection and use **Re-track range** for affected frames. One or two correction passes may be enough for a clean dataset, but inspect the result rather than relying on a fixed number of passes. For TIFF inputs, edit the original `.tif`/`.tiff` files; `masks/npy_converted/` is a regenerated cache.

## 3. Classify cells (optional)

### 3.1 Choose features and clustering settings

![PCA and clustering setup](<Epi-Quant%20GUI%20Tutos/Epi-Quant%20GUI%20Tutos-03.jpg>)

In **Classify Cells**, choose feature families and individual measurements before PCA and k-means. Basic geometry comes from **Extract Features**. Select optional shape or tissue-texture features only after computing them in Extract. EFD features represent the full two-dimensional contour and can distinguish lobed or puzzle-like cells that simple length/area measures may miss. Use **one specimen over time** for a time series; **unrelated images** fits each independent sample separately. Auto-clustering chooses a cluster number from the configured range; inspect the result because a statistical cluster need not equal a biological cell type.

### 3.2 Preview the result

![Classification preview](<Epi-Quant%20GUI%20Tutos/Epi-Quant%20GUI%20Tutos-04.jpg>)

Use the player to inspect several frames. Change cluster colours or switch between filled masks and outlines to see boundaries clearly. The clipboard copies the current view; sequence export writes frames using the selected display style. Before comparing groups, check whether the clusters remain interpretable across the movie.

### 4. Correct classification

![Manual classification correction](<Epi-Quant%20GUI%20Tutos/Epi-Quant%20GUI%20Tutos-05.jpg>)

In **Correct Cell Classification**, select the target cluster, click cells to reassign them, and save with **Save Corrections** or **Ctrl+S** (Windows/Linux) / **Cmd+S** (macOS). Review multiple frames if cell appearances change over time.

## 5. Track cells

![Tracking settings and local re-tracking](<Epi-Quant%20GUI%20Tutos/Epi-Quant%20GUI%20Tutos-06.jpg>)

In **Track Cells**, choose how to match cells between adjacent frames: centroid displacement, mask overlap (IoU), or both. The default is **Centroid + mask overlap**. **Close 1-frame gaps** can reconnect a track missing for one frame; **Register masks (drift)** accounts for whole-field movement. The `?` controls explain each setting. Run **Track Cells**, then inspect a sample of links throughout the movie and around divisions, cell loss, or poor segmentation.

> **Tip — practical starting settings:** For many movies, **Centroid only + Close 1-frame gaps + registration when the field drifts** is a useful first pass. Compare its visible mistakes with the default centroid-plus-overlap setting before keeping it. Registration and overlap settings should reflect the actual image motion; no preset guarantees correct identities.

> **Tip — local changes:** After correcting source segmentation masks, use **Detect changed frames** and **Re-track range** to recompute the affected interval. Check the links at both boundaries of that interval. A changed mask may also change extracted measurements, so re-extract features when needed.

## 6. Correct tracking by visual inspection

### 6.1 Inspect tracks and their life cycles

![Tracking correction overview](<Epi-Quant%20GUI%20Tutos/Epi-Quant%20GUI%20Tutos-07.jpg>)

**Correct Cell Tracking** provides the current frame, a selected track's history, and several overlays. Select a cell to inspect its ID and movement across frames. Use the inset/stepper view to compare adjacent frames. A link is correct only if the cell at frame *f* is the same physical cell at frame *f* + 1.

### 6.2 Use overlays to locate likely mask errors

![Tracking overlays for segmentation QC](<Epi-Quant%20GUI%20Tutos/Epi-Quant%20GUI%20Tutos-08.jpg>)

The **Track ID**, **Matched/Unmatched**, and **Track life-cycle** views make broken links visible. A sudden new track, a track ending, or an unmatched cell may indicate over-segmentation, under-segmentation, cell entry/exit, division, or a real tracking error. Confirm the cause against the microscopy image and the mask before editing an ID.

### 6.3 Reassign a track ID

![Reassigning a track](<Epi-Quant%20GUI%20Tutos/Epi-Quant%20GUI%20Tutos-09.jpg>)

Select the cell/track, enter the intended **Reassign to track #** ID, and click **Apply**. Use **Propagate to later** only when the identity should remain changed in subsequent frames. Inspect those frames after applying the change.

> **Tip — a target ID already appears later:** If the desired ID belongs to a track that first appears at the current or a later frame, a reassignment may collide with that track. The correction code can move the target's conflicting future segment to a fresh, unused ID (above the existing maximum), while assigning the requested ID to the edited track. Check the reported result and both tracks afterward; this is a conflict-handling behavior, not a way to merge two real cells.

### 6.4 Swap two tracks

![Swapping two tracks](<Epi-Quant%20GUI%20Tutos/Epi-Quant%20GUI%20Tutos-10.jpg>)

If two cells exchange identities, enable **Swap mode** (shortcut **X**), choose the two matched cells in the current frame, then click **Swap**. You can also enter both IDs. **Esc** or **X** exits swap mode. Use propagation only if the swap continues beyond the current frame.

### 6.5 Ask for a likely previous-frame match

![Suggest match in tracking correction](<Epi-Quant%20GUI%20Tutos/Epi-Quant%20GUI%20Tutos-11.jpg>)

**Suggest match** ranks candidate IDs from the preceding frame. Inspect the image, candidate positions, and neighbouring cells before accepting one. A ranked suggestion helps navigate complex cases; it is not a verified answer.

### 6.6 Review queues and the census

![Census-assisted tracking review](<Epi-Quant%20GUI%20Tutos/Epi-Quant%20GUI%20Tutos-12.jpg>)

The four Set choices below select links to inspect. They are review queues, **not** four independent measures of tracking accuracy. The first compares with an internally computed overlap reference, so it is not strictly reference-free. Only a human verdict or an independently established ground truth decides whether a link is correct.

| Review set in the GUI                                     | How it chooses links                                                                                                                         | Main limitation                                                                                                                                                        |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Every link the overlap reference disagrees with** | Compares the tracker's successor with the cell having the best mask overlap in the next frame. Disagreements are shown first.                | Agreement can still be wrong. The overlap reference depends on how the masks were registered.                                                                          |
| **Cells the environment score flags for repair**    | Compares each cell's neighbours between frames using a Dice-style overlap of neighbour sets; low scores suggest a changed local environment. | Genuine neighbour exchanges can lower the score. Small or incomplete neighbour sets can make it uninformative, and a coherent permutation of IDs may escape detection. |
| **Links the reverse solve disagrees with**          | Solves the same pair of frames backward and lists links that do not match the forward solution.                                              | A systematic mistake can be consistent in both directions. It shares the coherent-permutation blind spot with the environment score.                                   |
| **Links the solver was least sure of**              | Ranks assignments by the gap between the smallest and second-smallest feasible costs; a small gap indicates an ambiguous choice.             | It measures the tracker's own cost separation, not truth; margins are comparable only within the same cost settings.                                                   |

Use **Correct** to apply reviewed link fixes; use **Census** to record verdicts for a ground-truth set without editing tracking. The modes deliberately keep correction and ground-truth collection separate. In a review queue, **J** marks a link correct, **K** marks it wrong, and **L** records that it cannot be judged. Hold **Space** to alternate frames; follow the on-screen prompt after **K** to choose the correct successor or none. The queue selection itself is biased toward its own flagging rule, so a percentage from it alone is not an unbiased accuracy estimate.

> **Tip — combine evidence carefully:** Review flagged links together with ordinary, unflagged links and inspect the microscopy image. The environment and reverse-solve queues can miss the same coherent identity error; agreement between them is not independent confirmation.

## 7. Inspect trajectories and cell dynamics

### 7.1 Visual inspection of tracks

![Trajectory viewer](<Epi-Quant%20GUI%20Tutos/Epi-Quant%20GUI%20Tutos-13.jpg>)

Select a cell in **Trajectory View** to follow its path through time. Step through the frames and inspect discontinuities. The visual inspection tools can support track-by-track checking of a ground-truth dataset; keep a record of ambiguous cases rather than forcing a verdict.

### 7.2 Plot single-cell measurements

![Single-cell plots and exports](<Epi-Quant%20GUI%20Tutos/Epi-Quant%20GUI%20Tutos-14.jpg>)

Choose a track, measurement, and time window to plot its dynamics. Plot presets help with presentation. Export a figure and its underlying table where offered; retain the raw workbook for analysis and reproducibility.

### 7.3 Explore cell events

![Cell-event visualization](<Epi-Quant%20GUI%20Tutos/Epi-Quant%20GUI%20Tutos-15.jpg>)

Use **Cell dynamics** to inspect events such as neighbour exchanges and lineage changes. Event detection uses tracking, geometry, and topology across time; it does not infer division or cell loss from PCA clusters. Check candidate events against the frames, especially near segmentation mistakes or image boundaries. T1 filtering options help reduce short-lived contact changes; report the settings used.

## 8. Use Data Playground

### 8.1 Collective dynamics

![Collective dynamics in Data Playground](<Epi-Quant%20GUI%20Tutos/Epi-Quant%20GUI%20Tutos-16.jpg>)

Choose measurements and a time window for population-level plots. Select clusters when a within-project comparison is biologically meaningful. Review sample sizes and whether the plotted cells are the same across time.

### 8.2 Morphospace

![Morphospace view](<Epi-Quant%20GUI%20Tutos/Epi-Quant%20GUI%20Tutos-17.jpg>)

Select principal-component axes and optional overlays to explore how cells occupy a feature space. PC positions reflect the selected, scaled shape measurements; they do not encode a complete cell trajectory by themselves.

### 8.3 Cell maps

![Spatial cell map](<Epi-Quant%20GUI%20Tutos/Epi-Quant%20GUI%20Tutos-18.jpg>)

Select a measurement and colour scale to map values back onto cells in their tissue positions. Keep the colour limits consistent when comparing frames or groups. Use **Movie Maker** for an image sequence and include the exported settings with it.

### 8.4 Compare groups

![Inter-group comparison](<Epi-Quant%20GUI%20Tutos/Epi-Quant%20GUI%20Tutos-19.jpg>)

Select comparable projects or clusters, a measurement, and a time window. Check acquisition scale, frame interval, and feature settings before interpreting differences. Export the figure and its source table together.

### 8.5 Visualize flow

![Cell streamlines](<Epi-Quant%20GUI%20Tutos/Epi-Quant%20GUI%20Tutos-20.jpg>)

Choose the time window and registration setting for a streamline view. These paths summarize motion inferred from tracked cells; verify that drift correction is appropriate for the question being asked.

## Keyboard shortcuts

| Where                                 | Keys                                                      | Action                                                                                                                      |
| ------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Player, when visible                  | **← / →**, **Space**, **Home / End**  | Previous/next frame, play/pause, first/last frame. The magnifier or a review queue may use these keys instead while active. |
| Classification or tracking correction | **Ctrl+S** (Windows/Linux), **Cmd+S** (macOS) | Save corrections in the current correction screen.                                                                          |
| Tracking correction                   | **X**; **Esc**                                | Toggle swap mode; leave swap mode.                                                                                          |
| Census review                         | **J / K / L**, hold **Space**                 | Correct / wrong / cannot tell; alternate the two frames. After **K**, use the on-screen number choices.               |

## Collect Tissue Analyzer `handCorrection.tif` masks

Tissue Analyzer-style projects may store each frame in its own subfolder. The following commands copy each immediate subfolder's `handCorrection.tif` to a single Epi-Quant mask folder, naming each copy after its source folder. They do not change the originals. For example, `sample_0000/handCorrection.tif` becomes `masks/sample_0000_handCorrection.tif`. Keep the frame number as the last number in the subfolder name; Epi-Quant uses the last number in a mask filename as the frame index. Check that the number of copied masks equals the number of intended frames and that the matching original images have the same counters.

**macOS/Linux terminal** (replace the two paths):

```sh
src="/path/to/TissueAnalyzer/project"
dst="/path/to/EpiQuant/project/masks"
mkdir -p "$dst"
for dir in "$src"/*/; do
    [ -d "$dir" ] || continue
    file="${dir}handCorrection.tif"
    [ -f "$file" ] || continue
    name="$(basename "${dir%/}")"
    target="$dst/${name}_handCorrection.tif"
    if [ -e "$target" ]; then
        printf 'Already exists, skipped: %s\n' "$target"
        continue
    fi
    cp "$file" "$target"
done
```

**Windows PowerShell** (replace the two paths):

```powershell
$src = 'C:\path\to\TissueAnalyzer\project'
$dst = 'C:\path\to\EpiQuant\project\masks'
New-Item -ItemType Directory -Path $dst -Force | Out-Null
Get-ChildItem -LiteralPath $src -Directory | ForEach-Object {
    $file = Join-Path $_.FullName 'handCorrection.tif'
    if (Test-Path -LiteralPath $file -PathType Leaf) {
        $target = Join-Path $dst ($_.Name + '_handCorrection.tif')
        if (Test-Path -LiteralPath $target) {
            Write-Host "Already exists, skipped: $target"
        } else {
            Copy-Item -LiteralPath $file -Destination $target
        }
    }
}
```

If the original images are also stored one per frame folder, export or copy them into a separate `images/` folder with matching names. Do not put images and masks into the same input folder.
