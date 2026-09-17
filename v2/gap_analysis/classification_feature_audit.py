"""Archive of the scripts behind `v2/docs/shape-index-vs-circularity.md` and
`v2/docs/classification-feature-audit.md` (2026-09-09).

**Not a single runnable module** -- it is the eight scratch scripts as they were
run, concatenated in order and separated by banners. Each section is
self-contained: split a section out to a file and run it. `regen.py` must run
first (it writes the cached tables the rest read); `build.py` must sit on
`sys.path` as `build.py` for the sections that import it.

Data: `AS_LE_lateral view_E-cad` re-extracted from `masks/` with
`exclude_edge_cells=True` and `min_cell_area=21` (the "<=20 px excluded" rule),
6112 cells over 86 frames, labels from
`GT_cell_classification_output/cells_clustering_results.xlsx`.

Needs scipy, scikit-learn and pyefd, none of which the pipeline itself requires.
"""



# ==========================================================================
# ---- regen.py ----
# ==========================================================================
"""Regenerate AS_LE features from masks: exclude edge cells, drop cells <=20 px.
Also recompute v1's `Fitted Ellipse Angle` (deleted from the schema) for Q1."""
import os, sys, math, pickle
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.expanduser("~/mnt/EPI_Quant/v2"))
import data_input, shape_features, tissue_features

DS = os.path.expanduser("~/mnt/EPI_Quant/test_data/AS_LE_lateral view_E-cad")
MASKS = os.path.join(DS, "masks")
OUT = os.path.expanduser("~/an/regen")
os.makedirs(OUT, exist_ok=True)

data_input.set_min_cell_area(21)          # "<= 20 px" excluded -> threshold 21
print("min_cell_area:", data_input.get_min_cell_area(), flush=True)

info, nframes = data_input.extract_cell_info(MASKS, exclude_edge_cells=True)
print("info", info.shape, "frames", nframes, flush=True)
info.to_pickle(f"{OUT}/info.pkl")

shape = shape_features.extract_shape_features(MASKS, exclude_edge_cells=True, with_coefficients=True)
print("shape", shape.shape, flush=True)
shape.to_pickle(f"{OUT}/shape.pkl")

tissue = tissue_features.extract_tissue_features(MASKS, shape_info=shape)
print("tissue", tissue.shape, flush=True)
tissue.to_pickle(f"{OUT}/tissue.pkl")

# --- v1 Fitted Ellipse Angle, recomputed on exactly the rows above ----------
keep = set(info.index)
rows = {}
files = data_input.list_segmentation_files(MASKS)
fnum = data_input.resolve_frame_numbers(files)
for fn in files:
    dat = np.load(os.path.join(MASKS, fn), allow_pickle=True).item()
    outlines = dat["outlines"]
    fill_labels, removed = data_input.load_label_array(dat, with_removed=True)
    if removed:
        outlines = outlines.copy(); outlines[np.isin(outlines, removed)] = 0
    fid = str(fnum[fn])
    for cid in np.unique(outlines):
        if cid == 0:
            continue
        idx = f"Cell{fid}_{cid}"
        if idx not in keep:
            continue
        pos = np.where(outlines == cid)
        pts = np.array([pos[1], pos[0]]).T.reshape(-1, 1, 2)
        ell = cv2.fitEllipse(pts) if len(pts) >= 5 else None
        if ell is not None and min(ell[1]) < data_input._MIN_ELLIPSE_AXIS_PX:
            ell = None
        if ell is None:
            rows[idx] = (-1.0, -1.0)
        else:
            raw = ell[2]
            rows[idx] = (round(180 - raw if raw > 90 else raw, 2), round(raw, 2))
ang = pd.DataFrame.from_dict(rows, orient="index",
                             columns=["Fitted Ellipse Angle", "Ellipse Angle Raw"])
ang.index.name = "Cell Index"
print("angle", ang.shape, flush=True)
ang.to_pickle(f"{OUT}/angle.pkl")
print("DONE", flush=True)


# ==========================================================================
# ---- build.py ----
# ==========================================================================
import os, sys, math
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/mnt/EPI_Quant/v2"))
OUT = os.path.expanduser("~/an/regen")
GT = os.path.expanduser("~/mnt/EPI_Quant/test_data/AS_LE_lateral view_E-cad/GT_cell_classification_output/cells_clustering_results.xlsx")

def build():
    info = pd.read_pickle(f"{OUT}/info.pkl")
    shape = pd.read_pickle(f"{OUT}/shape.pkl")
    tissue = pd.read_pickle(f"{OUT}/tissue.pkl")
    ang = pd.read_pickle(f"{OUT}/angle.pkl")
    gt = pd.read_excel(GT).set_index("Cell_Index")
    X = info.join(shape, how="left").join(tissue, how="left").join(ang, how="left")
    X["Circularity"] = 4*math.pi*X["Cell Area"]/X["Cell Perimeter"]**2
    X["P-Value"] = 4*math.pi/X["Circularity"]**2
    y = (gt.loc[X.index, "Cell_Type"] == "cluster_2").astype(int).rename("is_AS")
    return X, y

BASIC = ["Cell Area","Cell Perimeter","Fitted Ellipse Major Axis","Fitted Ellipse Minor Axis",
         "Ellipse Major/Minor Axis Ratio","Approximate Polygon Vertex Count",
         "Circumscribed Circle Radius","Inscribed Circle Radius"]
BASIC_OFF = ["Maximum Horizontal Length","Maximum Vertical Length","Horizontal/Vertical Length Ratio"]
HULL = ["Solidity","Convexity","Shape Index"]
LOBE = ["Lobe Count","Max Concavity Depth","Mean Neck Width","EFD Irregularity"]
TISSUE = ["Neighbor Count","Local Hexagonal Fraction","Local Nematic Order",
          "Local Mean Shape Index","Local Mean Solidity"]
LEGACY13 = BASIC + BASIC_OFF + ["Fitted Ellipse Angle","Circularity"]


# ==========================================================================
# ---- q2_q3.py ----
# ==========================================================================
import math, os, glob
import numpy as np, pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

ROOT = os.path.expanduser("~/mnt/EPI_Quant/test_data")
ML = ["Cell Area","Fitted Ellipse Major Axis","Fitted Ellipse Minor Axis","Ellipse Major/Minor Axis Ratio",
      "Cell Perimeter","Maximum Horizontal Length","Maximum Vertical Length",
      "Horizontal/Vertical Length Ratio","Approximate Polygon Vertex Count",
      "Circumscribed Circle Radius","Inscribed Circle Radius"]

def load(ds):
    d = os.path.join(ROOT, ds, "cell_classification_output")
    info = pd.read_excel(os.path.join(d,"Cells_info.xlsx")).set_index("Cell Index")
    sp = os.path.join(d,"Cells_shape.xlsx")
    shape = pd.read_excel(sp).set_index("Cell Index") if os.path.exists(sp) else None
    return info, shape

datasets = [d for d in sorted(os.listdir(ROOT))
            if os.path.exists(os.path.join(ROOT,d,"cell_classification_output","Cells_info.xlsx"))]

print("="*100); print("Q2  Circularity vs Shape Index"); print("="*100)
rows=[]
for ds in datasets:
    info, shape = load(ds)
    ok = (info[ML] != -1).all(axis=1) & info["Cell Perimeter"].gt(0) & info["Cell Area"].gt(0)
    A = info.loc[ok,"Cell Area"].astype(float); P = info.loc[ok,"Cell Perimeter"].astype(float)
    circ = 4*math.pi*A/P**2          # v1 Circularity, recomputed from the two surviving columns
    p_info = P/np.sqrt(A)            # Shape Index from the SAME A,P
    r1 = stats.pearsonr(circ, p_info); s1 = stats.spearmanr(circ, p_info)
    row = dict(dataset=ds, n=len(A), pearson_same=r1[0], spearman_same=s1[0])
    if shape is not None:
        j = pd.concat([circ.rename("circ"), shape["Shape Index"]], axis=1, join="inner").dropna()
        r2 = stats.pearsonr(j.circ, j["Shape Index"]); s2 = stats.spearmanr(j.circ, j["Shape Index"])
        row.update(n_cross=len(j), pearson_cross=r2[0], spearman_cross=s2[0])
        # circularity computed self-consistently inside the shape table
        cm = 4*math.pi*shape["Mask Area"]/shape["Mask Perimeter"]**2
        jm = pd.concat([cm.rename("cm"), shape["Shape Index"]], axis=1).dropna()
        row.update(pearson_mask=stats.pearsonr(jm.cm, jm["Shape Index"])[0],
                   spearman_mask=stats.spearmanr(jm.cm, jm["Shape Index"])[0])
    rows.append(row)
t = pd.DataFrame(rows).set_index("dataset")
pd.set_option("display.width",200,"display.float_format",lambda v:f"{v:.4f}")
print(t.to_string()); print()

print("--- distribution shape (pooled over datasets with a shape table) ---")
for ds in datasets:
    info, shape = load(ds)
    ok = (info[ML] != -1).all(axis=1) & info["Cell Perimeter"].gt(0) & info["Cell Area"].gt(0)
    A = info.loc[ok,"Cell Area"].astype(float); P = info.loc[ok,"Cell Perimeter"].astype(float)
    circ = 4*math.pi*A/P**2; p = P/np.sqrt(A)
    zc = (circ-circ.mean())/circ.std(); zp = (p-p.mean())/p.std()
    print(f"{ds:34s} n={len(A):6d} | circ skew={stats.skew(circ):+6.3f} kurt={stats.kurtosis(circ):+7.3f} "
          f"frac|z|<0.5={np.mean(abs(zc)<0.5):.3f} max|z|={abs(zc).max():6.2f} | "
          f"p skew={stats.skew(p):+6.3f} kurt={stats.kurtosis(p):+7.3f} "
          f"frac|z|<0.5={np.mean(abs(zp)<0.5):.3f} max|z|={abs(zp).max():6.2f}")

print()
print("--- sensitivity to a perimeter bias (the documented +5.5% cv2.arcLength bias) ---")
for ds in datasets[:3]:
    info,_ = load(ds)
    ok = (info[ML] != -1).all(axis=1) & info["Cell Perimeter"].gt(0) & info["Cell Area"].gt(0)
    A = info.loc[ok,"Cell Area"].astype(float); P = info.loc[ok,"Cell Perimeter"].astype(float)
    for bias in (0.055,):
        c0=4*math.pi*A/P**2; c1=4*math.pi*A/(P*(1+bias))**2
        p0=P/np.sqrt(A);     p1=P*(1+bias)/np.sqrt(A)
        print(f"{ds:34s} +{bias*100:.1f}% on P -> circ shifts {abs(c1-c0).mean()/c0.std():.3f} SD, "
              f"shape index shifts {abs(p1-p0).mean()/p0.std():.3f} SD")

print()
print("="*100); print("Q3  do sentinel (-1) cells reach PCA?"); print("="*100)
for ds in datasets:
    info,_ = load(ds)
    bad = (info[ML] == -1)
    n_bad = bad.any(axis=1).sum()
    cols = {c:int(bad[c].sum()) for c in ML if bad[c].any()}
    print(f"{ds:34s} n={len(info):6d}  sentinel cells={n_bad:4d} ({n_bad/len(info)*100:5.2f}%)  columns={cols}")
    if n_bad:
        f = info.loc[:,ML].apply(pd.to_numeric, errors="coerce")
        print(f"{'':34s}   after dropna() (what api._feature_frame does): {len(f.dropna())} rows kept -> "
              f"{int(bad.any(axis=1)[f.dropna().index].sum())} sentinel rows STILL IN the PCA matrix")


# ==========================================================================
# ---- q1.py ----
# ==========================================================================
import os, sys, math
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/an")); sys.path.insert(0, os.path.expanduser("~/mnt/EPI_Quant/v2"))
from build import build, BASIC, BASIC_OFF, HULL, LOBE, TISSUE, LEGACY13
from sklearn.metrics import roc_auc_score, adjusted_rand_score, f1_score, balanced_accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pca_kmeans

X, y = build()
print(f"n={len(X)}  AS={y.sum()} ({y.mean()*100:.2f}%)  non-AS={len(y)-y.sum()}")

print("\n" + "="*92); print("Q1-a  single-feature AUC for AS vs non-AS (0.5 = useless)"); print("="*92)
cands = LEGACY13 + HULL + LOBE + TISSUE + ["Elongation","Sin2Theta","Cos2Theta","P-Value"]
auc = {}
for c in dict.fromkeys(cands):
    v = pd.to_numeric(X[c], errors="coerce"); m = v.notna()
    auc[c] = roc_auc_score(y[m], v[m])
s = pd.Series(auc).sort_values(key=lambda z: (z-0.5).abs(), ascending=False)
for k, v in s.items():
    print(f"  {k:36s} AUC={v:.4f}   |AUC-0.5|={abs(v-0.5):.4f}{'   <-- legacy_13' if k in LEGACY13 else ''}")

print("\n" + "="*92); print("Q1-b  what the angle actually contains"); print("="*92)
a = X["Fitted Ellipse Angle"]; raw = X["Ellipse Angle Raw"]; ratio = X["Ellipse Major/Minor Axis Ratio"]
print(f"  folded angle : AS median={a[y==1].median():5.1f}  non-AS median={a[y==0].median():5.1f}")
print(f"  raw    angle : AS median={raw[y==1].median():5.1f}  non-AS median={raw[y==0].median():5.1f}")
for lo, hi in [(1.0,1.1),(1.1,1.2),(1.2,1.5),(1.5,2.0),(2.0,999)]:
    m = (ratio>=lo)&(ratio<hi)
    if m.sum()>10:
        R = abs(np.exp(1j*np.deg2rad(raw[m]*2)).mean())
        print(f"  axis ratio [{lo},{hi}): n={m.sum():5d} ({m.mean()*100:4.1f}%)  circular concentration R={R:.3f}")

print("\n" + "="*92); print("Q1-c  rotation test: mount the same specimen d degrees rotated"); print("="*92)
def run(cols, Xf, prune=None, mode="global", var=0.85, k=2):
    f = Xf.loc[:, cols].apply(pd.to_numeric, errors="coerce").dropna()
    if prune is not None:
        f, _ = pca_kmeans.prune_correlated(f, threshold=prune)
    Z = pca_kmeans.standardize(f, mode=mode)
    if var is not None:
        Z = PCA(n_components=var).fit_transform(Z)
    return pd.Series(KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z), index=f.index)

def score(lab, y):
    yy = y.loc[lab.index]
    best = max([(f1_score(yy, (lab==c).astype(int)), c) for c in lab.unique()])
    pred = (lab==best[1]).astype(int)
    return dict(ARI=adjusted_rand_score(yy, lab), F1_AS=f1_score(yy, pred),
                BalAcc=balanced_accuracy_score(yy, pred))

base = run(LEGACY13, X); b = score(base, y)
print(f"  legacy_13 (with angle)  d= 0: ARI={b['ARI']:.4f} F1_AS={b['F1_AS']:.4f} BalAcc={b['BalAcc']:.4f}")
for d in (15, 30, 45, 60, 90):
    Xr = X.copy(); r = (X["Ellipse Angle Raw"] + d) % 180
    Xr["Fitted Ellipse Angle"] = np.where(r > 90, 180 - r, r)
    lr = run(LEGACY13, Xr); sc = score(lr, y)
    print(f"  legacy_13 (with angle)  d={d:2d}: ARI={sc['ARI']:.4f} F1_AS={sc['F1_AS']:.4f} "
          f"BalAcc={sc['BalAcc']:.4f}   labels flipped vs d=0: {(lr!=base.loc[lr.index]).mean()*100:5.2f}%")
noang = run([c for c in LEGACY13 if c != "Fitted Ellipse Angle"], X); sn = score(noang, y)
print(f"  legacy_13 MINUS angle       : ARI={sn['ARI']:.4f} F1_AS={sn['F1_AS']:.4f} BalAcc={sn['BalAcc']:.4f}"
      f"   ARI vs the with-angle partition: {adjusted_rand_score(base.loc[noang.index], noang):.4f}")

print("\n" + "="*92); print("Q1-d  the angle's share of the PCA subspace it feeds"); print("="*92)
f = X.loc[:, LEGACY13].apply(pd.to_numeric, errors="coerce").dropna()
Z = pca_kmeans.standardize(f, mode="global")
p = PCA(n_components=0.85).fit(Z)
w = pd.DataFrame(p.components_**2, columns=f.columns,
                 index=[f"PC{i+1} ({v*100:.1f}%)" for i, v in enumerate(p.explained_variance_ratio_)])
print((w[["Fitted Ellipse Angle","Circularity","Cell Area","Ellipse Major/Minor Axis Ratio"]]*100).round(2).to_string())
tot = (p.components_**2 * p.explained_variance_ratio_[:,None]).sum(0)
tot = pd.Series(tot/tot.sum(), index=f.columns).sort_values(ascending=False)
print("\n  variance-weighted share of the retained subspace:")
for k_, v in tot.items(): print(f"    {k_:36s} {v*100:5.2f}%")


# ==========================================================================
# ---- chk.py ----
# ==========================================================================
import os, sys, math
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/an")); sys.path.insert(0, os.path.expanduser("~/mnt/EPI_Quant/v2"))
from build import build
from sklearn.metrics import roc_auc_score
X, y = build()
h = X["Maximum Horizontal Length"]
print("AUC=1.0 check -- Maximum Horizontal Length")
print("  non-AS: max=%.1f  p99=%.1f  median=%.1f" % (h[y==0].max(), h[y==0].quantile(.99), h[y==0].median()))
print("  AS    : min=%.1f  p01=%.1f  median=%.1f" % (h[y==1].min(), h[y==1].quantile(.01), h[y==1].median()))
print("  gap between the two populations: %.1f px  (perfectly separable at any cut in it)" % (h[y==1].min()-h[y==0].max()))
print("  per-frame check: does one threshold work in every frame?")
fr = pd.Series([str(i).split('_')[0] for i in X.index], index=X.index)
bad=0
for f_, g in h.groupby(fr):
    yy = y.loc[g.index]
    if yy.sum() and (yy==0).sum():
        if g[yy==1].min() <= g[yy==0].max(): bad+=1
print(f"    frames where a single horizontal-length cut fails: {bad}/{fr.nunique()}")
print()
print("Circularity vs Shape Index: separating FORMULA from ESTIMATOR")
A_i = X["Cell Area"].astype(float); P_i = X["Cell Perimeter"].astype(float)
A_m = X["Mask Area"].astype(float);  P_m = X["Mask Perimeter"].astype(float)
variants = {
 "Circularity  4piA/P^2   (info A,P  = v1)": 4*math.pi*A_i/P_i**2,
 "Shape Index  P/sqrt(A)  (info A,P)      ": P_i/np.sqrt(A_i),
 "Circularity  4piA/P^2   (mask A,P)      ": 4*math.pi*A_m/P_m**2,
 "Shape Index  P/sqrt(A)  (mask A,P = shipped)": P_m/np.sqrt(A_m),
}
for k,v in variants.items():
    m=v.notna()&np.isfinite(v)
    print(f"  {k:46s} AUC={roc_auc_score(y[m],v[m]):.4f}  |AUC-0.5|={abs(roc_auc_score(y[m],v[m])-0.5):.4f}")
from scipy import stats
c=variants["Circularity  4piA/P^2   (info A,P  = v1)"]; p=variants["Shape Index  P/sqrt(A)  (mask A,P = shipped)"]
ci=variants["Shape Index  P/sqrt(A)  (info A,P)      "]
print(f"  Pearson(C_info, p_info)  = {stats.pearsonr(c,ci)[0]:+.4f}   Spearman = {stats.spearmanr(c,ci)[0]:+.4f}")
print(f"  Pearson(C_info, p_mask)  = {stats.pearsonr(c,p)[0]:+.4f}   Spearman = {stats.spearmanr(c,p)[0]:+.4f}")
cm=variants["Circularity  4piA/P^2   (mask A,P)      "]
print(f"  Pearson(C_mask, p_mask)  = {stats.pearsonr(cm,p)[0]:+.4f}   Spearman = {stats.spearmanr(cm,p)[0]:+.4f}")
print()
print("Neighbor Count NaN:", X["Neighbor Count"].isna().sum(), "of", len(X))
print("tissue NaN per column:"); print(X[["Neighbor Count","Local Hexagonal Fraction","Local Nematic Order","Local Mean Shape Index","Local Mean Solidity"]].isna().sum().to_string())


# ==========================================================================
# ---- q4.py ----
# ==========================================================================
import os, sys, itertools, math, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/an")); sys.path.insert(0, os.path.expanduser("~/mnt/EPI_Quant/v2"))
from build import build, BASIC, BASIC_OFF, HULL, LOBE, TISSUE, LEGACY13
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, f1_score, balanced_accuracy_score, matthews_corrcoef
import pca_kmeans

X, y = build()
POOL = list(dict.fromkeys(BASIC+BASIC_OFF+HULL+LOBE+TISSUE+
            ["Fitted Ellipse Angle","Circularity","Elongation","Sin2Theta","Cos2Theta","P-Value"]))

def evaluate(cols, prune=None, mode="global", var=0.85, k=2, seed=0):
    f = X.loc[:, list(cols)].apply(pd.to_numeric, errors="coerce").dropna()
    if len(f) < 50 or f.shape[1] == 0:
        return None
    kept = f
    if prune is not None:
        kept, _ = pca_kmeans.prune_correlated(f, threshold=prune)
    Z = pca_kmeans.standardize(kept, mode=mode)
    if var is not None:
        Z = PCA(n_components=var).fit_transform(Z)
    lab = pd.Series(KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z), index=kept.index)
    yy = y.loc[lab.index]
    _, best = max((f1_score(yy, (lab == c).astype(int)), c) for c in lab.unique())
    pred = (lab == best).astype(int)
    return dict(ARI=adjusted_rand_score(yy, lab), F1_AS=f1_score(yy, pred),
                BalAcc=balanced_accuracy_score(yy, pred), MCC=matthews_corrcoef(yy, pred),
                n=len(kept), d_in=f.shape[1], d_kept=kept.shape[1])

FAMS = {"basic":BASIC, "basic_off":BASIC_OFF, "hull":HULL, "lobe":LOBE, "tissue":TISSUE}
combos = {
  "legacy_13 (v1, with angle+circularity)": LEGACY13,
  "legacy_13 - angle": [c for c in LEGACY13 if c!="Fitted Ellipse Angle"],
  "legacy_13 - circularity": [c for c in LEGACY13 if c!="Circularity"],
  "legacy_13 - angle - circularity": [c for c in LEGACY13 if c not in ("Fitted Ellipse Angle","Circularity")],
  "basic (shipped default)": BASIC,
  "basic + basic_off": BASIC+BASIC_OFF,
  "basic + hull": BASIC+HULL,
  "basic + lobe": BASIC+LOBE,
  "basic + hull + lobe": BASIC+HULL+LOBE,
  "basic + basic_off + hull": BASIC+BASIC_OFF+HULL,
  "basic + basic_off + hull + lobe": BASIC+BASIC_OFF+HULL+LOBE,
  "basic + tissue": BASIC+TISSUE,
  "basic + hull + lobe + tissue": BASIC+HULL+LOBE+TISSUE,
  "hull only (dimensionless)": HULL,
  "hull + lobe (dimensionless)": HULL+LOBE,
  "ratios only": ["Ellipse Major/Minor Axis Ratio","Horizontal/Vertical Length Ratio","Solidity","Convexity","Shape Index"],
}
PRUNES = [None, 0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
rows=[]
for name, cols in combos.items():
    for pr in PRUNES:
        for mode in ("global","per_frame"):
            for var in (0.85, 0.95, None):
                r = evaluate(cols, prune=pr, mode=mode, var=var)
                if r: rows.append(dict(featureset=name, prune=pr, std=mode, var=var, **r))
R = pd.DataFrame(rows)
R.to_pickle(os.path.expanduser("~/an/grid.pkl"))
pd.set_option("display.width",240,"display.max_rows",300,"display.float_format",lambda v:f"{v:.4f}")
print("="*118); print("Q4-a  full grid, top 25 by ARI"); print("="*118)
print(R.sort_values("ARI",ascending=False).head(25).to_string(index=False))
print()
print("="*118); print("Q4-b  best config per feature set (var=0.85, the shipped setting)"); print("="*118)
sub = R[R["var"]==0.85]
print(sub.sort_values("ARI",ascending=False).groupby("featureset",sort=False).head(1)
        .sort_values("ARI",ascending=False).to_string(index=False))
print()
print("="*118); print("Q4-c  prune threshold sweep, holding featureset=legacy_13 and basic+hull+lobe"); print("="*118)
for fs in ["legacy_13 (v1, with angle+circularity)","basic + basic_off + hull + lobe","basic (shipped default)"]:
    print(f"\n  {fs}   (std=global, var=0.85)")
    q = R[(R.featureset==fs)&(R["std"]=="global")&(R["var"]==0.85)].sort_values("prune", na_position="first")
    print(q[["prune","d_in","d_kept","ARI","F1_AS","BalAcc","MCC","n"]].to_string(index=False))


# ==========================================================================
# ---- q4b.py ----
# ==========================================================================
import os, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/an")); sys.path.insert(0, os.path.expanduser("~/mnt/EPI_Quant/v2"))
from build import build, BASIC, BASIC_OFF, HULL, LOBE, TISSUE, LEGACY13
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, f1_score, roc_auc_score
import pca_kmeans
X, y = build()
pd.set_option("display.width",240,"display.float_format",lambda v:f"{v:.4f}")

h=X["Maximum Horizontal Length"]
print("exact AUC(Maximum Horizontal Length) = %.6f" % roc_auc_score(y,h))
n1,n0=int(y.sum()),int((y==0).sum())
print(f"  misordered pairs: {(1-roc_auc_score(y,h))*n1*n0:.0f} of {n1*n0} pairs")

def ev(cols, prune=None, mode="global", var=0.85, seed=0):
    f=X.loc[:,list(cols)].apply(pd.to_numeric,errors="coerce").dropna()
    kept,_=(pca_kmeans.prune_correlated(f,threshold=prune) if prune is not None else (f,None))
    Z=pca_kmeans.standardize(kept,mode=mode)
    if var is not None: Z=PCA(n_components=var).fit_transform(Z)
    lab=pd.Series(KMeans(n_clusters=2,n_init=10,random_state=seed).fit_predict(Z),index=kept.index)
    return adjusted_rand_score(y.loc[lab.index],lab), lab, list(kept.columns)

print("\nwhich features survive pruning on 'basic' (std=global, var=0.85):")
for pr in (0.99,0.95,0.90,0.85,0.80,0.70):
    a,_,k=ev(BASIC,prune=pr); print(f"  r>{pr}: ARI={a:.4f}  kept={k}")

POOL=list(dict.fromkeys(BASIC+BASIC_OFF+HULL+LOBE+["Fitted Ellipse Angle","Circularity","Elongation"]))
print("\ngreedy forward selection (prune=None, std=global, var=0.85), scored by ARI vs GT:")
chosen=[]; best=-1
for step in range(6):
    cand=[(ev(chosen+[c])[0],c) for c in POOL if c not in chosen]
    cand.sort(reverse=True)
    if cand[0][0] <= best + 1e-9 and step>0:
        print(f"  step {step+1}: no improvement (best candidate {cand[0][1]} -> {cand[0][0]:.4f}); stop"); break
    best,pick=cand[0]; chosen.append(pick)
    print(f"  step {step+1}: + {pick:36s} ARI={best:.4f}   runners-up: " +
          ", ".join(f"{c}={s:.3f}" for s,c in cand[1:4]))

print("\nseed / stability check on the winners (10 KMeans seeds):")
sets={"legacy_13 @prune0.80":(LEGACY13,0.80),"legacy_13 @no prune":(LEGACY13,None),
      "basic+basic_off @prune0.80":(BASIC+BASIC_OFF,0.80),"basic+basic_off @no prune":(BASIC+BASIC_OFF,None),
      "basic (shipped) @no prune":(BASIC,None),"greedy top-2":(chosen[:2],None),"greedy top-3":(chosen[:3],None)}
for nm,(cols,pr) in sets.items():
    a=[ev(cols,prune=pr,seed=s)[0] for s in range(10)]
    print(f"  {nm:30s} ARI mean={np.mean(a):.4f} min={np.min(a):.4f} max={np.max(a):.4f}")

print("\nleave-one-frame-out: fit on 85 frames, apply to the held-out frame (legacy_13 @0.80 vs basic):")
fr=pd.Series([str(i).split('_')[0] for i in X.index],index=X.index)
for nm,(cols,pr) in [("legacy_13 @0.80",(LEGACY13,0.80)),("basic+basic_off @0.80",(BASIC+BASIC_OFF,0.80)),("basic (shipped)",(BASIC,None))]:
    aris=[]
    f=X.loc[:,cols].apply(pd.to_numeric,errors="coerce").dropna()
    kept,_=(pca_kmeans.prune_correlated(f,threshold=pr) if pr else (f,None))
    for fid in sorted(fr.unique(), key=int)[:20]:
        tr=kept[fr.loc[kept.index]!=fid]; te=kept[fr.loc[kept.index]==fid]
        if len(te)<5 or y.loc[te.index].nunique()<2: continue
        from sklearn.preprocessing import StandardScaler
        sc=StandardScaler().fit(tr); p=PCA(n_components=0.85).fit(sc.transform(tr))
        km=KMeans(n_clusters=2,n_init=10,random_state=0).fit(p.transform(sc.transform(tr)))
        pred=km.predict(p.transform(sc.transform(te)))
        aris.append(adjusted_rand_score(y.loc[te.index],pred))
    print(f"  {nm:24s} held-out ARI over {len(aris)} frames: mean={np.mean(aris):.4f} min={np.min(aris):.4f}")


# ==========================================================================
# ---- q4c.py ----
# ==========================================================================
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/an")); sys.path.insert(0, os.path.expanduser("~/mnt/EPI_Quant/v2"))
from build import build, BASIC, BASIC_OFF, HULL, LOBE, LEGACY13
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, f1_score
import pca_kmeans
X,y=build()
fr=pd.Series([str(i).split('_')[0].replace('Cell','') for i in X.index],index=X.index)

def kept_of(cols,pr):
    f=X.loc[:,cols].apply(pd.to_numeric,errors="coerce").dropna()
    k,d=(pca_kmeans.prune_correlated(f,threshold=pr) if pr else (f,{}))
    return list(k.columns), d
for nm,cols,pr in [("legacy_13 @0.80",LEGACY13,0.80),("legacy_13 @0.90",LEGACY13,0.90),
                   ("legacy_13 @0.95",LEGACY13,0.95),("basic+basic_off @0.80",BASIC+BASIC_OFF,0.80)]:
    k,d=kept_of(cols,pr); print(f"{nm:24s} keeps {len(k)}: {k}")

print("\nleave-one-frame-out (fit on 85 frames, predict the held-out frame):")
for nm,cols,pr in [("legacy_13 @0.80",LEGACY13,0.80),("basic+basic_off @0.80",BASIC+BASIC_OFF,0.80),
                   ("basic (shipped, no prune)",BASIC,None),("greedy 2 (MaxHoriz+EllipseMajor)",["Maximum Horizontal Length","Fitted Ellipse Major Axis"],None)]:
    f=X.loc[:,cols].apply(pd.to_numeric,errors="coerce").dropna()
    kept,_=(pca_kmeans.prune_correlated(f,threshold=pr) if pr else (f,None))
    aris=[];f1s=[]
    for fid in sorted(fr.unique(),key=int):
        tr=kept[fr.loc[kept.index]!=fid]; te=kept[fr.loc[kept.index]==fid]
        if len(te)<5 or y.loc[te.index].nunique()<2: continue
        sc=StandardScaler().fit(tr); p=PCA(n_components=0.85).fit(sc.transform(tr))
        km=KMeans(n_clusters=2,n_init=10,random_state=0).fit(p.transform(sc.transform(tr)))
        pr_=pd.Series(km.predict(p.transform(sc.transform(te))),index=te.index)
        yy=y.loc[te.index]
        _,b=max((f1_score(yy,(pr_==c).astype(int)),c) for c in np.unique(km.labels_))
        aris.append(adjusted_rand_score(yy,pr_)); f1s.append(f1_score(yy,(pr_==b).astype(int)))
    print(f"  {nm:34s} frames={len(aris):3d} ARI mean={np.mean(aris):.4f} min={np.min(aris):.4f} | F1_AS mean={np.mean(f1s):.4f} min={np.min(f1s):.4f}")

print("\nrotation test redone (ARI of the partition against the d=0 partition, and against GT):")
def run(Xf,cols,pr):
    f=Xf.loc[:,cols].apply(pd.to_numeric,errors="coerce").dropna()
    k,_=(pca_kmeans.prune_correlated(f,threshold=pr) if pr else (f,None))
    Z=PCA(n_components=0.85).fit_transform(pca_kmeans.standardize(k,mode="global"))
    return pd.Series(KMeans(n_clusters=2,n_init=10,random_state=0).fit_predict(Z),index=k.index)
for pr,tag in [(None,"no prune"),(0.80,"@0.80")]:
    b=run(X,LEGACY13,pr); print(f"  legacy_13 {tag}: d= 0 ARI_GT={adjusted_rand_score(y.loc[b.index],b):.4f}")
    for d in (15,30,45,60,90):
        Xr=X.copy(); r=(X["Ellipse Angle Raw"]+d)%180
        Xr["Fitted Ellipse Angle"]=np.where(r>90,180-r,r)
        l=run(Xr,LEGACY13,pr)
        print(f"  legacy_13 {tag}: d={d:2d} ARI_GT={adjusted_rand_score(y.loc[l.index],l):.4f}  ARI_vs_d0={adjusted_rand_score(b.loc[l.index],l):.4f}")


# ==========================================================================
# ---- q4d.py ----
# ==========================================================================
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/an")); sys.path.insert(0, os.path.expanduser("~/mnt/EPI_Quant/v2"))
from build import build, BASIC, BASIC_OFF, HULL, LOBE, LEGACY13
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, f1_score, balanced_accuracy_score
import pca_kmeans
X,y=build()
fr=pd.Series([str(i).split('_')[0].replace('Cell','') for i in X.index],index=X.index)

print("rotation x pruning interaction -- which columns survive prune_correlated(0.80):")
for d in (0,15,30,45,90):
    Xr=X.copy(); r=(X["Ellipse Angle Raw"]+d)%180
    Xr["Fitted Ellipse Angle"]=np.where(r>90,180-r,r)
    f=Xr.loc[:,LEGACY13].apply(pd.to_numeric,errors="coerce").dropna()
    k,_=pca_kmeans.prune_correlated(f,threshold=0.80)
    print(f"  d={d:2d}: {list(k.columns)}")

print("\nLEAVE-ONE-FRAME-OUT, pooled scoring (all held-out predictions pooled into one confusion):")
cfgs={"legacy_13, no prune":(LEGACY13,None),
      "legacy_13 @0.80":(LEGACY13,0.80),
      "basic (shipped default)":(BASIC,None),
      "basic + basic_off":(BASIC+BASIC_OFF,None),
      "basic + basic_off @0.80":(BASIC+BASIC_OFF,0.80),
      "basic + basic_off + hull":(BASIC+BASIC_OFF+HULL,None),
      "MaxHoriz + EllipseMajor":(["Maximum Horizontal Length","Fitted Ellipse Major Axis"],None),
      "MaxHoriz + EllipseMajor + ShapeIndex":(["Maximum Horizontal Length","Fitted Ellipse Major Axis","Shape Index"],None),
      "MaxHoriz + Area + AxisRatio":(["Maximum Horizontal Length","Cell Area","Ellipse Major/Minor Axis Ratio"],None)}
for nm,(cols,pr) in cfgs.items():
    f=X.loc[:,cols].apply(pd.to_numeric,errors="coerce").dropna()
    kept,_=(pca_kmeans.prune_correlated(f,threshold=pr) if pr else (f,None))
    preds={}
    for fid in sorted(fr.unique(),key=int):
        tr=kept[fr.loc[kept.index]!=fid]; te=kept[fr.loc[kept.index]==fid]
        if len(te)==0: continue
        sc=StandardScaler().fit(tr); p=PCA(n_components=0.85).fit(sc.transform(tr))
        Ztr=p.transform(sc.transform(tr)); km=KMeans(n_clusters=2,n_init=10,random_state=0).fit(Ztr)
        # name the AS cluster on the TRAINING half, then apply to held-out
        ytr=y.loc[tr.index]; _,asc=max((f1_score(ytr,(km.labels_==c).astype(int)),c) for c in (0,1))
        for i,v in zip(te.index, km.predict(p.transform(sc.transform(te)))): preds[i]=int(v==asc)
    P=pd.Series(preds); yy=y.loc[P.index]
    print(f"  {nm:38s} n={len(P):5d} ARI={adjusted_rand_score(yy,P):.4f} F1_AS={f1_score(yy,P):.4f} "
          f"BalAcc={balanced_accuracy_score(yy,P):.4f}  kept={kept.shape[1]}")


# ==========================================================================
# ---- q4e.py ----
# ==========================================================================
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/an")); sys.path.insert(0, os.path.expanduser("~/mnt/EPI_Quant/v2"))
from build import build, BASIC, BASIC_OFF, LEGACY13
from sklearn.cluster import KMeans; from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, f1_score, balanced_accuracy_score
import pca_kmeans
X,y=build(); fr=pd.Series([str(i).split('_')[0].replace('Cell','') for i in X.index],index=X.index)

def prune(features, threshold, method="pearson"):
    corr=features.corr(method=method).abs(); kept=[]; dropped={}
    for c in features.columns:
        d=next((k for k in kept if pd.notna(corr.loc[c,k]) and corr.loc[c,k]>threshold), None)
        if d is None: kept.append(c)
        else: dropped[c]=d
    return features.loc[:,kept], dropped

def lofo(kept):
    preds={}
    for fid in sorted(fr.unique(),key=int):
        tr=kept[fr.loc[kept.index]!=fid]; te=kept[fr.loc[kept.index]==fid]
        if len(te)==0: continue
        sc=StandardScaler().fit(tr); p=PCA(n_components=0.85).fit(sc.transform(tr))
        km=KMeans(n_clusters=2,n_init=10,random_state=0).fit(p.transform(sc.transform(tr)))
        ytr=y.loc[tr.index]; _,asc=max((f1_score(ytr,(km.labels_==c).astype(int)),c) for c in (0,1))
        for i,v in zip(te.index, km.predict(p.transform(sc.transform(te)))): preds[i]=int(v==asc)
    P=pd.Series(preds); yy=y.loc[P.index]
    return adjusted_rand_score(yy,P), f1_score(yy,P), balanced_accuracy_score(yy,P)

cols = LEGACY13   # the only set that contains BOTH Circularity and Shape Index? add Shape Index in
cols = LEGACY13 + ["Shape Index"]
f=X.loc[:,cols].apply(pd.to_numeric,errors="coerce").dropna()
print("Pearson vs Spearman pruning on legacy_13 + Shape Index (Circularity and Shape Index both present)")
print(f"  raw |Pearson r| (Circularity, Shape Index) = {abs(f['Circularity'].corr(f['Shape Index'])):.4f}")
print(f"  raw |Spearman r|(Circularity, Shape Index) = {abs(f['Circularity'].corr(f['Shape Index'],method='spearman')):.4f}")
print()
print(f"  {'thr':>5} | {'pearson: kept':>13} {'C/p pair caught':>16} {'ARI':>7} {'F1_AS':>7} | {'spearman: kept':>14} {'C/p caught':>11} {'ARI':>7} {'F1_AS':>7}")
for t in (0.99,0.97,0.95,0.90,0.85,0.80):
    out=[]
    for m in ("pearson","spearman"):
        k,d=prune(f,t,m)
        caught = ("Shape Index" in d and d["Shape Index"]=="Circularity") or ("Circularity" in d)
        a,f1,_=lofo(k); out.append((k.shape[1],caught,a,f1))
    print(f"  {t:5.2f} | {out[0][0]:13d} {str(out[0][1]):>16} {out[0][2]:7.4f} {out[0][3]:7.4f} |"
          f" {out[1][0]:14d} {str(out[1][1]):>11} {out[1][2]:7.4f} {out[1][3]:7.4f}")


# ==========================================================================
# ---- q2doc.py ----
# ==========================================================================
import os, sys, math
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/an")); sys.path.insert(0, os.path.expanduser("~/mnt/EPI_Quant/v2"))
from build import build
from scipy import stats
X,y=build()
A=X["Mask Area"].astype(float); P=X["Mask Perimeter"].astype(float)
p=P/np.sqrt(A); C=4*math.pi*A/P**2
print("AS_LE (regenerated, edge cells + <=20px removed, n=%d)"%len(X))
print("  reference points:   circle p=3.5449 C=1.0000 | hexagon p=3.7224 C=0.9069 | p0=3.81 C=0.8656")
q=[0,1,25,50,75,99,100]
print("  Shape Index  percentiles:", " ".join(f"p{i}={np.percentile(p,i):.3f}" for i in q))
print("  Circularity  percentiles:", " ".join(f"p{i}={np.percentile(C,i):.3f}" for i in q))
print()
print("  resolution: dC/dp = -8pi/p^3")
for v in (3.6,4.0,5.0,6.0,8.0,10.0):
    print(f"    at p={v:5.2f} (C={4*math.pi/v**2:.3f}): one unit of p moves C by {8*math.pi/v**3:.4f}")
print(f"    -> from p=3.6 to p=10 the same unit step in p shrinks C by {(8*math.pi/3.6**3)/(8*math.pi/10**3):.1f}x")
print()
zc=(C-C.mean())/C.std(); zp=(p-p.mean())/p.std()
print(f"  class separation after StandardScaler (AS vs non-AS median gap, in SD):")
print(f"    Circularity {abs(zc[y==1].median()-zc[y==0].median()):.3f} SD   Shape Index {abs(zp[y==1].median()-zp[y==0].median()):.3f} SD")
print(f"  skew: C={stats.skew(C):+.3f}  p={stats.skew(p):+.3f}   |  excess kurtosis: C={stats.kurtosis(C):+.3f}  p={stats.kurtosis(p):+.3f}")
print(f"  cells with C>1 (impossible, digitisation): {(C>1).sum()}   cells with p<3.5449 (impossible): {(p<3.5449).sum()}")
print()
print("  error propagation:  dC/C = -2 dP/P + dA/A     dp/p = dP/P - 0.5 dA/A")
for b in (0.02,0.055,0.10):
    print(f"    a +{b*100:4.1f}% perimeter bias moves C by {((1/(1+b)**2)-1)*100:+6.2f}% and p by {b*100:+6.2f}%")


# ==========================================================================
# ---- q5.py / q6.py / q7.py  (2026-09-11, the prune_correlated correction) ----
# ==========================================================================
# q5: how many live-schema pairs does Spearman catch that Pearson does not?
# q6: is pooled Pearson inflated by the two-population structure?
# q7: which feature does each coefficient throw away first?

import os, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/an")); sys.path.insert(0, os.path.expanduser("~/mnt/EPI_Quant/v2"))
from build import build, BASIC, BASIC_OFF, HULL, LOBE
X,_=build()
LIVE = BASIC+BASIC_OFF+HULL+LOBE          # what Classify actually offers today
f = X.loc[:,LIVE].apply(pd.to_numeric,errors="coerce").dropna()
P = f.corr().abs(); S = f.corr(method="spearman").abs()
rows=[]
for a,b in itertools.combinations(f.columns,2):
    rows.append((a,b,P.loc[a,b],S.loc[a,b],S.loc[a,b]-P.loc[a,b]))
D=pd.DataFrame(rows,columns=["a","b","pearson","spearman","gap"])
pd.set_option("display.width",220,"display.float_format",lambda v:f"{v:.4f}")
print("live Classify feature set (basic+basic_off+hull+lobe), %d columns, %d pairs" % (f.shape[1],len(D)))
print("\npairs where Spearman clears a threshold that Pearson does not:")
for t in (0.99,0.95,0.90,0.85):
    m=D[(D.spearman>t)&(D.pearson<=t)]
    print(f"  thr {t}: {len(m)} pair(s)" + ("" if len(m)==0 else ""))
    for _,r in m.iterrows():
        print(f"      {r.a:34s} x {r.b:34s} pearson={r.pearson:.4f} spearman={r.spearman:.4f}")
print("\nthe reverse -- Pearson clears it but Spearman does not (Spearman would KEEP a column Pearson drops):")
for t in (0.99,0.95,0.90,0.85):
    m=D[(D.pearson>t)&(D.spearman<=t)]
    print(f"  thr {t}: {len(m)} pair(s)")
    for _,r in m.iterrows():
        print(f"      {r.a:34s} x {r.b:34s} pearson={r.pearson:.4f} spearman={r.spearman:.4f}")
print("\ntop 12 pairs by |spearman - pearson|:")
print(D.reindex(D.gap.abs().sort_values(ascending=False).index).head(12).to_string(index=False))

import os, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/an")); sys.path.insert(0, os.path.expanduser("~/mnt/EPI_Quant/v2"))
from build import build, BASIC, BASIC_OFF, HULL, LOBE
X,y=build()
LIVE=BASIC+BASIC_OFF+HULL+LOBE
f=X.loc[:,LIVE].apply(pd.to_numeric,errors="coerce").dropna()
yy=y.loc[f.index]
pairs=[("Cell Area","Inscribed Circle Radius"),("Cell Area","Fitted Ellipse Minor Axis"),
       ("Cell Area","Maximum Horizontal Length"),("Inscribed Circle Radius","Maximum Horizontal Length"),
       ("Cell Perimeter","Fitted Ellipse Major Axis"),("Cell Area","Cell Perimeter")]
print("is Pearson inflated by the two-cluster structure?  (pooled vs within the majority class)")
print(f"{'pair':>66}  {'pooled r':>9} {'pooled rho':>11} | {'non-AS r':>9} {'non-AS rho':>11} | {'AS r':>7} {'AS rho':>8}")
for a,b in pairs:
    g=f.loc[yy==0]; h=f.loc[yy==1]
    print(f"{a+' x '+b:>66}  {f[a].corr(f[b]):9.4f} {f[a].corr(f[b],method='spearman'):11.4f} |"
          f" {g[a].corr(g[b]):9.4f} {g[a].corr(g[b],method='spearman'):11.4f} |"
          f" {h[a].corr(h[b]):7.4f} {h[a].corr(h[b],method='spearman'):8.4f}")

print("\nsame question, averaged over the six test datasets that have no such bimodality issue?")
print("(here: how many of the 153 live pairs change side of a 0.90 cut between the two coefficients)")
for nm,sub in [("pooled (both populations)",f),("majority population only",f.loc[yy==0])]:
    P=sub.corr().abs(); S=sub.corr(method="spearman").abs()
    d=[(P.loc[a,b],S.loc[a,b]) for a,b in itertools.combinations(sub.columns,2)]
    d=pd.DataFrame(d,columns=["p","s"])
    print(f"  {nm:28s} pearson>0.90: {int((d.p>0.90).sum()):3d}   spearman>0.90: {int((d.s>0.90).sum()):3d}"
          f"   mean(p-s) = {(d.p-d.s).mean():+.4f}")

import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/an")); sys.path.insert(0, os.path.expanduser("~/mnt/EPI_Quant/v2"))
from build import build, BASIC, BASIC_OFF
from sklearn.metrics import roc_auc_score
X,y=build()
cols=BASIC+BASIC_OFF
f=X.loc[:,cols].apply(pd.to_numeric,errors="coerce").dropna()
auc={c:abs(roc_auc_score(y.loc[f.index],f[c])-0.5)+0.5 for c in cols}
def prune(features,t,method):
    corr=features.corr(method=method).abs(); kept=[]; dropped={}
    for c in features.columns:
        d=next((k for k in kept if corr.loc[c,k]>t),None)
        if d is None: kept.append(c)
        else: dropped[c]=d
    return kept,dropped
for t in (0.90,0.85,0.80):
    print(f"\nthreshold {t}")
    for m in ("pearson","spearman"):
        k,d=prune(f,t,m)
        lost=sorted(((auc[c],c,v) for c,v in d.items()),reverse=True)
        print(f"  {m:9s} keeps {len(k)}; drops: " + ", ".join(f"{c}(AUC {a:.3f}, as dup of {v})" for a,c,v in lost[:3]))
        best_dropped=max((auc[c] for c in d), default=0)
        print(f"            highest-AUC feature thrown away: {best_dropped:.4f}   (best feature overall AUC {max(auc.values()):.4f})")
