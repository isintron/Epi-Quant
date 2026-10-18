"""Manuscript plotting style for EPI_Quant figures + SVG hygiene for Illustrator.

Style is taken from figures/Epiquant_ShapeAnalysis_allMutants.ipynb:
genotype colours, 8/9/11 pt type, lw .7 for single traces (alpha .3) and
lw 2.2 + marker 'o' ms 4 capsize 2 for group means, grid alpha .2.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import Collection
from matplotlib.text import Text

GENOTYPE = {'WT_G': '#4878d0', 'tmc': '#ee854a', 'pyd': '#6acc64'}
ROWS     = ['#8ab6e2', '#4878d0', '#26427a']   # ordinal ramp, validated
AS       = '#ee854a'
NEUTRAL  = 'grey'
REF      = '0.7'

FS_TICK, FS_LABEL, FS_TITLE, FS_SUP = 8, 9, 10, 11
LW_SINGLE, LW_MEAN, MS_MEAN, CAPSIZE = .7, 2.2, 4, 2


def use():
    mpl.rcParams.update({
        # --- Illustrator-friendly SVG: real <text>, no glyph outlines, no font subsetting
        'svg.fonttype': 'none',
        'path.simplify': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        # --- manuscript type
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': FS_LABEL,
        'axes.labelsize': FS_LABEL, 'axes.titlesize': FS_TITLE,
        'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK,
        'legend.fontsize': FS_TICK, 'figure.titlesize': FS_SUP,
        'axes.linewidth': .8, 'xtick.major.width': .8, 'ytick.major.width': .8,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'legend.frameon': False,
        'axes.grid': False, 'grid.alpha': .2, 'grid.linewidth': .6,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'savefig.transparent': False,
    })


def unclip(ax):
    """Drop per-artist clip paths so the SVG has no <clipPath> soup.

    Matplotlib attaches a clip-path to every artist inside an Axes; Illustrator
    imports each one as a clipping mask. Images keep theirs (they must be
    clipped); everything else is drawn inside the limits anyway.
    """
    from matplotlib.image import AxesImage
    seen = []
    seen += list(ax.get_children())
    seen += ax.get_xgridlines() + ax.get_ygridlines()
    for axis in (ax.xaxis, ax.yaxis):          # minor gridlines and tick lines
        seen += list(axis.get_gridlines()) + list(axis.get_ticklines())
        seen += [t.gridline for t in axis.get_minor_ticks() + axis.get_major_ticks()]
    for c in ax.containers:                    # errorbar caps live here
        for part in getattr(c, 'lines', ()) or ():
            seen += list(part) if isinstance(part, (list, tuple)) else [part]
    for a in seen:
        if a is None or isinstance(a, AxesImage):
            continue
        try: a.set_clip_on(False)
        except Exception: pass


def save(fig, path, axes=None):
    for ax in (axes if axes is not None else fig.axes):
        unclip(ax)
    fig.savefig(path, format='svg', bbox_inches='tight', transparent=False)
    return path
