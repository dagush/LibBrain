# --------------------------------------------------------------------------------------
# Raincloud plot additions — drop-in complement to plotComparisonAcrossLabels2[Ax].
#
# A raincloud plot layers three elements per group:
#   • Cloud  — half-violin (KDE) drawn to the LEFT of the centre position
#   • Rain   — jittered strip of raw data points, drawn to the RIGHT
#   • Box    — a slim boxplot at the centre, used as the statannotations anchor
#
# The statannotations Annotator is initialised on an almost-invisible seaborn
# boxplot (linewidth=0, width≈0) so that all existing annotation options
# (BH correction, custom tests, star format …) work unchanged.
#
# Public API (identical signatures to the boxplot versions):
#   plotRaincloudAcrossLabels2Ax(ax, tests, ...)   ← axes-level
#   plotRaincloudAcrossLabels2(tests, ...)          ← figure-level convenience wrapper
# --------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from itertools import combinations
from scipy.stats import gaussian_kde
from statannotations.Annotator import Annotator

# Reuse helpers already defined in p_values.py
# (printStats, padEqualLengthDicts are imported at call-site or kept here for standalone use)


# ──────────────────────────────────────────────────────────────────────────────
# Internal drawing helper
# ──────────────────────────────────────────────────────────────────────────────

def _draw_raincloud_elements(
    ax,
    df_long: pd.DataFrame,
    order: list,
    palette=None,
    violin_width: float = 0.30,
    violin_offset: float = 0.30,   # leftward shift of cloud from centre tick
    strip_offset: float = 0.18,    # rightward shift of rain from centre tick
    jitter: float = 0.04,
    box_width: float = 0.06,
    box_linewidth: float = 1.4,
    point_size: float = 10.0,
    point_alpha: float = 0.55,
    kde_resolution: int = 256,
):
    """
    Draw half-violin + strip + slim boxplot for every group in *order*.

    Parameters
    ----------
    ax            : matplotlib Axes (already has an invisible seaborn boxplot)
    df_long       : tidy DataFrame with columns ['value', 'group']
    order         : list of group names, same order as the seaborn x-axis ticks
    palette       : dict {group: colour} or None (falls back to current colour cycle)
    violin_width  : maximum half-width of the KDE shape in data-x units
    violin_offset : how far left of tick centre the cloud is drawn
    strip_offset  : how far right of tick centre the rain dots are drawn
    jitter        : horizontal jitter amplitude for rain dots
    box_width     : half-width of the central boxplot whisker caps
    box_linewidth : line width of box/whisker elements
    point_size    : scatter dot size (points²)
    point_alpha   : opacity of rain dots
    kde_resolution: number of KDE evaluation points
    """
    colours = palette or {
        g: c for g, c in zip(order, plt.rcParams["axes.prop_cycle"].by_key()["color"])
    }

    for tick_pos, group in enumerate(order):
        vals = df_long.loc[df_long["group"] == group, "value"].dropna().values
        if len(vals) < 2:
            continue
        col = colours.get(group, "steelblue")

        # ── Cloud (half-violin, left side) ──────────────────────────────────
        kde = gaussian_kde(vals, bw_method="scott")
        y_grid = np.linspace(vals.min(), vals.max(), kde_resolution)
        density = kde(y_grid)
        density = density / density.max() * violin_width   # normalise to max width

        x_centre = tick_pos - violin_offset
        x_right  = x_centre + density          # fills leftward from x_centre
        ax.fill_betweenx(y_grid, x_centre, x_right, color=col, alpha=0.55, linewidth=0)
        ax.plot(x_right, y_grid, color=col, linewidth=0.8, alpha=0.8)

        # ── Rain (jittered strip, right side) ───────────────────────────────
        rng = np.random.default_rng(seed=42)
        x_rain = tick_pos + strip_offset + rng.uniform(-jitter, jitter, size=len(vals))
        ax.scatter(x_rain, vals, color=col, s=point_size, alpha=point_alpha,
                   linewidths=0, zorder=3)

        # ── Box (slim, aligned with rain to the right of violin) ────────────
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1
        lo_whisker = max(vals.min(), q1 - 1.5 * iqr)
        hi_whisker = min(vals.max(), q3 + 1.5 * iqr)

        box_centre = tick_pos + strip_offset  # same anchor as the rain dots

        # IQR box
        rect = mpatches.FancyBboxPatch(
            (box_centre - box_width, q1),
            2 * box_width, iqr,
            boxstyle="square,pad=0",
            linewidth=box_linewidth,
            edgecolor=col,
            facecolor="none",
            zorder=4,
        )
        ax.add_patch(rect)
        # Median line
        ax.plot([box_centre - box_width, box_centre + box_width], [med, med],
                color=col, linewidth=box_linewidth + 0.6, zorder=5)
        # Whiskers
        ax.plot([box_centre, box_centre], [q3, hi_whisker],
                color=col, linewidth=box_linewidth, zorder=4)
        ax.plot([box_centre, box_centre], [q1, lo_whisker],
                color=col, linewidth=box_linewidth, zorder=4)
        # Whisker caps
        for w in (lo_whisker, hi_whisker):
            ax.plot([box_centre - box_width * 0.6, box_centre + box_width * 0.6], [w, w],
                    color=col, linewidth=box_linewidth, zorder=4)


# ──────────────────────────────────────────────────────────────────────────────
# Public API — mirrors plotComparisonAcrossLabels2Ax / plotComparisonAcrossLabels2
# ──────────────────────────────────────────────────────────────────────────────

def plotComparisonAcrossLabels2Ax(
    ax,
    tests,
    custom_test=None,
    columnLables=None,
    graphLabel='',
    pairs=None,
    test='Mann-Whitney',
    comparisons_correction='BH',
    show_N=True,
    palette=None,
    violin_width: float = 0.30,
    violin_offset: float = 0.30,
    strip_offset: float = 0.18,
    jitter: float = 0.04,
    box_width: float = 0.06,
    point_size: float = 10.0,
):
    """
    Axes-level raincloud plot with statannotations significance brackets.

    Parameters shared with plotComparisonAcrossLabels2Ax
    ─────────────────────────────────────────────────────
    ax                    : matplotlib Axes to draw on
    tests                 : dict {label: array-like} — raw per-group data
    custom_test           : custom statannotations test object (overrides `test`)
    columnLables          : ordered list of group names (defaults to tests.keys())
    graphLabel            : axes title string
    pairs                 : list of (a, b) tuples to annotate; None → all pairs
    test                  : statannotations test name, e.g. 'Mann-Whitney'
    comparisons_correction: 'BH', 'Bonferroni', or None
    show_N                : append (N=…) to each x-tick label

    Raincloud-specific parameters
    ──────────────────────────────
    palette      : dict {label: colour} or None
    violin_width : max half-width of KDE shape in axis-x units  (default 0.30)
    violin_offset: leftward shift of cloud from tick centre      (default 0.30)
    strip_offset : rightward shift of rain from tick centre      (default 0.18)
    jitter       : horizontal scatter amplitude for rain dots    (default 0.04)
    box_width    : half-width of slim central boxplot            (default 0.06)
    point_size   : size of rain dots in points²                  (default 3.5)
    """
    # ── 0. Stats printout (same as boxplot version) ──────────────────────────
    from p_values import printStats, padEqualLengthDicts   # adjust import path as needed
    printStats(tests)

    # ── 1. Resolve column order ───────────────────────────────────────────────
    if columnLables is None:
        columnLables = list(tests.keys())
    else:
        columnLables = list(columnLables)

    # ── 2. Compute N before padding (padding adds NaNs) ──────────────────────
    n_per_group = {c: int(np.sum(~np.isnan(np.asarray(tests[c], dtype=float))))
                   for c in columnLables}

    # ── 3. Pad unequal-length arrays so DataFrame construction works ──────────
    if isinstance(tests, dict):
        tests_padded = padEqualLengthDicts(tests)
    else:
        tests_padded = tests

    df_wide = pd.DataFrame(tests_padded, columns=columnLables)

    # ── 4. Invisible seaborn boxplot — gives statannotations its tick positions
    #       We draw it with zero linewidth so it is completely hidden behind our
    #       manual raincloud elements drawn in step 6.
    sns.boxplot(
        data=df_wide,
        order=columnLables,
        ax=ax,
        width=1e-6,          # essentially invisible width
        linewidth=0,         # no visible lines
        fliersize=0,         # no outlier markers
        palette=palette,
        boxprops=dict(alpha=0),
        whiskerprops=dict(alpha=0),
        medianprops=dict(alpha=0),
        capprops=dict(alpha=0),
    )

    # ── 5. X-tick labels with optional N ─────────────────────────────────────
    tick_labels = (
        [f"{c}\n(N={n_per_group[c]})" for c in columnLables]
        if show_N else columnLables
    )
    ax.set_xticks(range(len(columnLables)))
    ax.set_xticklabels(tick_labels)

    # ── 6. Draw the actual raincloud elements ─────────────────────────────────
    # Build a tidy long-form frame (ignoring NaN padding)
    df_long = df_wide.melt(var_name="group", value_name="value").dropna(subset=["value"])
    _draw_raincloud_elements(
        ax, df_long, columnLables,
        palette=palette,
        violin_width=violin_width,
        violin_offset=violin_offset,
        strip_offset=strip_offset,
        jitter=jitter,
        box_width=box_width,
        point_size=point_size,
    )

    # Give the axes some breathing room on the left so the cloud isn't clipped
    ax.set_xlim(-0.5 - violin_offset - violin_width, len(columnLables) - 0.5 + strip_offset + box_width + 0.15)

    # ── 7. statannotations — identical to boxplot version ────────────────────
    if pairs is None:
        pairs = list(combinations(columnLables, 2))

    annotator = Annotator(ax, pairs, data=df_wide, order=columnLables)
    annotator.configure(text_format='star', loc='inside')
    if custom_test is None:
        annotator.configure(test=test)
    else:
        annotator.configure(test=custom_test)
    if comparisons_correction is not None:
        annotator.configure(
            comparisons_correction=comparisons_correction,
            correction_format="replace",
        )
    annotator.apply_and_annotate()

    ax.set_title(graphLabel)


def plotComparisonAcrossLabels2(
    tests,
    custom_test=None,
    columnLables=None,
    graphLabel='',
    pairs=None,
    test='Mann-Whitney',
    comparisons_correction='BH',
    show_N=True,
    palette=None,
    violin_width: float = 0.30,
    violin_offset: float = 0.30,
    strip_offset: float = 0.18,
    jitter: float = 0.04,
    box_width: float = 0.06,
    point_size: float = 10.0,
    figsize=None,
):
    """
    Figure-level convenience wrapper around plotRaincloudAcrossLabels2Ax.
    All parameters are forwarded unchanged — see that function for full docs.

    Extra parameter
    ───────────────
    figsize : (width, height) in inches, or None for matplotlib default
    """
    fig, ax = plt.subplots(figsize=figsize)
    plotComparisonAcrossLabels2Ax(
        ax, tests,
        custom_test=custom_test,
        columnLables=columnLables,
        graphLabel=graphLabel,
        pairs=pairs,
        test=test,
        comparisons_correction=comparisons_correction,
        show_N=show_N,
        palette=palette,
        violin_width=violin_width,
        violin_offset=violin_offset,
        strip_offset=strip_offset,
        jitter=jitter,
        box_width=box_width,
        point_size=point_size,
    )
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test — run this file directly to see a sample raincloud plot
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dummy = {
        "Control":   rng.normal(0.5, 0.15, 30),
        "Treatment": rng.normal(0.65, 0.20, 25),
        "Sham":      rng.normal(0.45, 0.12, 28),
    }
    plotComparisonAcrossLabels2(
        dummy,
        graphLabel="Example raincloud",
        comparisons_correction="BH",
    )
