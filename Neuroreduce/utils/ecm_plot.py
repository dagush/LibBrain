"""
Neuroreduce/utils/ecm_plot.py
------------------------------
Violin plots comparing edge-centric metastability (ECM) between the
original BOLD signal (source space) and its reconstruction after
dimensionality reduction (reconstructed space).

What is being plotted — correctly
----------------------------------
For each method (PCA, CHARM) and each subject:

    X_sub       = original BOLD for that subject  (N × T)
    X_sub_hat   = reducer.inverse_transform(reducer.transform(X_sub))  (N × T)

    ecm_source[sub]        = compute_ecm(X_sub)
    ecm_reconstructed[sub] = compute_ecm(X_sub_hat)

Both live in N-dimensional parcel space, so their ECM values are directly
comparable. The violin plots show the DISTRIBUTION of these ECM values
across subjects, for each method and each condition. The Pearson r between
ecm_source and ecm_reconstructed (across subjects) is annotated on each
violin — this is the key quality metric from Figure 2(b) of the paper.

Reference:
    Deco, G., Sanz Perl, Y., & Kringelbach, M. L. (2025). Complex harmonics
    reveal low-dimensional manifolds of critical brain dynamics.
    Physical Review E, 111(1). https://doi.org/10.1103/physreve.111.014410

Usage
-----
    from Neuroreduce.utils.ecm_plot import ECMPlotter

    plotter = ECMPlotter(
        ecm_source_rest        = ecm_src_rest,
        ecm_source_task        = ecm_src_task,
        ecm_recon_charm_rest   = ecm_charm_recon_rest,
        ecm_recon_charm_task   = ecm_charm_recon_task,
        ecm_recon_pca_rest     = ecm_pca_recon_rest,
        ecm_recon_pca_task     = ecm_pca_recon_task,
        group_labels           = ('REST', 'EMOTION'),
    )
    fig = plotter.plot_ecm_violins()
    fig.savefig('_Data_Produced/ecm_violins.pdf', bbox_inches='tight')
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy import stats


# ── colour palette — colourblind-friendly, print-safe ────────────────────────
_COLOURS = {
    'PCA':   {'REST': '#5B9BD5', 'TASK': '#1F5FAC'},
    'CHARM': {'REST': '#F4A261', 'TASK': '#C1440E'},
    'source': '#888888',
}
_VIOLIN_ALPHA  = 0.55
_DOT_SIZE      = 45
_DOT_ALPHA     = 0.85
_JITTER_SEED   = 42


class ECMPlotter:
    """
    Produces violin and scatter plots comparing ECM of original BOLD
    versus ECM of the reconstructed BOLD, for PCA and CHARM.

    Parameters
    ----------
    ecm_source_rest : np.ndarray, shape (n_subjects,)
        ECM of the ORIGINAL BOLD for the REST condition.
        Method-independent — the same reference for both PCA and CHARM.
    ecm_source_task : np.ndarray, shape (n_subjects,)
        ECM of the ORIGINAL BOLD for the TASK condition.
    ecm_recon_charm_rest : np.ndarray, shape (n_subjects,)
        ECM of the CHARM-RECONSTRUCTED BOLD for REST.
        X_hat = charm_reducer.inverse_transform(charm_reducer.transform(X_sub))
    ecm_recon_charm_task : np.ndarray, shape (n_subjects,)
        ECM of the CHARM-RECONSTRUCTED BOLD for TASK.
    ecm_recon_pca_rest : np.ndarray, shape (n_subjects,)
        ECM of the PCA-RECONSTRUCTED BOLD for REST.
    ecm_recon_pca_task : np.ndarray, shape (n_subjects,)
        ECM of the PCA-RECONSTRUCTED BOLD for TASK.
    group_labels : tuple of str
        Human-readable labels for the two conditions.
    """

    def __init__(
        self,
        ecm_source_rest:       np.ndarray,
        ecm_source_task:       np.ndarray,
        ecm_recon_charm_rest:  np.ndarray,
        ecm_recon_charm_task:  np.ndarray,
        ecm_recon_pca_rest:    np.ndarray,
        ecm_recon_pca_task:    np.ndarray,
        group_labels:          tuple[str, str] = ('REST', 'TASK'),
    ):
        self.ecm_source_rest      = np.asarray(ecm_source_rest)
        self.ecm_source_task      = np.asarray(ecm_source_task)
        self.ecm_recon_charm_rest = np.asarray(ecm_recon_charm_rest)
        self.ecm_recon_charm_task = np.asarray(ecm_recon_charm_task)
        self.ecm_recon_pca_rest   = np.asarray(ecm_recon_pca_rest)
        self.ecm_recon_pca_task   = np.asarray(ecm_recon_pca_task)
        self.group_labels         = group_labels
        self.n_subjects           = len(ecm_source_rest)

        lengths = [len(np.asarray(a)) for a in [
            ecm_source_rest, ecm_source_task,
            ecm_recon_charm_rest, ecm_recon_charm_task,
            ecm_recon_pca_rest,   ecm_recon_pca_task,
        ]]
        if len(set(lengths)) > 1:
            raise ValueError(f"All ECM arrays must have same length. Got: {lengths}")

    # -------------------------------------------------------------------------
    # Public: violin plot
    # -------------------------------------------------------------------------

    def plot_ecm_violins(
        self,
        figsize: tuple[float, float] = (9, 5),
        title:   Optional[str] = None,
        ax:      Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """
        Violin + strip plot of reconstructed ECM per method per condition.

        Each violin shows the distribution of RECONSTRUCTED ECM across
        subjects. Coloured dots = reconstructed ECM per subject; grey
        diamonds = source ECM (reference). Pearson r between source and
        reconstructed ECM is annotated above each violin.

        Parameters
        ----------
        figsize : tuple
        title : str or None
        ax : matplotlib.axes.Axes or None

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        rng = np.random.default_rng(_JITTER_SEED)

        # (method, condition_label, colour, recon_ecm, source_ecm)
        entries = [
            ('PCA',   self.group_labels[0], _COLOURS['PCA']['REST'],
             self.ecm_recon_pca_rest,   self.ecm_source_rest),
            ('PCA',   self.group_labels[1], _COLOURS['PCA']['TASK'],
             self.ecm_recon_pca_task,   self.ecm_source_task),
            ('CHARM', self.group_labels[0], _COLOURS['CHARM']['REST'],
             self.ecm_recon_charm_rest, self.ecm_source_rest),
            ('CHARM', self.group_labels[1], _COLOURS['CHARM']['TASK'],
             self.ecm_recon_charm_task, self.ecm_source_task),
        ]
        positions   = [1, 2, 4, 5]
        violin_data = [e[3] for e in entries]

        # ── violin bodies ─────────────────────────────────────────────────────
        vp = ax.violinplot(violin_data, positions=positions,
                           widths=0.7, showmeans=False,
                           showmedians=True, showextrema=True)
        for body, entry in zip(vp['bodies'], entries):
            body.set_facecolor(entry[2])
            body.set_edgecolor('white')
            body.set_alpha(_VIOLIN_ALPHA)
        for part in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
            vp[part].set_color('0.3')
            vp[part].set_linewidth(1.2)

        # ── individual subject dots ───────────────────────────────────────────
        for entry, pos in zip(entries, positions):
            method, group, colour, recon_ecm, source_ecm = entry

            # Reconstructed ECM — coloured circles
            jitter = rng.uniform(-0.12, 0.12, size=self.n_subjects)
            ax.scatter(pos + jitter, recon_ecm,
                       color=colour, s=_DOT_SIZE, alpha=_DOT_ALPHA,
                       edgecolors='white', linewidths=0.5, zorder=3)

            # Source ECM reference — grey diamonds
            jitter2 = rng.uniform(-0.12, 0.12, size=self.n_subjects)
            ax.scatter(pos + jitter2, source_ecm,
                       color=_COLOURS['source'], s=_DOT_SIZE * 0.65,
                       alpha=0.55, edgecolors='white', linewidths=0.4,
                       zorder=2, marker='D',
                       label='Source (original)' if pos == 1 else '_nolegend_')

            # ── Pearson r annotation ──────────────────────────────────────────
            # r = correlation(source_ecm, recon_ecm) across subjects.
            # This quantifies how well the reconstruction preserves dynamics.
            if self.n_subjects >= 3:
                r, p = stats.pearsonr(source_ecm, recon_ecm)
                stars = _pvalue_stars(p)
                ymax = float(np.max(recon_ecm))
                ax.text(pos, ymax, f'r={r:.2f}{stars}',
                        ha='center', va='bottom', fontsize=8,
                        color='0.2', fontweight='bold')

        # ── method separator and group labels ─────────────────────────────────
        ax.axvline(x=3, color='0.8', linestyle='--', linewidth=1.0, zorder=0)
        ymin_ax, ymax_ax = ax.get_ylim()
        label_y = ymax_ax + 0.03 * (ymax_ax - ymin_ax)
        ax.text(1.5, label_y, 'PCA',   ha='center', fontsize=11,
                fontweight='bold', color=_COLOURS['PCA']['TASK'])
        ax.text(4.5, label_y, 'CHARM', ha='center', fontsize=11,
                fontweight='bold', color=_COLOURS['CHARM']['TASK'])

        ax.set_xticks(positions)
        ax.set_xticklabels([f"{m}\n{g}" for m, g, *_ in entries], fontsize=9)

        # ── legend ────────────────────────────────────────────────────────────
        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=_COLOURS['PCA']['REST'],   markersize=7,
                   label=f'PCA  {self.group_labels[0]}  (reconstructed)'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=_COLOURS['PCA']['TASK'],   markersize=7,
                   label=f'PCA  {self.group_labels[1]}  (reconstructed)'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=_COLOURS['CHARM']['REST'], markersize=7,
                   label=f'CHARM {self.group_labels[0]}  (reconstructed)'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=_COLOURS['CHARM']['TASK'], markersize=7,
                   label=f'CHARM {self.group_labels[1]}  (reconstructed)'),
            Line2D([0], [0], marker='D', color='w',
                   markerfacecolor=_COLOURS['source'],        markersize=6,
                   label='Original BOLD  (source reference)'),
        ]
        ax.legend(handles=legend_elements, fontsize=7.5,
                  loc='lower right', framealpha=0.85)

        # ── styling ───────────────────────────────────────────────────────────
        ax.set_ylabel('Edge-centric metastability (ECM)', fontsize=10)
        ax.set_xlabel('Method  /  Condition', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(axis='y', alpha=0.2, linestyle=':', zorder=0)

        default_title = (
            'Edge-Centric Metastability: Original vs Reconstructed BOLD\n'
            f'n = {self.n_subjects} subjects per condition'
        )
        ax.set_title(title or default_title, fontsize=11, pad=14)
        fig.tight_layout()
        return fig

    # -------------------------------------------------------------------------
    # Public: scatter plot
    # -------------------------------------------------------------------------

    def plot_ecm_scatter(
        self,
        figsize: tuple[float, float] = (9, 4),
        title:   Optional[str] = None,
    ) -> plt.Figure:
        """
        Scatterplot of source ECM vs reconstructed ECM per subject.

        One panel per method. Each dot is one subject; REST and TASK shown
        as different markers. The dashed diagonal is the identity line
        (perfect reconstruction). Pearson r is annotated in each panel.

        Parameters
        ----------
        figsize : tuple
        title : str or None

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        panels = [
            ('PCA',   axes[0], self.ecm_recon_pca_rest,   self.ecm_recon_pca_task),
            ('CHARM', axes[1], self.ecm_recon_charm_rest, self.ecm_recon_charm_task),
        ]

        for method, ax, recon_rest, recon_task in panels:
            src_all   = np.concatenate([self.ecm_source_rest, self.ecm_source_task])
            recon_all = np.concatenate([recon_rest, recon_task])

            for cond, recon_ecm, src_ecm, marker, colour in [
                (self.group_labels[0], recon_rest, self.ecm_source_rest,
                 'o', _COLOURS[method]['REST']),
                (self.group_labels[1], recon_task, self.ecm_source_task,
                 's', _COLOURS[method]['TASK']),
            ]:
                ax.scatter(src_ecm, recon_ecm,
                           color=colour, marker=marker, s=55,
                           alpha=_DOT_ALPHA, edgecolors='white',
                           linewidths=0.6, label=cond, zorder=3)
                _draw_regression_line(ax, src_ecm, recon_ecm, colour)

            # Independent axis limits for x (source) and y (reconstructed).
            # Forcing both to the same range would clip CHARM dots off-screen
            # because the reconstructed BOLD amplitude is generally different
            # from the original — especially for CHARM where conet is
            # L2-normalised, producing a different ECM scale than source.
            def _margin(vals, frac=0.08):
                p = np.ptp(vals) if np.ptp(vals) > 0 else 1.0
                return [vals.min() - frac * p, vals.max() + frac * p]

            xlim = _margin(src_all)
            ylim = _margin(recon_all)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            # Diagonal reference line over the OVERLAPPING range of x and y.
            # When axes differ in scale, the line shows "equal relative change"
            # rather than "perfect reconstruction" in absolute terms.
            # We draw it only over the intersection so it does not dominate.
            diag_min = max(xlim[0], ylim[0])
            diag_max = min(xlim[1], ylim[1])
            if diag_min < diag_max:
                ax.plot([diag_min, diag_max], [diag_min, diag_max],
                        '--', color='0.75', linewidth=1.0,
                        zorder=1, label='y = x')

            # Overall Pearson r annotation (pooled across conditions)
            if len(src_all) >= 3:
                r_all, p_all = stats.pearsonr(src_all, recon_all)
                stars = _pvalue_stars(p_all)
                ax.text(0.05, 0.95, f'r = {r_all:.3f}{stars}',
                        transform=ax.transAxes,
                        ha='left', va='top', fontsize=9.5,
                        color='0.15', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', alpha=0.75,
                                  edgecolor='0.8'))

            ax.set_xlabel('ECM — original BOLD (source)', fontsize=9)
            ax.set_ylabel('ECM — reconstructed BOLD', fontsize=9)
            ax.set_title(method, fontsize=11, fontweight='bold',
                         color=_COLOURS[method]['TASK'])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(alpha=0.2, linestyle=':', zorder=0)
            ax.legend(fontsize=8, framealpha=0.85)

        default_title = (
            'ECM: Original BOLD vs Reconstructed BOLD per Subject\n'
            f'(n = {self.n_subjects} subjects per condition)'
        )
        fig.suptitle(title or default_title, fontsize=11, y=1.02)
        fig.tight_layout()
        return fig


# =============================================================================
# Private helpers
# =============================================================================

def _pvalue_stars(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


def _draw_regression_line(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray,
    colour: str, alpha: float = 0.45,
) -> None:
    if len(x) < 2:
        return
    m, b   = np.polyfit(x, y, deg=1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, m * x_line + b,
            color=colour, alpha=alpha, linewidth=1.5, zorder=2)
