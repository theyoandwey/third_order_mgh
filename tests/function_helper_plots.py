"""Visualisation helpers for regularized cubic models and Armijo linesearch."""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from function_helper import M_x_sig, f4


def plot_step_size_impact(
    func=None, x0=None, sigmas=None,
    x_range=(-0.5, 1.0), n_pts=600,
    outdir=".", filename="step_size_impact",
    vertical=False, y_lo_factor=0.12, y_hi_factor=0.12,
    show_legend=True, sol_color='white',
):
    """Show the effect of the regularization parameter σ on the model M_σ.

    For each σ value, plots the true function f (black) and the regularized
    cubic model M_σ (coloured dashed), marks the current iterate x_k and the
    minimizer of M_σ (proposed next step), and draws the step arrow.
    Saves {filename}.pdf (serif) and {filename}.png (Arial, slide-ready).
    """
    import matplotlib.pyplot as plt
    import matplotlib.text as _mtext

    if func   is None: func   = f4()
    if x0     is None: x0     = np.array([0.5])
    if sigmas is None: sigmas = [2.0, 16.0, 64.0]

    x_k  = float(x0[0])
    fx   = func.f(x0)
    gx   = func.grad(x0)
    Hx   = func.hess(x0)

    x_vals = np.linspace(x_range[0], x_range[1], n_pts)
    f_vals = np.array([func.f(np.array([x])) for x in x_vals])

    y_lo = np.min(f_vals) - y_lo_factor * (np.max(f_vals) - np.min(f_vals))
    y_hi = np.max(f_vals) + y_hi_factor * (np.max(f_vals) - np.min(f_vals))

    fs = 20
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset":   "dejavusans",
        "font.size":          fs,
        "axes.labelsize":     fs,
        "axes.titlesize":     fs + 2,
        "xtick.labelsize":    fs - 3,
        "ytick.labelsize":    fs - 3,
        "legend.fontsize":    fs - 4,
        "axes.linewidth":     1.0,
        "xtick.major.width":  1.0,
        "ytick.major.width":  1.0,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    })

    colors = ['#777777', '#777777', '#777777']

    if vertical:
        nrows, ncols = len(sigmas), 1
        figsize = (6.5, 4.2 * nrows)
    else:
        nrows, ncols = 1, len(sigmas)
        figsize = (5.5 * ncols, 5.2)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    if len(sigmas) == 1:
        axes = [axes]
    else:
        axes = list(axes.flat)

    h_f = h_m = h_xk = h_xn = h_arr = None

    for i, (ax, sigma, color) in enumerate(zip(axes, sigmas, colors)):
        model  = M_x_sig(sigma, x0, fx, gx, Hx)
        m_vals = np.array([model.f(np.array([x])) for x in x_vals])
        m_plot = np.where(
            (m_vals >= y_lo - abs(y_hi - y_lo)) & (m_vals <= y_hi + abs(y_hi - y_lo)),
            m_vals, np.nan)

        finite = np.isfinite(m_plot)
        if finite.any():
            best     = int(np.nanargmin(m_plot))
            x_next   = x_vals[best]
            m_x_next = float(m_plot[best])
            f_x_next = float(func.f(np.array([x_next])))
        else:
            x_next = None

        h_f, = ax.plot(x_vals, f_vals, color='black', linewidth=2.4, zorder=3)
        h_m, = ax.plot(x_vals, m_plot, color=color, linewidth=2.2,
                       linestyle='--', zorder=4)
        ax.axvline(x_k, color='gray', linewidth=1.0, linestyle=':', alpha=0.6, zorder=2)
        h_xk = ax.scatter([x_k], [fx], color='black', s=100, zorder=6, clip_on=False)

        if x_next is not None:
            h_xn = ax.scatter([x_next], [m_x_next], color=color, marker='D',
                              s=110, zorder=7, edgecolors='white', linewidths=1.3)
            h_fn = ax.scatter([x_next], [f_x_next], color='black', marker='o',
                              s=100, zorder=7,
                              facecolors=sol_color, edgecolors='black', linewidths=1.8)
            ax.plot([x_next, x_next], [m_x_next, f_x_next],
                    color='black', linewidth=1.2, linestyle=':', zorder=6)
            arrow_y = y_lo + 0.05 * (y_hi - y_lo)
            h_arr = ax.annotate(
                '', xy=(x_next, arrow_y), xytext=(x_k, arrow_y),
                arrowprops=dict(arrowstyle='->', color='dimgray',
                                lw=1.8, mutation_scale=16),
                zorder=5)

        ax.set_title(rf'$\sigma = {sigma}$', fontweight='bold', pad=10)
        ax.set_xlabel(r'$x$', labelpad=6)
        if i == 0 or vertical:
            ax.set_ylabel('function value', labelpad=8)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(x_range)
        ax.grid(True, color='0.88', linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)

    leg_handles = [h_f, h_m, h_xk]
    leg_labels  = [
        r'$f(x)$: true function',
        r'$M_\sigma(x)$: regularized cubic model',
        r'$x_k$: current iterate',
    ]
    if h_xn is not None:
        leg_handles += [h_xn, h_fn]
        leg_labels  += [
            r'$\hat{x}$: minimizer of $M_\sigma$ (proposed step)',
            r'$f(\hat{x})$: actual value at proposed step',
        ]

    if show_legend:
        leg_bottom = 0.06 if not vertical else 0.04
        leg_rect   = [0, 0.10, 1, 1] if not vertical else [0, 0.07, 1, 1]
        fig.legend(leg_handles, leg_labels,
                   loc='upper center', bbox_to_anchor=(0.5, leg_bottom),
                   ncol=2 if vertical else len(leg_handles),
                   framealpha=0.92, edgecolor='0.75',
                   handlelength=1.6, columnspacing=1.0, borderpad=0.8,
                   fontsize=fs - 5)
    else:
        leg_rect = [0, 0, 1, 1]

    plt.tight_layout(rect=leg_rect)
    os.makedirs(outdir, exist_ok=True)
    base = f"{outdir}/{filename}"
    fig.savefig(f"{base}.pdf", bbox_inches='tight')
    for obj in fig.findobj(_mtext.Text):
        obj.set_fontfamily('sans-serif')
        try:
            obj.set_fontname('Arial')
        except Exception:
            pass
    fig.savefig(f"{base}.png", dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {base}.pdf / .png")


def plot_armijo_illustration(outdir=".", filename="armijo_illustration"):
    """Two-panel illustration of Armijo backtracking: rejected step (left), accepted step (right)."""
    import matplotlib.pyplot as plt
    import matplotlib.text as _mtext
    from matplotlib.lines import Line2D

    def f(x):
        return x**3 - 4*x + 3

    def df(x):
        return 3*x**2 - 4

    c    = 0.5
    x_k  = -0.5
    fx_k = f(x_k)
    gx_k = df(x_k)
    d_k  = -gx_k

    t_rej, t_acc = 0.8, 0.4
    x_rej = x_k + t_rej * d_k
    x_acc = x_k + t_acc * d_k

    x_lo, x_hi = -1.5, 2.5
    y_lo, y_hi = -2.0, 8.5
    x_vals = np.linspace(x_lo, x_hi, 600)
    f_vals = f(x_vals)
    x_arm  = np.linspace(x_k, x_hi, 400)
    arm_vals = fx_k + c * gx_k * (x_arm - x_k)

    fs = 20
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset":  "dejavusans",
        "font.size":         fs,
        "axes.labelsize":    fs,
        "axes.titlesize":    fs,
        "xtick.labelsize":   fs - 4,
        "ytick.labelsize":   fs - 4,
        "legend.fontsize":   fs - 5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    1.0,
    })

    GREY  = '#777777'
    RED   = '#c0392b'
    GREEN = '#27ae60'

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    panels = [
        (axes[0], t_rej, x_rej, 'Initial step',      RED,   'REJECTED'),
        (axes[1], t_acc, x_acc, 'After backtracking', GREEN, 'ACCEPTED'),
    ]

    for i, (ax, t_step, x_step, panel_title, col, verdict) in enumerate(panels):
        f_step   = f(x_step)
        arm_step = fx_k + c * gx_k * (x_step - x_k)

        ax.plot(x_vals, f_vals, color='black', linewidth=2.4, zorder=3)
        ax.plot(x_arm, arm_vals, color=GREY, linewidth=1.8, linestyle='--', zorder=4)
        ax.scatter([x_k], [fx_k], color='black', s=110, zorder=7)
        ax.axvline(x_k, color='gray', linewidth=0.8, linestyle=':', alpha=0.4, zorder=1)
        ax.scatter([x_step], [f_step], color=col, s=130, zorder=8,
                   edgecolors='white', linewidths=1.5)
        y_bot, y_top = min(f_step, arm_step), max(f_step, arm_step)
        ax.plot([x_step, x_step], [y_bot, y_top],
                color=col, linewidth=1.6, linestyle=':', zorder=5)
        arrow_y = y_lo + 0.07 * (y_hi - y_lo)
        ax.annotate('', xy=(x_step, arrow_y), xytext=(x_k, arrow_y),
                    arrowprops=dict(arrowstyle='->', color='dimgray',
                                    lw=1.8, mutation_scale=16), zorder=6)
        ax.text(0.97, 0.97, verdict, transform=ax.transAxes, ha='right', va='top',
                fontsize=fs - 2, fontweight='bold', color=col,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=col, linewidth=1.5))
        ax.set_title(f'{panel_title}  ($t = {t_step}$)', pad=10)
        ax.set_xlabel(r'$x$', labelpad=6)
        if i == 0:
            ax.set_ylabel(r'$f(x)$', labelpad=8)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(x_lo, x_hi)
        ax.grid(True, color='0.90', linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

    leg_items = [
        (Line2D([0], [0], color='black', lw=2.4),               r'$f(x)$'),
        (Line2D([0], [0], color=GREY, lw=1.8, ls='--'),          r'Armijo bound'),
        (Line2D([0], [0], marker='o', color='black', ms=9, ls='none'), r'$x_k$'),
        (Line2D([0], [0], marker='o', color=RED,   ms=9, ls='none', mec='white'), 'rejected step'),
        (Line2D([0], [0], marker='o', color=GREEN, ms=9, ls='none', mec='white'), 'accepted step'),
    ]
    fig.legend([h for h, _ in leg_items], [lbl for _, lbl in leg_items],
               loc='upper center', bbox_to_anchor=(0.5, 0.07),
               ncol=len(leg_items), framealpha=0.93, edgecolor='0.75',
               handlelength=1.6, columnspacing=0.9, borderpad=0.7,
               fontsize=fs - 5)

    plt.tight_layout(rect=[0, 0.11, 1, 1])
    os.makedirs(outdir, exist_ok=True)
    base = f"{outdir}/{filename}"
    fig.savefig(f"{base}.pdf", bbox_inches='tight')
    for obj in fig.findobj(_mtext.Text):
        obj.set_fontfamily('sans-serif')
        try:
            obj.set_fontname('Arial')
        except Exception:
            pass
    fig.savefig(f"{base}.png", dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {base}.pdf / .png")


if __name__ == '__main__':
    plot_step_size_impact(
        func=f4(), x0=np.array([0.5]),
        sigmas=[2.0, 16.0, 64.0],
        x_range=(-0.5, 1.0),
        outdir=".",
        filename="step_size_impact",
    )
    plot_step_size_impact(
        func=f4(), x0=np.array([0.5]),
        sigmas=[2.0, 16.0],
        x_range=(-0.5, 1.0),
        outdir=".",
        filename="step_size_impact_slide",
        vertical=True,
        y_lo_factor=0.55,
        show_legend=False,
    )
    plot_step_size_impact(
        func=f4(), x0=np.array([0.5]),
        sigmas=[16.0],
        x_range=(0.1, 1.0),
        outdir=".",
        filename="step_size_impact_s16",
        y_lo_factor=0.08,
        y_hi_factor=0.06,
        show_legend=False,
        sol_color='#E07800',
    )
    plot_armijo_illustration(outdir=".", filename="armijo_illustration")
