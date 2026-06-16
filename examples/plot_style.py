"""Shared plot styling for publication-quality figures.

Import this at the top of any example to get consistent styling:
    from plot_style import apply_style, label_panel
"""

import matplotlib.pyplot as plt
import matplotlib as mpl


def apply_style():
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update({
        # Font
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 13,

        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 5,

        # Axes
        'axes.linewidth': 0.8,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Ticks
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,

        # Figure
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,

        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
    })


def label_panel(ax, label, x=-0.12, y=1.08):
    """Add a panel label like (a), (b), (c) to an axes."""
    ax.text(x, y, f'({label})', transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top', ha='right')


def raise_window(fig=None):
    """Raise the matplotlib window to foreground (TkAgg on Linux)."""
    try:
        manager = (fig or plt.gcf()).canvas.manager
        manager.window.attributes('-topmost', True)
        manager.window.update()
        manager.window.attributes('-topmost', False)
    except Exception:
        pass


# Standard colors for consistency across all examples
COLORS = {
    'primary': '#2166ac',      # blue
    'secondary': '#b2182b',    # red
    'tertiary': '#1b7837',     # green
    'quaternary': '#762a83',   # purple
    'light': '#92c5de',        # light blue
    'accent': '#f4a582',       # salmon
    'gray': '#636363',
    'lightgray': '#bdbdbd',
}
