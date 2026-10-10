"""Shared chart styling - one palette, applied by role, used by every figure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE     = "#fcfcfb"
TEXT_1      = "#0b0b0b"
TEXT_2      = "#52514e"
TEXT_MUTED  = "#8a8982"
GRID        = "#e6e5e1"

# Categorical slots, fixed order - never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948"]
# Sequential blue, light -> dark (magnitude only).
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf",
       "#1c5cab", "#184f95", "#104281", "#0d366b"]
NEUTRAL = "#c9c8c2"

def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT_1,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "legend.labelcolor": TEXT_2,
        "font.family": "DejaVu Sans",
        "figure.dpi": 150,
        "lines.linewidth": 2.0,
    })

def strip(ax, keep=("left", "bottom"), xgrid=False):
    """Recessive axes: drop the box, keep grid on the measure axis only."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    ax.grid(axis="x" if xgrid else "y")
    ax.grid(axis="y" if xgrid else "x", visible=False)
    ax.tick_params(length=0)

def subtitle(fig, text, y=0.945):
    fig.text(0.5, y, text, ha="center", va="top", fontsize=9.8, color=TEXT_2)
