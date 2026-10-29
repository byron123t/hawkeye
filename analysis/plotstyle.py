"""Minimal matplotlib style shared by the figure scripts (CVPR two-column page sizes)."""
FULL_WIDTH = 6.875        # inches, \textwidth
COL_WIDTH = 3.25          # inches, \columnwidth
TEXT_COLOR = "#222222"
BASE_COLOR = "#4c72b0"
HAWKEYE_COLOR = {"4b": "#008300", "8b": "#c0457a"}


def apply():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.titlesize": 6, "axes.labelsize": 6,
                         "axes.edgecolor": "#888888", "text.color": TEXT_COLOR,
                         "savefig.dpi": 250, "pdf.fonttype": 42})
    return plt
