import pandas as pd
import matplotlib.pyplot as plt
from ..config import PCA_RESULTS, FIG_DPI


def scatter_matrix(X, y, feature_names, save=True):
    """1.2 Матрица разброса признаков с окраской по истинным меткам."""
    df = pd.DataFrame(X, columns=feature_names)
    df["y"] = y

    axes = pd.plotting.scatter_matrix(
        df, figsize=(12, 12), c=df["y"], cmap="viridis", alpha=0.7,
        diagonal="hist", marker="o", edgecolors="k"
    )
    fig = axes[0, 0].get_figure()
    fig.suptitle("Матрица разброса признаков (цвет — истинные метки y)",
                 fontsize=14)
    fig.tight_layout()
    if save:
        fig.savefig(PCA_RESULTS / "1_2_scatter_matrix.png", dpi=FIG_DPI)
    plt.close(fig)