import numpy as np
import matplotlib.pyplot as plt
from ..config import CLUSTER_RESULTS, FIG_DPI


def plot_outliers(X, labels, title="Аномалии (label == -1)"):
    """2.4 Визуализация кластеров + аномалий."""
    labels = np.asarray(labels)
    mask = labels == -1
    fig, ax = plt.subplots(figsize=(9, 7))

    if (~mask).any():
        n_clusters = len(set(labels) - {-1})
        sc = ax.scatter(X[~mask, 0], X[~mask, 1],
                        c=labels[~mask], cmap="tab10",
                        s=35, alpha=0.8, edgecolors="k")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Cluster label")
    if mask.any():
        ax.scatter(X[mask, 0], X[mask, 1], c="red", marker="x",
                   s=55, label=f"Outliers (n={mask.sum()})")
    ax.legend()
    ax.set_title(f"{title}\nКластеров: {len(set(labels) - {-1})}, "
                 f"аномалий: {mask.sum()}")
    fig.tight_layout()
    fig.savefig(CLUSTER_RESULTS / f"2_4_{title.replace(' ', '_')}.png",
                dpi=FIG_DPI)
    plt.close(fig)