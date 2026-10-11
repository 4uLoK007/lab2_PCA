import numpy as np
import matplotlib.pyplot as plt
from ..config import CLUSTER_RESULTS, FIG_DPI, RANDOM_STATE

from sklearn.cluster import (
    KMeans, AffinityPropagation, MeanShift, SpectralClustering,
    AgglomerativeClustering, DBSCAN, OPTICS, Birch
)
from sklearn.mixture import GaussianMixture
from sklearn.cluster import BisectingKMeans
try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


def make_algorithms(X):
    """Возвращает словарь {имя: экземпляр модели} с подобранными параметрами."""
    n = len(X)
    return {
        "KMeans":        KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10),
        "AffinityProp":  AffinityPropagation(random_state=RANDOM_STATE),
        "MeanShift":     MeanShift(),
        "Spectral":      SpectralClustering(n_clusters=3, random_state=RANDOM_STATE,
                                             assign_labels="kmeans", affinity="nearest_neighbors"),
        "Ward":          AgglomerativeClustering(n_clusters=3, linkage="ward"),
        "Agglomerative": AgglomerativeClustering(n_clusters=3, linkage="average",
                                                  metric="euclidean"),
        "DBSCAN":        DBSCAN(eps=0.15, min_samples=5),
        "HDBSCAN":       hdbscan.HDBSCAN(min_cluster_size=15) if HAS_HDBSCAN else None,
        "OPTICS":        OPTICS(min_samples=5, xi=0.05),
        "GMM":           GaussianMixture(n_components=3, random_state=RANDOM_STATE),
        "BIRCH":         Birch(n_clusters=3, threshold=0.5),
        "BisectingKMeans": BisectingKMeans(n_clusters=3, random_state=RANDOM_STATE),
    }


def fit_all(X):
    """Обучает все алгоритмы, возвращает dict {имя: labels}."""
    models = make_algorithms(X)
    labels = {}
    for name, model in models.items():
        if model is None:
            continue
        try:
            if name == "GMM":
                lab = model.fit_predict(X)
            else:
                lab = model.fit_predict(X)
            labels[name] = lab
        except Exception as e:
            print(f"[!] {name}: {e}")
            labels[name] = np.zeros(len(X), dtype=int)
    return labels


def plot_all_clusterings(X, labels_dict, y_true=None, ncols=4):
    """2.2 Общий график plt.subplots для всех алгоритмов."""
    names = list(labels_dict.keys())
    n = len(names)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows))
    axes = axes.ravel()

    for ax, name in zip(axes, names):
        lab = labels_dict[name]
        # Точки выбросов (label == -1) — чёрные
        mask_noise = lab == -1
        ax.scatter(X[~mask_noise, 0], X[~mask_noise, 1],
                   c=lab[~mask_noise], cmap="tab10", s=14, alpha=0.8)
        if mask_noise.any():
            ax.scatter(X[mask_noise, 0], X[mask_noise, 1],
                       c="k", marker="x", s=25, label="noise")
        n_clusters = len(set(lab) - {-1})
        ax.set_title(f"{name}\n{n_clusters} clusters", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes[n:]:
        ax.axis("off")

    if y_true is not None:
        fig.suptitle("Результаты алгоритмов кластеризации", fontsize=14)
    fig.tight_layout()
    fig.savefig(CLUSTER_RESULTS / "2_2_all_algorithms.png", dpi=FIG_DPI)
    plt.close(fig)