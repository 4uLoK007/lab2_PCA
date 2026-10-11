import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.metrics import silhouette_score
try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

from ..config import CLUSTER_RESULTS, FIG_DPI


def tune_dbscan(X, eps_range=None, ms_range=(3, 5, 8, 12)):
    """Подбор eps и min_samples для DBSCAN."""
    if eps_range is None:
        eps_range = np.linspace(0.05, 0.4, 15)
    best = {"ari_proxy": -1}
    results = []
    for eps in eps_range:
        for ms in ms_range:
            lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
            n_clusters = len(set(lab) - {-1})
            n_noise = int((lab == -1).sum())
            mask = lab != -1
            sil = np.nan
            if 1 < n_clusters < len(X):
                try:
                    sil = silhouette_score(X[mask], lab[mask])
                except Exception:
                    pass
            results.append({"eps": eps, "min_samples": ms,
                            "n_clusters": n_clusters,
                            "n_noise": n_noise, "silhouette": sil})
            if n_clusters >= 2 and not np.isnan(sil) and sil > best["ari_proxy"]:
                best = {"eps": eps, "min_samples": ms,
                        "ari_proxy": sil, "labels": lab}
    return results, best


def plot_tuning_heatmap(results, param_x="eps", param_y="min_samples"):
    """Тепловая карта silhouette по (eps, min_samples)."""
    import pandas as pd
    import seaborn as sns
    df = pd.DataFrame(results).pivot(
        index=param_y, columns=param_x, values="silhouette"
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", ax=ax)
    ax.set_title("DBSCAN: silhouette по (eps, min_samples)")
    fig.tight_layout()
    fig.savefig(CLUSTER_RESULTS / "2_3_dbscan_tuning.png", dpi=FIG_DPI)
    plt.close(fig)


def tune_hdbscan(X, min_cluster_sizes=(5, 10, 15, 20, 30)):
    if not HAS_HDBSCAN:
        return []
    out = []
    for mcs in min_cluster_sizes:
        lab = hdbscan.HDBSCAN(min_cluster_size=mcs).fit_predict(X)
        n_clusters = len(set(lab) - {-1})
        n_noise = int((lab == -1).sum())
        mask = lab != -1
        sil = np.nan
        if 1 < n_clusters < len(X):
            try:
                sil = silhouette_score(X[mask], lab[mask])
            except Exception:
                pass
        out.append({"min_cluster_size": mcs, "n_clusters": n_clusters,
                    "n_noise": n_noise, "silhouette": sil, "labels": lab})
    return out