from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score
)
import numpy as np


def evaluate(X, y_true, labels):
    res = {}
    mask = labels != -1
    # Внешние метрики
    if mask.sum() > 1:
        res["ARI"] = adjusted_rand_score(y_true[mask], labels[mask])
        res["NMI"] = normalized_mutual_info_score(y_true[mask], labels[mask])
    else:
        res["ARI"] = res["NMI"] = np.nan

    n_unique = len(set(labels) - {-1})
    if 1 < n_unique < len(X):
        try:
            res["Silhouette"] = silhouette_score(X[mask], labels[mask])
        except Exception:
            res["Silhouette"] = np.nan
    else:
        res["Silhouette"] = np.nan
    res["n_clusters"] = n_unique
    res["n_noise"] = int((labels == -1).sum())
    return res