"""Запуск части 1: анализ признаков, PCA / SVD / ICA."""
import numpy as np
from src.data_loader import load_pca_dataset
from src.part1_pca.correlation import correlation_matrix
from src.part1_pca.scatter_matrix import scatter_matrix
from src.part1_pca.methods import prepare_data, run_pca, run_svd, run_ica
from src.part1_pca.component_analysis import (
    plot_component_importance, explained_variance_ratios
)
from src.part1_pca.biplot import make_biplots


def main():
    X, y, feature_names = load_pca_dataset(variant=1)
    print(f"Датасет: {X.shape[0]} объектов, {X.shape[1]} признаков")

    # 1.1
    corr = correlation_matrix(X, feature_names)
    print("\nМатрица корреляций:\n", corr.round(2))

    # 1.2
    scatter_matrix(X, y, feature_names)

    # 1.3
    Xs = prepare_data(X)
    pca, pca_scores = run_pca(Xs)
    svd, svd_scores = run_svd(Xs)
    ica, ica_scores = run_ica(Xs)

    # 1.4
    results = {}
    for name, model in [("PCA", pca), ("SVD", svd), ("ICA", ica)]:
        info = plot_component_importance(name, model, Xs)
        results[name] = info
        print(f"\n[{name}]")
        print("  Главные:      ", [i + 1 for i in info["primary"]])
        print("  Второстеп.:   ", [i + 1 for i in info["secondary"]])
        print("  Шумовые:      ", [i + 1 for i in info["noise"]])

    # 1.5 — biplot
    scores_loadings = {
        "PCA": (pca_scores, pca.components_),
        "SVD": (svd_scores, svd.components_),
        "ICA": (ica_scores, ica.components_),
    }
    explained = {
        "PCA": tuple(pca.explained_variance_ratio_[:2]),
        "SVD": tuple(svd.explained_variance_ratio_[:2]),
        "ICA": tuple(explained_variance_ratios("ICA", ica, Xs)[:2]),
    }
    make_biplots(scores_loadings, y, feature_names, explained)


if __name__ == "__main__":
    main()