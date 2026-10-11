"""Запуск части 2: кластеризация."""
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_clustering_dataset
from src.part2_clustering.algorithms import fit_all, plot_all_clusterings
from src.part2_clustering.tuning import tune_dbscan, plot_tuning_heatmap, tune_hdbscan
from src.part2_clustering.outliers import plot_outliers
from src.part2_clustering.comparison import compare_all, plot_comparison
from src.config import CLUSTER_RESULTS, FIG_DPI


def main():
    X, y, feature_names = load_clustering_dataset()
    print(f"Датасет: {X.shape[0]} объектов, {X.shape[1]} признаков")

    # 2.1–2.2
    labels_dict = fit_all(X)
    plot_all_clusterings(X, labels_dict, y_true=y)

    # 2.3 — подбор параметров для плотностных методов
    db_results, best_db = tune_dbscan(X)
    plot_tuning_heatmap(db_results)
    print(f"Лучший DBSCAN: eps={best_db.get('eps')}, "
          f"min_samples={best_db.get('min_samples')}, "
          f"silhouette={best_db.get('ari_proxy'):.3f}")

    hdb_results = tune_hdbscan(X)
    if hdb_results:
        best_hdb = max(hdb_results, key=lambda r:
                       r["silhouette"] if not np.isnan(r["silhouette"]) else -1)
        print(f"Лучший HDBSCAN: min_cluster_size="
              f"{best_hdb['min_cluster_size']}, "
              f"silhouette={best_hdb['silhouette']:.3f}")

    # 2.4 — визуализация аномалий
    for name in ("DBSCAN", "HDBSCAN", "OPTICS"):
        if name in labels_dict:
            plot_outliers(X, labels_dict[name],
                          title=f"{name}_outliers")

    # 2.5 — сравнение с эталоном
    df = compare_all(X, y, labels_dict)
    print("\nСравнение алгоритмов:\n", df.round(3))
    for m in ("ARI", "NMI", "Silhouette"):
        plot_comparison(df, metric=m)

    best_name = df["ARI"].idxmax()
    print(f"\n>>> Лучший алгоритм по ARI: {best_name} "
          f"(ARI={df.loc[best_name, 'ARI']:.3f})")

    # Итоговый график лучшего алгоритма
    fig, ax = plt.subplots(figsize=(8, 6))
    lab = labels_dict[best_name]
    mask = lab == -1
    ax.scatter(X[~mask, 0], X[~mask, 1], c=lab[~mask], cmap="tab10",
               s=30, edgecolors="k")
    if mask.any():
        ax.scatter(X[mask, 0], X[mask, 1], c="red", marker="x",
                   s=40, label="outliers")
    ax.set_title(f"Лучший алгоритм: {best_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CLUSTER_RESULTS / "2_5_best_algorithm.png", dpi=FIG_DPI)
    plt.close(fig)


if __name__ == "__main__":
    main()