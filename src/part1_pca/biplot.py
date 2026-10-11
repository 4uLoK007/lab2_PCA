import numpy as np
import matplotlib.pyplot as plt
from ..config import PCA_RESULTS, FIG_DPI


def biplot(scores, loadings, labels=None, targets=None,
           target_labels=None, title="", save_path=None):
    """Biplot: точки + векторы нагрузок признаков."""
    fig, ax = plt.subplots(figsize=(9, 7))

    scalex = 1.0 / (scores[:, 0].max() - scores[:, 0].min())
    scaley = 1.0 / (scores[:, 1].max() - scores[:, 1].min())

    if targets is not None:
        sc = ax.scatter(scores[:, 0] * scalex, scores[:, 1] * scaley,
                        c=targets, cmap="viridis", alpha=0.7,
                        edgecolors="k")
        handles, _ = sc.legend_elements()
        if target_labels is not None:
            ax.legend(handles, target_labels, title="Classes")
    else:
        ax.scatter(scores[:, 0] * scalex, scores[:, 1] * scaley,
                   alpha=0.7)

    n_features = loadings.shape[0]
    for i in range(n_features):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                 color="r", alpha=0.7, head_width=0.03,
                 head_length=0.03, linewidth=1.5)
        if labels is not None:
            ax.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15,
                    labels[i], color="r", fontsize=9)

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_title(title)

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=FIG_DPI)
    plt.close(fig)


def make_biplots(models_scores_loadings, y, feature_names, explained_pairs):
    """
    models_scores_loadings: dict {имя_метода: (scores, loadings)}
    explained_pairs: dict {имя_метода: (ratio1, ratio2)} — для заголовка
    """
    for name, (scores, loadings) in models_scores_loadings.items():
        r1, r2 = explained_pairs[name]
        biplot(
            scores[:, :2], loadings[:2, :].T,
            labels=feature_names, targets=y,
            title=f"{name}: {r1*100:.1f}% / {r2*100:.1f}% Variance",
            save_path=PCA_RESULTS / f"1_5_{name}_biplot.png",
        )