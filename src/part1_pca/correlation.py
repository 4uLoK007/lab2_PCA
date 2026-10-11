import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ..config import PCA_RESULTS, FIG_DPI


def correlation_matrix(X, feature_names, save=True):
    """1.1 Матрица взаимных корреляций между признаками."""
    df = pd.DataFrame(X, columns=feature_names)
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Матрица взаимных корреляций признаков")
    fig.tight_layout()
    if save:
        fig.savefig(PCA_RESULTS / "1_1_correlation_matrix.png", dpi=FIG_DPI)
    plt.close(fig)

    return corr