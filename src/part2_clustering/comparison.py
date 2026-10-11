import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ..config import CLUSTER_RESULTS, FIG_DPI
from ..utils.metrics import evaluate


def compare_all(X, y_true, labels_dict):
    rows = []
    for name, lab in labels_dict.items():
        res = evaluate(X, y_true, lab)
        res["algorithm"] = name
        rows.append(res)
    df = pd.DataFrame(rows).set_index("algorithm")
    df = df.sort_values("ARI", ascending=False)
    df.to_csv(CLUSTER_RESULTS / "2_5_comparison.csv", float_format="%.4f")
    return df


def plot_comparison(df, metric="ARI"):
    fig, ax = plt.subplots(figsize=(10, 5))
    df[metric].plot(kind="barh", ax=ax, color="tab:blue")
    ax.set_xlabel(metric)
    ax.set_title(f"Сравнение алгоритмов по {metric}")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(CLUSTER_RESULTS / f"2_5_bar_{metric}.png", dpi=FIG_DPI)
    plt.close(fig)