import numpy as np
import matplotlib.pyplot as plt
from ..config import PCA_RESULTS, FIG_DPI, VAR_PRIMARY, VAR_SECONDARY


def explained_variance_ratios(model_name, model, X_scaled):
    """Возвращает доли объяснённой дисперсии для PCA / SVD."""
    if model_name == "PCA":
        return model.explained_variance_ratio_
    if model_name == "SVD":
        ev = model.explained_variance_ratio_
        return ev
    if model_name == "ICA":
        # Для ICA — нормированная дисперсия компонент как прокси
        components = model.components_
        variances = np.var(model.transform(X_scaled), axis=0)
        return variances / variances.sum()
    raise ValueError(model_name)


def classify_components(ratios, var_primary=VAR_PRIMARY, var_secondary=VAR_SECONDARY):
    """Разбивает индексы компонент на главные / второстепенные / шумовые."""
    primary, secondary, noise = [], [], []
    for i, v in enumerate(ratios):
        if v >= var_primary:
            primary.append(i)
        elif v >= var_secondary:
            secondary.append(i)
        else:
            noise.append(i)
    return primary, secondary, noise


def plot_component_importance(model_name, model, X_scaled, save=True):
    """График значимости компонент + кумулятивная дисперсия."""
    ratios = explained_variance_ratios(model_name, model, X_scaled)
    cum = np.cumsum(ratios)
    idx = np.arange(1, len(ratios) + 1)

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.bar(idx, ratios, alpha=0.6, label="Individual explained variance")
    ax1.set_xlabel("Component index")
    ax1.set_ylabel("Explained variance ratio")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.step(idx, cum, where="mid", color="tab:blue",
             label="Cumulative explained variance")
    ax2.set_ylabel("Cumulative ratio")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title(f"{model_name}: значимость компонент")

    primary, secondary, noise = classify_components(ratios)
    for i in primary:
        ax1.axvspan(i + 0.5, i + 1.5, color="green", alpha=0.08)
    for i in secondary:
        ax1.axvspan(i + 0.5, i + 1.5, color="orange", alpha=0.08)
    for i in noise:
        ax1.axvspan(i + 0.5, i + 1.5, color="red", alpha=0.08)

    fig.tight_layout()
    if save:
        fig.savefig(PCA_RESULTS / f"1_4_{model_name}_importance.png", dpi=FIG_DPI)
    plt.close(fig)

    return {
        "ratios": ratios,
        "cumulative": cum,
        "primary": primary,
        "secondary": secondary,
        "noise": noise,
    }