import numpy as np
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.preprocessing import StandardScaler


def prepare_data(X):
    """Стандартизация перед анализом компонент."""
    return StandardScaler().fit_transform(X)


def run_pca(X_scaled, n_components=None):
    """PCA: возвращает модель и матрицу score-проекций."""
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X_scaled)
    return pca, scores


def run_svd(X_scaled, n_components=None):
    """SVD: сингулярное разложение."""
    svd = TruncatedSVD(n_components=n_components or X_scaled.shape[1] - 1,
                       random_state=42)
    scores = svd.fit_transform(X_scaled)
    return svd, scores


def run_ica(X_scaled, n_components=None):
    """ICA: анализ независимых компонент."""
    ica = FastICA(n_components=n_components or X_scaled.shape[1],
                  random_state=42, max_iter=1000, whiten="unit-variance")
    scores = ica.fit_transform(X_scaled)
    return ica, scores