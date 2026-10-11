from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PCA_RESULTS = RESULTS_DIR / "pca"
CLUSTER_RESULTS = RESULTS_DIR / "clustering"

for p in (PCA_RESULTS, CLUSTER_RESULTS):
    p.mkdir(parents=True, exist_ok=True)

PCA_DATASET = DATA_DIR / "Lab2_Dataset_PCA.xlsx"
CLUSTER_DATASET = DATA_DIR / "Lab2_Dataset_Clustering.xlsx"

RANDOM_STATE = 42
FIG_DPI = 120

# Пороги для классификации компонент (по доле объяснённой дисперсии)
VAR_PRIMARY = 0.10      # > 10% — главная
VAR_SECONDARY = 0.03    # 3–10% — второстепенная
                        # < 3% — шумовая