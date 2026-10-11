import pandas as pd
import numpy as np
from .config import PCA_DATASET, CLUSTER_DATASET


def _extract_variant_data(df, variant_number, n_features):
    """
    Простой и надежный парсер.
    1. Находит строку с названиями вариантов.
    2. Находит колонку, где начинается нужный вариант.
    3. Вырезает ровно n_features колонок признаков и 1 колонку 'y'.
    """
    # Убираем пустые строки
    df = df.dropna(how='all')

    # Ищем строку, где написано "Вариант 1", "Вариант 2" и т.д.
    header_row_idx = None
    for idx, row in df.iterrows():
        # Проверяем, есть ли в строке хотя бы одна ячейка со словом "Вариант"
        if row.astype(str).str.contains("Вариант").any():
            header_row_idx = idx
            break

    if header_row_idx is None:
        raise ValueError("Не найдена строка с заголовками 'Вариант ...'.")

    # Получаем названия колонок для этой строки
    header_row = df.iloc[header_row_idx].values

    # Ищем индекс колонки, где начинается нужный вариант
    # (если variant_number=None, берем первый вариант)
    start_col_idx = None
    if variant_number is None:
        # Ищем первый столбец, где есть "Вариант"
        for i, val in enumerate(header_row):
            if isinstance(val, str) and "Вариант" in val:
                start_col_idx = i
                break
        variant_name = "Вариант 1" # Условно
    else:
        target_name = f"Вариант {variant_number}"
        for i, val in enumerate(header_row):
            if isinstance(val, str) and target_name in val:
                start_col_idx = i
                break
        variant_name = target_name

    if start_col_idx is None:
        raise ValueError(f"Вариант {variant_number} не найден в заголовке.")

    # Проверяем, что после начала варианта есть достаточно колонок
    # (n_features признаков + 1 колонка 'y')
    if start_col_idx + n_features >= len(header_row):
         raise ValueError(f"Недостаточно колонок для варианта {variant_name}.")

    # Вырезаем данные, начиная со строки ПОСЛЕ строки с признаками
    # (строка с признаками 'a', 'b', 'y' идет сразу после строки с 'Вариант ...')
    data_start_row = header_row_idx + 2  # +1 для признаков, +1 для данных
    
    # Берем только нужные колонки и строки
    # Признаки
    X_cols = list(range(start_col_idx, start_col_idx + n_features))
    # Метка 'y' (она идет сразу после признаков)
    y_col = start_col_idx + n_features

    X_df = df.iloc[data_start_row:, X_cols].copy()
    y_series = df.iloc[data_start_row:, y_col].copy()

    # Убираем строки, где есть пустые значения (NaN)
    valid_mask = ~X_df.isna().any(axis=1) & ~y_series.isna()
    
    if valid_mask.sum() == 0:
        raise ValueError(f"Нет данных для варианта {variant_name}.")

    X_df = X_df[valid_mask]
    y_series = y_series[valid_mask]

    # Приводим типы
    X_df = X_df.astype(float)
    y_series = y_series.astype(int)

    # Переименовываем признаки
    feature_names = [f"Feature_{j+1}" for j in range(n_features)]
    X_df.columns = feature_names

    return X_df.values, y_series.values, feature_names


def load_pca_dataset(path=PCA_DATASET, variant=1):
    """
    Загружает данные для PCA.
    variant: номер варианта (1-24). По умолчанию 1.
    """
    df = pd.read_excel(path, header=None)
    # В PCA файле 5 признаков
    return _extract_variant_data(df, variant_number=variant, n_features=5)


def load_clustering_dataset(path=CLUSTER_DATASET, variant=1):
    """
    Загружает данные для кластеризации.
    variant: номер варианта (1-24). По умолчанию 1.
    """
    df = pd.read_excel(path, header=None)
    # В Clustering файле 2 признака
    return _extract_variant_data(df, variant_number=variant, n_features=2)