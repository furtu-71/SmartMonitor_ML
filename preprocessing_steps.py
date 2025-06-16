import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

# 1) VALIDACIÓN DE COLUMNAS ────────────────────────────────────────
REQUIRED_COLUMNS = [
    'experiment_ID', 'Timestamp',
    'L_1','L_2','L_3','L_4','L_5','L_6','L_7','L_8','L_9','L_10',
    'A_1','A_2','A_3','A_4','A_5',
    'B_1','B_2','B_3','B_4','B_5',
    'C_1','C_2','C_3','C_4','C_5'
]

def validate_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")
    return df


# 2) EXPERIMENT_ID → CATEGORÍA ─────────────────────────────────────
def convert_experiment_id_to_category(df):
    df = df.copy()
    df['experiment_ID'] = df['experiment_ID'].astype('category')
    return df


# 3) TIMESTAMP → FEATURES RELATIVAS ────────────────────────────────
def process_timestamp_features(df):
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ns', errors='coerce')
    frames = []

    for exp, g in df.groupby('experiment_ID'):
        g = g.dropna(subset=['Timestamp']).copy()
        if g.empty:
            continue
        t0 = g['Timestamp'].min()
        g['elapsed_seconds'] = (g['Timestamp'] - t0).dt.total_seconds()
        dt = pd.to_datetime(g['elapsed_seconds'], unit='s', origin='1970-01-01')

        g['year']        = dt.dt.year.astype('int32')
        g['month']       = dt.dt.month.astype('int32')
        g['day']         = dt.dt.day.astype('int32')
        g['hour']        = dt.dt.hour.astype('int32')
        g['minute']      = dt.dt.minute.astype('int32')
        g['second']      = dt.dt.second.astype('int32')
        g['day_of_week'] = dt.dt.dayofweek.astype('int32')

        frames.append(g)

    return pd.concat(frames, ignore_index=True)


# 4) ELIMINAR DUPLICADOS ───────────────────────────────────────────
def remove_duplicates(df):
    return df.drop_duplicates().copy()


# 5) AÑADIR COLUMNAS DE OUTLIERS (Modified Z-score) ────────────────
MONITORED_VARS = [
    'L_1','L_2','L_3','L_4','L_5','L_6','L_7','L_8','L_9','L_10',
    'A_1','A_2','A_3','A_4','A_5',
    'B_1','B_2','B_3','B_4','B_5',
    'C_1','C_2','C_3','C_4','C_5'
]

def _modified_z(series, thr=3.5):
    med = np.median(series)
    mad = np.median(np.abs(series - med)) or 1e-6
    z   = 0.6745 * (series - med) / mad
    return np.abs(z) > thr, z

def add_outlier_columns(df, columns=MONITORED_VARS, threshold=3.5):
    df = df.copy()
    for col in columns:
        out, z = _modified_z(df[col], threshold)
        df[f'{col}_modified_z'] = z
        df[f'{col}_outlier']    = out
    return df

# 5-B) IMPUTACIÓN MEDIANA  ───────────────────────────────────────
def median_impute_numeric(df):
    """
    Imputa la mediana sólo en las columnas numéricas
    y devuelve de nuevo un DataFrame.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    return df

# 6) ESCALADO ROBUSTO ──────────────────────────────────────────────
EXCLUDED = ['Timestamp', 'experiment_ID'] + [f'{c}_outlier' for c in MONITORED_VARS]

def robust_scale_features(df, excluded_cols=EXCLUDED):
    df = df.copy()
    feats = [c for c in df.columns if c not in excluded_cols]
    scaler = RobustScaler()
    df[feats] = scaler.fit_transform(df[feats])
    return df
