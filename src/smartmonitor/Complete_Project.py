# --- rutas ------------------------------------------------------------------
import sys, pathlib
SRC_DIR = pathlib.Path(__file__).resolve().parents[2] / "src"   # ← dos niveles arriba
if str(SRC_DIR) not in sys.path:                                # evita duplicados
    sys.path.append(str(SRC_DIR))

# --- resto de imports -------------------------------------------------------
import importlib
from smartmonitor import preprocessing_steps        # alias igual que antes
importlib.reload(preprocessing_steps)

from smartmonitor.preprocessing_steps import (
    validate_columns, convert_experiment_id_to_category,
    process_timestamp_features, remove_duplicates,
    add_outlier_columns, robust_scale_features,
    median_impute_numeric, MONITORED_VARS, EXCLUDED
)

###################################################
# Paso 1 – Crear el módulo preprocessing_steps.py #
###################################################
#%%writefile preprocessing_steps.py
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

###############################################################################
# Paso 2 – Importar las funciones del módulo y montar los FunctionTransformer #
###############################################################################
import importlib, preprocessing_steps
importlib.reload(preprocessing_steps)

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from preprocessing_steps import (
    validate_columns, convert_experiment_id_to_category,
    process_timestamp_features, remove_duplicates,
    add_outlier_columns, robust_scale_features,
    median_impute_numeric, MONITORED_VARS, EXCLUDED
)

validation_tf = FunctionTransformer(validate_columns,                validate=False)
id_cat_tf     = FunctionTransformer(convert_experiment_id_to_category,validate=False)
timestamp_tf  = FunctionTransformer(process_timestamp_features,       validate=False)
imputer_tf    = FunctionTransformer(median_impute_numeric,            validate=False)
dup_tf        = FunctionTransformer(remove_duplicates,                validate=False)
outlier_tf    = FunctionTransformer(
                   add_outlier_columns,
                   kw_args={'columns': MONITORED_VARS, 'threshold': 3.5},
                   validate=False)
scaling_tf    = FunctionTransformer(
                   robust_scale_features,
                   kw_args={'excluded_cols': EXCLUDED},
                   validate=False)

preprocessing_pipeline = Pipeline(steps=[
    ('validacion'        , validation_tf),
    ('to_category'       , id_cat_tf),
    ('timestamp_features', timestamp_tf),
    ('imputacion'        , imputer_tf),       # ← usa la función DEL MÓDULO
    ('remove_duplicates' , dup_tf),
    ('outlier_features'  , outlier_tf),
    ('robust_scaling'    , scaling_tf),
])

print("Pipeline construido ✔️")

##################################################################
# Paso 3 – Ajustar el pipeline con tus datos y guardarlo en .pkl #
##################################################################

# Paso 3 · Fit + dump del pipeline
import pandas as pd
from joblib import dump

# --- 3-A · Cargar los datos crudos -------------------------------
CSV_PATH = "/content/drive/MyDrive/Colab Notebooks/MASTER DATA SCIENCE/Proyecto final/date_production.csv"
df_raw = pd.read_csv(CSV_PATH)
print("Datos cargados ->", df_raw.shape)

# --- 3-B · Ajustar el pipeline que creamos en el Paso 2 ----------
preprocessing_pipeline.fit(df_raw)

# --- 3-C · Serializar a disco ------------------------------------
dump(preprocessing_pipeline, "preprocessing_pipeline.pkl")
print("✔️  Pipeline entrenado y guardado como preprocessing_pipeline.pkl")

######################################################
# Paso 4 – Exportar los artefactos y probar la carga #
######################################################

from google.colab import files
files.download('preprocessing_pipeline.pkl')      # modelo
files.download('preprocessing_steps.py')          # módulo

############################################
# Paso 5 - Reentrenar RandomForesClassifer #
############################################


##############################################
# Paso 5.1. - Obtención muestra estratificad #
##############################################

# --- 1-A · Función de muestreo por fases --------------------------
def sample_experiment_phases(df, fraction=0.1,
                             time_col="Timestamp",
                             phases=3,
                             weights=None,
                             random_state=42):
    import numpy as np, pandas as pd
    if weights is None:
        weights = [1.0 / phases] * phases
    df_sorted = df.sort_values(by=time_col).copy()
    n_total = len(df_sorted)
    total_samples = int(n_total * fraction)

    # dividir en fases
    quantiles = [i / phases for i in range(phases + 1)]
    boundaries = df_sorted[time_col].quantile(quantiles).values
    df_sorted["phase"] = pd.cut(df_sorted[time_col],
                                bins=boundaries,
                                labels=False,
                                include_lowest=True)

    out = []
    for phase in range(phases):
        phase_data = df_sorted[df_sorted["phase"] == phase]
        n_phase = int(round(total_samples * weights[phase]))
        n_phase = min(n_phase, len(phase_data))
        out.append(phase_data.sample(n=n_phase,
                                     random_state=random_state))
    return pd.concat(out)

# --- 1-B · Cargar datos crudos -----------------------------------
RAW_CSV  = "/content/drive/MyDrive/Colab Notebooks/MASTER DATA SCIENCE/Proyecto final/date_production.csv"
df_raw   = pd.read_csv(RAW_CSV)
print("Dataset completo:", df_raw.shape)

# --- 1-C · Muestra 10 % estratificada -----------------------------
samples  = []
for exp_id, g in df_raw.groupby("experiment_ID"):
    samples.append(sample_experiment_phases(
        g, fraction=0.10,
        phases=3,
        weights=[0.2, 0.3, 0.5],
        random_state=42))
df_sample_strat = pd.concat(samples).reset_index(drop=True)
print("Muestra estratificada:", df_sample_strat.shape)

#############################################################
# Paso 5.2. - Preprocesado y obtención del data _modified_z #
#############################################################
import joblib, pandas as pd

PIPE_PKL = "/content/preprocessing_pipeline.pkl"   # ← cambia si lo tienes en otra ruta
pipe     = joblib.load(PIPE_PKL)

X_sample_trans = pipe.transform(df_sample_strat)   # DataFrame con nombres
print("Shape muestra tras pipeline:", X_sample_trans.shape)

# columnas *_modified_z (deben ser 25)
sensor_cols = [c for c in X_sample_trans.columns if "_modified_z" in c]
X_clust = X_sample_trans[sensor_cols].to_numpy()
print("Array para clustering:", X_clust.shape)

#########################################################################################################
# Paso 5.3. - Entrenar y guardar K-Means, OPTICS Y GMM (grid-search: min_samples = 3…9, xi = 0.01…0.10) #
#########################################################################################################

from sklearn.cluster import KMeans, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import numpy as np, joblib, time

# --------- datos de entrada (ya creados en PASO 2) ------------
# X_clust  →  np.array  (22 838 × 25)

k = 3
min_samples_range = range(3, 10)          # 3 … 9
xi_range = np.linspace(0.01, 0.1, 10)     # 0.01 … 0.10

tic = time.time()

############ K-MEANS ###########################################
kmeans = KMeans(n_clusters=k, init="k-means++",
                n_init=10, max_iter=300,
                random_state=42).fit(X_clust)
labels_kmeans = kmeans.labels_
print("K-Means listo – clusters:", np.unique(labels_kmeans))

############ OPTICS – grid search ##############################
best_score, best_params, best_labels = -1, None, None
for ms in tqdm(min_samples_range, desc="min_samples"):
    for xi in tqdm(xi_range, leave=False, desc="xi"):
        opt = OPTICS(min_samples=ms, xi=xi, cluster_method="xi")
        lbl = opt.fit_predict(X_clust)
        # excluir ruido
        if len(np.unique(lbl[lbl != -1])) < 2:
            continue
        try:
            score = silhouette_score(X_clust, lbl)
        except ValueError:
            score = -1
        if score > best_score:
            best_score, best_params, best_labels = score, (ms, xi), lbl
            optics = opt     # guarda el modelo ganador

print(f"Mejor OPTICS: min_samples={best_params[0]}, xi={best_params[1]}  "
      f"→ Silhouette {best_score:.3f}")

labels_optics = best_labels            # definitivos

############ GMM ################################################
gmm = GaussianMixture(n_components=k, random_state=42).fit(X_clust)
labels_gmm = gmm.predict(X_clust)
print("GMM listo – clusters:", np.unique(labels_gmm))

############ Guardar pickles ###################################
joblib.dump(kmeans, "kmeans.pkl")
joblib.dump(optics, "optics.pkl")
joblib.dump(gmm   , "gmm.pkl")
print(f"✓ KMeans, OPTICS, GMM guardados  ({time.time()-tic:.1f} s)")

######################################################
# PASO 5.4. - Matriz de coasociación + Agglomerative #
######################################################


# ‑ necesita en RAM:
#   • labels_kmeans, labels_optics, labels_gmm  (sobre la muestra 10 %)
#   • X_clust  (array 22 838 × 25 con *_modified_z)

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics  import (silhouette_score,
                              calinski_harabasz_score,
                              davies_bouldin_score)
import numpy as np, joblib, time

def compute_coassociation(labels_list):
    """
    Devuelve una matriz n×n donde cada celda es la fracción de algoritmos
    que asignan a la misma etiqueta el par (i, j).
    """
    n = len(labels_list[0])
    co = np.zeros((n, n), dtype=np.float32)
    for lab in labels_list:
        co += (lab[:, None] == lab[None, :]).astype(np.float32)
    return co / len(labels_list)

tic = time.time()

labels_list = [labels_kmeans, labels_optics, labels_gmm]
coassoc     = compute_coassociation(labels_list)           # (22 838 × 22 838)
distance    = 1.0 - coassoc                                # disimilaridad

# AgglomerativeClustering: metric≥1.4  •  affinity<1.4
try:
    ensemble = AgglomerativeClustering(
        n_clusters=3,
        metric="precomputed",
        linkage="average"
    ).fit(distance)
except TypeError:  # compatibilidad con versiones viejas
    ensemble = AgglomerativeClustering(
        n_clusters=3,
        affinity="precomputed",
        linkage="average"
    ).fit(distance)

ensemble_labels = ensemble.labels_
print(f"Ensemble creado en {time.time()-tic:.1f}s "
      f"– clusters: {np.unique(ensemble_labels)}")

# ─────────── Métricas en la muestra ───────────
sil = silhouette_score(X_clust, ensemble_labels)
cal = calinski_harabasz_score(X_clust, ensemble_labels)
dav = davies_bouldin_score(X_clust, ensemble_labels)

print("\nMÉTRICAS DEL ENSEMBLE (muestra 10 %):")
print(f"• Silhouette Score        : {sil:.3f}")
print(f"• Calinski-Harabasz Score : {cal:.1f}")
print(f"• Davies-Bouldin Score    : {dav:.3f}")

# ─────────── Guardar artefactos ───────────
joblib.dump(ensemble, "ensemble_agglom.pkl")
np.save("coassoc.npy", coassoc)  # (~2 GB; borra si no la necesitas)
print("✓ ensemble_agglom.pkl y coassoc.npy guardados")


#################################################
# X_sample_trans.pkl   y   y_cluster_sample.pkl #
#################################################

#En el cuaderno ya calculamos y_cluster para la muestra de 22 838 filas 
# cuando hicimos el ensemble, pero no lo guardamos en disco como y_cluster_sample.pkl 
# ni tampoco guardamos X_sample_trans.pkl. Tras perder la sesión de Colab, 
# cualquier variable que sólo vivía en RAM desapareció.

# ================================================================
# CREA  X_sample_trans.pkl   y   y_cluster_sample.pkl
# (muestra 10 % estratificada por fases · 22 838 filas aprox.)
# ================================================================

from google.colab import drive
import sys, os, joblib, pandas as pd, numpy as np
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ───────────────── 1 · Montar y preparar rutas ───────────────────
drive.mount('/content/drive')

MODEL_DIR = "/content/drive/MyDrive/Colab Notebooks/MASTER DATA SCIENCE/Proyecto final/models"
RAW_CSV   = "/content/drive/MyDrive/Colab Notebooks/MASTER DATA SCIENCE/Proyecto final/date_production.csv"

if MODEL_DIR not in sys.path:      # para que se pueda importar preprocessing_steps
    sys.path.append(MODEL_DIR)

pipe   = joblib.load(os.path.join(MODEL_DIR, "preprocessing_pipeline.pkl"))
kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))
optics = joblib.load(os.path.join(MODEL_DIR, "optics.pkl"))
gmm    = joblib.load(os.path.join(MODEL_DIR, "gmm.pkl"))

# ───────────────── 2 · Función de muestreo estratificado ─────────
def sample_experiment_phases(df, fraction=0.10,
                             time_col="Timestamp",
                             phases=3, weights=(0.2, 0.3, 0.5),
                             random_state=42):
    """Devuelve ~fraction de filas, estratificando en 3 tramos temporales."""
    df = df.sort_values(time_col).copy()
    q = [i / phases for i in range(phases + 1)]
    bins = df[time_col].quantile(q).values
    df["phase"] = pd.cut(df[time_col], bins=bins, labels=False, include_lowest=True)

    out = []
    total = int(len(df) * fraction)
    for p in range(phases):
        g = df[df["phase"] == p]
        n = int(round(total * weights[p]))
        n = min(n, len(g))
        out.append(g.sample(n=n, random_state=random_state))
    return pd.concat(out)

# ───────────────── 3 · Tomar la muestra y transformarla ──────────
df_raw = pd.read_csv(RAW_CSV)
sampled = []
for exp_id, g in df_raw.groupby("experiment_ID"):
    sampled.append(sample_experiment_phases(g))
df_sample = pd.concat(sampled).reset_index(drop=True)

X_sample_trans = pipe.transform(df_sample)          # DataFrame (22 8xx × 86)

# ───────────────── 4 · Ensemble de clustering sobre la muestra ───
sensor_cols = [c for c in X_sample_trans.columns if "_modified_z" in c]
X_clust     = X_sample_trans[sensor_cols].to_numpy()

labels_k = kmeans.predict(X_clust)
labels_g = gmm.predict(X_clust)
labels_o = optics.fit_predict(X_clust)              # OPTICS no tiene .predict

def coassoc(labels_list):
    n = len(labels_list[0])
    mat = np.zeros((n, n), dtype=np.float32)
    for lbl in labels_list:
        mat += (lbl[:, None] == lbl[None, :]).astype(np.float32)
    return mat / len(labels_list)

dist   = 1.0 - coassoc([labels_k, labels_o, labels_g])
ensemble = AgglomerativeClustering(n_clusters=3,
                                   metric="precomputed",
                                   linkage="average").fit(dist)
y_cluster = ensemble.labels_

print("Muestra transformada:", X_sample_trans.shape)
print("Etiquetas ensemble  :", np.bincount(y_cluster))

# ───────────────── 5 · Guardar los artefactos ────────────────────
X_sample_trans.to_pickle(os.path.join(MODEL_DIR, "X_sample_trans.pkl"))
pd.Series(y_cluster, name="cluster").to_pickle(
    os.path.join(MODEL_DIR, "y_cluster_sample.pkl")
)

print("✅  X_sample_trans.pkl  y  y_cluster_sample.pkl guardados en", MODEL_DIR)

########################################
# PASO 5.5.- ENTRENAR MODELO AUXILIAR. #
########################################

############### CARGA DE LOS ARTEFACTOS ###############

import sys, os, joblib, pandas as pd
from google.colab import drive
drive.mount("/content/drive")

MODEL_DIR = "/content/drive/MyDrive/Colab Notebooks/MASTER DATA SCIENCE/Proyecto final/models"
if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

pipe       = joblib.load(os.path.join(MODEL_DIR, "preprocessing_pipeline.pkl"))
rf_sample  = None          # aún no lo tenemos: lo crearemos en el punto 3

############### PONER EN MEMORIA LA MUESTRA TRANSFORMADA Y SUS ETIQUETAS DE ENSAMBLE ###############

X_sample_trans = pd.read_pickle(os.path.join(MODEL_DIR, "X_sample_trans.pkl"))   # 22 838 × 86
y_cluster      = pd.read_pickle(os.path.join(MODEL_DIR, "y_cluster_sample.pkl")) # 22 838

############### ENTRENA EL RANDOM FOREST AUXILIAR CON LA MUESTRA ###############

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
import numpy as np

# ─── columnas que quitamos ────────────────────────────
COLS_DROP = ["experiment_ID", "Timestamp", "phase"]   # ← añade phase

X_num   = X_sample_trans.drop(columns=COLS_DROP)
groups  = X_sample_trans["experiment_ID"].cat.codes

rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42
)

scores = cross_val_score(
            rf, X_num, y_cluster,
            cv=GroupKFold(n_splits=5).split(X_num, y_cluster, groups),
            scoring="accuracy", n_jobs=-1)
print("GroupKFold acc:", scores.round(4), "→ media", scores.mean().round(4))

rf.fit(X_num, y_cluster)
joblib.dump(rf, os.path.join(MODEL_DIR, "predictive_aux_model.pkl"))

############### PREDICE EL CLUSTER PARA TODO EL CSV CUANDO LO NECESITES ###############

pipe = joblib.load(os.path.join(MODEL_DIR, "preprocessing_pipeline.pkl"))
rf   = joblib.load(os.path.join(MODEL_DIR, "predictive_aux_model.pkl"))

RAW_CSV = ("/content/drive/MyDrive/Colab Notebooks/"
           "MASTER DATA SCIENCE/Proyecto final/date_production.csv")
df_raw  = pd.read_csv(RAW_CSV)

X_full  = pipe.transform(df_raw)

# Los mismos descartes que en el fit
COLS_DROP_INF = ["experiment_ID", "Timestamp"]
if "phase" in X_full.columns:
    COLS_DROP_INF.append("phase")

X_num = X_full.drop(columns=COLS_DROP_INF)
y_pred = rf.predict(X_num)

# (opcional) guardar
pd.Series(y_pred, name="cluster").to_pickle(
    f"{MODEL_DIR}/y_pred_full.pkl")

