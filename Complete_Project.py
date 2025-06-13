import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from joblib import load
from sklearn.preprocessing import RobustScaler

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, OPTICS, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold

from joblib import dump
from google.colab import files








#------------------------------------------------------------------------------#
# CARGAR LOS DATOS                                                             #
#------------------------------------------------------------------------------#

# Definir la ruta al archivo CSV (ajusta según la ubicación real)
csv_file_path = "/content/drive/MyDrive/Colab Notebooks/MASTER DATA SCIENCE/Proyecto final/date_production.csv"

# Cargar el DataFrame preprocesado desde el archivo CSV usando pd.read_csv
df_production = pd.read_csv(csv_file_path)

#------------------------------------------------------------------------------#
# VERIFICACIÓN DE LA EXISTENCIAS DE LAS COLUMNAS NECESARIAS                    #
#------------------------------------------------------------------------------#

# Función de validación existente
def validate_dataframe_columns(df, required_columns):
    """
    Valida que el DataFrame 'df' contenga todas las columnas especificadas en
    'required_columns'. Si falta alguna de estas columnas, lanza un ValueError.
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
      raise ValueError(f"El DataFrame no contiene las columnas requeridas: {missing_cols}")
    return df

# Definir las columnas requeridas
required_columns = [
    'experiment_ID', 'Timestamp', 'L_1', 'L_2', 'A_1', 'A_2',
    'B_1', 'B_2', 'C_1', 'C_2', 'A_3', 'A_4', 'B_3', 'B_4',
    'C_3', 'C_4', 'L_3', 'L_4', 'L_5', 'L_6', 'L_7', 'L_8',
    'L_9', 'L_10', 'A_5', 'B_5', 'C_5'
]

# Envolver la función de validación en un FunctionTransformer
# Como la función requiere dos argumentos (df y required_columns), usamos una lambda o pasamos los argumentos mediante kw_args.
def validate_columns(df):
    return validate_dataframe_columns(df, required_columns)

validation_transformer = FunctionTransformer(validate_columns, validate=False)


# Crear el pipeline para el paso de validación
pipeline_validation = Pipeline(steps=[
    ('validate_columns', validation_transformer)
])


# Aplicar el pipeline de validación
df_validated = pipeline_validation.fit_transform(df_production)
print("El DataFrame contiene todas las columnas necesarias.")

#------------------------------------------------------------------------------#
# DATOS FALTANTES                                                              #
#------------------------------------------------------------------------------#

'''
SimpleImputer: Este transformador se encarga de detectar los valores faltantes
(por ejemplo, NaN) en tus columnas numéricas y los reemplaza con la mediana
calculada para cada columna. La mediana es una medida robusta ante valores
extremos.
'''

# Supongamos que df_validated es el DataFrame validado que contiene la columna 'experiment_ID'
# Creamos un subconjunto sin 'experiment_ID' dado que es un string
df_numeric = df_validated.drop(columns=['experiment_ID'])

# Definimos el pipeline para la imputación (aplicable sólo a columnas numéricas)
pipeline_missing_data_global = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Aplicamos el pipeline y reconstruimos el DataFrame
df_imputed_array = pipeline_missing_data_global.fit_transform(df_numeric)
'''Para conservar la estructura del DataFrame (etiquetas de columnas), se convierte:'''
df_imputed_numeric = pd.DataFrame(df_imputed_array, columns=df_numeric.columns)

# Añadimos nuevamente la columna 'experiment_ID'
df_imputed = df_imputed_numeric.copy()
df_imputed['experiment_ID'] = df_validated['experiment_ID']

print("DataFrame después de la imputación:")
print(df_imputed.head())


#------------------------------------------------------------------------------#
# CONVERSIÓN A CATEGORICO                                                      #
#------------------------------------------------------------------------------#

'''
Definimos una función que convierta 'experiment_ID' a categórico.
'''
def convert_experiment_id_to_category(df):
    # Realizamos una copia para evitar modificar el DataFrame original
    df = df.copy()
    df['experiment_ID'] = df['experiment_ID'].astype('category')
    return df

# Envolvemos la función con FunctionTransformer.
experiment_id_transformer = FunctionTransformer(convert_experiment_id_to_category, validate=False)

# Creamos un pipeline que solo realice esta transformación
pipeline_convert_experiment_ID = Pipeline(steps=[
    ('to_category', experiment_id_transformer)
])

# Aplicamos el pipeline:
df_transformed = pipeline_convert_experiment_ID.fit_transform(df_imputed)

# Verificamos que 'experiment_ID' es ahora de tipo categoría
print(df_transformed['experiment_ID'].dtype)


#------------------------------------------------------------------------------#
# Convertir Timestamp a datetime y extraer Features, "resetenado" el tiempo    #
# para cada experimento y mateniendo el caracter de "tiempo relativo"          #
#------------------------------------------------------------------------------#

def process_timestamp_features(df):
    """
    Para cada experimento (identificado en 'experiment_ID'):
      1. Convierten la columna 'Timestamp' a datetime.
      2. Filtran los registros de ese experimento y eliminan aquellos con Timestamp inválido (NaT).
      3. Calculan el tiempo transcurrido relativo a partir del mínimo de ese experimento.
      4. Convierte ese tiempo relativo (un timedelta) a segundos y luego a un objeto datetime
         (usando '1970-01-01' como origen) para extraer features temporales.
      5. Extrae features: year, month, day, hour, minute, second, day_of_week y conserva 'elapsed_seconds'.
      6. Finalmente, concatena los resultados de todos los experimentos.
    """
    # Hacemos una copia para no modificar el DataFrame original
    df = df.copy()

    # 1. Convertir 'Timestamp' a datetime. Ajusta 'unit' según la forma en que estén codificados (p.ej., 's' o 'ns')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ns', errors='coerce')

    # Conservar los valores únicos de experiment_ID
    experiment_ids = df['experiment_ID'].unique()

    processed_frames = []

    # Iterar sobre cada experimento y procesar de forma individual
    for exp_id in experiment_ids:
        # Filtrar el DataFrame para el experimento actual
        df_exp = df[df['experiment_ID'] == exp_id].copy()

        # Eliminar registros con Timestamp inválido
        df_exp = df_exp.dropna(subset=['Timestamp'])

        if df_exp.empty:
            continue  # si no hay registros válidos para este experimento, se salta

        # 2. Calcular la diferencia respecto al mínimo Timestamp de este experimento
        min_time = df_exp['Timestamp'].min()
        df_exp['elapsed'] = df_exp['Timestamp'] - min_time

        # 3. Convertir el timedelta a segundos para tener valores numéricos
        df_exp['elapsed_seconds'] = df_exp['elapsed'].dt.total_seconds()

        # 4. Convertir esos segundos a datetime (la parte de fecha es irrelevante, nos interesa extraer componentes)
        df_exp['Datetime'] = pd.to_datetime(df_exp['elapsed_seconds'], unit='ns', origin='1970-01-01')

        # 5. Extraer las features temporales y convertirlas a enteros
        df_exp['year'] = df_exp['Datetime'].dt.year.astype('int32')
        df_exp['month'] = df_exp['Datetime'].dt.month.astype('int32')
        df_exp['day'] = df_exp['Datetime'].dt.day.astype('int32')
        df_exp['hour'] = df_exp['Datetime'].dt.hour.astype('int32')
        df_exp['minute'] = df_exp['Datetime'].dt.minute.astype('int32')
        df_exp['second'] = df_exp['Datetime'].dt.second.astype('int32')
        df_exp['day_of_week'] = df_exp['Datetime'].dt.dayofweek.astype('int32')

        # 6. Eliminar columnas auxiliares que ya no se necesitan
        df_exp.drop(columns=['elapsed', 'Datetime'], inplace=True)

        processed_frames.append(df_exp)

    # Concatenar los DataFrames procesados para cada experimento
    df_processed = pd.concat(processed_frames, ignore_index=True)

    return df_processed

# Envolver la función con FunctionTransformer
timestamp_transformer = FunctionTransformer(process_timestamp_features, validate=False)

# Crear el pipeline para procesar el timestamp y extraer las features
pipeline_timestamp_features = Pipeline(steps=[
    ('timestamp_processing', timestamp_transformer)
])

# Aplicar el pipeline sobre el DataFrame que ya tiene la conversión a categórico (df_transformed)
df_timestamp_processed = pipeline_timestamp_features.fit_transform(df_transformed)

# Mostrar las primeras filas para verificar el resultado
print("DataFrame tras el procesamiento del Timestamp y extracción de features:")
print(df_timestamp_processed.head())


#------------------------------------------------------------------------------#
# Detectar y eliminar registros redundantes                                    #
#------------------------------------------------------------------------------#

def remove_duplicates(df):
    """
    Elimina los registros duplicados del DataFrame.
    Se crea una copia para no modificar el DataFrame original.
    """
    df = df.copy()
    # Elimina duplicados considerando todas las columnas.
    df_clean = df.drop_duplicates()
    return df_clean

# Envolver la función en un FunctionTransformer.
duplicates_transformer = FunctionTransformer(remove_duplicates, validate=False)

# Crear el pipeline para eliminar registros redundantes.
pipeline_duplicates_removal = Pipeline(steps=[
    ('remove_duplicates', duplicates_transformer)
])

df_final = pipeline_duplicates_removal.fit_transform(df_timestamp_processed)
print("Shape final del DataFrame después de eliminar duplicados:", df_final.shape)



#------------------------------------------------------------------------------#
# Función que calcula el modified z-score para una serie                       #
#------------------------------------------------------------------------------#
def modified_z_score(series, threshold=3.5):
    """
    Calcula el Modified Z-score para cada dato de la serie usando la Mediana Absoluta de las Desviaciones (MAD).

    Parámetros:
      series (pd.Series o np.array): La serie de datos a analizar.
      threshold (float): Valor umbral para etiquetar como outlier. Por defecto se utiliza 3.5.

    Retorna:
      outliers (np.array): Array booleano donde True indica que el valor es un outlier.
      z_scores (np.array): Array con los Modified Z-scores de cada dato.
    """
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    if mad == 0:  # Evitar división por cero
        mad = 1e-6
    z_scores = 0.6745 * (series - median) / mad
    outliers = np.abs(z_scores) > threshold
    return outliers, z_scores

# Función que, dado un DataFrame y una lista de columnas,
# agrega dos columnas nuevas para cada parámetro: una con el modified z-score y otra indicadora de outlier.
def add_outlier_columns(df, columns, threshold=3.5):
    df = df.copy()
    for col in columns:
        # Calcular outliers y z-scores para la columna actual
        outliers, z_scores = modified_z_score(df[col], threshold=threshold)
        # Crear dos nuevas columnas
        df[col + "_modified_z"] = z_scores
        df[col + "_outlier"] = outliers
    return df

# Supongamos que V_Monitorizados es la lista de columnas con los parámetros de sensores, por ejemplo:
V_Monitorizados = ['L_1', 'L_2', 'A_1', 'A_2', 'B_1', 'B_2', 'C_1', 'C_2', 'A_3', 'A_4',
                   'B_3', 'B_4', 'C_3', 'C_4', 'L_3', 'L_4', 'L_5', 'L_6', 'L_7', 'L_8',
                   'L_9', 'L_10', 'A_5', 'B_5', 'C_5']

# Envolver la función en un FunctionTransformer, pasando los argumentos necesarios a través de kw_args.
outlier_transformer = FunctionTransformer(
    add_outlier_columns,
    kw_args={'columns': V_Monitorizados, 'threshold': 3.5},
    validate=False
)

# Crear el pipeline para añadir las columnas de outliers
pipeline_outliers = Pipeline(steps=[
    ('add_outlier_features', outlier_transformer)
])

# Aplicar el pipeline sobre el DataFrame final (df_final)
df_final_with_outliers = pipeline_outliers.fit_transform(df_final)

# Mostrar algunas filas para comprobar las nuevas columnas
print("DataFrame final con columnas de outliers añadidas:")
print(df_final_with_outliers.head())


# ===============================================================
# Función de muestreo estratificado por fases (definida de forma externa)
# ===============================================================
def sample_experiment_phases(df, fraction=0.1, time_col="Timestamp", phases=3, weights=None, random_state=42):
    """
    Toma una muestra estratificada del DataFrame de un experimento,
    preservando la variabilidad temporal al dividirlo en 'phases'.

    Parámetros:
      - df: DataFrame del experimento.
      - fraction: Fracción de registros a muestrear (ej. 0.1 para el 10%).
      - time_col: Nombre de la columna temporal.
      - phases: Número de fases en que se divide el experimento (p.ej. 3: inicio, medio y fin).
      - weights: Lista de pesos para cada fase (debe sumar 1). Si es None, se distribuye uniformemente.
      - random_state: Semilla para reproducibilidad.

    Retorna:
      - DataFrame con la muestra estratificada.
    """

    # Si no se proporcionan pesos, se asignan iguales a cada fase.
    if weights is None:
        weights = [1.0 / phases] * phases

    # Ordenar el DataFrame por la columna temporal
    df_sorted = df.sort_values(by=time_col).copy()
    n_total = len(df_sorted)
    total_samples = int(n_total * fraction)

    # Calcular los cuantiles para dividir el experimento en "phases"
    quantiles = [i / phases for i in range(phases + 1)]
    boundaries = df_sorted[time_col].quantile(quantiles).values

    # Asignar a cada registro una etiqueta de fase usando pd.cut
    df_sorted["phase"] = pd.cut(df_sorted[time_col],
                                bins=boundaries,
                                labels=False,
                                include_lowest=True)

    sampled_df = pd.DataFrame()
    # Muestrear cada fase con el peso especificado
    for phase in range(phases):
        phase_data = df_sorted[df_sorted["phase"] == phase]
        n_phase = int(round(total_samples * weights[phase]))
        n_phase = min(n_phase, len(phase_data))  # Evitar solicitar más muestras de las disponibles
        sampled_phase = phase_data.sample(n=n_phase, random_state=random_state)
        sampled_df = pd.concat([sampled_df, sampled_phase])

    return sampled_df

# ===============================================================
# Código final: Aplicar el muestreo estratificado al dataset con outliers
# ===============================================================
import pandas as pd

# Se trabaja sobre el DataFrame df_final_with_outliers, que contiene, entre otras columnas,
# "experiment_ID" y "Timestamp". Se agrupa por "experiment_ID" para procesar cada experimento.
samples_list = []
for exp_id, df_exp in df_final_with_outliers.groupby("experiment_ID"):
    sample_exp = sample_experiment_phases(
        df_exp,
        fraction=0.1,
        time_col="Timestamp",
        phases=3,
        weights=[0.2, 0.3, 0.5],
        random_state=42
    )
    samples_list.append(sample_exp)

# Combinar las muestras de todos los experimentos en un único DataFrame y renombrarlo a df_final_with_outliers_strat
df_final_with_outliers_strat = pd.concat(samples_list).reset_index(drop=True)

# (Opcional) Eliminar la columna auxiliar 'phase' ya que solo se usó para el muestreo
df_final_with_outliers_strat = df_final_with_outliers_strat.drop(columns=["phase"])

# Visualizar las dimensiones del dataset original y de la muestra estratificada
print("Dimensión original:", df_final_with_outliers.shape)
print("Dimensión de la muestra estratificada (10%):", df_final_with_outliers_strat.shape)


# -----------------------------------------------------------------------------
# ESCALADO CON ROBUSTSCALER
# -----------------------------------------------------------------------------

# Definir la lista de columnas a excluir (no se modificarán)
excluded_cols = [
    'Timestamp',
    'experiment_ID',
    'L_1_outlier', 'L_2_outlier', 'L_3_outlier', 'L_4_outlier', 'L_5_outlier',
    'L_6_outlier', 'L_7_outlier', 'L_8_outlier', 'L_9_outlier', 'L_10_outlier',
    'A_1_outlier', 'A_2_outlier', 'A_3_outlier', 'A_4_outlier', 'A_5_outlier',
    'B_1_outlier', 'B_2_outlier', 'B_3_outlier', 'B_4_outlier', 'B_5_outlier',
    'C_1_outlier', 'C_2_outlier', 'C_3_outlier', 'C_4_outlier', 'C_5_outlier'
]

def robust_scale_features(df, excluded_cols):
    """
    Escala las columnas numéricas del DataFrame (aquellas que no están en excluded_cols)
    utilizando RobustScaler, y une el resultado con las columnas excluidas.
    """
    df = df.copy()
    # Seleccionar las columnas a escalar (aquellas que no se encuentran en excluded_cols)
    features_to_scale = [col for col in df.columns if col not in excluded_cols]

    # Aplicar RobustScaler sobre las columnas seleccionadas
    scaler = RobustScaler()
    scaled_array = scaler.fit_transform(df[features_to_scale])

    # Convertir el array escalado en un DataFrame conservando índices y nombres originales
    df_scaled = pd.DataFrame(scaled_array, columns=features_to_scale, index=df.index)

    # Concatenar con las columnas excluidas sin modificar
    df_final_scaled = pd.concat([df_scaled, df[excluded_cols]], axis=1)
    return df_final_scaled

# Envolver la función con FunctionTransformer, pasando 'excluded_cols' como argumento
scaling_transformer = FunctionTransformer(robust_scale_features, kw_args={'excluded_cols': excluded_cols}, validate=False)

# Crear el pipeline que aplica el escalado robusto
pipeline_scaling = Pipeline(steps=[
    ('scaling', scaling_transformer)
])

# Se utiliza el DataFrame 'df_final_with_outliers_strat' (obtenido tras el muestreo estratificado)
# como entrada para la estandarización robusta
df_sample_final_scaled = pipeline_scaling.fit_transform(df_final_with_outliers_strat)

# Visualizar las primeras filas para verificar la aplicación del RobustScaler
print("Primeras filas del DataFrame escalado (RobustScaler):")
print(df_sample_final_scaled.head())

# Integrar el pipeline de escalado junto con los demás pasos de preprocesamiento en un pipeline final
pipeline_final = Pipeline(steps=[
    ('validacion', pipeline_validation),
    ('imputacion', pipeline_missing_data_global),
    ('conversion_categoria', pipeline_convert_experiment_ID),
    ('procesamiento_timestamp', pipeline_timestamp_features),
    ('eliminar_duplicados', pipeline_duplicates_removal),
    ('agregar_outliers', pipeline_outliers),
    ('escalado', pipeline_scaling)
])

# Guardar el pipeline final en un archivo PKL
dump(pipeline_final, 'preprocessing_pipeline.pkl')

# Descargar el archivo PKL al dispositivo local (compatible con Google Colab)
files.download('preprocessing_pipeline.pkl')

# ===============================================================
# 3. ENSAMBLE CLUSTERING
# ===============================================================
# Se utiliza el DataFrame escalado proveniente del pipeline robusto: df_sample_final_scaled
# 3.1. Seleccionar las características de sensores (se asume que aquellas transformadas tienen '_modified_z' en el nombre)
sensor_features = [col for col in df_sample_final_scaled.columns if '_modified_z' in col]
X = df_sample_final_scaled[sensor_features].values
print("Dimensión de X:", X.shape)

# 3.2. K-Means++ clustering
k = 3  # Número de clusters deseado
kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
labels_kmeans = kmeans.fit_predict(X)
print("K-Means++ - Número de clusters:", np.unique(labels_kmeans).size)

# 3.3. OPTICS: Grid Search mediante tqdm
def tune_optics(X, min_samples_range, xi_range):
    best_score = -1
    best_params = None
    best_labels = None
    for min_samples in tqdm(min_samples_range, desc="Min Samples Loop"):
        for xi in tqdm(xi_range, leave=False, desc="Xi Loop"):
            optics = OPTICS(min_samples=min_samples, xi=xi, cluster_method='xi')
            labels = optics.fit_predict(X)
            # Excluir el ruido (etiqueta -1) y verificar que se formen al menos 2 clusters
            unique_clusters = np.unique(labels[labels != -1])
            if len(unique_clusters) < 2:
                continue
            try:
                score = silhouette_score(X, labels)
            except Exception:
                score = -1
            if score > best_score:
                best_score = score
                best_params = (min_samples, xi)
                best_labels = labels
    return best_params, best_score, best_labels

min_samples_range = range(3, 10)          # por ejemplo, de 3 a 9
xi_range = np.linspace(0.01, 0.1, 10)       # 10 valores entre 0.01 y 0.1
best_params_optics, best_score_optics, best_labels_optics = tune_optics(X, min_samples_range, xi_range)
print("OPTICS tuning - Mejores parámetros encontrados:")
print("min_samples =", best_params_optics[0], "xi =", best_params_optics[1])
print("OPTICS - Mejor Silhouette Score:", best_score_optics)
print("OPTICS - Número de clusters (excluyendo ruido):", np.unique(best_labels_optics[best_labels_optics != -1]).size)
labels_optics = best_labels_optics

# 3.4. Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=k, random_state=42)
labels_gmm = gmm.fit_predict(X)
print("GMM - Número de clusters:", np.unique(labels_gmm).size)

# 3.5. Guardar resultados intermedios para evitar repetir estas operaciones

with open('clustering_intermediate_results.pkl', 'wb') as f:
    pickle.dump({'X': X,
                 'labels_kmeans': labels_kmeans,
                 'labels_optics': labels_optics,
                 'labels_gmm': labels_gmm}, f)

# 3.6. Ensemble clustering mediante matriz de coasociación
def compute_coassociation_matrix(labels_list):
    n_samples = len(labels_list[0])
    coassoc = np.zeros((n_samples, n_samples))
    for labels in labels_list:
        matrix = (labels[:, None] == labels[None, :]).astype(int)
        coassoc += matrix
    coassoc /= len(labels_list)
    return coassoc

labels_list = [labels_kmeans, labels_optics, labels_gmm]
coassoc_matrix = compute_coassociation_matrix(labels_list)
distance_matrix = 1 - coassoc_matrix

ensemble_cluster = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
ensemble_labels = ensemble_cluster.fit_predict(distance_matrix)
print("Ensemble Clustering - Número de clusters:", np.unique(ensemble_labels).size)

# 3.7. Evaluación del Ensemble
silhouette_val = silhouette_score(X, ensemble_labels)
calinski_val = calinski_harabasz_score(X, ensemble_labels)
davies_val = davies_bouldin_score(X, ensemble_labels)
print("\nEvaluación del Ensemble:")
print("Silhouette Score:", silhouette_val)
print("Calinski-Harabasz Score:", calinski_val)
print("Davies-Bouldin Score:", davies_val)

# Visualización de clusters con PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(10,6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=ensemble_labels, palette="viridis", s=50)
plt.title("Visualización de Clusters (Ensemble) con PCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(title="Cluster")
plt.show()

# ===============================================================
# 4. MODELO PREDICTIVO AUXILIAR CON VALIDACIÓN CRUZADA AGRUPADA
# ===============================================================


# Se trabaja con el DataFrame escalado obtenido (df_sample_final_scaled)
# y se integra la información de clustering contenida en ensemble_labels.
df_aux = df_sample_final_scaled.copy()
df_aux['ensemble_cluster'] = ensemble_labels

# Seleccionar las características relevantes (columnas que contienen '_modified_z')
sensor_features = [col for col in df_aux.columns if '_modified_z' in col]

# Definir X (características) e y (etiqueta de clúster)
X_aux = df_aux[sensor_features]
y_aux = df_aux['ensemble_cluster']

# Definir los grupos basados en la columna 'experiment_ID' para asegurar la trazabilidad de cada experimento
groups = df_aux['experiment_ID']

# Configurar validación cruzada agrupada con 5 pliegues
gkf = GroupKFold(n_splits=5)
fold_accuracies = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_aux, y_aux, groups), start=1):
    # Dividir los datos en conjuntos de entrenamiento y prueba para el pliegue actual
    X_train, X_test = X_aux.iloc[train_idx], X_aux.iloc[test_idx]
    y_train, y_test = y_aux.iloc[train_idx], y_aux.iloc[test_idx]

    # Entrenar el modelo predictivo auxiliar (RandomForestClassifier)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Evaluar el desempeño del modelo en el pliegue actual
    y_pred = rf_classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    fold_accuracies.append(acc)

    print(f"\nFold {fold}:")
    print("Accuracy:", acc)
    print("Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))

# Mostrar la precisión promedio a lo largo de los pliegues
print("\nAccuracy promedio en validación cruzada agrupada:", np.mean(fold_accuracies))


# ===============================================================
# 5. GUARDAR (SERIALIZAR) EL MODELO PREDICTIVO AUXILIAR EN UN ARCHIVO PKL
# ===============================================================
with open('predictive_aux_model.pkl', 'wb') as f:
    pickle.dump(rf_classifier, f)
print("Archivo PKL generado: predictive_aux_model.pkl")

# (Opcional) Descargar el archivo PKL a tu equipo local (compatible con Google Colab)

files.download('predictive_aux_model.pkl')




# ------------------------------------------------------------------------------
# VALIDACIÓN CRUZADA AGRUPADA SOBRE EL ENSAMBLE CLUSTERING
# ------------------------------------------------------------------------------
# 'df_sample_final_scaled' es el DataFrame de la muestra escalado previamente,
# que ya incluye la columna 'experiment_ID' (obtenida al seleccionar la muestra del 10%
# del df_monitoring conservando todas las características).
# Se asume que ya hemos calculado 'ensemble_labels' (las etiquetas obtenidas del ensamble).

from sklearn.model_selection import GroupKFold
from sklearn.metrics import silhouette_score
import numpy as np

# Definir los grupos a partir de la columna 'experiment_ID'
groups = df_sample_final_scaled['experiment_ID']

# Seleccionar las características de sensor utilizadas para el clustering
sensor_features = [col for col in df_sample_final_scaled.columns if '_modified_z' in col]
X_cv = df_sample_final_scaled[sensor_features].values
print("Dimensión de X_cv:", X_cv.shape)

# Configurar GroupKFold con 5 pliegues
gkf = GroupKFold(n_splits=5)

# Lista para almacenar las métricas de Silhouette en cada fold
silhouette_scores_cv = []

# Iterar sobre cada partición generada por GroupKFold
# Se evaluarán las etiquetas del ensamble ya calculadas
for fold, (train_idx, test_idx) in enumerate(gkf.split(X_cv, groups=groups), start=1):
    # Subconjunto de datos correspondiente al fold de validación
    X_fold = X_cv[test_idx]
    labels_fold = ensemble_labels[test_idx]

    # Verificar que en el fold existan al menos 2 clusters para poder calcular el Silhouette Score
    if len(np.unique(labels_fold)) < 2:
        print(f"Fold {fold}: menos de 2 clusters, se omite la evaluación.")
        continue

    sil_score = silhouette_score(X_fold, labels_fold)
    silhouette_scores_cv.append(sil_score)
    print(f"Fold {fold}: Silhouette Score = {sil_score:.3f}")

# Calcular el promedio de Silhouette Score a lo largo de los folds
if silhouette_scores_cv:
    print("\nAverage Silhouette Score en validación cruzada agrupada:", np.mean(silhouette_scores_cv))
else:
    print("No se pudo evaluar el Silhouette Score debido a la falta de variabilidad en algunos folds.")

