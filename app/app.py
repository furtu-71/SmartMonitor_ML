import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

# app.py ───────────────────────────────────────────────────────────
from smartmonitor import preprocessing_steps          # registra las 6 funciones
from pathlib import Path
from PIL import Image
import joblib, pandas as pd, streamlit as st, plotly.express as px
from sklearn.decomposition import PCA

# 🟡── Config global ───────────────────────────────────────────────
ICON_PATH = Path("assets/icono smartmonitor.png")
icon_img  = Image.open(ICON_PATH)

st.set_page_config(
    page_title="SmartMonitor – Mantenimiento Predictivo",
    page_icon=icon_img,
    layout="wide",
)
st.title("🔧 SmartMonitor – Mantenimiento Predictivo")

# 🟡── Rutas absolutas ─────────────────────────────────────────────
DATA_PATH   = Path("data/date_production.zip")          # CSV completo
PICKLE_X    = Path("models/X_sample_trans.pkl")         # 22 838 × 25
PICKLE_Y    = Path("models/y_cluster_sample.pkl")       # 22 838 labels

# 🟡── Carga de datos base ─────────────────────────────────────────
@st.cache_data
def load_full_csv() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, compression="zip")

df = load_full_csv()
st.success(f"Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
st.dataframe(df.head())

# 🟡── Botón para ver el scatter de referencia ─────────────────────
st.subheader("Visualización de clústeres (muestra 10 %)")

def load_reference_sample():
    if PICKLE_X.exists() and PICKLE_Y.exists():
        X_s = pd.read_pickle(PICKLE_X)
        y_s = pd.read_pickle(PICKLE_Y).astype(str).values
        return X_s, y_s
    else:
        return None, None

if st.button("Mostrar scatter de referencia"):
    X_s, y_s = load_reference_sample()

    if X_s is None:
        st.error(
            "❌ No encuentro los archivos:\n"
            f"→ {PICKLE_X.name}\n"
            f"→ {PICKLE_Y.name}\n"
            "Cópialos a la carpeta *models/* y vuelve a recargar la página."
        )
        st.stop()

    # 🟡 PCA 2 D idéntica a la del Colab (misma semilla) ──────────
    sensor_cols = [c for c in X_s.columns if "_modified_z" in c]
    coords = PCA(n_components=2, random_state=42).fit_transform(X_s[sensor_cols])

    plot_df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "cluster": y_s
    })

    fig = px.scatter(
        plot_df, x="PC1", y="PC2",
        color="cluster",
        opacity=0.7,
        height=650,
        color_discrete_sequence=px.colors.qualitative.Vivid
    ).update_layout(
        title="Distribución de clústeres – proyección PCA 2D",
        legend_title_text="Clúster"
    )

    st.plotly_chart(fig, use_container_width=True)
