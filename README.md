# SmartMonitor – Pipeline de Machine Learning + Dashboard Streamlit  
Monitorización de sensores industriales · clustering · detección de anomalías

![Banner](assets/banner.png)

---

## 1. ¿Qué hay en este repo?

| Carpeta               | Para qué sirve                                                                                                  |
| :-------------------- | :--------------------------------------------------------------------------------------------------------------- |
| **app/**              | Front-end en Streamlit (`app.py`) con paneles de KPIs y visualización de clústeres.                              |
| **src/smartmonitor/** | Paquete Python reutilizable → pasos de preprocesado y script de entrenamiento (`Complete_Project.py`).          |
| **data/**             | Muestra reducida del CSV original (`date_production.zip`).                                                       |
| **assets/**           | Imágenes que usa el dashboard.                                                                                   |
| **models/**           | (en `.gitignore`) Pickles generados al entrenar: `preprocessing_pipeline.pkl`, `kmeans.pkl`, `gmm.pkl`, etc.     |

---

## 2. Arranque exprés (⚡ 30 s)

```bash
git clone https://github.com/furtu-71/SmartMonitor_ML.git
cd SmartMonitor_ML

# Python ≥ 3.10 recomendado
python -m venv venv
# .\venv\Scripts\activate        # Windows
source venv/bin/activate         # Linux / macOS

pip install -r requirements.txt

# lanza el dashboard
python -m streamlit run app/app.py
```

Se abrirá en **http://localhost:8501** y podrás explorar tendencias, clústeres y anomalías.

---

## 3. ¿Cómo funciona el pipeline?

| Paso                    | Archivo / función                            | Descripción breve |
|-------------------------|---------------------------------------------|-------------------|
| Pre-procesado           | `src/smartmonitor/preprocessing_steps.py`   | Validación de columnas, categóricas, ingeniería temporal, duplicados, Modified Z-score, imputación mediana, escalado robusto. |
| Muestreo 10 % estrat.   | `sample_experiment_phases()`                | Equilibrio temporal (20-30-50 %). |
| Clustering base         | `Paso 5.3` de `Complete_Project.py`         | Entrena K-Means, OPTICS y GMM sobre 25 columnas *_modified_z_. |
| Ensamblado de clústeres | `Paso 5.4`                                  | Matriz de co-asociación → Agglomerative (average linkage). |
| Modelo auxiliar         | `Paso 5.5`                                  | Random Forest predice el clúster ensemble en tiempo real. |

---

## 4. Reentrenar todo desde cero

```bash
python src/smartmonitor/Complete_Project.py
```

Genera de nuevo el pipeline y los modelos en **models/**  
(tiempo ≈ 12-15 min en un portátil reciente).

---

## 5. Solución de problemas

| Síntoma / mensaje                             | Arreglo rápido |
|----------------------------------------------|----------------|
| `ModuleNotFoundError: preprocessing_steps`   | Ejecuta la app desde la raíz (`streamlit run app/app.py`). |
| Falta de RAM en la co-asociación             | Baja la fracción de muestra al 5 % o usa ≥ 16 GB. |
| Dashboard arranca lento                      | Usa `@st.cache_resource` o comenta modelos que no necesites. |

---

## 6. Contribuir

1. Haz *fork*, crea tu rama (`git checkout -b mejora-x`).  
2. Formatea con `black .` y asegúrate de que los tests (próximamente) pasan.  
3. Abre un Pull Request describiendo la mejora y las pruebas.

---

## 7. Licencia y autoría

Proyecto bajo licencia MIT.  
Creado con 🛠️ y ☕ por **Fernando González**  
([@furtu-71](https://github.com/furtu-71)).

> Los datos industriales pueden ser caóticos; el código, no.
