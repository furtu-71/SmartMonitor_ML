## Cómo reproducir el entorno

```bash
python -m venv venv                # crea el entorno
source venv/bin/activate           # Linux / macOS
# .\venv\Scripts\activate          # ← Windows
pip install -r requirements.txt
```

## Cargar el dataset comprimido

```python
import pandas as pd

df = pd.read_csv(
    "data/date_production.zip",
    compression="zip"
)
```
