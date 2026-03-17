# Telco Customer Churn - Clasificación y Dashboard

Este repositorio cumple el reto de aprendizaje supervisado usando el dataset **Telco Customer Churn**.

## Estructura
```text
/data
  - telco_batch_template.csv
/notebooks
  - telco_customer_churn_clasificacion.ipynb
/app
  - main.py
requirements.txt
README.md
```

## Qué incluye
- Notebook comentado con:
  - EDA
  - preprocesamiento
  - ingeniería de características
  - entrenamiento y ajuste de hiperparámetros
  - validación robusta con cross-validation
  - métricas en test set
- Dashboard en Streamlit con:
  - predicción individual por formulario
  - predicción por lote con archivo CSV
  - gráfico de feature importance
  - visualización de métricas del modelo

## Cómo ejecutarlo localmente
```bash
pip install -r requirements.txt
streamlit run app/main.py
```

## Despliegue sugerido
Sube el repositorio a GitHub y luego conéctalo a **Streamlit Cloud**.  
Archivo principal: `app/main.py`

## Evidencia de pruebas de usuario
Puedes documentar en GitHub o en tu entrega:
1. Predicción individual exitosa desde el formulario.
2. Predicción por lote exitosa con `data/telco_batch_template.csv`.
3. Visualización del gráfico de feature importance.
4. Capturas de pantalla del dashboard funcionando.

## Notas
- El modelo se entrena automáticamente la primera vez que se ejecuta la app.
- Luego queda cacheado en `app/artifacts/`.
- El dataset se carga desde una copia pública del CSV de IBM.
