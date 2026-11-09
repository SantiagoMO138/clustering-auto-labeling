# Etiquetado automático mediante clustering

Experimento reproducible para evaluar la generación de etiquetas en el dataset
Breast Cancer Wisconsin (Diagnostic) con K-Means, clustering jerárquico y
Gaussian Mixture Models (GMM).

## Ejecución

```powershell
python -m pip install -r requirements.txt
python run_experiment.py
pytest -q
```

La ejecución descarga el dataset incluido en scikit-learn, corre 10
repeticiones estratificadas 80/20 y genera tablas, figuras y el informe en
`outputs/` y `reports/informe_resultados.md`.

## Convenciones

- La etiqueta `0` de scikit-learn corresponde a **maligno** y se usa como
  clase positiva para sensibilidad, F1 y ROC-AUC.
- Las etiquetas reales solo se usan para evaluar y para simular el pequeño
  subconjunto de apoyo experto (5 % y 10 %). Nunca se entregan al algoritmo
  de clustering.
- Los tres esquemas supervisados son: pseudoetiquetas generadas por
  clustering, etiquetas reales completas y etiquetas reales del subconjunto
  de apoyo.
