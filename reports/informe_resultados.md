# Informe de resultados: etiquetado automático mediante clustering

## Objetivo y diseño

Se evaluó la generación de pseudoetiquetas con K-Means, clustering jerárquico Ward y Gaussian Mixture Models (GMM) en Breast Cancer Wisconsin (Diagnostic). El dataset contiene 569 observaciones y 30 variables morfológicas. La clase `0` (maligno) se trató como positiva.

Se realizaron 10 particiones estratificadas entrenamiento/prueba de 80/20. En cada entrenamiento se estandarizaron las variables sin usar datos de prueba y se simuló apoyo experto estratificado del 5 % y 10 %. Las etiquetas de diagnóstico no se entregaron a los algoritmos de clustering.

## Criterio de viabilidad

Un enfoque se considera viable si la exactitud media de etiquetado con la correspondencia obtenida del apoyo experto es al menos 90 %, y las brechas absolutas de exactitud y sensibilidad maligna frente al clasificador entrenado con etiquetas completas no exceden 5 puntos porcentuales.

Se identificaron **2** combinaciones viables de 12 evaluadas.

## Mejor resultado por desempeño promedio

La combinación líder fue **GMM**, apoyo del **5 %** y **logistic_regression**, con exactitud media de **0.929 +/- 0.017** y sensibilidad maligna de **0.905 +/- 0.021**.

## Resultados detallados

Los resultados crudos, resúmenes, ranking y evaluación de viabilidad se encuentran en `outputs/`. Las figuras principales son `pca_clusters.png`, `generated_accuracy.png` y `performance_gaps.png`.

## Limitaciones

Este resultado corresponde a un dataset clínico curado y de tamaño reducido; no sustituye diagnóstico clínico ni justifica extrapolación a otros hospitales, poblaciones o problemas de clasificación.

Variables utilizadas: mean radius, mean texture, mean perimeter, mean area, mean smoothness, mean compactness, mean concavity, mean concave points, mean symmetry, mean fractal dimension, radius error, texture error, perimeter error, area error, smoothness error, compactness error, concavity error, concave points error, symmetry error, fractal dimension error, worst radius, worst texture, worst perimeter, worst area, worst smoothness, worst compactness, worst concavity, worst concave points, worst symmetry, worst fractal dimension.
