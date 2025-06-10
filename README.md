# UR5Inv

Proyecto desarrollado como parte de la materia de Consultoría de la Maestría en Cómputo Estadístico del CIMAT, en colaboración con CIDESI.  
El objetivo es modelar la **cinemática inversa del robot UR5** mediante técnicas de aprendizaje supervisado, evaluando enfoques deterministas (FNN, LSTM, Random Forest) y generativos (CGAN, INN).

## Estructura del Repositorio

- `Data/`: enlace externo a los **conjuntos de datos** utilizados (simulados y reales).  
  [Acceder a los datos](ENLACE_DRIVE_DATOS)

- `Modelos/`: enlace externo a los **modelos entrenados** y guardados en Drive.  
  [Acceder a los modelos](ENLACE_DRIVE_MODELOS)

- `UR5Inv/`: contiene los **módulos y funciones** auxiliares en Python desarrolladas para procesamiento, entrenamiento e inferencia.

## Notebooks Principales

- `Datasets.ipynb`: carga, limpieza y exploración de los conjuntos de datos.
- `MLP.ipynb`: análisis exploratorio específico sobre datos del conjunto **Comandos**.
- `Optuna_FNN.ipynb`: búsqueda de hiperparámetros y entrenamiento del modelo FNN.
- `Optuna_LTSM.ipynb`: optimización y entrenamiento del modelo LSTM.
- `Optuna_RF.ipynb`: optimización y entrenamiento del modelo Random Forest.

## Requisitos
- PyTorch, cuML, Optuna, Scikit-learn, Pandas, NumPy, Matplotlib

## Licencia

Este proyecto es de uso académico y no cuenta con licencia comercial.
