import numpy as np
import pandas as pd
from cuml.ensemble import RandomForestRegressor as cuRF
from cuml.metrics import mean_squared_error as cuml_mse
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import os
import json
from .utils import select_m_representative_points, pdist_manual, squareform_manual


class RFInversaUR5Simple:
    def __init__(self, ruta_datos, n_estimators=100, max_depth=16,
                 test_size=0.2, random_state=42, cols_excluir=None):
        """
        Inicializa una instancia de Random Forest para cinemática inversa del UR5 con RAPIDS cuML.
        """
        self.ruta = ruta_datos
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.test_size = test_size
        self.random_state = random_state
        self.cols_excluir = cols_excluir or []

        # Cargar y dividir datos
        self.X, self.Y = self._cargar_datos()
        self.input_dim = self.X.shape[1]
        self.output_dim = self.Y.shape[1]
        self.X_train, self.X_test, self.Y_train, self.Y_test = self._dividir_datos()

        # Modelo cuML
        self.modelo = cuRF(n_estimators=self.n_estimators,
                           max_depth=self.max_depth,
                           random_state=self.random_state)

        self.train_loss = None
        self.test_loss = None

    def _cargar_datos(self):
        if self.ruta.endswith('.xlsx'):
            df = pd.read_excel(self.ruta)
        else:
            df = pd.read_csv(self.ruta)

        cols_entrada = [c for c in df.columns if c not in [f'q{i}' for i in range(10)] and c not in self.cols_excluir]
        cols_salida = [f'q{i}' for i in range(10) if f'q{i}' in df.columns]

        X = df[cols_entrada].to_numpy(dtype=np.float32)
        Y = df[cols_salida].to_numpy(dtype=np.float32)
        return X, Y

    def _dividir_datos(self, pmax=10000):
        """
        Divide los datos en conjuntos de entrenamiento y prueba.
        Usa selección representativa si el tamaño es manejable; de lo contrario, usa división aleatoria.
        
        Parámetros:
            pmax (int): Límite máximo de muestras para usar selección representativa.

        Retorna:
            (X_train, X_test, Y_train, Y_test)
        """
        if self.test_size <= 0:
            return self.X, np.empty((0, self.X.shape[1])), self.Y, np.empty((0, self.Y.shape[1]))

        n_total = len(self.X)
        n_train = int(n_total * (1 - self.test_size))

        if n_total > pmax:
            return train_test_split(self.X, self.Y, train_size=n_train, random_state=self.random_state)

        D = squareform(pdist(self.X))
        train_idx = select_m_representative_points(D, n_train)
        test_idx = list(set(range(n_total)) - set(train_idx))

        return self.X[train_idx], self.X[test_idx], self.Y[train_idx], self.Y[test_idx]

    def entrenar(self):
        """
        Entrena un modelo Random Forest independiente por cada salida q_i usando cuML.
        """
        self.modelos = []
        self.train_loss = []
        self.test_loss = []

        for i in range(self.output_dim):
            modelo_i = cuRF(n_estimators=self.n_estimators,
                            max_depth=self.max_depth,
                            random_state=self.random_state)

            y_train_i = self.Y_train[:, i]
            y_test_i = self.Y_test[:, i]

            modelo_i.fit(self.X_train, y_train_i)
            self.modelos.append(modelo_i)

            y_pred_train = modelo_i.predict(self.X_train)
            y_pred_test = modelo_i.predict(self.X_test)

            mse_train_i = cuml_mse(y_train_i, y_pred_train)
            mse_test_i = cuml_mse(y_test_i, y_pred_test)

            self.train_loss.append(mse_train_i.get())
            self.test_loss.append(mse_test_i.get())


        print(f"MSE Promedio Entrenamiento: {np.mean(self.train_loss):.6f}")
        print(f"MSE Promedio Prueba:        {np.mean(self.test_loss):.6f}")

    def graficar_perdidas(self):
        """
        Grafica el MSE promedio de entrenamiento y prueba como barras.
        """
        if self.train_loss is None or self.test_loss is None:
            raise RuntimeError("El modelo no ha sido entrenado.")
        
        mse_train_promedio = np.mean(self.train_loss)
        mse_test_promedio = np.mean(self.test_loss)
        
        plt.bar(["Entrenamiento", "Prueba"], [mse_train_promedio, mse_test_promedio])
        plt.ylabel("MSE promedio")
        plt.title("Error cuadrático medio por conjunto")
        plt.grid(True)
        plt.show()

    def graficar_perdidas_por_articulacion(self):
        """
        Grafica el MSE por articulación, tanto en entrenamiento como en prueba.
        """
        etiquetas = [f"q{i}" for i in range(self.output_dim)]
        x = np.arange(len(etiquetas))
        width = 0.35

        plt.bar(x - width/2, self.train_loss, width, label='Entrenamiento')
        plt.bar(x + width/2, self.test_loss, width, label='Prueba')
        plt.xticks(x, etiquetas)
        plt.ylabel("MSE")
        plt.title("MSE por articulación")
        plt.legend()
        plt.grid(True, axis='y')
        plt.show()

    def graficar_offsets(self, pmax=10000):
        """
        Visualiza el error entre valores reales y predichos usando MDS.
        Usa funciones manuales en lugar de pdist y squareform.
        """
        Y_pred = np.column_stack([modelo.predict(self.X) for modelo in self.modelos])

        # Submuestreo si es necesario
        if len(self.Y) > pmax:
            print(f"Usando {pmax} puntos representativos para MDS...")
            Dvec = pdist_manual(self.Y)
            D = squareform_manual(Dvec)
            idx = select_m_representative_points(D, pmax)
            Y_true = self.Y[idx]
            Y_pred = Y_pred[idx]
        else:
            Y_true = self.Y
            Y_pred = Y_pred

        concat = np.vstack([Y_true, Y_pred])
        D_concat = pdist_manual(concat)
        stress_base = np.sum(D_concat ** 2) / 2
        mds = MDS(n_components=2, dissimilarity='euclidean', random_state=0, n_jobs=-1)
        emb = mds.fit_transform(concat)

        y_true_2d = emb[:len(Y_true)]
        y_pred_2d = emb[len(Y_true):]

        self.stress = mds.stress_
        self.stress_norm = self.stress / stress_base

        plt.figure(figsize=(10, 10), dpi=300)
        plt.plot([y_true_2d[:, 0], y_pred_2d[:, 0]],
                [y_true_2d[:, 1], y_pred_2d[:, 1]],
                'r-', alpha=0.2, linewidth=0.5)
        plt.scatter(y_true_2d[:, 0], y_true_2d[:, 1], c="blue", s=10, label="Verdaderos")
        plt.scatter(y_pred_2d[:, 0], y_pred_2d[:, 1], c="green", s=10, label="Predichos")
        plt.xlabel("MDS 1")
        plt.ylabel("MDS 2")
        plt.title(f"Proyección MDS del error ({len(Y_true)} puntos)\nStress: {self.stress_norm:.4f}")
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluar(self):
        """
        Evalúa el desempeño del modelo Random Forest sobre todos los datos y conjuntos por separado.
        Retorna:
            dict: métricas globales, por conjunto y por articulación.
        """
        # Reconstruir predicciones totales
        Y_pred_total = np.column_stack([
            modelo.predict(self.X) for modelo in self.modelos
        ])
        Y_pred_train = np.column_stack([
            modelo.predict(self.X_train) for modelo in self.modelos
        ])
        Y_pred_test = np.column_stack([
            modelo.predict(self.X_test) for modelo in self.modelos
        ])

        results = {
            'total': {
                'mse_rad': mean_squared_error(self.Y, Y_pred_total),
                'mse_deg': mean_squared_error(np.rad2deg(self.Y), np.rad2deg(Y_pred_total)),
                'r2': r2_score(self.Y, Y_pred_total),
                'Y_pred': Y_pred_total
            },
            'train': {
                'mse_rad': mean_squared_error(self.Y_train, Y_pred_train),
                'mse_deg': mean_squared_error(np.rad2deg(self.Y_train), np.rad2deg(Y_pred_train)),
                'r2': r2_score(self.Y_train, Y_pred_train),
                'Y_pred': Y_pred_train
            },
            'test': {
                'mse_rad': mean_squared_error(self.Y_test, Y_pred_test),
                'mse_deg': mean_squared_error(np.rad2deg(self.Y_test), np.rad2deg(Y_pred_test)),
                'r2': r2_score(self.Y_test, Y_pred_test),
                'Y_pred': Y_pred_test
            },
            'error_por_artic': np.mean((self.Y - Y_pred_total)**2, axis=0)
        }

        print(f"MSE total (radianes²): {results['total']['mse_rad']:.6f}")
        print(f"MSE total (grados²):   {results['total']['mse_deg']:.6f}")
        print(f"R²: {results['total']['r2']:.6f}")

        return results

    def summary(self):
        """
        Imprime un resumen detallado del modelo Random Forest y sus métricas de evaluación.
        """
        results = self.evaluar()

        print("=" * 50)
        print("Modelo: RF")
        print(f"Arquitectura: {self.input_dim}-{self.output_dim}")
        print(f"Parámetros: n_estimators={self.n_estimators}, max_depth={self.max_depth}")
        print(f"Opt: Simple")

        print("\nMétricas globales (todo el conjunto de datos):")
        print(f"R²: {results['total']['r2']:.6f}")
        print(f"MSE (radianes): {results['total']['mse_rad']:.6f}")
        print(f"MSE (grados): {results['total']['mse_deg']:.6f}")

        print("\nMétricas de entrenamiento:")
        print(f"R²: {results['train']['r2']:.6f}")
        print(f"MSE (radianes): {results['train']['mse_rad']:.6f}")
        print(f"MSE (grados): {results['train']['mse_deg']:.6f}")

        print("\nMétricas de prueba:")
        print(f"R²: {results['test']['r2']:.6f}")
        print(f"MSE (radianes): {results['test']['mse_rad']:.6f}")
        print(f"MSE (grados): {results['test']['mse_deg']:.6f}")

        print("\nError por articulación (global):")
        for i, err in enumerate(results['error_por_artic']):
            print(f"  q{i}: {err:.6f}")
        print("=" * 50)

    def guardar(self, nombre_base="modelo_rf"):
        """
        Guarda los modelos entrenados (uno por articulación) y la configuración del entrenamiento.
        
        Parámetros:
            nombre_base (str): Carpeta destino para guardar los archivos.
        """
        os.makedirs(nombre_base, exist_ok=True)

        # Guardar cada modelo individualmente
        for i, modelo in enumerate(self.modelos):
            ruta = os.path.join(nombre_base, f"rf_model_q{i}.pkl")
            joblib.dump(modelo, ruta)

        # Guardar configuración como JSON
        config = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "test_size": self.test_size,
            "random_state": self.random_state
        }
        with open(os.path.join(nombre_base, "configuracion.json"), "w") as f:
            json.dump(config, f, indent=4)

        print(f"Modelos y configuración guardados en: {nombre_base}/")

class RFInversaUR5:
    def __init__(self, ruta_datos, test_size=0.2, random_state=42, cols_excluir=None):
        self.ruta = ruta_datos
        self.test_size = test_size
        self.random_state = random_state
        self.cols_excluir = cols_excluir or []
        self.device = "cuda"

        self.X, self.Y = self._cargar_datos()
        self.input_dim = self.X.shape[1]
        self.output_dim = self.Y.shape[1]

        self.X_train, self.X_test, self.Y_train, self.Y_test = self._dividir_datos()
        self.study = None
        self.modelo_final = None
        self.stress = None
        self.stress_norm = None
        self.train_losses = []
        self.val_losses = []

    def _cargar_datos(self):
        if self.ruta.endswith('.xlsx'):
            df = pd.read_excel(self.ruta)
        else:
            df = pd.read_csv(self.ruta)
        cols_entrada = [c for c in df.columns if c not in [f'q{i}' for i in range(10)] and c not in self.cols_excluir]
        cols_salida = [f'q{i}' for i in range(10) if f'q{i}' in df.columns]
        X = df[cols_entrada].to_numpy(dtype=np.float32)
        Y = df[cols_salida].to_numpy(dtype=np.float32)
        return X, Y

    def _espacio(self, t):
        return {
            "n_estimators": t.suggest_categorical("n_estimators", [32, 64, 100]),
            "max_depth": t.suggest_categorical("max_depth", [8, 12, 16]),
            "max_features": t.suggest_categorical("max_features", [0.2, "sqrt", "log2"])
        }
    
    def _dividir_datos(self, pmax=500):
        if self.test_size <= 0:
            return self.X, np.empty((0, self.X.shape[1])), self.Y, np.empty((0, self.Y.shape[1]))
        n_total = len(self.X)
        n_train = int(n_total * (1 - self.test_size))
        if n_total > pmax:
            return train_test_split(self.X, self.Y, train_size=n_train, random_state=self.random_state)
        Dvec = pdist_manual(self.X)
        D = squareform_manual(Dvec)
        train_idx = select_m_representative_points(D, n_train)
        test_idx = list(set(range(n_total)) - set(train_idx))
        return self.X[train_idx], self.X[test_idx], self.Y[train_idx], self.Y[test_idx]

    def _train_once(self, p, Xtr, Ytr, Xval, Yval):
        modelos = []
        val_preds = []
        for i in range(self.output_dim):
            modelo_i = cuRF(n_estimators=p["n_estimators"], max_depth=p["max_depth"], random_state=self.random_state)
            modelo_i.fit(Xtr, Ytr[:, i])
            modelos.append(modelo_i)
            pred_i = modelo_i.predict(Xval)
            val_preds.append(pred_i.get() if hasattr(pred_i, 'get') else pred_i)
        Y_pred = np.column_stack(val_preds)
        mse = mean_squared_error(np.rad2deg(Yval), np.rad2deg(Y_pred))
        return mse, modelos, Y_pred

    def _objetivo(self, trial):
        p = self._espacio(trial)
        Xtr, Xval, Ytr, Yval = self._dividir_datos()
        mse, _, _ = self._train_once(p, Xtr, Ytr, Xval, Yval)
        return -mse

    def optimizar(self, n_trials=20, nombre_est="estudio_rf"):
        import optuna
        self.study = optuna.create_study(direction="maximize", study_name=nombre_est)
        self.study.optimize(self._objetivo, n_trials=n_trials)

    def entrenar_mejor_modelo(self):
        """
        Entrena el modelo final con los mejores hiperparámetros hallados por Optuna
        y reporta el error cuadrático medio (en radianes²). También calcula pérdidas
        por articulación (MSE y MAE) para visualización.
        """
        if self.study is None:
            raise RuntimeError("Ejecute optimizar() primero.")

        p = {**self._espacio(self.study.best_trial), **self.study.best_params}
        Xtr, Xval, Ytr, Yval = self._dividir_datos()
        mse_rad, modelos, _ = self._train_once(p, Xtr, Ytr, Xval, Yval)

        self.modelo_final = modelos
        print(f"MSE en radianes² del modelo final (validación): {mse_rad:.6f}")

        # Cálculo de errores por articulación
        self.train_loss = []
        self.test_loss = []
        self.train_loss_l1 = []
        self.test_loss_l1 = []

        for i, modelo in enumerate(self.modelo_final):
            pred_train = modelo.predict(self.X_train)
            pred_test = modelo.predict(self.X_test)

            self.train_loss.append(mean_squared_error(self.Y_train[:, i], pred_train))
            self.test_loss.append(mean_squared_error(self.Y_test[:, i], pred_test))

            self.train_loss_l1.append(np.mean(np.abs(self.Y_train[:, i] - pred_train)))
            self.test_loss_l1.append(np.mean(np.abs(self.Y_test[:, i] - pred_test)))

        return mse_rad
    
    def predecir(self, X):
        if self.modelo_final is None:
            raise RuntimeError("Modelo no entrenado.")
        return np.column_stack([m.predict(X) for m in self.modelo_final])

    def graficar_offsets(self, pmax=10000):
        Y_pred = self.predecir(self.X)
        if len(self.Y) > pmax:
            Dvec = pdist_manual(self.Y)
            D = squareform_manual(Dvec)
            idx = select_m_representative_points(D, pmax)
            Y_true = self.Y[idx]
            Y_pred = Y_pred[idx]
        else:
            Y_true = self.Y
        concat = np.vstack([Y_true, Y_pred])
        D_concat = pdist_manual(concat)
        stress_base = np.sum(D_concat ** 2) / 2
        mds = MDS(n_components=2, dissimilarity='euclidean', random_state=0, n_jobs=-1)
        emb = mds.fit_transform(concat)
        y_true_2d = emb[:len(Y_true)]
        y_pred_2d = emb[len(Y_true):]
        self.stress = mds.stress_
        self.stress_norm = self.stress / stress_base
        plt.figure(figsize=(10, 10), dpi=300)
        plt.plot([y_true_2d[:, 0], y_pred_2d[:, 0]], [y_true_2d[:, 1], y_pred_2d[:, 1]], 'r-', alpha=0.2, linewidth=0.5)
        plt.scatter(y_true_2d[:, 0], y_true_2d[:, 1], c="blue", s=10, label="Verdaderos")
        plt.scatter(y_pred_2d[:, 0], y_pred_2d[:, 1], c="green", s=10, label="Predichos")
        plt.xlabel("MDS 1"); plt.ylabel("MDS 2")
        plt.title(f"Proyección MDS del error ({len(Y_true)} puntos)\nStress: {self.stress_norm:.4f}")
        plt.legend(); plt.grid(True); plt.show()

    def summary(self):
        if self.modelo_final is None:
            raise RuntimeError("Modelo no entrenado.")

        # Predicciones globales
        Y_pred = self.predecir(self.X)
        Y_pred_train = self.predecir(self.X_train)
        Y_pred_test = self.predecir(self.X_test)

        def calcular_metricas(Y_true, Y_pred):
            return {
                "r2": r2_score(Y_true, Y_pred),
                "mse_rad": mean_squared_error(Y_true, Y_pred),
                "mse_deg": mean_squared_error(np.rad2deg(Y_true), np.rad2deg(Y_pred)),
                "mae_rad": np.mean(np.abs(Y_true - Y_pred)),
                "mae_deg": np.mean(np.abs(np.rad2deg(Y_true) - np.rad2deg(Y_pred)))
            }

        res_total = calcular_metricas(self.Y, Y_pred)
        res_train = calcular_metricas(self.Y_train, Y_pred_train)
        res_test  = calcular_metricas(self.Y_test, Y_pred_test)

        error_artic_l2 = np.mean((self.Y - Y_pred)**2, axis=0)
        error_artic_l1 = np.mean(np.abs(self.Y - Y_pred), axis=0)

        print("=" * 50)
        print("Modelo: RF")
        print(f"Arquitectura: {self.input_dim}-{self.output_dim}")
        print(f"Parámetros: {self.study.best_params if self.study else 'Manual'}")
        print(f"Opt: {'Optuna' if self.study else 'Simple'}")

        print("\nMétricas globales (todo el conjunto de datos):")
        print(f"R²: {res_total['r2']:.6f}")
        print(f"MSE (radianes²): {res_total['mse_rad']:.6f}")
        print(f"MSE (grados²):  {res_total['mse_deg']:.6f}")
        print(f"MAE (radianes): {res_total['mae_rad']:.6f}")
        print(f"MAE (grados):   {res_total['mae_deg']:.6f}")

        print("\nMétricas de entrenamiento:")
        print(f"R²: {res_train['r2']:.6f}")
        print(f"MSE (radianes²): {res_train['mse_rad']:.6f}")
        print(f"MSE (grados²):  {res_train['mse_deg']:.6f}")
        print(f"MAE (radianes): {res_train['mae_rad']:.6f}")
        print(f"MAE (grados):   {res_train['mae_deg']:.6f}")

        print("\nMétricas de prueba:")
        print(f"R²: {res_test['r2']:.6f}")
        print(f"MSE (radianes²): {res_test['mse_rad']:.6f}")
        print(f"MSE (grados²):  {res_test['mse_deg']:.6f}")
        print(f"MAE (radianes): {res_test['mae_rad']:.6f}")
        print(f"MAE (grados):   {res_test['mae_deg']:.6f}")

        print("\nError por articulación (global):")
        for i in range(self.output_dim):
            print(f"  q{i}: L2={error_artic_l2[i]:.6f}  |  L1={error_artic_l1[i]:.6f}")

        if hasattr(self, "stress") and self.stress is not None:
            print(f"\nMDS stress: {self.stress:.6f}")
            print(f"MDS stress normalizado: {self.stress_norm:.6f}")

        print("=" * 50)

    def graficar_perdidas(self):
        if self.train_loss is None or self.test_loss is None:
            raise RuntimeError("El modelo no ha sido entrenado.")
        mse_train_promedio = np.mean(self.train_loss)
        mse_test_promedio = np.mean(self.test_loss)
        plt.bar(["Entrenamiento", "Prueba"], [mse_train_promedio, mse_test_promedio])
        plt.ylabel("MSE promedio")
        plt.title("Error cuadrático medio por conjunto")
        plt.grid(True)
        plt.show()

    def graficar_perdidas_por_articulacion(self, L1=False):
        """
        Grafica el error por articulación, tanto en entrenamiento como en prueba.
        Parámetros:
            L1 (bool): Si True, grafica error absoluto medio (MAE); si False, error cuadrático medio (MSE).
        """
        etiquetas = [f"q{i}" for i in range(self.output_dim)]
        x = np.arange(len(etiquetas))
        width = 0.35

        if L1:
            if not hasattr(self, 'train_loss_l1') or not hasattr(self, 'test_loss_l1'):
                raise RuntimeError("No se han calculado las pérdidas L1. Entrene con _train_once que calcule MAE.")
            y_train = self.train_loss_l1
            y_test = self.test_loss_l1
            ylabel = "MAE"
            titulo = "MAE por articulación"
        else:
            y_train = self.train_loss
            y_test = self.test_loss
            ylabel = "MSE"
            titulo = "MSE por articulación"

        plt.bar(x - width/2, y_train, width, label='Entrenamiento')
        plt.bar(x + width/2, y_test, width, label='Prueba')
        plt.xticks(x, etiquetas)
        plt.ylabel(ylabel)
        plt.title(titulo)
        plt.legend()
        plt.grid(True, axis='y')
        plt.show()

    def guardar(self, nombre_base="modelo_RF"):
        """
        Guarda el modelo final entrenado, resultados de Optuna y el historial de pérdidas en la carpeta especificada.

        Parámetros:
            nombre_base (str): Nombre de la carpeta destino donde se guardarán todos los archivos.
                            Por defecto es "modelo_RF".

        Acciones:
            - Crea la carpeta especificada si no existe.
            - Guarda los modelos RF (uno por articulación) en formato joblib.
            - Guarda resultados y gráficas de Optuna.
            - Guarda historial de pérdidas y sus visualizaciones.
        """
        import joblib
        import optuna.visualization.matplotlib as vis
        import pandas as pd
        import json
        
        if self.modelo_final is None or self.study is None:
            raise RuntimeError("Optimice y entrene antes de guardar.")
        
        # Crear directorio si no existe
        os.makedirs(nombre_base, exist_ok=True)
        
        # Guardar modelos (uno por articulación)
        for i, modelo in enumerate(self.modelo_final):
            joblib.dump(modelo, f"{nombre_base}/rf_modelo_q{i}.joblib")
            
        # Guardar configuración y parámetros
        config = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "best_params": self.study.best_params
        }
        with open(f"{nombre_base}/configuracion.json", "w") as f:
            json.dump(config, f, indent=4)
        
        # Guardar resultados de Optuna
        self.study.trials_dataframe().to_csv(f"{nombre_base}/optuna_resultados.csv", index=False)
        
        # Guardar gráficas de Optuna
        try:
            vis.plot_optimization_history(self.study)
            plt.savefig(f"{nombre_base}/opt_history.png")
            plt.clf()
            
            vis.plot_param_importances(self.study)
            plt.savefig(f"{nombre_base}/opt_param_importance.png")
            plt.clf()
        except Exception as e:
            print(f"Advertencia: No se pudieron generar las gráficas de Optuna: {e}")
        
        # Guardar historial de pérdidas
        if hasattr(self, 'train_loss') and self.train_loss is not None:
            # Crear DataFrame con pérdidas MSE por articulación
            train_df = pd.DataFrame({f'q{i}': [loss] for i, loss in enumerate(self.train_loss)})
            train_df['conjunto'] = 'entrenamiento'
            
            test_df = pd.DataFrame({f'q{i}': [loss] for i, loss in enumerate(self.test_loss)})
            test_df['conjunto'] = 'prueba'
            
            loss_df = pd.concat([train_df, test_df], ignore_index=True)
            loss_df.to_csv(f"{nombre_base}/perdidas_por_articulacion.csv", index=False)
            
            # También guardar MAE si está disponible
            if hasattr(self, 'train_loss_l1') and self.train_loss_l1 is not None:
                train_l1_df = pd.DataFrame({f'q{i}': [loss] for i, loss in enumerate(self.train_loss_l1)})
                train_l1_df['conjunto'] = 'entrenamiento'
                
                test_l1_df = pd.DataFrame({f'q{i}': [loss] for i, loss in enumerate(self.test_loss_l1)})
                test_l1_df['conjunto'] = 'prueba'
                
                loss_l1_df = pd.concat([train_l1_df, test_l1_df], ignore_index=True)
                loss_l1_df.to_csv(f"{nombre_base}/mae_por_articulacion.csv", index=False)
            
            # Guardar gráfica de pérdidas MSE por conjunto
            plt.figure(figsize=(10, 6))
            plt.bar(["Entrenamiento", "Prueba"], 
                    [np.mean(self.train_loss), np.mean(self.test_loss)])
            plt.ylabel("MSE promedio")
            plt.title("Error cuadrático medio por conjunto")
            plt.grid(True)
            plt.savefig(f"{nombre_base}/mse_por_conjunto.png")
            plt.close()
            
            # Guardar gráfica de pérdidas por articulación
            etiquetas = [f"q{i}" for i in range(self.output_dim)]
            x = np.arange(len(etiquetas))
            width = 0.35
            
            plt.figure(figsize=(12, 6))
            plt.bar(x - width/2, self.train_loss, width, label='Entrenamiento')
            plt.bar(x + width/2, self.test_loss, width, label='Prueba')
            plt.xticks(x, etiquetas)
            plt.ylabel("MSE")
            plt.title("MSE por articulación")
            plt.legend()
            plt.grid(True, axis='y')
            plt.savefig(f"{nombre_base}/mse_por_articulacion.png")
            plt.close()
            
            # Guardar gráfica de pérdidas MAE por articulación si está disponible
            if hasattr(self, 'train_loss_l1') and self.train_loss_l1 is not None:
                plt.figure(figsize=(12, 6))
                plt.bar(x - width/2, self.train_loss_l1, width, label='Entrenamiento')
                plt.bar(x + width/2, self.test_loss_l1, width, label='Prueba')
                plt.xticks(x, etiquetas)
                plt.ylabel("MAE")
                plt.title("MAE por articulación")
                plt.legend()
                plt.grid(True, axis='y')
                plt.savefig(f"{nombre_base}/mae_por_articulacion.png")
                plt.close()
        
        print(f"Modelo y resultados guardados en carpeta: {nombre_base}")
        
    def predecir(self, X):
        """
        Realiza una predicción utilizando los modelos RF entrenados.

        Parámetros
        ----------
        X : array-like
            Datos de entrada para la predicción. Debe ser un array de numpy o similar.

        Retorna
        -------
        numpy.ndarray
            Resultado de la predicción como un array con las predicciones para cada articulación.

        Lanza
        -----
        RuntimeError
            Si los modelos no han sido entrenados previamente (modelo_final es None).
        """
        if self.modelo_final is None:
            raise RuntimeError("Modelo no entrenado.")
            
        # Convertir entrada a numpy array si no lo es
        X = np.asarray(X)
        
        # Manejar tanto entradas individuales como lotes
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Realizar predicciones con cada modelo (uno por articulación)
        predicciones = []
        for modelo in self.modelo_final:
            pred = modelo.predict(X)
            # Convertir a numpy si es un objeto de cuML
            if hasattr(pred, 'get'):
                pred = pred.get()
            predicciones.append(pred)
        
        # Combinar predicciones de todas las articulaciones
        return np.column_stack(predicciones)

