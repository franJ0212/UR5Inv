import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
import optuna, joblib, matplotlib.pyplot as plt
import optuna.visualization.matplotlib as vis
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from .utils import select_m_representative_points

# Red neuronal totalmente conectada
class _FNN(nn.Module):
    """
    Red neuronal feedforward (FNN) para tareas de regresión o clasificación.
    Parámetros:
        input_dim (int): Dimensión de entrada de los datos. Por defecto es 3.
        output_dim (int): Dimensión de salida de la red. Por defecto es 6.
        hidden_layers (tuple): Tupla que indica el número de neuronas en cada capa oculta. Por defecto es (64, 32).
        dropout (float): Tasa de abandono (dropout) aplicada después de cada capa oculta. Por defecto es 0.0.
    Métodos:
        forward(x): Propaga la entrada 'x' a través de la red y retorna la salida.
    """
    def __init__(self, input_dim=3, output_dim=6, hidden_layers=(64, 32), dropout=0.0):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Clase principal
class FNNInversaUR5:
    """
    FNNInversaUR5
    Clase para entrenar y evaluar una red neuronal feedforward (FNN) para el problema de cinemática inversa del robot UR5.
    Permite cargar datos, optimizar hiperparámetros con Optuna, entrenar el modelo, evaluar su desempeño y visualizar resultados.
    Incluye métodos para graficar pérdidas, analizar errores mediante MDS, guardar el modelo y realizar predicciones.
    
    Parámetros:
    - ruta_datos (str): Ruta al archivo de datos (CSV o Excel).
    - test_size (float): Proporción de datos para validación/test.
    - random_state (int): Semilla para la aleatorización.
    - dispositivo (str): Dispositivo de cómputo ('cuda' o 'cpu').
    - cols_excluir (list): Lista de columnas a excluir de las entradas.
    
    Métodos principales:
    - optimizar: Realiza la optimización de hiperparámetros.
    - entrenar_mejor_modelo: Entrena el modelo con los mejores parámetros encontrados.
    - graficar_perdidas: Muestra la evolución de la pérdida durante el entrenamiento.
    - graficar_offsets: Visualiza el error de predicción usando MDS.
    - evaluar_total: Evalúa el modelo en todo el conjunto de datos.
    - summary: Imprime un resumen del desempeño del modelo.
    - guardar: Guarda el modelo entrenado y los resultados de la optimización.
    - predecir: Realiza predicciones con el modelo entrenado.
    """
    
    def __init__(self, ruta_datos, test_size=0.2, random_state=42, dispositivo=None, cols_excluir=None):
        """
        Inicializa una instancia de la clase para el manejo y entrenamiento de una red neuronal.
        Parámetros:
        -----------
        ruta_datos : str
            Ruta al archivo de datos que será utilizado para cargar los datos de entrada y salida.
        test_size : float, opcional (por defecto=0.2)
            Proporción de los datos que se utilizarán para el conjunto de prueba.
        random_state : int, opcional (por defecto=42)
            Semilla para la generación de números aleatorios, asegurando la reproducibilidad.
        dispositivo : str o None, opcional
            Dispositivo a utilizar para el entrenamiento ('cuda' para GPU o 'cpu'). Si es None, se selecciona automáticamente.
        cols_excluir : list o None, opcional
            Lista de nombres de columnas a excluir de los datos de entrada. Si es None, no se excluye ninguna columna.
        """
        self.ruta = ruta_datos
        self.test_size = test_size
        self.random_state = random_state  # Corregido: ahora se usa random_state en vez de random
        self.device = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cols_excluir = cols_excluir or []
        
        self.X, self.Y = self._cargar_datos()
        self.input_dim = self.X.shape[1]
        self.output_dim = self.Y.shape[1]
        self.train_losses = []
        self.val_losses = []
        self.study = None
        self.modelo_final = None
        self.stress = None
        self.stress_norm = None
        
    def _cargar_datos(self):
        """
        Carga los datos desde un archivo especificado en la ruta del objeto, detectando automáticamente las columnas de entrada y salida.
        Si el archivo es de tipo '.xlsx', se utiliza pandas.read_excel; en caso contrario, se utiliza pandas.read_csv.
        Las columnas de entrada se determinan como aquellas que no corresponden a las salidas (q0, q1, ..., q9) ni a las columnas excluidas especificadas en self.cols_excluir.
        Las columnas de salida se consideran aquellas con nombres 'q0' a 'q9' presentes en el archivo.
        Devuelve:
            X (np.ndarray): Matriz de características de entrada, de tipo float32.
            Y (np.ndarray): Matriz de salidas, de tipo float32.
        """
        if self.ruta.endswith('.xlsx'):
            df = pd.read_excel(self.ruta)
        else:
            df = pd.read_csv(self.ruta)
            
        # Detectar automáticamente columnas de entrada y salida
        cols_entrada = [c for c in df.columns if c not in [f'q{i}' for i in range(10)] and c not in self.cols_excluir]
        cols_salida = [f'q{i}' for i in range(10) if f'q{i}' in df.columns]
        
        X = df[cols_entrada].to_numpy(dtype=np.float32)
        Y = df[cols_salida].to_numpy(dtype=np.float32)
        return X, Y
        
    def _espacio(self, t):
        """
        Genera un espacio de búsqueda de hiperparámetros para la optimización de una red neuronal.

        Parámetros:
            t: objeto de prueba (trial) de Optuna utilizado para sugerir valores de hiperparámetros.

        Retorna:
            dict: Un diccionario con los siguientes hiperparámetros sugeridos:
                - "hidden_layers": tupla con el número de neuronas en cada capa oculta, seleccionada de varias opciones predefinidas.
                - "dropout": valor flotante para la tasa de dropout, entre 0.0 y 0.4.
                - "lr": tasa de aprendizaje (learning rate), valor flotante entre 1e-4 y 5e-3 (escala logarítmica).
                - "batch": tamaño del batch, seleccionado entre 32, 64 o 128.
                - "epochs": número de épocas de entrenamiento, entero entre 50 y 300.
        """
        hidden_options = {
            "64-32": (64, 32),
            "128-64-32": (128, 64, 32),
            "256-128-64": (256, 128, 64),
            "512-256-128": (512, 256, 128),
            "1024-512-256": (1024, 512, 256)
        }
        key = t.suggest_categorical("hidden_key", list(hidden_options.keys()))
        return {
            "hidden_layers": hidden_options[key],
            "dropout": t.suggest_float("dropout", 0.0, 0.4),
            "lr": t.suggest_float("lr", 1e-4, 5e-3, log=True),
            "batch": t.suggest_categorical("batch", [64]),
            "epochs": t.suggest_int("epochs", 50, 500)
        }
    
    def _dividir_datos(self):
        """
        Divide los datos en conjuntos de entrenamiento y prueba de manera representativa.
        Si `test_size` es menor o igual a 0, todos los datos se asignan al conjunto de entrenamiento y el conjunto de prueba queda vacío.
        Si `test_size` es mayor que 0 y hay más de 5000 muestras, usa train_test_split para una división aleatoria.
        Si hay menos de 5000 muestras, selecciona un subconjunto representativo utilizando distancias entre muestras.
        
        Returns:
            X_train (np.ndarray): Conjunto de características para entrenamiento.
            X_test (np.ndarray): Conjunto de características para prueba.
            Y_train (np.ndarray): Conjunto de etiquetas para entrenamiento.
            Y_test (np.ndarray): Conjunto de etiquetas para prueba.
        """
        if self.test_size <= 0:
            # Caso 1: No hay división (todo para entrenamiento)
            return self.X, np.empty((0, self.X.shape[1])), self.Y, np.empty((0, self.Y.shape[1]))
        
        # Caso 2: Si hay más de 5000 muestras, usar train_test_split
        if len(self.X) > 5000:
            from sklearn.model_selection import train_test_split
            X_train, X_test, Y_train, Y_test = train_test_split(
                self.X, self.Y, test_size=self.test_size, random_state=self.random_state
            )
            return X_train, X_test, Y_train, Y_test
        
        # Caso 3: Si hay menos de 5000 muestras, usar el método original
        from scipy.spatial.distance import pdist, squareform
        
        try:
            from .utils import select_m_representative_points
        except (ImportError, ValueError):
            # Implementación básica si no está disponible
            def select_m_representative_points(distances, m):
                import numpy as np
                n = distances.shape[0]
                if m >= n:
                    return list(range(n))
                
                np.random.seed(self.random_state)
                selected_indices = [np.random.randint(0, n)]
                
                for _ in range(m - 1):
                    min_distances = np.min([distances[i, selected_indices] for i in range(n)], axis=1)
                    min_distances[selected_indices] = -1
                    next_point = np.argmax(min_distances)
                    selected_indices.append(next_point)
                
                return selected_indices
        
        n_train = int(len(self.X) * (1 - self.test_size))
        distances = squareform(pdist(self.X))
        train_idx = select_m_representative_points(distances, n_train)
        test_idx = list(set(range(len(self.X))) - set(train_idx))
        
        X_train, X_test = self.X[train_idx], self.X[test_idx]
        Y_train, Y_test = self.Y[train_idx], self.Y[test_idx]
        return X_train, X_test, Y_train, Y_test
    
    def _train_once(self, p, Xtr, Ytr, Xval, Yval):
        """
        Entrena una vez una red neuronal feedforward (_FNN) utilizando los datos de entrenamiento y validación proporcionados.
        Parámetros
        ----------
        p : dict
            Diccionario de hiperparámetros que incluye:
                - "hidden_layers": lista con el número de neuronas por capa oculta.
                - "dropout": tasa de dropout.
                - "lr": tasa de aprendizaje.
                - "batch": tamaño de lote.
                - "epochs": número de épocas de entrenamiento.
        Xtr : array-like
            Datos de entrada para entrenamiento.
        Ytr : array-like
            Etiquetas/valores objetivo para entrenamiento.
        Xval : array-like
            Datos de entrada para validación.
        Yval : array-like
            Etiquetas/valores objetivo para validación.
        Retorna
        -------
        mse : float
            Error cuadrático medio (MSE) en el conjunto de validación (o entrenamiento si no hay validación), convertido a grados si aplica.
        model : _FNN
            Modelo entrenado de red neuronal.
        Y_pred : numpy.ndarray
            Predicciones del modelo sobre el conjunto de validación (o entrenamiento si no hay validación).
        Notas
        -----
        - Guarda las pérdidas de entrenamiento y validación en los atributos `self.train_losses` y `self.val_losses`.
        - Utiliza Adam como optimizador y MSE como función de pérdida.
        - Si no se proporcionan datos de validación, la evaluación final se realiza sobre los datos de entrenamiento.
        """
        model = _FNN(self.input_dim, self.output_dim, p["hidden_layers"], p["dropout"]).to(self.device)
        opt = optim.Adam(model.parameters(), lr=p["lr"])
        loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr)), 
                           batch_size=p["batch"], shuffle=True)
        mse_fn = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(p["epochs"]):
            model.train()
            epoch_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                loss = mse_fn(model(xb), yb)
                epoch_loss += loss.item() * len(xb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            train_losses.append(epoch_loss / len(Xtr))
            
            # Validación
            if len(Xval) > 0:
                model.eval()
                with torch.no_grad():
                    Y_pred = model(torch.tensor(Xval).to(self.device)).cpu().numpy()
                val_mse = mean_squared_error(Yval, Y_pred)
                val_losses.append(val_mse)
        
        # Guardar pérdidas
        self.train_losses = train_losses
        self.val_losses = val_losses
        
        # Evaluación final
        model.eval()
        with torch.no_grad():
            if len(Xval) > 0:
                Y_pred = model(torch.tensor(Xval).to(self.device)).cpu().numpy()
                mse = mean_squared_error(np.rad2deg(Yval), np.rad2deg(Y_pred))
            else:
                Y_pred = model(torch.tensor(Xtr).to(self.device)).cpu().numpy()
                mse = mean_squared_error(np.rad2deg(Ytr), np.rad2deg(Y_pred))
                
        return mse, model, Y_pred
        
    def _objetivo(self, trial):
        """
        Función objetivo para la optimización de hiperparámetros.

        Esta función se utiliza en un proceso de optimización (por ejemplo, con Optuna) para evaluar un conjunto de hiperparámetros propuestos por el objeto `trial`.
        Divide los datos en conjuntos de entrenamiento y validación,
        entrena el modelo una vez con los hiperparámetros sugeridos y 
        calcula el error cuadrático medio (MSE) en el conjunto de validación.

        Args:
            trial: Objeto de prueba de la librería de optimización, que sugiere un conjunto de hiperparámetros.

        Returns:
            float: El negativo del error cuadrático medio (MSE) en el conjunto de validación, ya que el objetivo es maximizar esta función.
        """
        p = self._espacio(trial)
        Xtr, Xval, Ytr, Yval = self._dividir_datos()
        mse, _, _ = self._train_once(p, Xtr, Ytr, Xval, Yval)
        return -mse  # maximizar
        
    def optimizar(self, n_trials=30, nombre_est="estudio_ur5_fnn"):
        """
        Optimiza los hiperparámetros del modelo utilizando Optuna.

        Este método crea un estudio de Optuna y ejecuta la optimización del objetivo definido en el método `_objetivo`.
        El objetivo es maximizar la métrica especificada por el modelo.

        Parámetros:
            n_trials (int): Número de iteraciones (pruebas) para la optimización. Por defecto es 30.
            nombre_est (str): Nombre del estudio de Optuna. Por defecto es "estudio_ur5_fnn".

        Retorna:
            None. El resultado del estudio se almacena en el atributo `self.study`.
        """
        self.study = optuna.create_study(direction="maximize", study_name=nombre_est)
        self.study.optimize(self._objetivo, n_trials=n_trials, n_jobs=1)
        
    def entrenar_mejor_modelo(self):
        """
        Entrena el mejor modelo encontrado durante la optimización de hiperparámetros y lo almacena como modelo final.

        Si no se ha ejecutado previamente el proceso de optimización, lanza una excepción. 
        Utiliza los mejores parámetros encontrados para entrenar el modelo sobre los datos de entrenamiento y validación, 
        calcula el error cuadrático medio (MSE) en la validación y guarda el modelo entrenado como atributo de la clase.

        Returns:
            float: El error cuadrático medio (MSE) en grados² sobre el conjunto de validación.

        Raises:
            RuntimeError: Si no se ha ejecutado el método de optimización previamente.
        """
        if self.study is None:
            raise RuntimeError("Ejecute optimizar() primero.")
        p = {**self._espacio(self.study.best_trial), **self.study.best_params}
        Xtr, Xval, Ytr, Yval = self._dividir_datos()
        mse, model, _ = self._train_once(p, Xtr, Ytr, Xval, Yval)
        self.modelo_final = model
        print(f"MSE grados² validación: {mse:.6f}")
        return mse
        
    def graficar_perdidas(self):
        """
        Genera una gráfica de la evolución de la pérdida (MSE) durante el entrenamiento del modelo.

        Muestra la curva de pérdida para el conjunto de entrenamiento y, si está disponible, para el conjunto de validación.
        La gráfica incluye etiquetas, título, leyenda y una cuadrícula para facilitar la interpretación visual.

        Parámetros:
            No recibe parámetros adicionales. Utiliza los atributos `train_losses` y `val_losses` de la instancia.

        Retorna:
            None. Muestra la gráfica en pantalla.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Entrenamiento')
        if len(self.val_losses) > 0:
            plt.plot(self.val_losses, label='Validación')
        plt.xlabel('Época')
        plt.ylabel('Pérdida (MSE)')
        plt.title('Evolución de la pérdida durante el entrenamiento')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def graficar_offsets(self, pmax=10000):
        """
        Visualiza el error entre valores reales y predichos usando MDS,
        solo sobre pmax puntos seleccionados (sin interpolación).

        Parámetros:
            pmax (int): Número máximo de puntos representativos a usar para MDS.
        """

        if self.modelo_final is None:
            raise RuntimeError("Modelo no entrenado.")

        # Evaluación del modelo
        self.modelo_final.eval()
        with torch.no_grad():
            Y_pred = self.modelo_final(torch.tensor(self.X, dtype=torch.float32).to(self.device)).cpu().numpy()

        # Calcular distancias sobre Y verdadero
        distances = squareform(pdist(self.Y))

        # Seleccionar puntos representativos si es necesario
        if len(self.Y) > pmax:
            print(f"Usando {pmax} puntos representativos para MDS...")
            selected_indices = select_m_representative_points(distances, pmax)
            Y_true = self.Y[selected_indices]
            Y_pred = Y_pred[selected_indices]
        else:
            selected_indices = np.arange(len(self.Y))
            Y_true = self.Y
            Y_pred = Y_pred

        # Aplicar MDS directamente con paralelización
        concat = np.vstack([Y_true, Y_pred])
        mds = MDS(n_components=2, dissimilarity='euclidean', random_state=0, n_jobs=-1)
        emb = mds.fit_transform(concat)
        y_true_2d = emb[:len(Y_true)]
        y_pred_2d = emb[len(Y_true):]

        # Calcular stress normalizado
        self.stress = mds.stress_
        self.stress_norm = self.stress / np.sum(pdist(concat) ** 2 / 2)

        # Visualización optimizada con método vectorizado
        plt.figure(figsize=(10, 10), dpi=300)
        
        # Método vectorizado para dibujar líneas (más eficiente)
        plt.plot(
            [y_true_2d[:, 0], y_pred_2d[:, 0]],
            [y_true_2d[:, 1], y_pred_2d[:, 1]],
            'r-', alpha=0.2, linewidth=0.5
        )
        
        plt.scatter(y_true_2d[:, 0], y_true_2d[:, 1], c="blue", s=10, label="Verdaderos")
        plt.scatter(y_pred_2d[:, 0], y_pred_2d[:, 1], c="green", s=10, label="Predichos")
        plt.xlabel("MDS 1")
        plt.ylabel("MDS 2")
        plt.title(f"Proyección MDS del error ({len(Y_true)} puntos)\n"
                f"Stress: {self.stress_norm:.4f}")
        plt.legend()
        plt.grid(True)
        plt.show()
       
    def evaluar_total(self):
        """
        Evalúa el desempeño del modelo final sobre todo el conjunto de datos, así como sobre los subconjuntos de entrenamiento y prueba si corresponde.
        Calcula las siguientes métricas:
            - Error cuadrático medio (MSE) en radianes y grados.
            - Coeficiente de determinación R².
            - Error medio por articulación.
        Si existe una división de datos (train/test), también calcula las métricas por separado para cada subconjunto.
        Returns:
            dict: Diccionario con las métricas calculadas para el conjunto total, entrenamiento y prueba (si aplica), así como el error por articulación y las predicciones correspondientes.
        Raises:
            RuntimeError: Si el modelo no ha sido entrenado (`modelo_final` es None).
        """

        if self.modelo_final is None:
            raise RuntimeError("Modelo no entrenado.")
        
        # Evaluación sobre todo el conjunto de datos
        self.modelo_final.eval()
        with torch.no_grad():
            Y_pred_total = self.modelo_final(
                torch.tensor(self.X, dtype=torch.float32).to(self.device)
            ).cpu().numpy()
        
        mse_rad_total = mean_squared_error(self.Y, Y_pred_total)
        mse_deg_total = mean_squared_error(np.rad2deg(self.Y), np.rad2deg(Y_pred_total))
        r2_total = r2_score(self.Y, Y_pred_total)
        
        # Error por articulación
        error_por_artic = np.mean((self.Y - Y_pred_total)**2, axis=0)
        
        # Resultados de entrenamiento y prueba si corresponde
        results = {
            'total': {
                'mse_rad': mse_rad_total,
                'mse_deg': mse_deg_total,
                'r2': r2_total,
                'Y_pred': Y_pred_total
            },
            'train': None,
            'test': None,
            'error_por_artic': error_por_artic
        }
        
        # Si hay división de datos, calcular métricas por separado
        if self.test_size > 0:
            X_train, X_test, Y_train, Y_test = self._dividir_datos()
            
            # Métricas de entrenamiento
            with torch.no_grad():
                Y_pred_train = self.modelo_final(
                    torch.tensor(X_train, dtype=torch.float32).to(self.device)
                ).cpu().numpy()
            
            mse_rad_train = mean_squared_error(Y_train, Y_pred_train)
            mse_deg_train = mean_squared_error(np.rad2deg(Y_train), np.rad2deg(Y_pred_train))
            r2_train = r2_score(Y_train, Y_pred_train)
            
            results['train'] = {
                'mse_rad': mse_rad_train,
                'mse_deg': mse_deg_train,
                'r2': r2_train,
                'Y_pred': Y_pred_train
            }
            
            # Métricas de prueba
            with torch.no_grad():
                Y_pred_test = self.modelo_final(
                    torch.tensor(X_test, dtype=torch.float32).to(self.device)
                ).cpu().numpy()
            
            mse_rad_test = mean_squared_error(Y_test, Y_pred_test)
            mse_deg_test = mean_squared_error(np.rad2deg(Y_test), np.rad2deg(Y_pred_test))
            r2_test = r2_score(Y_test, Y_pred_test)
            
            results['test'] = {
                'mse_rad': mse_rad_test,
                'mse_deg': mse_deg_test,
                'r2': r2_test,
                'Y_pred': Y_pred_test
            }
        
        # Imprimir solo las métricas globales
        print(f"MSE total (radianes²): {mse_rad_total:.6f}")
        print(f"MSE total (grados²):   {mse_deg_total:.6f}")
        print(f"R²: {r2_total:.6f}")
        
        return results
          
    def summary(self):
        """
        Muestra un resumen detallado del modelo FNN entrenado, incluyendo arquitectura, parámetros óptimos,
        métricas de desempeño globales, de entrenamiento y prueba, así como errores por articulación y 
        métricas de MDS si están disponibles.
        Lanza:
            RuntimeError: Si el modelo no ha sido entrenado (`self.modelo_final` es None).
        Salida:
            Imprime en consola la información relevante sobre el modelo y su desempeño.
        """

        if self.modelo_final is None:
            raise RuntimeError("Modelo no entrenado.")
        
        results = self.evaluar_total()
        
        print("=" * 50)
        print("Modelo: FNN")
        print(f"Arquitectura: {self.input_dim}-{'-'.join(map(str, self.study.best_params.get('hidden_layers', [])))}-{self.output_dim}")
        print(f"Parámetros: {self.study.best_params}")
        print(f"Opt: Optuna")
        
        # Métricas globales
        print("\nMétricas globales (todo el conjunto de datos):")
        print(f"MSE (grados): {results['total']['mse_deg']:.6f}")
        
        # Métricas de entrenamiento y prueba
        if results['train'] is not None and results['test'] is not None:
            print("\nMétricas de entrenamiento:")
            print(f"MSE (radianes): {results['train']['mse_rad']:.6f}")
            
            print("\nMétricas de prueba:")
            print(f"MSE (radianes): {results['test']['mse_rad']:.6f}")
        
        # Error por articulación
        print("\nError por articulación (global):")
        for i, err in enumerate(results['error_por_artic']):
            print(f"  q{i}: {err:.6f}")
            
        # Información de MDS
        if self.stress is not None:
            print(f"\nMDS stress: {self.stress:.6f}")
            print(f"MDS stress normalizado: {self.stress_norm:.6f}")
        
        print("=" * 50) 
        
    def guardar(self, nombre_base="modelo_FNN"):
        """
        Guarda el modelo final entrenado, resultados de Optuna y el historial de pérdidas en la carpeta especificada.

        Parámetros:
            nombre_base (str): Nombre de la carpeta destino donde se guardarán todos los archivos.
                            Por defecto es "modelo_FNN".

        Acciones:
            - Crea la carpeta especificada si no existe.
            - Guarda el modelo, resultados y gráficas directamente en esa carpeta.
        """
        if self.modelo_final is None or self.study is None:
            raise RuntimeError("Optimice y entrene antes de guardar.")
        
        # Crear directorio si no existe
        os.makedirs(nombre_base, exist_ok=True)
        
        # Guardar modelo
        torch.save(self.modelo_final.state_dict(), f"{nombre_base}/fnn_modelo.pt")
        
        # Guardar resultados de Optuna
        self.study.trials_dataframe().to_csv(f"{nombre_base}/optuna_resultados.csv", index=False)
        
        # Guardar gráficas de Optuna
        vis.plot_optimization_history(self.study)
        plt.savefig(f"{nombre_base}/opt_history.png")
        plt.clf()
        
        vis.plot_param_importances(self.study)
        plt.savefig(f"{nombre_base}/opt_param_importance.png")
        plt.clf()
        
        # Guardar historial de pérdidas en CSV
        if len(self.train_losses) > 0:
            import pandas as pd
            loss_df = pd.DataFrame({'epoch': range(1, len(self.train_losses) + 1), 
                                'train_loss': self.train_losses})
            
            if len(self.val_losses) > 0:
                loss_df['val_loss'] = self.val_losses
                
            loss_df.to_csv(f"{nombre_base}/train_val_losses.csv", index=False)
            
            # Guardar gráfica de pérdidas
            plt.figure(figsize=(10, 6))
            plt.plot(self.train_losses, label='Entrenamiento')
            if len(self.val_losses) > 0:
                plt.plot(self.val_losses, label='Validación')
            plt.xlabel('Época')
            plt.ylabel('Pérdida (MSE)')
            plt.title('Evolución de la pérdida durante el entrenamiento')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{nombre_base}/loss_history.png")
            plt.close()
        
        print(f"Modelo y resultados guardados en carpeta: {nombre_base}")
        
        def predecir(self, p_xyz):
            """
            Realiza una predicción utilizando el modelo final entrenado.

            Parámetros
            ----------
            p_xyz : array-like
                Datos de entrada para la predicción. Debe ser convertible a un tensor de tipo float32.

            Retorna
            -------
            numpy.ndarray
                Resultado de la predicción realizado por el modelo.

            Lanza
            -----
            RuntimeError
                Si el modelo no ha sido entrenado previamente (modelo_final es None).

            Notas
            -----
            El método pone el modelo en modo evaluación y desactiva el cálculo de gradientes para realizar la predicción.
            """
            if self.modelo_final is None:
                raise RuntimeError("Modelo no entrenado.")
            self.modelo_final.eval()
            with torch.no_grad():
                x = torch.tensor(p_xyz, dtype=torch.float32).to(self.device)
                return self.modelo_final(x).cpu().numpy()

# Clase simplificada
class FNNInversaUR5Simple:
    """
    Red neuronal feedforward (FNN) para resolver la cinemática inversa de un robot UR5.
    Esta clase permite cargar datos de entrenamiento, construir y entrenar una red neuronal
    para aproximar la función inversa del robot UR5 (de posición a ángulos articulares).
    Incluye utilidades para evaluación, visualización de pérdidas y errores, y guardado del modelo.
    Parámetros
    ----------
    ruta_datos : str
        Ruta al archivo de datos (CSV o Excel) con entradas y salidas.
    hidden_layers : tuple, opcional
        Tamaños de las capas ocultas de la red (por defecto (128, 64)).
    dropout : float, opcional
        Proporción de dropout para regularización (por defecto 0.1).
    lr : float, opcional
        Tasa de aprendizaje para el optimizador (por defecto 1e-3).
    batch_size : int, opcional
        Tamaño de lote para entrenamiento (por defecto 64).
    epochs : int, opcional
        Número de épocas de entrenamiento (por defecto 100).
    test_size : float, opcional
        Proporción o cantidad de datos para validación (por defecto 0.2).
    random_state : int, opcional
        Semilla para la división de datos (por defecto 42).
    cols_excluir : list, opcional
        Lista de nombres de columnas a excluir de las entradas.
    Métodos
    -------
    entrenar():
        Entrena la red neuronal con los datos cargados.
    graficar_perdidas():
        Grafica la evolución de la pérdida durante el entrenamiento.
    graficar_offsets():
        Visualiza el error de predicción usando MDS.
    evaluar():
        Evalúa el modelo sobre todos los datos y muestra métricas de error.
    summary():
        Imprime un resumen de arquitectura, métricas y errores.
    guardar(nombre_base):
        Guarda los pesos del modelo entrenado en disco.
    predecir(p_xyz):
        Realiza una predicción de ángulos articulares para una entrada dada.
    """
    
    def __init__(self, ruta_datos, hidden_layers=(128, 64), dropout=0.1,
                 lr=1e-3, batch_size=64, epochs=100, test_size=0.2, 
                 random_state=42, cols_excluir=None):

        """
        Inicializa una instancia de la clase para entrenar una red neuronal feedforward (FNN) con PyTorch.
        
        Parámetros:
            ruta_datos (str): Ruta al archivo de datos que se utilizará para el entrenamiento y prueba.
            hidden_layers (tuple, opcional): Tupla que indica el número de neuronas en cada capa oculta. Por defecto es (128, 64).
            dropout (float, opcional): Tasa de abandono (dropout) para regularización. Por defecto es 0.1.
            lr (float, opcional): Tasa de aprendizaje para el optimizador. Por defecto es 1e-3.
            batch_size (int, opcional): Tamaño del lote para el entrenamiento. Por defecto es 64.
            epochs (int, opcional): Número de épocas de entrenamiento. Por defecto es 100.
            test_size (float, opcional): Proporción de los datos reservados para prueba. Por defecto es 0.2.
            random_state (int, opcional): Semilla para la aleatorización de los datos. Por defecto es 42.
            cols_excluir (list, opcional): Lista de nombres de columnas a excluir de los datos. Por defecto es None.
            
        Atributos:
            ruta (str): Ruta al archivo de datos.
            hidden_layers (tuple): Configuración de las capas ocultas.
            dropout (float): Tasa de abandono.
            lr (float): Tasa de aprendizaje.
            batch_size (int): Tamaño del lote.
            epochs (int): Número de épocas.
            test_size (float): Proporción de datos de prueba.
            random_state (int): Semilla de aleatorización.
            cols_excluir (list): Columnas excluidas.
            device (str): Dispositivo de cómputo utilizado ('cuda' o 'cpu').
            X (np.ndarray): Datos de entrada.
            Y (np.ndarray): Datos de salida.
            input_dim (int): Dimensión de entrada.
            output_dim (int): Dimensión de salida.
            modelo (_FNN): Modelo de red neuronal.
            train_losses (list): Historial de pérdidas de entrenamiento.
            val_losses (list): Historial de pérdidas de validación.
            stress (None): Atributo reservado para futuros cálculos de estrés.
            stress_norm (None): Atributo reservado para futuros cálculos de normalización de estrés.
        """

        
        self.ruta = ruta_datos
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.test_size = test_size
        self.random_state = random_state
        self.cols_excluir = cols_excluir or []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Entrenando en dispositivo: {self.device.upper()}")
        self.X, self.Y = self._cargar_datos()
        self.input_dim = self.X.shape[1]
        self.output_dim = self.Y.shape[1]
        self.modelo = _FNN(self.input_dim, self.output_dim, hidden_layers, dropout).to(self.device)
        self.train_losses = []
        self.val_losses = []
        self.stress = None
        self.stress_norm = None
        
    def _cargar_datos(self):
        """
        Carga los datos desde un archivo especificado en la ruta, detectando automáticamente las columnas de entrada y salida.
        Si el archivo es de tipo .xlsx, se utiliza pandas.read_excel; en caso contrario, se utiliza pandas.read_csv.
        Las columnas de entrada se determinan excluyendo aquellas que corresponden a las salidas (q0 a q9) y las especificadas en self.cols_excluir.
        Las columnas de salida se consideran aquellas con nombres 'q0' a 'q9' presentes en el archivo.
        Retorna:
            X (np.ndarray): Matriz de características de entrada, de tipo float32.
            Y (np.ndarray): Matriz de salidas, de tipo float32.
        """
        if self.ruta.endswith('.xlsx'):
            df = pd.read_excel(self.ruta)
        else:
            df = pd.read_csv(self.ruta)
            
        # Detectar automáticamente columnas de entrada y salida
        cols_entrada = [c for c in df.columns if c not in [f'q{i}' for i in range(10)] and c not in self.cols_excluir]
        cols_salida = [f'q{i}' for i in range(10) if f'q{i}' in df.columns]
        
        X = df[cols_entrada].to_numpy(dtype=np.float32)
        Y = df[cols_salida].to_numpy(dtype=np.float32)
        return X, Y
        
    def _dividir_datos(self):
        """
        Divide los datos en conjuntos de entrenamiento y prueba de manera representativa.
        Si `test_size` es menor o igual a 0, todos los datos se asignan al conjunto de entrenamiento y el conjunto de prueba queda vacío.
        Si `test_size` es mayor que 0 y hay más de 5000 muestras, usa train_test_split para una división aleatoria.
        Si hay menos de 5000 muestras, selecciona un subconjunto representativo utilizando distancias entre muestras.
        
        Returns:
            X_train (np.ndarray): Conjunto de características para entrenamiento.
            X_test (np.ndarray): Conjunto de características para prueba.
            Y_train (np.ndarray): Conjunto de etiquetas para entrenamiento.
            Y_test (np.ndarray): Conjunto de etiquetas para prueba.
        """
        if self.test_size <= 0:
            # Caso 1: No hay división (todo para entrenamiento)
            return self.X, np.empty((0, self.X.shape[1])), self.Y, np.empty((0, self.Y.shape[1]))
        
        # Caso 2: Si hay más de 5000 muestras, usar train_test_split
        if len(self.X) > 5000:
            from sklearn.model_selection import train_test_split
            X_train, X_test, Y_train, Y_test = train_test_split(
                self.X, self.Y, test_size=self.test_size, random_state=self.random_state
            )
            return X_train, X_test, Y_train, Y_test
        
        # Caso 3: Si hay menos de 5000 muestras, usar el método original
        from scipy.spatial.distance import pdist, squareform
        from .utils import select_m_representative_points
        
        n_train = int(len(self.X) * (1 - self.test_size))
        distances = squareform(pdist(self.X))
        train_idx = select_m_representative_points(distances, n_train)
        test_idx = list(set(range(len(self.X))) - set(train_idx))
        
        X_train, X_test = self.X[train_idx], self.X[test_idx]
        Y_train, Y_test = self.Y[train_idx], self.Y[test_idx]
        return X_train, X_test, Y_train, Y_test   
        
    def entrenar(self):
        """
        Entrena el modelo de red neuronal utilizando los datos de entrenamiento y validación.
        Este método divide los datos en conjuntos de entrenamiento y validación, configura el optimizador y la función de pérdida,
        y realiza el ciclo de entrenamiento durante el número de épocas especificado. Durante cada época, calcula la pérdida de entrenamiento
        y, si hay datos de validación, también calcula la pérdida de validación (MSE). Al final de cada 10 épocas (o en la última época),
        imprime el MSE de entrenamiento y validación. Almacena las pérdidas de entrenamiento y validación en los atributos `train_losses` y `val_losses`.
        Parámetros:
            No recibe parámetros directos, pero utiliza los atributos de la clase:
                - self.batch_size: Tamaño del batch para el entrenamiento.
                - self.lr: Tasa de aprendizaje para el optimizador.
                - self.epochs: Número de épocas de entrenamiento.
                - self.device: Dispositivo (CPU o GPU) donde se ejecuta el modelo.
                - self.modelo: Red neuronal a entrenar.
        Retorna:
            None. Actualiza los atributos `train_losses` y `val_losses` de la instancia.
        """
        Xtr, Xval, Ytr, Yval = self._dividir_datos()
        
        loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr)),
                           batch_size=self.batch_size, shuffle=True)
        optimizador = optim.Adam(self.modelo.parameters(), lr=self.lr)
        mse_fn = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for epoca in range(1, self.epochs + 1):
            self.modelo.train()
            epoch_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.modelo(xb)
                loss = mse_fn(pred, yb)
                epoch_loss += loss.item() * len(xb)
                optimizador.zero_grad()
                loss.backward()
                optimizador.step()
                
            train_losses.append(epoch_loss / len(Xtr))
            
            if len(Xval) > 0:
                self.modelo.eval()
                with torch.no_grad():
                    Y_pred = self.modelo(torch.tensor(Xval).to(self.device)).cpu().numpy()
                val_mse = mean_squared_error(Yval, Y_pred)
                val_losses.append(val_mse)
                
            if epoca % 10 == 0 or epoca == self.epochs:
                print(f"Época {epoca:3d}: MSE train = {train_losses[-1]:.6f}" + 
                     (f", MSE val = {val_losses[-1]:.6f}" if len(val_losses) > 0 else ""))
                
        self.train_losses = train_losses
        self.val_losses = val_losses
        
    def graficar_perdidas(self):
        """
        Genera una gráfica que muestra la evolución de la función de pérdida (MSE) durante el entrenamiento del modelo.

        La función grafica las pérdidas de entrenamiento almacenadas en `self.train_losses` y, si están disponibles,
        también las pérdidas de validación almacenadas en `self.val_losses`. El eje X representa las épocas y el eje Y
        representa el valor de la pérdida. Incluye leyenda, título y cuadrícula para facilitar la interpretación.

        Parámetros:
            No recibe parámetros externos. Utiliza los atributos internos `self.train_losses` y `self.val_losses`.

        Retorna:
            None. Muestra la gráfica en pantalla.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Entrenamiento')
        if len(self.val_losses) > 0:
            plt.plot(self.val_losses, label='Validación')
        plt.xlabel('Época')
        plt.ylabel('Pérdida (MSE)')
        plt.title('Evolución de la pérdida durante el entrenamiento')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def graficar_offsets(self, pmax=10000):
        """
        Genera una visualización MDS del error entre valores reales y predichos.
        Utiliza únicamente pmax puntos representativos seleccionados para aplicar MDS directamente.
            
        Parámetros:
            pmax (int): Número máximo de puntos a utilizar directamente en el cálculo MDS.
        """
        
        # Evaluación del modelo de manera más eficiente
        self.modelo.eval()
        with torch.no_grad():
            # Procesar todos los datos de una vez o en lotes grandes si es necesario
            Y_pred = self.modelo(torch.tensor(self.X, dtype=torch.float32).to(self.device)).cpu().numpy()
        
        # Calcular distancias sobre datos reales
        D = squareform(pdist(self.Y))
        
        # Seleccionar puntos representativos si es necesario
        if len(self.Y) > pmax:
            print(f"Usando {pmax} puntos representativos para MDS...")
            selected_indices = select_m_representative_points(D, pmax)
            Y_true = self.Y[selected_indices]
            Y_pred = Y_pred[selected_indices]
        else:
            selected_indices = np.arange(len(self.Y))
            Y_true = self.Y
            Y_pred = Y_pred
        
        # Concatenar y aplicar MDS con paralelización
        concat = np.vstack([Y_true, Y_pred])
        mds = MDS(n_components=2, dissimilarity='euclidean', random_state=0, n_jobs=-1)
        emb = mds.fit_transform(concat)
        y_true_2d = emb[:len(Y_true)]
        y_pred_2d = emb[len(Y_true):]
        
        # Guardar stress
        self.stress = mds.stress_
        self.stress_norm = self.stress / np.sum(pdist(concat)**2 / 2)
        
        # Visualización optimizada
        plt.figure(figsize=(10, 10), dpi=300)
        
        # Método vectorizado para dibujar líneas (más eficiente)
        plt.plot(
            [y_true_2d[:, 0], y_pred_2d[:, 0]],
            [y_true_2d[:, 1], y_pred_2d[:, 1]],
            'r-', alpha=0.2, linewidth=0.5
        )
        
        plt.scatter(y_true_2d[:, 0], y_true_2d[:, 1], c="blue", s=10, label="Verdaderos")
        plt.scatter(y_pred_2d[:, 0], y_pred_2d[:, 1], c="green", s=10, label="Predichos")
        plt.xlabel("MDS 1")
        plt.ylabel("MDS 2")
        plt.title(f"Proyección MDS del error ({len(Y_true)} puntos)\n"
                f"Stress: {self.stress_norm:.4f}")
        plt.legend()
        plt.grid(True)
        plt.show()
     
    def evaluar(self):
        """
        Evalúa el desempeño del modelo neuronal en los conjuntos de datos.
        Retorna:
            mse_rad (float): Error cuadrático medio total en radianes².
            mse_deg (float): Error cuadrático medio total en grados².
            r2 (float): Coeficiente de determinación R².
            error_por_artic (np.ndarray): Error cuadrático medio por articulación (vector).
            Y_pred (np.ndarray): Predicciones del modelo sobre los datos de entrada.
        Imprime:
            - MSE total en radianes².
            - MSE total en grados².
            - R².
        """
        # Evaluación sobre todo el conjunto de datos
        self.modelo.eval()
        with torch.no_grad():
            Y_pred_total = self.modelo(torch.tensor(self.X, dtype=torch.float32).to(self.device)).cpu().numpy()
            
        mse_rad_total = mean_squared_error(self.Y, Y_pred_total)
        mse_deg_total = mean_squared_error(np.rad2deg(self.Y), np.rad2deg(Y_pred_total))
        r2_total = r2_score(self.Y, Y_pred_total)
        
        # Error por articulación
        error_por_artic = np.mean((self.Y - Y_pred_total)**2, axis=0)
        
        # Resultados de entrenamiento y prueba si corresponde
        results = {
            'total': {
                'mse_rad': mse_rad_total,
                'mse_deg': mse_deg_total,
                'r2': r2_total,
                'Y_pred': Y_pred_total
            },
            'train': None,
            'test': None,
            'error_por_artic': error_por_artic
        }
        
        # Si hay división de datos, calcular métricas por separado
        if self.test_size > 0:
            X_train, X_test, Y_train, Y_test = self._dividir_datos()
            
            # Métricas de entrenamiento
            with torch.no_grad():
                Y_pred_train = self.modelo(torch.tensor(X_train, dtype=torch.float32).to(self.device)).cpu().numpy()
            
            mse_rad_train = mean_squared_error(Y_train, Y_pred_train)
            mse_deg_train = mean_squared_error(np.rad2deg(Y_train), np.rad2deg(Y_pred_train))
            r2_train = r2_score(Y_train, Y_pred_train)
            
            results['train'] = {
                'mse_rad': mse_rad_train,
                'mse_deg': mse_deg_train,
                'r2': r2_train,
                'Y_pred': Y_pred_train
            }
            
            # Métricas de prueba
            with torch.no_grad():
                Y_pred_test = self.modelo(torch.tensor(X_test, dtype=torch.float32).to(self.device)).cpu().numpy()
            
            mse_rad_test = mean_squared_error(Y_test, Y_pred_test)
            mse_deg_test = mean_squared_error(np.rad2deg(Y_test), np.rad2deg(Y_pred_test))
            r2_test = r2_score(Y_test, Y_pred_test)
            
            results['test'] = {
                'mse_rad': mse_rad_test,
                'mse_deg': mse_deg_test,
                'r2': r2_test,
                'Y_pred': Y_pred_test
            }
        
        # Imprimir solo las métricas globales
        print(f"MSE total (radianes²): {mse_rad_total:.6f}")
        print(f"MSE total (grados²):   {mse_deg_total:.6f}")
        print(f"R²: {r2_total:.6f}")
        
        return results

    def summary(self):
        """
        Muestra un resumen detallado del modelo FNN entrenado, incluyendo arquitectura,
        parámetros, métricas de desempeño globales, de entrenamiento y prueba,
        así como errores L1 y L2 por articulación en cada conjunto y
        métricas de MDS si están disponibles.
        """
        # Verificación del modelo (usamos self.modelo en vez de self.modelo_final)
        if self.modelo is None:
            raise RuntimeError("Modelo no entrenado.")
        
        # Evaluar el modelo y obtener todas las métricas
        results = self.evaluar()
        
        # Obtener predicciones para todos los conjuntos
        self.modelo.eval()
        with torch.no_grad():
            Y_pred_total = self.modelo(torch.tensor(self.X, dtype=torch.float32).to(self.device)).cpu().numpy()
        
        # Calcular errores L1 y L2 por articulación para el conjunto global
        error_L1_total = np.mean(np.abs(self.Y - Y_pred_total), axis=0)
        error_L2_total = np.sqrt(np.mean((self.Y - Y_pred_total)**2, axis=0))
        
        # Imprimir encabezado
        print("=" * 50)
        print("Modelo: FNN")
        print(f"Arquitectura: {self.input_dim}-{'-'.join(map(str, self.hidden_layers))}-{self.output_dim}")
        print(f"Parámetros: dropout={self.dropout}, lr={self.lr}, batch_size={self.batch_size}, epochs={self.epochs}")
        print(f"Opt: Simple")
        
        # Métricas globales
        print("\nMétricas globales (todo el conjunto de datos):")
        print(f"R²: {results['total']['r2']:.6f}")
        print(f"MSE (radianes): {results['total']['mse_rad']:.6f}")
        print(f"MSE (grados): {results['total']['mse_deg']:.6f}")
        
        # Error por articulación global (L1 y L2)
        print("\nError por articulación (global):")
        for i in range(len(error_L1_total)):
            print(f"  q{i}: L1={error_L1_total[i]:.6f}, L2={error_L2_total[i]:.6f}")
        
        # Métricas y errores por articulación para entrenamiento y prueba
        if results['train'] is not None and results['test'] is not None:
            # Obtener conjuntos de datos
            X_train, X_test, Y_train, Y_test = self._dividir_datos()
            
            # Predicciones para entrenamiento
            with torch.no_grad():
                Y_pred_train = self.modelo(torch.tensor(X_train, dtype=torch.float32).to(self.device)).cpu().numpy()
            
            # Errores L1 y L2 por articulación para entrenamiento
            error_L1_train = np.mean(np.abs(Y_train - Y_pred_train), axis=0)
            error_L2_train = np.sqrt(np.mean((Y_train - Y_pred_train)**2, axis=0))
            
            # Predicciones para prueba
            with torch.no_grad():
                Y_pred_test = self.modelo(torch.tensor(X_test, dtype=torch.float32).to(self.device)).cpu().numpy()
            
            # Errores L1 y L2 por articulación para prueba
            error_L1_test = np.mean(np.abs(Y_test - Y_pred_test), axis=0)
            error_L2_test = np.sqrt(np.mean((Y_test - Y_pred_test)**2, axis=0))
            
            # Métricas de entrenamiento
            print("\nMétricas de entrenamiento:")
            print(f"R²: {results['train']['r2']:.6f}")
            print(f"MSE (radianes): {results['train']['mse_rad']:.6f}")
            print(f"MSE (grados): {results['train']['mse_deg']:.6f}")
            
            # Error por articulación en entrenamiento
            print("\nError por articulación (entrenamiento):")
            for i in range(len(error_L1_train)):
                print(f"  q{i}: L1={error_L1_train[i]:.6f}, L2={error_L2_train[i]:.6f}")
            
            # Métricas de prueba
            print("\nMétricas de prueba:")
            print(f"R²: {results['test']['r2']:.6f}")
            print(f"MSE (radianes): {results['test']['mse_rad']:.6f}")
            print(f"MSE (grados): {results['test']['mse_deg']:.6f}")
            
            # Error por articulación en prueba
            print("\nError por articulación (prueba):")
            for i in range(len(error_L1_test)):
                print(f"  q{i}: L1={error_L1_test[i]:.6f}, L2={error_L2_test[i]:.6f}")
        
        # Información de MDS
        if self.stress is not None:
            print(f"\nMDS stress: {self.stress:.6f}")
            print(f"MDS stress normalizado: {self.stress_norm:.6f}")
        
        print("=" * 50)