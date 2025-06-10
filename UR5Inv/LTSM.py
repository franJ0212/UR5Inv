import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.distance import pdist, squareform
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from .utils import select_m_representative_points
import os
import optuna
import joblib
import optuna.visualization.matplotlib as vis
from sklearn.manifold import MDS


class _SeqDataset(Dataset):
    """
    Clase interna _SeqDataset para manejar conjuntos de datos secuenciales en PyTorch.
    Esta clase extiende Dataset y permite crear secuencias de longitud fija a partir de datos de entrada (X) y etiquetas (Y).
    Es útil para tareas de series temporales o modelos que requieren entradas secuenciales, como LSTM o RNN.
    Atributos:
        X (np.ndarray): Matriz de características de entrada, convertida a float32.
        Y (np.ndarray): Vector de etiquetas o valores objetivo, convertido a float32.
        seq_len (int): Longitud de la secuencia que se utilizará como entrada.
        valid (np.ndarray): Índices válidos para generar secuencias completas de longitud seq_len.
    Métodos:
        __len__(): Devuelve el número de secuencias válidas que se pueden extraer del conjunto de datos.
        __getitem__(idx): Devuelve una tupla (x_seq, y_t) donde x_seq es una secuencia de longitud seq_len y y_t es el valor objetivo correspondiente al último elemento de la secuencia.
    Parámetros:
        X (np.ndarray): Datos de entrada.
        Y (np.ndarray): Etiquetas o valores objetivo.
        seq_len (int): Longitud de la secuencia.
    Ejemplo de uso:
        dataset = _SeqDataset(X, Y, seq_len=10)
        x_seq, y_t = dataset[0]
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, seq_len: int):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.seq_len = seq_len
        self.valid = np.arange(seq_len - 1, len(X))

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        end = self.valid[idx] + 1
        start = end - self.seq_len
        x_seq = self.X[start:end]
        y_t = self.Y[end - 1]
        return torch.tensor(x_seq), torch.tensor(y_t)

class _LSTM(nn.Module):
    """
    Clase _LSTM que implementa una red neuronal LSTM para tareas de modelado secuencial.
    Parámetros
    ----------
    input_dim : int
        Dimensión de entrada de las secuencias.
    output_dim : int
        Dimensión de salida de la red (número de clases o variables de salida).
    hidden_size : int
        Número de unidades ocultas en cada capa LSTM.
    num_layers : int
        Número de capas LSTM apiladas.
    dropout : float
        Tasa de abandono (dropout) aplicada entre las capas LSTM.
    Métodos
    -------
    forward(x)
        Realiza una pasada hacia adelante de la entrada `x` a través de la red LSTM y la capa lineal final.
        Devuelve la salida correspondiente a la última posición temporal de la secuencia.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1]
        return self.fc(last)

class LSTMInversaUR5Simple:
    """"
    Clase para modelar la cinemática inversa de un robot UR5 usando una red LSTM simple.
    
        ruta_datos (str): Ruta al archivo de datos (CSV o Excel).
        seq_len (int): Longitud de la secuencia de entrada para la LSTM.
        hidden_size (int): Número de unidades ocultas en la LSTM.
        num_layers (int): Número de capas LSTM.
        dropout (float): Tasa de dropout en la LSTM.
        lr (float): Tasa de aprendizaje para el optimizador.
        batch_size (int): Tamaño del batch para entrenamiento.
        epochs (int): Número de épocas de entrenamiento.
        test_size (float): Proporción del conjunto de datos para validación.
        random_state (int): Semilla para la aleatoriedad.
        cols_excluir (List[str], opcional): Columnas a excluir de las entradas.
        
    Métodos principales:
        entrenar(): Entrena el modelo LSTM con los datos proporcionados.
        graficar_perdidas(): Grafica la evolución de la pérdida durante el entrenamiento.
        graficar_offsets(pmax): Visualiza los errores de predicción usando MDS.
        summary(): Muestra un resumen de desempeño y arquitectura del modelo.
        evaluar(): Calcula y muestra métricas globales de desempeño.
        predecir(fila_actual): Predice los valores articulares dados nuevos datos de entrada.
        guardar(nombre_base): Guarda el modelo entrenado en disco.
    """

    POS_COLS = ["X", "Y", "Z", "RX", "RY", "RZ"]
    Q_COLS = [f"q{i}" for i in range(6)]

    def __init__(self,
                 ruta_datos: str,
                 seq_len: int = 20,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 lr: float = 1e-3,
                 batch_size: int = 64,
                 epochs: int = 100,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 cols_excluir: Optional[List[str]] = None):

        """
        Inicializa una instancia de la clase para el modelo LSTM aplicado a series temporales.
        
        Parámetros:
            ruta_datos (str): Ruta al archivo de datos de entrada.
            seq_len (int, opcional): Longitud de la secuencia de entrada para el modelo LSTM. Por defecto es 20.
            hidden_size (int, opcional): Número de unidades ocultas en cada capa LSTM. Por defecto es 128.
            num_layers (int, opcional): Número de capas LSTM apiladas. Por defecto es 2.
            dropout (float, opcional): Tasa de abandono (dropout) entre capas LSTM. Por defecto es 0.1.
            lr (float, opcional): Tasa de aprendizaje para el optimizador. Por defecto es 1e-3.
            batch_size (int, opcional): Tamaño del lote para el entrenamiento. Por defecto es 64.
            epochs (int, opcional): Número de épocas de entrenamiento. Por defecto es 100.
            test_size (float, opcional): Proporción de los datos reservados para prueba. Por defecto es 0.2.
            random_state (int, opcional): Semilla para la aleatorización de los datos. Por defecto es 42.
            cols_excluir (Optional[List[str]], opcional): Lista de nombres de columnas a excluir de los datos de entrada. Por defecto es None. 
        """
        self.ruta = ruta_datos
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.test_size = test_size
        self.random_state = random_state
        self.cols_excluir = cols_excluir or []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Dispositivo: {self.device.upper()}")
        self.X, self.Y = self._cargar_datos()
        self.input_dim = self.X.shape[1]
        self.output_dim = self.Y.shape[1]

        self.model = _LSTM(self.input_dim, self.output_dim,
                           hidden_size, num_layers, dropout).to(self.device)

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def _cargar_dataframe(self) -> pd.DataFrame:
        """
        Carga un archivo de datos en un DataFrame de pandas.

        Dependiendo de la extensión del archivo especificado en `self.ruta`, 
        utiliza `pd.read_excel` para archivos con extensión `.xlsx` o 
        `pd.read_csv` para otros tipos de archivos.

        Returns:
            pd.DataFrame: El DataFrame cargado desde el archivo especificado.
        """
        if self.ruta.lower().endswith(".xlsx"):
            return pd.read_excel(self.ruta)
        return pd.read_csv(self.ruta)

    def _cargar_datos(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga y procesa los datos desde un DataFrame, verificando la existencia de las columnas requeridas
        y separando las variables de entrada (X) y salida (Y).
        Retorna:
            Tuple[np.ndarray, np.ndarray]: Una tupla que contiene:
                - X: Arreglo NumPy de tipo float32 con las variables de entrada (posiciones y covariables).
                - Y: Arreglo NumPy de tipo float32 con las variables objetivo (Q_COLS).
        Lanza:
            ValueError: Si faltan columnas requeridas en el DataFrame.
        """
        df = self._cargar_dataframe()

        faltantes = [c for c in self.POS_COLS + self.Q_COLS if c not in df.columns]
        if faltantes:
            raise ValueError(f"Columnas faltantes en archivo: {faltantes}")

        base_in = self.POS_COLS.copy()
        covars = [c for c in df.columns if c not in base_in + self.Q_COLS + self.cols_excluir]
        cols_in = base_in + covars

        X = df[cols_in].to_numpy(dtype=np.float32)
        Y = df[self.Q_COLS].to_numpy(dtype=np.float32)
        return X, Y

    def _split_indices(self):
        """
        Divide los índices de la serie temporal en conjuntos de entrenamiento y validación.

        Calcula el número total de secuencias posibles según la longitud de la secuencia (`seq_len`),
        baraja los índices de forma aleatoria utilizando la semilla `random_state` y separa los índices
        en dos conjuntos: uno para entrenamiento y otro para validación, según la proporción definida
        por `test_size`.

        Returns:
            tuple: Una tupla (train_idx, val_idx) donde:
                - train_idx (np.ndarray): Índices para el conjunto de entrenamiento.
                - val_idx (np.ndarray): Índices para el conjunto de validación.
        """
        n_total = len(self.X) - (self.seq_len - 1)
        rng = np.random.default_rng(self.random_state)
        idx = np.arange(n_total)
        rng.shuffle(idx)
        n_val = int(self.test_size * n_total)
        return idx[n_val:], idx[:n_val]

    def _prepare_loaders(self):
        """
        Prepara y retorna los DataLoaders para entrenamiento y validación a partir de los datos de entrada.

        Si `test_size` es menor o igual a 0, retorna solo un DataLoader para entrenamiento con todos los datos.
        Si `test_size` es mayor que 0, divide los datos en conjuntos de entrenamiento y validación según los índices generados por `_split_indices()`,
        y retorna un DataLoader para cada conjunto.

        Retorna:
            tuple: (loader_train, loader_val)
                - loader_train (DataLoader): DataLoader para el conjunto de entrenamiento.
                - loader_val (DataLoader o None): DataLoader para el conjunto de validación, o None si no se realiza partición.
        """
        dataset = _SeqDataset(self.X, self.Y, self.seq_len)
        if self.test_size <= 0:
            loader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=True, drop_last=True)
            return loader, None
        train_idx, val_idx = self._split_indices()
        train_ds = torch.utils.data.Subset(dataset, train_idx)
        val_ds = torch.utils.data.Subset(dataset, val_idx)
        loader_train = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True, drop_last=True)
        loader_val = DataLoader(val_ds, batch_size=self.batch_size,
                                shuffle=False)
        return loader_train, loader_val

    def entrenar(self):
        """
        Entrena el modelo utilizando el conjunto de datos de entrenamiento y, opcionalmente, valida el desempeño en un conjunto de validación.
        Durante cada época, el método realiza los siguientes pasos:
        - Ejecuta el ciclo de entrenamiento sobre los lotes del conjunto de entrenamiento, calculando la pérdida MSE y actualizando los parámetros del modelo mediante el optimizador Adam.
        - Calcula y almacena la pérdida promedio de entrenamiento por época.
        - Si se proporciona un conjunto de validación, evalúa el modelo en este conjunto sin actualizar los parámetros y almacena la pérdida promedio de validación.
        - Imprime el progreso cada 10 épocas o al finalizar el entrenamiento, mostrando la pérdida de entrenamiento y, si corresponde, la de validación.
        Parámetros:
            No recibe parámetros adicionales.
        Efectos secundarios:
            - Actualiza los pesos del modelo.
            - Almacena las pérdidas de entrenamiento y validación en las listas `self.train_losses` y `self.val_losses`.
            - Imprime el progreso del entrenamiento en consola.
        """
        loader_train, loader_val = self._prepare_loaders()
        mse_fn = nn.MSELoss()
        opt = optim.Adam(self.model.parameters(), lr=self.lr)

        for ep in range(1, self.epochs + 1):
            self.model.train()
            running = 0.0
            for xb, yb in loader_train:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = self.model(xb)
                loss = mse_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.item() * len(xb)
            train_loss = running / len(loader_train.dataset)
            self.train_losses.append(train_loss)

            if loader_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss_acc = 0.0
                    for xv, yv in loader_val:
                        xv = xv.to(self.device)
                        yv = yv.to(self.device)
                        pv = self.model(xv)
                        val_loss_acc += mse_fn(pv, yv).item() * len(xv)
                val_loss = val_loss_acc / len(loader_val.dataset)
                self.val_losses.append(val_loss)
            if ep % 10 == 0 or ep == self.epochs:
                msg = f"Época {ep:3d}: MSE train = {train_loss:.6f}"
                if loader_val is not None:
                    msg += f", MSE val = {val_loss:.6f}"
                print(msg)

    def graficar_perdidas(self):
        """
        Genera una gráfica que muestra la evolución de la función de pérdida (MSE) durante el entrenamiento del modelo.

        La función grafica las pérdidas de entrenamiento almacenadas en `self.train_losses` y, si están disponibles,
        también las pérdidas de validación almacenadas en `self.val_losses`. La gráfica incluye etiquetas, título,
        leyenda y una cuadrícula para facilitar la interpretación visual del desempeño del modelo a lo largo de las épocas.

        Parámetros:
            No recibe parámetros externos. Utiliza los atributos internos `self.train_losses` y `self.val_losses`.

        Retorna:
            None: La función solo muestra la gráfica y no retorna ningún valor.
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
        Grafica los offsets entre valores verdaderos y predicciones e imprime métricas de stress.
        """
        # Evaluación del modelo
        dataset = _SeqDataset(self.X, self.Y, self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        Y_pred, Y_true = [], []
        self.model.eval()
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                preds = self.model(xb).cpu().numpy()
                Y_pred.append(preds)
                Y_true.append(yb.numpy())
        Y_pred = np.vstack(Y_pred)
        Y_true = np.vstack(Y_true)
        
        # Muestreo con el método original
        n = len(Y_true)
        if pmax < n:
            D = squareform(pdist(Y_true))
            idx = select_m_representative_points(D, pmax)
            Y_true = Y_true[idx]
            Y_pred = Y_pred[idx]
        
        # MDS optimizado
        concat = np.vstack([Y_true, Y_pred])
        mds = MDS(n_components=2, dissimilarity='euclidean', random_state=0, n_jobs=-1)
        embed = mds.fit_transform(concat)
        
        # Imprimir métricas de stress
        stress = mds.stress_
        stress_normalizado = stress / (np.sum(pdist(concat)**2) / 2)
        print(f"Stress del MDS: {stress:.4f}")
        print(f"Stress normalizado: {stress_normalizado:.4f}")
        
        # Visualización optimizada
        yt, yp = embed[:len(Y_true)], embed[len(Y_true):]
        plt.figure(figsize=(10, 10))
        
        # Método rápido de dibujo de líneas
        plt.plot(
            [yt[:, 0], yp[:, 0]],
            [yt[:, 1], yp[:, 1]],
            'r-', alpha=0.2, linewidth=0.5
        )
        
        plt.scatter(yt[:, 0], yt[:, 1], c='blue', s=10, label='Verdaderos')
        plt.scatter(yp[:, 0], yp[:, 1], c='green', s=10, label='Predicciones')
        plt.title(f"Proyección MDS de los errores (stress={stress_normalizado:.4f})")
        plt.legend()
        plt.grid(True)
        plt.show()

    def summary(self):
        """
        Genera un resumen de la evaluación del modelo LSTM sobre el conjunto de datos completo, así como métricas específicas para los subconjuntos de entrenamiento y prueba si corresponde.
        Este método calcula y muestra en consola las siguientes métricas:
        - Error cuadrático medio (MSE) en radianes y grados para todo el conjunto de datos.
        - Coeficiente de determinación R² para todo el conjunto de datos.
        - Error cuadrático medio por articulación.
        - Arquitectura y parámetros del modelo.
        - Si se ha definido un conjunto de prueba, también muestra métricas de entrenamiento y prueba por separado.
        Parámetros:
            No recibe parámetros adicionales.
        Notas:
            - Utiliza el modelo entrenado y los datos almacenados en la instancia.
            - Requiere que los atributos `self.X`, `self.Y`, `self.seq_len`, `self.batch_size`, `self.device`, `self.input_dim`, 
              `self.hidden_size`, `self.num_layers`, `self.output_dim`, `self.dropout`, `self.lr`, `self.epochs`, y `self.test_size` 
              estén correctamente definidos.
            - Imprime los resultados en consola.
        """
        # Evaluación completa
        dataset = _SeqDataset(self.X, self.Y, self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                preds.append(self.model(xb).cpu().numpy())
                trues.append(yb.cpu().numpy())
        Y_pred = np.vstack(preds)
        Y_true = np.vstack(trues)

        mse_rad_total = mean_squared_error(Y_true, Y_pred)
        mse_deg_total = mean_squared_error(np.rad2deg(Y_true), np.rad2deg(Y_pred))
        r2_total = r2_score(Y_true, Y_pred)
        error_por_artic = np.mean((Y_true - Y_pred)**2, axis=0)

        print("=" * 50)
        print(f"MSE total (radianes²): {mse_rad_total:.6f}")
        print(f"MSE total (grados²):   {mse_deg_total:.6f}")
        print(f"R²: {r2_total:.6f}")
        print("=" * 50)
        print("Modelo: LSTM")
        print(f"Arquitectura: Input({self.input_dim}) -> LSTM({self.hidden_size}) x {self.num_layers} -> Output({self.output_dim})")
        print(f"Parámetros: dropout={self.dropout}, lr={self.lr}, batch_size={self.batch_size}, epochs={self.epochs}")
        print(f"Opt: Simple")

        if self.test_size > 0:
            train_idx, test_idx = self._split_indices()
            dataset = _SeqDataset(self.X, self.Y, self.seq_len)
            loader_train = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=self.batch_size)
            loader_test = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=self.batch_size)

            def calcular_métricas(loader):
                ps, ys = [], []
                with torch.no_grad():
                    for xb, yb in loader:
                        xb = xb.to(self.device)
                        ps.append(self.model(xb).cpu().numpy())
                        ys.append(yb.numpy())
                P = np.vstack(ps)
                Y = np.vstack(ys)
                return {
                    'mse_rad': mean_squared_error(Y, P),
                    'mse_deg': mean_squared_error(np.rad2deg(Y), np.rad2deg(P)),
                    'r2': r2_score(Y, P)
                }

            r_train = calcular_métricas(loader_train)
            r_test = calcular_métricas(loader_test)

            print("\nMétricas globales (todo el conjunto de datos):")
            print(f"R²: {r2_total:.6f}")
            print(f"MSE (radianes): {mse_rad_total:.6f}")
            print(f"MSE (grados): {mse_deg_total:.6f}")

            print("\nMétricas de entrenamiento:")
            print(f"R²: {r_train['r2']:.6f}")
            print(f"MSE (radianes): {r_train['mse_rad']:.6f}")
            print(f"MSE (grados): {r_train['mse_deg']:.6f}")

            print("\nMétricas de prueba:")
            print(f"R²: {r_test['r2']:.6f}")
            print(f"MSE (radianes): {r_test['mse_rad']:.6f}")
            print(f"MSE (grados): {r_test['mse_deg']:.6f}")

        print("\nError por articulación (global):")
        for i, err in enumerate(error_por_artic):
            print(f"  q{i}: {err:.6f}")
        print("=" * 50)

    def _predict_numpy(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Realiza una predicción utilizando el modelo entrenado a partir de una secuencia de entrada en formato numpy.

        Parámetros
        ----------
        X_seq : np.ndarray
            Secuencia de entrada con las características para la predicción. Debe ser un arreglo de numpy.

        Retorna
        -------
        np.ndarray
            Predicción generada por el modelo, convertida a un arreglo de numpy y ajustada en sus dimensiones.
        """
        self.model.eval()
        X_t = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y_pred = self.model(X_t).cpu().numpy().squeeze()
        return y_pred

    def evaluar(self):
        """
        Evalúa el desempeño del modelo en el conjunto de datos completo utilizando MSE y R².

        Este método crea un DataLoader a partir de los datos de entrada y salida, realiza la inferencia
        del modelo en modo evaluación (sin gradientes), y calcula las métricas de error cuadrático medio (MSE)
        y coeficiente de determinación (R²) entre las predicciones y los valores reales.

        Imprime los valores globales de MSE y R² en consola.

        Returns:
            tuple: Una tupla (mse, r2) donde mse es el error cuadrático medio y r2 es el coeficiente de determinación.
        """
        dataset = _SeqDataset(self.X, self.Y, self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                preds.append(self.model(xb).cpu().numpy())
                trues.append(yb.cpu().numpy())
        Y_pred = np.vstack(preds)
        Y_true = np.vstack(trues)
        mse = mean_squared_error(Y_true, Y_pred)
        r2 = r2_score(Y_true, Y_pred)
        print(f"MSE global: {mse:.6f}\nR² global: {r2:.6f}")
        return mse, r2

    def predecir(self, fila_actual: pd.Series) -> np.ndarray:
        """
        Predice el siguiente valor de la secuencia utilizando el modelo LSTM.

        Parámetros
        ----------
        fila_actual : pd.Series
            Serie de pandas que contiene los valores actuales de las variables de entrada.

        Retorna
        -------
        np.ndarray
            Arreglo de numpy con la predicción generada por el modelo para la secuencia actualizada.

        Notas
        -----
        - La función selecciona las columnas relevantes de la fila actual, las convierte a un arreglo numpy,
          y las añade a la secuencia de entrada.
        - Se utiliza la longitud de secuencia (`seq_len`) para formar la entrada al modelo.
        - La predicción se realiza llamando al método interno `_predict_numpy`.
        """
        new_x = fila_actual[self.POS_COLS + [c for c in fila_actual.index if c not in self.POS_COLS + self.Q_COLS + self.cols_excluir]].to_numpy(dtype=np.float32)
        self.X = np.vstack([self.X, new_x])
        seq = self.X[-self.seq_len:]
        return self._predict_numpy(seq)

    def guardar(self, nombre_base: str = "modelo_lstm_ur5.pth"):
        """
        Guarda el estado actual del modelo LSTM en un archivo.

        Parámetros:
            nombre_base (str): Nombre del archivo donde se guardará el modelo. 
                               Por defecto es "modelo_lstm_ur5.pth".

        El método utiliza torch.save para almacenar los parámetros del modelo en el archivo especificado
        y muestra un mensaje indicando la ubicación del archivo guardado.
        """
        torch.save(self.model.state_dict(), nombre_base)
        print(f"Modelo guardado en {nombre_base}")

class LSTMInversaUR5:
    """
    Clase para entrenar y evaluar un modelo LSTM para la cinemática inversa del robot UR5.
    Permite cargar datos, optimizar hiperparámetros con Optuna, entrenar el modelo,
    evaluar su desempeño y visualizar resultados.
    
    Parámetros:
        ruta_datos (str): Ruta al archivo de datos (CSV o Excel).
        test_size (float): Proporción de datos para validación/test.
        random_state (int): Semilla para la aleatorización.
        dispositivo (str): Dispositivo de cómputo ('cuda' o 'cpu').
        cols_excluir (list): Lista de columnas a excluir de las entradas.
    """
    
    POS_COLS = ["X", "Y", "Z", "RX", "RY", "RZ"]
    Q_COLS = [f"q{i}" for i in range(6)]
    
    def __init__(self, ruta_datos: str, test_size: float = 0.2, 
                 random_state: int = 42, dispositivo: Optional[str] = None, 
                 cols_excluir: Optional[List[str]] = None):
        
        self.ruta = ruta_datos
        self.test_size = test_size
        self.random_state = random_state
        self.device = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cols_excluir = cols_excluir or []
        
        print(f"Entrenando en dispositivo: {self.device.upper()}")
        self.X, self.Y = self._cargar_datos()
        self.input_dim = self.X.shape[1]
        self.output_dim = self.Y.shape[1]
        self.train_losses = []
        self.val_losses = []
        self.study = None
        self.modelo_final = None
        self.stress = None
        self.stress_norm = None
        
    def _cargar_datos(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga y procesa los datos desde el archivo especificado.
        
        Retorna:
            Tuple[np.ndarray, np.ndarray]: Una tupla con los datos de entrada (X) y salida (Y).
        """
        if self.ruta.endswith('.xlsx'):
            df = pd.read_excel(self.ruta)
        else:
            df = pd.read_csv(self.ruta)
            
        # Verificar columnas requeridas
        faltantes = [c for c in self.POS_COLS + self.Q_COLS if c not in df.columns]
        if faltantes:
            raise ValueError(f"Columnas faltantes en archivo: {faltantes}")
        
        # Seleccionar columnas de entrada y salida
        base_in = self.POS_COLS.copy()
        covars = [c for c in df.columns if c not in base_in + self.Q_COLS + self.cols_excluir]
        cols_in = base_in + covars
        
        X = df[cols_in].to_numpy(dtype=np.float32)
        Y = df[self.Q_COLS].to_numpy(dtype=np.float32)
        return X, Y
        
    def _espacio(self, t) -> Dict[str, Any]:
        """
        Define el espacio de búsqueda para la optimización de hiperparámetros.
        
        Parámetros:
            t: Un objeto trial de Optuna.
            
        Retorna:
            Dict[str, Any]: Un diccionario con los hiperparámetros sugeridos.
        """
        hidden_options = {
            "64": 64,
            "128": 128,
            "256": 256,
            "512": 512
        }
        
        key = t.suggest_categorical("hidden_key", list(hidden_options.keys()))
        
        return {
            "hidden_size": hidden_options[key],
            "num_layers": t.suggest_int("num_layers", 1, 3),
            "seq_len": t.suggest_int("seq_len", 1, 30),
            "dropout": t.suggest_float("dropout", 0.0, 0.4),
            "lr": t.suggest_float("lr", 1e-3, 5e-3, log=True),
            "batch": t.suggest_categorical("batch", [32, 64, 128]),
            "epochs": t.suggest_int("epochs", 50, 300)
        }
    
    def _dividir_datos(self):
        """
        Divide los datos en conjuntos de entrenamiento y prueba.
        
        Retorna:
            Tuple: Índices para los conjuntos de entrenamiento y prueba.
        """
        n_total = len(self.X)
        if self.test_size <= 0:
            return np.arange(n_total), np.array([], dtype=int)
        
        if n_total > 5000:
            # Para conjuntos grandes, división aleatoria simple
            rng = np.random.RandomState(self.random_state)
            indices = np.arange(n_total)
            rng.shuffle(indices)
            n_val = int(self.test_size * n_total)
            return indices[n_val:], indices[:n_val]
        else:
            # Para conjuntos pequeños, selección representativa
            distances = squareform(pdist(self.X))
            n_train = int(n_total * (1 - self.test_size))
            train_idx = select_m_representative_points(distances, n_train)
            test_idx = list(set(range(n_total)) - set(train_idx))
            return np.array(train_idx), np.array(test_idx)
    
    def _prepare_dataset(self, seq_len: int, idx_train=None, idx_val=None):
        """
        Prepara los DataLoaders para entrenamiento y validación.
        
        Parámetros:
            seq_len (int): Longitud de la secuencia.
            idx_train (np.ndarray): Índices para entrenamiento.
            idx_val (np.ndarray): Índices para validación.
            
        Retorna:
            Tuple: DataLoaders para entrenamiento y validación.
        """
        dataset = _SeqDataset(self.X, self.Y, seq_len)
        n_valid = len(dataset)
        
        if idx_train is None or idx_val is None:
            valid_indices = np.arange(n_valid)
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(valid_indices)
            n_val = int(self.test_size * n_valid)
            idx_train, idx_val = valid_indices[n_val:], valid_indices[:n_val]
        
        # Ajustar para secuencias válidas
        # Solo considerar índices dentro del rango válido para secuencias
        valid_seq_indices = dataset.valid
        idx_train = np.array([i for i in idx_train if i < n_valid and i in valid_seq_indices])
        idx_val = np.array([i for i in idx_val if i < n_valid and i in valid_seq_indices])
        
        if len(idx_val) == 0:
            dataset_train = dataset
            loader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
            return loader_train, None, idx_train, idx_val
        
        dataset_train = Subset(dataset, idx_train)
        dataset_val = Subset(dataset, idx_val)
        
        loader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
        loader_val = DataLoader(dataset_val, batch_size=64, shuffle=False)
        
        return loader_train, loader_val, idx_train, idx_val
    
    def _train_once(self, p, idx_train, idx_val):
        """
        Entrena el modelo una vez con los parámetros dados.
        
        Parámetros:
            p (Dict[str, Any]): Diccionario de hiperparámetros.
            idx_train (np.ndarray): Índices para entrenamiento.
            idx_val (np.ndarray): Índices para validación.
            
        Retorna:
            Tuple: MSE, modelo entrenado y predicciones.
        """
        # Crear modelo
        model = _LSTM(
            self.input_dim, 
            self.output_dim, 
            p["hidden_size"], 
            p["num_layers"], 
            p["dropout"]
        ).to(self.device)
        
        # Preparar dataset y loaders
        loader_train, loader_val, _, _ = self._prepare_dataset(
            p["seq_len"], idx_train, idx_val
        )
        
        # Optimizador y función de pérdida
        opt = optim.Adam(model.parameters(), lr=p["lr"])
        mse_fn = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        # Entrenamiento
        for epoch in range(p["epochs"]):
            model.train()
            epoch_loss = 0
            total_samples = 0
            
            for xb, yb in loader_train:
                xb, yb = xb.to(self.device), yb.to(self.device)
                loss = mse_fn(model(xb), yb)
                epoch_loss += loss.item() * len(xb)
                total_samples += len(xb)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            train_loss = epoch_loss / total_samples
            train_losses.append(train_loss)
            
            # Validación
            if loader_val is not None:
                model.eval()
                val_loss = 0
                val_samples = 0
                Y_pred = []
                Y_val = []
                
                with torch.no_grad():
                    for xv, yv in loader_val:
                        xv, yv = xv.to(self.device), yv.to(self.device)
                        pred = model(xv)
                        val_loss += mse_fn(pred, yv).item() * len(xv)
                        val_samples += len(xv)
                        Y_pred.append(pred.cpu().numpy())
                        Y_val.append(yv.cpu().numpy())
                
                val_loss = val_loss / val_samples
                val_losses.append(val_loss)
        
        # Guardar pérdidas
        self.train_losses = train_losses
        self.val_losses = val_losses
        
        # Evaluación final
        model.eval()
        with torch.no_grad():
            if loader_val is not None:
                preds, trues = [], []
                for xv, yv in loader_val:
                    xv = xv.to(self.device)
                    preds.append(model(xv).cpu().numpy())
                    trues.append(yv.numpy())
                    
                Y_pred = np.vstack(preds)
                Y_val = np.vstack(trues)
                
                mse = mean_squared_error(np.rad2deg(Y_val), np.rad2deg(Y_pred))
            else:
                # Si no hay datos de validación, evaluar en entrenamiento
                preds, trues = [], []
                for xb, yb in loader_train:
                    xb = xb.to(self.device)
                    preds.append(model(xb).cpu().numpy())
                    trues.append(yb.numpy())
                
                Y_pred = np.vstack(preds)
                Y_train = np.vstack(trues)
                
                mse = mean_squared_error(np.rad2deg(Y_train), np.rad2deg(Y_pred))
                Y_val = Y_train
                
        return mse, model, Y_pred
        
    def _objetivo(self, trial):
        """
        Función objetivo para la optimización de hiperparámetros.
        
        Parámetros:
            trial: Objeto trial de Optuna.
            
        Retorna:
            float: Negativo del MSE (para maximizar).
        """
        p = self._espacio(trial)
        idx_train, idx_val = self._dividir_datos()
        mse, _, _ = self._train_once(p, idx_train, idx_val)
        return -mse  # Maximizar el negativo del MSE (minimizar MSE)
        
    def optimizar(self, n_trials=30, nombre_est="estudio_ur5_lstm"):
        """
        Optimiza los hiperparámetros del modelo utilizando Optuna.
        
        Parámetros:
            n_trials (int): Número de pruebas.
            nombre_est (str): Nombre del estudio.
        """
        self.study = optuna.create_study(direction="maximize", study_name=nombre_est)
        self.study.optimize(self._objetivo, n_trials=n_trials, n_jobs=1)
        
    def entrenar_mejor_modelo(self):
        """
        Entrena el modelo con los mejores hiperparámetros encontrados.
        
        Retorna:
            float: MSE en grados².
        """
        if self.study is None:
            raise RuntimeError("Ejecute optimizar() primero.")
            
        p = {**self._espacio(self.study.best_trial), **self.study.best_params}
        idx_train, idx_val = self._dividir_datos()
        mse, model, _ = self._train_once(p, idx_train, idx_val)
        self.modelo_final = model
        print(f"MSE grados² validación: {mse:.6f}")
        return mse
        
    def graficar_perdidas(self):
        """
        Grafica las pérdidas durante el entrenamiento.
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
        Visualiza el error entre valores reales y predichos usando MDS.
        
        Parámetros:
            pmax (int): Número máximo de puntos para MDS.
        """
        if self.modelo_final is None:
            raise RuntimeError("Modelo no entrenado.")
        
        # Obtener los mejores parámetros
        p = {**self._espacio(self.study.best_trial), **self.study.best_params}
        
        # Crear dataset con la secuencia óptima
        dataset = _SeqDataset(self.X, self.Y, p["seq_len"])
        loader = DataLoader(dataset, batch_size=p["batch"], shuffle=False)
        
        # Evaluación del modelo
        Y_pred, Y_true = [], []
        self.modelo_final.eval()
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                preds = self.modelo_final(xb).cpu().numpy()
                Y_pred.append(preds)
                Y_true.append(yb.numpy())
        
        Y_pred = np.vstack(Y_pred)
        Y_true = np.vstack(Y_true)
        
        # Si hay más puntos que pmax, seleccionar muestra representativa
        if len(Y_true) > pmax:
            print(f"Usando {pmax} puntos representativos para MDS...")
            distances = squareform(pdist(Y_true))
            selected_indices = select_m_representative_points(distances, pmax)
            Y_true = Y_true[selected_indices]
            Y_pred = Y_pred[selected_indices]
        
        # Aplicar MDS
        concat = np.vstack([Y_true, Y_pred])
        mds = MDS(n_components=2, dissimilarity='euclidean', random_state=0, n_jobs=-1)
        emb = mds.fit_transform(concat)
        y_true_2d = emb[:len(Y_true)]
        y_pred_2d = emb[len(Y_true):]
        
        # Calcular stress normalizado
        self.stress = mds.stress_
        self.stress_norm = self.stress / np.sum(pdist(concat) ** 2 / 2)
        
        # Visualización
        plt.figure(figsize=(10, 10), dpi=300)
        
        # Método vectorizado para dibujar líneas
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
        Evalúa el modelo en todos los datos.
        
        Retorna:
            Dict: Resultados de la evaluación.
        """
        if self.modelo_final is None:
            raise RuntimeError("Modelo no entrenado.")
        
        # Obtener los mejores parámetros
        p = {**self._espacio(self.study.best_trial), **self.study.best_params}
        
        # Crear dataset con la secuencia óptima
        dataset = _SeqDataset(self.X, self.Y, p["seq_len"])
        loader = DataLoader(dataset, batch_size=p["batch"], shuffle=False)
        
        # Evaluación sobre todo el conjunto de datos
        Y_pred_total = []
        Y_true_total = []
        
        self.modelo_final.eval()
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                preds = self.modelo_final(xb).cpu().numpy()
                Y_pred_total.append(preds)
                Y_true_total.append(yb.numpy())
        
        Y_pred_total = np.vstack(Y_pred_total)
        Y_true_total = np.vstack(Y_true_total)
        
        mse_rad_total = mean_squared_error(Y_true_total, Y_pred_total)
        mse_deg_total = mean_squared_error(np.rad2deg(Y_true_total), np.rad2deg(Y_pred_total))
        r2_total = r2_score(Y_true_total, Y_pred_total)
        
        # Error por articulación
        error_por_artic = np.mean((Y_true_total - Y_pred_total)**2, axis=0)
        
        # Resultados
        results = {
            'total': {
                'mse_rad': mse_rad_total,
                'mse_deg': mse_deg_total,
                'r2': r2_total,
                'Y_pred': Y_pred_total,
                'Y_true': Y_true_total
            },
            'train': None,
            'test': None,
            'error_por_artic': error_por_artic
        }
        
        # Si hay división de datos, calcular métricas por separado
        if self.test_size > 0:
            idx_train, idx_val = self._dividir_datos()
            
            # Crear loaders para train y test
            loader_train, loader_val, _, _ = self._prepare_dataset(
                p["seq_len"], idx_train, idx_val
            )
            
            # Métricas de entrenamiento
            Y_pred_train = []
            Y_true_train = []
            
            with torch.no_grad():
                for xb, yb in loader_train:
                    xb = xb.to(self.device)
                    preds = self.modelo_final(xb).cpu().numpy()
                    Y_pred_train.append(preds)
                    Y_true_train.append(yb.numpy())
            
            Y_pred_train = np.vstack(Y_pred_train)
            Y_true_train = np.vstack(Y_true_train)
            
            mse_rad_train = mean_squared_error(Y_true_train, Y_pred_train)
            mse_deg_train = mean_squared_error(np.rad2deg(Y_true_train), np.rad2deg(Y_pred_train))
            r2_train = r2_score(Y_true_train, Y_pred_train)
            
            results['train'] = {
                'mse_rad': mse_rad_train,
                'mse_deg': mse_deg_train,
                'r2': r2_train,
                'Y_pred': Y_pred_train,
                'Y_true': Y_true_train
            }
            
            # Métricas de prueba
            if loader_val is not None:
                Y_pred_test = []
                Y_true_test = []
                
                with torch.no_grad():
                    for xb, yb in loader_val:
                        xb = xb.to(self.device)
                        preds = self.modelo_final(xb).cpu().numpy()
                        Y_pred_test.append(preds)
                        Y_true_test.append(yb.numpy())
                
                Y_pred_test = np.vstack(Y_pred_test)
                Y_true_test = np.vstack(Y_true_test)
                
                mse_rad_test = mean_squared_error(Y_true_test, Y_pred_test)
                mse_deg_test = mean_squared_error(np.rad2deg(Y_true_test), np.rad2deg(Y_pred_test))
                r2_test = r2_score(Y_true_test, Y_pred_test)
                
                results['test'] = {
                    'mse_rad': mse_rad_test,
                    'mse_deg': mse_deg_test,
                    'r2': r2_test,
                    'Y_pred': Y_pred_test,
                    'Y_true': Y_true_test
                }
        
        # Imprimir solo las métricas globales
        print(f"MSE total (radianes²): {mse_rad_total:.6f}")
        print(f"MSE total (grados²):   {mse_deg_total:.6f}")
        print(f"R²: {r2_total:.6f}")
        
        return results
          
    def summary(self):
        """
        Muestra un resumen detallado del modelo y su desempeño.
        """
        if self.modelo_final is None:
            raise RuntimeError("Modelo no entrenado.")
        
        results = self.evaluar_total()
        
        # Obtener los mejores parámetros
        p = {**self._espacio(self.study.best_trial), **self.study.best_params}
        
        print("=" * 50)
        print("Modelo: LSTM")
        print(f"Arquitectura: Input({self.input_dim}) -> LSTM({p['hidden_size']}) x {p['num_layers']} -> Output({self.output_dim})")
        print(f"Parámetros: seq_len={p['seq_len']}, dropout={p['dropout']}, lr={p['lr']}, batch={p['batch']}, epochs={p['epochs']}")
        print(f"Opt: Optuna")
        
        # Métricas globales
        print("\nMétricas globales (todo el conjunto de datos):")
        print(f"R²: {results['total']['r2']:.6f}")
        print(f"MSE (radianes): {results['total']['mse_rad']:.6f}")
        print(f"MSE (grados): {results['total']['mse_deg']:.6f}")
        
        # Métricas de entrenamiento y prueba
        if results['train'] is not None:
            print("\nMétricas de entrenamiento:")
            print(f"R²: {results['train']['r2']:.6f}")
            print(f"MSE (radianes): {results['train']['mse_rad']:.6f}")
            print(f"MSE (grados): {results['train']['mse_deg']:.6f}")
            
            if results['test'] is not None:
                print("\nMétricas de prueba:")
                print(f"R²: {results['test']['r2']:.6f}")
                print(f"MSE (radianes): {results['test']['mse_rad']:.6f}")
                print(f"MSE (grados): {results['test']['mse_deg']:.6f}")
        
        # Error por articulación
        print("\nError por articulación (global):")
        for i, err in enumerate(results['error_por_artic']):
            print(f"  q{i}: {err:.6f}")
            
        # Información de MDS
        if self.stress is not None:
            print(f"\nMDS stress: {self.stress:.6f}")
            print(f"MDS stress normalizado: {self.stress_norm:.6f}")
        
        print("=" * 50) 
        
    def guardar(self, nombre_base="modelo"):
        """
        Guarda el modelo entrenado y los resultados del estudio.
        
        Parámetros:
            nombre_base (str): Nombre base para la carpeta.
        """
        if self.modelo_final is None or self.study is None:
            raise RuntimeError("Optimice y entrene antes de guardar.")
            
        nombre_base += "_LSTM"
        os.makedirs(nombre_base, exist_ok=True)
        torch.save(self.modelo_final.state_dict(), f"{nombre_base}/lstm_modelo.pt")
        self.study.trials_dataframe().to_csv(f"{nombre_base}/optuna_resultados.csv", index=False)
        
        vis.plot_optimization_history(self.study)
        plt.savefig(f"{nombre_base}/opt_history.png"); plt.clf()
        vis.plot_param_importances(self.study)
        plt.savefig(f"{nombre_base}/opt_param_importance.png"); plt.clf()
        
        print(f"Modelo y resultados guardados en carpeta: {nombre_base}")
        
    def predecir(self, X_seq):
        """
        Realiza una predicción con el modelo entrenado.
        
        Parámetros:
            X_seq (np.ndarray): Secuencia de entrada.
            
        Retorna:
            np.ndarray: Predicción.
        """
        if self.modelo_final is None:
            raise RuntimeError("Modelo no entrenado.")
            
        # Obtener los mejores parámetros
        p = {**self._espacio(self.study.best_trial), **self.study.best_params}
        
        # Verificar que la secuencia tenga la longitud correcta
        if len(X_seq) < p["seq_len"]:
            raise ValueError(f"La secuencia debe tener al menos {p['seq_len']} puntos")
        
        # Tomar los últimos seq_len puntos si la secuencia es más larga
        if len(X_seq) > p["seq_len"]:
            X_seq = X_seq[-p["seq_len"]:]
            
        # Convertir a tensor y evaluar
        self.modelo_final.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            y_pred = self.modelo_final(x_tensor).cpu().numpy().squeeze()
        
        return y_pred
        
    def predecir_serie(self, fila_actual):
        """
        Predice los valores articulares para una nueva fila de datos.
        Actualiza el historial de datos para mantener la secuencia.
        
        Parámetros:
            fila_actual (pd.Series): Serie de pandas con los valores actuales.
            
        Retorna:
            np.ndarray: Predicción de valores articulares.
        """
        if self.modelo_final is None:
            raise RuntimeError("Modelo no entrenado.")
            
        # Obtener los mejores parámetros
        p = {**self._espacio(self.study.best_trial), **self.study.best_params}
        
        # Seleccionar columnas relevantes de la fila actual
        cols_input = self.POS_COLS + [c for c in fila_actual.index 
                                     if c not in self.POS_COLS + self.Q_COLS + self.cols_excluir]
        new_x = fila_actual[cols_input].to_numpy(dtype=np.float32)
        
        # Actualizar el historial de datos
        self.X = np.vstack([self.X, new_x.reshape(1, -1)])
        
        # Tomar los últimos seq_len puntos para la predicción
        seq = self.X[-p["seq_len"]:]
        
        return self.predecir(seq)
