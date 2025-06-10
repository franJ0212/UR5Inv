import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
import optuna
import joblib
import optuna.visualization.matplotlib as vis
from typing import List, Tuple, Dict, Any, Optional, Union
from FrEIA.framework import InputNode, OutputNode, Node, ConditionNode, ReversibleGraphNet
from FrEIA.modules import AllInOneBlock
from .utils import select_m_representative_points
import pandas as pd

class CINNInversaUR5:
    """
    Clase para entrenar y evaluar un modelo CINN (Conditional Invertible Neural Network)
    para el problema de cinemática inversa del robot UR5.
    
    Permite cargar datos, optimizar hiperparámetros con Optuna, entrenar el modelo,
    evaluar su desempeño, generar múltiples soluciones y visualizar resultados.
    
    Parámetros:
        ruta_datos (str): Ruta al archivo de datos (CSV o Excel).
        test_size (float): Proporción de datos para validación/test.
        random_state (int): Semilla para la aleatorización.
        dispositivo (str): Dispositivo de cómputo ('cuda' o 'cpu').
        cols_excluir (list): Lista de columnas a excluir de las entradas.
    """
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
        
        # En CINN, el input (lo que entra) son las articulaciones y las condiciones son la posición
        self.input_dim = self.Y.shape[1]  # q0...q5
        self.cond_dim = self.X.shape[1]   # x, y, z, Rx, Ry, Rz, contexto
        
        # Inicializar a None, se crearán durante la optimización/entrenamiento
        self.modelo = None
        self.optimizer = None
        
        # Historial y resultados
        self.loss_hist = []
        self.val_hist = []
        self.study = None
        self.stress = None
        self.stress_norm = None
        
    def _cargar_datos(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga los datos desde un archivo CSV o Excel.
        
        Retorna:
            Tuple[np.ndarray, np.ndarray]: Una tupla con los datos de condición (X) y articulaciones (Y).
        """
        if self.ruta.endswith('.xlsx'):
            df = pd.read_excel(self.ruta)
        else:
            df = pd.read_csv(self.ruta)
            
        # Detectar columnas automáticamente
        columnas_salida = [f'q{i}' for i in range(6)]
        for col in columnas_salida:
            if col not in df.columns:
                raise ValueError(f"La columna requerida '{col}' no está presente en el archivo.")
                
        columnas_entrada = [c for c in df.columns
                            if c not in columnas_salida and c not in self.cols_excluir]
        
        X = df[columnas_entrada].to_numpy(dtype=np.float32)
        Y = df[columnas_salida].to_numpy(dtype=np.float32)
        return X, Y
    
    def _dividir_datos(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Divide los datos en conjuntos de entrenamiento y prueba.
        
        Retorna:
            Tuple: X_train, X_test, Y_train, Y_test.
        """
        if self.test_size <= 0:
            return self.X, np.empty((0, self.X.shape[1])), self.Y, np.empty((0, self.Y.shape[1]))
        
        # Para conjuntos grandes, usar división aleatoria
        if len(self.X) > 5000:
            return train_test_split(
                self.X, self.Y, test_size=self.test_size, random_state=self.random_state
            )
        
        # Para conjuntos pequeños, seleccionar puntos representativos
        n_train = int(len(self.X) * (1 - self.test_size))
        distances = squareform(pdist(self.X))
        train_idx = select_m_representative_points(distances, n_train)
        test_idx = list(set(range(len(self.X))) - set(train_idx))
        
        X_train, X_test = self.X[train_idx], self.X[test_idx]
        Y_train, Y_test = self.Y[train_idx], self.Y[test_idx]
        return X_train, X_test, Y_train, Y_test
    
    def _subred(self, in_ch: int, out_ch: int, hidden_dim: int, subnet_depth: int, dropout: float) -> nn.Sequential:
        """
        Crea una subred para el bloque de acoplamiento.
        
        Parámetros:
            in_ch (int): Dimensión de entrada.
            out_ch (int): Dimensión de salida.
            hidden_dim (int): Dimensión de las capas ocultas.
            subnet_depth (int): Número de capas ocultas.
            dropout (float): Tasa de dropout.
            
        Retorna:
            nn.Sequential: Una secuencia de capas como subred.
        """
        capas = []
        
        capas.append(nn.Linear(in_ch, hidden_dim))
        capas.append(nn.ReLU())
        if dropout > 0:
            capas.append(nn.Dropout(dropout))
            
        for _ in range(subnet_depth - 1):
            capas.append(nn.Linear(hidden_dim, hidden_dim))
            capas.append(nn.ReLU())
            if dropout > 0:
                capas.append(nn.Dropout(dropout))
                
        capas.append(nn.Linear(hidden_dim, out_ch))
        return nn.Sequential(*capas)
    
    def _construir_modelo(self, n_bloques: int, hidden_dim: int, subnet_depth: int, dropout: float) -> ReversibleGraphNet:
        """
        Construye la arquitectura invertible utilizando FrEIA.
        
        Parámetros:
            n_bloques (int): Número de bloques invertibles.
            hidden_dim (int): Dimensión de las capas ocultas.
            subnet_depth (int): Profundidad de la subred.
            dropout (float): Tasa de dropout.
            
        Retorna:
            ReversibleGraphNet: Un modelo invertible de FrEIA.
        """
        nodes = [InputNode(self.input_dim, name='input')]
        cond = ConditionNode(self.cond_dim, name='cond')
        
        # Creamos un closure para pasar los parámetros adicionales a _subred
        def subnet_constructor(in_ch, out_ch):
            return self._subred(in_ch, out_ch, hidden_dim, subnet_depth, dropout)
        
        for k in range(n_bloques):
            nodes.append(
                Node(nodes[-1],
                    AllInOneBlock,
                    {'subnet_constructor': subnet_constructor,
                     'affine_clamping': 2.0},
                    conditions=[cond],
                    name=f'block_{k}')
            )
            
        nodes.append(OutputNode(nodes[-1], name='output'))  
        nodes.append(cond)
        
        model = ReversibleGraphNet(nodes, verbose=False).to(self.device)
        return model
    
    def _espacio(self, t) -> Dict[str, Any]:
        """
        Define el espacio de búsqueda para la optimización de hiperparámetros.
        
        Parámetros:
            t: Un objeto trial de Optuna.
            
        Retorna:
            Dict[str, Any]: Un diccionario con los hiperparámetros sugeridos.
        """
        return {
            "n_bloques": t.suggest_int("n_bloques", 4, 10),
            "hidden_dim": t.suggest_categorical("hidden_dim", [64, 128, 256, 512]),
            "subnet_depth": t.suggest_int("subnet_depth", 1, 3),
            "dropout": t.suggest_float("dropout", 0.0, 0.5),
            "lr": t.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch": t.suggest_categorical("batch", [32, 64, 128]),
            "epochs": t.suggest_int("epochs", 50, 300)
        }
    
    def _train_once(self, p: Dict[str, Any], Xtr: np.ndarray, Ytr: np.ndarray, 
                   Xval: np.ndarray, Yval: np.ndarray) -> Tuple[float, ReversibleGraphNet]:
        """
        Entrena un modelo CINN una vez con los parámetros dados.
        
        Parámetros:
            p (Dict[str, Any]): Diccionario de hiperparámetros.
            Xtr (np.ndarray): Datos de condición para entrenamiento.
            Ytr (np.ndarray): Datos de articulaciones para entrenamiento.
            Xval (np.ndarray): Datos de condición para validación.
            Yval (np.ndarray): Datos de articulaciones para validación.
            
        Retorna:
            Tuple: MSE en validación y modelo entrenado.
        """
        # Construir modelo
        model = self._construir_modelo(
            p["n_bloques"], 
            p["hidden_dim"], 
            p["subnet_depth"], 
            p["dropout"]
        )
        
        # Optimizador
        optimizer = optim.Adam(model.parameters(), lr=p["lr"])
        
        # Preparar dataset y dataloader
        # Nota: En CINN, Y entra al modelo y X es la condición
        train_dataset = TensorDataset(
            torch.tensor(Ytr, dtype=torch.float32),
            torch.tensor(Xtr, dtype=torch.float32)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=p["batch"], shuffle=True
        )
        
        # Convertir datos de validación a tensores
        X_val_tensor = torch.tensor(Xval, dtype=torch.float32).to(self.device)
        Y_val_tensor = torch.tensor(Yval, dtype=torch.float32).to(self.device)
        
        # Historial de pérdidas
        train_losses = []
        val_losses = []
        
        # Entrenamiento
        model.train()
        for epoch in range(p["epochs"]):
            running_loss = 0.0
            batches = 0
            
            for batch_y, batch_x in train_loader:
                batches += 1
                
                # Transferir datos al dispositivo
                batch_y = batch_y.to(self.device)
                batch_x = batch_x.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass: el modelo toma articulaciones y condiciones
                output = model(batch_y, c=batch_x)
                
                # Extraer z y log_jac según la estructura de retorno
                if isinstance(output, tuple):
                    z = output[0]  # El primer elemento siempre es z
                    if len(output) > 1:
                        log_jac = output[1]  # El segundo elemento es log_jac
                    else:
                        log_jac = model.log_jacobian(batch_y, c=batch_x)
                else:
                    z = output
                    log_jac = model.log_jacobian(batch_y, c=batch_x)
                
                # Pérdida log-likelihood negativa
                loss = 0.5 * torch.sum(z**2, dim=1) - log_jac
                loss = torch.mean(loss)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * len(batch_y)
            
            # Pérdida promedio de la época
            epoch_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_loss)
            
            # Evaluación en validación
            if len(Xval) > 0:
                model.eval()
                with torch.no_grad():
                    output_val = model(Y_val_tensor, c=X_val_tensor)
                    
                    # Extraer z y log_jac para validación
                    if isinstance(output_val, tuple):
                        z_val = output_val[0]
                        if len(output_val) > 1:
                            log_jac_val = output_val[1]
                        else:
                            log_jac_val = model.log_jacobian(Y_val_tensor, c=X_val_tensor)
                    else:
                        z_val = output_val
                        log_jac_val = model.log_jacobian(Y_val_tensor, c=X_val_tensor)
                    
                    val_loss = 0.5 * torch.sum(z_val**2, dim=1) - log_jac_val
                    val_loss = val_loss.mean().item()
                    val_losses.append(val_loss)
        
        # Guardar historial de pérdidas
        self.loss_hist = train_losses
        self.val_hist = val_losses
        
        # Evaluar MSE en validación para selección de hiperparámetros
        # Primero generamos varios puntos latentes aleatorios
        model.eval()
        val_mse = 0
        if len(Xval) > 0:
            with torch.no_grad():
                n_samples = 10  # Número de muestras a promediar
                predictions = []
                
                for _ in range(n_samples):
                    # Generar puntos latentes aleatorios
                    z = torch.randn(len(Xval), self.input_dim, device=self.device)
                    # Generar predicciones: modelo invertido de z a y, dada la condición x
                    y_pred = model(z, c=X_val_tensor, rev=True)
                    if isinstance(y_pred, tuple):
                        y_pred = y_pred[0]
                    predictions.append(y_pred.cpu().numpy())
                
                # Promediar las predicciones
                avg_pred = np.mean(predictions, axis=0)
                # Calcular MSE
                val_mse = mean_squared_error(Yval, avg_pred)
        
        return val_mse, model
    
    def _objetivo(self, trial):
        """
        Función objetivo para la optimización de hiperparámetros.
        
        Parámetros:
            trial: Objeto trial de Optuna.
            
        Retorna:
            float: Negativo del MSE (para maximización).
        """
        p = self._espacio(trial)
        Xtr, Xval, Ytr, Yval = self._dividir_datos()
        mse, _ = self._train_once(p, Xtr, Ytr, Xval, Yval)
        return -mse  # Maximizar el negativo del MSE (minimizar MSE)
    
    def optimizar(self, n_trials=30, nombre_est="estudio_ur5_cinn"):
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
        Entrena el modelo CINN con los mejores hiperparámetros encontrados.
        
        Retorna:
            float: MSE en grados².
        """
        if self.study is None:
            raise RuntimeError("Ejecute optimizar() primero.")
            
        p = {**self._espacio(self.study.best_trial), **self.study.best_params}
        Xtr, Xval, Ytr, Yval = self._dividir_datos()
        mse, model = self._train_once(p, Xtr, Ytr, Xval, Yval)
        
        self.modelo = model
        self.optimizer = optim.Adam(model.parameters(), lr=p["lr"])
        
        print(f"MSE grados² validación: {mse:.6f}")
        return mse
    
    def entrenar(self, params=None):
        """
        Entrena el modelo CINN con parámetros específicos.
        
        Parámetros:
            params (Dict[str, Any], opcional): Diccionario de hiperparámetros. Si es None, 
                                              se utilizan los mejores parámetros encontrados.
        
        Retorna:
            float: MSE en grados².
        """
        if params is None:
            if self.study is None:
                raise RuntimeError("Ejecute optimizar() primero o proporcione parámetros.")
            params = {**self._espacio(self.study.best_trial), **self.study.best_params}
        
        Xtr, Xval, Ytr, Yval = self._dividir_datos()
        mse, model = self._train_once(params, Xtr, Ytr, Xval, Yval)
        
        self.modelo = model
        self.optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        
        print(f"MSE grados² validación: {mse:.6f}")
        return mse
    
    def graficar_perdidas(self):
        """
        Grafica la evolución de las pérdidas durante el entrenamiento.
        """
        if not self.loss_hist:
            raise RuntimeError("No hay historial de entrenamiento disponible.")
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_hist, label='Entrenamiento')
        if self.val_hist:
            plt.plot(self.val_hist, label='Validación')
        plt.xlabel('Época')
        plt.ylabel('Pérdida (NLL)')
        plt.title('Evolución de la pérdida durante el entrenamiento')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def graficar_offsets(self, pmax=10000):
        """
        Visualiza el error entre valores reales y predichos usando MDS.
        
        Parámetros:
            pmax (int): Número máximo de puntos para el análisis MDS.
        """
        if self.modelo is None:
            raise RuntimeError("Modelo no entrenado.")
        
        # Evaluación del modelo
        self.modelo.eval()
        with torch.no_grad():
            # Generar múltiples predicciones y promediar
            n_samples = 10
            preds = []
            for _ in range(n_samples):
                # Generar un conjunto diferente de puntos latentes para cada muestra
                z = torch.randn(len(self.X), self.input_dim, device=self.device)
                X_tensor = torch.tensor(self.X, dtype=torch.float32).to(self.device)
                
                # Generar predicciones: inversa de z a y dado x
                y_pred = self.modelo(z, c=X_tensor, rev=True)
                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]
                preds.append(y_pred.cpu().numpy())
            
            # Promediar predicciones
            Y_pred = np.mean(preds, axis=0)
        
        # Calcular distancias sobre Y verdadero
        distances = squareform(pdist(self.Y))
        
        # Seleccionar puntos representativos si es necesario
        if len(self.Y) > pmax:
            print(f"Usando {pmax} puntos representativos para MDS...")
            selected_indices = select_m_representative_points(distances, pmax)
            Y_true = self.Y[selected_indices]
            Y_pred = Y_pred[selected_indices]
        else:
            Y_true = self.Y
        
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
    
    def evaluar_total(self, n_muestras=10):
        """
        Evalúa el modelo en todo el conjunto de datos.
        
        Parámetros:
            n_muestras (int): Número de muestras a promediar para las predicciones.
            
        Retorna:
            Dict: Resultados de la evaluación.
        """
        if self.modelo is None:
            raise RuntimeError("Modelo no entrenado.")
        
        results = {}
        
        # Evaluación sobre todo el conjunto
        self.modelo.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(self.X, dtype=torch.float32).to(self.device)
            
            # Generar múltiples predicciones y promediar
            preds = []
            for _ in range(n_muestras):
                z = torch.randn(len(self.X), self.input_dim, device=self.device)
                y_pred = self.modelo(z, c=X_tensor, rev=True)
                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]
                preds.append(y_pred.cpu().numpy())
            
            Y_pred_total = np.mean(preds, axis=0)
        
        mse_rad_total = mean_squared_error(self.Y, Y_pred_total)
        mse_deg_total = mean_squared_error(np.rad2deg(self.Y), np.rad2deg(Y_pred_total))
        r2_total = r2_score(self.Y, Y_pred_total)
        
        # Error por articulación
        error_por_artic = np.mean((self.Y - Y_pred_total)**2, axis=0)
        
        # Resultados
        results['total'] = {
            'mse_rad': mse_rad_total,
            'mse_deg': mse_deg_total,
            'r2': r2_total,
            'Y_pred': Y_pred_total
        }
        results['error_por_artic'] = error_por_artic
        
        # Si hay división de datos, calcular métricas por separado
        if self.test_size > 0:
            X_train, X_test, Y_train, Y_test = self._dividir_datos()
            
            # Métricas de entrenamiento
            with torch.no_grad():
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
                
                preds_train = []
                for _ in range(n_muestras):
                    z_train = torch.randn(len(X_train), self.input_dim, device=self.device)
                    y_pred_train = self.modelo(z_train, c=X_train_tensor, rev=True)
                    if isinstance(y_pred_train, tuple):
                        y_pred_train = y_pred_train[0]
                    preds_train.append(y_pred_train.cpu().numpy())
                
                Y_pred_train = np.mean(preds_train, axis=0)
            
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
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
                
                preds_test = []
                for _ in range(n_muestras):
                    z_test = torch.randn(len(X_test), self.input_dim, device=self.device)
                    y_pred_test = self.modelo(z_test, c=X_test_tensor, rev=True)
                    if isinstance(y_pred_test, tuple):
                        y_pred_test = y_pred_test[0]
                    preds_test.append(y_pred_test.cpu().numpy())
                
                Y_pred_test = np.mean(preds_test, axis=0)
            
            mse_rad_test = mean_squared_error(Y_test, Y_pred_test)
            mse_deg_test = mean_squared_error(np.rad2deg(Y_test), np.rad2deg(Y_pred_test))
            r2_test = r2_score(Y_test, Y_pred_test)
            
            results['test'] = {
                'mse_rad': mse_rad_test,
                'mse_deg': mse_deg_test,
                'r2': r2_test,
                'Y_pred': Y_pred_test
            }
        
        # Imprimir métricas globales
        print(f"MSE total (radianes²): {mse_rad_total:.6f}")
        print(f"MSE total (grados²):   {mse_deg_total:.6f}")
        print(f"R²: {r2_total:.6f}")
        
        return results
    
    def inferir_muestras(self, x: np.ndarray, n: int = 5) -> np.ndarray:
        """
        Genera n muestras diferentes a partir de una condición dada.
        
        Parámetros:
            x (np.ndarray): Condición (posición del efector final).
            n (int): Número de soluciones a generar.
            
        Retorna:
            np.ndarray: Conjunto de soluciones generadas.
        """
        if self.modelo is None:
            raise RuntimeError("Modelo no entrenado.")
            
        self.modelo.eval()
        with torch.no_grad():
            # Asegurar que x es un array 2D
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
                
            # Generar n puntos latentes aleatorios
            z_samples = torch.randn(n, self.input_dim, device=self.device)
            
            # Preparar condición
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            # Repetir la condición n veces si hay solo una
            if x.shape[0] == 1 and n > 1:
                x_tensor = x_tensor.repeat(n, 1)
            
            # Realizar inferencia inversa (z, x -> y)
            y_pred = self.modelo(z_samples, c=x_tensor, rev=True)
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]
                
            return y_pred.cpu().numpy()
    
    def comparar_soluciones(self, posicion: np.ndarray, n_soluciones: int = 5, 
                          mostrar_grafica: bool = True) -> np.ndarray:
        """
        Genera y compara múltiples soluciones para una posición específica.
        
        Parámetros:
            posicion (np.ndarray): Vector de posición del efector final.
            n_soluciones (int): Número de soluciones a generar.
            mostrar_grafica (bool): Si es True, muestra una gráfica comparativa.
            
        Retorna:
            np.ndarray: Matriz con las soluciones generadas.
        """
        if self.modelo is None:
            raise RuntimeError("Modelo no entrenado.")
            
        # Asegurarse de que posicion es un array 2D
        if len(posicion.shape) == 1:
            posicion = posicion.reshape(1, -1)
        
        # Generar soluciones
        soluciones = self.inferir_muestras(posicion, n_soluciones)
        
        # Mostrar soluciones numéricas
        print(f"Comparación de {n_soluciones} soluciones para la posición:")
        print(f"Posición: {posicion.flatten()}")
        print("\nSoluciones generadas (ángulos en radianes):")
        
        for i, solucion in enumerate(soluciones):
            print(f"Solución {i+1}:")
            for j, angulo in enumerate(solucion):
                print(f"  q{j}: {angulo:.6f}")
            print("")
        
        # Calcular varianza entre soluciones
        varianza = np.var(soluciones, axis=0)
        print("Varianza entre soluciones por articulación:")
        for j, var in enumerate(varianza):
            print(f"  q{j}: {var:.6f}")
        
        # Visualización gráfica
        if mostrar_grafica and n_soluciones > 1:
            # Crear gráfica
            plt.figure(figsize=(12, 6))
            
            # Gráfica por articulación con diferentes soluciones en colores
            ax1 = plt.subplot(1, 2, 1)
            x = np.arange(self.input_dim)  # Articulaciones en el eje X
            width = 0.8 / n_soluciones
            offset = width * np.arange(n_soluciones) - width * (n_soluciones - 1) / 2
            
            for i in range(n_soluciones):
                ax1.bar(x + offset[i], soluciones[i], 
                        width=width, label=f'Sol {i+1}')
            
            ax1.set_xlabel('Articulación')
            ax1.set_ylabel('Ángulo (radianes)')
            ax1.set_xticks(x)
            ax1.set_xticklabels([f'q{i}' for i in range(self.input_dim)])
            ax1.set_title('Comparación de soluciones por articulación')
            ax1.legend()
            
            # Gráfica de varianza
            ax2 = plt.subplot(1, 2, 2)
            ax2.bar(np.arange(self.input_dim), np.sqrt(varianza), color='orange')
            ax2.set_xlabel('Articulación')
            ax2.set_ylabel('Desviación estándar (radianes)')
            ax2.set_xticks(np.arange(self.input_dim))
            ax2.set_xticklabels([f'q{j}' for j in range(self.input_dim)])
            ax2.set_title('Variabilidad entre soluciones')
            
            plt.tight_layout()
            plt.show()
        
        return soluciones
    
    def directa(self, q: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Evalúa el modelo en modo directo: f(q | x) = z
        
        Parámetros:
            q (np.ndarray): Configuración articular.
            x (np.ndarray): Condición (posición del efector final).
            
        Retorna:
            np.ndarray: Vector latente z.
        """
        if self.modelo is None:
            raise RuntimeError("Modelo no entrenado.")
            
        self.modelo.eval()
        with torch.no_grad():
            # Asegurar que q y x son arrays 2D
            if len(q.shape) == 1:
                q = q.reshape(1, -1)
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            
            # Repetir x si es necesario para que coincida con el tamaño de q
            if x.shape[0] == 1 and q.shape[0] > 1:
                x = np.repeat(x, q.shape[0], axis=0)
            
            # Convertir a tensores
            q_tensor = torch.tensor(q, dtype=torch.float32).to(self.device)
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            
            # Forward pass
            output = self.modelo(q_tensor, c=x_tensor)
            if isinstance(output, tuple):
                z = output[0]
            else:
                z = output
                
            return z.cpu().numpy()
    
    def summary(self):
        """
        Muestra un resumen detallado del modelo CINN y su desempeño.
        """
        if self.modelo is None:
            raise RuntimeError("Modelo no entrenado.")
        
        results = self.evaluar_total()
        
        # Obtener los mejores parámetros
        if self.study is not None:
            p = {**self._espacio(self.study.best_trial), **self.study.best_params}
        else:
            p = {"n_bloques": "?", "hidden_dim": "?", "subnet_depth": "?", "dropout": "?"}
        
        print("=" * 50)
        print("Modelo: CINN")
        print(f"Arquitectura: {self.input_dim} (q) -> CINN con {p['n_bloques']} bloques -> {self.input_dim} (z)")
        print(f"Condiciones: {self.cond_dim} dimensiones (x)")
        print(f"Subredes: hidden_dim={p['hidden_dim']}, depth={p['subnet_depth']}, dropout={p['dropout']}")
        print(f"Opt: {'Optuna' if self.study is not None else 'Manual'}")
        
        # Métricas globales
        print("\nMétricas globales (todo el conjunto de datos):")
        print(f"R²: {results['total']['r2']:.6f}")
        print(f"MSE (radianes): {results['total']['mse_rad']:.6f}")
        print(f"MSE (grados): {results['total']['mse_deg']:.6f}")
        
        # Métricas de entrenamiento y prueba
        if 'train' in results:
            print("\nMétricas de entrenamiento:")
            print(f"R²: {results['train']['r2']:.6f}")
            print(f"MSE (radianes): {results['train']['mse_rad']:.6f}")
            print(f"MSE (grados): {results['train']['mse_deg']:.6f}")
            
            if 'test' in results:
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
        if self.modelo is None:
            raise RuntimeError("Entrene el modelo antes de guardar.")
            
        nombre_base += "_CINN"
        os.makedirs(nombre_base, exist_ok=True)
        
        # Guardar modelo
        torch.save(self.modelo.state_dict(), f"{nombre_base}/cinn_modelo.pt")
        
        # Guardar historial de pérdidas
        np.savez(f"{nombre_base}/historial.npz", 
                 loss_hist=np.array(self.loss_hist),
                 val_hist=np.array(self.val_hist) if self.val_hist else np.array([]))
        
        # Guardar información del modelo
        if self.study is not None:
            self.study.trials_dataframe().to_csv(f"{nombre_base}/optuna_resultados.csv", index=False)
            vis.plot_optimization_history(self.study)
            plt.savefig(f"{nombre_base}/opt_history.png"); plt.clf()
            vis.plot_param_importances(self.study)
            plt.savefig(f"{nombre_base}/opt_param_importance.png"); plt.clf()
        
        print(f"Modelo y resultados guardados en carpeta: {nombre_base}")
    
    def cargar(self, nombre_base="modelo_CINN", params=None):
        """
        Carga un modelo previamente guardado.
        
        Parámetros:
            nombre_base (str): Nombre de la carpeta donde se encuentra el modelo.
            params (Dict[str, Any], opcional): Parámetros para reconstruir el modelo si no se conocen.
        """
        if not os.path.exists(nombre_base):
            raise RuntimeError(f"La carpeta {nombre_base} no existe.")
        
        # Obtener parámetros para reconstruir el modelo
        if params is None:
            if self.study is not None:
                params = {**self._espacio(self.study.best_trial), **self.study.best_params}
            else:
                raise ValueError("Se necesitan los parámetros para reconstruir el modelo.")
        
        # Construir modelo
        modelo = self._construir_modelo(
            params["n_bloques"], 
            params["hidden_dim"], 
            params["subnet_depth"], 
            params["dropout"]
        )
        
        # Cargar pesos
        modelo.load_state_dict(torch.load(f"{nombre_base}/cinn_modelo.pt", map_location=self.device))
        self.modelo = modelo
        
        # Cargar historial de pérdidas si existe
        if os.path.exists(f"{nombre_base}/historial.npz"):
            historial = np.load(f"{nombre_base}/historial.npz")
            self.loss_hist = historial['loss_hist'].tolist()
            if 'val_hist' in historial and len(historial['val_hist']) > 0:
                self.val_hist = historial['val_hist'].tolist()
        
        print(f"Modelo cargado correctamente desde: {nombre_base}")
    
    def predecir(self, X, n_muestras=10):
        """
        Realiza una predicción con el modelo entrenado, promediando múltiples muestras.
        
        Parámetros:
            X (np.ndarray): Vector o matriz de posiciones.
            n_muestras (int): Número de muestras a promediar.
            
        Retorna:
            np.ndarray: Predicción de ángulos articulares.
        """
        if self.modelo is None:
            raise RuntimeError("Modelo no entrenado.")
            
        # Asegurar que X es un array 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        # Verificar dimensiones
        if X.shape[1] != self.cond_dim:
            raise ValueError(f"La entrada debe tener {self.cond_dim} dimensiones, pero tiene {X.shape[1]}")
        
        # Realizar predicción promediando múltiples muestras
        self.modelo.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            
            preds = []
            for _ in range(n_muestras):
                z = torch.randn(len(X), self.input_dim, device=self.device)
                y_pred = self.modelo(z, c=X_tensor, rev=True)
                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]
                preds.append(y_pred.cpu().numpy())
            
            prediction = np.mean(preds, axis=0)
        
        # Si solo hay una muestra, devolver vector
        if len(X) == 1:
            return prediction[0]
        
        return prediction

class CINNInversaUR5Simple:
    def __init__(self, ruta_datos, n_bloques=6, hidden_dim=128, subnet_depth=2,
                 dropout=0.0, lr=1e-3, batch_size=64, epochs=100, test_size=0.2,
                 random_state=42, cols_excluir=None):
        
        self.ruta = ruta_datos
        self.n_bloques = n_bloques
        self.hidden_dim = hidden_dim
        self.subnet_depth = subnet_depth
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.test_size = test_size
        self.random_state = random_state
        self.cols_excluir = cols_excluir or []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.X, self.Y = self._cargar_datos()
        self.input_dim = self.Y.shape[1]  # q0...q5
        self.cond_dim = self.X.shape[1]   # x, y, z, Rx, Ry, Rz, contexto

        self.modelo = self._construir_modelo()
        self.optimizer = torch.optim.Adam(self.modelo.parameters(), lr=self.lr)

    def _cargar_datos(self):
        """
        Carga el archivo de datos y separa las variables de condición (x)
        y las articulaciones objetivo (q0 a q5) como salida.
        """
        if self.ruta.endswith('.xlsx'):
            df = pd.read_excel(self.ruta)
        else:
            df = pd.read_csv(self.ruta)

        columnas_salida = [f'q{i}' for i in range(6)]
        for col in columnas_salida:
            if col not in df.columns:
                raise ValueError(f"La columna requerida '{col}' no está presente en el archivo.")

        columnas_entrada = [c for c in df.columns
                            if c not in columnas_salida and c not in (self.cols_excluir or [])]

        X = df[columnas_entrada].to_numpy(dtype=np.float32)
        Y = df[columnas_salida].to_numpy(dtype=np.float32)

        return X, Y

    def _dividir_datos(self):
        """
        Divide los datos en conjuntos de entrenamiento y prueba.
        """
        if self.test_size <= 0:
            return self.X, np.empty((0, self.X.shape[1])), self.Y, np.empty((0, self.Y.shape[1]))

        if len(self.X) > 5000:
            from sklearn.model_selection import train_test_split
            return train_test_split(
                self.X, self.Y, test_size=self.test_size, random_state=self.random_state
            )
        
        from scipy.spatial.distance import pdist, squareform
        from utils import select_m_representative_points
        
        n_train = int(len(self.X) * (1 - self.test_size))
        D = squareform(pdist(self.X))
        idx_train = select_m_representative_points(D, n_train)
        idx_test = list(set(range(len(self.X))) - set(idx_train))

        return self.X[idx_train], self.X[idx_test], self.Y[idx_train], self.Y[idx_test]

    def _subred(self, in_ch, out_ch):
        """
        Crea una subred para el bloque de acoplamiento.
        """
        capas = []

        capas.append(nn.Linear(in_ch, self.hidden_dim))
        capas.append(nn.ReLU())
        if self.dropout > 0:
            capas.append(nn.Dropout(self.dropout))

        for _ in range(self.subnet_depth - 1):
            capas.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            capas.append(nn.ReLU())
            if self.dropout > 0:
                capas.append(nn.Dropout(self.dropout))

        capas.append(nn.Linear(self.hidden_dim, out_ch))
        return nn.Sequential(*capas)

    def _construir_modelo(self):
        """
        Construye la arquitectura invertible utilizando FrEIA.
        """
        nodes = [InputNode(self.input_dim, name='input')]
        cond = ConditionNode(self.cond_dim, name='cond')

        for k in range(self.n_bloques):
            nodes.append(
                Node(nodes[-1],
                    AllInOneBlock,
                    {'subnet_constructor': self._subred,
                    'affine_clamping': 2.0},
                    conditions=[cond],
                    name=f'block_{k}')
            )

        nodes.append(OutputNode(nodes[-1], name='output'))  
        nodes.append(cond)

        model = ReversibleGraphNet(nodes, verbose=False).to(self.device)
        return model

    def entrenar(self):
        """
        Entrena el modelo INN utilizando el método estándar de maximización de verosimilitud.
        """
        Xtr, Xval, Ytr, Yval = self._dividir_datos()

        Xtr_tensor = torch.tensor(Xtr, dtype=torch.float32).to(self.device)
        Ytr_tensor = torch.tensor(Ytr, dtype=torch.float32).to(self.device)
        Xval_tensor = torch.tensor(Xval, dtype=torch.float32).to(self.device)
        Yval_tensor = torch.tensor(Yval, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(Ytr_tensor, Xtr_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.loss_hist = []
        self.val_hist = []
        self.modelo.train()

        for epoca in range(self.epochs):
            epoch_loss = 0

            for batch_q, batch_x in loader:
                self.optimizer.zero_grad()

                # Manejo flexible de la salida del modelo de FrEIA
                output = self.modelo(batch_q, c=batch_x)
                
                # Extracción de z y log_jac según la estructura de retorno
                if isinstance(output, tuple):
                    z = output[0]  # El primer elemento siempre es z
                    if len(output) > 1:
                        log_jac = output[1]  # El segundo elemento es log_jac si está disponible
                    else:
                        # Si no hay log_jac disponible, usar otro método para obtenerlo
                        log_jac = self.modelo.log_jacobian(batch_q, c=batch_x)
                else:
                    # Si la salida no es una tupla, asumimos que es solo z
                    z = output
                    # Intentar obtener el jacobiano de otra manera
                    log_jac = self.modelo.log_jacobian(batch_q, c=batch_x)
                
                # Pérdida log-likelihood negativa
                loss = 0.5 * torch.sum(z**2, dim=1) - log_jac
                loss = torch.mean(loss)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * len(batch_q)

            epoch_loss /= len(loader.dataset)
            self.loss_hist.append(epoch_loss)

            # Evaluación con datos de validación
            with torch.no_grad():
                output_val = self.modelo(Yval_tensor, c=Xval_tensor)
                
                # Extracción de z y log_jac para validación
                if isinstance(output_val, tuple):
                    z_val = output_val[0]
                    if len(output_val) > 1:
                        log_jac_val = output_val[1]
                    else:
                        log_jac_val = self.modelo.log_jacobian(Yval_tensor, c=Xval_tensor)
                else:
                    z_val = output_val
                    log_jac_val = self.modelo.log_jacobian(Yval_tensor, c=Xval_tensor)
                
                val_loss = 0.5 * torch.sum(z_val**2, dim=1) - log_jac_val
                val_loss = val_loss.mean().item()
                self.val_hist.append(val_loss)

            # Reporte cada 10 épocas
            if (epoca + 1) % 10 == 0 or epoca == self.epochs - 1:
                print(f"Época {epoca + 1}/{self.epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")

    def graficar_perdidas(self):
        """
        Grafica la evolución de las pérdidas durante el entrenamiento.
        """
        if not hasattr(self, 'loss_hist'):
            raise RuntimeError("No se ha entrenado el modelo aún.")

        plt.figure(figsize=(8, 4))
        plt.plot(self.loss_hist, label="Train Loss")
        if hasattr(self, 'val_hist'):
            plt.plot(self.val_hist, label="Val Loss")
        plt.xlabel("Época")
        plt.ylabel("Loss (NLL)")
        plt.title("Evolución de la función de pérdida")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def inversa(self, x, z):
        """
        Genera configuraciones articulares a partir de condiciones y vectores latentes.
        """
        self.modelo.eval()
        with torch.no_grad():
            x = np.atleast_2d(x)
            z = np.atleast_2d(z)

            if x.shape[0] == 1 and z.shape[0] > 1:
                x = np.repeat(x, z.shape[0], axis=0)

            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            z_tensor = torch.tensor(z, dtype=torch.float32).to(self.device)

            # En FrEIA, el método inverso espera que c sea un iterable con las condiciones
            output = self.modelo(z_tensor, c=x_tensor, rev=True)
            
            # Extraer q_hat (resultado de la inversa) del output
            if isinstance(output, tuple):
                q_hat = output[0]
            else:
                q_hat = output
                
            return q_hat.cpu().numpy()

    def inferir_muestras(self, x, n=5):
        """
        Genera n muestras diferentes a partir de una condición dada.
        """
        z_samples = np.random.randn(n, self.input_dim).astype(np.float32)
        return self.inversa(x, z_samples)

    def directa(self, q, x):
        """
        Evalúa el modelo en modo directo: f(q | x) = z
        """
        self.modelo.eval()
        with torch.no_grad():
            q = np.atleast_2d(q)
            x = np.atleast_2d(x)

            if x.shape[0] == 1 and q.shape[0] > 1:
                x = np.repeat(x, q.shape[0], axis=0)

            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            q_tensor = torch.tensor(q, dtype=torch.float32).to(self.device)

            output = self.modelo(q_tensor, c=x_tensor)
            
            # Extraer z del output
            if isinstance(output, tuple):
                z = output[0]
            else:
                z = output
                
            return z.cpu().numpy()

    def evaluar(self, n_muestras=10):
        """
        Evalúa el modelo y devuelve métricas de rendimiento.
        """
        resultados = {}

        # Evaluación global
        X_full = np.asarray(self.X)
        Y_full = np.asarray(self.Y)

        self.modelo.eval()
        with torch.no_grad():
            preds = []
            for _ in range(n_muestras):
                z_i = np.random.randn(len(X_full), self.input_dim).astype(np.float32)
                pred_i = self.inversa(X_full, z_i)
                preds.append(pred_i)
            pred = np.mean(preds, axis=0)

            mse = mean_squared_error(Y_full, pred)
            r2 = r2_score(Y_full, pred)
            error_l2 = np.mean((Y_full - pred) ** 2, axis=0)
            error_l1 = np.mean(np.abs(Y_full - pred), axis=0)

            resultados['Global'] = {
                'MSE': mse,
                'R2': r2,
                'Error_L2_por_articulacion': error_l2,
                'Error_L1_por_articulacion': error_l1
            }

            # Evaluación por conjuntos separados
            Xtr, Xval, Ytr, Yval = self._dividir_datos()
            for conjunto, X, Y in [('Train', Xtr, Ytr), ('Test', Xval, Yval)]:
                X = np.asarray(X)
                Y = np.asarray(Y)

                preds = []
                for _ in range(n_muestras):
                    z_i = np.random.randn(len(X), self.input_dim).astype(np.float32)
                    pred_i = self.inversa(X, z_i)
                    preds.append(pred_i)
                pred = np.mean(preds, axis=0)

                mse = mean_squared_error(Y, pred)
                r2 = r2_score(Y, pred)
                error_l2 = np.mean((Y - pred) ** 2, axis=0)
                error_l1 = np.mean(np.abs(Y - pred), axis=0)

                resultados[conjunto] = {
                    'MSE': mse,
                    'R2': r2,
                    'Error_L2_por_articulacion': error_l2,
                    'Error_L1_por_articulacion': error_l1
                }

        return resultados
   
    def summary(self):
        """
        Imprime un resumen de desempeño del modelo.
        """
        resultados = self.evaluar()
        print("=" * 50)
        print("Resumen de evaluación del modelo")
        for conjunto, metricas in resultados.items():
            print(f"\n[{conjunto}]:")
            print(f"MSE total: {metricas['MSE']:.6f}")
            print(f"R² total: {metricas['R2']:.6f}")
            print("Error L2 por articulación:")
            print(metricas['Error_L2_por_articulacion'])
            print("Error L1 por articulación:")
            print(metricas['Error_L1_por_articulacion'])
        print("=" * 50)
