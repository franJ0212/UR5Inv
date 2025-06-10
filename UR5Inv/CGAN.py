import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from typing import List, Tuple, Dict, Any, Optional
import optuna
import joblib
import optuna.visualization.matplotlib as vis
from .utils import select_m_representative_points

class _Generator(nn.Module):
    """
    Generador para CGAN que produce ángulos articulares a partir de posiciones y ruido.
    Utiliza tanh en la capa de salida para limitar los valores al rango [-π, π].
    """
    def __init__(self, input_dim=3, noise_dim=10, output_dim=6, hidden_layers=(128, 64)):
        super().__init__()
        total_input = input_dim + noise_dim
        
        # Capas ocultas con activaciones
        hidden_layers_list = []
        in_dim = total_input
        for h in hidden_layers:
            hidden_layers_list.extend([
                nn.Linear(in_dim, h),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(h)
            ])
            in_dim = h
            
        self.hidden_layers = nn.Sequential(*hidden_layers_list)
        
        # Capa de salida separada (sin activación tanh por ahora)
        self.output_layer = nn.Linear(in_dim, output_dim)
        
        # Constante π para multiplicar la activación tanh
        self.pi = np.pi
        
    def forward(self, x, z):
        # Asegurar que x y z tengan el mismo tamaño de lote
        if x.size(0) != z.size(0):
            raise ValueError(f"Las dimensiones de batch no coinciden: x={x.size(0)}, z={z.size(0)}")
            
        # Combinar entrada y ruido
        x_combined = torch.cat([x, z], dim=1)
        
        # Pasar por capas ocultas
        features = self.hidden_layers(x_combined)
        
        # Capa de salida con activación tanh * π para limitar al rango [-π, π]
        output = torch.tanh(self.output_layer(features)) * self.pi
        
        return output

class _Discriminator(nn.Module):
    """
    Discriminador para CGAN que clasifica pares (posición, ángulos) como reales o generados.
    """
    def __init__(self, input_dim=3, target_dim=6, hidden_layers=(64, 32)):
        super().__init__()
        total_input = input_dim + target_dim
        layers = []
        in_dim = total_input
        for h in hidden_layers:
            layers.extend([nn.Linear(in_dim, h), nn.LeakyReLU(0.2), nn.Dropout(0.3)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, y):
        x_combined = torch.cat([x, y], dim=1)
        return self.net(x_combined)

class CGANInversaUR5Simple:
    """
    CGAN para resolver la cinemática inversa del robot UR5.
    
    Esta clase implementa un Generative Adversarial Network Condicional (CGAN)
    para aproximar la función de cinemática inversa, mapeando posiciones 
    cartesianas a ángulos articulares, con capacidad de generar múltiples 
    soluciones válidas dada una posición.
    
    Parámetros:
        ruta_datos (str): Ruta al archivo de datos (CSV o Excel).
        noise_dim (int): Dimensión del vector de ruido.
        gen_hidden (tuple): Capas ocultas del generador.
        disc_hidden (tuple): Capas ocultas del discriminador.
        lr_gen (float): Tasa de aprendizaje del generador.
        lr_disc (float): Tasa de aprendizaje del discriminador.
        batch_size (int): Tamaño del lote.
        epochs (int): Número de épocas.
        test_size (float): Proporción de datos para test.
        random_state (int): Semilla aleatoria.
        cols_excluir (list): Columnas a excluir.
    """
    def __init__(self, ruta_datos, noise_dim=10, 
                 gen_hidden=(128, 64), disc_hidden=(64, 32),
                 lr_gen=0.0002, lr_disc=0.0002, 
                 batch_size=64, epochs=100, 
                 test_size=0.2, random_state=42, cols_excluir=None):
        
        self.ruta = ruta_datos
        self.noise_dim = noise_dim
        self.gen_hidden = gen_hidden
        self.disc_hidden = disc_hidden
        self.lr_gen = lr_gen
        self.lr_disc = lr_disc
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
        
        # Inicializar modelos
        self.generator = _Generator(
            input_dim=self.input_dim,
            noise_dim=self.noise_dim,
            output_dim=self.output_dim,
            hidden_layers=self.gen_hidden
        ).to(self.device)
        
        self.discriminator = _Discriminator(
            input_dim=self.input_dim,
            target_dim=self.output_dim,
            hidden_layers=self.disc_hidden
        ).to(self.device)
        
        # Historial de pérdidas
        self.gen_losses = []
        self.disc_losses = []
        self.mse_history = []
        
    def _cargar_datos(self):
        """
        Carga datos de entrada desde archivo CSV/Excel.
        """
        if self.ruta.endswith('.xlsx'):
            df = pd.read_excel(self.ruta)
        else:
            df = pd.read_csv(self.ruta)
            
        # Detectar columnas automáticamente
        cols_entrada = [c for c in df.columns if c not in [f'q{i}' for i in range(10)] and c not in self.cols_excluir]
        cols_salida = [f'q{i}' for i in range(10) if f'q{i}' in df.columns]
        
        X = df[cols_entrada].to_numpy(dtype=np.float32)
        Y = df[cols_salida].to_numpy(dtype=np.float32)
        return X, Y
    
    def _dividir_datos(self):
        """
        Divide los datos en conjuntos de entrenamiento y prueba.
        """
        if self.test_size <= 0:
            return self.X, np.empty((0, self.X.shape[1])), self.Y, np.empty((0, self.Y.shape[1]))
        
        # Para conjuntos grandes, usar división aleatoria
        if len(self.X) > 5000:
            from sklearn.model_selection import train_test_split
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
    
    def _genera_ruido(self, batch_size):
        """
        Genera un vector de ruido aleatorio para el generador.
        
        Parámetros:
            batch_size (int): Tamaño del lote para el que generar ruido.
            
        Retorna:
            torch.Tensor: Tensor de ruido con forma (batch_size, noise_dim).
        """
        return torch.randn(batch_size, self.noise_dim, device=self.device)
    
    def entrenar(self):
        """
        Entrena el modelo CGAN durante el número de épocas especificado.
        """
        X_train, X_test, Y_train, Y_test = self._dividir_datos()
        
        # Preparar dataloaders
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(Y_train, dtype=torch.float32)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Optimizadores
        optimizer_g = optim.Adam(self.generator.parameters(), lr=self.lr_gen, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.lr_disc, betas=(0.5, 0.999))
        
        # Criterios de pérdida
        adversarial_loss = nn.BCELoss()
        content_loss = nn.MSELoss()
        
        # Etiquetas para el entrenamiento
        real_label = 1.0
        fake_label = 0.0
        
        # Historial de pérdidas
        gen_losses = []
        disc_losses = []
        mse_history = []
        
        # Bucle de entrenamiento
        for epoch in range(1, self.epochs + 1):
            running_disc_loss = 0.0
            running_gen_loss = 0.0
            running_mse = 0.0
            batches = 0
            
            for x_real, y_real in train_loader:
                batch_size = x_real.size(0)
                batches += 1
                
                # Transferir datos al dispositivo
                x_real = x_real.to(self.device)
                y_real = y_real.to(self.device)
                
                # ---------------------
                # Entrenar Discriminador
                # ---------------------
                optimizer_d.zero_grad()
                
                # Datos reales
                real_target = torch.full((batch_size, 1), real_label, device=self.device)
                output_real = self.discriminator(x_real, y_real)
                loss_real = adversarial_loss(output_real, real_target)
                
                # Datos generados
                noise = self._genera_ruido(batch_size)
                y_fake = self.generator(x_real, noise)
                fake_target = torch.full((batch_size, 1), fake_label, device=self.device)
                output_fake = self.discriminator(x_real, y_fake.detach())
                loss_fake = adversarial_loss(output_fake, fake_target)
                
                # Pérdida total del discriminador
                loss_d = loss_real + loss_fake
                loss_d.backward()
                optimizer_d.step()
                
                # ---------------------
                # Entrenar Generador
                # ---------------------
                optimizer_g.zero_grad()
                
                # Generar nuevas salidas y calcular pérdida adversarial
                output = self.discriminator(x_real, y_fake)
                loss_g_adv = adversarial_loss(output, real_target)
                
                # Añadir pérdida de contenido (MSE)
                loss_g_content = content_loss(y_fake, y_real)
                
                # Pérdida total del generador (combina adversarial y contenido)
                loss_g = loss_g_adv + 10 * loss_g_content
                loss_g.backward()
                optimizer_g.step()
                
                # Acumular pérdidas
                running_disc_loss += loss_d.item()
                running_gen_loss += loss_g.item()
                running_mse += loss_g_content.item()
            
            # Registrar pérdidas promedio
            avg_disc_loss = running_disc_loss / batches
            avg_gen_loss = running_gen_loss / batches
            avg_mse = running_mse / batches
            
            self.disc_losses.append(avg_disc_loss)
            self.gen_losses.append(avg_gen_loss)
            self.mse_history.append(avg_mse)
            
            # Mostrar progreso cada 10 épocas
            if epoch % 10 == 0 or epoch == self.epochs:
                print(f"Época {epoch:3d}: D Loss: {avg_disc_loss:.6f}, G Loss: {avg_gen_loss:.6f}, MSE: {avg_mse:.6f}")
                
                # Evaluar en conjunto de prueba si existe
                if len(X_test) > 0:
                    with torch.no_grad():
                        x_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
                        y_test = torch.tensor(Y_test, dtype=torch.float32).to(self.device)
                        noise = self._genera_ruido(len(x_test))
                        y_pred = self.generator(x_test, noise)
                        test_mse = content_loss(y_pred, y_test).item()
                        print(f"          Test MSE: {test_mse:.6f}")
    
    def graficar_perdidas(self):
        """
        Grafica la evolución de las pérdidas durante el entrenamiento.
        """
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.gen_losses, label='Generador')
        plt.plot(self.disc_losses, label='Discriminador')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.title('Pérdidas de GAN')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.mse_history, color='green', label='MSE')
        plt.xlabel('Época')
        plt.ylabel('MSE')
        plt.title('Error Cuadrático Medio')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def generar_soluciones(self, X, n_soluciones=5):
        """
        Genera múltiples posibles soluciones para un conjunto de posiciones.
        
        Parámetros:
            X (np.ndarray): Matriz de posiciones (n_muestras, input_dim).
            n_soluciones (int): Número de soluciones a generar.
            
        Retorna:
            np.ndarray: Conjunto de soluciones generadas con forma (n_soluciones, n_muestras, n_articulaciones).
        """
        # Asegurar que X es un array 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        # Comprobar que X tiene las dimensiones de entrada correctas
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Las entradas deben tener {self.input_dim} dimensiones, pero se proporcionaron {X.shape[1]}")
        
        self.generator.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            soluciones = []
            
            for _ in range(n_soluciones):
                # Generar ruido diferente para cada solución
                noise = self._genera_ruido(len(X_tensor))
                Y_pred = self.generator(X_tensor, noise).cpu().numpy()
                soluciones.append(Y_pred)
                
        return np.array(soluciones)
    
    def comparar_soluciones(self, posicion, n_soluciones=5, mostrar_grafica=True):
        """
        Genera y compara múltiples soluciones para una posición específica.
        
        Parámetros:
            posicion (np.ndarray): Vector de posición del efector final.
            n_soluciones (int): Número de soluciones a generar.
            mostrar_grafica (bool): Si es True, muestra una gráfica comparativa.
            
        Retorna:
            np.ndarray: Matriz con las soluciones generadas.
        """
        # Asegurarse de que posicion es un array 2D
        if len(posicion.shape) == 1:
            posicion = posicion.reshape(1, -1)
        
        # Generar soluciones
        soluciones = self.generar_soluciones(posicion, n_soluciones)
        
        # Mostrar soluciones numéricas
        print(f"Comparación de {n_soluciones} soluciones para la posición:")
        print(f"Posición: {posicion.flatten()}")
        print("\nSoluciones generadas (ángulos en radianes):")
        
        for i, solucion in enumerate(soluciones):
            print(f"Solución {i+1}:")
            for j, angulo in enumerate(solucion[0]):
                print(f"  q{j}: {angulo:.6f}")
            print("")
        
        # Calcular varianza entre soluciones
        varianza = np.var(soluciones, axis=0)[0]
        print("Varianza entre soluciones por articulación:")
        for j, var in enumerate(varianza):
            print(f"  q{j}: {var:.6f}")
        
        # Visualización gráfica
        if mostrar_grafica and n_soluciones > 1:
            # Preparar datos para gráfica
            soluciones_flat = soluciones[:, 0, :]  # Extraer soluciones (n_soluciones, n_articulaciones)
            
            # Crear gráfica
            plt.figure(figsize=(12, 6))
            
            # Gráfica por articulación con diferentes soluciones en colores
            ax1 = plt.subplot(1, 2, 1)
            x = np.arange(self.output_dim)  # Articulaciones en el eje X
            width = 0.8 / n_soluciones
            offset = width * np.arange(n_soluciones) - width * (n_soluciones - 1) / 2
            
            for i in range(n_soluciones):
                ax1.bar(x + offset[i], soluciones_flat[i], 
                        width=width, label=f'Sol {i+1}')
            
            ax1.set_xlabel('Articulación')
            ax1.set_ylabel('Ángulo (radianes)')
            ax1.set_xticks(x)
            ax1.set_xticklabels([f'q{i}' for i in range(self.output_dim)])
            ax1.set_title('Comparación de soluciones por articulación')
            ax1.legend()
            
            # Gráfica de varianza
            ax2 = plt.subplot(1, 2, 2)
            ax2.bar(np.arange(self.output_dim), np.sqrt(varianza), color='orange')
            ax2.set_xlabel('Articulación')
            ax2.set_ylabel('Desviación estándar (radianes)')
            ax2.set_xticks(np.arange(self.output_dim))
            ax2.set_xticklabels([f'q{j}' for j in range(self.output_dim)])
            ax2.set_title('Variabilidad entre soluciones')
            
            plt.tight_layout()
            plt.show()
        
        return soluciones
    
    def graficar_offsets(self, pmax=10000):
        """
        Visualiza el error entre valores reales y predichos usando MDS.
        """
        self.generator.eval()
        with torch.no_grad():
            # Generar una predicción
            X_tensor = torch.tensor(self.X, dtype=torch.float32).to(self.device)
            noise = self._genera_ruido(len(X_tensor))
            Y_pred = self.generator(X_tensor, noise).cpu().numpy()
        
        # Calcular distancias
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
        
        # Aplicar MDS
        concat = np.vstack([Y_true, Y_pred])
        mds = MDS(n_components=2, dissimilarity='euclidean', random_state=0, n_jobs=-1)
        emb = mds.fit_transform(concat)
        y_true_2d = emb[:len(Y_true)]
        y_pred_2d = emb[len(Y_true):]
        
        # Calcular stress
        self.stress = mds.stress_
        self.stress_norm = self.stress / np.sum(pdist(concat) ** 2 / 2)
        
        # Visualización
        plt.figure(figsize=(10, 10), dpi=300)
        
        # Dibujar líneas entre pares
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
        Evalúa el modelo en todo el conjunto de datos.
        """
        self.generator.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(self.X, dtype=torch.float32).to(self.device)
            noise = self._genera_ruido(len(X_tensor))
            Y_pred = self.generator(X_tensor, noise).cpu().numpy()
        
        # Dividir datos
        X_train, X_test, Y_train, Y_test = self._dividir_datos()
        
        # Evaluar en cada conjunto
        with torch.no_grad():
            # Predicciones de entrenamiento
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
            noise_train = self._genera_ruido(len(X_train_tensor))
            Y_pred_train = self.generator(X_train_tensor, noise_train).cpu().numpy()
            
            # Predicciones de prueba si existen
            if len(X_test) > 0:
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
                noise_test = self._genera_ruido(len(X_test_tensor))
                Y_pred_test = self.generator(X_test_tensor, noise_test).cpu().numpy()
            else:
                Y_pred_test = np.empty((0, self.output_dim))
        
        # Calcular métricas
        mse_total = mean_squared_error(self.Y, Y_pred)
        mse_deg_total = mean_squared_error(np.rad2deg(self.Y), np.rad2deg(Y_pred))
        r2_total = r2_score(self.Y, Y_pred)
        
        # Error por articulación
        error_por_artic = np.mean((self.Y - Y_pred)**2, axis=0)
        
        # Métricas para cada conjunto
        results = {
            'total': {
                'mse_rad': mse_total,
                'mse_deg': mse_deg_total,
                'r2': r2_total,
                'Y_pred': Y_pred
            },
            'error_por_artic': error_por_artic
        }
        
        # Añadir resultados de entrenamiento
        if len(X_train) > 0:
            mse_train = mean_squared_error(Y_train, Y_pred_train)
            mse_deg_train = mean_squared_error(np.rad2deg(Y_train), np.rad2deg(Y_pred_train))
            r2_train = r2_score(Y_train, Y_pred_train)
            
            results['train'] = {
                'mse_rad': mse_train,
                'mse_deg': mse_deg_train,
                'r2': r2_train,
                'Y_pred': Y_pred_train
            }
        
        # Añadir resultados de prueba
        if len(X_test) > 0:
            mse_test = mean_squared_error(Y_test, Y_pred_test)
            mse_deg_test = mean_squared_error(np.rad2deg(Y_test), np.rad2deg(Y_pred_test))
            r2_test = r2_score(Y_test, Y_pred_test)
            
            results['test'] = {
                'mse_rad': mse_test,
                'mse_deg': mse_deg_test,
                'r2': r2_test,
                'Y_pred': Y_pred_test
            }
        
        # Imprimir métricas globales
        print(f"MSE total (radianes²): {mse_total:.6f}")
        print(f"MSE total (grados²):   {mse_deg_total:.6f}")
        print(f"R²: {r2_total:.6f}")
        
        return results
    
    def guardar(self, nombre_base="modelo_cgan"):
        """
        Guarda el modelo entrenado en disco.
        """
        import os
        os.makedirs(nombre_base, exist_ok=True)
        
        # Guardar modelos
        torch.save(self.generator.state_dict(), f"{nombre_base}/generador.pt")
        torch.save(self.discriminator.state_dict(), f"{nombre_base}/discriminador.pt")
        
        # Guardar configuración e historial de entrenamiento
        import json
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'noise_dim': self.noise_dim,
            'gen_hidden': self.gen_hidden,
            'disc_hidden': self.disc_hidden,
            'lr_gen': self.lr_gen,
            'lr_disc': self.lr_disc,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }
        
        with open(f"{nombre_base}/config.json", 'w') as f:
            json.dump(config, f)
            
        # Guardar historial de pérdidas
        import numpy as np
        np.savez(f"{nombre_base}/historial.npz", 
                 gen_losses=self.gen_losses,
                 disc_losses=self.disc_losses,
                 mse_history=self.mse_history)
        
        print(f"Modelo y configuración guardados en carpeta: {nombre_base}")
    
    def cargar(self, nombre_base="modelo_cgan"):
        """
        Carga un modelo previamente guardado.
        """
        # Cargar configuración
        import json
        with open(f"{nombre_base}/config.json", 'r') as f:
            config = json.load(f)
        
        # Actualizar parámetros
        self.noise_dim = config['noise_dim']
        self.gen_hidden = tuple(config['gen_hidden'])
        self.disc_hidden = tuple(config['disc_hidden'])
        
        # Recrear modelos
        self.generator = _Generator(
            input_dim=self.input_dim,
            noise_dim=self.noise_dim,
            output_dim=self.output_dim,
            hidden_layers=self.gen_hidden
        ).to(self.device)
        
        self.discriminator = _Discriminator(
            input_dim=self.input_dim,
            target_dim=self.output_dim,
            hidden_layers=self.disc_hidden
        ).to(self.device)
        
        # Cargar pesos
        self.generator.load_state_dict(torch.load(f"{nombre_base}/generador.pt", map_location=self.device))
        self.discriminator.load_state_dict(torch.load(f"{nombre_base}/discriminador.pt", map_location=self.device))
        
        # Cargar historial
        import numpy as np
        historial = np.load(f"{nombre_base}/historial.npz")
        self.gen_losses = historial['gen_losses'].tolist()
        self.disc_losses = historial['disc_losses'].tolist()
        self.mse_history = historial['mse_history'].tolist()
        
        print(f"Modelo cargado correctamente desde: {nombre_base}")
    
    def summary(self):
        """
        Imprime un resumen del modelo y su desempeño, incluyendo errores L1 y L2 por articulación.
        """
        self.generator.eval()
        with torch.no_grad():
            # Predicciones globales
            X_tensor = torch.tensor(self.X, dtype=torch.float32).to(self.device)
            noise = self._genera_ruido(len(X_tensor))
            Y_pred = self.generator(X_tensor, noise).cpu().numpy()
            
            # Dividir datos para métricas por conjunto
            X_train, X_test, Y_train, Y_test = self._dividir_datos()
            
            # Predicciones de entrenamiento
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
            noise_train = self._genera_ruido(len(X_train_tensor))
            Y_pred_train = self.generator(X_train_tensor, noise_train).cpu().numpy()
            
            # Predicciones de prueba
            if len(X_test) > 0:
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
                noise_test = self._genera_ruido(len(X_test_tensor))
                Y_pred_test = self.generator(X_test_tensor, noise_test).cpu().numpy()
        
        # Calcular métricas L1 y L2 por articulación
        error_l1_global = np.mean(np.abs(self.Y - Y_pred), axis=0)
        error_l2_global = np.sqrt(np.mean((self.Y - Y_pred)**2, axis=0))
        
        error_l1_train = np.mean(np.abs(Y_train - Y_pred_train), axis=0)
        error_l2_train = np.sqrt(np.mean((Y_train - Y_pred_train)**2, axis=0))
        
        if len(X_test) > 0:
            error_l1_test = np.mean(np.abs(Y_test - Y_pred_test), axis=0)
            error_l2_test = np.sqrt(np.mean((Y_test - Y_pred_test)**2, axis=0))
        
        # Calcular métricas globales
        mse_total = mean_squared_error(self.Y, Y_pred)
        mse_deg_total = mean_squared_error(np.rad2deg(self.Y), np.rad2deg(Y_pred))
        
        mse_train = mean_squared_error(Y_train, Y_pred_train)
        mse_deg_train = mean_squared_error(np.rad2deg(Y_train), np.rad2deg(Y_pred_train))
        
        if len(X_test) > 0:
            mse_test = mean_squared_error(Y_test, Y_pred_test)
            mse_deg_test = mean_squared_error(np.rad2deg(Y_test), np.rad2deg(Y_pred_test))
        
        # Imprimir información
        print("=" * 50)
        print("Modelo: CGAN")
        print(f"Arquitectura Generador: {self.input_dim}+{self.noise_dim}-{'-'.join(map(str, self.gen_hidden))}-{self.output_dim}")
        print(f"Arquitectura Discriminador: {self.input_dim}+{self.output_dim}-{'-'.join(map(str, self.disc_hidden))}-1")
        print(f"Parámetros: lr_gen={self.lr_gen}, lr_disc={self.lr_disc}, batch_size={self.batch_size}, epochs={self.epochs}")
        
        # Métricas globales
        print("\nMétricas globales (todo el conjunto de datos):")
        print(f"MSE (radianes): {mse_total:.6f}")
        print(f"MSE (grados): {mse_deg_total:.6f}")
        
        # Métricas por conjunto
        print("\nMétricas de entrenamiento:")
        print(f"MSE (radianes): {mse_train:.6f}")
        print(f"MSE (grados): {mse_deg_train:.6f}")
        
        if len(X_test) > 0:
            print("\nMétricas de prueba:")
            print(f"MSE (radianes): {mse_test:.6f}")
            print(f"MSE (grados): {mse_deg_test:.6f}")
        
        # Error por articulación - Global
        print("\nError por articulación (global):")
        for i in range(self.output_dim):
            print(f"  q{i}: L1={error_l1_global[i]:.6f}, L2={error_l2_global[i]:.6f}")
        
        # Error por articulación - Entrenamiento
        print("\nError por articulación (entrenamiento):")
        for i in range(self.output_dim):
            print(f"  q{i}: L1={error_l1_train[i]:.6f}, L2={error_l2_train[i]:.6f}")
        
        # Error por articulación - Prueba
        if len(X_test) > 0:
            print("\nError por articulación (prueba):")
            for i in range(self.output_dim):
                print(f"  q{i}: L1={error_l1_test[i]:.6f}, L2={error_l2_test[i]:.6f}")
        
        # Información de MDS
        if hasattr(self, "stress") and self.stress is not None:
            print(f"\nMDS stress: {self.stress:.6f}")
            print(f"MDS stress normalizado: {self.stress_norm:.6f}")
        
        print("=" * 50)

class CGANInversaUR5:
    """
    Clase para entrenar y evaluar un modelo CGAN para el problema de cinemática inversa del robot UR5.
    Permite cargar datos, optimizar hiperparámetros con Optuna, entrenar el modelo, evaluar su desempeño,
    generar múltiples soluciones válidas y visualizar resultados.
    
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
        self.input_dim = self.X.shape[1]
        self.output_dim = self.Y.shape[1]
        
        # Inicializar a None, se crearán durante la optimización/entrenamiento
        self.generator = None
        self.discriminator = None
        self.noise_dim = None
        
        # Historial y resultados
        self.gen_losses = []
        self.disc_losses = []
        self.mse_history = []
        self.study = None
        self.stress = None
        self.stress_norm = None
        
    def _cargar_datos(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga los datos desde un archivo CSV o Excel.
        
        Retorna:
            Tuple[np.ndarray, np.ndarray]: Una tupla con los datos de entrada (X) y salida (Y).
        """
        if self.ruta.endswith('.xlsx'):
            df = pd.read_excel(self.ruta)
        else:
            df = pd.read_csv(self.ruta)
            
        # Detectar columnas automáticamente
        cols_entrada = [c for c in df.columns if c not in [f'q{i}' for i in range(10)] and c not in self.cols_excluir]
        cols_salida = [f'q{i}' for i in range(10) if f'q{i}' in df.columns]
        
        X = df[cols_entrada].to_numpy(dtype=np.float32)
        Y = df[cols_salida].to_numpy(dtype=np.float32)
        return X, Y
    
    def _espacio(self, t) -> Dict[str, Any]:
        """
        Define el espacio de búsqueda para la optimización de hiperparámetros.
        
        Parámetros:
            t: Un objeto trial de Optuna.
            
        Retorna:
            Dict[str, Any]: Un diccionario con los hiperparámetros sugeridos.
        """
        # Opciones para las capas ocultas del generador
        gen_options = {
            "128-64": (128, 64),
            "256-128": (256, 128),
            "512-256": (512, 256),
            "256-128-64": (256, 128, 64),
            "512-256-128": (512, 256, 128)
        }
        
        # Opciones para las capas ocultas del discriminador
        disc_options = {
            "64-32": (64, 32),
            "128-64": (128, 64),
            "256-128": (256, 128),
            "128-64-32": (128, 64, 32)
        }
        
        gen_key = t.suggest_categorical("gen_key", list(gen_options.keys()))
        disc_key = t.suggest_categorical("disc_key", list(disc_options.keys()))
        
        return {
            "noise_dim": t.suggest_int("noise_dim", 5, 50),
            "gen_hidden": gen_options[gen_key],
            "disc_hidden": disc_options[disc_key],
            "lr_gen": t.suggest_float("lr_gen", 1e-5, 1e-3, log=True),
            "lr_disc": t.suggest_float("lr_disc", 1e-5, 1e-3, log=True),
            "batch": t.suggest_categorical("batch", [32, 64, 128]),
            "content_weight": t.suggest_float("content_weight", 5.0, 20.0),
            "epochs": t.suggest_int("epochs", 50, 300)
        }
    
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
            from sklearn.model_selection import train_test_split
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
    
    def _genera_ruido(self, batch_size: int) -> torch.Tensor:
        """
        Genera un tensor de ruido aleatorio para el generador.
        
        Parámetros:
            batch_size (int): Tamaño del lote.
            
        Retorna:
            torch.Tensor: Tensor de ruido.
        """
        return torch.randn(batch_size, self.noise_dim, device=self.device)
    
    def _train_once(self, p: Dict[str, Any], Xtr: np.ndarray, Ytr: np.ndarray, 
                   Xval: np.ndarray, Yval: np.ndarray) -> Tuple[float, _Generator, _Discriminator]:
        """
        Entrena un modelo CGAN una vez con los parámetros dados.
        
        Parámetros:
            p (Dict[str, Any]): Diccionario de hiperparámetros.
            Xtr (np.ndarray): Datos de entrada para entrenamiento.
            Ytr (np.ndarray): Datos de salida para entrenamiento.
            Xval (np.ndarray): Datos de entrada para validación.
            Yval (np.ndarray): Datos de salida para validación.
            
        Retorna:
            Tuple: MSE en validación, generador entrenado, discriminador entrenado.
        """
        # Actualizar dimensión de ruido
        self.noise_dim = p["noise_dim"]
        
        # Inicializar modelos
        generator = _Generator(
            input_dim=self.input_dim,
            noise_dim=self.noise_dim,
            output_dim=self.output_dim,
            hidden_layers=p["gen_hidden"]
        ).to(self.device)
        
        discriminator = _Discriminator(
            input_dim=self.input_dim,
            target_dim=self.output_dim,
            hidden_layers=p["disc_hidden"]
        ).to(self.device)
        
        # Preparar dataset y dataloader
        train_dataset = TensorDataset(
            torch.tensor(Xtr, dtype=torch.float32),
            torch.tensor(Ytr, dtype=torch.float32)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=p["batch"], shuffle=True
        )
        
        # Optimizadores
        optimizer_g = optim.Adam(generator.parameters(), lr=p["lr_gen"], betas=(0.5, 0.999))
        optimizer_d = optim.Adam(discriminator.parameters(), lr=p["lr_disc"], betas=(0.5, 0.999))
        
        # Criterios de pérdida
        adversarial_loss = nn.BCELoss()
        content_loss = nn.MSELoss()
        
        # Etiquetas para el entrenamiento
        real_label = 1.0
        fake_label = 0.0
        
        # Historial de pérdidas
        gen_losses = []
        disc_losses = []
        mse_history = []
        
        # Bucle de entrenamiento
        for epoch in range(p["epochs"]):
            running_disc_loss = 0.0
            running_gen_loss = 0.0
            running_mse = 0.0
            batches = 0
            
            for x_real, y_real in train_loader:
                batch_size = x_real.size(0)
                batches += 1
                
                # Transferir datos al dispositivo
                x_real = x_real.to(self.device)
                y_real = y_real.to(self.device)
                
                # ---------------------
                # Entrenar Discriminador
                # ---------------------
                optimizer_d.zero_grad()
                
                # Datos reales
                real_target = torch.full((batch_size, 1), real_label, device=self.device)
                output_real = discriminator(x_real, y_real)
                loss_real = adversarial_loss(output_real, real_target)
                
                # Datos generados
                noise = self._genera_ruido(batch_size)
                y_fake = generator(x_real, noise)
                fake_target = torch.full((batch_size, 1), fake_label, device=self.device)
                output_fake = discriminator(x_real, y_fake.detach())
                loss_fake = adversarial_loss(output_fake, fake_target)
                
                # Pérdida total del discriminador
                loss_d = loss_real + loss_fake
                loss_d.backward()
                optimizer_d.step()
                
                # ---------------------
                # Entrenar Generador
                # ---------------------
                optimizer_g.zero_grad()
                
                # Generar nuevas salidas y calcular pérdida adversarial
                output = discriminator(x_real, y_fake)
                loss_g_adv = adversarial_loss(output, real_target)
                
                # Añadir pérdida de contenido (MSE)
                loss_g_content = content_loss(y_fake, y_real)
                
                # Pérdida total del generador (combina adversarial y contenido)
                loss_g = loss_g_adv + p["content_weight"] * loss_g_content
                loss_g.backward()
                optimizer_g.step()
                
                # Acumular pérdidas
                running_disc_loss += loss_d.item()
                running_gen_loss += loss_g.item()
                running_mse += loss_g_content.item()
            
            # Registrar pérdidas promedio
            avg_disc_loss = running_disc_loss / batches
            avg_gen_loss = running_gen_loss / batches
            avg_mse = running_mse / batches
            
            disc_losses.append(avg_disc_loss)
            gen_losses.append(avg_gen_loss)
            mse_history.append(avg_mse)
        
        # Guardar historial de pérdidas
        self.gen_losses = gen_losses
        self.disc_losses = disc_losses
        self.mse_history = mse_history
        
        # Evaluación final en validación
        val_mse = 0
        if len(Xval) > 0:
            generator.eval()
            with torch.no_grad():
                X_val_tensor = torch.tensor(Xval, dtype=torch.float32).to(self.device)
                Y_val_tensor = torch.tensor(Yval, dtype=torch.float32).to(self.device)
                noise = self._genera_ruido(len(X_val_tensor))
                Y_pred = generator(X_val_tensor, noise)
                val_mse = mean_squared_error(np.rad2deg(Yval), np.rad2deg(Y_pred.cpu().numpy()))
        else:
            # Si no hay datos de validación, evaluar en entrenamiento
            generator.eval()
            with torch.no_grad():
                X_tr_tensor = torch.tensor(Xtr, dtype=torch.float32).to(self.device)
                Y_tr_tensor = torch.tensor(Ytr, dtype=torch.float32).to(self.device)
                noise = self._genera_ruido(len(X_tr_tensor))
                Y_pred = generator(X_tr_tensor, noise)
                val_mse = mean_squared_error(np.rad2deg(Ytr), np.rad2deg(Y_pred.cpu().numpy()))
        
        return val_mse, generator, discriminator
    
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
        mse, _, _ = self._train_once(p, Xtr, Ytr, Xval, Yval)
        return -mse  # Maximizar el negativo del MSE (minimizar MSE)
    
    def optimizar(self, n_trials=30, nombre_est="estudio_ur5_cgan"):
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
        Entrena el modelo CGAN con los mejores hiperparámetros encontrados.
        
        Retorna:
            float: MSE en grados².
        """
        if self.study is None:
            raise RuntimeError("Ejecute optimizar() primero.")
            
        p = {**self._espacio(self.study.best_trial), **self.study.best_params}
        Xtr, Xval, Ytr, Yval = self._dividir_datos()
        mse, generator, discriminator = self._train_once(p, Xtr, Ytr, Xval, Yval)
        
        self.generator = generator
        self.discriminator = discriminator
        self.noise_dim = p["noise_dim"]
        
        print(f"MSE grados² validación: {mse:.6f}")
        return mse
    
    def entrenar(self, params=None):
        """
        Entrena el modelo CGAN con parámetros específicos.
        
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
        mse, generator, discriminator = self._train_once(params, Xtr, Ytr, Xval, Yval)
        
        self.generator = generator
        self.discriminator = discriminator
        self.noise_dim = params["noise_dim"]
        
        print(f"MSE grados² validación: {mse:.6f}")
        return mse
    
    def graficar_perdidas(self):
        """
        Grafica la evolución de las pérdidas durante el entrenamiento.
        """
        if not self.gen_losses or not self.disc_losses:
            raise RuntimeError("No hay historial de entrenamiento disponible.")
            
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.gen_losses, label='Generador')
        plt.plot(self.disc_losses, label='Discriminador')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.title('Pérdidas de GAN')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.mse_history, color='green', label='MSE')
        plt.xlabel('Época')
        plt.ylabel('MSE')
        plt.title('Error Cuadrático Medio')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def graficar_offsets(self, pmax=10000):
        """
        Visualiza el error entre valores reales y predichos usando MDS.
        
        Parámetros:
            pmax (int): Número máximo de puntos para el análisis MDS.
        """
        if self.generator is None:
            raise RuntimeError("Modelo no entrenado.")
        
        # Evaluación del modelo
        self.generator.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(self.X, dtype=torch.float32).to(self.device)
            noise = self._genera_ruido(len(X_tensor))
            Y_pred = self.generator(X_tensor, noise).cpu().numpy()
        
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
    
    def evaluar_total(self):
        """
        Evalúa el modelo en todo el conjunto de datos.
        
        Retorna:
            Dict: Resultados de la evaluación.
        """
        if self.generator is None:
            raise RuntimeError("Modelo no entrenado.")
        
        # Evaluación sobre todo el conjunto
        self.generator.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(self.X, dtype=torch.float32).to(self.device)
            noise = self._genera_ruido(len(X_tensor))
            Y_pred_total = self.generator(X_tensor, noise).cpu().numpy()
        
        mse_rad_total = mean_squared_error(self.Y, Y_pred_total)
        mse_deg_total = mean_squared_error(np.rad2deg(self.Y), np.rad2deg(Y_pred_total))
        r2_total = r2_score(self.Y, Y_pred_total)
        
        # Error por articulación
        error_por_artic = np.mean((self.Y - Y_pred_total)**2, axis=0)
        
        # Resultados
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
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
                noise_train = self._genera_ruido(len(X_train_tensor))
                Y_pred_train = self.generator(X_train_tensor, noise_train).cpu().numpy()
            
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
                noise_test = self._genera_ruido(len(X_test_tensor))
                Y_pred_test = self.generator(X_test_tensor, noise_test).cpu().numpy()
            
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
    
    def generar_soluciones(self, X, n_soluciones=5):
        """
        Genera múltiples posibles soluciones para un conjunto de posiciones.
        
        Parámetros:
            X (np.ndarray): Matriz de posiciones.
            n_soluciones (int): Número de soluciones a generar.
            
        Retorna:
            np.ndarray: Conjunto de soluciones generadas.
        """
        if self.generator is None:
            raise RuntimeError("Modelo no entrenado.")
            
        # Asegurar que X es un array 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        # Comprobar que X tiene las dimensiones de entrada correctas
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Las entradas deben tener {self.input_dim} dimensiones, pero se proporcionaron {X.shape[1]}")
        
        self.generator.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            soluciones = []
            
            for _ in range(n_soluciones):
                # Generar ruido diferente para cada solución
                noise = self._genera_ruido(len(X_tensor))
                Y_pred = self.generator(X_tensor, noise).cpu().numpy()
                soluciones.append(Y_pred)
                
        return np.array(soluciones)
    
    def comparar_soluciones(self, posicion, n_soluciones=5, mostrar_grafica=True):
        """
        Genera y compara múltiples soluciones para una posición específica.
        
        Parámetros:
            posicion (np.ndarray): Vector de posición del efector final.
            n_soluciones (int): Número de soluciones a generar.
            mostrar_grafica (bool): Si es True, muestra una gráfica comparativa.
            
        Retorna:
            np.ndarray: Matriz con las soluciones generadas.
        """
        if self.generator is None:
            raise RuntimeError("Modelo no entrenado.")
            
        # Asegurarse de que posicion es un array 2D
        if len(posicion.shape) == 1:
            posicion = posicion.reshape(1, -1)
        
        # Generar soluciones
        soluciones = self.generar_soluciones(posicion, n_soluciones)
        
        # Mostrar soluciones numéricas
        print(f"Comparación de {n_soluciones} soluciones para la posición:")
        print(f"Posición: {posicion.flatten()}")
        print("\nSoluciones generadas (ángulos en radianes):")
        
        for i, solucion in enumerate(soluciones):
            print(f"Solución {i+1}:")
            for j, angulo in enumerate(solucion[0]):
                print(f"  q{j}: {angulo:.6f}")
            print("")
        
        # Calcular varianza entre soluciones
        varianza = np.var(soluciones, axis=0)[0]
        print("Varianza entre soluciones por articulación:")
        for j, var in enumerate(varianza):
            print(f"  q{j}: {var:.6f}")
        
        # Visualización gráfica
        if mostrar_grafica and n_soluciones > 1:
            # Preparar datos para gráfica
            soluciones_flat = soluciones[:, 0, :]  # Extraer soluciones (n_soluciones, n_articulaciones)
            
            # Crear gráfica
            plt.figure(figsize=(12, 6))
            
            # Gráfica por articulación con diferentes soluciones en colores
            ax1 = plt.subplot(1, 2, 1)
            x = np.arange(self.output_dim)  # Articulaciones en el eje X
            width = 0.8 / n_soluciones
            offset = width * np.arange(n_soluciones) - width * (n_soluciones - 1) / 2
            
            for i in range(n_soluciones):
                ax1.bar(x + offset[i], soluciones_flat[i], 
                        width=width, label=f'Sol {i+1}')
            
            ax1.set_xlabel('Articulación')
            ax1.set_ylabel('Ángulo (radianes)')
            ax1.set_xticks(x)
            ax1.set_xticklabels([f'q{i}' for i in range(self.output_dim)])
            ax1.set_title('Comparación de soluciones por articulación')
            ax1.legend()
            
            # Gráfica de varianza
            ax2 = plt.subplot(1, 2, 2)
            ax2.bar(np.arange(self.output_dim), np.sqrt(varianza), color='orange')
            ax2.set_xlabel('Articulación')
            ax2.set_ylabel('Desviación estándar (radianes)')
            ax2.set_xticks(np.arange(self.output_dim))
            ax2.set_xticklabels([f'q{j}' for j in range(self.output_dim)])
            ax2.set_title('Variabilidad entre soluciones')
            
            plt.tight_layout()
            plt.show()
        
        return soluciones
        
    def summary(self):
        """
        Muestra un resumen detallado del modelo CGAN y su desempeño.
        """
        if self.generator is None:
            raise RuntimeError("Modelo no entrenado.")
        
        results = self.evaluar_total()
        
        # Obtener los mejores parámetros
        if self.study is not None:
            p = {**self._espacio(self.study.best_trial), **self.study.best_params}
        else:
            p = {"noise_dim": self.noise_dim}
        
        print("=" * 50)
        print("Modelo: CGAN")
        print(f"Arquitectura Generador: {self.input_dim}+{self.noise_dim}->[ocultas]->{self.output_dim}")
        print(f"Parámetros: {p}")
        print(f"Opt: {'Optuna' if self.study is not None else 'Manual'}")
        
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
        if self.generator is None or self.discriminator is None:
            raise RuntimeError("Entrene el modelo antes de guardar.")
            
        nombre_base += "_CGAN"
        os.makedirs(nombre_base, exist_ok=True)
        
        # Guardar modelos
        torch.save(self.generator.state_dict(), f"{nombre_base}/generador.pt")
        torch.save(self.discriminator.state_dict(), f"{nombre_base}/discriminador.pt")
        
        # Guardar historial de pérdidas
        np.savez(f"{nombre_base}/historial.npz", 
                 gen_losses=self.gen_losses,
                 disc_losses=self.disc_losses,
                 mse_history=self.mse_history)
        
        # Guardar información del modelo
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'noise_dim': self.noise_dim,
        }
        
        with open(f"{nombre_base}/config.json", 'w') as f:
            json.dump(config, f)
        
        # Guardar resultados de Optuna si existen
        if self.study is not None:
            self.study.trials_dataframe().to_csv(f"{nombre_base}/optuna_resultados.csv", index=False)
            vis.plot_optimization_history(self.study)
            plt.savefig(f"{nombre_base}/opt_history.png"); plt.clf()
            vis.plot_param_importances(self.study)
            plt.savefig(f"{nombre_base}/opt_param_importance.png"); plt.clf()
        
        print(f"Modelo y resultados guardados en carpeta: {nombre_base}")
        
    def cargar(self, nombre_base="modelo_CGAN"):
        """
        Carga un modelo previamente guardado.
        
        Parámetros:
            nombre_base (str): Nombre de la carpeta donde se encuentra el modelo.
        """
        if not os.path.exists(nombre_base):
            raise RuntimeError(f"La carpeta {nombre_base} no existe.")
        
        # Cargar configuración
        with open(f"{nombre_base}/config.json", 'r') as f:
            config = json.load(f)
        
        # Actualizar dimensiones
        self.noise_dim = config.get('noise_dim', 10)
        
        # Crear modelos
        gen_hidden = (128, 64)  # Valores por defecto
        disc_hidden = (64, 32)  # Valores por defecto
        
        self.generator = _Generator(
            input_dim=self.input_dim,
            noise_dim=self.noise_dim,
            output_dim=self.output_dim,
            hidden_layers=gen_hidden
        ).to(self.device)
        
        self.discriminator = _Discriminator(
            input_dim=self.input_dim,
            target_dim=self.output_dim,
            hidden_layers=disc_hidden
        ).to(self.device)
        
        # Cargar pesos
        self.generator.load_state_dict(
            torch.load(f"{nombre_base}/generador.pt", map_location=self.device)
        )
        self.discriminator.load_state_dict(
            torch.load(f"{nombre_base}/discriminador.pt", map_location=self.device)
        )
        
        # Cargar historial de pérdidas si existe
        if os.path.exists(f"{nombre_base}/historial.npz"):
            historial = np.load(f"{nombre_base}/historial.npz")
            self.gen_losses = historial['gen_losses'].tolist()
            self.disc_losses = historial['disc_losses'].tolist()
            self.mse_history = historial['mse_history'].tolist()
        
        print(f"Modelo cargado correctamente desde: {nombre_base}")
        
    def predecir(self, X):
        """
        Realiza una predicción con el modelo entrenado.
        
        Parámetros:
            X (np.ndarray): Vector o matriz de posiciones.
            
        Retorna:
            np.ndarray: Predicción de ángulos articulares.
        """
        if self.generator is None:
            raise RuntimeError("Modelo no entrenado.")
            
        # Asegurar que X es un array 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        # Verificar dimensiones
        if X.shape[1] != self.input_dim:
            raise ValueError(f"La entrada debe tener {self.input_dim} dimensiones, pero tiene {X.shape[1]}")
        
        # Realizar predicción
        self.generator.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            noise = self._genera_ruido(len(X_tensor))
            prediction = self.generator(X_tensor, noise).cpu().numpy()
        
        # Si solo hay una muestra, devolver vector
        if len(X) == 1:
            return prediction[0]
        
        return prediction

    