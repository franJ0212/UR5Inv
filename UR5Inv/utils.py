import numpy as np
from random import choice
from math import ceil, sqrt

def select_m_representative_points(Dissim, m):
    """
    Selecciona m puntos representativos de un conjunto, maximizando la diversidad según una matriz de disimilitud.
    Parámetros
    ----------
    Dissim : numpy.ndarray
        Matriz cuadrada de disimilitud de tamaño (n, n), donde Dissim[i, j] representa la disimilitud entre los puntos i y j.
    m : int
        Número de puntos representativos a seleccionar (m <= n).
    Retorna
    -------
    selected_points : list of int
        Lista de índices de los m puntos seleccionados como representativos.
    Lanza
    -----
    ValueError
        Si m es mayor que el número de puntos (n).
        
    Notas
    -----
    El algoritmo selecciona iterativamente puntos que maximizan la distancia mínima a los ya seleccionados, 
    alternando con la selección de puntos cuya distancia mínima es cercana a la mediana de las distancias restantes,
    para asegurar diversidad y representatividad.
    """
    n = Dissim.shape[0]
    if m > n:
        raise ValueError("m cannot be greater than the number of points (n).")
    
    selected_points = []
    remaining_points = set(range(n))
    min_distances = np.full(n, np.inf)
    
    i = choice(list(remaining_points))
    selected_points.append(i)
    remaining_points.remove(i)
    min_distances = np.minimum(min_distances, Dissim[i])
    
    while len(selected_points) < m:
        j_prime = np.argmax(min_distances[list(remaining_points)])
        j_prime = list(remaining_points)[j_prime]
        
        selected_points.append(j_prime)
        remaining_points.remove(j_prime)
        if len(selected_points) >= m:
            break
            
        min_distances = np.minimum(min_distances, Dissim[j_prime])
        
        remaining_distances = min_distances[list(remaining_points)]
        median_distance = np.median(remaining_distances)
        
        i = list(remaining_points)[np.argmin(np.abs(remaining_distances - median_distance))]
        
        selected_points.append(i)
        remaining_points.remove(i)
        
        min_distances = np.minimum(min_distances, Dissim[i])
    
    return selected_points

def interpolacion_Gower(dist_matrix, selected_indices, Xc_landmarks):
    """
    Interpola las coordenadas MDS usando la fórmula de Gower.

    Parámetros:
        dist_matrix      : np.array, matriz de distancias completa (n_total x n_total)
        selected_indices : array-like, índices de los landmarks
        Xc_landmarks     : np.array, coordenadas MDS de los landmarks (m x r)

    Retorna:
        Xc_interpolated  : np.array, coordenadas interpoladas para todos los puntos
    """
    N = dist_matrix.shape[0]
    m = len(selected_indices)

    # Obtener matriz de distancias al cuadrado para los puntos a interpolar
    A21 = dist_matrix[~np.isin(range(N), selected_indices)][:, selected_indices]**2

    # Calcular S (matriz de escala)
    S = (Xc_landmarks.T @ Xc_landmarks) / (m - 1)

    # Calcular q1
    P1 = np.eye(m) - (1 / m) * np.ones((m, m))
    Q1 = -0.5 * (P1 @ (dist_matrix[np.ix_(selected_indices, selected_indices)] ** 2.0) @ P1.T)
    q1 = np.diagonal(Q1).reshape(m, 1)

    try:
        # Opción 4: Usar scipy.linalg.pinvh para mayor estabilidad numérica
        S_inv = pinvh(S)
    
    except np.linalg.LinAlgError:
        print("SVD no convergió en pinvh. Probando con regularización...")

        try:
            # Opción 3: Regularizar S sumándole un término diagonal
            lambda_reg = 1e-6
            S_reg = S + lambda_reg * np.eye(S.shape[0])
            S_inv = np.linalg.inv(S_reg)

        except np.linalg.LinAlgError:
            print("S sigue siendo singular después de la regularización. Probando con pinv con rcond...")

            # Opción 2: Usar np.linalg.pinv con rcond
            S_inv = np.linalg.pinv(S, rcond=1e-6)

    # Aplicar fórmula de Gower
    X2 = ((np.ones((N - m, 1)) @ q1.T - A21) @ Xc_landmarks @ S_inv) / (2.0 * m)

    # Construir solución completa
    Xc_interpolated = np.zeros((N, Xc_landmarks.shape[1]))
    Xc_interpolated[selected_indices] = Xc_landmarks
    mask = ~np.isin(range(N), selected_indices)
    Xc_interpolated[mask] = X2

    return Xc_interpolated

def pdist_manual(X):
    """
    Calcula las distancias euclidianas entre todas las filas de X.
    Retorna el vector condensado (como pdist).
    """
    n = X.shape[0]
    D = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(X[i] - X[j])
            D.append(d)
    return np.array(D, dtype=np.float32)

def squareform_manual(dvec):
    """
    Convierte un vector de distancias condensado en una matriz simétrica completa.
    """
    n = int(ceil((1 + sqrt(1 + 8 * len(dvec))) / 2))
    D = np.zeros((n, n), dtype=np.float32)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = dvec[idx]
            D[j, i] = dvec[idx]
            idx += 1
    return D

