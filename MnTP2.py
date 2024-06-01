import sys
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import StratifiedKFold
import time 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy.linalg as npAlg
sys.path.append('build')
import my_module

# Rutas de los archivos CSV
X_train = np.loadtxt("data/X_train.csv", delimiter=",")
y_train = np.loadtxt("data/y_train.csv", delimiter=",").astype(int)
X_test = np.loadtxt("data/X_test.csv", delimiter=",")
y_test = np.loadtxt("data/y_test.csv", delimiter=",").astype(int)

##############################################################################################################################################
##############################################################################################################################################

print("ejercicio 1")

def KNN(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        # Calcular todas las similitudes coseno entre el punto de prueba y todos los puntos de entrenamiento
        similarities = 1 - np.dot(X_train, test_point) / (np.linalg.norm(X_train, axis=1) * np.linalg.norm(test_point))
        
        # Obtener los índices de los k vecinos más cercanos
        nearest_indices = np.argsort(similarities)[:k]
        
        # Obtener las etiquetas de los k vecinos más cercanos
        nearest_labels = y_train[nearest_indices]
        
        # Contar las ocurrencias de cada etiqueta
        unique_labels, counts = np.unique(nearest_labels, return_counts=True)
        
        # Predecir la etiqueta más frecuente
        predicted_label = unique_labels[np.argmax(counts)]
        
        # Agregar la predicción a la lista de predicciones
        predictions.append(predicted_label)
    
    return np.array(predictions)

##############################################################################################################################################
##############################################################################################################################################

print("ejercicio 2)a) test con truco de la matriz Householder")
tol = 10e-7

# Definición de la matriz diagonal D
N = 10
D = np.diag(range(N, 0, -1))

# Definición y normalización del vector v
v = np.ones((D.shape[0], 1))
v = v / np.linalg.norm(v)

# Construcción de la matriz de Householder B
B = np.eye(D.shape[0]) - 2 * (v @ v.T)

# Transformación de la matriz D
M = B.T @ D @ B

# Cálculo de los valores y vectores propios usando el método de la potencia
max_iter = 100000
eigenvalues = []
eigenvectors = []

for _ in range(N):
    aus, auv, _ = my_module.metodoPotenciaDeflacion(M, tol, max_iter)
    eigenvalues.extend(aus)
    eigenvectors.extend(auv)

# Convertir listas a arrays para facilitar el manejo
eigenvalues = np.array(eigenvalues)
eigenvectors = np.array(eigenvectors)

print("Valores propios:", eigenvalues)
print("Vectores propios:", eigenvectors)

##############################################################################################################################################
##############################################################################################################################################

print("ejercicio 2)b)")


def generar_matrizH_aleatoria(autovalores):
    n = len(autovalores)
    # Crear una matriz diagonal con los autovalores dados
    D = np.diag(autovalores)
    # Generar una matriz ortogonal aleatoria usando QR descomposición
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    # Crear la matriz simétrica con los autovalores dados
    H = Q @ D @ Q.T
    return H

def calcular_error(autovalores, autovectores, matriz):
    errores = []
    for i in range(len(autovalores)):
        error = np.linalg.norm(np.dot(matriz, autovectores[i]) - autovalores[i] * autovectores[i])
        errores.append(error)
    return errores

# Definir valores de epsilon
epsilons = np.logspace(-4, 2, num=10)  # Reducir el número de puntos a 10 para pruebas iniciales

errores_promedio = []
pasos_promedio = []

# Realizar mediciones para cada valor de epsilon
for epsilon in epsilons:
    # Generar matriz de Householder aleatoria
    autovalores_conocidos = [10, 10 - epsilon, 5, 2, 1]
    matriz_H = generar_matrizH_aleatoria(autovalores_conocidos)
    errores = []
    pasos = []
    for _ in range(10):  # Realizar 10 mediciones para cada valor de epsilon
        autovalores, autovectores, pasos_iteracion = my_module.metodoPotenciaDeflacion(matriz_H, tol, max_iter)
        errores.extend(calcular_error(autovalores, autovectores, matriz_H))
        pasos.extend(pasos_iteracion)
    
    # Calcular promedio y desvío estándar de errores y cantidad de pasos
    errores_promedio.append(np.mean(errores))
    pasos_promedio.append(np.mean(pasos))

# Graficar el error del Método de la Potencia vs Epsilon
plt.figure(figsize=(8, 6))
plt.plot(epsilons, errores_promedio, marker='o', linestyle='-')
plt.title('Error del Método de la Potencia vs Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Error Promedio')
plt.grid(True)
plt.show()

# Graficar el número de pasos vs Epsilon
plt.figure(figsize=(8, 6))
plt.plot(epsilons, pasos_promedio, marker='o', linestyle='-')
plt.title('Número de Pasos vs Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Número de Pasos Promedio')
plt.grid(True)
plt.show()

##############################################################################################################################################
##############################################################################################################################################

print("ejercicio 3)a)")

print("knn de sklearn")
# haciendolo con la libreria que viene dada

# Paso 2: Crear el modelo KNN con k = 5
knn_model = KNeighborsClassifier(n_neighbors=5)

# Paso 3: Entrenar el modelo
knn_model.fit(X_train, y_train)

# Paso 4: Evaluar el modelo en el conjunto de prueba
y_pred = knn_model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)

# Paso 5: Mostrar la precisión del modelo
print("Precisión del modelo KNN con k=5:", accuracy)

print("knn propio")

X_newtrain, X_dev, y_newtrain, y_dev = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

y_pred = KNN(X_train, y_train, X_test, 5)

resultados1 = KNN(X_newtrain, y_newtrain, X_dev, 5)

print("El algoritmo ha acertado para knn propio: ", np.sum(resultados1 == y_dev),"de", y_dev.size, "veces")

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)

# Mostrar la precisión del modelo
print("Precisión del modelo KNN propio:", accuracy)


##############################################################################################################################################
##############################################################################################################################################


print("ejercicio 3)b)")

# Definir valores de k a probar
k_values = [1, 3, 5, 7, 9]

# Inicializar lista para almacenar las medidas de exactitud promedio
exactitud_promedio = []

# Definir el número de folds para la validación cruzada
num_folds = 5

# Crear objeto StratifiedKFold
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Iterar sobre los valores de k
for k in k_values:
    fold_accuracies = []
    
    # Validación cruzada
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        # Predecir sobre el conjunto de validación usando la función KNN
        y_val_pred = KNN(X_train_fold, y_train_fold, X_val_fold, k)
        
        # Calcular la precisión y agregarla a la lista de precisiones
        accuracy = accuracy_score(y_val_fold, y_val_pred)
        fold_accuracies.append(accuracy)
    
    # Calcular la precisión promedio para el valor actual de k
    avg_accuracy = np.mean(fold_accuracies)
    exactitud_promedio.append(avg_accuracy)
    print(f'k={k}, Exactitud promedio={avg_accuracy:.4f}')

##############################################################################################################################################
##############################################################################################################################################

print("ejercicio 3)c)")

# Verificar el tamaño de X_train. 
print(f"Tamaño de X_train: {X_train.shape[0]}")
#Entonces tendremos 5000 caracteristicas principales en los componentes originales

# mi funcion de pca
def PCA(A):
    # Paso 1: Restar el promedio de cada columna de A
    A_mean = np.mean(A, axis=0)
    A_centered = A - A_mean
    
    # Paso 2: Construir la matriz de covarianza
    n = A_centered.shape[1]
    matriz_cov = ((A_centered.T @ A_centered) / (n - 1))
    
    # Paso 3: Encontrar los autovalores y autovectores de CA
    autovalores, autovectores, _ = my_module.metodoPotenciaDeflacion(matriz_cov,1e-7,1000)
    
    # Paso 4: Construir la matriz de cambio de base V con los autovectores
    V = autovectores

    return V, A_mean

# Aplicar PCA a la matriz de datos A
V, A_mean = PCA(X_train)

#transformar datos de entrenamiento utilizando la matriz de cambio de base V
X_train_pca = (X_train - A_mean) @ V

# Calcular la varianza explicada
explained_variance = np.var(X_train_pca, axis=0) / np.var(X_train, axis=0).sum()

# Gráfico de Varianza Explicada Acumulada
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(explained_variance))
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Explicada Acumulada')
plt.title('Varianza Explicada Acumulada por Componentes Principales')
plt.grid(True)
plt.show()

num_component = 50

# Varianza total explicada por los primeros 50 componentes principales
total_explained_variance = np.sum(explained_variance[:num_components])
print(f"Varianza total explicada por los primeros 50 componentes principales: {total_explained_variance:.2f}")

##############################################################################################################################################
##############################################################################################################################################

print("ejercicio 3) d)")

# Parámetros a explorar
k_valores = [1, 3, 5, 7, 9]
p_valores = [10, 20, 30, 40, 50]  # Número de componentes principales

# Preparar la validación cruzada
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

mejor_exactitud = 0
mejor_k = None
mejor_p = None

# Exploración conjunta de hiperparámetros
for k in k_valores:
    for p in p_valores:
        accuracies = []
        
        for train_index, val_index in kf.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            
            # Aplicar PCA a los datos de entrenamiento del fold
            V, A_mean = PCA(X_train_fold)
            X_train_fold_pca = (X_train_fold - A_mean) @ V[:, :p]
            X_val_fold_pca = (X_val_fold - A_mean) @ V[:, :p]
            
            # Predecir sobre el conjunto de validación usando la función KNN
            y_val_pred = KNN(X_train_fold_pca, y_train_fold, X_val_fold_pca, k)
            accuracy = accuracy_score(y_val_fold, y_val_pred)
            accuracies.append(accuracy)
        
        mean_accuracy = np.mean(accuracies)
        print(f"promedio de exactitud para: k={k}, p={p}: {mean_accuracy:.4f}")
        
        if mean_accuracy > mejor_exactitud:
            mejor_exactitud = mean_accuracy
            mejor_k = k
            mejor_p = p

print(f"mejor k: {mejor_k}, mejor p: {mejor_p}, mejor exactitud: {mejor_exactitud:.4f}")

# Entrenar con todos los datos de entrenamiento usando los mejores hiperparámetros
V, A_mean = PCA(X_train)
X_train_pca = (X_train - A_mean) @ V[:, :mejor_p]
X_test_pca = (X_test - A_mean) @ V[:, :mejor_p]

# Usar la función KNN para predecir en el conjunto de prueba
y_test_pred = KNN(X_train_pca, y_train, X_test_pca, mejor_k)
test_exactitud = accuracy_score(y_test, y_test_pred)

print(f"Test accuracy with best k and p: {test_exactitud:.4f}")