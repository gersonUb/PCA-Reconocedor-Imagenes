import sys
from sklearn.decomposition import PCA
import numpy as np
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


matriz = np.ones((2, 2), dtype=np.float32)
tol = 1e-7
# Calcular la norma infinita de la matriz
matriz_norm_inf = np.linalg.norm(matriz, np.inf)
# Calcular el número máximo de iteraciones usando la norma infinita y la tolerancia
max_iter = 100  # Limitar el número máximo de iteraciones a 100

# Definir la matriz matrix y los vectores conocidos (autovectores)
autovalor, autovector = my_module.metodoPotencia(matriz, tol, max_iter)

print("Autovalor:")
print(autovalor)
print("\nAutovector:")
print(autovector)

# DE ANTEMANO SE QUE ESTOS SON LOS AUTOVECTORES [-1,1], [1,1] DE matriz

print("ejercicio 2)a) test con truco de la matriz Householder")

# Obtener las columnas de la matriz
columnas = [matriz[:, i] for i in range(matriz.shape[1])]

# Aplicar la transformación de Householder a cada columna
matrices_h = [my_module.householder(columna) for columna in columnas]

# Imprimir las matrices resultantes
for matriz_h in matrices_h:
    print(matriz_h)
    
print("ejercicio 2)b)")

def generar_matriz_aleatoria(autovalores):
    n = len(autovalores)
    matriz = np.diag(autovalores)  # Matriz diagonal con los autovalores dados
    tol = 1e-7
    matriz_norm_inf = np.linalg.norm(matriz, np.inf)
    max_iter = 100
    
    autovalores_calculados = []
    for i in range(n):
        autovalor, _ = my_module.metodoPotencia(matriz, tol, max_iter)
        autovalores_calculados.append(autovalor)
        # Perturba la matriz ligeramente para intentar obtener diferentes autovalores
        matriz[i, i] += 0.1 * np.random.rand()
    
    return np.diag(np.array(autovalores_calculados, dtype=np.float32))

def calcular_error(autovalores, autovectores, matriz):
    errores = []
    for i in range(len(autovalores)):
        error = np.linalg.norm(np.dot(matriz, autovector) - autovalores[i] * autovector)
        errores.append(error)
    return errores

# Definir valores de epsilon
epsilons = np.logspace(-4, 2, num=5)  # Reducir el número de puntos a 5 para pruebas iniciales

errores_promedio = []
pasos_promedio = []

for epsilon in epsilons:
    start_time = time.time()  # Medir el tiempo de inicio
    autovalores_conocidos = [10, 10 - epsilon, 5, 2, 1]
    matriz = generar_matriz_aleatoria(autovalores_conocidos)
    autovalores = []  # Inicializar lista de autovalores
    for _ in range(len(autovalores_conocidos)):  # Realizar el método de la potencia para cada autovalor
        autovalor, autovector = my_module.metodoPotencia(matriz, tol, max_iter)
        autovalores.append(autovalor)  # Agregar el autovalor a la lista
    errores = calcular_error(autovalores_conocidos, autovector.reshape(-1, 1), matriz)
    errores_promedio.append(np.mean(errores))
    pasos_promedio.append(len(autovalores))
    end_time = time.time()  # Medir el tiempo de finalización
    print(f"Epsilon: {epsilon}, Tiempo: {end_time - start_time:.4f} segundos")  
print(pasos_promedio)

# Graficar los resultados
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(epsilons, errores_promedio)
plt.title('Error del Método de la Potencia vs Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Error Promedio')

plt.subplot(1, 2, 2)
plt.plot(epsilons, pasos_promedio)
plt.title('Número de Pasos vs Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Número de Pasos Promedio')

plt.tight_layout()
plt.show()

print("ejercicio (3)")

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

# Tomar solo una pequeña parte de los datos de prueba y entrenamiento
X_train_small = X_train[:100]
y_train_small = y_train[:100]
X_test_small = X_test[:10]

# Ejecutar tu función KNN con k=5 en este conjunto de datos pequeño
predictions1 = KNN(X_train_small, y_train_small, X_test_small, 2)
# Imprimir las predicciones y las etiquetas verdaderas para verificar
print("Predicciones:", predictions1)
print("Etiquetas verdaderas con k=2:", y_test[:10])

predictions2 = KNN(X_train_small, y_train_small, X_test_small, 5)
# Imprimir las predicciones y las etiquetas verdaderas para verificar
print("Predicciones:", predictions2)
print("Etiquetas verdaderas con k=5:", y_test[:10])

# NO TERMINA DE COMPILAR NUNCA ...
"""
# Entrenar el modelo utilizando mi función KNN
y_pred = KNN(X_train, y_train, X_test, 5)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)

# Mostrar la precisión del modelo
print("Precisión del modelo KNN propio:", accuracy)
"""

print("ejercicio 3)b)")

# Definir valores de k a probar
k_values = [1, 3, 5, 7, 9]

# Inicializar lista para almacenar las medidas de exactitud promedio
mean_accuracy_scores = []

# Definir el número de folds para la validación cruzada
num_folds = 5

# Dividir los datos en los índices de los folds
fold_indices = np.array_split(np.arange(len(X_train)), num_folds)

# Iterar sobre cada valor de k
for k in k_values:
    fold_accuracies = []  # Lista para almacenar las medidas de exactitud en cada fold
    
    # Iterar sobre cada fold para la validación cruzada
    for val_fold_index in range(num_folds):
        # Obtener los índices de entrenamiento y validación para este fold
        val_indices = fold_indices[val_fold_index]
        train_indices = np.concatenate([fold_indices[i] for i in range(num_folds) if i != val_fold_index])
        
        # Obtener los conjuntos de entrenamiento y validación
        X_train_fold, X_val_fold = X_train[train_indices], X_train[val_indices]
        y_train_fold, y_val_fold = y_train[train_indices], y_train[val_indices]
        
        # Entrenar el modelo KNN
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(X_train_fold, y_train_fold)
        
        # Realizar predicciones en el fold de validación
        y_pred_val = knn_model.predict(X_val_fold)
        
        # Calcular la medida de exactitud en el fold de validación
        fold_accuracy = accuracy_score(y_val_fold, y_pred_val)
        fold_accuracies.append(fold_accuracy)
    
    # Calcular el promedio de las medidas de exactitud en todos los folds para este valor de k
    mean_accuracy = np.mean(fold_accuracies)
    mean_accuracy_scores.append(mean_accuracy)

# Encontrar el mejor valor de k
best_k_index = np.argmax(mean_accuracy_scores)
best_k = k_values[best_k_index]
best_accuracy = mean_accuracy_scores[best_k_index]

print(f"El mejor valor de k encontrado es: {best_k}")
print(f"Exactitud promedio correspondiente: {best_accuracy}")

print("usando pca")

# Verificar el tamaño de X_train
print(f"Tamaño de X_train: {X_train.shape[0]}")

# Calcular la matriz de covarianza
cov_matrix = np.cov(X_train, rowvar=False)

# Encontrar los autovectores principales usando el método de la potencia
num_components = 50
autovectores = []
for _ in range(num_components):
    _ , autovector = my_module.metodoPotencia(cov_matrix,tol,100)
    autovectores.append(autovector)
    # Deflación de la matriz de covarianza
    cov_matrix -= autovector[:, np.newaxis] * autovector[np.newaxis, :]

autovectores = np.array(autovectores)

# Transformar los datos usando los autovectores encontrados
X_train_pca = np.dot(X_train, autovectores.T)
X_test_pca = np.dot(X_test, autovectores.T)

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

# Varianza total explicada por los primeros 50 componentes principales
total_explained_variance = np.sum(explained_variance[:num_components])
print(f"Varianza total explicada por los primeros 50 componentes principales: {total_explained_variance:.2f}")